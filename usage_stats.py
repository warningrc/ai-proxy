"""
Token 用量统计模块。

职责：
- SQLite 建表（tenants / usage_log）
- 启动时从 config 同步租户信息到 DB（upsert）
- 运行时通过 Bearer key 匹配租户
- 记录每次请求的 token 用量
- 提供聚合查询接口
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
#  数据结构
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    req_id: str
    tenant_id: str
    provider: str
    model: str
    endpoint: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    duration_ms: float = 0.0
    prompt: Optional[str] = None
    status_code: Optional[int] = None
    client_ip: Optional[str] = None


@dataclass
class TenantInfo:
    id: str
    name: str
    status: str = "active"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _mask_key(raw_key: str) -> str:
    """脱敏显示：前 6 + ... + 后 4"""
    if len(raw_key) <= 12:
        return "***"
    return f"{raw_key[:6]}***{raw_key[-4:]}"


# ---------------------------------------------------------------------------
#  UsageStats 主类
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    api_key_hash TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime'))
);

CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
    req_id      TEXT    NOT NULL,
    tenant_id   TEXT    NOT NULL DEFAULT 'default',
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    prompt      TEXT,
    status_code INTEGER,
    client_ip   TEXT,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms     REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_request_tenant_model_ts ON request_log (tenant_id, model, ts);
CREATE INDEX IF NOT EXISTS idx_request_ts              ON request_log (ts);
"""


class UsageStats:
    """线程安全的用量统计管理器。"""

    def __init__(self, db_path: str = "./stats.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        # 内存中的 hash -> TenantInfo 映射，用于快速匹配
        self._key_hash_to_tenant: Dict[str, TenantInfo] = {}
        # 内存中的 tenant_id -> TenantInfo 映射，用于快速根据 ID 查询
        self._id_to_tenant: Dict[str, TenantInfo] = {}

    # ------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        
        # 1. 检测并升级旧表名 usage_log -> request_log
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='usage_log'")
            if cursor.fetchone()[0] > 0:
                try:
                    self._conn.execute("ALTER TABLE usage_log RENAME TO request_log")
                    self._conn.execute("DROP INDEX IF EXISTS idx_usage_tenant_model_ts")
                    self._conn.execute("DROP INDEX IF EXISTS idx_usage_ts")
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

        # 2. 执行建表脚本
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        # 3. 对已有的 request_log 表追加新列（如果不存在的话）
        with self._lock:
            for col_def in [
                ("prompt", "TEXT"),
                ("status_code", "INTEGER"),
                ("client_ip", "TEXT")
            ]:
                try:
                    self._conn.execute(f"ALTER TABLE request_log ADD COLUMN {col_def[0]} {col_def[1]}")
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

    def upsert_tenants(self, tenants: Sequence[Dict[str, Any]]) -> None:
        """
        从 config 同步租户到 DB 和内存。

        tenants: [{"id": "...", "name": "...", "api_key": "...", "status": "..."}]
        """
        if not self._conn:
            raise RuntimeError("DB not initialized; call init_db() first")

        self._key_hash_to_tenant.clear()
        self._id_to_tenant.clear()

        with self._lock:
            # 1. 写入数据库
            for t in tenants:
                tid = t["id"]
                name = t["name"]
                key_hash = _hash_key(t["api_key"])
                status = t.get("status", "active")

                self._conn.execute(
                    """
                    INSERT INTO tenants (id, name, api_key_hash, status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        api_key_hash = excluded.api_key_hash,
                        status = excluded.status
                    """,
                    (tid, name, key_hash, status),
                )
            self._conn.commit()

            # 2. 从数据库加载所有租户到内存（为了获得最新的 status）
            cursor = self._conn.execute("SELECT id, name, api_key_hash, status FROM tenants")
            rows = cursor.fetchall()
            for row in rows:
                tid, name, key_hash, status = row
                info = TenantInfo(id=tid, name=name, status=status)
                self._key_hash_to_tenant[key_hash] = info
                self._id_to_tenant[tid] = info

    # ------------------------------------------------------------------
    #  租户识别
    # ------------------------------------------------------------------

    def resolve_tenant(self, raw_key: Optional[str]) -> str:
        """
        通过 raw_key 的 sha256 在内存映射中查找 tenant_id。
        未匹配返回 'default'。
        """
        if not raw_key:
            return "default"
        h = _hash_key(raw_key)
        info = self._key_hash_to_tenant.get(h)
        return info.id if info else "default"

    def get_tenant_info_by_id(self, tenant_id: str) -> Optional[TenantInfo]:
        """通过 tenant_id 获取租户详细信息。"""
        if tenant_id == "default":
            return TenantInfo(id="default", name="Default Tenant", status="active")
        with self._lock:
            return self._id_to_tenant.get(tenant_id)

    # ------------------------------------------------------------------
    #  记录用量
    # ------------------------------------------------------------------

    def record_usage(self, record: UsageRecord) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO request_log
                    (req_id, tenant_id, provider, model, endpoint,
                     prompt, status_code, client_ip,
                     input_tokens, output_tokens, cache_read_tokens,
                     cache_creation_tokens, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.req_id,
                    record.tenant_id,
                    record.provider,
                    record.model,
                    record.endpoint,
                    record.prompt,
                    record.status_code,
                    record.client_ip,
                    record.input_tokens,
                    record.output_tokens,
                    record.cache_read_tokens,
                    record.cache_creation_tokens,
                    record.duration_ms,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    #  查询统计
    # ------------------------------------------------------------------

    def query_stats(
        self,
        *,
        group_by: str = "model",
        since: Optional[str] = None,
        until: Optional[str] = None,
        model: Optional[str] = None,
        tenant: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._conn:
            return {"stats": [], "period": {}}

        allowed_dimensions = {"model", "provider", "tenant_id", "tenant"}
        group_parts = [g.strip() for g in group_by.split(",")]
        # 规范化：外部用 "tenant" 但 DB 列名是 "tenant_id"
        select_cols = []
        group_cols = []
        for g in group_parts:
            if g == "tenant":
                select_cols.append("tenant_id")
                group_cols.append("tenant_id")
            elif g in allowed_dimensions:
                select_cols.append(g)
                group_cols.append(g)
            else:
                select_cols.append("model")
                group_cols.append("model")

        if not group_cols:
            group_cols = ["model"]
            select_cols = ["model"]

        select_str = ", ".join(select_cols)
        group_str = ", ".join(group_cols)

        where_clauses: List[str] = []
        params: List[Any] = []

        if since:
            where_clauses.append("ts >= ?")
            params.append(since)
        if until:
            where_clauses.append("ts <= ?")
            params.append(until)
        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if tenant:
            where_clauses.append("tenant_id = ?")
            params.append(tenant)
        if provider:
            where_clauses.append("provider = ?")
            params.append(provider)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT {select_str},
                   COUNT(*) as request_count,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(total_tokens) as total_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(cache_creation_tokens) as cache_creation_tokens
            FROM request_log
            {where_sql}
            GROUP BY {group_str}
            ORDER BY request_count DESC
        """

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        stats = []
        for row in rows:
            item: Dict[str, Any] = {}
            for i, col in enumerate(select_cols):
                key = "tenant_id" if col == "tenant_id" else col
                item[key] = row[i]
            offset = len(select_cols)
            item["request_count"] = row[offset]
            item["input_tokens"] = row[offset + 1]
            item["output_tokens"] = row[offset + 2]
            item["total_tokens"] = row[offset + 3]
            item["cache_read_tokens"] = row[offset + 4]
            item["cache_creation_tokens"] = row[offset + 5]
            stats.append(item)

        # 确定实际查询的时间范围
        period: Dict[str, Optional[str]] = {
            "since": since,
            "until": until,
        }
        if not since or not until:
            with self._lock:
                bounds = self._conn.execute(
                    "SELECT MIN(ts), MAX(ts) FROM request_log"
                ).fetchone()
            if bounds and bounds[0]:
                if not since:
                    period["since"] = bounds[0]
                if not until:
                    period["until"] = bounds[1]

        return {"stats": stats, "period": period}

    # ------------------------------------------------------------------
    #  关闭
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
