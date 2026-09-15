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
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

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


# request_log.ts 的存储格式（见 _SCHEMA_SQL 的 DEFAULT），定宽，可直接按字符串比较/截断
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


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


# 未匹配到任何租户时的占位身份。是所有请求共享的不可变对象，避免每次查询都新建。
_DEFAULT_TENANT = TenantInfo(id="default", name="Default Tenant", status="active")

# 关机时塞进写队列的叫醒信号：writer 线程可能正阻塞在 get(timeout=flush_interval)
# 上，光设 _stop 得等它自然超时才能退出。取出即丢弃，不落库。
_WAKEUP = object()

_INSERT_SQL = """
    INSERT INTO request_log
        (req_id, tenant_id, provider, model, endpoint,
         prompt, status_code, client_ip,
         input_tokens, output_tokens, cache_read_tokens,
         cache_creation_tokens, duration_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# 抢不到写锁时最多等多久（毫秒）。
#
# 多 worker 下 N 个进程共用一把写锁，凡是不肯多等的地方都会在这一刻报
# "database is locked"。取 30s 是被 VACUUM 逼出来的：VACUUM 要重写整个库文件、
# 全程独占写锁（实测 2.8GB 的库约 8s），这期间其它进程的写入只能排队。原先的 5s
# 比这个窗口短，于是「清理跑起来之后的那几秒」里所有入库都失败：生产日志里
# init_db、upsert_tenants、cleanup 三条路径同时报锁，其中两条直接把 worker 带崩。
_BUSY_TIMEOUT_MS = 30_000


def _is_locked_error(e: BaseException) -> bool:
    """是否属于「写锁没等到」这类**重试有意义**的错误。

    多 worker 下这是常态而不是故障：同一个库只有一把写锁，别人正写着就得等。
    """
    return isinstance(e, sqlite3.Error) and "lock" in str(e).lower()


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """对一条连接施加运行时 PRAGMA。

    - busy_timeout **必须第一个设**：它决定「抢不到写锁时等多久」，而下面几条
      语句本身就要拿锁。放后面的话，那些语句用的是 sqlite3.connect 的默认值
      （5s），这里配的值对它们等于是没生效。
    - journal_mode=WAL：读不阻塞写、写不阻塞读。原先是默认的 rollback journal，
      一次后台全表扫描就能让所有写入排队；多进程（uvicorn --workers）下更是必须。
    - synchronous=NORMAL：WAL 下这个档位不会因掉电损坏库，最多丢最后几个已提交
      事务。统计数据的价值不值得每次 commit 都做一次 fsync。
    - temp_store=MEMORY：聚合查询的临时 B 树放内存，避免落盘。
    - journal_size_limit：WAL 文件在 checkpoint 后收缩到该上限。默认不限制，
      长期运行下 -wal 会只涨不消（实测 WAL 已到 2.4MB 而主库仍是 4KB）。

    注意 journal_mode=WAL 是持久化在库文件里的设置，但每条连接都执行一次也无害。
    """
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.execute("PRAGMA journal_size_limit=67108864")  # 64MB


def _truncate_prompt(prompt: Optional[str], limit: int) -> Optional[str]:
    """按 limit 截断 prompt 后入库。

    请求体全文对计费没有价值，却曾经占到本库 88% 的体积（实测平均 300KB/行）。
    limit <= 0 表示完全不存。
    """
    if prompt is None or limit <= 0:
        return None
    return prompt if len(prompt) <= limit else prompt[:limit]


def _to_row(record: UsageRecord, prompt_limit: int) -> Tuple[Any, ...]:
    """UsageRecord → INSERT 参数元组（由 writer 线程统一构造）。"""
    return (
        record.req_id,
        record.tenant_id,
        record.provider,
        record.model,
        record.endpoint,
        _truncate_prompt(record.prompt, prompt_limit),
        record.status_code,
        record.client_ip,
        record.input_tokens,
        record.output_tokens,
        record.cache_read_tokens,
        record.cache_creation_tokens,
        record.duration_ms,
    )


# ---------------------------------------------------------------------------
#  聚合查询白名单
#
#  下面三张表把「外部参数 -> SQL 片段」的映射集中在一处。外部传入的字符串只用于
#  字典查表，永远不会直接拼进 SQL，因此不存在注入面。新增维度／指标只需在这里加一行。
#
#  ts 是本地时间文本 "YYYY-MM-DDTHH:MM:SS"（见 _SCHEMA_SQL 的 DEFAULT），所以 substr
#  切片、字符串比较、strftime 都能直接用，无需类型转换。
# ---------------------------------------------------------------------------

# 分组维度：外部名 -> SELECT/GROUP BY 表达式（key 即响应里输出的字段名）
_DIMENSION_SQL: Dict[str, str] = {
    "model": "model",
    "provider": "provider",
    "tenant": "tenant_id",
    "tenant_id": "tenant_id",
    "day": "substr(ts, 1, 10)",
}

# 聚合指标：外部名 -> 聚合表达式
_METRIC_SQL: Dict[str, str] = {
    "total_tokens": "SUM(total_tokens)",
    "input_tokens": "SUM(input_tokens)",
    "output_tokens": "SUM(output_tokens)",
    "cache_read_tokens": "SUM(cache_read_tokens)",
    "cache_creation_tokens": "SUM(cache_creation_tokens)",
    "request_count": "COUNT(*)",
}

# 时间分桶：外部名 -> 分桶表达式。
# 注意 strftime('%W') 的周定义（周一为一周起点，00-53）与 Python date.strftime('%W')
# 一致，补缺口时两边能对上。
_BUCKET_SQL: Dict[str, str] = {
    "day": "substr(ts, 1, 10)",
    "week": "strftime('%Y-W%W', ts)",
    "month": "substr(ts, 1, 7)",
}

# 补缺口的安全阀：桶数超过这个值就放弃补齐，只返回实际存在的桶
_MAX_FILLED_BUCKETS = 400


def _format_bucket(day: _date, bucket: str) -> str:
    """把 date 格式化成与 _BUCKET_SQL 中同名表达式一致的字符串。"""
    if bucket == "month":
        return day.strftime("%Y-%m")
    if bucket == "week":
        return day.strftime("%Y-W%W")
    return day.strftime("%Y-%m-%d")


def _first_bucket_date(start: _date, bucket: str) -> _date:
    """补缺口的起点：month 对齐到当月 1 号，其余保持原样。"""
    if bucket == "month":
        return start.replace(day=1)
    return start


def _next_bucket(day: _date, bucket: str) -> _date:
    """按分桶粒度推进一步。"""
    if bucket == "month":
        # 跳到下个月 1 号
        return (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    if bucket == "week":
        return day + timedelta(days=7)
    return day + timedelta(days=1)


def _normalize_range(
    since: Optional[str], until: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    归一化时间参数：把裸日期补全成完整时间戳。

    前端用 <input type="date"> 只会给出 "2026-09-15"。而 WHERE 是裸的 `ts <= ?`，
    若直接传裸日期，`until=2026-09-15` 会把当天 00:00 之后的数据全部漏掉。
    """
    if since and len(since) == 10:
        since = since + "T00:00:00"
    if until and len(until) == 10:
        until = until + "T23:59:59"
    return since, until


class UsageStats:
    """用量统计管理器。

    连接分工（别把两者混用）：
    - ``_conn``     写连接。只被 writer 线程和 schema 变更（init_db / upsert_tenants
      / cleanup_old_records / vacuum）使用，全部在 ``_lock`` 内操作。
    - ``_read_conn`` 读连接。查询走它，且设了 ``query_only``。WAL 模式下读不阻塞写，
      所以管理端的一次全表扫描不会再卡住入库。

    写入路径非阻塞：``record_usage`` 只入队，真正的 INSERT 由 ``_writer_loop`` 攒批
    （每 ``flush_interval`` 秒或攒够 ``batch_size`` 条）用一次 ``executemany`` 提交。
    事件循环上的请求因此永远不用为 SQLite 的 fsync 买单。
    """

    # 全时段排名缓存有效期（秒）
    _RANK_TTL = 300.0

    # 写队列上限。writer 线程被长时间阻塞时用来兜底 —— 最典型的场景是 VACUUM：
    # 它要重写整个库，期间一直占着写锁，队列会在这段时间里堆积。prompt 截断后单条
    # 记录通常只有几百字节，按 2KB 上限估算，打满约 40MB。
    # 注意这只堵 writer 线程，不堵请求路径（record_usage 只入队），所以打满时的
    # 表现是「统计漏记 + 报错日志」，而不是请求变慢。丢数据必须让人看见。
    _MAX_QUEUE_SIZE = 20_000

    def __init__(
        self,
        db_path: str = "./stats.db",
        *,
        prompt_max_chars: int = 2048,
        flush_interval: float = 0.5,
        batch_size: int = 500,
    ):
        self._db_path = db_path
        self._prompt_max_chars = prompt_max_chars
        self._flush_interval = max(0.01, flush_interval)
        self._batch_size = max(1, batch_size)

        self._conn: Optional[sqlite3.Connection] = None
        self._read_conn: Optional[sqlite3.Connection] = None
        # 保护 _conn 上的写操作（writer 线程 + schema 变更 + 清理/VACUUM）
        self._lock = threading.Lock()
        # 保护 _read_conn。与 _lock 分离，长查询不会挡住 writer 线程。
        self._read_lock = threading.Lock()

        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=self._MAX_QUEUE_SIZE)
        self._stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._dropped = 0
        # close() 置位后 record_usage 直接返回：drain 的退出条件是「队列恰好为空」，
        # 收尾期间还能入队的话，那批记录会永远留在队列里（既没落库也没人再处理）。
        self._closing = False
        # 累计写入失败的条数。flush() 靠它判定「哨兵被置位但记录没进库」这种情况，
        # 否则写失败也照样返回 True。
        self._write_failed = 0
        # 「统计库不可用、正在丢弃记录」是否已经报过（见 record_usage）
        self._no_db_warned = False

        # 内存中的 hash -> TenantInfo 映射，用于快速匹配
        self._key_hash_to_tenant: Dict[str, TenantInfo] = {}
        # 内存中的 tenant_id -> TenantInfo 映射，用于快速根据 ID 查询
        self._id_to_tenant: Dict[str, TenantInfo] = {}
        # (时间戳, 名字列表) 的全时段用量排名缓存，见 model_rank() / provider_rank()
        self._rank_cache: Optional[Tuple[float, List[str]]] = None
        self._provider_cache: Optional[Tuple[float, List[str]]] = None

    # ------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """建库/升级 schema，成功后启动批量写入线程。失败时**不留残留状态**。

        原子性在这里是有实际意义的：多 worker 同时启动时 N 个进程会同时调这个
        函数，而这里每一步都要拿写锁，任何一步都可能撞上别人正在写而抛
        "database is locked"。抛异常时连接要么还没建起来、要么被关掉，
        ``_conn`` 保持 None，实例就是一个干净的「没有统计库」状态：
        ``record_usage`` 直接丢弃、查询返回空。要是留下半初始化的状态，writer
        线程和读连接缺一个，表现就成了「请求都正常、统计默默变少」。
        """
        # 全程用局部变量，最后才落到 self 上 —— 中途失败不用收拾半个实例。
        # 此时 self._conn 还是 None，没有别的线程会碰这些局部对象，故不必持锁。
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        read_conn: Optional[sqlite3.Connection] = None
        try:
            _apply_pragmas(conn)

            # 1. 检测并升级旧表名 usage_log -> request_log
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='usage_log'")
            if cursor.fetchone()[0] > 0:
                try:
                    conn.execute("ALTER TABLE usage_log RENAME TO request_log")
                    conn.execute("DROP INDEX IF EXISTS idx_usage_tenant_model_ts")
                    conn.execute("DROP INDEX IF EXISTS idx_usage_ts")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

            # 2. 执行建表脚本
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

            # 3. 对已有的 request_log 表追加新列（如果不存在的话）
            for col_def in [
                ("prompt", "TEXT"),
                ("status_code", "INTEGER"),
                ("client_ip", "TEXT")
            ]:
                try:
                    conn.execute(f"ALTER TABLE request_log ADD COLUMN {col_def[0]} {col_def[1]}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

            # 4. schema 就绪后再开读连接
            read_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            _apply_pragmas(read_conn)
            # 读连接只读：将来若有人在查询路径里误加写入，这里会立刻报错，而不是悄悄去
            # 抢写锁、把 writer 线程堵住。
            read_conn.execute("PRAGMA query_only=ON")
        except BaseException:
            conn.close()
            if read_conn is not None:
                read_conn.close()
            raise

        # 5. 全部就绪才切换实例状态，最后启动 writer 线程
        self._conn = conn
        self._read_conn = read_conn
        self._start_writer()

    def upsert_tenants(self, tenants: Sequence[Dict[str, Any]]) -> None:
        """
        从 config 同步租户到 DB 和内存。

        tenants: [{"id": "...", "name": "...", "api_key": "...", "status": "..."}]

        会抛异常（库没初始化 / 写锁没等到）。启动和热重载路径请用 sync_tenants：
        那边不允许把「写不进库」升级成「网关起不来」或「所有 key 401」。
        """
        if not self._conn:
            raise RuntimeError("DB not initialized; call init_db() first")

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

        self._install_tenant_maps(rows)

    def _install_tenant_maps(self, rows: Iterable[Tuple[str, str, str, str]]) -> None:
        """把 (id, name, key_hash, status) 行装进内存映射。

        先建全新的映射，最后两个赋值整体换掉引用。
        不要改成「就地 clear() 再逐条填」——那样在填充完成前，并发的鉴权请求会
        查到空表，把刚 reload 完的这几毫秒里所有正常请求判成 401。
        """
        by_hash: Dict[str, TenantInfo] = {}
        by_id: Dict[str, TenantInfo] = {}
        for tid, name, key_hash, status in rows:
            info = TenantInfo(id=tid, name=name, status=status)
            by_hash[key_hash] = info
            by_id[tid] = info
        # 单次属性赋值在 CPython 里是原子的，读方永远看到完整的一份映射
        self._key_hash_to_tenant = by_hash
        self._id_to_tenant = by_id

    def sync_tenants(self, tenants: Sequence[Dict[str, Any]], *, attempts: int = 3) -> bool:
        """同步租户，**永不抛异常**；返回是否真的写进了库。

        鉴权只认内存映射（见 resolve_tenant），而这份映射平时是从库里读回来的
        （为了拿到库里可能被改过的 status）。所以「库写不进去」必须降级成
        「用配置里的租户填内存映射」，否则一个记账问题会升级成整个网关 401。

        写锁被占是常态（多 worker 同时启动、别的进程正在 VACUUM），所以先重试
        几次再降级。降级只用配置里的 status，与库里可能不一致 —— 日志会写明。
        """
        if self._conn is None:
            logger.warning(
                "usage_stats | 统计库不可用，租户映射改由配置填充（未写库）"
            )
            self.seed_tenants_from_config(tenants)
            return False

        delays = (0.5, 1.0)
        for attempt in range(len(delays) + 1):
            try:
                self.upsert_tenants(tenants)
                return True
            except sqlite3.Error as e:
                if _is_locked_error(e) and attempt < len(delays):
                    logger.warning(
                        "usage_stats | 租户写库被占用，%.1fs 后重试（第 %d/%d 次）: %s",
                        delays[attempt], attempt + 2, len(delays) + 1, e,
                    )
                    time.sleep(delays[attempt])
                    continue
                logger.exception("usage_stats | 租户写库失败，改用配置里的租户填内存映射")
                break
            except Exception:
                logger.exception("usage_stats | 租户写库失败，改用配置里的租户填内存映射")
                break

        self.seed_tenants_from_config(tenants)
        return False

    def seed_tenants_from_config(self, tenants: Sequence[Dict[str, Any]]) -> None:
        """不碰数据库，直接用配置里的租户填内存映射（降级路径，见 sync_tenants）。"""
        self._install_tenant_maps([
            (t["id"], t["name"], _hash_key(t["api_key"]), t.get("status", "active"))
            for t in tenants
        ])

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
        """通过 tenant_id 获取租户详细信息。

        不加锁：_id_to_tenant 只会被 upsert_tenants 整体换引用，读到的要么是旧的
        完整一份、要么是新的完整一份，不存在半成品状态。这里**每个请求**都会走，
        加锁就等于让所有请求排队等 writer 线程提交事务。
        """
        if tenant_id == "default":
            return _DEFAULT_TENANT
        return self._id_to_tenant.get(tenant_id)

    # ------------------------------------------------------------------
    #  记录用量
    # ------------------------------------------------------------------

    def record_usage(self, record: UsageRecord) -> None:
        """记录一条用量（非阻塞）。

        只做一次入队，不碰数据库，所以可以直接在事件循环里调用。真正的 INSERT
        由 writer 线程攒批提交。
        """
        if self._conn is None or self._closing:
            # 统计库没起来（init_db 失败）时静默丢弃会让人以为「只是没数据」，
            # 所以第一次丢就报一条 —— 启动时容错不等于可以悄悄不记账。
            if self._conn is None and not self._no_db_warned:
                self._no_db_warned = True
                logger.error(
                    "usage_stats | 统计库不可用，此后所有用量记录都会被丢弃"
                    "（服务本身不受影响，检查启动日志里统计库初始化失败的原因）"
                )
            return
        try:
            self._queue.put_nowait(_to_row(record, self._prompt_max_chars))
        except queue.Full:
            self._dropped += 1
            # 每丢 1000 条报一次：既不刷屏，也不会被彻底忽略
            if self._dropped % 1000 == 1:
                logger.error(
                    "usage_stats | 写入队列已满（上限 %d），累计丢弃 %d 条用量记录",
                    self._MAX_QUEUE_SIZE, self._dropped,
                )

    def reconfigure(
        self,
        *,
        prompt_max_chars: int,
        flush_interval: float,
        batch_size: int,
    ) -> None:
        """热重载时更新截断/批量参数。

        只影响之后入队的记录；已经在队列里的按入队时的值写入（截断在入队时就做完了）。
        """
        self._prompt_max_chars = prompt_max_chars
        self._flush_interval = max(0.01, flush_interval)
        self._batch_size = max(1, batch_size)

    def flush(self, timeout: float = 5.0) -> bool:
        """阻塞到此前入队的记录全部落盘为止；有条目没写进去就返回 False。

        往队尾塞一个 Event 哨兵，writer 线程按 FIFO 处理到它时置位。队列是 FIFO 的，
        所以哨兵被置位时，排在它前面的记录必定已经提交（提交发生在置位之前）。

        返回值是「可信」的判据：线程死了、哨兵超时没置位、或者期间有记录写入失败，
        都返回 False。这三个信号以前都是 True，于是关机收尾和调用方都以为落盘了。
        """
        thread = self._writer_thread
        if thread is None:
            # 从未启动，或已经 close() 过。close() 会先 drain 再置 None，
            # 所以这里返回 True 的前提是「收尾已经跑完」。
            return True
        if not thread.is_alive():
            logger.error(
                "usage_stats | writer 线程已退出，队列里的记录不会落库（flush 不可信）"
            )
            return False
        failed_before = self._write_failed
        done = threading.Event()
        try:
            self._queue.put(done, timeout=timeout)
        except queue.Full:
            return False
        if not done.wait(timeout):
            return False
        if self._write_failed != failed_before:
            logger.error(
                "usage_stats | 本次 flush 期间有 %d 条记录写入失败",
                self._write_failed - failed_before,
            )
            return False
        return True

    # ------------------------------------------------------------------
    #  writer 线程：攒批 + 单事务提交
    # ------------------------------------------------------------------

    def _start_writer(self) -> None:
        self._stop.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="usage-stats-writer", daemon=True,
        )
        self._writer_thread.start()

    def _collect_batch(self, timeout: Optional[float]) -> List[Any]:
        """取一批待处理项。timeout=None 表示有多少取多少，绝不阻塞。

        两种「提前收手」的信号：
        - _WAKEUP：唯一目的就是把阻塞在 get() 上的线程叫醒，好让 close() 不用白等
          一个完整的 flush_interval。
        - flush() 塞进来的 Event 哨兵：它排在「本次 flush 之前入队的全部记录」后面，
          所以收到它就意味着该提交的都在这批里了。不在这里收手的话，队列恰好被取空时
          线程会继续按 flush_interval 等下一批，flush() 就要白等到超时 —— 等了多久
          取决于 batch_size 和 flush_interval，跟「数据落盘了没有」完全无关。
        """
        batch: List[Any] = []
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(batch) < self._batch_size:
            if deadline is None:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0 and batch:
                    break
                try:
                    item = self._queue.get(timeout=max(remaining, 0.0))
                except queue.Empty:
                    break
            batch.append(item)
            if item is _WAKEUP or isinstance(item, threading.Event):
                break
        return batch

    def _flush_batch(self, batch: List[Tuple[Any, ...]]) -> None:
        """整批写入；失败则回滚 + 逐条重试，把坏行单独摘掉。

        单位必须是「行」：一条坏记录（典型是上游给的 token 数超出 SQLite INTEGER
        范围，绑定时报 OverflowError）不该让同批的其它记录一起消失，更不该把
        writer 线程带走 —— 线程一死，此后所有用量都只入队不落库，flush() 还会
        因为线程不在而谎报成功。所以这里捕 Exception，不是只捕 sqlite3.Error。
        """
        with self._lock:
            # 关机时 close() 可能已经把连接收走了，判空必须在锁内
            if self._conn is None:
                # 已出队的这批确实写不进去了，但不能装作没发生：记账 + 告警
                self._write_failed += len(batch)
                logger.error(
                    "usage_stats | 写入连接已关闭，丢弃 %d 条未落库的用量记录（累计 %d 条）",
                    len(batch), self._write_failed,
                )
                return
            try:
                self._conn.executemany(_INSERT_SQL, batch)
                self._conn.commit()
                return
            except Exception as e:
                logger.error(
                    "usage_stats | 批量写入失败（%d 条），回滚后逐条重试: %s", len(batch), e
                )
                self._rollback_locked()
            # 逐条重试。慢，但只在失败路径上跑；目的是「只丢掉真正写不进去的那几条」。
            for row in batch:
                try:
                    self._conn.execute(_INSERT_SQL, row)
                    self._conn.commit()
                except Exception as e:
                    self._rollback_locked()
                    self._write_failed += 1
                    # 只记 req_id，不记整行：行里有 prompt（属于用户内容）
                    logger.error(
                        "usage_stats | 这条用量记录写不进去，已丢弃: %s | req_id=%r",
                        e, row[0] if row else None,
                    )

    def _rollback_locked(self) -> None:
        """回滚当前事务。调用方需持 _lock。

        必须回滚：executemany 报错后 Python sqlite3 的隐式事务仍然开着，坏行之前
        的行还挂在事务里，会被之后任意一次 commit 连带提交（那样日志报的丢弃条数
        就和事实对不上了）；而且悬空写事务会一直占着 WAL 写锁，把同库的其它 worker
        一起拖死。
        """
        try:
            self._conn.rollback()
        except Exception as e:
            logger.error("usage_stats | 回滚失败: %s", e)

    def _process(self, timeout: Optional[float]) -> bool:
        """取一批 → 落盘 → 置位其中的 flush 哨兵。返回是否处理了东西。

        这里兜住一切异常。writer 线程一旦死掉，后果是「此后所有用量静默不落库」，
        比丢掉某一批严重得多，所以宁可丢这一批、也不能让异常逃出去终止线程。
        异常路径上不置位哨兵：置了 flush() 就会在数据其实没落盘时报成功。
        """
        items: List[Any] = []
        try:
            items = self._collect_batch(timeout)
            if not items:
                return False
            # 只有 tuple 是数据行；threading.Event 是 flush 哨兵，_WAKEUP 是叫醒信号
            rows = [it for it in items if isinstance(it, tuple)]
            if rows:
                self._flush_batch(rows)
        except Exception:
            logger.exception(
                "usage_stats | 处理一批用量时出错，丢弃这一批（writer 线程继续服务）"
            )
            return bool(items)
        # 哨兵必须在数据提交之后才置位，否则 flush() 会在数据可见前就返回
        for it in items:
            if isinstance(it, threading.Event):
                it.set()
        return True

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            self._process(self._flush_interval)
        # 收尾：把停机前入队的记录全部落盘。
        # 复用 _process（而不是自己再写一遍 collect+flush）是有意的：收尾路径同样
        # 需要那层异常兜底，否则一个坏行就能让「关机前的最后一批」被整个丢掉。
        while self._process(None):
            pass

    # ------------------------------------------------------------------
    #  数据保留（清理超期记录）
    # ------------------------------------------------------------------

    # 单事务删除的行数上限。老库一次 DELETE 掉上百万行会写出一份同样大的回滚日志，
    # 期间一直占着写锁；分批删 + 分批提交，把单次事务压到毫秒级。
    _CLEANUP_BATCH_SIZE = 5000

    # 空闲页占比超过这个阈值才 VACUUM。VACUUM 会重写整个库文件（耗时、且需要等量
    # 的临时磁盘空间），所以只在「删了很多、文件明显虚胖」时才做。
    _VACUUM_FREE_PAGE_RATIO = 0.2

    # 跨进程认领清理用的锁文件（跟在库文件后面）。空文件，只是拿来 flock。
    _CLEANUP_LOCK_SUFFIX = ".cleanup.lock"

    @contextmanager
    def claim_global_cleanup(self) -> Iterator[bool]:
        """试着认领这一轮「全局清理」，yield 是否认领到了。

        清理（DELETE + VACUUM）是对**整个库**的操作，不是每个 worker 各自的事。
        多 worker 下 N 个进程同时跑同一份清理，除了互相抢写锁没别的作用：VACUUM
        要独占写锁重写整个库，别的进程在这几秒里一个字节都写不进去。生产上就是
        这样崩的 —— 4 个 worker 同时启动，其中一个在 VACUUM，另外三个的 init_db
        和租户写入全部超时报 database is locked，两个 worker 直接启动失败退出。

        用文件锁认领：进程被 kill -9 时锁由内核回收，不会留下需要人清的死锁。
        拿不到锁说明别的进程正在做这件事，本轮直接跳过 —— 清理间隔 24h，
        少跑一轮没有任何影响。
        """
        # 内存库（测试用）没有跨进程问题，也不该在仓库里生成锁文件
        if not self._db_path or self._db_path == ":memory:":
            yield True
            return

        try:
            import fcntl
        except ImportError:  # Windows 没有 flock：退回每个进程各跑一轮
            yield True
            return

        try:
            fd = open(self._db_path + self._CLEANUP_LOCK_SUFFIX, "a+")
        except OSError as e:
            # 锁文件建不出来（目录只读等）不该变成「清理永远不跑」
            logger.warning("usage_stats | 清理锁文件打不开（%s），本轮照常执行清理", e)
            yield True
            return

        try:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            yield True
        finally:
            fd.close()  # 关掉 fd 即释放 flock

    def cleanup_old_records(self, retention_days: int = 30) -> int:
        """
        删除 ts 早于 retention_days 天前的请求记录，返回删除的行数。

        retention_days <= 0 表示关闭清理（返回 0，不做任何事）。

        ts 是本地时间的 ISO 文本，定宽格式下字符串比较等价于时间比较，所以
        cutoff 直接用同样的格式拼出来即可，不需要在 SQL 里做时间转换。
        """
        if retention_days <= 0 or not self._conn:
            return 0

        cutoff = (datetime.now() - timedelta(days=retention_days)).strftime(_TS_FORMAT)

        deleted_total = 0
        while True:
            with self._lock:
                # 清理跑在线程池里，关机时可能和 close() 抢跑；conn 判空放在锁内
                # （close() 同样持锁），保证不会拿着已关闭的连接执行语句。
                if not self._conn:
                    break
                # rowid 子查询 + LIMIT：SQLite 默认没开 SQLITE_ENABLE_UPDATE_DELETE_LIMIT
                # 编译选项，DELETE ... LIMIT 不一定可用，走 rowid 是通用写法。
                cursor = self._conn.execute(
                    "DELETE FROM request_log WHERE rowid IN ("
                    "    SELECT rowid FROM request_log WHERE ts < ? LIMIT ?)",
                    (cutoff, self._CLEANUP_BATCH_SIZE),
                )
                batch = cursor.rowcount or 0
                self._conn.commit()
            deleted_total += batch
            if batch < self._CLEANUP_BATCH_SIZE:
                break

        if deleted_total:
            # 数据变了，全时段排名/厂商列表缓存必须失效，否则前端还按旧排名配色
            self._rank_cache = None
            self._provider_cache = None
        return deleted_total

    def clear_oversized_prompts(
        self, max_chars: int = 2048, batch_size: Optional[int] = None
    ) -> int:
        """
        把历史遗留的「超长 prompt」置为 NULL，返回处理的行数。

        用于截断策略上线之前落库的数据：那时 prompt 存的是请求体全文（实测平均
        269KB/行、最大 3.5MB），一度占到本库 89% 的体积。按行数清理清不掉它们 ——
        这些行本身还在保留期内 —— 只能就地抹掉字段内容。

        判定用「长度」而不是「时间」：凡是超过当前上限的行，都是现行策略下不会再
        产生的数据，与它哪天写入的无关。反过来，恰好合规的历史行不必动。

        batch_size 默认沿用 _CLEANUP_BATCH_SIZE。对**线上库**建议调小：单批是一个
        事务、全程占着写锁，而这些行本身很大（单批 5000 行 = 上 GB 的页改写），
        批次太大会把正在记录用量的写入挡在锁外，超过 busy_timeout 就是丢数据。
        批与批之间别的进程能插进来写。

        UPDATE 同样只把页标成空闲，文件不会自己变小，调用方需要接着
        vacuum(force=True) —— 注意这种批量 UPDATE 腾出的空间不进 freelist_count。
        """
        limit_chars = max_chars
        if limit_chars <= 0 or not self._conn:
            return 0
        batch_limit = batch_size or self._CLEANUP_BATCH_SIZE

        cleared_total = 0
        while True:
            with self._lock:
                if not self._conn:
                    break
                cursor = self._conn.execute(
                    "UPDATE request_log SET prompt = NULL WHERE rowid IN ("
                    "    SELECT rowid FROM request_log"
                    "     WHERE prompt IS NOT NULL AND LENGTH(prompt) > ? LIMIT ?)",
                    (limit_chars, batch_limit),
                )
                batch = cursor.rowcount or 0
                self._conn.commit()
            cleared_total += batch
            if batch < batch_limit:
                break
        return cleared_total

    def vacuum(self, force: bool = False) -> bool:
        """
        VACUUM 把删除后残留的空闲页还给文件系统。

        SQLite 的 DELETE/UPDATE 只把页标成空闲供后续插入复用，文件本身不会变小；
        不 VACUUM 的话 stats.db 会一直停在历史峰值大小。默认按空闲页占比节流
        （VACUUM 会重写整个库文件，日常清理那点增量不值得每次都跑）。
        返回是否真的执行了 VACUUM。

        force=True 跳过节流，无条件执行，给一次性维护用。**批量 UPDATE 腾出的空间
        不会出现在 freelist_count 里**：实测 2045 页的库清空 6MB prompt 后 freelist
        仍是 0，而 VACUUM 能把文件缩到 66 页。这种情况按阈值判断会直接跳过。
        """
        with self._lock:
            if not self._conn:
                return False
            if not force:
                page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
                free_pages = self._conn.execute("PRAGMA freelist_count").fetchone()[0]
                if page_count <= 0 or free_pages / page_count < self._VACUUM_FREE_PAGE_RATIO:
                    return False
                logger.info(
                    "stats VACUUM | 空闲页 %d/%d，开始收缩数据库（期间会占用等量临时磁盘）",
                    free_pages, page_count,
                )
            else:
                logger.info("stats VACUUM | 强制收缩数据库（期间会占用等量临时磁盘）")

            self._conn.execute("VACUUM")

            # WAL 下 VACUUM 重写后的内容先落进 WAL，主库文件要等下一次 checkpoint 才
            # 截断（实测不 checkpoint 的话 page_count 已降到 66 而文件仍是 8.3MB）。
            # 主动 TRUNCATE 一次，别让文件大小取决于后台时机。多 worker 下这个
            # checkpoint 可能拿不齐读者，拿不到就算了 —— 库本身已经紧凑。
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as e:
                logger.debug("stats VACUUM | wal_checkpoint 未完成: %s", e)
            return True

    def db_size_bytes(self) -> int:
        """数据库的逻辑大小（字节）。VACUUM 前后对比用这个。

        用 page_count * page_size，而不是 os.path.getsize(主库文件)：开了 WAL 之后
        新写入先落在 -wal 里，主库文件在 checkpoint 之前几乎不增长 —— 实测写入
        2000 行后主库仍是 4096 字节，量文件大小只会得到毫无意义的数字。逻辑页数
        才反映「库有多大」，也正是 VACUUM 真正回收的东西。
        """
        with self._lock:
            if not self._conn:
                return 0
            try:
                page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
                page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
            except sqlite3.Error:
                return 0
            return int(page_count) * int(page_size)

    def wal_size_bytes(self) -> int:
        """WAL 文件占用的磁盘空间（字节），仅用于日志/排障。

        正常情况下 WAL 会在 checkpoint 后复用，受 journal_size_limit 约束；
        持续增长说明有长事务或长查询一直挡着 checkpoint。
        """
        try:
            return os.path.getsize(self._db_path + "-wal")
        except OSError:
            return 0

    # ------------------------------------------------------------------
    #  查询统计
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        model: Optional[str] = None,
        tenant: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        """构造 WHERE 子句与参数（query_stats / query_daily_usage 共用）。"""
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
        return where_sql, params

    def _read_locked(self, run: Callable[[sqlite3.Connection], Any], default: Any) -> Any:
        """在 _read_lock 内取读连接并执行 run(conn)；连接已关闭则返回 default。

        读连接**必须**在锁内取。close() 也持这把锁来关闭并置空 _read_conn，
        如果在锁外读 self._read_conn，就可能拿到已经 close 掉的连接 —— sqlite3
        是 C 扩展，对已关闭连接执行不是抛异常而是直接段错误，整个 worker 一起没。
        """
        with self._read_lock:
            conn = self._read_conn
            if conn is None:
                return default
            return run(conn)

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
        if not self._read_conn:
            return {"stats": [], "period": {}}

        since, until = _normalize_range(since, until)

        # 外部名 -> SQL 表达式；key 即响应中输出的字段名
        select_cols: List[str] = []
        keys: List[str] = []
        group_cols: List[str] = []
        has_day = False

        for g in (part.strip() for part in group_by.split(",")):
            if not g:
                continue
            expr = _DIMENSION_SQL.get(g)
            if expr is None:
                # 保持历史行为：不认识的维度回退成按 model 分组（不报错），但记一条日志，
                # 否则调用方会拿到一份「看起来正常、实际答非所问」的数据。
                logger.warning(
                    "query_stats: 未知分组维度 %r，已回退为按 model 分组", g
                )
                expr, g = _DIMENSION_SQL["model"], "model"
            if g == "day":
                has_day = True
            select_cols.append(expr)
            # 保持历史契约：外部维度名 "tenant" 在响应里输出为 "tenant_id"
            keys.append("tenant_id" if g == "tenant" else g)
            group_cols.append(expr)

        if not group_cols:
            select_cols, keys, group_cols = ["model"], ["model"], ["model"]

        select_str = ", ".join(select_cols)
        group_str = ", ".join(group_cols)

        # 时间序列必须按时间升序；其余维度维持按请求数降序（即「排行榜」语义）
        if has_day:
            order_str = f"{_DIMENSION_SQL['day']} ASC, request_count DESC"
        else:
            order_str = "request_count DESC"

        where_sql, params = self._build_where(
            since=since, until=until, model=model, tenant=tenant, provider=provider
        )

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
            ORDER BY {order_str}
        """

        rows = self._read_locked(lambda c: c.execute(sql, params).fetchall(), [])

        stats = []
        for row in rows:
            item: Dict[str, Any] = {}
            for i, key in enumerate(keys):
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
            bounds = self._read_locked(
                lambda c: c.execute(
                    "SELECT MIN(ts), MAX(ts) FROM request_log"
                ).fetchone(),
                None,
            )
            if bounds and bounds[0]:
                if not since:
                    period["since"] = bounds[0]
                if not until:
                    period["until"] = bounds[1]

        return {"stats": stats, "period": period}

    # ------------------------------------------------------------------
    #  时间序列查询（按天 × 模型）
    # ------------------------------------------------------------------

    def _cached_rank(self, dim: str, cache_attr: str) -> List[str]:
        """按 dim 分组的全时段用量排名（降序）。dim 走 _DIMENSION_SQL 查表，
        外部输入永远进不了 SQL —— 这里的 KeyError 只可能是编程错误。

        这是一次全表 GROUP BY（随库增长线性变慢），所以结果按 dim 各自缓存 _RANK_TTL 秒。
        走读连接：扫表期间不会阻塞 writer 线程入库。
        """
        cache = getattr(self, cache_attr)
        now = time.monotonic()
        if cache and now - cache[0] < self._RANK_TTL:
            return cache[1]

        if not self._read_conn:
            return []
        col = _DIMENSION_SQL[dim]
        rows = self._read_locked(
            lambda c: c.execute(
                f"SELECT {col}, SUM(total_tokens) AS t FROM request_log "
                f"GROUP BY {col} ORDER BY t DESC, {col} ASC"
            ).fetchall(),
            [],
        )
        rank = [r[0] for r in rows]
        setattr(self, cache_attr, (now, rank))
        return rank

    def model_rank(self) -> List[str]:
        """
        全时段（不受任何筛选条件影响）的模型用量排名，按 total_tokens 降序。

        前端用它给模型分配固定配色槽位。用全时段排名而不是「当前区间的排名」，
        是为了让配色在切换时间范围时保持稳定：如果按区间排名分配，同一个模型在
        不同区间里会被涂成不同颜色，而读者一旦记住「deepseek-flash 是蓝色」，
        换一个区间它就变色，就是在误导人。

        注意同样要**忽略 provider 筛选**：模型名可以跨厂商（本库里 deepseek-flash
        同时在 deepseek 和 yunshu 下），按厂商过滤后再排名的话，换个厂商同一个模型
        就换一种颜色了。所以这里始终是全量的排名。
        """
        return self._cached_rank("model", "_rank_cache")

    def provider_rank(self) -> List[str]:
        """
        全时段的厂商列表，按 total_tokens 降序。

        给前端渲染厂商下拉框用。同样取全时段而不是当前区间：否则换时间范围时
        下拉框里的选项会忽增忽减，当前选中的那个甚至可能整个消失。
        """
        return self._cached_rank("provider", "_provider_cache")

    def query_daily_usage(
        self,
        *,
        metric: str = "total_tokens",
        bucket: str = "day",
        since: Optional[str] = None,
        until: Optional[str] = None,
        model: Optional[str] = None,
        tenant: Optional[str] = None,
        provider: Optional[str] = None,
        max_series: int = 60,
    ) -> Dict[str, Any]:
        """
        返回「时间 × 模型」的透视序列，供前端画堆叠柱状图 / 折线图。

        与 query_stats 的区别：
        - 固定按 bucket(时间) × model 两维分组，结果在 Python 侧透视成矩阵；
        - **补齐缺口**：区间内没有任何请求的桶也会出现在 buckets 里（值为 0）。否则
          没有请求的日子会从 x 轴上消失，折线图会把不相邻的两天直接连起来，读起来
          就是错的；
        - series 按区间内该指标合计降序排列；超过 max_series 的长尾合并成「其他」。

        返回::

            {
              "buckets": ["2026-08-27", ...],
              "series": [{"key": "glm-5.2", "total": 123, "values": [0, 5, ...]}],
              "metric": "total_tokens", "bucket": "day",
              "period": {"since": ..., "until": ...},
              "totals": {"request_count": ..., "total_tokens": ..., "active_models": ...},
              "bucket_filled": true,
              "truncated_series": 0,
              "model_rank": [...], "provider_rank": [...],
            }
        """
        if not self._read_conn:
            return {
                "buckets": [], "series": [], "metric": metric, "bucket": bucket,
                "period": {}, "totals": {}, "bucket_filled": False,
                "truncated_series": 0, "model_rank": [], "provider_rank": [],
            }

        if metric not in _METRIC_SQL:
            raise ValueError(f"未知指标 {metric!r}，可选: {sorted(_METRIC_SQL)}")
        if bucket not in _BUCKET_SQL:
            raise ValueError(f"未知分桶 {bucket!r}，可选: {sorted(_BUCKET_SQL)}")

        since, until = _normalize_range(since, until)
        where_sql, params = self._build_where(
            since=since, until=until, model=model, tenant=tenant, provider=provider
        )
        bucket_expr = _BUCKET_SQL[bucket]
        metric_expr = _METRIC_SQL[metric]

        # 1) 区间边界 + 总量。走 idx_request_ts，代价很低；用它来补缺口，
        #    这样就不必把桶字符串反解析回日期（strftime('%W') 在 Python 侧无法还原）。
        bounds = self._read_locked(
            lambda c: c.execute(
                f"SELECT MIN(ts), MAX(ts), COUNT(*), "
                f"COUNT(DISTINCT model), SUM(total_tokens) "
                f"FROM request_log {where_sql}",
                params,
            ).fetchone(),
            None,
        )

        totals: Dict[str, Any] = {
            "request_count": 0, "total_tokens": 0, "active_models": 0,
        }
        if not bounds or not bounds[0]:
            return {
                "buckets": [], "series": [], "metric": metric, "bucket": bucket,
                "period": {"since": since, "until": until},
                "totals": totals, "bucket_filled": False, "truncated_series": 0,
                "model_rank": self.model_rank(),
                "provider_rank": self.provider_rank(),
            }

        totals = {
            "request_count": bounds[2] or 0,
            "total_tokens": bounds[4] or 0,
            "active_models": bounds[3] or 0,
        }
        period = {"since": since or bounds[0], "until": until or bounds[1]}

        # 2) 主查询：bucket × model 的稀疏矩阵
        sql = f"""
            SELECT {bucket_expr} AS b, model, {metric_expr} AS v
            FROM request_log
            {where_sql}
            GROUP BY b, model
            ORDER BY b ASC
        """
        rows = self._read_locked(lambda c: c.execute(sql, params).fetchall(), [])

        observed: List[str] = []
        sparse: Dict[str, Dict[str, float]] = {}
        per_model_total: Dict[str, float] = {}
        for b, m, v in rows:
            b = str(b)
            if b not in sparse:
                sparse[b] = {}
                observed.append(b)
            sparse[b][m] = v or 0
            per_model_total[m] = per_model_total.get(m, 0) + (v or 0)

        # 3) 补缺口：从区间起点按分桶粒度走到终点，用与 SQL 相同的格式化成字符串
        try:
            start = datetime.fromisoformat(period["since"]).date()
            end = datetime.fromisoformat(period["until"]).date()
        except (ValueError, TypeError):
            # 时间参数格式不对时不猜，直接退回实际出现的桶
            logger.warning(
                "query_daily_usage: 无法解析区间 %r ~ %r，跳过缺口补齐",
                period["since"], period["until"],
            )
            start = end = None

        buckets: List[str] = []
        cursor = _first_bucket_date(start, bucket) if start else None
        # 安全阀：区间过宽时不补齐，避免返回一个几千列的轴（前端会自行降级分桶粒度）
        while cursor is not None and end is not None and cursor <= end \
                and len(buckets) <= _MAX_FILLED_BUCKETS:
            buckets.append(_format_bucket(cursor, bucket))
            cursor = _next_bucket(cursor, bucket)

        # 循环最多产出 _MAX_FILLED_BUCKETS + 1 个桶：超出即视为区间过宽，退回实际桶
        filled = 0 < len(buckets) <= _MAX_FILLED_BUCKETS
        if not filled:
            buckets = observed

        # 4) 透视 + 长尾合并
        ordered = sorted(per_model_total.items(), key=lambda kv: kv[1], reverse=True)
        truncated = max(0, len(ordered) - max_series)
        named = [m for m, _ in ordered[:max_series]] if truncated else [m for m, _ in ordered]

        series = []
        for m in named:
            values = [sparse.get(b, {}).get(m, 0) for b in buckets]
            series.append({"key": m, "total": per_model_total.get(m, 0), "values": values})

        if truncated:
            tail = [m for m, _ in ordered[max_series:]]
            values = [sum(sparse.get(b, {}).get(m, 0) for m in tail) for b in buckets]
            series.append({
                "key": "其他",
                "total": sum(per_model_total[m] for m in tail),
                "values": values,
            })

        return {
            "buckets": buckets,
            "series": series,
            "metric": metric,
            "bucket": bucket,
            "period": period,
            "totals": totals,
            "bucket_filled": filled,
            "truncated_series": truncated,
            # 全时段排名，供前端分配固定配色槽位（见 model_rank 的说明）
            "model_rank": self.model_rank(),
            # 全时段厂商列表，供前端渲染厂商下拉框（见 provider_rank 的说明）
            "provider_rank": self.provider_rank(),
        }

    # ------------------------------------------------------------------
    #  关闭
    # ------------------------------------------------------------------

    def close(self) -> None:
        """停 writer 线程、落盘剩余记录、关闭两条连接。

        writer 线程最多还要等一个 flush_interval 才会从 _collect_batch 醒过来，
        随后它会做一次完整的收尾 drain，所以这里 join 超时给得比较宽裕。
        它是 daemon 线程，即使超时也不会拖住进程退出。
        """
        # 先关闭收单，再唤醒 writer 去 drain。顺序不能反：drain 的唯一退出条件是
        # 「队列恰好为空」，如果这期间 record_usage 还在成功入队，那批记录会永远
        # 留在队列里（既没落库也没人再处理），而 close() 的语义是「落盘剩余记录」。
        self._closing = True
        self._stop.set()
        # 立刻叫醒 writer，否则它要等满一个 flush_interval 才会发现 _stop 已被置位
        try:
            self._queue.put_nowait(_WAKEUP)
        except queue.Full:
            # 队列满说明 writer 不在阻塞等待，它会立刻把队列排空并看到 _stop
            pass

        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self._writer_thread = None

        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        # 读连接也必须持 _read_lock 才关：所有查询路径都是在锁内 execute 的，
        # 不加锁就是「一边 execute 一边 close」，sqlite3 是 C 扩展，这种竞态
        # 不是抛异常而是直接段错误，整个 worker 进程会被带走。
        with self._read_lock:
            if self._read_conn is not None:
                try:
                    self._read_conn.close()
                except sqlite3.Error as e:
                    logger.debug("usage_stats | 关闭读连接失败: %s", e)
                self._read_conn = None
