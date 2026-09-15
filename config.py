"""
配置加载：只读 config.toml。

设计原则：
- 单一事实源：所有配置都来自 TOML 文件，**不支持环境变量覆盖配置值**。
- 唯一的外部输入是文件路径：默认 ``./config.toml``，可通过 ``CONFIG_FILE``
  环境变量指定其他路径（这只是"去哪儿读文件"，不构成对配置内容的覆盖）。
- 严格校验，配置错误一律 fail-fast 退出，避免运行时再炸。
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

PROTOCOL_OPENAI = "openai"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOLS = (PROTOCOL_OPENAI, PROTOCOL_ANTHROPIC)

# 鉴权风格
AUTH_STYLE_BEARER = "bearer"        # Authorization: Bearer <api_key>
AUTH_STYLE_ANTHROPIC = "anthropic"  # x-api-key + anthropic-version
AUTH_STYLE_NONE = "none"            # 不注入默认鉴权头（由 headers 子表自行处理）
AUTH_STYLES = (AUTH_STYLE_BEARER, AUTH_STYLE_ANTHROPIC, AUTH_STYLE_NONE)

# 协议默认 auth_style（厂商若不一致，可在子表 auth_style 字段覆盖）
DEFAULT_AUTH_STYLE_BY_PROTOCOL = {
    PROTOCOL_OPENAI: AUTH_STYLE_BEARER,
    PROTOCOL_ANTHROPIC: AUTH_STYLE_ANTHROPIC,
}


# ---------------------------------------------------------------------------
#  数据结构
# ---------------------------------------------------------------------------

@dataclass
class ProviderEndpoint:
    """一个 provider 在特定协议下的连接信息。"""
    base_url: str
    api_key: str
    # 鉴权风格：决定默认注入哪些鉴权头。
    # - "bearer"    -> Authorization: Bearer <api_key>
    # - "anthropic" -> x-api-key: <api_key> + anthropic-version: 2023-06-01
    # - "none"      -> 不注入默认鉴权头（自行用 headers 子表处理）
    auth_style: str = AUTH_STYLE_BEARER
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """
    一个上游服务商。

    每个 provider 可以同时支持多种协议；至少声明一个端点。
    某个协议字段为 None 表示该 provider 不原生支持该协议。
    """
    id: str
    openai: Optional[ProviderEndpoint] = None
    anthropic: Optional[ProviderEndpoint] = None
    # 手动指定的模型列表（可选）。非空时代替上游 GET /models 的自动发现，
    # 适用于不提供 /models 接口的上游（如火山方舟部分端点）。
    models: List[str] = field(default_factory=list)

    def endpoint(self, protocol: str) -> Optional[ProviderEndpoint]:
        if protocol == PROTOCOL_OPENAI:
            return self.openai
        if protocol == PROTOCOL_ANTHROPIC:
            return self.anthropic
        return None

    def supported_protocols(self) -> List[str]:
        return [p for p in PROTOCOLS if self.endpoint(p) is not None]


@dataclass
class ModelRoute:
    """一条模型路由。upstream_model 为空表示透传客户端模型名。"""
    provider_id: str
    upstream_model: Optional[str] = None


@dataclass
class TenantConfig:
    """一个租户的配置信息。"""
    id: str
    name: str
    api_key: str  # 明文，仅在启动时哈希后用于匹配
    status: str = "active"


@dataclass
class AdminConfig:
    """管理界面认证配置。

    二选一：
      - password       : 明文密码（仅开发期方便，与 tenant api_key 明文存储风格一致）
      - password_hash  : sha256(password) 的十六进制（生产推荐）
    两者都配置时优先校验 password_hash。
    """
    username: str
    password: Optional[str] = None
    password_hash: Optional[str] = None


# ---------------------------------------------------------------------------
#  辅助
# ---------------------------------------------------------------------------

def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"CRITICAL ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


# 顶层 provider 表里若出现这些字段，属于旧 schema 残留，明确拒绝：
# - base_url：OpenAI / Anthropic 协议根路径天然不同，不能共享；必须各自写在协议子表里
# - headers ：当前未支持顶层共享头；如有共享需求请显式写在各协议子表
# 注意：api_key **不在此列**——顶层 api_key 是合法的"共享默认值"，子表存在则覆盖。
_LEGACY_PROVIDER_FIELDS = ("base_url", "headers")


# ---------------------------------------------------------------------------
#  TOML 解析
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _die(
            f"配置文件不存在: {path}\n"
            f"  请在项目根创建 config.toml（参考 config.example.toml），\n"
            f"  或通过 CONFIG_FILE 环境变量指定路径。"
        )
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        _die(f"配置文件 TOML 解析失败: {path}\n  {e}")
    except OSError as e:
        _die(f"配置文件读取失败: {path}\n  {e}")


def _parse_endpoint(
    raw: Any,
    provider_id: str,
    protocol: str,
    shared_api_key: Optional[str],
) -> ProviderEndpoint:
    """
    解析单个协议子表。

    api_key 的取值顺序：
      1. 子表 ``api_key`` 字段（如果存在且非空，覆盖共享默认）
      2. provider 顶层共享的 ``shared_api_key``
    两处都没有则 fail-fast。
    """
    if not isinstance(raw, dict):
        _die(
            f"providers[{provider_id!r}].{protocol} 必须是表，"
            f"实际为 {type(raw).__name__}"
        )

    base = str(raw.get("base_url", "")).strip()
    if not base:
        _die(f"providers[{provider_id!r}].{protocol} 缺少 base_url")

    sub_key = str(raw.get("api_key", "")).strip()
    key = sub_key or (shared_api_key or "")
    if not key:
        _die(
            f"providers[{provider_id!r}].{protocol} 缺少 api_key"
            f"（子表与 provider 顶层均未提供）"
        )

    auth_style_raw = raw.get("auth_style")
    if auth_style_raw is None:
        auth_style = DEFAULT_AUTH_STYLE_BY_PROTOCOL[protocol]
    else:
        if not isinstance(auth_style_raw, str):
            _die(
                f"providers[{provider_id!r}].{protocol}.auth_style 必须是字符串"
            )
        auth_style = auth_style_raw.strip().lower()
        if auth_style not in AUTH_STYLES:
            _die(
                f"providers[{provider_id!r}].{protocol}.auth_style={auth_style_raw!r} 非法，"
                f"允许值: {list(AUTH_STYLES)}"
            )

    headers = raw.get("headers", {}) or {}
    if not isinstance(headers, dict):
        _die(
            f"providers[{provider_id!r}].{protocol}.headers 必须是表（key/value 字符串）"
        )

    return ProviderEndpoint(
        base_url=_ensure_trailing_slash(base),
        api_key=key,
        auth_style=auth_style,
        headers={str(k): str(v) for k, v in headers.items()},
    )


def _parse_providers(raw: Any) -> List[ProviderConfig]:
    if not raw:
        _die("配置缺少 [[providers]]：至少需要声明一个上游服务商。")
    if not isinstance(raw, list):
        _die("配置 providers 必须是数组（请使用 [[providers]] 数组表）。")

    seen_ids: set[str] = set()
    providers: List[ProviderConfig] = []

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            _die(f"providers[{i}] 必须是表（table），实际为 {type(item).__name__}")

        pid = str(item.get("id", "")).strip()
        if not pid:
            _die(f"providers[{i}] 缺少 id")
        if "/" in pid:
            _die(
                f"providers[{i}].id 不能包含 '/' 字符"
                f"（用于 /v1/models 的 provider_id/model_name 命名空间路由）: {pid!r}"
            )
        if pid in seen_ids:
            _die(f"providers 中存在重复 id: {pid!r}")
        seen_ids.add(pid)

        # 顶层不允许 base_url / headers（含义不可共享或当前未支持）
        legacy_hits = [f for f in _LEGACY_PROVIDER_FIELDS if f in item]
        if legacy_hits:
            _die(
                f"providers[{pid!r}] 顶层不允许字段 {legacy_hits}：\n"
                f"  - base_url：两个协议根路径不同，请写在 [providers.openai] / [providers.anthropic] 各自的子表里。\n"
                f"  - headers ：暂不支持顶层共享，请写在各协议子表里。\n"
                f"  详见 config.example.toml。"
            )

        # 顶层 api_key 作为共享默认值（可选）；任一协议子表里的 api_key 会覆盖它。
        shared_api_key = str(item.get("api_key", "")).strip() or None

        # 手动模型列表（可选）：非空时代替该 provider 的 /models 自动发现。
        raw_models = item.get("models")
        models: List[str] = []
        if raw_models is not None:
            if not isinstance(raw_models, list):
                _die(
                    f"providers[{pid!r}].models 必须是字符串数组"
                    f"（models = [\"model-a\", \"model-b\"]）"
                )
            seen_models: set[str] = set()
            for j, m in enumerate(raw_models):
                if not isinstance(m, str) or not m.strip():
                    _die(f"providers[{pid!r}].models[{j}] 必须是非空字符串")
                mm = m.strip()
                if mm not in seen_models:
                    seen_models.add(mm)
                    models.append(mm)

        openai_ep = (
            _parse_endpoint(item["openai"], pid, PROTOCOL_OPENAI, shared_api_key)
            if "openai" in item else None
        )
        anthropic_ep = (
            _parse_endpoint(item["anthropic"], pid, PROTOCOL_ANTHROPIC, shared_api_key)
            if "anthropic" in item else None
        )

        if openai_ep is None and anthropic_ep is None:
            _die(
                f"providers[{pid!r}] 至少需要声明一个协议子表："
                f"[providers.openai] 或 [providers.anthropic]"
            )

        providers.append(ProviderConfig(
            id=pid,
            openai=openai_ep,
            anthropic=anthropic_ep,
            models=models,
        ))

    return providers


def _parse_tenants(raw: Any) -> List[TenantConfig]:
    """解析 [[tenants]]。允许为空（表示无租户配置，所有请求落到 'default'）。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        _die("配置 tenants 必须是数组（请使用 [[tenants]] 数组表）。")

    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    tenants: List[TenantConfig] = []

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            _die(f"tenants[{i}] 必须是表（table），实际为 {type(item).__name__}")

        tid = str(item.get("id", "")).strip()
        if not tid:
            _die(f"tenants[{i}] 缺少 id")
        if tid == "default":
            _die(f"tenants[{i}].id 不能为 'default'（保留给未匹配租户的占位）")
        if tid in seen_ids:
            _die(f"tenants 中存在重复 id: {tid!r}")
        seen_ids.add(tid)

        name = str(item.get("name", "")).strip()
        if not name:
            _die(f"tenants[{tid!r}] 缺少 name")

        api_key = str(item.get("api_key", "")).strip()
        if not api_key:
            _die(f"tenants[{tid!r}] 缺少 api_key")
        if api_key in seen_keys:
            _die(f"tenants[{tid!r}].api_key 与其他租户重复")
        seen_keys.add(api_key)

        status = str(item.get("status", "active")).strip().lower()
        if status not in ("active", "disabled"):
            _die(f"tenants[{tid!r}].status 必须是 'active' 或 'disabled'")

        tenants.append(TenantConfig(id=tid, name=name, api_key=api_key, status=status))

    return tenants


def _parse_model_routes(
    raw: Any,
    known_provider_ids: set[str],
) -> Dict[str, List[ModelRoute]]:
    """
    解析 [model_routes] 段，value 接受两种形式：

    1. inline 表：{ provider = "p", model = "m" }     -> 单条路由（当前唯一使用）
    2. 数组表  ：[[model_routes."x"]] ...              -> 多条路由（未来 failover 预留）

    model 字段可省略，省略时透传客户端模型名。
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        _die("model_routes 必须是表（[model_routes] 段）")

    result: Dict[str, List[ModelRoute]] = {}
    for client_model, value in raw.items():
        if not isinstance(client_model, str) or not client_model.strip():
            _die(f"model_routes 存在无效 key: {client_model!r}")
        client_model = client_model.strip()

        items = value if isinstance(value, list) else [value]
        routes: List[ModelRoute] = []
        for j, entry in enumerate(items):
            if not isinstance(entry, dict):
                _die(
                    f"model_routes[{client_model!r}][{j}] 必须是表，"
                    f"实际为 {type(entry).__name__}"
                )

            pid = str(entry.get("provider", "")).strip()
            if not pid:
                _die(f"model_routes[{client_model!r}][{j}] 缺少 provider")
            if pid not in known_provider_ids:
                _die(
                    f"model_routes[{client_model!r}][{j}] 指向未知 provider={pid!r}"
                    f"（已知: {sorted(known_provider_ids)}）"
                )

            upstream_model = entry.get("model")
            if upstream_model is not None:
                if not isinstance(upstream_model, str) or not upstream_model.strip():
                    _die(
                        f"model_routes[{client_model!r}][{j}].model 必须是非空字符串或省略"
                    )
                upstream_model = upstream_model.strip()

            routes.append(ModelRoute(provider_id=pid, upstream_model=upstream_model))

        if routes:
            result[client_model] = routes
    return result


def _parse_admin(raw: Any) -> Optional[AdminConfig]:
    """解析 [admin] 段。允许缺失（返回 None，表示未启用管理界面认证）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _die("admin 必须是表（[admin] 段）")

    username = str(raw.get("username", "")).strip()
    if not username:
        _die("admin.username 不能为空")

    password = str(raw.get("password", "")).strip() or None
    password_hash = str(raw.get("password_hash", "")).strip().lower() or None

    if not password and not password_hash:
        _die("admin 至少需要配置 password 或 password_hash 之一")

    return AdminConfig(username=username, password=password, password_hash=password_hash)


# ---------------------------------------------------------------------------
#  Settings
# ---------------------------------------------------------------------------

CONFIG_FILE_ENV = "CONFIG_FILE"
DEFAULT_CONFIG_PATH = "config.toml"


class Settings:
    PROVIDERS: Dict[str, ProviderConfig]
    DEFAULT_PROVIDER_ID: str
    MODEL_ROUTES: Dict[str, List[ModelRoute]]
    TENANTS: List[TenantConfig]
    ADMIN: Optional[AdminConfig]
    STATS_DB: str

    LOG_LEVEL: str
    MAX_BODY_SIZE: int
    STATS_RETENTION_DAYS: int
    STATS_PROMPT_MAX_CHARS: int
    STATS_FLUSH_INTERVAL: float
    STATS_BATCH_SIZE: int
    POOL_MAX_CONNECTIONS: int
    POOL_MAX_KEEPALIVE: int
    POOL_KEEPALIVE_EXPIRY: float
    CONFIG_WATCH_INTERVAL: float
    TIMEOUT_CONNECT: float
    TIMEOUT_READ: float
    TIMEOUT_WRITE: float
    TIMEOUT_POOL: float

    CONFIG_PATH: Path

    def __init__(self):
        path = Path(os.getenv(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)).expanduser()
        self.CONFIG_PATH = path
        data = _load_toml(path)
        self._populate_from(data)

    def _populate_from(self, data: Dict[str, Any]) -> None:
        """从已解析的 TOML dict 填充各字段。reload 时复用。

        解析阶段只写局部变量，全部成功后才一次性写回 self。任何 _die（配置非法）
        都发生在 self 被改动之前，因此 reload 失败时运行态保持原样 —— 后台配置
        监听依赖这个性质：配置文件被外部改坏，不能让正在服务的进程变成半新半旧。
        """
        # --- 1. providers ---
        providers_list = _parse_providers(data.get("providers"))
        providers = {p.id: p for p in providers_list}

        # --- 2. default_provider ---
        default = str(data.get("default_provider", "")).strip()
        if not default:
            default = providers_list[0].id
        elif default not in providers:
            _die(
                f"default_provider={default!r} 不在 providers 中"
                f"（已知: {sorted(providers)}）"
            )

        # --- 3. model_routes ---
        model_routes = _parse_model_routes(
            data.get("model_routes"),
            known_provider_ids=set(providers),
        )

        # --- 4. tenants ---
        tenants = _parse_tenants(data.get("tenants"))

        # --- 5. admin ---
        admin = _parse_admin(data.get("admin"))

        # --- 6. 其他全局参数 ---
        log_level = str(data.get("log_level", "INFO")).upper()
        max_body_size = int(data.get("max_body_size", 15728640))
        stats_db = str(data.get("stats_db", "./stats.db")).strip()
        # 统计数据保留天数：超过这个天数的 request_log 记录会被后台任务清理。
        # 设为 <=0 表示关闭自动清理（保留全部历史）。
        stats_retention_days = int(data.get("stats_retention_days", 30))
        # 入库的 prompt 前缀长度上限（字符）。请求体全文对计费没有价值，却曾经占到
        # stats.db 的 88%；<=0 表示完全不存 prompt。
        stats_prompt_max_chars = int(data.get("stats_prompt_max_chars", 2048))
        # 用量写库的批量参数：攒够 stats_batch_size 条或等满 stats_flush_interval 秒
        # 就提交一次。写库发生在后台线程，不影响请求延迟。
        stats_flush_interval = float(data.get("stats_flush_interval", 0.5))
        stats_batch_size = int(data.get("stats_batch_size", 500))
        # 每个 provider 的上游连接池上限。httpx 默认 max_connections=100，对网关来说
        # 是硬性并发天花板：第 101 个请求只能在池里排队，超时即 502。
        pool_max_connections = int(data.get("pool_max_connections", 500))
        pool_max_keepalive = int(data.get("pool_max_keepalive", 100))
        # 空闲连接保持时间。httpx 默认 5s，对「持续有流量但速率不高」的网关会导致
        # 频繁重连（含 TLS 握手），调大明显省事。
        pool_keepalive_expiry = float(data.get("pool_keepalive_expiry", 60.0))
        # 后台配置监听轮询间隔（秒）。多 worker 下靠它把配置变更广播到所有进程；
        # <=0 关闭监听（仅单进程或由外部负责重启时使用）。
        config_watch_interval = float(data.get("config_watch_interval", 2.0))

        timeouts = data.get("timeouts", {}) or {}
        if not isinstance(timeouts, dict):
            _die("timeouts 必须是表（[timeouts] 段）")
        timeout_connect = float(timeouts.get("connect", 5.0))
        timeout_read = float(timeouts.get("read", 300.0))
        timeout_write = float(timeouts.get("write", 20.0))
        timeout_pool = float(timeouts.get("pool", 10.0))

        # --- 提交阶段：以上全部解析通过，才改动 self ---
        self.PROVIDERS = providers
        self.DEFAULT_PROVIDER_ID = default
        self.MODEL_ROUTES = model_routes
        self.TENANTS = tenants
        self.ADMIN = admin

        self.LOG_LEVEL = log_level
        self.MAX_BODY_SIZE = max_body_size
        self.STATS_DB = stats_db
        self.STATS_RETENTION_DAYS = stats_retention_days
        self.STATS_PROMPT_MAX_CHARS = stats_prompt_max_chars
        self.STATS_FLUSH_INTERVAL = stats_flush_interval
        self.STATS_BATCH_SIZE = stats_batch_size
        self.POOL_MAX_CONNECTIONS = pool_max_connections
        self.POOL_MAX_KEEPALIVE = pool_max_keepalive
        self.POOL_KEEPALIVE_EXPIRY = pool_keepalive_expiry
        self.CONFIG_WATCH_INTERVAL = config_watch_interval

        self.TIMEOUT_CONNECT = timeout_connect
        self.TIMEOUT_READ = timeout_read
        self.TIMEOUT_WRITE = timeout_write
        self.TIMEOUT_POOL = timeout_pool

    def to_dict(self) -> Dict[str, Any]:
        """把当前 settings 序列化为可写回 TOML 的 dict（与配置文件结构一致）。

        api_key 保留明文（仅用于写回磁盘，不经网络传输；GET /admin/api/config
        会在路由层做脱敏）。
        """
        providers_out: List[Dict[str, Any]] = []
        for p in self.PROVIDERS.values():
            item: Dict[str, Any] = {"id": p.id}

            openai_key = p.openai.api_key if p.openai else None
            anthropic_key = p.anthropic.api_key if p.anthropic else None
            keys = [k for k in (openai_key, anthropic_key) if k is not None]
            shared = len(keys) > 0 and len(set(keys)) == 1

            if shared:
                item["api_key"] = keys[0]

            if p.models:
                item["models"] = list(p.models)

            if p.openai:
                openai_tbl: Dict[str, Any] = {"base_url": p.openai.base_url}
                if not shared:
                    openai_tbl["api_key"] = p.openai.api_key
                if p.openai.auth_style != DEFAULT_AUTH_STYLE_BY_PROTOCOL[PROTOCOL_OPENAI]:
                    openai_tbl["auth_style"] = p.openai.auth_style
                if p.openai.headers:
                    openai_tbl["headers"] = dict(p.openai.headers)
                item["openai"] = openai_tbl

            if p.anthropic:
                anth_tbl: Dict[str, Any] = {"base_url": p.anthropic.base_url}
                if not shared:
                    anth_tbl["api_key"] = p.anthropic.api_key
                if p.anthropic.auth_style != DEFAULT_AUTH_STYLE_BY_PROTOCOL[PROTOCOL_ANTHROPIC]:
                    anth_tbl["auth_style"] = p.anthropic.auth_style
                if p.anthropic.headers:
                    anth_tbl["headers"] = dict(p.anthropic.headers)
                item["anthropic"] = anth_tbl

            providers_out.append(item)

        routes_out: Dict[str, Any] = {}
        for client_model, routes in self.MODEL_ROUTES.items():
            # 当前只使用 inline 单条路由形式
            r = routes[0]
            entry: Dict[str, str] = {"provider": r.provider_id}
            if r.upstream_model is not None:
                entry["model"] = r.upstream_model
            routes_out[client_model] = entry

        tenants_out: List[Dict[str, Any]] = [
            {"id": t.id, "name": t.name, "api_key": t.api_key, "status": t.status}
            for t in self.TENANTS
        ]

        result: Dict[str, Any] = {
            "log_level": self.LOG_LEVEL,
            "max_body_size": self.MAX_BODY_SIZE,
            "default_provider": self.DEFAULT_PROVIDER_ID,
            "stats_db": self.STATS_DB,
            "stats_retention_days": self.STATS_RETENTION_DAYS,
            "stats_prompt_max_chars": self.STATS_PROMPT_MAX_CHARS,
            "stats_flush_interval": self.STATS_FLUSH_INTERVAL,
            "stats_batch_size": self.STATS_BATCH_SIZE,
            "pool_max_connections": self.POOL_MAX_CONNECTIONS,
            "pool_max_keepalive": self.POOL_MAX_KEEPALIVE,
            "pool_keepalive_expiry": self.POOL_KEEPALIVE_EXPIRY,
            "config_watch_interval": self.CONFIG_WATCH_INTERVAL,
            "timeouts": {
                "connect": self.TIMEOUT_CONNECT,
                "read": self.TIMEOUT_READ,
                "write": self.TIMEOUT_WRITE,
                "pool": self.TIMEOUT_POOL,
            },
            "providers": providers_out,
            "model_routes": routes_out,
            "tenants": tenants_out,
        }

        if self.ADMIN is not None:
            admin_tbl: Dict[str, Any] = {"username": self.ADMIN.username}
            if self.ADMIN.password_hash:
                admin_tbl["password_hash"] = self.ADMIN.password_hash
            elif self.ADMIN.password:
                admin_tbl["password"] = self.ADMIN.password
            result["admin"] = admin_tbl

        return result


def reload() -> None:
    """从磁盘重读 config.toml，原地替换模块级 ``settings`` 的字段。

    用于热重载：保留 ``settings`` 对象引用不变，只更新其属性，避免其他模块
    已绑定的 ``from config import settings`` 失效。
    """
    data = _load_toml(settings.CONFIG_PATH)
    settings._populate_from(data)


settings = Settings()
