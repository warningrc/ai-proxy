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
    STATS_DB: str

    LOG_LEVEL: str
    MAX_BODY_SIZE: int
    TIMEOUT_CONNECT: float
    TIMEOUT_READ: float
    TIMEOUT_WRITE: float
    TIMEOUT_POOL: float

    CONFIG_PATH: Path

    def __init__(self):
        path = Path(os.getenv(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)).expanduser()
        self.CONFIG_PATH = path

        data = _load_toml(path)

        # --- 1. providers ---
        providers_list = _parse_providers(data.get("providers"))
        self.PROVIDERS = {p.id: p for p in providers_list}

        # --- 2. default_provider ---
        default = str(data.get("default_provider", "")).strip()
        if not default:
            default = providers_list[0].id
        elif default not in self.PROVIDERS:
            _die(
                f"default_provider={default!r} 不在 providers 中"
                f"（已知: {sorted(self.PROVIDERS)}）"
            )
        self.DEFAULT_PROVIDER_ID = default

        # --- 3. model_routes ---
        self.MODEL_ROUTES = _parse_model_routes(
            data.get("model_routes"),
            known_provider_ids=set(self.PROVIDERS),
        )

        # --- 4. tenants ---
        self.TENANTS = _parse_tenants(data.get("tenants"))

        # --- 5. 其他全局参数 ---
        self.LOG_LEVEL = str(data.get("log_level", "INFO")).upper()
        self.MAX_BODY_SIZE = int(data.get("max_body_size", 15728640))
        self.STATS_DB = str(data.get("stats_db", "./stats.db")).strip()

        timeouts = data.get("timeouts", {}) or {}
        if not isinstance(timeouts, dict):
            _die("timeouts 必须是表（[timeouts] 段）")
        self.TIMEOUT_CONNECT = float(timeouts.get("connect", 5.0))
        self.TIMEOUT_READ = float(timeouts.get("read", 300.0))
        self.TIMEOUT_WRITE = float(timeouts.get("write", 20.0))
        self.TIMEOUT_POOL = float(timeouts.get("pool", 10.0))


settings = Settings()
