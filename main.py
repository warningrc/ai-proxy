import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

import config as config_module
from config import (
    AUTH_STYLE_ANTHROPIC,
    AUTH_STYLE_BEARER,
    AUTH_STYLE_NONE,
    PROTOCOL_ANTHROPIC,
    PROTOCOL_OPENAI,
    ModelRoute,
    ProviderEndpoint,
    settings,
)
from schemas import ClaudeChatRequest
from converters import claude_to_openai_request, openai_to_claude_response, openai_to_claude_stream
from request_log import (
    json_preview,
    log_json_preview,
    log_stream_chunk_debug,
    summarize_openai_request,
    summarize_openai_response,
)
from usage_stats import UsageStats, UsageRecord, _is_locked_error, _mask_key
from admin_reloader import register_reload_callback
from config_watcher import ConfigWatcher, register_watcher
from sse import aiter_raw_lines

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ai-proxy")

# Anthropic 协议默认要求的版本头。可被 endpoint.headers 覆盖。
ANTHROPIC_VERSION = "2023-06-01"

# http_clients[provider_id][protocol] -> AsyncClient
http_clients: Dict[str, Dict[str, httpx.AsyncClient]] = {}

# 用量统计
usage_stats = UsageStats(
    settings.STATS_DB,
    prompt_max_chars=settings.STATS_PROMPT_MAX_CHARS,
    flush_interval=settings.STATS_FLUSH_INTERVAL,
    batch_size=settings.STATS_BATCH_SIZE,
)

# 上游模型发现缓存：namespaced_id（"provider_id/model_name"）→ 模型条目 dict。
# 仅用于 /v1/models 聚合展示；路由本身由命名空间前缀解析，不依赖此缓存。
_upstream_models_cache: Dict[str, Dict[str, Any]] = {}
_upstream_models_cache_ts: float = 0.0
_upstream_models_lock = asyncio.Lock()
MODELS_DISCOVERY_TTL = 60.0


def _invalidate_upstream_models_cache() -> None:
    """清空上游模型发现缓存（热重载后 provider 可能变化，需要重建）。"""
    global _upstream_models_cache, _upstream_models_cache_ts
    _upstream_models_cache = {}
    _upstream_models_cache_ts = 0.0


def _default_auth_headers(auth_style: str, api_key: str) -> Dict[str, str]:
    """根据 auth_style 派发默认鉴权头（可被 endpoint.headers 覆盖）。"""
    if auth_style == AUTH_STYLE_BEARER:
        return {"Authorization": f"Bearer {api_key}"}
    if auth_style == AUTH_STYLE_ANTHROPIC:
        return {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
    if auth_style == AUTH_STYLE_NONE:
        return {}
    raise ValueError(f"unknown auth_style: {auth_style!r}")


def _build_client(protocol: str, endpoint: ProviderEndpoint) -> httpx.AsyncClient:
    headers = _default_auth_headers(endpoint.auth_style, endpoint.api_key)
    headers.update(endpoint.headers or {})
    return httpx.AsyncClient(
        base_url=endpoint.base_url,
        headers=headers,
        timeout=httpx.Timeout(
            connect=settings.TIMEOUT_CONNECT,
            read=settings.TIMEOUT_READ,
            write=settings.TIMEOUT_WRITE,
            pool=settings.TIMEOUT_POOL,
        ),
        # httpx 默认 max_connections=100，对网关来说就是硬性并发天花板：第 101 个
        # 请求只能在池里排队干等，超过 TIMEOUT_POOL 直接 502。必须显式放大。
        limits=httpx.Limits(
            max_connections=settings.POOL_MAX_CONNECTIONS,
            max_keepalive_connections=settings.POOL_MAX_KEEPALIVE,
            keepalive_expiry=settings.POOL_KEEPALIVE_EXPIRY,
        ),
    )


def build_http_clients(s) -> Dict[str, Dict[str, httpx.AsyncClient]]:
    """根据 settings 构建全部 provider 的 http clients。

    lifespan 启动与热重载共用此函数。
    """
    clients: Dict[str, Dict[str, httpx.AsyncClient]] = {}
    for pid, p in s.PROVIDERS.items():
        clients[pid] = {}
        if p.openai:
            clients[pid][PROTOCOL_OPENAI] = _build_client(PROTOCOL_OPENAI, p.openai)
        if p.anthropic:
            # Anthropic 端点路径里 v1 由代码自动拼，base_url 不应再带 /v1。
            if p.anthropic.base_url.rstrip("/").endswith("/v1"):
                logger.warning(
                    "Provider %r anthropic.base_url=%r 末尾带了 '/v1'；"
                    "代码会再拼一次 v1/messages，最终路径会出现两次 v1。"
                    "请把 base_url 末尾的 /v1 去掉。",
                    pid, p.anthropic.base_url,
                )
            clients[pid][PROTOCOL_ANTHROPIC] = _build_client(PROTOCOL_ANTHROPIC, p.anthropic)
    return clients


async def _close_http_clients(clients: Dict[str, Dict[str, httpx.AsyncClient]]) -> None:
    """关闭并清空给定 clients 集合（用于热重载时释放旧连接）。"""
    for pid, by_proto in clients.items():
        for protocol, client in by_proto.items():
            try:
                await client.aclose()
            except Exception as e:
                logger.debug("reload | aclose %s/%s failed: %s", pid, protocol, e)


# 客户端世代号：每次重建 http_clients 就 +1。新集合属于新世代，退役集合保留
# 它当「现役」时的那个世代号。
_client_generation: int = 0

# 世代号 -> 从这个世代开始处理、目前还没结束的请求数。只在事件循环线程里增减，
# 不需要锁。
#
# 为什么不能按时间猜宽限期：httpx 的 read timeout 是「每次读取」的上限，不是整个
# 响应的预算。一个持续吐 token 的 SSE 流，只要每次间隔小于 read timeout，就能活过
# 任意长的宽限期 —— 之前按 max(TIMEOUT_READ, 30) 定时关闭，热重载会把正在流式返回
# 的会话拦腰掐断（响应头早发出去了，只能看到流中途断掉，状态码已经补救不了）。
# 改成按在途请求判断：某个世代的集合只可能被「从它这一代或更早世代开始处理」的
# 请求用到 —— 请求可能过了几个 await 才真正去取 client，那时现役的已经是更新的
# 集合了；反过来，一个请求绝不可能取到比它开始处理时更旧的集合。于是当所有在途
# 请求的世代号都大于它时，它就绝不会再被使用，可以安全关闭。在途请求总会有尽头
# （读超时/客户端断开/cancel），所以不会无限期攒着不关。
_inflight_by_generation: Dict[int, int] = {}

# 退役集合的扫描间隔。只影响「用完多久后真正释放」，不影响正确性。
_RETIRED_SWEEP_INTERVAL = 5.0

# 已退役、但可能还有在途请求在用的旧 clients：(世代号, clients)。
_retired_clients: List[Tuple[int, Dict[str, Dict[str, httpx.AsyncClient]]]] = []


def _mark_client_use() -> int:
    """登记「本请求开始于当前世代」，返回世代号（结束时要交给 _unmark_client_use）。"""
    gen = _client_generation
    _inflight_by_generation[gen] = _inflight_by_generation.get(gen, 0) + 1
    return gen


def _unmark_client_use(gen: int) -> None:
    left = _inflight_by_generation.get(gen, 0) - 1
    if left > 0:
        _inflight_by_generation[gen] = left
    else:
        _inflight_by_generation.pop(gen, None)


def _retire_http_clients(
    clients: Dict[str, Dict[str, httpx.AsyncClient]], generation: int
) -> None:
    if clients:
        _retired_clients.append((generation, clients))


async def _sweep_retired_clients(force: bool = False) -> None:
    """关闭已不可能再有请求使用的退役 clients。

    ``force=True``（进程退出）时不再等待在途请求，直接关干净。
    """
    keep: List[Tuple[int, Dict[str, Dict[str, httpx.AsyncClient]]]] = []
    for gen, clients in _retired_clients:
        # 这个集合只可能被「从 gen 或更早世代开始」的请求用着
        in_use = any(
            _inflight_by_generation.get(g) for g in range(gen + 1)
        )
        if in_use and not force:
            keep.append((gen, clients))
        else:
            await _close_http_clients(clients)
    _retired_clients[:] = keep


async def _sweep_retired_clients_loop() -> None:
    """周期性回收退役 clients（判据见 _inflight_by_generation 的说明）。"""
    while True:
        await asyncio.sleep(_RETIRED_SWEEP_INTERVAL)
        try:
            await _sweep_retired_clients()
        except Exception:
            logger.exception("reload | sweep retired clients failed")


async def reload_runtime() -> None:
    """热重载运行态：换掉 http_clients，按重载后的 settings 重建，并同步租户。

    调用前 config.reload() 应已完成（settings 已更新为新配置）。
    """
    global http_clients, _client_generation
    old = http_clients
    old_generation = _client_generation
    _client_generation += 1
    new = build_http_clients(settings)
    http_clients = new  # 切换引用；旧 _get_client 走新 dict
    # 旧 clients 延迟退役：此刻可能还有请求正拿它读上游 SSE
    _retire_http_clients(old, old_generation)

    # provider 可能变化，丢弃旧的模型发现缓存，下次 /v1/models 重新发现
    _invalidate_upstream_models_cache()

    # 统计相关的可调参数也跟着配置走（prompt 截断长度、批量大小等）
    usage_stats.reconfigure(
        prompt_max_chars=settings.STATS_PROMPT_MAX_CHARS,
        flush_interval=settings.STATS_FLUSH_INTERVAL,
        batch_size=settings.STATS_BATCH_SIZE,
    )

    # 别再直接调 upsert_tenants：它抛异常会把热重载截断在半路（新 key 没生效），
    # 而且异常冒到 admin 接口就是 500、冒到配置 watcher 就是一条 reload failed。
    # 写不进统计库不该算热重载失败 —— 配置和内存映射都已经换好了。
    _sync_tenants("reload")

    logger.info("reload | http_clients rebuilt: %s", list(http_clients.keys()))

register_reload_callback(reload_runtime)


async def _reload_from_watcher() -> None:
    """配置监视器发现 config.toml 被改时调用（多 worker 下各进程各自执行一遍）。

    ``config.reload()`` 校验失败会走 ``_die`` -> ``sys.exit``，在后台任务里
    直接抛 SystemExit 会带走整个 worker，所以必须在这里兜住：配置坏了就继续
    用旧配置服务，把错误写进日志等人来修。
    """
    try:
        config_module.reload()
    except SystemExit as e:
        logger.error(
            "config watcher | 配置校验失败，继续沿用旧配置（已忽略本次变更）: %s", e
        )
        return
    await reload_runtime()


# ---------------------------------------------------------------------------
#  统计数据保留：定时清理超期记录
# ---------------------------------------------------------------------------

# 清理间隔。启动时先跑一轮（老库可能已经很大，且 VACUUM 只在启动后这一轮
# 最可能真正触发），之后每 24h 一轮。
STATS_CLEANUP_INTERVAL_SECONDS = 24 * 3600


# 统计库初始化失败后的重试节奏（秒）。多 worker 同时启动时 N 个进程一起抢写锁，
# 失败基本都是「别人正在写」，等一下就过去了。
_STATS_INIT_RETRY_DELAYS = (0.5, 1.0, 2.0)


def _init_stats() -> bool:
    """初始化统计库，返回是否可用。**绝不抛异常**。

    统计是旁路能力：记不了账绝不能拦着网关启动。原先这里让异常直接冒到 lifespan
    外面，于是一个正在 VACUUM（独占写锁）的进程就足以让其它 worker 反复起不来 ——
    uvicorn 只会打印 "Application startup failed. Exiting." 然后不停重启这个 worker。
    """
    for attempt in range(len(_STATS_INIT_RETRY_DELAYS) + 1):
        try:
            usage_stats.init_db()
            return True
        except sqlite3.OperationalError as e:
            if _is_locked_error(e) and attempt < len(_STATS_INIT_RETRY_DELAYS):
                delay = _STATS_INIT_RETRY_DELAYS[attempt]
                logger.warning(
                    "Startup | 统计库被占用，%.1fs 后重试（第 %d/%d 次）: %s",
                    delay, attempt + 2, len(_STATS_INIT_RETRY_DELAYS) + 1, e,
                )
                time.sleep(delay)
                continue
            logger.exception("Startup | 统计库初始化失败，本次启动不记录统计")
            return False
        except Exception:
            logger.exception("Startup | 统计库初始化失败，本次启动不记录统计")
            return False
    logger.error("Startup | 统计库持续被占用，本次启动不记录统计")
    return False


def _sync_tenants(where: str) -> None:
    """把配置里的租户同步给 usage_stats（写库 + 填内存映射）。

    ``usage_stats.sync_tenants`` 不会抛异常：写不进库就降级成只用配置填内存映射。
    这一点对鉴权是必须的 —— 鉴权只认那份内存映射，库写不进去时若映射还空着，
    所有租户的 key 都会被判成非法，一个记账问题就升级成了整个网关 401。
    """
    if not settings.TENANTS:
        return
    tenants = [
        {"id": t.id, "name": t.name, "api_key": t.api_key, "status": t.status}
        for t in settings.TENANTS
    ]
    ids = [t["id"] for t in tenants]
    if usage_stats.sync_tenants(tenants):
        logger.info("%s | tenants synced: %s", where, ids)
    else:
        logger.warning("%s | tenants 未写入统计库，鉴权已改用配置里的租户: %s", where, ids)


def _stats_cleanup_once() -> None:
    """同步执行一轮「删除超期记录 + 按需 VACUUM」，由线程池调用。"""
    retention_days = settings.STATS_RETENTION_DAYS
    # 清理是全局任务：多 worker 下由拿到了锁的那个进程跑，其余直接跳过。
    # 不加这一步的话，N 个进程会同时 DELETE + VACUUM，VACUUM 期间独占写锁，
    # 别的进程连租户都同步不进去（见 usage_stats.claim_global_cleanup）。
    with usage_stats.claim_global_cleanup() as claimed:
        if not claimed:
            logger.debug("stats cleanup | 已被其它进程认领，本轮跳过")
            return
        size_before = usage_stats.db_size_bytes()
        deleted = usage_stats.cleanup_old_records(retention_days)
        # 只有删过东西才可能产生大块空闲页，没删就不用白跑一次 VACUUM
        vacuumed = usage_stats.vacuum() if deleted else False
        logger.info(
            "stats cleanup | retention=%dd | deleted=%d rows | %.1fMB -> %.1fMB%s",
            retention_days,
            deleted,
            size_before / 1e6,
            usage_stats.db_size_bytes() / 1e6,
            " | vacuumed" if vacuumed else "",
        )


async def _stats_cleanup_loop() -> None:
    """后台定时清理任务；单轮失败不影响服务，下一轮继续。"""
    while True:
        try:
            await asyncio.get_running_loop().run_in_executor(None, _stats_cleanup_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stats cleanup 失败，将在下一轮重试")
        await asyncio.sleep(STATS_CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_clients, _client_generation
    _client_generation += 1
    http_clients = build_http_clients(settings)

    providers_summary: Dict[str, Dict[str, Dict[str, str]]] = {}
    for pid, p in settings.PROVIDERS.items():
        summary: Dict[str, Dict[str, str]] = {}
        if p.openai:
            summary[PROTOCOL_OPENAI] = {
                "base_url": p.openai.base_url,
                "auth_style": p.openai.auth_style,
            }
        if p.anthropic:
            summary[PROTOCOL_ANTHROPIC] = {
                "base_url": p.anthropic.base_url,
                "auth_style": p.anthropic.auth_style,
            }
        providers_summary[pid] = summary

    routes_summary = {
        client_model: [
            {"provider": r.provider_id, "model": r.upstream_model or "<passthrough>"}
            for r in routes
        ]
        for client_model, routes in settings.MODEL_ROUTES.items()
    }
    logger.info(
        "Startup | providers=%s | default_provider=%r | log_level=%s | routes=%s",
        providers_summary,
        settings.DEFAULT_PROVIDER_ID,
        settings.LOG_LEVEL,
        routes_summary or "<none>",
    )

    # 统计库不可用也必须继续启动（记不了账不等于不能服务），而租户映射无论如何
    # 都要填上 —— 鉴权只认它，空了就是所有 key 401。
    stats_ok = _init_stats()
    _sync_tenants("Startup")

    # 后台预热上游模型发现缓存（best-effort，不阻塞启动）
    warmup_task = asyncio.create_task(discover_upstream_models(force=True))

    # 后台轮询 config.toml：多 worker 下其余进程靠它跟上管理接口改的配置
    config_watcher = ConfigWatcher(
        settings.CONFIG_PATH,
        settings.CONFIG_WATCH_INTERVAL,
        _reload_from_watcher,
    )
    register_watcher(config_watcher)
    config_watcher.start()

    # 后台回收热重载时退役的旧 http clients（等宽限期过了再关，避免掐断在途流）
    sweep_task = asyncio.create_task(_sweep_retired_clients_loop())

    # 后台定时清理超期统计数据（stats_retention_days <= 0 表示关闭）
    cleanup_task: Optional[asyncio.Task] = None
    if settings.STATS_RETENTION_DAYS <= 0:
        logger.info("Startup | stats cleanup disabled (stats_retention_days<=0)")
    elif not stats_ok:
        # 库都没起来，跑清理只会每轮报一次「库不可用」，没有意义
        logger.warning("Startup | stats cleanup disabled (统计库不可用)")
    else:
        cleanup_task = asyncio.create_task(_stats_cleanup_loop())
        logger.info(
            "Startup | stats cleanup enabled: retention=%dd, interval=%dh",
            settings.STATS_RETENTION_DAYS,
            STATS_CLEANUP_INTERVAL_SECONDS // 3600,
        )

    yield

    warmup_task.cancel()
    try:
        await warmup_task
    except (asyncio.CancelledError, Exception):
        pass

    await config_watcher.stop()

    sweep_task.cancel()
    try:
        await sweep_task
    except (asyncio.CancelledError, Exception):
        pass

    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):
            pass

    await _close_http_clients(http_clients)
    http_clients = {}
    # 进程正要退出，不必再等在途请求跑完，把退役的旧 clients 一并关干净
    await _sweep_retired_clients(force=True)
    usage_stats.close()
    logger.info("Shutdown | all upstream HTTP clients closed")


class _ClientLifetimeMiddleware:
    """把「本世代有在途请求」的标记持到整个响应体发完为止。

    用纯 ASGI 中间件而不是 ``@app.middleware("http")``：StreamingResponse 的分片
    是在 handler 返回**之后**才由 starlette 逐个发出去的，纯 ASGI 包装到的
    ``await self.app(...)`` 要等响应体全部发完才返回，标记的覆盖范围正好是
    「从请求进来到最后一片 SSE 发出去」；这正是退役 client 不能被关的时间窗。
    若挂在 handler 上，返回 StreamingResponse 的那一刻标记就注销了，长流照样被掐断。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        generation = _mark_client_use()
        try:
            await self.app(scope, receive, send)
        finally:
            _unmark_client_use(generation)


app = FastAPI(lifespan=lifespan)
app.add_middleware(_ClientLifetimeMiddleware)

# 挂载管理界面路由（Web 管理 + 用量统计仪表盘）
from admin_api import router as admin_router  # noqa: E402

app.include_router(admin_router)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def _prompt_preview(body: bytes, limit: int) -> Optional[str]:
    """请求体的入库预览：只解码前 limit 个字符。

    prompt 字段只用于排障，而请求体全文对计费毫无价值 —— 实测它曾占到 stats.db 的
    88%（平均 300KB/行，最大 3.5MB）。整段 decode 出来的 str 要一直活到入库为止，
    在事件循环上这笔开销很实在。这里按 UTF-8 最长 4 字节/字符取前缀再解码，
    结果一定不超过 limit。
    """
    if limit <= 0:
        return None
    return body[: limit * 4].decode("utf-8", errors="replace")[:limit]


def _authenticate_request(request: Request, req_id: str) -> str:
    """从请求头提取 Bearer key 或 x-api-key，匹配并校验租户身份。"""
    raw_key: Optional[str] = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw_key = auth[7:].strip()
    if not raw_key:
        raw_key = request.headers.get("x-api-key", "").strip() or None

    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Authentication credentials were not provided."
        )

    tenant_id = usage_stats.resolve_tenant(raw_key)
    if tenant_id == "default":
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key."
        )
    return tenant_id


def resolve_model_route(model: str | None, req_id: str) -> Tuple[str, str | None]:
    """
    把客户端传入的 model 解析为 (provider_id, upstream_model)。

    解析优先级：
    1. 命中 MODEL_ROUTES：取第一条路由（未来 failover 在此扩展为遍历）。
       若 upstream_model 为空则透传客户端模型名。
    2. 命名空间形式 "provider_id/model_name"：前缀是已知 provider 时，
       路由到该 provider，剩余部分作为上游模型名（/v1/models 上游模型
       统一用这种前缀名返回）。按第一个 '/' 切分：provider id 已在配置
       校验阶段禁止包含 '/'，因此剩余部分可安全包含 '/'（如 "MiniMax/M3"）。
    3. 未命中：使用默认 provider，且模型名原样透传。
    """
    if not model:
        return settings.DEFAULT_PROVIDER_ID, model

    routes = settings.MODEL_ROUTES.get(model)
    if routes:
        if len(routes) > 1:
            logger.warning(
                "[%s] model route | %r has %d routes; multi-provider failover not yet implemented, using first",
                req_id, model, len(routes),
            )
        route = routes[0]
        upstream_model = route.upstream_model or model
        logger.info(
            "[%s] model route | %r -> provider=%r model=%r",
            req_id, model, route.provider_id, upstream_model,
        )
        return route.provider_id, upstream_model

    # 命名空间解析：仅当前缀是已知 provider 且剩余模型名非空时才生效。
    if "/" in model:
        prefix, upstream = model.split("/", 1)
        if prefix in settings.PROVIDERS and upstream:
            logger.info(
                "[%s] model route | %r -> provider=%r model=%r (namespaced)",
                req_id, model, prefix, upstream,
            )
            return prefix, upstream

    return settings.DEFAULT_PROVIDER_ID, model


def _get_client(provider_id: str, protocol: str) -> Optional[httpx.AsyncClient]:
    """返回 client；找不到返回 None（调用方决定 502 还是 500）。"""
    return http_clients.get(provider_id, {}).get(protocol)


def _append_query(path: str, query_string: str) -> str:
    """把客户端原请求的 query string 附加到上游相对路径。"""
    if not query_string:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}{query_string}"


def _protocol_unsupported_response(
    req_id: str, provider_id: str, protocol: str
) -> JSONResponse:
    """provider 未声明该协议端点时统一的 502 响应。"""
    msg = (
        f"Provider {provider_id!r} does not expose {protocol!r} protocol"
    )
    logger.warning("[%s] %s", req_id, msg)
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "type": "provider_protocol_unsupported",
                "message": msg,
            }
        },
    )


# ---------------------------------------------------------------------------
#  Usage 提取辅助函数
# ---------------------------------------------------------------------------

def _record_usage(record: UsageRecord) -> None:
    """记录一条用量。

    直接调 usage_stats.record_usage 即可：它内部只做一次队列入队，不做 IO，
    不会阻塞事件循环。原先这里要绕 run_in_executor 是因为当时每条记录都会同步
    写一次 SQLite 并 commit。
    """
    usage_stats.record_usage(record)


def _check_tenant_allowance(tenant_id: str) -> None:
    """
    检查租户的配额和准入状态。
    如果租户被禁用，抛出 403 异常。
    """
    info = usage_stats.get_tenant_info_by_id(tenant_id)
    if info and info.status != "active":
        raise HTTPException(
            status_code=403,
            detail=f"Tenant account {tenant_id!r} ({info.name}) is {info.status!r}",
        )
    # 未来可在此扩展 quota 消费检查


def _record_anthropic_usage(
    req_id: str, tenant_id: str, provider_id: str,
    model: str, duration_ms: float, data: dict,
    prompt: Optional[str] = None, status_code: int = 200,
    client_ip: Optional[str] = None,
) -> None:
    """从 Anthropic 非流式响应中提取 usage 并记录。"""
    u = data.get("usage") or {}
    _record_usage(UsageRecord(
        req_id=req_id,
        tenant_id=tenant_id,
        provider=provider_id,
        model=model,
        endpoint="messages",
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_tokens=u.get("cache_read_input_tokens", 0),
        cache_creation_tokens=u.get("cache_creation_input_tokens", 0),
        duration_ms=duration_ms,
        prompt=prompt,
        status_code=status_code,
        client_ip=client_ip,
    ))


def _record_openai_usage(
    req_id: str, tenant_id: str, provider_id: str,
    model: str, endpoint: str, duration_ms: float, data: dict,
    prompt: Optional[str] = None, status_code: int = 200,
    client_ip: Optional[str] = None,
) -> None:
    """从 OpenAI 非流式响应中提取 usage 并记录。"""
    u = data.get("usage") or {}
    cache_read = 0
    details = u.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cache_read = details.get("cached_tokens", 0)
    if cache_read == 0:
        cache_read = u.get("prompt_cache_hit_tokens", 0)
    _record_usage(UsageRecord(
        req_id=req_id,
        tenant_id=tenant_id,
        provider=provider_id,
        model=model,
        endpoint=endpoint,
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        cache_read_tokens=cache_read,
        cache_creation_tokens=0,
        duration_ms=duration_ms,
        prompt=prompt,
        status_code=status_code,
        client_ip=client_ip,
    ))


async def stream_generator(response: httpx.Response, req_id: str, t_upstream_start: float):
    """Parse OpenAI SSE lines; log first chunk, optional usage, and summary on close."""
    chunk_index = 0
    last_usage = None
    upstream_id = None
    upstream_model = None
    parse_errors = 0
    stream_error: str | None = None
    try:
        try:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        logger.info(
                            "[%s] upstream SSE | event=[DONE] | chunks_parsed=%s | parse_errors=%s",
                            req_id,
                            chunk_index,
                            parse_errors,
                        )
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        logger.warning(
                            "[%s] upstream SSE | bad_json | err=%s | head=%r",
                            req_id,
                            e,
                            data_str[:120],
                        )
                        continue

                    chunk_index += 1
                    log_stream_chunk_debug(req_id, chunk_index, data)

                    if chunk_index == 1:
                        upstream_id = data.get("id")
                        upstream_model = data.get("model")
                        logger.info(
                            "[%s] upstream SSE | first_chunk | http_already_200 | id=%r model=%r",
                            req_id,
                            upstream_id,
                            upstream_model,
                        )

                    if data.get("usage"):
                        last_usage = data["usage"]

                    fr = None
                    for ch in data.get("choices") or []:
                        fr = fr or ch.get("finish_reason")
                    if fr:
                        logger.info(
                            "[%s] upstream SSE | chunk | finish_reason=%r | usage_this_chunk=%s",
                            req_id,
                            fr,
                            data.get("usage"),
                        )

                    yield data
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.StreamError) as exc:
            # 上游连接在中途断开（常见于负载均衡/上游模型超时）。
            # 此时响应头已发给客户端，不能再返回 502，这里记录并优雅结束流。
            stream_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[%s] upstream SSE | stream_aborted | chunks=%s | err=%s",
                req_id,
                chunk_index,
                stream_error,
            )
    finally:
        try:
            await response.aclose()
        except Exception as close_exc:
            logger.debug("[%s] upstream SSE | aclose_error | %s", req_id, close_exc)
        elapsed_ms = (time.perf_counter() - t_upstream_start) * 1000
        logger.info(
            "[%s] upstream SSE | stream_closed | chunks=%s parse_errors=%s id=%r model=%r "
            "last_usage=%s duration_ms=%.0f stream_error=%s",
            req_id,
            chunk_index,
            parse_errors,
            upstream_id,
            upstream_model,
            last_usage,
            elapsed_ms,
            stream_error,
        )


async def _anthropic_stream_passthrough(
    response: httpx.Response, req_id: str, t_start: float,
    tenant_id: str = "default", provider_id: str = "", model: str = "",
    prompt_str: Optional[str] = None, client_ip: Optional[str] = None,
):
    """转发 Anthropic SSE 流，同时解析 usage 事件用于统计。"""
    bytes_count = 0
    stream_error: Optional[str] = None
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0

    try:
        try:
            current_event = ""
            async for raw_line in aiter_raw_lines(response):
                # 原样透传，不做 str 往返：上面切出来的是 bytes，直接补个换行发走。
                # len() 数出来的就是上游给的真实字节数（含我们补的那个 \n）。
                bytes_count += len(raw_line) + 1
                yield raw_line + b"\n"

                # SSE 解析：提取 event type 和 data
                if raw_line.startswith(b"event: "):
                    current_event = raw_line[7:].decode("utf-8", "replace").strip()
                elif raw_line.startswith(b"data: ") and current_event:
                    try:
                        # json.loads 直接吃 bytes，内部按 UTF-8 解码
                        data = json.loads(raw_line[6:])
                        if current_event == "message_start":
                            msg = data.get("message", {})
                            u = msg.get("usage", {})
                            input_tokens += u.get("input_tokens", 0)
                            cache_read_tokens += u.get("cache_read_input_tokens", 0)
                            cache_creation_tokens += u.get("cache_creation_input_tokens", 0)
                        elif current_event == "message_delta":
                            u = data.get("usage", {})
                            output_tokens += u.get("output_tokens", 0)
                    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                        pass
                elif not raw_line:
                    current_event = ""
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.StreamError) as exc:
            stream_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[%s] anthropic passthrough | stream_aborted | bytes=%s | err=%s",
                req_id, bytes_count, stream_error,
            )
    finally:
        try:
            await response.aclose()
        except Exception as close_exc:
            logger.debug("[%s] anthropic passthrough | aclose_error | %s", req_id, close_exc)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "[%s] anthropic passthrough | stream_closed | bytes=%s | duration_ms=%.0f | "
            "input=%d output=%d cache_read=%d | stream_error=%s",
            req_id, bytes_count, elapsed_ms,
            input_tokens, output_tokens, cache_read_tokens, stream_error,
        )
        _record_usage(UsageRecord(
            req_id=req_id,
            tenant_id=tenant_id,
            provider=provider_id,
            model=model,
            endpoint="messages",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            duration_ms=elapsed_ms,
            prompt=prompt_str,
            status_code=200,
            client_ip=client_ip,
        ))


async def _handle_messages_anthropic_passthrough(
    req_id: str,
    provider_id: str,
    upstream_model: Optional[str],
    claude_body_dict: dict,
    query_string: str = "",
    tenant_id: str = "default",
    prompt_str: Optional[str] = None,
    client_ip: Optional[str] = None,
):
    """provider 原生支持 Anthropic 协议 → 仅替换 model 字段，直接透传 body 与 query。"""
    client = _get_client(provider_id, PROTOCOL_ANTHROPIC)
    if client is None:
        # 启动期已保证至少一个端点；理论上不会到这
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
        ))
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_ANTHROPIC)

    if upstream_model is not None:
        claude_body_dict["model"] = upstream_model

    # Anthropic 协议端点路径为 /v1/messages（v1 是端点的一部分，按官方 SDK 约定，
    # 不由 base_url 携带）。OpenAI 那边相反，v1 由 base_url 携带。
    upstream_path = _append_query("v1/messages", query_string)
    is_stream = bool(claude_body_dict.get("stream"))
    logger.info(
        "[%s] /v1/messages | provider=%r | mode=anthropic-passthrough | model=%r | stream=%s | upstream_path=%r",
        req_id, provider_id, claude_body_dict.get("model"), is_stream, upstream_path,
    )
    log_json_preview(logger, "proxy -> upstream | anthropic_request_json", req_id, claude_body_dict)

    t0 = time.perf_counter()

    if is_stream:
        built = client.build_request("POST", upstream_path, json=claude_body_dict)
        try:
            r = await client.send(built, stream=True)
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "[%s] upstream | connection_failed | after_ms=%.0f | %s",
                req_id, elapsed_ms, exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=claude_body_dict.get("model", ""), endpoint="messages",
                prompt=prompt_str, status_code=502, client_ip=client_ip,
                duration_ms=elapsed_ms
            ))
            raise HTTPException(status_code=502, detail="Upstream connection failed")

        if r.status_code != 200:
            body = await r.aread()
            text = body.decode("utf-8", errors="replace")
            logger.error(
                "[%s] upstream | error | status=%s | preview=%r",
                req_id, r.status_code, text[:2000],
            )
            try:
                err_content = json.loads(text)
            except json.JSONDecodeError:
                err_content = {
                    "type": "error",
                    "error": {"type": "upstream_error", "message": text[:500]},
                }
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=claude_body_dict.get("model", ""), endpoint="messages",
                prompt=prompt_str, status_code=r.status_code, client_ip=client_ip,
                duration_ms=(time.perf_counter() - t0) * 1000
            ))
            return JSONResponse(status_code=r.status_code, content=err_content)

        return StreamingResponse(
            _anthropic_stream_passthrough(
                r, req_id, t0,
                tenant_id=tenant_id, provider_id=provider_id,
                model=claude_body_dict.get("model", ""),
                prompt_str=prompt_str, client_ip=client_ip,
            ),
            media_type="text/event-stream",
        )

    # 非流式
    try:
        resp = await client.post(upstream_path, json=claude_body_dict)
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "[%s] upstream | connection_failed | after_ms=%.0f | %s",
            req_id, elapsed_ms, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream connection failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] upstream | anthropic passthrough | status=%s | duration_ms=%.0f",
        req_id, resp.status_code, elapsed_ms,
    )

    if resp.status_code != 200:
        try:
            err_content = resp.json()
        except Exception:
            err_content = {
                "type": "error",
                "error": {"type": "upstream_error", "message": resp.text[:500]},
            }
        logger.error(
            "[%s] upstream | error | status=%s | %s",
            req_id, resp.status_code, json_preview(err_content),
        )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=resp.status_code, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error(
            "[%s] upstream | invalid_json_body | %s | preview=%r",
            req_id, e, resp.text[:500],
        )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    _record_anthropic_usage(req_id, tenant_id, provider_id,
                            claude_body_dict.get("model", ""), elapsed_ms, data,
                            prompt=prompt_str, status_code=200, client_ip=client_ip)
    return JSONResponse(content=data)


async def _handle_messages_openai_convert(
    req_id: str,
    provider_id: str,
    upstream_model: Optional[str],
    claude_body_dict: dict,
    tenant_id: str = "default",
    prompt_str: Optional[str] = None,
    client_ip: Optional[str] = None,
):
    """provider 仅支持 OpenAI 协议 → 走 claude→openai 转换 + openai→claude 反转。"""
    client = _get_client(provider_id, PROTOCOL_OPENAI)
    if client is None:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
        ))
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_OPENAI)

    # 这条分支需要严格 Claude schema 才能转换
    try:
        validated_req = ClaudeChatRequest(**claude_body_dict)
        claude_body = validated_req.model_dump(exclude_none=True)
    except Exception as e:
        logger.warning("[%s] reject | validation | %s", req_id, e)
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=claude_body_dict.get("model") or "unknown", endpoint="messages",
            prompt=prompt_str, status_code=422, client_ip=client_ip,
        ))
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")

    openai_req = claude_to_openai_request(claude_body)
    if upstream_model is not None:
        openai_req["model"] = upstream_model

    logger.info(
        "[%s] /v1/messages | provider=%r | mode=openai-convert | %s",
        req_id, provider_id, summarize_openai_request(openai_req),
    )
    log_json_preview(logger, "proxy -> upstream | openai_request_json", req_id, openai_req)

    if openai_req.get("stream", False):
        built = client.build_request("POST", "chat/completions", json=openai_req)
        logger.info(
            "[%s] proxy -> upstream | POST %schat/completions | provider=%r | model=%r | stream=true",
            req_id, client.base_url, provider_id, openai_req.get("model"),
        )
        t0 = time.perf_counter()
        try:
            r = await client.send(built, stream=True)
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "[%s] upstream | connection_failed | after_ms=%.0f | %s",
                req_id,
                elapsed_ms,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=openai_req.get("model", ""), endpoint="messages",
                prompt=prompt_str, status_code=502, client_ip=client_ip,
                duration_ms=elapsed_ms
            ))
            raise HTTPException(status_code=502, detail="Upstream connection failed")

        logger.info(
            "[%s] upstream | response_headers | status=%s | content_type=%r | "
            "x_request_id=%r | cf_ray=%r",
            req_id,
            r.status_code,
            r.headers.get("content-type"),
            r.headers.get("x-request-id") or r.headers.get("request-id"),
            r.headers.get("cf-ray"),
        )

        if r.status_code != 200:
            body = await r.aread()
            text = body.decode("utf-8", errors="replace")
            preview_obj: dict | str
            try:
                preview_obj = json.loads(text)
            except json.JSONDecodeError:
                preview_obj = {"raw": text[:2000]}
            logger.error(
                "[%s] upstream | error_body | status=%s | bytes=%s | preview=%s",
                req_id,
                r.status_code,
                len(body),
                json_preview(preview_obj),
            )
            try:
                err_content = json.loads(text)
            except json.JSONDecodeError:
                err_content = {
                    "type": "error",
                    "error": {"type": "upstream_error", "message": text[:500]},
                }
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=openai_req.get("model", ""), endpoint="messages",
                prompt=prompt_str, status_code=r.status_code, client_ip=client_ip,
                duration_ms=(time.perf_counter() - t0) * 1000
            ))
            return JSONResponse(status_code=r.status_code, content=err_content)

        def on_usage(u: dict):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _record_usage(UsageRecord(
                req_id=req_id,
                tenant_id=tenant_id,
                provider=provider_id,
                model=openai_req.get("model", ""),
                endpoint="messages",
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cache_read_tokens=u.get("cache_read_tokens", 0),
                cache_creation_tokens=u.get("cache_creation_tokens", 0),
                duration_ms=elapsed_ms,
                prompt=prompt_str,
                status_code=200,
                client_ip=client_ip,
            ))

        return StreamingResponse(
            openai_to_claude_stream(
                stream_generator(r, req_id, t0),
                on_usage_done=on_usage,
            ),
            media_type="text/event-stream",
        )

    logger.info(
        "[%s] proxy -> upstream | POST %schat/completions | provider=%r | model=%r | stream=false",
        req_id, client.base_url, provider_id, openai_req.get("model"),
    )
    t0 = time.perf_counter()
    try:
        resp = await client.post("chat/completions", json=openai_req)
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "[%s] upstream | connection_failed | after_ms=%.0f | %s",
            req_id,
            elapsed_ms,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream connection failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] upstream | response_headers | status=%s | content_type=%r | "
        "x_request_id=%r | duration_ms=%.0f",
        req_id,
        resp.status_code,
        resp.headers.get("content-type"),
        resp.headers.get("x-request-id") or resp.headers.get("request-id"),
        elapsed_ms,
    )

    if resp.status_code != 200:
        try:
            err_content = resp.json()
            logger.error(
                "[%s] upstream | error_json | status=%s | %s",
                req_id,
                resp.status_code,
                json_preview(err_content),
            )
        except Exception:
            err_content = {
                "type": "error",
                "error": {"type": "upstream_error", "message": resp.text[:500]},
            }
            logger.error(
                "[%s] upstream | error_text | status=%s | preview=%r",
                req_id,
                resp.status_code,
                resp.text[:800],
            )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=resp.status_code, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error("[%s] upstream | invalid_json_body | %s | preview=%r", req_id, e, resp.text[:500])
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="messages",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    logger.info("[%s] upstream | ok | %s", req_id, summarize_openai_response(data))
    log_json_preview(logger, "upstream | response_body", req_id, data)

    _record_openai_usage(
        req_id=req_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        model=openai_req.get("model", ""),
        endpoint="messages",
        duration_ms=elapsed_ms,
        data=data,
        prompt=prompt_str,
        status_code=200,
        client_ip=client_ip,
    )
    return JSONResponse(content=openai_to_claude_response(data))


@app.post("/v1/messages")
async def handle_messages(request: Request):
    if not http_clients:
        logger.error("Server not initialized properly")
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    client_ip = _get_client_ip(request)
    prompt_str = None
    tenant_id = "default"
    provider_id = settings.DEFAULT_PROVIDER_ID
    model_name = "unknown"

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_BODY_SIZE:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="messages", prompt=f"<body size {content_length} exceeds limit>",
            status_code=413, client_ip=client_ip
        ))
        logger.warning("[%s] reject | body too large (header) | %s bytes", req_id, content_length)
        raise HTTPException(status_code=413, detail="Request entity too large")

    # 鉴权放在读 body 之前。未通过鉴权就没必要把最多 15MB 的请求体读进内存、解码、
    # 解析 JSON，更没必要写一条带请求体的记录 —— 否则拿着错 key 的洪水请求，单个
    # 成本比正常请求还高。代价只是这条记录里没有 model 名，用默认占位。
    try:
        tenant_id = _authenticate_request(request, req_id)
        _check_tenant_allowance(tenant_id)
    except HTTPException as exc:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="messages", status_code=exc.status_code, client_ip=client_ip
        ))
        raise

    try:
        body_bytes = await request.body()
        if len(body_bytes) > settings.MAX_BODY_SIZE:
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
                endpoint="messages", prompt=f"<body size {len(body_bytes)} exceeds limit>",
                status_code=413, client_ip=client_ip
            ))
            logger.warning("[%s] reject | body too large | %s bytes", req_id, len(body_bytes))
            raise HTTPException(status_code=413, detail="Request entity too large")

        prompt_str = _prompt_preview(body_bytes, settings.STATS_PROMPT_MAX_CHARS)
        claude_body_dict = json.loads(body_bytes)
    except json.JSONDecodeError:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="messages", prompt=prompt_str or "<invalid json>",
            status_code=400, client_ip=client_ip
        ))
        logger.warning("[%s] reject | invalid JSON body", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(claude_body_dict, dict):
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="messages", prompt=prompt_str,
            status_code=400, client_ip=client_ip
        ))
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    model_name = claude_body_dict.get("model") or "unknown"

    provider_id, upstream_model = resolve_model_route(claude_body_dict.get("model"), req_id)
    provider = settings.PROVIDERS.get(provider_id)
    if provider is None:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="messages", prompt=prompt_str,
            status_code=500, client_ip=client_ip
        ))
        logger.error("[%s] provider not initialized: %r", req_id, provider_id)
        raise HTTPException(status_code=500, detail=f"Provider {provider_id!r} not configured")

    # 优先用 provider 原生 Anthropic 端点透传；否则走 OpenAI 转换路径
    if provider.anthropic is not None:
        return await _handle_messages_anthropic_passthrough(
            req_id, provider_id, upstream_model, claude_body_dict,
            query_string=request.url.query,
            tenant_id=tenant_id,
            prompt_str=prompt_str,
            client_ip=client_ip,
        )
    # openai-convert 分支不透传 query：原 Anthropic 协议的 query（如 ?beta=true）
    # 对 OpenAI 上游没意义，强行带上反而可能引起 400。
    return await _handle_messages_openai_convert(
        req_id, provider_id, upstream_model, claude_body_dict,
        tenant_id=tenant_id,
        prompt_str=prompt_str,
        client_ip=client_ip,
    )


# ---------------------------------------------------------------------------
#  OpenAI Chat Completions 透传接口
# ---------------------------------------------------------------------------

async def _openai_stream_passthrough(
    response: httpx.Response, req_id: str, t_start: float,
    tenant_id: str = "default", provider_id: str = "", model: str = "",
    prompt_str: Optional[str] = None, client_ip: Optional[str] = None,
):
    """直接透传上游 SSE 数据，并解析其中包含的 usage 记录用量。"""
    chunk_count = 0
    done_sent = False
    stream_error: str | None = None
    last_usage = None
    try:
        try:
            async for raw_line in aiter_raw_lines(response):
                if raw_line.startswith(b"data: "):
                    data_str = raw_line[6:]
                    if data_str.strip() == b"[DONE]":
                        yield b"data: [DONE]\n\n"
                        done_sent = True
                        logger.info("[%s] openai passthrough | [DONE] | chunks=%s", req_id, chunk_count)
                        break
                    chunk_count += 1
                    try:
                        data = json.loads(data_str)
                        if data.get("usage"):
                            last_usage = data["usage"]
                        if chunk_count == 1:
                            logger.info(
                                "[%s] openai passthrough | first_chunk | id=%r model=%r",
                                req_id, data.get("id"), data.get("model"),
                            )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    # 上游那一段字节原样发出去，只有外层报文是本地拼的
                    yield b"data: " + data_str + b"\n\n"
                elif raw_line.strip():
                    yield raw_line + b"\n\n"
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.StreamError) as exc:
            # 上游中途断开连接：响应头已发回客户端，无法再改状态码，
            # 这里记录日志，向客户端补一个 [DONE] 让它干净结束，避免悬挂。
            stream_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[%s] openai passthrough | stream_aborted | chunks=%s | err=%s",
                req_id, chunk_count, stream_error,
            )
            if not done_sent:
                try:
                    err_payload = json.dumps({
                        "error": {
                            "type": "upstream_stream_error",
                            "message": "Upstream connection closed before completion",
                        }
                    })
                    yield b"data: " + err_payload.encode("utf-8") + b"\n\n"
                    yield b"data: [DONE]\n\n"
                    done_sent = True
                except Exception:
                    pass
    finally:
        try:
            await response.aclose()
        except Exception as close_exc:
            logger.debug("[%s] openai passthrough | aclose_error | %s", req_id, close_exc)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "[%s] openai passthrough | stream_closed | chunks=%s | duration_ms=%.0f | stream_error=%s | last_usage=%s",
            req_id, chunk_count, elapsed_ms, stream_error, last_usage,
        )
        cache_read = 0
        input_tokens = 0
        output_tokens = 0
        if last_usage:
            input_tokens = last_usage.get("prompt_tokens", 0)
            output_tokens = last_usage.get("completion_tokens", 0)
            details = last_usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cache_read = details.get("cached_tokens", 0)
            if cache_read == 0:
                cache_read = last_usage.get("prompt_cache_hit_tokens", 0)

        _record_usage(UsageRecord(
            req_id=req_id,
            tenant_id=tenant_id,
            provider=provider_id,
            model=model,
            endpoint="chat/completions",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=0,
            duration_ms=elapsed_ms,
            prompt=prompt_str,
            status_code=200,
            client_ip=client_ip,
        ))


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    if not http_clients:
        logger.error("Server not initialized properly")
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    client_ip = _get_client_ip(request)
    prompt_str = None
    tenant_id = "default"
    provider_id = settings.DEFAULT_PROVIDER_ID
    model_name = "unknown"

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_BODY_SIZE:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="chat/completions", prompt=f"<body size {content_length} exceeds limit>",
            status_code=413, client_ip=client_ip
        ))
        logger.warning("[%s] reject | body too large (header) | %s bytes", req_id, content_length)
        raise HTTPException(status_code=413, detail="Request entity too large")

    # 鉴权放在读 body 之前，理由同 /v1/messages：未通过鉴权就不该为请求体付出
    # 读内存 / 解码 / 解析 / 入库的成本。
    try:
        tenant_id = _authenticate_request(request, req_id)
        _check_tenant_allowance(tenant_id)
    except HTTPException as exc:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="chat/completions", status_code=exc.status_code, client_ip=client_ip
        ))
        raise

    try:
        body_bytes = await request.body()
        if len(body_bytes) > settings.MAX_BODY_SIZE:
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
                endpoint="chat/completions", prompt=f"<body size {len(body_bytes)} exceeds limit>",
                status_code=413, client_ip=client_ip
            ))
            logger.warning("[%s] reject | body too large | %s bytes", req_id, len(body_bytes))
            raise HTTPException(status_code=413, detail="Request entity too large")
        prompt_str = _prompt_preview(body_bytes, settings.STATS_PROMPT_MAX_CHARS)
        openai_req = json.loads(body_bytes)
    except json.JSONDecodeError:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="chat/completions", prompt=prompt_str or "<invalid json>",
            status_code=400, client_ip=client_ip
        ))
        logger.warning("[%s] reject | invalid JSON body", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(openai_req, dict):
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=model_name,
            endpoint="chat/completions", prompt=prompt_str,
            status_code=400, client_ip=client_ip
        ))
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    model_name = openai_req.get("model") or "unknown"

    provider_id, upstream_model = resolve_model_route(model_name, req_id)
    if upstream_model is not None:
        openai_req["model"] = upstream_model

    client = _get_client(provider_id, PROTOCOL_OPENAI)
    if client is None:
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id, model=openai_req.get("model", ""),
            endpoint="chat/completions", prompt=prompt_str,
            status_code=502, client_ip=client_ip
        ))
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_OPENAI)

    upstream_path = _append_query("chat/completions", request.url.query)

    logger.info(
        "[%s] proxy -> upstream | provider=%r | upstream_path=%r | %s",
        req_id, provider_id, upstream_path, summarize_openai_request(openai_req),
    )
    log_json_preview(logger, "proxy -> upstream | openai_request_json", req_id, openai_req)

    is_stream = openai_req.get("stream", False)

    t0 = time.perf_counter()
    if is_stream:
        if "stream_options" not in openai_req:
            openai_req["stream_options"] = {"include_usage": True}
        elif isinstance(openai_req["stream_options"], dict):
            openai_req["stream_options"]["include_usage"] = True

        built = client.build_request("POST", upstream_path, json=openai_req)
        logger.info(
            "[%s] proxy -> upstream | POST %s%s | provider=%r | model=%r | stream=true (passthrough)",
            req_id, client.base_url, upstream_path, provider_id, openai_req.get("model"),
        )
        try:
            r = await client.send(built, stream=True)
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "[%s] upstream | connection_failed | after_ms=%.0f | %s",
                req_id, elapsed_ms, exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=openai_req.get("model", ""), endpoint="chat/completions",
                prompt=prompt_str, status_code=502, client_ip=client_ip,
                duration_ms=elapsed_ms
            ))
            raise HTTPException(status_code=502, detail="Upstream connection failed")

        if r.status_code != 200:
            body = await r.aread()
            text = body.decode("utf-8", errors="replace")
            logger.error("[%s] upstream | error | status=%s | preview=%r", req_id, r.status_code, text[:2000])
            try:
                err_content = json.loads(text)
            except json.JSONDecodeError:
                err_content = {"error": {"message": text[:500], "type": "upstream_error"}}
            _record_usage(UsageRecord(
                req_id=req_id, tenant_id=tenant_id, provider=provider_id,
                model=openai_req.get("model", ""), endpoint="chat/completions",
                prompt=prompt_str, status_code=r.status_code, client_ip=client_ip,
                duration_ms=(time.perf_counter() - t0) * 1000
            ))
            return JSONResponse(status_code=r.status_code, content=err_content)

        return StreamingResponse(
            _openai_stream_passthrough(
                r, req_id, t0,
                tenant_id=tenant_id, provider_id=provider_id,
                model=openai_req.get("model", ""),
                prompt_str=prompt_str,
                client_ip=client_ip,
            ),
            media_type="text/event-stream",
        )

    # 非流式
    logger.info(
        "[%s] proxy -> upstream | POST %s%s | provider=%r | model=%r | stream=false (passthrough)",
        req_id, client.base_url, upstream_path, provider_id, openai_req.get("model"),
    )
    try:
        resp = await client.post(upstream_path, json=openai_req)
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "[%s] upstream | connection_failed | after_ms=%.0f | %s",
            req_id, elapsed_ms, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="chat/completions",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream connection failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] upstream | response | status=%s | duration_ms=%.0f",
        req_id, resp.status_code, elapsed_ms,
    )

    if resp.status_code != 200:
        try:
            err_content = resp.json()
        except Exception:
            err_content = {"error": {"message": resp.text[:500], "type": "upstream_error"}}
        logger.error("[%s] upstream | error | status=%s | %s", req_id, resp.status_code, json_preview(err_content))
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="chat/completions",
            prompt=prompt_str, status_code=resp.status_code, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error("[%s] upstream | invalid_json_body | %s | preview=%r", req_id, e, resp.text[:500])
        _record_usage(UsageRecord(
            req_id=req_id, tenant_id=tenant_id, provider=provider_id,
            model=openai_req.get("model", ""), endpoint="chat/completions",
            prompt=prompt_str, status_code=502, client_ip=client_ip,
            duration_ms=elapsed_ms
        ))
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    logger.info("[%s] upstream | ok | %s", req_id, summarize_openai_response(data))
    _record_openai_usage(
        req_id=req_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        model=openai_req.get("model", ""),
        endpoint="chat/completions",
        duration_ms=elapsed_ms,
        data=data,
        prompt=prompt_str,
        status_code=200,
        client_ip=client_ip,
    )
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
#  Models 列表透传接口
# ---------------------------------------------------------------------------

def _route_model_entries(
    model_routes: Dict[str, List[ModelRoute]],
    default_provider_id: str,
) -> Dict[str, Dict[str, Any]]:
    """把 model_routes 转成 /v1/models 条目（客户端模型名、带标记）。"""
    entries: Dict[str, Dict[str, Any]] = {}
    for name, routes in model_routes.items():
        route = routes[0] if routes else None
        provider_id = route.provider_id if route else default_provider_id
        upstream_model = (route.upstream_model or name) if route else name
        entries[name] = {
            "id": name,
            "object": "model",
            "source": "model_routes",
            "provider": provider_id,
            "upstream_model": upstream_model,
        }
    return entries


def _normalize_upstream_models(
    provider_id: str,
    upstream_items: Iterable[Any],
    models_source: str = "api",
) -> Dict[str, Dict[str, Any]]:
    """把一个 provider 的模型列表标准化为命名空间条目。

    - id 统一为 "provider_id/model_name"（全部加前缀，保证 id→provider 无歧义）。
    - 打 source="upstream"、provider、upstream_model、models_source 标记。
    - models_source="api"（来自上游 /models）或 "manual"（来自 provider.models 配置）。
    """
    entries: Dict[str, Dict[str, Any]] = {}
    for item in upstream_items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id is None:
            continue
        model_id = str(model_id).strip()
        if not model_id:
            continue
        namespaced_id = f"{provider_id}/{model_id}"
        entry = dict(item)
        entry.update({
            "id": namespaced_id,
            "object": "model",
            "source": "upstream",
            "provider": provider_id,
            "upstream_model": model_id,
            "models_source": models_source,
        })
        entries[namespaced_id] = entry
    return entries


async def discover_upstream_models(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """汇总所有 provider 的可用模型，返回命名空间化后的模型表。

    - provider 配置了 models 手动列表时优先采用，不再请求上游 GET /models
      （适用于不提供该接口的上游，如火山方舟部分端点）。
    - 未配置手动列表且带 OpenAI 端点的 provider：请求 GET /models 发现。
    - 结果按 MODELS_DISCOVERY_TTL 缓存；force=True 强制刷新。
    - best-effort：单个 provider 失败只记日志，不影响其它 provider。
    """
    global _upstream_models_cache, _upstream_models_cache_ts

    now = time.monotonic()
    if not force and _upstream_models_cache and (now - _upstream_models_cache_ts) < MODELS_DISCOVERY_TTL:
        return _upstream_models_cache

    async with _upstream_models_lock:
        now = time.monotonic()
        if not force and _upstream_models_cache and (now - _upstream_models_cache_ts) < MODELS_DISCOVERY_TTL:
            return _upstream_models_cache

        results: Dict[str, Dict[str, Any]] = {}
        targets: list = []
        manual_provider_ids: list = []
        for pid, p in settings.PROVIDERS.items():
            # 手动模型列表优先：直接采用，跳过上游 /models 请求
            if getattr(p, "models", None):
                manual_provider_ids.append(pid)
                results.update(_normalize_upstream_models(
                    pid, [{"id": m} for m in p.models], models_source="manual",
                ))
                continue
            client = _get_client(pid, PROTOCOL_OPENAI)
            if client is not None:
                targets.append((pid, client))

        async def _fetch(pid: str, client: httpx.AsyncClient) -> None:
            try:
                resp = await client.get("models")
                if resp.status_code != 200:
                    logger.warning(
                        "[models-discovery] provider=%r status=%s", pid, resp.status_code,
                    )
                    return
                data = resp.json()
                items = data.get("data", []) if isinstance(data, dict) else []
                results.update(_normalize_upstream_models(pid, items or []))
            except httpx.RequestError as exc:
                logger.warning("[models-discovery] provider=%r connection_failed | %s", pid, exc)
            except Exception as exc:
                logger.warning("[models-discovery] provider=%r error | %s", pid, exc)

        await asyncio.gather(*(_fetch(pid, client) for pid, client in targets))

        _upstream_models_cache = results
        _upstream_models_cache_ts = time.monotonic()
        logger.info(
            "[models-discovery] providers=%s discovered=%s (manual_providers=%s)",
            len(targets), len(results), manual_provider_ids,
        )
        return results


def _models_payload(
    models_by_id: Dict[str, Dict[str, Any]],
    route_names: Iterable[str],
) -> Dict[str, Any]:
    """组装 OpenAI /v1/models 响应体。

    - model_routes 优先：按配置顺序排在最前。
    - 其余（上游模型）按名称排序。
    """
    route_names = list(route_names)
    route_ids = [name for name in route_names if name in models_by_id]
    route_id_set = set(route_ids)

    route_entries = [models_by_id[name] for name in route_ids]
    other_entries = sorted(
        (entry for model_id, entry in models_by_id.items() if model_id not in route_id_set),
        key=lambda m: str(m.get("id", "")).lower(),
    )
    return {"object": "list", "data": route_entries + other_entries}


@app.get("/v1/models")
async def handle_list_models(request: Request):
    if not http_clients:
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]

    # 强制校验 API key 并检测租户状态
    tenant_id = _authenticate_request(request, req_id)
    _check_tenant_allowance(tenant_id)

    # 1) 本地 model_routes 条目（优先，带标记）
    route_entries = _route_model_entries(settings.MODEL_ROUTES, settings.DEFAULT_PROVIDER_ID)

    # 2) 所有 provider 的上游模型（命名空间化，带标记）
    upstream_entries = await discover_upstream_models()

    # 合并：model_routes 优先（若 key 与命名空间名撞车，以 model_routes 为准）
    models_by_id = dict(upstream_entries)
    models_by_id.update(route_entries)

    if not models_by_id:
        logger.warning(
            "[%s] /v1/models | no models available (no model_routes, no openai upstream)",
            req_id,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "no_models_available",
                    "message": "No models available: no model_routes configured and no OpenAI upstream discovered.",
                }
            },
        )

    logger.info(
        "[%s] /v1/models | routes=%s | upstream=%s | total=%s | tenant=%r",
        req_id, len(route_entries), len(upstream_entries), len(models_by_id), tenant_id,
    )
    return JSONResponse(content=_models_payload(models_by_id, settings.MODEL_ROUTES))


# ---------------------------------------------------------------------------
#  统计查询接口
# ---------------------------------------------------------------------------

@app.get("/stats")
async def get_stats(
    group_by: str = "model",
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    tenant: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    用量统计查询接口。
    """
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: usage_stats.query_stats(
                group_by=group_by,
                since=since,
                until=until,
                model=model,
                tenant=tenant,
                provider=provider,
            )
        )
        return JSONResponse(content=res)
    except Exception as e:
        logger.error("Failed to query stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to query stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5432)
