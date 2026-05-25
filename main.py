import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config import (
    AUTH_STYLE_ANTHROPIC,
    AUTH_STYLE_BEARER,
    AUTH_STYLE_NONE,
    PROTOCOL_ANTHROPIC,
    PROTOCOL_OPENAI,
    ProviderEndpoint,
    settings,
)
from schemas import ClaudeChatRequest
from converters import claude_to_openai_request, openai_to_claude_response, openai_to_claude_stream
from request_log import (
    json_preview,
    log_stream_chunk_debug,
    summarize_openai_request,
    summarize_openai_response,
)
from usage_stats import UsageStats, UsageRecord, _mask_key

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
usage_stats = UsageStats(settings.STATS_DB)


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
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    providers_summary: Dict[str, Dict[str, Dict[str, str]]] = {}
    for pid, p in settings.PROVIDERS.items():
        http_clients[pid] = {}
        summary: Dict[str, Dict[str, str]] = {}
        if p.openai:
            http_clients[pid][PROTOCOL_OPENAI] = _build_client(PROTOCOL_OPENAI, p.openai)
            summary[PROTOCOL_OPENAI] = {
                "base_url": p.openai.base_url,
                "auth_style": p.openai.auth_style,
            }
        if p.anthropic:
            # Anthropic 端点路径里 v1 由代码自动拼，base_url 不应再带 /v1，
            # 否则最终会变成 .../v1/v1/messages。这里给一条显式提醒。
            if p.anthropic.base_url.rstrip("/").endswith("/v1"):
                logger.warning(
                    "Provider %r anthropic.base_url=%r 末尾带了 '/v1'；"
                    "代码会再拼一次 v1/messages，最终路径会出现两次 v1。"
                    "请把 base_url 末尾的 /v1 去掉。",
                    pid, p.anthropic.base_url,
                )
            http_clients[pid][PROTOCOL_ANTHROPIC] = _build_client(PROTOCOL_ANTHROPIC, p.anthropic)
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

    usage_stats.init_db()
    if settings.TENANTS:
        usage_stats.upsert_tenants([
            {"id": t.id, "name": t.name, "api_key": t.api_key, "status": t.status}
            for t in settings.TENANTS
        ])
        logger.info(
            "Startup | tenants synced: %s",
            [t.id for t in settings.TENANTS],
        )

    yield
    for pid, by_proto in http_clients.items():
        for protocol, client in by_proto.items():
            try:
                await client.aclose()
            except Exception as e:
                logger.debug("Shutdown | aclose %s/%s failed: %s", pid, protocol, e)
    http_clients.clear()
    usage_stats.close()
    logger.info("Shutdown | all upstream HTTP clients closed")


app = FastAPI(lifespan=lifespan)


def _extract_tenant(request: Request, req_id: str) -> str:
    """从请求头提取 Bearer key 或 x-api-key，匹配租户。"""
    raw_key: Optional[str] = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw_key = auth[7:].strip()
    if not raw_key:
        raw_key = request.headers.get("x-api-key", "").strip() or None

    tenant_id = usage_stats.resolve_tenant(raw_key)
    if tenant_id != "default" and raw_key:
        logger.debug("[%s] tenant=%r (key=%s)", req_id, tenant_id, _mask_key(raw_key))
    return tenant_id


def resolve_model_route(model: str | None, req_id: str) -> Tuple[str, str | None]:
    """
    把客户端传入的 model 解析为 (provider_id, upstream_model)。

    - 命中 MODEL_ROUTES：取第一条路由（未来 failover 在此扩展为遍历）。
      若 upstream_model 为空则透传客户端模型名。
    - 未命中：使用默认 provider，且模型名原样透传。
    - 命中时打 INFO 日志，未命中保持安静。
    """
    if not model:
        return settings.DEFAULT_PROVIDER_ID, model

    routes = settings.MODEL_ROUTES.get(model)
    if not routes:
        return settings.DEFAULT_PROVIDER_ID, model

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

def record_usage_async(record: UsageRecord) -> None:
    """异步记录用量，避免 SQLite 同步 IO 阻塞事件循环。"""
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, usage_stats.record_usage, record)
    except RuntimeError:
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
) -> None:
    """从 Anthropic 非流式响应中提取 usage 并记录。"""
    u = data.get("usage") or {}
    record_usage_async(UsageRecord(
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
    ))


def _record_openai_usage(
    req_id: str, tenant_id: str, provider_id: str,
    model: str, endpoint: str, duration_ms: float, data: dict,
) -> None:
    """从 OpenAI 非流式响应中提取 usage 并记录。"""
    u = data.get("usage") or {}
    cache_read = 0
    details = u.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cache_read = details.get("cached_tokens", 0)
    if cache_read == 0:
        cache_read = u.get("prompt_cache_hit_tokens", 0)
    record_usage_async(UsageRecord(
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
            async for line in response.aiter_lines():
                raw_line = line + "\n"
                bytes_count += len(raw_line.encode("utf-8"))
                yield raw_line

                # SSE 解析：提取 event type 和 data
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: ") and current_event:
                    try:
                        data = json.loads(line[6:])
                        if current_event == "message_start":
                            msg = data.get("message", {})
                            u = msg.get("usage", {})
                            input_tokens += u.get("input_tokens", 0)
                            cache_read_tokens += u.get("cache_read_input_tokens", 0)
                            cache_creation_tokens += u.get("cache_creation_input_tokens", 0)
                        elif current_event == "message_delta":
                            u = data.get("usage", {})
                            output_tokens += u.get("output_tokens", 0)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif line == "":
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
        record_usage_async(UsageRecord(
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
        ))


async def _handle_messages_anthropic_passthrough(
    req_id: str,
    provider_id: str,
    upstream_model: Optional[str],
    claude_body_dict: dict,
    query_string: str = "",
    tenant_id: str = "default",
):
    """provider 原生支持 Anthropic 协议 → 仅替换 model 字段，直接透传 body 与 query。"""
    client = _get_client(provider_id, PROTOCOL_ANTHROPIC)
    if client is None:
        # 启动期已保证至少一个端点；理论上不会到这
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
    logger.debug("[%s] proxy -> upstream | anthropic_request_json=%s", req_id, json_preview(claude_body_dict))

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
            return JSONResponse(status_code=r.status_code, content=err_content)

        return StreamingResponse(
            _anthropic_stream_passthrough(
                r, req_id, t0,
                tenant_id=tenant_id, provider_id=provider_id,
                model=claude_body_dict.get("model", ""),
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
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error(
            "[%s] upstream | invalid_json_body | %s | preview=%r",
            req_id, e, resp.text[:500],
        )
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    _record_anthropic_usage(req_id, tenant_id, provider_id,
                            claude_body_dict.get("model", ""), elapsed_ms, data)
    return JSONResponse(content=data)


async def _handle_messages_openai_convert(
    req_id: str,
    provider_id: str,
    upstream_model: Optional[str],
    claude_body_dict: dict,
    tenant_id: str = "default",
):
    """provider 仅支持 OpenAI 协议 → 走 claude→openai 转换 + openai→claude 反转。"""
    client = _get_client(provider_id, PROTOCOL_OPENAI)
    if client is None:
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_OPENAI)

    # 这条分支需要严格 Claude schema 才能转换
    try:
        validated_req = ClaudeChatRequest(**claude_body_dict)
        claude_body = validated_req.model_dump(exclude_none=True)
    except Exception as e:
        logger.warning("[%s] reject | validation | %s", req_id, e)
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")

    openai_req = claude_to_openai_request(claude_body)
    if upstream_model is not None:
        openai_req["model"] = upstream_model

    logger.info(
        "[%s] /v1/messages | provider=%r | mode=openai-convert | %s",
        req_id, provider_id, summarize_openai_request(openai_req),
    )
    logger.debug("[%s] proxy -> upstream | openai_request_json=%s", req_id, json_preview(openai_req))

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
            return JSONResponse(status_code=r.status_code, content=err_content)

        def on_usage(u: dict):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            record_usage_async(UsageRecord(
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
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error("[%s] upstream | invalid_json_body | %s | preview=%r", req_id, e, resp.text[:500])
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    logger.info("[%s] upstream | ok | %s", req_id, summarize_openai_response(data))
    logger.debug("[%s] upstream | response_body=%s", req_id, json_preview(data))

    _record_openai_usage(
        req_id=req_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        model=openai_req.get("model", ""),
        endpoint="messages",
        duration_ms=elapsed_ms,
        data=data,
    )
    return JSONResponse(content=openai_to_claude_response(data))


@app.post("/v1/messages")
async def handle_messages(request: Request):
    if not http_clients:
        logger.error("Server not initialized properly")
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    client_host = request.client.host if request.client else "?"
    logger.info(
        "[%s] client -> proxy | POST /v1/messages | remote=%s | ua=%r",
        req_id,
        client_host,
        request.headers.get("user-agent", "")[:200],
    )

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_BODY_SIZE:
        logger.warning("[%s] reject | body too large (header) | %s bytes", req_id, content_length)
        raise HTTPException(status_code=413, detail="Request entity too large")

    try:
        body_bytes = await request.body()
        if len(body_bytes) > settings.MAX_BODY_SIZE:
            logger.warning("[%s] reject | body too large | %s bytes", req_id, len(body_bytes))
            raise HTTPException(status_code=413, detail="Request entity too large")

        claude_body_dict = json.loads(body_bytes)
    except json.JSONDecodeError:
        logger.warning("[%s] reject | invalid JSON body", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(claude_body_dict, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    tenant_id = _extract_tenant(request, req_id)
    _check_tenant_allowance(tenant_id)
    provider_id, upstream_model = resolve_model_route(claude_body_dict.get("model"), req_id)
    provider = settings.PROVIDERS.get(provider_id)
    if provider is None:
        logger.error("[%s] provider not initialized: %r", req_id, provider_id)
        raise HTTPException(status_code=500, detail=f"Provider {provider_id!r} not configured")

    # 优先用 provider 原生 Anthropic 端点透传；否则走 OpenAI 转换路径
    if provider.anthropic is not None:
        return await _handle_messages_anthropic_passthrough(
            req_id, provider_id, upstream_model, claude_body_dict,
            query_string=request.url.query,
            tenant_id=tenant_id,
        )
    # openai-convert 分支不透传 query：原 Anthropic 协议的 query（如 ?beta=true）
    # 对 OpenAI 上游没意义，强行带上反而可能引起 400。
    return await _handle_messages_openai_convert(
        req_id, provider_id, upstream_model, claude_body_dict,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
#  OpenAI Chat Completions 透传接口
# ---------------------------------------------------------------------------

async def _openai_stream_passthrough(
    response: httpx.Response, req_id: str, t_start: float,
    tenant_id: str = "default", provider_id: str = "", model: str = "",
):
    """直接透传上游 SSE 数据，并解析其中包含的 usage 记录用量。"""
    chunk_count = 0
    done_sent = False
    stream_error: str | None = None
    last_usage = None
    try:
        try:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
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
                    except json.JSONDecodeError:
                        pass
                    yield f"data: {data_str}\n\n"
                elif line.strip():
                    yield f"{line}\n\n"
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
                    yield f"data: {err_payload}\n\n"
                    yield "data: [DONE]\n\n"
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
        if last_usage:
            cache_read = 0
            details = last_usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cache_read = details.get("cached_tokens", 0)
            if cache_read == 0:
                cache_read = last_usage.get("prompt_cache_hit_tokens", 0)
            record_usage_async(UsageRecord(
                req_id=req_id,
                tenant_id=tenant_id,
                provider=provider_id,
                model=model,
                endpoint="chat/completions",
                input_tokens=last_usage.get("prompt_tokens", 0),
                output_tokens=last_usage.get("completion_tokens", 0),
                cache_read_tokens=cache_read,
                cache_creation_tokens=0,
                duration_ms=elapsed_ms,
            ))


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    if not http_clients:
        logger.error("Server not initialized properly")
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    client_host = request.client.host if request.client else "?"
    logger.info(
        "[%s] client -> proxy | POST /v1/chat/completions | remote=%s | ua=%r",
        req_id, client_host, request.headers.get("user-agent", "")[:200],
    )

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_BODY_SIZE:
        logger.warning("[%s] reject | body too large (header) | %s bytes", req_id, content_length)
        raise HTTPException(status_code=413, detail="Request entity too large")

    try:
        body_bytes = await request.body()
        if len(body_bytes) > settings.MAX_BODY_SIZE:
            logger.warning("[%s] reject | body too large | %s bytes", req_id, len(body_bytes))
            raise HTTPException(status_code=413, detail="Request entity too large")
        openai_req = json.loads(body_bytes)
    except json.JSONDecodeError:
        logger.warning("[%s] reject | invalid JSON body", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    client_model = openai_req.get("model") if isinstance(openai_req, dict) else None
    tenant_id = _extract_tenant(request, req_id)
    _check_tenant_allowance(tenant_id)
    provider_id, upstream_model = resolve_model_route(client_model, req_id)
    if isinstance(openai_req, dict) and upstream_model is not None:
        openai_req["model"] = upstream_model

    client = _get_client(provider_id, PROTOCOL_OPENAI)
    if client is None:
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_OPENAI)

    upstream_path = _append_query("chat/completions", request.url.query)

    logger.info(
        "[%s] proxy -> upstream | provider=%r | upstream_path=%r | %s",
        req_id, provider_id, upstream_path, summarize_openai_request(openai_req),
    )
    logger.debug("[%s] proxy -> upstream | openai_request_json=%s", req_id, json_preview(openai_req))

    is_stream = openai_req.get("stream", False)

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
        t0 = time.perf_counter()
        try:
            r = await client.send(built, stream=True)
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "[%s] upstream | connection_failed | after_ms=%.0f | %s",
                req_id, elapsed_ms, exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise HTTPException(status_code=502, detail="Upstream connection failed")

        if r.status_code != 200:
            body = await r.aread()
            text = body.decode("utf-8", errors="replace")
            logger.error("[%s] upstream | error | status=%s | preview=%r", req_id, r.status_code, text[:2000])
            try:
                err_content = json.loads(text)
            except json.JSONDecodeError:
                err_content = {"error": {"message": text[:500], "type": "upstream_error"}}
            return JSONResponse(status_code=r.status_code, content=err_content)

        return StreamingResponse(
            _openai_stream_passthrough(
                r, req_id, t0,
                tenant_id=tenant_id, provider_id=provider_id,
                model=openai_req.get("model", ""),
            ),
            media_type="text/event-stream",
        )

    # 非流式
    logger.info(
        "[%s] proxy -> upstream | POST %s%s | provider=%r | model=%r | stream=false (passthrough)",
        req_id, client.base_url, upstream_path, provider_id, openai_req.get("model"),
    )
    t0 = time.perf_counter()
    try:
        resp = await client.post(upstream_path, json=openai_req)
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "[%s] upstream | connection_failed | after_ms=%.0f | %s",
            req_id, elapsed_ms, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
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
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error("[%s] upstream | invalid_json_body | %s | preview=%r", req_id, e, resp.text[:500])
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
    )
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
#  Models 列表透传接口
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def handle_list_models(request: Request):
    if not http_clients:
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    provider_id = settings.DEFAULT_PROVIDER_ID
    client = _get_client(provider_id, PROTOCOL_OPENAI)
    if client is None:
        return _protocol_unsupported_response(req_id, provider_id, PROTOCOL_OPENAI)
    upstream_path = _append_query("models", request.url.query)
    logger.info(
        "[%s] client -> proxy | GET /v1/models | provider=%r | upstream_path=%r",
        req_id, provider_id, upstream_path,
    )

    t0 = time.perf_counter()
    try:
        resp = await client.get(upstream_path)
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error("[%s] upstream | connection_failed | after_ms=%.0f | %s", req_id, elapsed_ms, exc)
        raise HTTPException(status_code=502, detail="Upstream connection failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("[%s] upstream | response | status=%s | duration_ms=%.0f", req_id, resp.status_code, elapsed_ms)

    if resp.status_code != 200:
        try:
            err_content = resp.json()
        except Exception:
            err_content = {"error": {"message": resp.text[:500], "type": "upstream_error"}}
        return JSONResponse(status_code=resp.status_code, content=err_content)

    try:
        data = resp.json()
    except Exception as e:
        logger.error("[%s] upstream | invalid_json_body | %s", req_id, e)
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")

    model_count = len(data.get("data", []))
    logger.info("[%s] upstream | ok | models_count=%s", req_id, model_count)
    return JSONResponse(content=data)


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

    uvicorn.run(app, host="0.0.0.0", port=8000)
