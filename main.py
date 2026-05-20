import json
import logging
import time
import uuid
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config import settings
from schemas import ClaudeChatRequest
from converters import claude_to_openai_request, openai_to_claude_response, openai_to_claude_stream
from request_log import (
    json_preview,
    log_stream_chunk_debug,
    summarize_openai_request,
    summarize_openai_response,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ai-proxy")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=settings.OPENAI_API_BASE,
        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        timeout=httpx.Timeout(
            connect=settings.TIMEOUT_CONNECT,
            read=settings.TIMEOUT_READ,
            write=settings.TIMEOUT_WRITE,
            pool=settings.TIMEOUT_POOL,
        ),
    )
    logger.info(
        "Startup | upstream_base_url=%s | log_level=%s | model_aliases=%s",
        settings.OPENAI_API_BASE,
        settings.LOG_LEVEL,
        settings.MODEL_ALIASES or "<none>",
    )
    yield
    if http_client:
        await http_client.aclose()
        logger.info("Shutdown | upstream HTTP client closed")


app = FastAPI(lifespan=lifespan)


def resolve_model_alias(model: str | None, req_id: str) -> str | None:
    """
    根据 settings.MODEL_ALIASES 把客户端模型名映射到上游模型名。

    - 命中映射：返回映射后的名字，并打 INFO 日志记录原始名与映射后的名。
    - 未命中或入参为空：原样返回（透传）。
    """
    if not model:
        return model
    mapped = settings.MODEL_ALIASES.get(model)
    if mapped and mapped != model:
        logger.info(
            "[%s] model alias | %r -> %r",
            req_id,
            model,
            mapped,
        )
        return mapped
    return model


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


@app.post("/v1/messages")
async def handle_messages(request: Request):
    if http_client is None:
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

    try:
        validated_req = ClaudeChatRequest(**claude_body_dict)
        claude_body = validated_req.model_dump(exclude_none=True)
    except Exception as e:
        logger.warning("[%s] reject | validation | %s", req_id, e)
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")

    openai_req = claude_to_openai_request(claude_body)
    openai_req["model"] = resolve_model_alias(openai_req.get("model"), req_id)
    logger.info("[%s] proxy -> upstream | prepared | %s", req_id, summarize_openai_request(openai_req))
    logger.debug("[%s] proxy -> upstream | openai_request_json=%s", req_id, json_preview(openai_req))

    if openai_req.get("stream", False):
        built = http_client.build_request("POST", "chat/completions", json=openai_req)
        logger.info(
            "[%s] proxy -> upstream | POST %schat/completions | stream=true",
            req_id,
            settings.OPENAI_API_BASE,
        )
        t0 = time.perf_counter()
        try:
            r = await http_client.send(built, stream=True)
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

        return StreamingResponse(
            openai_to_claude_stream(stream_generator(r, req_id, t0)),
            media_type="text/event-stream",
        )

    logger.info(
        "[%s] proxy -> upstream | POST %schat/completions | stream=false",
        req_id,
        settings.OPENAI_API_BASE,
    )
    t0 = time.perf_counter()
    try:
        resp = await http_client.post("chat/completions", json=openai_req)
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

    return JSONResponse(content=openai_to_claude_response(data))


# ---------------------------------------------------------------------------
#  OpenAI Chat Completions 透传接口
# ---------------------------------------------------------------------------

async def _openai_stream_passthrough(response: httpx.Response, req_id: str, t_start: float):
    """直接透传上游 SSE 数据，仅做日志记录。"""
    chunk_count = 0
    done_sent = False
    stream_error: str | None = None
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
                    if chunk_count == 1:
                        try:
                            first = json.loads(data_str)
                            logger.info(
                                "[%s] openai passthrough | first_chunk | id=%r model=%r",
                                req_id, first.get("id"), first.get("model"),
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
            "[%s] openai passthrough | stream_closed | chunks=%s | duration_ms=%.0f | stream_error=%s",
            req_id, chunk_count, elapsed_ms, stream_error,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    if http_client is None:
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

    if isinstance(openai_req, dict) and openai_req.get("model"):
        openai_req["model"] = resolve_model_alias(openai_req["model"], req_id)

    logger.info("[%s] proxy -> upstream | prepared | %s", req_id, summarize_openai_request(openai_req))
    logger.debug("[%s] proxy -> upstream | openai_request_json=%s", req_id, json_preview(openai_req))

    is_stream = openai_req.get("stream", False)

    if is_stream:
        built = http_client.build_request("POST", "chat/completions", json=openai_req)
        logger.info(
            "[%s] proxy -> upstream | POST %schat/completions | stream=true (passthrough)",
            req_id, settings.OPENAI_API_BASE,
        )
        t0 = time.perf_counter()
        try:
            r = await http_client.send(built, stream=True)
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
            _openai_stream_passthrough(r, req_id, t0),
            media_type="text/event-stream",
        )

    # 非流式
    logger.info(
        "[%s] proxy -> upstream | POST %schat/completions | stream=false (passthrough)",
        req_id, settings.OPENAI_API_BASE,
    )
    t0 = time.perf_counter()
    try:
        resp = await http_client.post("chat/completions", json=openai_req)
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
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
#  Models 列表透传接口
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def handle_list_models(request: Request):
    if http_client is None:
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    req_id = uuid.uuid4().hex[:12]
    logger.info("[%s] client -> proxy | GET /v1/models", req_id)

    t0 = time.perf_counter()
    try:
        resp = await http_client.get("models")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
