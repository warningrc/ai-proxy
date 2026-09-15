"""
热重载时旧上游 client 的回收判据。

热重载会换掉整批 httpx client。旧的那批不能立刻 aclose —— 此刻可能还有请求正拿
它读上游 SSE，而响应头早就发给客户端了，掐断只能表现为「流中途断掉」，状态码已经
补救不了。

以前的判据是「退役满 max(read_timeout, 30) 秒就关」，但 httpx 的 read timeout 是
**每次读取**的上限，不是整个响应的预算：一个持续吐 token 的流，只要每次间隔小于
read timeout 就能活过任意长的宽限期。所以判据改成「按在途请求」——某个世代的
client 集合只可能被「世代号不大于它」的请求用到（请求可能在重载之后才真正去取
client，所以是「不大于」而不是「等于」），等这些请求都结束了才关。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

import main


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _client_set() -> Dict[str, Dict[str, _FakeClient]]:
    return {"p1": {"openai": _FakeClient()}}


def _closed(clients) -> bool:
    return clients["p1"]["openai"].closed


@pytest.fixture(autouse=True)
def _clean_state():
    """世代号和退役列表都是模块级全局，测试之间必须清干净。"""
    main._retired_clients.clear()
    main._inflight_by_generation.clear()
    yield
    main._retired_clients.clear()
    main._inflight_by_generation.clear()


# ---------------------------------------------------------------------------
#  回收判据
# ---------------------------------------------------------------------------

def test_retired_set_closes_once_nobody_uses_it(monkeypatch):
    monkeypatch.setattr(main, "_client_generation", 1)
    old = _client_set()
    main._retire_http_clients(old, 1)

    asyncio.run(main._sweep_retired_clients())
    assert _closed(old)
    assert main._retired_clients == []


def test_long_stream_keeps_its_client_set_alive(monkeypatch):
    """核心回归：一个还在读 SSE 的请求，必须挡住它那个世代的 client 被关闭。

    这条流比任何固定的宽限期都长（每片都在 read timeout 之内，整体没有上限），
    按时间猜的宽限期必然要么掐断它、要么无限期不回收。
    """
    monkeypatch.setattr(main, "_client_generation", 1)
    old = _client_set()
    main._retire_http_clients(old, 1)

    inflight = main._mark_client_use()          # 请求进来，开始读流
    assert inflight == 1
    monkeypatch.setattr(main, "_client_generation", 2)   # 热重载：新建了一批

    asyncio.run(main._sweep_retired_clients())
    assert not _closed(old), "流还没结束就把 client 关了，客户端会看到流中断"
    assert len(main._retired_clients) == 1

    main._unmark_client_use(inflight)           # 流正常结束
    asyncio.run(main._sweep_retired_clients())
    assert _closed(old), "没人用了还不回收，连接池会一直攒着"


def test_request_that_started_after_reload_does_not_protect_old_sets(monkeypatch):
    """重载之后才进来的请求不可能用到上一代 client（它取到的是新 dict），
    所以它不该挡住旧集合的回收。"""
    monkeypatch.setattr(main, "_client_generation", 1)
    old = _client_set()
    main._retire_http_clients(old, 1)

    monkeypatch.setattr(main, "_client_generation", 2)
    inflight = main._mark_client_use()
    assert inflight == 2

    asyncio.run(main._sweep_retired_clients())
    assert _closed(old)
    main._unmark_client_use(inflight)


def test_request_may_still_pick_up_a_newer_set(monkeypatch):
    """请求可能在重载之后才真正去取 client（取的是当时现役的那批）——所以判据是
    「在途请求的世代号 <= 集合的世代号」，老的世代号同样挡住新一代集合的回收。"""
    monkeypatch.setattr(main, "_client_generation", 1)
    inflight = main._mark_client_use()          # 请求进来（此刻现役是第 1 代）

    monkeypatch.setattr(main, "_client_generation", 2)
    old = _client_set()
    main._retire_http_clients(old, 1)
    newer = _client_set()
    main._retire_http_clients(newer, 2)

    asyncio.run(main._sweep_retired_clients())
    assert not _closed(old) and not _closed(newer)

    main._unmark_client_use(inflight)
    asyncio.run(main._sweep_retired_clients())
    assert _closed(old) and _closed(newer)


def test_independent_generations_are_reclaimed_independently(monkeypatch):
    """互不相关的两代互不牵连：只有还被用着的那代留着。"""
    monkeypatch.setattr(main, "_client_generation", 1)
    gen1 = _client_set()
    main._retire_http_clients(gen1, 1)

    monkeypatch.setattr(main, "_client_generation", 2)
    inflight = main._mark_client_use()          # 只有第 2 代的请求在途

    monkeypatch.setattr(main, "_client_generation", 3)
    gen2 = _client_set()
    main._retire_http_clients(gen2, 2)

    asyncio.run(main._sweep_retired_clients())
    assert _closed(gen1), "没人可能用到第 1 代了"
    assert not _closed(gen2), "第 2 代还有在途请求"

    main._unmark_client_use(inflight)
    asyncio.run(main._sweep_retired_clients())
    assert _closed(gen2)


def test_shutdown_forces_close(monkeypatch):
    """进程退出时不再等流跑完，全部关干净。"""
    monkeypatch.setattr(main, "_client_generation", 1)
    old = _client_set()
    main._retire_http_clients(old, 1)
    inflight = main._mark_client_use()

    asyncio.run(main._sweep_retired_clients(force=True))
    assert _closed(old)
    main._unmark_client_use(inflight)


def test_empty_client_set_is_not_retired():
    main._retire_http_clients({}, 1)
    assert main._retired_clients == []


# ---------------------------------------------------------------------------
#  在途标记的覆盖范围
# ---------------------------------------------------------------------------

def test_middleware_holds_the_mark_until_the_body_is_sent():
    """标记必须覆盖到「最后一片响应体发出去」，而不是 handler 返回为止。

    StreamingResponse 的分片是 handler 返回之后才逐个发出去的，用纯 ASGI 中间件
    包住整个调用才覆盖得住这段。若把标记挂在 handler 上，返回 StreamingResponse
    的那一刻标记就注销了 —— 长流照样会在下一次热重载时被掐断。
    """
    marks: List[Dict[int, int]] = []

    async def _send(message):
        pass

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        marks.append(dict(main._inflight_by_generation))   # 正在发分片
        await send({"type": "http.response.body", "body": b"data: 1\n\n"})
        marks.append(dict(main._inflight_by_generation))   # 正在发分片
        await send({"type": "http.response.body", "body": b""})

    asyncio.run(main._ClientLifetimeMiddleware(_app)({"type": "http"}, None, _send))

    assert marks and all(m for m in marks), f"发分片期间标记丢了: {marks}"
    assert not main._inflight_by_generation, "请求结束后标记必须注销"


def test_middleware_marks_are_released_even_on_failure():
    async def _app(scope, receive, send):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(
            main._ClientLifetimeMiddleware(_app)({"type": "http"}, None, None)
        )
    assert not main._inflight_by_generation, "异常路径漏掉注销 = 旧 client 永不回收"


def test_middleware_ignores_non_http_scopes():
    """lifespan / websocket 不是请求，不该记账（否则计数永远回不到 0）。"""
    seen: List[Any] = []

    async def _app(scope, receive, send):
        seen.append(dict(main._inflight_by_generation))

    asyncio.run(
        main._ClientLifetimeMiddleware(_app)({"type": "lifespan"}, None, None)
    )
    assert seen == [{}]
    assert not main._inflight_by_generation


class _FakeStats:
    """替掉真实 UsageStats：reload_runtime 只跟它要这两个调用，别去碰真实统计库。"""

    def __init__(self) -> None:
        self.upserted: List[Any] = []

    def reconfigure(self, **kwargs) -> None:
        pass

    def sync_tenants(self, tenants) -> bool:
        self.upserted.append(list(tenants))
        return True


def test_generation_increments_on_reload(monkeypatch):
    """热重载必须给新集合分配新的世代号，否则新旧混在一起没法判定。"""
    old = _client_set()
    monkeypatch.setattr(main, "http_clients", old)
    monkeypatch.setattr(main, "usage_stats", _FakeStats())
    monkeypatch.setattr(main, "build_http_clients", lambda s: _client_set())

    before = main._client_generation
    asyncio.run(main.reload_runtime())

    assert main._client_generation == before + 1, "新集合没拿到新世代号"
    assert main._retired_clients == [(before, old)], "退役的旧集合没带上它原来的世代号"
    assert main.http_clients is not old


def test_middleware_covers_a_real_streaming_response():
    """用真的 Starlette StreamingResponse 验证「纯 ASGI 中间件能覆盖住流式响应体」。

    这是整个修法的立足点：分片是 handler 返回之后才由 starlette 逐个发出去的，
    如果 ``await self.app(scope, receive, send)`` 在 handler 返回时就返回了，
    标记会提前注销，长流照样会被热重载掐断 —— 而且单元测试和集成测试都看不出来。
    """
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    marks_during_stream: List[Dict[int, int]] = []

    async def _gen():
        for i in range(3):
            marks_during_stream.append(dict(main._inflight_by_generation))
            yield f"data: {i}\n\n".encode()

    async def _endpoint(request):
        return StreamingResponse(_gen(), media_type="text/event-stream")

    app = main._ClientLifetimeMiddleware(
        Starlette(routes=[Route("/s", _endpoint)])
    )

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/s")

    resp = asyncio.run(_call())

    assert resp.text.count("data:") == 3
    assert len(marks_during_stream) == 3
    assert all(marks_during_stream), (
        f"产分片时标记已经注销了，说明中间件没覆盖到响应体发送阶段: "
        f"{marks_during_stream}"
    )
    assert not main._inflight_by_generation


def test_real_app_serves_requests_with_the_middleware():
    """中间件挂在真 app 上，一个真实请求要走得通、且事后不留下残留的记账。

    纯 ASGI 中间件写错（比如签名不对、记账不配对）不会在单元测试里暴露，
    只会表现为「整个服务 500」或者「旧 client 永不回收」。
    """
    import httpx

    async def _call():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/admin")

    resp = asyncio.run(_call())

    assert resp.status_code == 200
    assert not main._inflight_by_generation, "请求结束后记账没清空"

