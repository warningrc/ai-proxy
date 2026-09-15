"""
SSE 透传路径测试：字节原样转发 + 顺带解析 usage。

透传的契约是「客户端收到的字节 == 上游给的字节」，所以这里的主断言就是
逐字节相等 —— 顺手解析 usage 不能反过来改动转发出去的内容。
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable, List, Optional, Tuple

import httpx

import main
from main import _anthropic_stream_passthrough, _openai_stream_passthrough


class _ChunkedStream(httpx.AsyncByteStream):
    """按指定块边界吐出字节；可选在吐完 fail_after 块后抛异常，模拟上游中断。

    注意末尾那次 raise：中断发生在「请求下一块数据」时，所以即使 chunks 正好
    用完，只要设了 fail_after 就得再抛一次 —— 否则流会被当成正常结束
    （正常结束会 flush 缓冲区里的残行，中断则不会，两者行为不同）。
    """

    def __init__(self, chunks: List[bytes], fail_after: Optional[int] = None,
                 exc: Optional[Exception] = None):
        self._chunks = chunks
        self._fail_after = fail_after
        self._exc = exc

    async def __aiter__(self) -> AsyncIterator[bytes]:
        delivered = 0
        for c in self._chunks:
            if self._fail_after is not None and delivered >= self._fail_after:
                raise self._exc or httpx.ReadError("upstream died")
            yield c
            delivered += 1
        if self._fail_after is not None:
            raise self._exc or httpx.ReadError("upstream died")

    async def aclose(self) -> None:
        pass


def _run(
    factory: Callable[[httpx.Response], AsyncIterator[bytes]],
    chunks: List[bytes],
    *,
    fail_after: Optional[int] = None,
    exc: Optional[Exception] = None,
) -> List[bytes]:
    async def go() -> List[bytes]:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=_ChunkedStream(chunks, fail_after, exc),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with client.stream("POST", "http://upstream.test/v1/messages") as resp:
                return [chunk async for chunk in factory(resp)]

    return asyncio.run(go())


def _by(chunks: List[bytes]) -> List[List[bytes]]:
    """把每个字节单独作为一块 —— 最恶劣的分块边界。"""
    return [[bytes([b])] for chunk in chunks for b in chunk]


# ---------------------------------------------------------------------------
#  Anthropic 透传
# ---------------------------------------------------------------------------

ANTHROPIC_HEAD = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-x",'
    b'"usage":{"input_tokens":10,"cache_read_input_tokens":3,'
    b'"cache_creation_input_tokens":2}}}\n'
    b'\n'
)

ANTHROPIC_SSE = (
    ANTHROPIC_HEAD
    + b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\xe4\xbd\xa0\xe5\xa5\xbd"}}\n'
    b'\n'
    b'event: message_delta\n'
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":7}}\n'
    b'\n'
    b'event: message_stop\n'
    b'data: {"type":"message_stop"}\n'
    b'\n'
)


def _anthropic(chunks: List[bytes], **kwargs) -> Tuple[List[bytes], List]:
    recorded = []
    orig = main._record_usage
    main._record_usage = recorded.append
    try:
        out = _run(
            lambda r: _anthropic_stream_passthrough(
                r, "req1", 0.0, tenant_id="t1", provider_id="p1", model="claude-x",
            ),
            chunks, **kwargs,
        )
    finally:
        main._record_usage = orig
    return out, recorded


def test_anthropic_passthrough_is_byte_identical():
    """客户端拿到的必须是上游原样的字节流。"""
    out, _ = _anthropic([ANTHROPIC_SSE])
    assert b"".join(out) == ANTHROPIC_SSE


def test_anthropic_passthrough_byte_at_a_time_is_identical():
    """分块边界不能影响输出：逐字节投喂时结果必须一样。"""
    out, _ = _anthropic([bytes([b]) for b in ANTHROPIC_SSE])
    assert b"".join(out) == ANTHROPIC_SSE


def test_anthropic_passthrough_extracts_usage():
    _, recorded = _anthropic([ANTHROPIC_SSE])
    assert len(recorded) == 1
    rec = recorded[0]
    assert (rec.input_tokens, rec.output_tokens) == (10, 7)
    assert rec.cache_read_tokens == 3
    assert rec.cache_creation_tokens == 2
    assert (rec.tenant_id, rec.provider, rec.model) == ("t1", "p1", "claude-x")


def test_anthropic_passthrough_yields_chunks_incrementally():
    """必须边收边发，不能攒成一大坨 —— 否则流式就没意义了。"""
    parts, _ = _anthropic([ANTHROPIC_SSE])
    assert len(parts) > 5
    assert parts[0] == b"event: message_start\n"


def test_anthropic_passthrough_survives_upstream_abort(caplog):
    """上游中途断开：已整行转发的保留，usage 照记，不能把异常抛给客户端。"""
    with caplog.at_level("WARNING", logger="ai-proxy"):
        out, recorded = _anthropic([ANTHROPIC_HEAD, ANTHROPIC_SSE[len(ANTHROPIC_HEAD):]],
                                   fail_after=1)
    assert b"".join(out) == ANTHROPIC_HEAD       # 断点前的内容照常发出
    assert len(recorded) == 1                    # 断开也要落一条用量
    assert recorded[0].input_tokens == 10        # message_start 已经解析过了
    assert recorded[0].output_tokens == 0
    assert any("stream_aborted" in r.message for r in caplog.records)


def test_anthropic_passthrough_drops_incomplete_trailing_line():
    """断在半行上时，那半行没有换行符、客户端也解析不了，直接丢弃。

    与 httpx.aiter_lines() 在异常路径下的行为一致（只有正常读完才会 flush 残行）。
    """
    partial = b'event: content_block_delta\ndata: {"type":"content_bl'
    out, _ = _anthropic([ANTHROPIC_HEAD + partial], fail_after=1)
    sent = b"".join(out)
    # 断点前那条完整的 event 行照发；没有换行符的半行丢弃
    assert sent == ANTHROPIC_HEAD + b"event: content_block_delta\n"
    assert b'{"type":"content_bl' not in sent


def test_anthropic_passthrough_ignores_malformed_data_line():
    """上游偶发坏 JSON 时，行照样透传，只是不参与统计。"""
    bad = (
        b'event: message_start\n'
        b'data: {"usage": not-json}\n'
        b'\n'
        b'event: message_delta\n'
        b'data: {"usage":{"output_tokens":5}}\n'
        b'\n'
    )
    out, recorded = _anthropic([bad])
    assert b"".join(out) == bad
    assert recorded[0].output_tokens == 5


def test_anthropic_passthrough_data_without_event_is_not_parsed():
    """没有前置 event 行的 data 不参与统计（保持既有语义）。"""
    orphan = b'data: {"usage":{"output_tokens":99}}\n\n'
    out, recorded = _anthropic([orphan])
    assert b"".join(out) == orphan
    assert recorded[0].output_tokens == 0


# ---------------------------------------------------------------------------
#  OpenAI 透传
# ---------------------------------------------------------------------------

def _chunk(content: str, usage: Optional[dict] = None) -> bytes:
    obj = {"id": "cc_1", "model": "gpt-x", "choices": [{"delta": {"content": content}}]}
    if usage:
        obj["usage"] = usage
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


OPENAI_SSE = (
    _chunk("你")
    + _chunk("好")
    + _chunk("", usage={"prompt_tokens": 11, "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 6}})
    + b"data: [DONE]\n\n"
)


def _openai(chunks: List[bytes], **kwargs) -> Tuple[List[bytes], List]:
    recorded = []
    orig = main._record_usage
    main._record_usage = recorded.append
    try:
        out = _run(
            lambda r: _openai_stream_passthrough(
                r, "req2", 0.0, tenant_id="t1", provider_id="p1", model="gpt-x",
            ),
            chunks, **kwargs,
        )
    finally:
        main._record_usage = orig
    return out, recorded


def test_openai_passthrough_is_byte_identical():
    out, _ = _openai([OPENAI_SSE])
    assert b"".join(out) == OPENAI_SSE


def test_openai_passthrough_byte_at_a_time_is_identical():
    out, _ = _openai([bytes([b]) for b in OPENAI_SSE])
    assert b"".join(out) == OPENAI_SSE


def test_openai_passthrough_extracts_usage():
    _, recorded = _openai([OPENAI_SSE])
    rec = recorded[0]
    assert (rec.input_tokens, rec.output_tokens) == (11, 4)
    assert rec.cache_read_tokens == 6
    assert rec.endpoint == "chat/completions"


def test_openai_passthrough_stops_after_done():
    """[DONE] 之后的内容不该再转发（上游在 [DONE] 后挂住时靠这个收尾）。"""
    out, _ = _openai([OPENAI_SSE + b'data: {"id":"late"}\n\n'])
    assert b"".join(out) == OPENAI_SSE
    assert out[-1] == b"data: [DONE]\n\n"


def test_openai_passthrough_synthesizes_done_on_abort(caplog):
    """上游中断且没发过 [DONE]：补一个错误帧 + [DONE]，让客户端干净结束。"""
    head = _chunk("你")
    with caplog.at_level("WARNING", logger="ai-proxy"):
        out, recorded = _openai([head, OPENAI_SSE[len(head):]], fail_after=1)
    joined = b"".join(out)
    assert joined.startswith(head)
    assert b"upstream_stream_error" in joined
    assert joined.endswith(b"data: [DONE]\n\n")
    assert joined.count(b"data: [DONE]") == 1     # 只补一个，不能重复
    assert len(recorded) == 1
    assert any("stream_aborted" in r.message for r in caplog.records)


def test_openai_passthrough_forwards_non_data_lines():
    """注释/ping 行（如 `: keep-alive`）也要原样转发，否则连接会被判死。"""
    stream = b": keep-alive\n\n" + OPENAI_SSE
    out, _ = _openai([stream])
    assert b"".join(out) == stream
