"""
SSE 字节级切行（sse.aiter_raw_lines）测试。

这是透传路径的地基：切行切错了，客户端看到的就是碎掉的 JSON，而且
上游 SSE 的分块边界完全随机，所以必须覆盖「终止符跨块」和「多字节字符跨块」
这两类真实会发生的情况。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import httpx

from sse import aiter_raw_lines


class _FakeResponse:
    """只需要 aiter_bytes()，不依赖 httpx 连接。"""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c


def _lines(chunks: List[bytes]) -> List[bytes]:
    async def run() -> List[bytes]:
        return [line async for line in aiter_raw_lines(_FakeResponse(chunks))]

    return asyncio.run(run())


# ---------------------------------------------------------------------------
#  基础切分
# ---------------------------------------------------------------------------

def test_splits_on_lf():
    assert _lines([b"a\nb\nc\n"]) == [b"a", b"b", b"c"]


def test_terminator_is_stripped():
    assert _lines([b"data: {}\n\n"]) == [b"data: {}", b""]


def test_blank_lines_survive():
    """空行是 SSE 的事件分隔符，绝不能被丢掉。"""
    assert _lines([b"event: x\ndata: 1\n\n"]) == [b"event: x", b"data: 1", b""]


def test_crlf_counts_as_one_break():
    assert _lines([b"a\r\nb\r\n"]) == [b"a", b"b"]


def test_bare_cr_counts_as_break():
    """SSE 规范里 \\r 单独也是合法行终止符。"""
    assert _lines([b"a\rb\rc"]) == [b"a", b"b", b"c"]


def test_trailing_partial_line_is_flushed():
    """上游没补最后一个换行时，残行也必须发出去，不能吞掉。"""
    assert _lines([b"a\nb"]) == [b"a", b"b"]


def test_empty_stream_yields_nothing():
    assert _lines([]) == []
    assert _lines([b""]) == []


# ---------------------------------------------------------------------------
#  跨块边界
# ---------------------------------------------------------------------------

def test_line_split_across_chunks():
    """上游一个 TCP 包里塞半行是常态。"""
    assert _lines([b"data: 12", b"3\n\n"]) == [b"data: 123", b""]


def test_crlf_split_across_chunks():
    """\\r 落在一块末尾、\\n 落在下一块 —— 不能当成两个换行。"""
    assert _lines([b"a\r", b"\nb\r", b"\n"]) == [b"a", b"b"]


def test_trailing_cr_at_eof():
    """流以裸 \\r 结尾时，它是终止符而不是行内容（与 httpx 行为对齐）。"""
    assert _lines([b"a\r"]) == [b"a"]
    assert _lines([b"a\r\r"]) == [b"a", b""]


def test_byte_at_a_time():
    """极端情况：上游每块只给一个字节。"""
    assert _lines([bytes([c]) for c in b"data: 1\n\n"]) == [b"data: 1", b""]


def test_multibyte_char_split_across_chunks():
    """UTF-8 字符被切成两半时，字节必须原样拼回来，不能出现替换字符。"""
    payload = "你好".encode("utf-8")
    chunks = [b"data: " + payload[:2], payload[2:] + b"\n\n"]
    out = _lines(chunks)
    assert out == [b"data: " + payload, b""]
    assert out[0][6:].decode("utf-8") == "你好"


def test_multibyte_content_never_confused_with_newline():
    """所有非 ASCII 字节都 >= 0x80，不可能被误判成换行。"""
    text = "".join(chr(c) for c in range(0x4E00, 0x4E20))  # 一批三字节汉字
    raw = f"data: {text}\n\n".encode("utf-8")
    out = _lines([raw])
    assert out[0][6:].decode("utf-8") == text
    assert out[1] == b""


def test_many_lines_in_chunks():
    """长流：每行都恰好产出一次，不重不漏。"""
    chunks = [f"data: {i}\n\n".encode() for i in range(1000)]
    out = _lines(chunks)
    assert len(out) == 2000
    assert out[0] == b"data: 0"
    assert out[-2] == b"data: 999"


# ---------------------------------------------------------------------------
#  与 httpx.aiter_lines 的差分对比
# ---------------------------------------------------------------------------

class _ChunkedStream(httpx.AsyncByteStream):
    """按调用方指定的块边界吐出字节，模拟任意的上游分块。"""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c

    async def aclose(self) -> None:
        pass


def _httpx_lines(chunks: List[bytes]) -> List[str]:
    """参考实现：真 httpx 响应走 aiter_lines()。"""

    async def run() -> List[str]:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_ChunkedStream(chunks))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with client.stream("GET", "http://upstream.test/sse") as resp:
                return [line async for line in resp.aiter_lines()]

    return asyncio.run(run())


# 覆盖各种块边界：整块 / 逐字节 / 终止符被劈开 / 多字节字符被劈开 / 无尾换行
_BOUNDARY_CASES = [
    [b"event: message_start\ndata: {\"a\": 1}\n\n"],
    [b"event: message_start\r\ndata: {\"a\": 1}\r\n\r\n"],   # CRLF 版
    [b"data: 1\n", b"data: 2\n", b"data: 3\n"],
    [bytes([b]) for b in b"data: hello\n\n"],
    [b"data: \xe4\xbd", b"\xa0\xe5\xa5\xbd\n\n"],          # 「你好」被劈开
    [b"a\r", b"\nb\r", b"\n"],                              # CRLF 被劈开
    [b"a\rb\r"],                                            # 裸 CR
    [b"a\r"],                                               # 结尾单个 CR
    [b"a\r\r"],                                             # 结尾两个 CR
    [b"no trailing newline"],
    [b"a\n\n"],
    [b""],
]


def test_matches_httpx_aiter_lines():
    """差分测试：字节级切行必须和 aiter_lines 的行语义完全一致。

    换实现的收益是不做全量编解码，前提是行切分一个字都不能变 ——
    上游 SSE 的分块边界随机，靠手写用例穷举不现实，直接拿 httpx 当基准。
    """
    for chunks in _BOUNDARY_CASES:
        expected = _httpx_lines(chunks)
        got = [line.decode("utf-8") for line in _lines(chunks)]
        assert got == expected, f"chunks={chunks!r}"
