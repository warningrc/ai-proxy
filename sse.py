"""
上游 SSE 流的字节级按行切分。

透传路径（Anthropic passthrough / OpenAI passthrough）只需要把上游字节原样搬运
给客户端，但又要顺带解析 data 行来统计 usage。用 httpx 的 ``aiter_lines()`` 会
先把整个流解成 str、调用方再 ``.encode()`` 回去数字节数，Starlette 最后还要编码
一次 —— 同一个流全量编解码三遍，纯属白烧 CPU。

这里直接在 bytes 上切行，只有真正要解析的 data 行才解码。

为什么按字节找 ``\\n``/``\\r`` 安全：UTF-8 的多字节序列里，续字节和首字节都
>= 0x80，不可能等于 0x0A/0x0D，所以换行字节永远不会落在多字节字符内部。
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

_CR = 0x0D
_LF = 0x0A


async def aiter_raw_lines(response: httpx.Response) -> AsyncIterator[bytes]:
    """按 SSE 行终止符切分上游字节流，逐行 yield（不含终止符）。

    与 ``httpx.Response.aiter_lines()`` 的行语义一致：``\\n`` / ``\\r\\n`` / ``\\r``
    都算一个换行，行尾终止符不返回；流末尾没有终止符的残行也会返回。

    注意用的是 ``aiter_bytes()`` 而非 ``aiter_raw()``：上游若开了 gzip，
    必须解压后再切行。
    """
    buf = b""
    scan = 0       # buf 中尚未扫描到的下标
    emit_from = 0  # 当前行的起始下标

    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        buf += chunk
        i = scan
        n = len(buf)
        while i < n:
            c = buf[i]
            if c == _LF:
                yield buf[emit_from:i]
                i += 1
                emit_from = i
            elif c == _CR:
                if i + 1 == n:
                    # \r 可能是 \r\n 的前半，等下一块再判，不能急着断行
                    break
                yield buf[emit_from:i]
                i += 2 if buf[i + 1] == _LF else 1
                emit_from = i
            else:
                i += 1
        scan = i
        if emit_from:
            # 已发出的行不再需要留在缓冲区里，否则长连接下的缓冲区会无限增长
            buf = buf[emit_from:]
            scan -= emit_from
            emit_from = 0

    if buf.endswith(b"\r"):
        # 收尾时这个 \r 后面不可能再有字节了，所以它一定是裸 \r，按终止符处理：
        # 把前面积压的行发出去（可能是空行），不能让它跟着 \r 一起当内容。
        yield buf[:-1]
    elif buf:
        yield buf
