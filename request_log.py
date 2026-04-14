"""
Structured logging helpers for upstream OpenAI calls (no secrets, truncated payloads).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("ai-proxy")


def json_preview(obj: Any, max_chars: int = 6000) -> str:
    """Serialize to JSON and truncate for logs."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception as e:
        return f"<json.dumps failed: {e}>"
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + f"...<+{len(s) - max_chars} chars>"


def summarize_openai_request(req: Dict[str, Any]) -> str:
    """One-line safe summary for POST chat/completions body."""
    model = req.get("model", "?")
    stream = req.get("stream", False)
    msgs = req.get("messages") or []
    n_msg = len(msgs)
    roles = []
    for m in msgs[:12]:
        if isinstance(m, dict):
            roles.append(m.get("role", "?"))
        else:
            roles.append("?")
    extra = f" +{n_msg - 12} more" if n_msg > 12 else ""
    tools = req.get("tools")
    tool_hint = f" tools={len(tools)}" if tools else ""
    tc = req.get("tool_choice")
    tc_hint = f" tool_choice={tc!r}" if tc is not None else ""
    return (
        f"model={model!r} stream={stream} messages={n_msg} roles={roles}{extra}"
        f"{tool_hint}{tc_hint} max_tokens={req.get('max_tokens')!r}"
    )


def summarize_openai_response(resp_json: Dict[str, Any]) -> str:
    """One-line summary of chat completion JSON."""
    cid = resp_json.get("id", "?")
    model = resp_json.get("model", "?")
    choices = resp_json.get("choices") or []
    ch0 = choices[0] if choices else {}
    finish = ch0.get("finish_reason")
    msg = ch0.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        content_hint = f"text_len={len(content)}"
    elif isinstance(content, list):
        content_hint = f"parts={len(content)}"
    else:
        content_hint = f"type={type(content).__name__}"
    tc = msg.get("tool_calls")
    if tc:
        content_hint += f" tool_calls={len(tc)}"
    u = resp_json.get("usage") or {}
    usage_s = (
        f"prompt={u.get('prompt_tokens', '?')} completion={u.get('completion_tokens', '?')} "
        f"total={u.get('total_tokens', '?')}"
    )
    return f"id={cid!r} model={model!r} finish={finish!r} {content_hint} usage[{usage_s}]"


def log_stream_chunk_debug(req_id: str, chunk_index: int, data: Dict[str, Any]) -> None:
    """Log a single parsed SSE chunk at DEBUG (can be noisy)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    preview = json_preview(data, max_chars=4000)
    logger.debug("[%s] upstream SSE chunk #%s %s", req_id, chunk_index, preview)
