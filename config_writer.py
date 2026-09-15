"""
配置写回：把 dict 序列化为 config.toml 文本，原子写回磁盘。

零依赖：不引入 tomli_w / tomlkit，仅用标准库手写针对性序列化器。
仅覆盖本项目 config.toml 实际出现的类型：str / int / float / bool / dict / list。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# 配置中可能出现的 TOML 值类型
_TomlValue = Union[str, int, float, bool, Dict[str, "_TomlValue"], List["_TomlValue"]]


# ---------------------------------------------------------------------------
#  标量/inline 值序列化
# ---------------------------------------------------------------------------

def _format_scalar(value: Any) -> str:
    """把 Python 标量格式化为 TOML 字面量。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr 保证 round-trip（如 300.0 不丢精度）
        return repr(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"不支持的标量类型: {type(value).__name__}")


def _format_string(s: str) -> str:
    """字符串 → TOML 基本字符串字面量（双引号包裹，转义控制字符）。"""
    escaped = (
        s.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    return f"\"{escaped}\""


def _format_inline_value(value: Any) -> str:
    """递归把任意 TOML 值格式化为 inline 表示（用于 value = {...} 或数组）。"""
    if isinstance(value, dict):
        items = [
            f"{_format_key(k)} = {_format_inline_value(v)}"
            for k, v in value.items()
        ]
        return "{ " + ", ".join(items) + " }"
    if isinstance(value, list):
        items = [_format_inline_value(v) for v in value]
        return "[" + ", ".join(items) + "]"
    return _format_scalar(value)


def _format_key(key: str) -> str:
    """表 key：纯字母数字/下划线/连字符用裸 key，否则加引号。"""
    if key and all(c.isalnum() or c in "_-" for c in key):
        return key
    return _format_string(key)


# ---------------------------------------------------------------------------
#  TOML 文档构建
# ---------------------------------------------------------------------------

def _section_header(path: List[str]) -> str:
    """构建表头，如 [providers.openai] / [[providers]]。"""
    return ".".join(_format_key(p) for p in path)


def _build_toml_lines(data: Dict[str, Any]) -> List[str]:
    """把 dict 转为 TOML 文本行列表，按项目约定的段顺序输出。

    输出结构（与 config.example.toml 一致）：
      顶层标量 → [timeouts] → [[providers]]... → [model_routes] → [[tenants]] → [admin]
    非顶层、嵌套的 dict/list 统一以子表/数组表形式展开。
    """
    lines: List[str] = []

    # ---- 顶层标量（保持原始插入顺序）----
    # 顶层标量的固定输出顺序（与 config.example.toml 一致）。不在此列的顶层标量
    # 会被下面的兜底循环按插入顺序补在最后 —— 所以这里主要是为了顺序稳定，
    # 漏加不会导致配置丢失。
    known_top_keys = (
        "log_level",
        "max_body_size",
        "default_provider",
        "stats_db",
        "stats_retention_days",
        "stats_prompt_max_chars",
        "stats_flush_interval",
        "stats_batch_size",
        "pool_max_connections",
        "pool_max_keepalive",
        "pool_keepalive_expiry",
        "config_watch_interval",
    )
    for k in known_top_keys:
        if k in data and not isinstance(data[k], (dict, list)):
            lines.append(f"{_format_key(k)} = {_format_scalar(data[k])}")

    # 其他顶层标量（万一有）
    for k, v in data.items():
        if k in known_top_keys or isinstance(v, (dict, list)):
            continue
        lines.append(f"{_format_key(k)} = {_format_scalar(v)}")

    # ---- [timeouts] ----
    timeouts = data.get("timeouts")
    if isinstance(timeouts, dict):
        lines.append("")
        lines.append("[timeouts]")
        for k, v in timeouts.items():
            lines.append(f"{_format_key(str(k))} = {_format_scalar(v)}")

    # ---- [[providers]] ----
    providers = data.get("providers")
    if isinstance(providers, list):
        for p in providers:
            if not isinstance(p, dict):
                continue
            lines.append("")
            lines.append("[[providers]]")
            # 顶层字段：id / api_key（共享）/ models（手动模型列表）
            for k in ("id", "api_key"):
                if k in p and not isinstance(p[k], (dict, list)):
                    lines.append(f"{_format_key(k)} = {_format_scalar(p[k])}")
            models = p.get("models")
            if isinstance(models, list) and models:
                lines.append(f"models = {_format_inline_value(models)}")
            # 协议子表
            for proto in ("openai", "anthropic"):
                sub = p.get(proto)
                if isinstance(sub, dict):
                    lines.append("")
                    lines.append(f"[providers.{proto}]")
                    for k, v in sub.items():
                        if isinstance(v, dict):
                            # headers 等子表用 inline 表
                            lines.append(f"{_format_key(str(k))} = {_format_inline_value(v)}")
                        else:
                            lines.append(f"{_format_key(str(k))} = {_format_scalar(v)}")

    # ---- [model_routes] ----
    routes = data.get("model_routes")
    if isinstance(routes, dict) and routes:
        lines.append("")
        lines.append("[model_routes]")
        for client_model, value in routes.items():
            lines.append(f"{_format_key(client_model)} = {_format_inline_value(value)}")

    # ---- [[tenants]] ----
    tenants = data.get("tenants")
    if isinstance(tenants, list):
        for t in tenants:
            if not isinstance(t, dict):
                continue
            lines.append("")
            lines.append("[[tenants]]")
            for k in ("id", "name", "api_key", "status"):
                if k in t:
                    lines.append(f"{_format_key(k)} = {_format_scalar(t[k])}")

    # ---- [admin] ----
    admin = data.get("admin")
    if isinstance(admin, dict):
        lines.append("")
        lines.append("[admin]")
        for k in ("username", "password", "password_hash"):
            if k in admin and admin[k] is not None and admin[k] != "":
                lines.append(f"{_format_key(k)} = {_format_scalar(admin[k])}")

    return lines


def dumps(data: Dict[str, Any]) -> str:
    """把配置 dict 序列化为 TOML 文本。"""
    lines = _build_toml_lines(data)
    # 文件以单个换行结尾；段间空行由构建逻辑负责
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
#  原子写回
# ---------------------------------------------------------------------------

def write_toml(path: Path, data: Dict[str, Any]) -> None:
    """原子写回：先写临时文件，再 os.replace 覆盖目标，避免写坏 config.toml。"""
    text = dumps(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        # 写失败：清理临时文件，确保目标文件不被破坏
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
