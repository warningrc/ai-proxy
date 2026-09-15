"""
配置热重载编排：前端配置 JSON → 校验 → 写回 config.toml → 切换运行态。

流程（先校验后切换，失败不破坏运行态）：
1. 把前端 dict 序列化为 TOML 文本（config_writer.dumps）。
2. 用 tomllib 重新解析，确认 TOML 语法合法。
3. 在**临时文件**上构造一个临时 Settings 实例，复用 config.py 全部 _parse_*
   校验链路；通过捕获 SystemExit（_die 触发）拿到校验错误信息，不影响主进程。
4. 校验通过 → 原子写回真实 config.toml → config.reload() 更新模块级 settings
   → main.reload_runtime() 重建 http_clients + 同步租户。
5. 任一步异常：保留旧 settings/clients，返回错误详情。

为什么用临时文件而非直接构造：config.py 的 _parse_* 在校验失败时调 _die →
sys.exit，这是为 CLI fail-fast 设计的。在 API 上下文我们通过捕获 SystemExit
复用这套校验，避免重复维护一份校验逻辑。
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import tomllib
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import config as config_module
import config_watcher
import config_writer


@dataclass
class ReloadResult:
    success: bool
    error: Optional[str] = None
    detail: Optional[str] = None


def _validate_via_temp_settings(data: Dict[str, Any]) -> Optional[str]:
    """用临时文件构造 Settings 做校验。

    返回 None 表示校验通过；返回字符串表示错误信息。
    """
    toml_text = config_writer.dumps(data)

    # 先确认 TOML 语法合法
    try:
        tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as e:
        return f"TOML 语法错误: {e}"

    # 写入临时文件，借用 Settings() 走完整 _parse_* 校验链路
    fd, tmp_path = tempfile.mkstemp(prefix="config-validate-", suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(toml_text)

        # _die 会 print 到 stderr 后 sys.exit；捕获两者
        err_buf = io.StringIO()
        old_env = os.environ.get(config_module.CONFIG_FILE_ENV)
        os.environ[config_module.CONFIG_FILE_ENV] = tmp_path
        try:
            with redirect_stderr(err_buf):
                config_module.Settings()  # noqa: F841 — 触发校验，构造失败会抛 SystemExit
            return None  # 校验通过
        except SystemExit:
            msg = err_buf.getvalue().strip()
            # 去掉 _die 的 "CRITICAL ERROR: " 前缀
            if msg.startswith("CRITICAL ERROR:"):
                msg = msg[len("CRITICAL ERROR:"):].strip()
            return msg or "配置校验失败"
        finally:
            if old_env is None:
                os.environ.pop(config_module.CONFIG_FILE_ENV, None)
            else:
                os.environ[config_module.CONFIG_FILE_ENV] = old_env
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def apply_config(config_dict: Dict[str, Any]) -> ReloadResult:
    """同步入口：校验 + 写回 + 重载 settings（不含 http_clients 重建）。

    http_clients 重建是 async 的，由调用方在事件循环里执行
    ``await main.reload_runtime()``。
    """
    # 1. 校验
    err = _validate_via_temp_settings(config_dict)
    if err is not None:
        return ReloadResult(success=False, error="配置校验失败", detail=err)

    # 2. 写回真实 config.toml（原子）
    target_path: Path = config_module.settings.CONFIG_PATH
    try:
        config_writer.write_toml(target_path, config_dict)
    except Exception as e:
        return ReloadResult(
            success=False,
            error="写入 config.toml 失败",
            detail=f"{type(e).__name__}: {e}",
        )

    # 3. 重载模块级 settings（原地替换字段，引用不变）
    try:
        config_module.reload()
    except SystemExit as e:
        # reload 内部 _load_toml/_parse_* 可能 _die；理论上步骤 1 已校验过，
        # 这里兜底防止进程被杀
        return ReloadResult(
            success=False,
            error="重载配置时失败（配置已写盘但未生效）",
            detail=f"SystemExit: {e}",
        )

    # 4. 抬高配置监视器基线：本进程已经应用了这份配置，watcher 不必再触发一次。
    #    必须在任何 await 之前做完 —— 这里一旦让出事件循环，watcher 任务就可能
    #    看到新文件跟着重载一遍（多 worker 下其余 worker 正是靠那条路径生效）。
    config_watcher.note_config_applied()

    return ReloadResult(success=True)


_reload_callbacks = []

def register_reload_callback(cb) -> None:
    """注册运行态重载回调（在 settings 更新后调用）。"""
    _reload_callbacks.append(cb)


async def apply_config_and_runtime(config_dict: Dict[str, Any]) -> ReloadResult:
    """完整热重载：校验 + 写回 + 重载 settings + 触发运行态回调（重建 http_clients + 同步租户）。

    供 admin_api PUT /admin/api/config 调用。
    """
    result = apply_config(config_dict)
    if not result.success:
        return result

    # settings 已更新；触发运行态重建
    try:
        for cb in _reload_callbacks:
            import inspect
            if asyncio.iscoroutinefunction(cb) or inspect.isawaitable(cb):
                await cb()
            else:
                cb()
    except Exception as e:
        return ReloadResult(
            success=False,
            error="配置已保存并重载，但重建运行态失败（可能需要重启服务）",
            detail=f"{type(e).__name__}: {e}",
        )

    return ReloadResult(success=True)
