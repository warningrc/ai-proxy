"""
配置监视器测试（多 worker 热重载）。

这条链路出错的方式都很隐蔽：要么重复重载（管理接口刚应用完，watcher 又应用一遍），
要么配置改坏了直接把 worker 带走。所以下面重点覆盖去重和容错。
"""

from __future__ import annotations

import asyncio

import pytest

import config_watcher
from config_watcher import ConfigWatcher, note_config_applied, register_watcher


class _Recorder:
    """假的 on_reload 回调：记账，可选抛异常。"""

    def __init__(self, exc: BaseException | None = None):
        self.calls = 0
        self.exc = exc

    async def __call__(self) -> None:
        self.calls += 1
        if self.exc is not None:
            raise self.exc


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('log_level = "INFO"\n', encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _clear_registry():
    """注册表是模块级全局，测试之间必须清干净。"""
    config_watcher._watchers.clear()
    yield
    config_watcher._watchers.clear()


def _check(w: ConfigWatcher) -> bool:
    return asyncio.run(w.check_once())


# ---------------------------------------------------------------------------
#  变更检测
# ---------------------------------------------------------------------------

def test_no_change_does_not_reload(cfg):
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    assert _check(w) is False
    assert _check(w) is False
    assert rec.calls == 0


def test_content_change_triggers_reload(cfg):
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    assert _check(w) is True
    assert rec.calls == 1


def test_reload_happens_only_once_per_change(cfg):
    """同一次变更只能触发一次，否则每 2 秒都会重建一遍连接池。"""
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    _check(w)
    _check(w)
    _check(w)
    assert rec.calls == 1


def test_second_change_triggers_again(cfg):
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    _check(w)
    cfg.write_text('log_level = "WARNING"\n', encoding="utf-8")
    _check(w)
    assert rec.calls == 2


def test_touch_without_content_change_is_ignored(cfg):
    """编辑器「保存但内容没变」不该触发重载 —— 靠内容哈希而不是 mtime。"""
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text('log_level = "INFO"\n', encoding="utf-8")  # 内容相同，mtime 变了
    assert _check(w) is False
    assert rec.calls == 0


def test_missing_file_is_not_a_change(cfg):
    """文件被临时删掉（比如原子写回的空档）不能当成变更。"""
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.unlink()
    assert _check(w) is False
    assert rec.calls == 0


# ---------------------------------------------------------------------------
#  去重：管理接口刚应用过的配置不能再应用一遍
# ---------------------------------------------------------------------------

def test_note_applied_suppresses_self_reload(cfg):
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    register_watcher(w)
    # 模拟管理接口：写盘 → config.reload() → note_config_applied()
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    note_config_applied()
    assert _check(w) is False
    assert rec.calls == 0


def test_note_config_applied_reaches_all_watchers(cfg):
    recs = [_Recorder(), _Recorder()]
    ws = [ConfigWatcher(cfg, 1.0, r) for r in recs]
    for w in ws:
        register_watcher(w)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    note_config_applied()
    for w in ws:
        assert _check(w) is False


def test_later_change_still_reloads_after_note_applied(cfg):
    """抬高基线不能把后续的真实变更也一起吞掉。"""
    rec = _Recorder()
    w = ConfigWatcher(cfg, 1.0, rec)
    register_watcher(w)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    note_config_applied()
    cfg.write_text('log_level = "WARNING"\n', encoding="utf-8")
    assert _check(w) is True
    assert rec.calls == 1


# ---------------------------------------------------------------------------
#  容错
# ---------------------------------------------------------------------------

def test_reload_failure_does_not_kill_the_watcher(cfg, caplog):
    """回调抛异常时，watcher 必须活下来，且不能每轮疯狂重试。"""
    rec = _Recorder(exc=RuntimeError("boom"))
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    with caplog.at_level("ERROR", logger="ai-proxy"):
        assert _check(w) is True
    assert rec.calls == 1
    assert _check(w) is False          # 基线已推进，不再重试
    assert rec.calls == 1
    assert any("reload failed" in r.message for r in caplog.records)


def test_system_exit_from_bad_config_is_contained(cfg):
    """config.reload() 对坏配置会 sys.exit；不能被它带走后台任务。"""
    rec = _Recorder(exc=SystemExit(1))
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text("这不是 toml", encoding="utf-8")
    assert _check(w) is True           # SystemExit 是 BaseException，也要被兜住
    assert rec.calls == 1


def test_recovers_after_failure_when_config_fixed(cfg):
    rec = _Recorder(exc=RuntimeError("boom"))
    w = ConfigWatcher(cfg, 1.0, rec)
    cfg.write_text("坏配置", encoding="utf-8")
    _check(w)
    rec.exc = None
    cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
    assert _check(w) is True
    assert rec.calls == 2


# ---------------------------------------------------------------------------
#  生命周期
# ---------------------------------------------------------------------------

def test_start_stop_are_idempotent(cfg):
    rec = _Recorder()
    w = ConfigWatcher(cfg, 0.01, rec)

    async def run() -> None:
        w.start()
        w.start()
        await asyncio.sleep(0.05)
        await w.stop()
        await w.stop()

    asyncio.run(run())
    assert rec.calls == 0


def test_interval_has_a_floor(cfg):
    """间隔配置成 0 会让每个 worker 空转读文件，必须兜住。"""
    w = ConfigWatcher(cfg, 0.0, _Recorder())
    assert w._interval >= config_watcher._MIN_INTERVAL


def test_loop_detects_change(cfg):
    """跑真循环（不手动 check_once）：改文件后能自己发现。"""
    rec = _Recorder()
    w = ConfigWatcher(cfg, 0.01, rec)

    async def run() -> None:
        w.start()
        await asyncio.sleep(0.03)
        assert rec.calls == 0        # 还没改，不能触发
        cfg.write_text('log_level = "DEBUG"\n', encoding="utf-8")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if rec.calls:
                break
        await w.stop()

    asyncio.run(run())
    assert rec.calls == 1
