"""
多 worker 下统计库抢写锁的三种后果，以及它们各自不能再发生。

生产现场（2026-09-15 20:27，4 worker 重启）：一个进程在 VACUUM（独占写锁重写
2.8GB 的库，约 8s），另外几个进程此刻正在启动，于是

  1. `init_db` 里的 PRAGMA 报 database is locked —— 异常从 lifespan 冒出去，
     uvicorn 打印 "Application startup failed. Exiting."，worker 反复重启；
  2. `upsert_tenants` 同样报锁 —— 这是**鉴权路径**：内存里的 key->tenant 映射
     就靠它填，填不上等于所有租户的 key 都判非法（401）；
  3. 每个 worker 都各跑一份清理，谁都想 DELETE + VACUUM，抢锁本身成了常态。

对应到这一组测试：启动不能被统计库带崩、鉴权不能依赖统计库写入、全局清理同时
只能有一个进程在跑。
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import time

import pytest

import main
import usage_stats
from usage_stats import UsageStats


def _locked() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database is locked")


@pytest.fixture
def stats(tmp_path):
    s = UsageStats(str(tmp_path / "stats.db"))
    s.init_db()
    yield s
    s.close()


# ---------------------------------------------------------------------------
#  1. 启动不能被统计库带崩
# ---------------------------------------------------------------------------

def test_init_db_failure_leaves_no_half_initialized_state(tmp_path, monkeypatch):
    """init_db 抛异常后，实例必须是干净的「没有统计库」，不能留半个。

    留下半个更糟：_conn 有、writer 线程和读连接没起，表现是「请求都正常、
    统计默默变少」，比直接报错难查得多。
    """
    s = UsageStats(str(tmp_path / "stats.db"))
    monkeypatch.setattr(
        usage_stats, "_apply_pragmas", lambda conn: (_ for _ in ()).throw(_locked())
    )

    with pytest.raises(sqlite3.OperationalError):
        s.init_db()

    assert s._conn is None
    assert s._read_conn is None
    assert s._writer_thread is None
    # 后续调用必须安全降级，而不是 KeyError/NoneType 之类的二次伤害
    s.record_usage(usage_stats.UsageRecord(req_id="r1", tenant_id="t1", provider="p", model="m", endpoint="e"))
    assert s._queue.empty(), "没有统计库还往队列里塞记录，等于攒着没人写的垃圾"
    assert usage_stats.UsageStats.resolve_tenant(s, "any-key") == "default"


def test_startup_survives_a_locked_stats_db(monkeypatch):
    """锁冲突重试用尽后，启动必须继续 —— 统计是旁路，不是启动的前置条件。"""
    monkeypatch.setattr(
        main.usage_stats, "init_db", lambda: (_ for _ in ()).throw(_locked())
    )
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    assert main._init_stats() is False  # 不抛


def test_startup_retries_a_transiently_locked_stats_db(monkeypatch):
    """N 个 worker 同时启动时抢锁是常态：第一次失败、第二次成功要能起来。"""
    calls = []

    def _init_db():
        calls.append(1)
        if len(calls) == 1:
            raise _locked()

    monkeypatch.setattr(main.usage_stats, "init_db", _init_db)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    assert main._init_stats() is True
    assert len(calls) == 2


def test_non_lock_failures_are_not_retried(monkeypatch):
    """库根本打不开（磁盘满、文件损坏）重试没有意义，别白等 3.5 秒。"""
    calls = []

    def _init_db():
        calls.append(1)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(main.usage_stats, "init_db", _init_db)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    assert main._init_stats() is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------
#  2. 鉴权不能依赖统计库写入
# ---------------------------------------------------------------------------

_TENANTS = [{"id": "id-0001", "name": "n", "api_key": "sk-0001", "status": "active"}]


def test_auth_still_works_when_the_tenant_write_fails(stats, monkeypatch):
    """实测生产影响：租户写不进去时，内存映射还得有值，否则所有 key 401。

    鉴权链路是 resolve_tenant(raw_key) -> 内存映射 -> 匹配不上就 401，
    跟统计库没有任何关系 —— 只是那份映射平时是从库里读回来的。
    """
    monkeypatch.setattr(
        stats, "upsert_tenants", lambda t: (_ for _ in ()).throw(_locked())
    )
    monkeypatch.setattr(usage_stats.time, "sleep", lambda s: None)

    assert stats.sync_tenants(_TENANTS) is False
    assert stats.resolve_tenant("sk-0001") == "id-0001", "租户写不进库就把 key 判成非法"


def test_auth_still_works_when_the_db_was_never_initialized(tmp_path):
    """库压根没起来（init_db 失败）时同样不能把请求全变成 401。"""
    s = UsageStats(str(tmp_path / "stats.db"))
    assert s.sync_tenants(_TENANTS) is False
    assert s.resolve_tenant("sk-0001") == "id-0001"


def test_sync_tenants_retries_then_succeeds(stats, monkeypatch):
    """重试成功时映射要来自库（status 以库里的为准），而不是降级值。"""
    real = stats.upsert_tenants
    calls = []

    def flaky(tenants):
        calls.append(1)
        if len(calls) == 1:
            raise _locked()
        real(tenants)

    monkeypatch.setattr(stats, "upsert_tenants", flaky)
    monkeypatch.setattr(usage_stats.time, "sleep", lambda s: None)

    assert stats.sync_tenants(_TENANTS) is True
    assert len(calls) == 2
    assert stats.resolve_tenant("sk-0001") == "id-0001"
    assert stats.get_tenant_info_by_id("id-0001").status == "active"


def test_degraded_mode_uses_config_status(stats, monkeypatch):
    """降级路径用配置里的 status —— 库被改过的状态这时拿不到，日志要写明。"""
    monkeypatch.setattr(
        stats, "upsert_tenants", lambda t: (_ for _ in ()).throw(_locked())
    )
    monkeypatch.setattr(usage_stats.time, "sleep", lambda s: None)

    stats.sync_tenants([{**_TENANTS[0], "status": "disabled"}])
    assert stats.get_tenant_info_by_id("id-0001").status == "disabled"


# ---------------------------------------------------------------------------
#  3. 全局清理同一时刻只能有一个进程在跑
# ---------------------------------------------------------------------------

def test_only_one_claim_succeeds_at_a_time(stats):
    """同一份库上重复认领要失败；放掉之后下一个进程才能接手。"""
    with stats.claim_global_cleanup() as first:
        assert first is True
        with stats.claim_global_cleanup() as second:
            assert second is False, "两个进程同时跑 DELETE+VACUUM，写锁会互相顶死"
    with stats.claim_global_cleanup() as third:
        assert third is True


def test_claim_is_released_even_if_the_round_raises(stats):
    """清理中途抛异常也必须放锁，否则库会永远没人清理。"""
    with pytest.raises(RuntimeError):
        with stats.claim_global_cleanup() as claimed:
            assert claimed is True
            raise RuntimeError("VACUUM 中途炸了")

    with stats.claim_global_cleanup() as again:
        assert again is True


def test_claim_works_across_processes(tmp_path):
    """真正的跨进程验证：多 worker 是不同的进程，线程内测不出来。

    每个子进程都去认领同一份库，同时只有一个该成功。
    """
    db = str(tmp_path / "stats.db")
    UsageStats(db).init_db()
    script = (
        "import sys, time; sys.path.insert(0, %r);"
        "from usage_stats import UsageStats;"
        "s = UsageStats(%r);"
        "import contextlib;"
        "c = s.claim_global_cleanup(); claimed = c.__enter__();"
        "print('WIN' if claimed else 'SKIP', flush=True);"
        "time.sleep(0.4);"
        "c.__exit__(None, None, None)"
    ) % (str(__import__("pathlib").Path(usage_stats.__file__).parent), db)

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        for _ in range(3)
    ]
    outs = [p.communicate(timeout=30)[0].strip() for p in procs]

    assert outs.count("WIN") == 1, f"认领不是互斥的: {outs}"
    assert outs.count("SKIP") == 2


def test_cleanup_skips_the_round_when_another_worker_claimed_it(monkeypatch):
    """认领失败的那几个 worker 必须什么都不做（不 DELETE、不 VACUUM）。"""
    ran = []
    monkeypatch.setattr(main.usage_stats, "claim_global_cleanup", _not_claimed)
    monkeypatch.setattr(main.usage_stats, "cleanup_old_records", lambda d: ran.append("delete"))
    monkeypatch.setattr(main.usage_stats, "vacuum", lambda: ran.append("vacuum"))

    main._stats_cleanup_once()

    assert ran == []


def test_cleanup_runs_when_it_wins_the_claim(stats, monkeypatch):
    """拿到锁的那一个进程照常清理，功能不能被这层新的条件挡住。"""
    from datetime import datetime, timedelta

    old = (datetime.now() - timedelta(days=40)).strftime(usage_stats._TS_FORMAT)
    stats._conn.execute(
        "INSERT INTO request_log (ts, req_id, tenant_id, provider, model, endpoint,"
        " input_tokens, output_tokens) VALUES (?, 'r-old', 't1', 'p1', 'm1', 'messages', 1, 1)",
        (old,),
    )
    stats._conn.commit()

    monkeypatch.setattr(main, "usage_stats", stats)
    monkeypatch.setattr(main.settings, "STATS_RETENTION_DAYS", 30)

    main._stats_cleanup_once()

    left = stats._conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
    assert left == 0, "拿到锁却没删掉超期记录"


class _ClaimCM:
    def __init__(self, ok: bool) -> None:
        self._ok = ok

    def __enter__(self) -> bool:
        return self._ok

    def __exit__(self, *exc) -> None:
        return None


def _not_claimed():
    return _ClaimCM(False)


# ---------------------------------------------------------------------------
#  4. busy_timeout 要长于 VACUUM 占锁的时间
# ---------------------------------------------------------------------------

def test_busy_timeout_outlasts_a_vacuum(stats):
    """5s 等不过一个 VACUUM（实测 2.8GB 约 8s），写入会成片失败。"""
    assert usage_stats._BUSY_TIMEOUT_MS >= 20_000
    for conn in (stats._conn, stats._read_conn):
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == usage_stats._BUSY_TIMEOUT_MS


def test_busy_timeout_is_set_before_anything_that_takes_a_lock(monkeypatch):
    """PRAGMA 的顺序有意义：busy_timeout 必须第一个设。

    放后面的话，它前面那些要拿锁的语句用的是连接对象的默认超时，配置值等于没生效。
    """
    seen = []

    class _Recorder:
        def execute(self, sql, *a):
            seen.append(sql)

    usage_stats._apply_pragmas(_Recorder())
    assert seen[0].startswith("PRAGMA busy_timeout"), seen
