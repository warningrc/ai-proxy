"""
统计数据保留（UsageStats.cleanup_old_records / vacuum）测试。

覆盖：
- 超过保留期的记录被删除，保留期内的原样留下
- 边界：正好等于 cutoff 的记录不算「超过 N 天」，不删
- retention_days <= 0 关闭清理，一行都不删
- 分批删除跨批时总数正确
- 删除会让全时段排名缓存失效（否则前端还按旧排名配色）
- vacuum 在空闲页占比低时不执行
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from usage_stats import UsageStats, _TS_FORMAT


@pytest.fixture
def stats(tmp_path):
    s = UsageStats(str(tmp_path / "stats.db"))
    s.init_db()
    yield s
    s.close()


def _insert(s: UsageStats, ts: str, model: str = "m1", req_id: str = "r") -> None:
    """直接落库，以便指定 ts（record_usage 用的是 DEFAULT now）。"""
    s._conn.execute(
        """
        INSERT INTO request_log
            (ts, req_id, tenant_id, provider, model, endpoint,
             input_tokens, output_tokens)
        VALUES (?, ?, 't1', 'p1', ?, 'messages', 1, 1)
        """,
        (ts, req_id, model),
    )
    s._conn.commit()


def _ts(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime(_TS_FORMAT)


def _count(s: UsageStats) -> int:
    return s._conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]


# ---------------------------------------------------------------------------
#  cleanup_old_records
# ---------------------------------------------------------------------------

def test_deletes_only_expired_rows(stats):
    _insert(stats, _ts(40), req_id="old-40d")
    _insert(stats, _ts(31), req_id="old-31d")
    _insert(stats, _ts(29), req_id="fresh-29d")
    _insert(stats, _ts(1), req_id="fresh-1d")
    _insert(stats, _ts(0), req_id="now")

    deleted = stats.cleanup_old_records(30)

    assert deleted == 2
    remaining = {r[0] for r in stats._conn.execute("SELECT req_id FROM request_log")}
    assert remaining == {"fresh-29d", "fresh-1d", "now"}


def test_keeps_rows_exactly_at_cutoff(stats):
    """「超过 30 天」不含正好 30 天 —— cutoff 是 ts < cutoff，不是 <=。"""
    _insert(stats, _ts(30), req_id="exactly-30d")
    assert stats.cleanup_old_records(30) == 0
    assert _count(stats) == 1


def test_disabled_when_retention_non_positive(stats):
    _insert(stats, _ts(365), req_id="ancient")
    _insert(stats, _ts(3650), req_id="prehistoric")

    assert stats.cleanup_old_records(0) == 0
    assert stats.cleanup_old_records(-1) == 0
    assert _count(stats) == 2


def test_batched_delete_covers_all_batches(stats, monkeypatch):
    """删除是分批的，跨多批时不能漏删（monkeypatch 把批大小压到 2）。"""
    monkeypatch.setattr(UsageStats, "_CLEANUP_BATCH_SIZE", 2)
    for i in range(7):
        _insert(stats, _ts(100), req_id=f"old-{i}")
    _insert(stats, _ts(1), req_id="fresh")

    assert stats.cleanup_old_records(30) == 7
    assert _count(stats) == 1


def test_no_rows_to_delete_returns_zero(stats):
    _insert(stats, _ts(1))
    assert stats.cleanup_old_records(30) == 0


def test_invalidates_rank_cache(stats):
    """删完之后排名缓存必须失效，否则配色还按已删除数据的排名分配。"""
    _insert(stats, _ts(100), model="gone", req_id="old")
    _insert(stats, _ts(1), model="kept", req_id="new")

    assert stats.model_rank() == ["gone", "kept"]  # 先预热缓存

    stats.cleanup_old_records(30)

    assert stats._rank_cache is None
    assert stats.model_rank() == ["kept"]


def test_no_cache_invalidation_when_nothing_deleted(stats):
    _insert(stats, _ts(1), model="kept")
    stats.model_rank()
    cached = stats._rank_cache

    assert stats.cleanup_old_records(30) == 0
    assert stats._rank_cache is cached


# ---------------------------------------------------------------------------
#  clear_oversized_prompts
# ---------------------------------------------------------------------------

def _insert_prompt(s: UsageStats, ts: str, prompt: str, req_id: str = "r",
                   input_tokens: int = 10) -> None:
    s._conn.execute(
        """
        INSERT INTO request_log
            (ts, req_id, tenant_id, provider, model, endpoint, prompt,
             input_tokens, output_tokens)
        VALUES (?, ?, 't1', 'p1', 'm1', 'messages', ?, ?, 1)
        """,
        (ts, req_id, prompt, input_tokens),
    )
    s._conn.commit()


def test_clear_prompt_only_touches_oversized(stats):
    _insert_prompt(stats, _ts(1), "x" * 5000, req_id="legacy-huge")
    _insert_prompt(stats, _ts(1), "y" * 2048, req_id="at-limit")
    _insert_prompt(stats, _ts(1), "z" * 2047, req_id="under-limit")

    assert stats.clear_oversized_prompts(2048) == 1

    rows = dict(stats._conn.execute(
        "SELECT req_id, LENGTH(prompt) FROM request_log"
    ).fetchall())
    assert rows == {"at-limit": 2048, "under-limit": 2047, "legacy-huge": None}


def test_clear_prompt_keeps_rows_and_usage(stats):
    """抹掉的是字段内容，不是整行 —— 计费数据必须原样保留。"""
    _insert_prompt(stats, _ts(1), "x" * 9000, req_id="huge", input_tokens=1234)
    _insert_prompt(stats, _ts(1), "y" * 10, req_id="small", input_tokens=7)
    before = stats.query_stats(group_by="model")["stats"]

    assert stats.clear_oversized_prompts(2048) == 1

    assert _count(stats) == 2
    after = stats.query_stats(group_by="model")["stats"]
    assert after == before
    assert after[0]["input_tokens"] == 1241


def test_clear_prompt_batched(stats, monkeypatch):
    monkeypatch.setattr(UsageStats, "_CLEANUP_BATCH_SIZE", 2)
    for i in range(7):
        _insert_prompt(stats, _ts(1), "x" * 3000, req_id=f"huge-{i}")

    assert stats.clear_oversized_prompts(2048) == 7
    assert stats._conn.execute(
        "SELECT SUM(prompt IS NOT NULL) FROM request_log"
    ).fetchone()[0] == 0


def test_clear_prompt_disabled_when_limit_non_positive(stats):
    _insert_prompt(stats, _ts(1), "x" * 5000)
    assert stats.clear_oversized_prompts(0) == 0
    assert stats.clear_oversized_prompts(-1) == 0
    assert stats._conn.execute(
        "SELECT LENGTH(prompt) FROM request_log"
    ).fetchone()[0] == 5000


def test_clear_prompt_is_idempotent(stats):
    _insert_prompt(stats, _ts(1), "x" * 5000)
    assert stats.clear_oversized_prompts(2048) == 1
    assert stats.clear_oversized_prompts(2048) == 0


def test_clear_prompt_then_vacuum_reclaims(stats):
    for i in range(2000):
        _insert_prompt(stats, _ts(1), "x" * 3000, req_id=f"blob-{i}")
    size_before = stats.db_size_bytes()

    stats.clear_oversized_prompts(2048)

    # 批量 UPDATE 腾出的空间不进 freelist_count，阈值门会误判成「不用收」——
    # 这正是必须 force 的原因，顺带把这条行为固化下来。
    assert stats._conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
    assert stats.vacuum() is False
    assert stats.vacuum(force=True) is True

    assert stats.db_size_bytes() < size_before
    assert _count(stats) == 2000  # 行还在，只是字段空了


# ---------------------------------------------------------------------------
#  vacuum
# ---------------------------------------------------------------------------

def test_vacuum_skipped_when_db_is_compact(stats):
    _insert(stats, _ts(1))
    assert stats.vacuum() is False


def test_vacuum_after_massive_delete_shrinks_file(stats):
    # 造一批足够大的数据，删掉大部分后空闲页占比才会超过阈值
    for i in range(2000):
        _insert(stats, _ts(100), req_id=f"blob-{i}" * 40)
    size_before = stats.db_size_bytes()

    stats.cleanup_old_records(30)
    assert stats.vacuum() is True

    assert stats.db_size_bytes() < size_before
    assert _count(stats) == 0
    # VACUUM 之后库仍可正常写入
    _insert(stats, _ts(0), req_id="after-vacuum")
    assert _count(stats) == 1


def test_cleanup_on_closed_db_is_noop(tmp_path):
    s = UsageStats(str(tmp_path / "stats.db"))
    s.init_db()
    s.close()

    assert s.cleanup_old_records(30) == 0
    assert s.vacuum() is False
