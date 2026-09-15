"""
用量统计时间序列（query_daily_usage / query_stats 的 day 维度）测试。

覆盖：
- 缺口补齐：区间内没有请求的日子也要出现在 buckets 里
- 透视正确性：values 与 buckets 等长，求和等于该模型区间合计
- 指标切换、分桶粒度（day/week/month）
- 日期归一化：until="2026-09-15" 要能覆盖到当天 23:59
- 非法 metric / bucket 被拒
- 长尾合并为「其他」
- query_stats(group_by="day,model") 按时间升序
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

import usage_stats
from usage_stats import UsageRecord, UsageStats


@pytest.fixture
def stats(tmp_path):
    s = UsageStats(str(tmp_path / "stats.db"))
    s.init_db()
    yield s
    s.close()


def _insert(
    s: UsageStats,
    ts: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider: str = "p1",
    tenant_id: str = "t1",
    req_id: str = "r",
) -> None:
    """直接落库，以便指定 ts（record_usage 用的是 DEFAULT now）。"""
    s._conn.execute(
        """
        INSERT INTO request_log
            (ts, req_id, tenant_id, provider, model, endpoint,
             input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, 'messages', ?, ?)
        """,
        (ts, req_id, tenant_id, provider, model, input_tokens, output_tokens),
    )
    s._conn.commit()


@pytest.fixture
def seeded(stats):
    """三个模型，分布在 2026-09-01 / 09-03 / 09-04（09-02 故意留空）。"""
    _insert(stats, "2026-09-01T10:00:00", "alpha", input_tokens=100, output_tokens=10)
    _insert(stats, "2026-09-01T11:00:00", "beta", input_tokens=50, output_tokens=5)
    _insert(stats, "2026-09-03T09:00:00", "alpha", input_tokens=200, output_tokens=20)
    _insert(stats, "2026-09-03T09:30:00", "alpha", input_tokens=1, output_tokens=1)
    _insert(stats, "2026-09-04T23:30:00", "beta", input_tokens=7, output_tokens=3)
    return stats


# ---------------------------------------------------------------------------
#  缺口补齐
# ---------------------------------------------------------------------------

def test_fills_missing_days(seeded):
    res = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", metric="total_tokens"
    )
    assert res["buckets"] == [
        "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
    ]
    assert res["bucket_filled"] is True


def test_missing_day_is_zero_not_absent(seeded):
    res = seeded.query_daily_usage(since="2026-09-01", until="2026-09-04")
    by_key = {s["key"]: s["values"] for s in res["series"]}
    # 09-02 无人使用，必须是 0 而不是被跳过
    assert by_key["alpha"][1] == 0
    assert by_key["beta"][1] == 0


def test_values_align_with_buckets(seeded):
    res = seeded.query_daily_usage(since="2026-09-01", until="2026-09-04")
    n = len(res["buckets"])
    for s in res["series"]:
        assert len(s["values"]) == n, s["key"]
        # 区间合计应等于 series.total（区间恰好覆盖全部数据）
        assert sum(s["values"]) == s["total"]


def test_series_sorted_by_total_desc(seeded):
    res = seeded.query_daily_usage(since="2026-09-01", until="2026-09-04")
    totals = [s["total"] for s in res["series"]]
    assert totals == sorted(totals, reverse=True)
    # alpha: (100+10) + (200+20) + (1+1) = 332
    assert res["series"][0]["key"] == "alpha"
    assert res["series"][0]["total"] == 332


# ---------------------------------------------------------------------------
#  指标切换
# ---------------------------------------------------------------------------

def test_metric_switches_values(seeded):
    total = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", metric="total_tokens"
    )
    count = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", metric="request_count"
    )
    out = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", metric="output_tokens"
    )
    assert total["totals"]["request_count"] == 5
    assert sum(s["total"] for s in count["series"]) == 5
    assert total["series"][0]["total"] == 332
    assert out["series"][0]["total"] == 31  # alpha: 10 + 20 + 1


def test_invalid_metric_rejected(seeded):
    with pytest.raises(ValueError, match="未知指标"):
        seeded.query_daily_usage(metric="rm -rf")


def test_invalid_bucket_rejected(seeded):
    with pytest.raises(ValueError, match="未知分桶"):
        seeded.query_daily_usage(bucket="hour'); DROP TABLE request_log;--")


# ---------------------------------------------------------------------------
#  日期归一化
# ---------------------------------------------------------------------------

def test_bare_until_covers_whole_day(seeded):
    """until="2026-09-04" 必须覆盖 23:30 那条，否则会静默漏掉当天数据。"""
    res = seeded.query_daily_usage(since="2026-09-04", until="2026-09-04")
    assert res["buckets"] == ["2026-09-04"]
    assert res["totals"]["request_count"] == 1
    assert res["totals"]["total_tokens"] == 10  # beta 7 + 3


def test_bare_since_starts_at_midnight(seeded):
    res = seeded.query_daily_usage(since="2026-09-03", until="2026-09-03")
    assert res["totals"]["request_count"] == 2  # 09-03 的两条 alpha
    assert res["buckets"] == ["2026-09-03"]


def test_open_ended_range_uses_data_bounds(seeded):
    res = seeded.query_daily_usage()
    assert res["buckets"][0] == "2026-09-01"
    assert res["buckets"][-1] == "2026-09-04"
    assert res["period"]["since"].startswith("2026-09-01")


def test_empty_range_returns_empty(seeded):
    res = seeded.query_daily_usage(
        since="2030-01-01", until="2030-01-05"
    )
    assert res["buckets"] == []
    assert res["series"] == []
    assert res["totals"]["request_count"] == 0


# ---------------------------------------------------------------------------
#  分桶粒度
# ---------------------------------------------------------------------------

def test_month_bucket(seeded):
    res = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-30", bucket="month"
    )
    assert res["buckets"] == ["2026-09"]
    assert res["series"][0]["values"] == [332]


def test_week_bucket_labels_and_length(seeded):
    res = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", bucket="week"
    )
    # 09-01 ~ 09-04 全在同一周（2026-09-01 是周二，%W -> W35）
    assert len(res["buckets"]) == 1
    assert res["buckets"][0].startswith("2026-W")
    for s in res["series"]:
        assert len(s["values"]) == len(res["buckets"])


# ---------------------------------------------------------------------------
#  长尾合并
# ---------------------------------------------------------------------------

def test_tail_folded_into_other(stats):
    for i in range(5):
        # 主模型：量最大
        _insert(stats, "2026-09-01T10:00:00", "big", input_tokens=1000, output_tokens=0)
        # 长尾：每个只出现一次
        _insert(
            stats, "2026-09-01T10:00:00", f"tiny-{i}",
            input_tokens=10 + i, output_tokens=0, req_id=f"r{i}",
        )
    res = stats.query_daily_usage(
        since="2026-09-01", until="2026-09-01", max_series=2
    )
    keys = [s["key"] for s in res["series"]]
    # max_series=2 -> 留 big(5000) 与 tiny-4(14)，其余 tiny-0..3 折叠
    assert keys == ["big", "tiny-4", "其他"]
    assert res["truncated_series"] == 4
    other = next(s for s in res["series"] if s["key"] == "其他")
    assert other["total"] == 10 + 11 + 12 + 13
    # 折叠不丢数：合计仍然守恒
    assert sum(s["total"] for s in res["series"]) == 1000 * 5 + 10 + 11 + 12 + 13 + 14


# ---------------------------------------------------------------------------
#  query_stats 的 day 维度
# ---------------------------------------------------------------------------

def test_query_stats_day_dimension_is_chronological(seeded):
    res = seeded.query_stats(group_by="day,model")
    days = [row["day"] for row in res["stats"]]
    assert days == sorted(days), "时间序列必须按时间升序，不能按请求数降序"
    assert days[0] == "2026-09-01"
    # 每天内部仍按请求数降序：09-03 有两条 alpha、零条 beta
    assert res["stats"][0]["day"] == "2026-09-01"


def test_query_stats_day_totals(seeded):
    res = seeded.query_stats(group_by="day")
    by_day = {row["day"]: row for row in res["stats"]}
    assert by_day["2026-09-01"]["request_count"] == 2
    assert by_day["2026-09-01"]["total_tokens"] == 165  # 110 + 55
    assert by_day["2026-09-03"]["request_count"] == 2
    assert by_day["2026-09-03"]["total_tokens"] == 222


def test_query_stats_unknown_dimension_falls_back(seeded, caplog):
    """历史行为：未知维度回退成按 model 分组，但必须留下日志。"""
    with caplog.at_level("WARNING", logger="usage_stats"):
        res = seeded.query_stats(group_by="nonsense")
    assert res["stats"][0]["model"] == "alpha"
    assert any("未知分组维度" in r.message for r in caplog.records)


def test_query_stats_tenant_alias_unchanged(seeded):
    """回归：tenant 别名要仍然映射到 tenant_id。"""
    res = seeded.query_stats(group_by="tenant")
    assert res["stats"][0]["tenant_id"] == "t1"


def test_query_stats_bare_date_range(seeded):
    """query_stats 也应享受日期归一化。"""
    res = seeded.query_stats(group_by="model", since="2026-09-04", until="2026-09-04")
    assert len(res["stats"]) == 1
    assert res["stats"][0]["model"] == "beta"


def test_not_initialized_returns_empty():
    s = UsageStats(":memory:")  # 未调用 init_db
    assert s.query_daily_usage()["series"] == []
    assert s.query_stats()["stats"] == []


# ---------------------------------------------------------------------------
#  全时段模型排名（前端配色槽位的依据）
# ---------------------------------------------------------------------------

def test_model_rank_is_all_time_not_range_filtered(seeded):
    """排名必须与筛选条件无关，否则换个时间范围配色就会重排。"""
    full = seeded.model_rank()
    assert full[0] == "alpha"          # alpha 合计 332 > beta 65

    narrow = seeded.query_daily_usage(since="2026-09-04", until="2026-09-04")
    # 该区间里只有 beta 有数据，但排名仍是全时段的，alpha 依然排第一
    assert [s["key"] for s in narrow["series"]] == ["beta"]
    assert narrow["model_rank"] == full


def test_model_rank_present_in_empty_response(seeded):
    res = seeded.query_daily_usage(since="2030-01-01", until="2030-01-05")
    assert res["series"] == []
    assert res["model_rank"] == ["alpha", "beta"]


def test_model_rank_is_cached(seeded):
    assert seeded.model_rank() == ["alpha", "beta"]
    # 缓存期间新增数据不改变排名（TTL 到期后才刷新）
    _insert(seeded, "2026-09-05T00:00:00", "gamma", input_tokens=99999, output_tokens=0)
    assert "gamma" not in seeded.model_rank()
    seeded._rank_cache = None          # 模拟 TTL 到期
    assert seeded.model_rank()[0] == "gamma"


def test_model_rank_without_connection():
    assert UsageStats(":memory:").model_rank() == []


# ---------------------------------------------------------------------------
#  过滤器
# ---------------------------------------------------------------------------

def test_provider_and_model_filters(seeded):
    res = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", model="alpha"
    )
    assert [s["key"] for s in res["series"]] == ["alpha"]
    assert res["totals"]["request_count"] == 3

    res = seeded.query_daily_usage(
        since="2026-09-01", until="2026-09-04", provider="nope"
    )
    assert res["series"] == []


def test_max_filled_buckets_guard(stats, monkeypatch):
    """区间过宽时放弃缺口补齐，退回实际出现的桶。"""
    monkeypatch.setattr(usage_stats, "_MAX_FILLED_BUCKETS", 3)
    for day in ("2026-01-01", "2026-12-31"):
        _insert(stats, f"{day}T10:00:00", "alpha", input_tokens=1, output_tokens=0)
    res = stats.query_daily_usage(since="2026-01-01", until="2026-12-31")
    assert res["bucket_filled"] is False
    assert res["buckets"] == ["2026-01-01", "2026-12-31"]
    # 退化成稀疏桶后，透视仍然是对的
    assert sum(res["series"][0]["values"]) == 2


def test_record_usage_still_works(stats):
    """回归：新增代码不能弄坏写入路径。

    record_usage 现在是「入队即返回」，落库由后台 writer 线程攒批完成，
    所以要 flush() 之后才能查到。这条断言顺带锁住了这个语义。
    """
    stats.record_usage(
        UsageRecord(
            req_id="x", tenant_id="t1", provider="p1", model="alpha",
            endpoint="messages", input_tokens=3, output_tokens=4,
        )
    )
    stats.flush()
    res = stats.query_stats(group_by="model")
    assert res["stats"][0]["model"] == "alpha"
    assert res["stats"][0]["total_tokens"] == 7


# ---------------------------------------------------------------------------
#  厂商列表与厂商过滤
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_provider(stats):
    """两个厂商，且有一个模型跨厂商（真实库里 deepseek-flash 就是这样）。

    p_big  : shared(600) + only-big(50)
    p_small: shared(10)
    合计   : shared 610, only-big 50
    """
    _insert(stats, "2026-09-01T10:00:00", "shared", input_tokens=600, output_tokens=0,
            provider="p_big", req_id="a1")
    _insert(stats, "2026-09-01T10:00:00", "shared", input_tokens=10, output_tokens=0,
            provider="p_small", req_id="a2")
    _insert(stats, "2026-09-01T11:00:00", "only-big", input_tokens=50, output_tokens=0,
            provider="p_big", req_id="a3")
    return stats


def test_provider_rank_all_time_desc(multi_provider):
    assert multi_provider.provider_rank() == ["p_big", "p_small"]


def test_provider_rank_present_in_response(multi_provider):
    res = multi_provider.query_daily_usage()
    assert res["provider_rank"] == ["p_big", "p_small"]


def test_provider_rank_ignores_provider_filter(multi_provider):
    """下拉框的选项必须与当前筛选无关，否则选中一个厂商后其它选项就消失了。"""
    res = multi_provider.query_daily_usage(provider="p_small")
    assert res["provider_rank"] == ["p_big", "p_small"]
    assert [s["key"] for s in res["series"]] == ["shared"]


def test_provider_rank_ignores_range_filter(multi_provider):
    """换时间范围不能让下拉框选项忽增忽减。"""
    res = multi_provider.query_daily_usage(since="2030-01-01", until="2030-01-05")
    assert res["series"] == []
    assert res["provider_rank"] == ["p_big", "p_small"]


def test_provider_rank_is_cached(multi_provider):
    assert multi_provider.provider_rank() == ["p_big", "p_small"]
    _insert(multi_provider, "2026-09-02T00:00:00", "x", input_tokens=99999,
            output_tokens=0, provider="p_huge", req_id="z")
    assert "p_huge" not in multi_provider.provider_rank()
    multi_provider._provider_cache = None       # 模拟 TTL 到期
    assert multi_provider.provider_rank()[0] == "p_huge"


def test_provider_rank_cache_is_independent_of_model_rank(multi_provider):
    """两个缓存互不覆盖：取模型排名不能顺手把厂商缓存的 TTL 也刷新掉。

    如果两种排名共用一个缓存槽位，下面这步 model_rank() 就会重新查库，
    于是 provider_rank() 会看到刚插入的 p_huge —— 缓存就白做了。
    """
    assert multi_provider.provider_rank() == ["p_big", "p_small"]
    _insert(multi_provider, "2026-09-02T00:00:00", "x", input_tokens=99999,
            output_tokens=0, provider="p_huge", req_id="z")
    assert multi_provider.model_rank()[0] == "x"        # 这次查库，且改变了模型排名
    assert multi_provider.provider_rank() == ["p_big", "p_small"]   # 厂商缓存仍在有效期内


def test_provider_rank_without_connection():
    assert UsageStats(":memory:").provider_rank() == []


def test_provider_filter_sums_only_that_provider(multi_provider):
    big = multi_provider.query_daily_usage(provider="p_big")
    assert big["totals"]["total_tokens"] == 650      # shared 600 + only-big 50
    assert big["totals"]["request_count"] == 2

    small = multi_provider.query_daily_usage(provider="p_small")
    assert small["totals"]["total_tokens"] == 10
    assert small["totals"]["request_count"] == 1

    # 两家之和 == 不过滤的总计，过滤不丢数也不重数
    assert big["totals"]["total_tokens"] + small["totals"]["total_tokens"] == 660


def test_model_rank_stays_global_under_provider_filter(multi_provider):
    """跨厂商的模型必须保持同一个配色槽位，所以 model_rank 不能跟着 provider 变。"""
    full = multi_provider.model_rank()
    assert full == ["shared", "only-big"]

    res = multi_provider.query_daily_usage(provider="p_small")
    assert res["model_rank"] == full
    # 该厂商下只有 shared，但排名里 only-big 仍在 —— 颜色不因过滤而重排
    assert [s["key"] for s in res["series"]] == ["shared"]


def test_query_stats_provider_filter(multi_provider):
    res = multi_provider.query_stats(group_by="model", provider="p_small")
    assert [r["model"] for r in res["stats"]] == ["shared"]
    assert res["stats"][0]["total_tokens"] == 10
