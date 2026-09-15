"""
用量写入路径测试：prompt 截断、批量落库、队列满时丢数据不阻塞。

这条路径是「请求线程只入队、后台线程批量落库」，出问题的方式是静默的：
要么 prompt 没截断把库撑爆，要么队列满了把请求卡死，要么批量写把数据写丢。
"""

from __future__ import annotations

import threading
import time

import pytest

from usage_stats import UsageRecord, UsageStats


def _record(**kw) -> UsageRecord:
    base = dict(
        req_id="r1", tenant_id="t1", provider="p1", model="m",
        endpoint="messages", input_tokens=1, output_tokens=2,
    )
    base.update(kw)
    return UsageRecord(**base)


@pytest.fixture
def make_stats(tmp_path):
    created = []

    def _make(**kwargs):
        kwargs.setdefault("flush_interval", 0.05)
        s = UsageStats(str(tmp_path / "stats.db"), **kwargs)
        s.init_db()
        created.append(s)
        return s

    yield _make
    for s in created:
        s.close()


def _stored_prompt(s, req_id):
    row = s._read_conn.execute(
        "SELECT prompt FROM request_log WHERE req_id = ?", (req_id,)
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
#  prompt 截断
# ---------------------------------------------------------------------------

def test_prompt_is_truncated_to_limit(make_stats):
    s = make_stats(prompt_max_chars=100)
    s.record_usage(_record(req_id="long", prompt="啊" * 5000))
    s.flush()
    assert len(_stored_prompt(s, "long")) == 100


def test_short_prompt_is_stored_verbatim(make_stats):
    s = make_stats(prompt_max_chars=100)
    s.record_usage(_record(req_id="short", prompt="hello"))
    s.flush()
    assert _stored_prompt(s, "short") == "hello"


def test_zero_limit_disables_prompt_storage(make_stats):
    """0 = 完全不存 prompt（不落截断后的残段，而是整列为 NULL）。"""
    s = make_stats(prompt_max_chars=0)
    s.record_usage(_record(req_id="none", prompt="x" * 1000))
    s.flush()
    assert _stored_prompt(s, "none") is None


def test_negative_limit_disables_prompt_storage(make_stats):
    s = make_stats(prompt_max_chars=-1)
    s.record_usage(_record(req_id="neg", prompt="x" * 1000))
    s.flush()
    assert _stored_prompt(s, "neg") is None


def test_none_prompt_stays_none(make_stats):
    s = make_stats(prompt_max_chars=100)
    s.record_usage(_record(req_id="nil", prompt=None))
    s.flush()
    assert _stored_prompt(s, "nil") is None


def test_long_prompt_does_not_bloat_db(make_stats):
    """截断的直接目的：不能让单条记录把库撑大。"""
    s = make_stats(prompt_max_chars=64)
    for i in range(50):
        s.record_usage(_record(req_id=f"r{i}", prompt="y" * 200_000))
    s.flush()
    # 50 条 × 64 字符，库不可能到 MB 级
    assert s.db_size_bytes() < 1_000_000


def test_reconfigure_applies_new_limit(make_stats):
    """热重载改了配置，后续记录要按新长度截断（老记录不动）。"""
    s = make_stats(prompt_max_chars=100)
    s.record_usage(_record(req_id="before", prompt="z" * 500))
    s.flush()
    s.reconfigure(prompt_max_chars=10, flush_interval=0.05, batch_size=500)
    s.record_usage(_record(req_id="after", prompt="z" * 500))
    s.flush()
    assert len(_stored_prompt(s, "before")) == 100
    assert len(_stored_prompt(s, "after")) == 10


# ---------------------------------------------------------------------------
#  批量落库
# ---------------------------------------------------------------------------

def test_record_usage_does_not_block(make_stats):
    """入队必须立即返回：这是把写入踢出请求路径的关键。"""
    s = make_stats(prompt_max_chars=0)
    t0 = time.perf_counter()
    for i in range(2000):
        s.record_usage(_record(req_id=f"r{i}"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"入队 2000 条花了 {elapsed:.3f}s，说明又同步落库了"


def test_all_records_reach_disk(make_stats):
    """批量写不能丢数据：队列排空后条数必须对得上。"""
    s = make_stats(prompt_max_chars=0, batch_size=50)
    for i in range(500):
        s.record_usage(_record(req_id=f"r{i}"))
    s.flush()
    n = s._read_conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
    assert n == 500


def test_batch_size_controls_commit_granularity(make_stats):
    """攒批的意义在于减少 commit：500 条只应触发个位数次提交。"""
    s = make_stats(prompt_max_chars=0, batch_size=500, flush_interval=60.0)
    commits = []
    real_conn = s._conn

    class _Counting:
        def __getattr__(self, name):
            return getattr(real_conn, name)

        def commit(self):
            commits.append(1)
            return real_conn.commit()

    s._conn = _Counting()
    for i in range(500):
        s.record_usage(_record(req_id=f"r{i}"))
    s.flush()
    assert commits, "没有任何提交，数据没落库"
    assert len(commits) <= 3, f"500 条记录触发了 {len(commits)} 次 commit"


def test_records_written_before_flush_interval(make_stats):
    """没攒够 batch_size 也要按时落库，不能一直压在内存里。"""
    s = make_stats(prompt_max_chars=0, batch_size=10_000, flush_interval=0.05)
    s.record_usage(_record(req_id="solo"))
    time.sleep(0.3)
    n = s._read_conn.execute(
        "SELECT COUNT(*) FROM request_log WHERE req_id='solo'"
    ).fetchone()[0]
    assert n == 1


def test_flush_is_idempotent(make_stats):
    s = make_stats(prompt_max_chars=0)
    s.record_usage(_record(req_id="a"))
    s.flush()
    s.flush()
    n = s._read_conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
    assert n == 1


def test_full_queue_drops_instead_of_blocking(make_stats, monkeypatch, caplog):
    """队列满时必须丢弃并告警，绝不能反压到请求线程。

    用 _MAX_QUEUE_SIZE 让构造出来的队列本身就是小的：直接在 init_db() 之后整个换掉
    self._queue 是不行的 —— writer 线程可能已经阻塞在旧队列的 get() 上，永远等不到
    新队列里的东西。
    """
    monkeypatch.setattr(UsageStats, "_MAX_QUEUE_SIZE", 5)
    s = make_stats(prompt_max_chars=0, batch_size=10_000, flush_interval=60.0)

    # 按住写锁，让 writer 卡在 _flush_batch 里不再消费，队列才会真的被灌满
    s._lock.acquire()
    try:
        for i in range(50):
            s.record_usage(_record(req_id=f"r{i}"))   # 不能卡住，也不能抛异常
        assert s._dropped > 0
    finally:
        s._lock.release()

    # 丢数据是静默故障，必须留日志，否则线上只会看到统计数字凭空变少
    assert "丢弃" in caplog.text


def test_flush_returns_without_waiting_for_a_full_batch(make_stats):
    """flush() 的耗时不能取决于 batch_size / flush_interval。

    队列被取空后如果还按 flush_interval 等下一批，flush() 就得等到超时才返回 ——
    那时候「等多久」跟数据落没落盘已经没关系了。
    """
    s = make_stats(prompt_max_chars=0, batch_size=10_000, flush_interval=60.0)
    s.record_usage(_record(req_id="a"))
    t0 = time.perf_counter()
    assert s.flush(timeout=5.0) is True
    assert time.perf_counter() - t0 < 1.0


def test_flush_returns_false_when_writer_is_stuck(make_stats):
    """writer 卡住时 flush() 要如实返回 False，不能假装写成功。"""
    s = make_stats(prompt_max_chars=0, flush_interval=60.0)
    s._lock.acquire()
    try:
        s.record_usage(_record(req_id="stuck"))
        assert s.flush(timeout=0.2) is False
    finally:
        s._lock.release()


def test_close_flushes_pending_records(make_stats):
    """优雅退出不能丢队列里还没落库的数据。"""
    s = make_stats(prompt_max_chars=0, batch_size=10_000, flush_interval=60.0)
    for i in range(20):
        s.record_usage(_record(req_id=f"r{i}"))
    s.close()
    check = UsageStats(str(s._db_path))
    check.init_db()
    try:
        n = check._read_conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
        assert n == 20
    finally:
        check.close()


def test_close_is_fast_when_queue_is_empty(make_stats):
    """close() 不能干等到 flush_interval 超时才返回（否则每个测试都要多等半秒）。"""
    s = make_stats(prompt_max_chars=0, flush_interval=5.0)
    t0 = time.perf_counter()
    s.close()
    assert time.perf_counter() - t0 < 1.0


# ---------------------------------------------------------------------------
#  单条坏记录不能拖垮整条写入路径
# ---------------------------------------------------------------------------

def _row_count(s, req_id):
    return s._read_conn.execute(
        "SELECT COUNT(*) FROM request_log WHERE req_id = ?", (req_id,)
    ).fetchone()[0]


def test_oversized_int_does_not_kill_the_writer(make_stats):
    """单条写不进去的记录不能让 writer 线程死掉。

    Python int 超出 SQLite 的 64 位整数范围时 sqlite3 抛 OverflowError，而它
    **不是** sqlite3.Error 的子类。以前这个异常会从 writer 线程里冒出去把线程
    带走，此后所有用量都只入队不落库，而 flush() 还是返回 True（它只等哨兵，
    哨兵也被那个已经死掉的线程处理了），等于静默地永久丢数据。
    契约：坏记录自己写失败、如实上报，writer 必须活着继续处理后面的记录。
    """
    s = make_stats(prompt_max_chars=0)
    s.record_usage(_record(req_id="before"))
    s.record_usage(_record(req_id="bad", input_tokens=2 ** 63))  # 超出 int64
    s.record_usage(_record(req_id="after"))

    # 有 1 条写失败，flush() 要如实返回 False
    assert s.flush() is False
    assert _row_count(s, "before") == 1
    assert _row_count(s, "after") == 1
    assert _row_count(s, "bad") == 0

    # 关键：线程还活着，后续记录照常落库
    assert s._writer_thread is not None and s._writer_thread.is_alive()
    assert s._write_failed == 1
    s.record_usage(_record(req_id="later"))
    assert s.flush() is True
    assert _row_count(s, "later") == 1


def test_bad_row_does_not_roll_back_the_whole_batch(make_stats):
    """整批失败后退回逐条写：同批里没问题的记录必须留下。"""
    s = make_stats(prompt_max_chars=0, batch_size=10_000, flush_interval=60.0)
    for i in range(5):
        s.record_usage(_record(req_id=f"good{i}"))
    s.record_usage(_record(req_id="bad", output_tokens=-(2 ** 63) - 1))
    for i in range(5):
        s.record_usage(_record(req_id=f"good{i + 5}"))

    assert s.flush() is False
    n = s._read_conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
    assert n == 10
    assert _row_count(s, "bad") == 0


def test_failed_batch_leaves_no_open_transaction(make_stats):
    """批量写失败后必须 rollback：留着没结束的事务会一直攥着 WAL 写锁，
    别的 worker 的写入会被拖到 busy_timeout 超时。"""
    s = make_stats(prompt_max_chars=0)
    s.record_usage(_record(req_id="bad", output_tokens=2 ** 70))
    assert s.flush() is False
    with s._lock:
        assert s._conn.in_transaction is False


# ---------------------------------------------------------------------------
#  关库：读连接必须和读路径拿同一把锁
# ---------------------------------------------------------------------------

def test_close_waits_for_the_read_lock(make_stats):
    """close() 关读连接必须持 _read_lock。

    所有查询路径都是在 _read_lock 里 execute 的，close() 不加锁就是「一边 execute
    一边 close」。sqlite3 是 C 扩展，这类竞态不是抛异常而是段错误，会带走整个
    worker 进程 —— 而且是偶发的，最难查。
    """
    s = make_stats(prompt_max_chars=0)
    s._read_lock.acquire()
    try:
        t = threading.Thread(target=s.close, daemon=True)
        t.start()
        t.join(1.0)
        assert t.is_alive(), "close() 没等 _read_lock，说明它没持锁就关了读连接"
        # 写连接已经关完了，说明它确实是卡在读连接这把锁上
        assert s._conn is None
    finally:
        s._read_lock.release()
    t.join(5.0)
    assert not t.is_alive()
    assert s._read_conn is None


def test_close_is_idempotent(make_stats):
    s = make_stats(prompt_max_chars=0)
    s.close()
    s.close()
    assert s._read_conn is None and s._conn is None


def test_queries_during_close_never_touch_a_closed_connection(make_stats):
    """读连接被关掉之后，在途查询要么拿到结果要么返回空，不能抛异常，
    更不能撞上「已关闭的连接」——那是进程级崩溃，不是可捕获的错误。"""
    s = make_stats(prompt_max_chars=0)
    s.record_usage(_record(req_id="x"))
    s.flush()

    errors: list = []
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            try:
                s.query_stats()
                s.query_daily_usage()
                s.model_rank()
                s.provider_rank()
            except Exception as e:  # noqa: BLE001 - 就是要抓住任何异常
                errors.append(repr(e))

    threads = [threading.Thread(target=_reader, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    s.close()
    time.sleep(0.05)
    stop.set()
    for t in threads:
        t.join(5.0)

    assert errors == []


# ---------------------------------------------------------------------------
#  数据库初始化
# ---------------------------------------------------------------------------

def test_wal_mode_enabled(make_stats):
    """WAL 是「读不阻塞写」的前提，必须开着。"""
    s = make_stats()
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_read_connection_is_read_only(make_stats):
    """统计查询走只读连接：写不动，也就不会跟写入抢写锁。"""
    s = make_stats()
    with pytest.raises(Exception):
        s._read_conn.execute("INSERT INTO request_log (req_id) VALUES ('x')")


def test_two_instances_can_share_one_db(tmp_path):
    """多 worker 下每个进程一个实例、同一个库文件，读写不能互相锁死。"""
    db = str(tmp_path / "shared.db")
    a = UsageStats(db, prompt_max_chars=0, flush_interval=0.05)
    b = UsageStats(db, prompt_max_chars=0, flush_interval=0.05)
    a.init_db()
    b.init_db()
    try:
        for i in range(100):
            a.record_usage(_record(req_id=f"a{i}"))
            b.record_usage(_record(req_id=f"b{i}"))
        a.flush()
        b.flush()
        n = a._read_conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
        assert n == 200
    finally:
        a.close()
        b.close()
