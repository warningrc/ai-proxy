"""
一次性维护脚本：抹掉库中历史遗留的超长 prompt，并回收磁盘空间。

背景
----
prompt 字段只用于排障。在 stats_prompt_max_chars 上线之前，这里存的是请求体
全文（实测平均 269KB/行、最大 3.5MB），一度占 stats.db 的 89%。这些行仍在保留
期内，按天数是清不掉的，只能就地抹掉字段内容。

判定标准是「长度 > stats_prompt_max_chars」而不是「写入时间」：凡是超过当前上限
的行，都是现行策略下不会再产生的数据；恰好合规的历史行不必动。

用法
----
    ./.venv/bin/python clear_prompt_history.py --dry-run   # 先看要清多少
    ./.venv/bin/python clear_prompt_history.py             # 真清（不可撤销）

注意：UPDATE 和 DELETE 一样只把页标成空闲，文件不会自己变小，所以结尾会 VACUUM。
服务运行中也能跑，但 VACUUM 期间会独占写锁数秒，其它写入排队等待。
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import settings
from usage_stats import UsageStats

logger = logging.getLogger("clear_prompt_history")


def _stats(usage: UsageStats, max_chars: int) -> tuple[int, int, int]:
    """返回 (总行数, 超长行数, 超长 prompt 合计字节数)。"""
    row = usage._conn.execute(
        "SELECT COUNT(*),"
        "       SUM(prompt IS NOT NULL AND LENGTH(prompt) > ?),"
        "       SUM(CASE WHEN prompt IS NOT NULL AND LENGTH(prompt) > ?"
        "                THEN LENGTH(CAST(prompt AS BLOB)) ELSE 0 END)"
        "  FROM request_log",
        (max_chars, max_chars),
    ).fetchone()
    return row[0] or 0, row[1] or 0, row[2] or 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=settings.STATS_DB,
        help="stats.db 路径（默认取 config.toml 的 stats_db）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=settings.STATS_PROMPT_MAX_CHARS,
        help="超过这个长度即视为遗留数据（默认取 config.toml 的 stats_prompt_max_chars）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只统计不修改",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="单事务处理的行数（默认 200）。线上库别调大：单批全程占写锁，"
             "而这些行本身很大，批次过大会挡住服务正在进行的用量写入。",
    )
    args = parser.parse_args()

    if args.max_chars <= 0:
        print(
            f"stats_prompt_max_chars = {args.max_chars}（<=0 表示不存 prompt），"
            "这种情况请用 DELETE 清理整行，本脚本不适用。",
            file=sys.stderr,
        )
        return 2

    usage = UsageStats(args.db)
    usage.init_db()  # 顺带施加 WAL / busy_timeout 等 pragma
    try:
        total, oversized, payload = _stats(usage, args.max_chars)
        size_before = usage.db_size_bytes()

        print(f"库文件      : {args.db}  ({size_before / 1e6:,.1f} MB)")
        print(f"总行数      : {total:,}")
        print(f"超长 prompt : {oversized:,} 行（上限 {args.max_chars} 字符）")
        print(f"可回收      : {payload / 1e9:,.2f} GB")

        if not oversized:
            print("没有需要清理的数据。")
            return 0
        if args.dry_run:
            print("\n--dry-run：未做任何修改。")
            return 0

        cleared = usage.clear_oversized_prompts(args.max_chars, args.batch_size)
        print(f"\n已置空 {cleared:,} 行的 prompt")

        # 必须 force：批量 UPDATE 腾出的空间不进 freelist_count，按阈值判断会跳过
        vacuumed = usage.vacuum(force=True)
        print(f"VACUUM      : {'已执行' if vacuumed else '跳过'}")
        print(f"库文件      : {size_before / 1e6:,.1f} MB -> "
              f"{usage.db_size_bytes() / 1e6:,.1f} MB")
        return 0
    finally:
        usage.close()


if __name__ == "__main__":
    raise SystemExit(main())
