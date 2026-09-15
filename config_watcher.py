"""
config.toml 变更轮询：让每个 worker 都能感知到「别人」写的配置。

单进程时热重载由管理接口自己完成。但 ``uvicorn --workers N`` 下每个 worker 是
独立进程，只有收到 ``PUT /admin/api/config`` 的那个 worker 会 reload，其余
进程仍在用旧配置 —— 表现为「配置改了，但一部分请求还是老行为，刷新几次才变」。
这里让每个 worker 各起一个后台任务轮询 config.toml，内容一变就自己重载一次。

去重：管理接口那条路径写完盘会调 :func:`note_config_applied`，把新内容记为
基线，所以「自己刚应用的配置」不会被自己再应用一遍。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

logger = logging.getLogger("ai-proxy")

# 文件指纹 = 内容哈希。故意只算内容、不看 mtime
Fingerprint = str

# 轮询下限。配置是给人改的，比这更密没有意义，只会白读文件。
_MIN_INTERVAL = 0.2


def _fingerprint(path: Path) -> Optional[Fingerprint]:
    """算出文件指纹；不存在 / 读不到返回 None。

    只认内容、不认 mtime：mtime 会被 `touch`、会被编辑器「保存但内容没变」骗到，
    那些都会触发一次白重载（重建全部 HTTP 连接池）。config.toml 只有几 KB，
    每轮读一次的开销可以忽略，不值得为省这点 IO 去掺进 mtime。
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:16]


class ConfigWatcher:
    """轮询 config.toml 内容，发现变更就触发一次重载回调。"""

    def __init__(
        self,
        path: Path,
        interval: float,
        on_reload: Callable[[], Awaitable[None]],
    ) -> None:
        self._path = Path(path)
        self._interval = max(_MIN_INTERVAL, float(interval))
        self._on_reload = on_reload
        # 启动基线 = 进程启动时读到的配置（config.py import 时已经 load 过一遍）
        self._baseline = _fingerprint(self._path)
        self._task: Optional[asyncio.Task] = None
        self._busy = False

    @property
    def baseline(self) -> Optional[Fingerprint]:
        return self._baseline

    def note_applied(self) -> None:
        """把当前磁盘内容标记为「本进程已应用」，watcher 不会再重复应用它。"""
        self._baseline = _fingerprint(self._path)

    async def check_once(self) -> bool:
        """检查一次；触发了重载则返回 True。

        独立成方法是为了可测：单测直接调它，不用等 sleep。
        """
        if self._busy:
            return False
        current = _fingerprint(self._path)
        if current is None or current == self._baseline:
            return False

        self._busy = True
        try:
            logger.info(
                "config watcher | %s changed, reloading runtime", self._path
            )
            await self._on_reload()
        except (Exception, SystemExit):
            # 重载失败不能让后台任务死掉，也不能影响正在服务的请求：旧配置继续用。
            # SystemExit 要单独接住 —— config.reload() 对坏配置走 _die -> sys.exit，
            # 漏出去就直接把 worker 带走了（它继承 BaseException，except Exception 抓不到）。
            logger.exception("config watcher | reload failed, keeping old config")
        finally:
            self._busy = False

        # 不论成败都把基线推到最新：手改坏了配置不该每 2 秒重试一轮，
        # 那只会把日志刷爆（真正的原因在上一行的 traceback 里，改好文件自然会再触发）
        self._baseline = current
        return True

    async def run(self) -> None:
        logger.info(
            "config watcher | watching %s every %.1fs", self._path, self._interval
        )
        while True:
            await asyncio.sleep(self._interval)
            await self.check_once()

    def start(self) -> None:
        """在当前事件循环里起后台任务（重复调用无副作用）。"""
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
#  全局注册表
# ---------------------------------------------------------------------------
# 每个进程只有一个 watcher，但用注册表而不是单例：admin 侧只需 import 本模块
# 调一个函数，不必反向依赖 main（否则 main <-> admin_reloader 会成环）。

_watchers: List[ConfigWatcher] = []


def register_watcher(watcher: ConfigWatcher) -> None:
    _watchers.append(watcher)


def note_config_applied() -> None:
    """本进程刚把配置写盘并生效，通知所有 watcher 抬高基线，避免自己重载自己。"""
    for w in _watchers:
        w.note_applied()
