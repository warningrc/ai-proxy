#!/bin/bash
# ai-proxy 后台管理
# 用法: ./run.sh [start|stop|restart|status|logs]

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
PID_FILE="$DIR/logs/ai-proxy.pid"

# worker 进程数。每个 worker 是独立进程、各自持有连接池，靠 CPU 核数铺开；
# 可用 AI_PROXY_WORKERS 覆盖。注意 worker 之间不共享内存，配置热重载靠
# config_watcher 轮询 config.toml 广播（见 config_watcher.py）。
WORKERS="${AI_PROXY_WORKERS:-4}"

mkdir -p "$LOG_DIR"

stop_tree() {
    # 优雅停止：先给整个进程组发 TERM，让 uvicorn master 有机会收掉它 fork 的
    # N 个 worker，并让各 worker 走完 lifespan 收尾（刷统计队列、关连接池）。
    #
    # 不能直接 `kill -9 master`：worker 是 master 的孤儿进程，SIGKILL 没法转发，
    # master 一死 N 个 worker 会继续占着 5432 端口，下次 start 就 "address already
    # in use"，而且统计队列里没落库的数据直接丢。
    local pid="$1"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    # 15 秒还不退（多半是某个 worker 卡在上游长连接上），再强杀进程组
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    sleep 1
}

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "已在运行 (PID: $(cat "$PID_FILE"))"
            exit 0
        fi
        cd "$DIR"
        # 用 `uvicorn main:app` 启动（而非 `python main.py`），保证 main 以模块名
        # "main" 导入，避免 admin_api 里 `import main` 产生第二个模块实例。
        #
        # set -m：让服务起在独立进程组里，stop 才能按组回收 master + 全部 worker，
        # 而不会误伤脚本自己（不加的话后台任务和脚本同组）。
        set -m
        nohup ./.venv/bin/python -m uvicorn main:app \
            --host 0.0.0.0 --port 5432 \
            --workers "$WORKERS" \
            --loop uvloop --http httptools \
            >> "$LOG_DIR/app.log" 2>&1 &
        set +m
        echo $! > "$PID_FILE"
        echo "已启动 (PID: $!, workers: $WORKERS), 日志: $LOG_DIR/app.log"
        ;;
    stop)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            stop_tree "$(cat "$PID_FILE")"
            rm -f "$PID_FILE"
            echo "已停止"
        else
            echo "未在运行"
            rm -f "$PID_FILE"
        fi
        ;;
    restart)
        "$0" stop
        sleep 1
        "$0" start
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "运行中 (PID: $(cat "$PID_FILE"))"
        else
            echo "未运行"
            rm -f "$PID_FILE"
        fi
        ;;
    logs)
        tail -f "$LOG_DIR/app.log"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        ;;
esac
