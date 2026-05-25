#!/bin/bash
# ai-proxy 后台管理
# 用法: ./run.sh [start|stop|restart|status|logs]

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
PID_FILE="$DIR/logs/ai-proxy.pid"

mkdir -p "$LOG_DIR"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "已在运行 (PID: $(cat "$PID_FILE"))"
            exit 0
        fi
        cd "$DIR"
        nohup ./.venv/bin/python main.py >> "$LOG_DIR/app.log" 2>&1 &
        echo $! > "$PID_FILE"
        echo "已启动 (PID: $!), 日志: $LOG_DIR/app.log"
        ;;
    stop)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            kill -9 "$(cat "$PID_FILE")"
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
