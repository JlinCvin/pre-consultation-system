#!/bin/bash

echo "正在重启服务..."

# 获取当前应用的 Gunicorn 主进程 PID
PID=$(ps aux | grep "gunicorn main:app" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "找到 Gunicorn 主进程 PID: $PID"
    # 优雅地终止主进程（这会同时终止所有工作进程）
    kill -TERM $PID
    # 等待进程终止
    sleep 2
    # 如果进程仍然存在，强制终止
    if ps -p $PID > /dev/null; then
        echo "进程未正常终止，强制终止..."
        kill -9 $PID
    fi
else
    echo "未找到运行中的 Gunicorn 进程"
fi

# 启动新的服务
./start_prod.sh

echo "服务已重启"
echo "可以通过 'tail -f logs/access.log' 查看访问日志"
echo "可以通过 'tail -f logs/error.log' 查看错误日志" 