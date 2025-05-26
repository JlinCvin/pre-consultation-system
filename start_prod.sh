#!/bin/bash

# 设置环境变量
export ENVIRONMENT=production

# 创建日志目录
mkdir -p logs

# 使用 Gunicorn 启动应用（后台运行）
nohup gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --log-level warning \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --timeout 120 \
    --daemon

echo "服务已在后台启动"
echo "可以通过 'tail -f logs/access.log' 查看访问日志"
echo "可以通过 'tail -f logs/error.log' 查看错误日志" 