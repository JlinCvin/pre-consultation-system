@echo off
echo 正在重启服务...

:: 查找当前应用的 Gunicorn 进程
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq gunicorn.exe" /fo list ^| find "PID:"') do (
    set PID=%%a
)

if defined PID (
    echo 找到 Gunicorn 进程 PID: %PID%
    :: 尝试正常终止进程
    taskkill /PID %PID% /F
    :: 等待进程终止
    timeout /t 2 /nobreak
) else (
    echo 未找到运行中的 Gunicorn 进程
)

:: 启动新的服务
call start_prod.bat

echo 服务已重启
echo 可以通过 'type logs\access.log' 查看访问日志
echo 可以通过 'type logs\error.log' 查看错误日志 