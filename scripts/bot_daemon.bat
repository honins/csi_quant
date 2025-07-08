@echo off
:: 增强版交易机器人守护进程启动脚本 (Windows版本)

setlocal enabledelayedexpansion

:: 脚本配置
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "BOT_NAME=enhanced_trading_bot"
set "PID_FILE=%PROJECT_ROOT%\results\daily_trading\%BOT_NAME%.pid"
set "LOG_FILE=%PROJECT_ROOT%\logs\daemon.log"
set "VENV_PATH=%PROJECT_ROOT%\venv"

:: 颜色定义 (Windows 10+)
set "COLOR_RESET="
set "COLOR_GREEN=echo [92m"
set "COLOR_YELLOW=echo [93m"
set "COLOR_RED=echo [91m"
set "COLOR_BLUE=echo [94m"

:: 打印消息函数
:print_status
echo [%date% %time%] %~1
goto :eof

:print_success
echo [%date% %time%] ✅ %~1
goto :eof

:print_warning
echo [%date% %time%] ⚠️  %~1
goto :eof

:print_error
echo [%date% %time%] ❌ %~1
goto :eof

:: 检查虚拟环境
:check_venv
if exist "%VENV_PATH%" (
    call :print_status "检测到虚拟环境: %VENV_PATH%"
    if exist "%VENV_PATH%\Scripts\activate.bat" (
        call "%VENV_PATH%\Scripts\activate.bat"
        call :print_success "虚拟环境已激活"
    ) else (
        call :print_warning "虚拟环境目录存在但无法激活"
    )
) else (
    call :print_warning "未找到虚拟环境，使用系统Python"
)
goto :eof

:: 检查依赖
:check_dependencies
call :print_status "检查Python依赖..."

:: 检查必需的包
python -c "import psutil" 2>nul
if errorlevel 1 set "MISSING_DEPS=psutil %MISSING_DEPS%"

python -c "import git" 2>nul
if errorlevel 1 set "MISSING_DEPS=GitPython %MISSING_DEPS%"

python -c "import schedule" 2>nul
if errorlevel 1 set "MISSING_DEPS=schedule %MISSING_DEPS%"

python -c "import pandas" 2>nul
if errorlevel 1 set "MISSING_DEPS=pandas %MISSING_DEPS%"

python -c "import numpy" 2>nul
if errorlevel 1 set "MISSING_DEPS=numpy %MISSING_DEPS%"

if defined MISSING_DEPS (
    call :print_error "缺少依赖包: %MISSING_DEPS%"
    call :print_status "正在安装缺少的依赖..."
    
    if exist "%PROJECT_ROOT%\requirements.txt" (
        pip install -r "%PROJECT_ROOT%\requirements.txt"
    ) else (
        pip install %MISSING_DEPS%
    )
    
    if errorlevel 1 (
        call :print_error "依赖安装失败"
        exit /b 1
    ) else (
        call :print_success "依赖安装完成"
    )
) else (
    call :print_success "所有依赖已满足"
)
goto :eof

:: 检查是否已运行
:is_running
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /FI "PID eq !PID!" | find "!PID!" >nul 2>&1
    if errorlevel 1 (
        :: PID文件存在但进程不存在，删除陈旧的PID文件
        del "%PID_FILE%" 2>nul
        exit /b 1
    ) else (
        exit /b 0
    )
) else (
    exit /b 1
)

:: 启动守护进程
:start_daemon
call :print_status "启动增强版交易机器人守护进程..."

call :is_running
if not errorlevel 1 (
    set /p PID=<"%PID_FILE%"
    call :print_warning "守护进程已在运行 (PID: !PID!)"
    exit /b 0
)

:: 创建必要的目录
if not exist "%PROJECT_ROOT%\results\daily_trading" mkdir "%PROJECT_ROOT%\results\daily_trading"
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

:: 切换到项目根目录
cd /d "%PROJECT_ROOT%"

:: 启动守护进程 (Windows下使用start启动后台进程)
start /B python run.py bot -m daemon --daemon > "%LOG_FILE%" 2>&1

:: 等待启动确认
timeout /t 3 /nobreak >nul

:: 检查进程是否启动成功
python -c "
import time, os, psutil
time.sleep(1)
for proc in psutil.process_iter(['pid', 'cmdline']):
    try:
        if 'python' in ' '.join(proc.info['cmdline']).lower() and 'daemon' in ' '.join(proc.info['cmdline']):
            with open(r'%PID_FILE%', 'w') as f:
                f.write(str(proc.info['pid']))
            print('守护进程启动成功 (PID: {})'.format(proc.info['pid']))
            exit(0)
    except:
        pass
print('守护进程启动失败')
exit(1)
"

if errorlevel 1 (
    call :print_error "守护进程启动失败"
    exit /b 1
) else (
    call :print_success "守护进程启动完成"
    call :print_status "日志文件: %LOG_FILE%"
    call :print_status "PID文件: %PID_FILE%"
    
    :: 显示定时任务
    call :print_status "定时任务配置:"
    call :print_status "  - 每天15:05 自动数据拉取和提交"
    call :print_status "  - 每天09:30 日常交易流程"
    call :print_status "  - 每天01:00 系统健康检查"
    call :print_status "  - 每周日02:00 数据备份"
    call :print_status "  - 每小时性能指标收集"
)
goto :eof

:: 停止守护进程
:stop_daemon
call :print_status "停止增强版交易机器人守护进程..."

call :is_running
if errorlevel 1 (
    call :print_warning "守护进程未运行"
    exit /b 0
)

set /p PID=<"%PID_FILE%"

:: 尝试正常终止进程
taskkill /PID %PID% >nul 2>&1

:: 等待进程退出
set /a COUNT=0
:wait_stop
set /a COUNT+=1
if %COUNT% gtr 30 goto force_kill

tasklist /FI "PID eq %PID%" | find "%PID%" >nul 2>&1
if errorlevel 1 (
    call :print_success "守护进程已停止 (PID: %PID%)"
    del "%PID_FILE%" 2>nul
    exit /b 0
)

timeout /t 1 /nobreak >nul
goto wait_stop

:force_kill
call :print_warning "正在强制终止守护进程..."
taskkill /F /PID %PID% >nul 2>&1
del "%PID_FILE%" 2>nul
call :print_success "守护进程已强制终止"
goto :eof

:: 重启守护进程
:restart_daemon
call :print_status "重启增强版交易机器人守护进程..."
call :stop_daemon
timeout /t 2 /nobreak >nul
call :start_daemon
goto :eof

:: 检查状态
:check_status
call :print_status "检查增强版交易机器人守护进程状态..."

call :is_running
if not errorlevel 1 (
    set /p PID=<"%PID_FILE%"
    call :print_success "守护进程正在运行 (PID: !PID!)"
    
    :: 显示进程信息
    call :print_status "进程信息:"
    tasklist /FI "PID eq !PID!" /FO table
    
    :: 显示最近的日志
    if exist "%LOG_FILE%" (
        call :print_status "最近日志 (最后10行):"
        powershell -Command "Get-Content '%LOG_FILE%' | Select-Object -Last 10"
    )
    
    :: 调用机器人状态检查
    call :print_status "详细状态信息:"
    cd /d "%PROJECT_ROOT%"
    python run.py bot -m status 2>nul
    
) else (
    call :print_warning "守护进程未运行"
    
    :: 检查是否有陈旧的日志
    if exist "%LOG_FILE%" (
        call :print_status "最后运行日志:"
        powershell -Command "Get-Content '%LOG_FILE%' | Select-Object -Last 5"
    )
)
goto :eof

:: 查看日志
:view_logs
if exist "%LOG_FILE%" (
    call :print_status "实时查看守护进程日志 (Ctrl+C退出):"
    powershell -Command "Get-Content '%LOG_FILE%' -Wait"
) else (
    call :print_warning "日志文件不存在: %LOG_FILE%"
)
goto :eof

:: 执行健康检查
:health_check
call :print_status "执行系统健康检查..."
cd /d "%PROJECT_ROOT%"
python run.py bot -m health
goto :eof

:: 执行备份
:backup_data
call :print_status "执行手动数据备份..."
cd /d "%PROJECT_ROOT%"
python run.py bot -m backup
goto :eof

:: 显示帮助
:show_help
echo 增强版交易机器人守护进程管理脚本 (Windows版本)
echo.
echo 用法: %~nx0 {start^|stop^|restart^|status^|logs^|health^|backup^|help}
echo.
echo 命令说明:
echo   start   - 启动守护进程
echo   stop    - 停止守护进程
echo   restart - 重启守护进程
echo   status  - 检查运行状态
echo   logs    - 实时查看日志
echo   health  - 执行健康检查
echo   backup  - 执行数据备份
echo   help    - 显示此帮助信息
echo.
echo 示例:
echo   %~nx0 start         # 启动守护进程
echo   %~nx0 status        # 查看运行状态
echo   %~nx0 logs          # 实时查看日志
echo.
echo 配置文件: %PROJECT_ROOT%\config\config_core.yaml
echo 日志文件: %LOG_FILE%
echo PID文件: %PID_FILE%
goto :eof

:: 主逻辑
:main
if "%~1"=="start" (
    call :check_venv
    call :check_dependencies
    call :start_daemon
) else if "%~1"=="stop" (
    call :stop_daemon
) else if "%~1"=="restart" (
    call :check_venv
    call :check_dependencies
    call :restart_daemon
) else if "%~1"=="status" (
    call :check_status
) else if "%~1"=="logs" (
    call :view_logs
) else if "%~1"=="health" (
    call :check_venv
    call :health_check
) else if "%~1"=="backup" (
    call :check_venv
    call :backup_data
) else if "%~1"=="help" (
    call :show_help
) else if "%~1"=="--help" (
    call :show_help
) else if "%~1"=="-h" (
    call :show_help
) else (
    call :print_error "无效命令: %~1"
    echo.
    call :show_help
    exit /b 1
)
goto :eof

:: 确保脚本在项目根目录执行
if not exist "%PROJECT_ROOT%\run.py" (
    call :print_error "未找到run.py文件，请确保脚本在正确的项目目录中运行"
    exit /b 1
)

:: 执行主逻辑
call :main %* 