@echo off
chcp 65001 >nul
title 日常交易机器人启动器

echo ================================================================
echo                   🚀 日常交易机器人启动器
echo ================================================================
echo.

cd /d "%~dp0\.."

REM 检查虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo ✅ 发现虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  未发现虚拟环境，使用系统Python
)

echo.
:start
echo 请选择运行模式:
echo [1] 单次执行 (立即执行一次交易流程)
echo [2] 定时执行 (每天9:30自动执行)
echo [3] 查看状态 (查看机器人运行状态)
echo [4] 手动训练 (执行模型训练)
echo [5] 手动预测 (执行预测)
echo [0] 退出
echo.

set /p choice="请输入选择 (0-5): "

if "%choice%"=="1" goto single_run
if "%choice%"=="2" goto scheduled_run
if "%choice%"=="3" goto status_check
if "%choice%"=="4" goto manual_training
if "%choice%"=="5" goto manual_prediction
if "%choice%"=="0" goto exit
goto invalid_choice

:single_run
echo.
echo 🚀 启动单次执行模式...
python run.py bot -m run
echo.
echo 按任意键继续...
pause >nul
goto start

:scheduled_run
echo.
echo ⏰ 启动定时执行模式...
echo 💡 提示: 程序将在每天9:30自动执行交易流程
echo 💡 按 Ctrl+C 可以停止定时任务
python run.py bot -m schedule
goto end

:status_check
echo.
echo 📊 查看机器人状态...
python run.py bot -m status
echo.
echo 按任意键继续...
pause >nul
goto start

:manual_training
echo.
echo 🤖 执行手动训练...
python run.py ai -m incremental
echo.
echo 按任意键继续...
pause >nul
goto start

:manual_prediction
echo.
echo 🔮 执行手动预测...
python run.py ai -m demo
echo.
echo 按任意键继续...
pause >nul
goto start

:invalid_choice
echo.
echo ❌ 无效选择，请重新输入
timeout /t 2 >nul
goto start

:end
echo.
echo ================================================================
echo 执行完成！按任意键退出...
pause >nul

:exit
echo.
echo 👋 再见！
timeout /t 1 >nul 