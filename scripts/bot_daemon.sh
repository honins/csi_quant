#!/bin/bash

# 指数交易机器人守护进程启动脚本

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BOT_NAME="enhanced_trading_bot"
PID_FILE="$PROJECT_ROOT/results/daily_trading/$BOT_NAME.pid"
LOG_FILE="$PROJECT_ROOT/logs/daemon.log"
PYTHON_CMD="python"
VENV_PATH="$PROJECT_ROOT/venv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✅ $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ❌ $1"
}

# 检查虚拟环境
check_venv() {
    if [ -d "$VENV_PATH" ]; then
        print_status "检测到虚拟环境: $VENV_PATH"
        if [ -f "$VENV_PATH/bin/activate" ]; then
            . "$VENV_PATH/bin/activate"
            print_success "虚拟环境已激活"
        elif [ -f "$VENV_PATH/Scripts/activate" ]; then
            . "$VENV_PATH/Scripts/activate"
            print_success "虚拟环境已激活 (Windows)"
        else
            print_warning "虚拟环境目录存在但无法激活"
        fi
    else
        print_warning "未找到虚拟环境，使用系统Python"
    fi
}

# 检查依赖
check_dependencies() {
    print_status "检查Python依赖..."
    
    # 使用兼容sh的方式检查依赖
    required_packages="psutil GitPython schedule pandas numpy"
    missing_packages=""
    
    for package in $required_packages; do
        if ! python -c "import $package" 2>/dev/null; then
            if [ -z "$missing_packages" ]; then
                missing_packages="$package"
            else
                missing_packages="$missing_packages $package"
            fi
        fi
    done
    
    if [ -n "$missing_packages" ]; then
        print_error "缺少依赖包: $missing_packages"
        print_status "正在安装缺少的依赖..."
        
        if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
            pip install -r "$PROJECT_ROOT/requirements.txt"
        else
            pip install $missing_packages
        fi
        
        if [ $? -eq 0 ]; then
            print_success "依赖安装完成"
        else
            print_error "依赖安装失败"
            exit 1
        fi
    else
        print_success "所有依赖已满足"
    fi
}

# 检查是否已运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            return 0
        else
            # PID文件存在但进程不存在，删除陈旧的PID文件
            rm -f "$PID_FILE"
            return 1
        fi
    else
        return 1
    fi
}

# 启动守护进程
start_daemon() {
    print_status "启动指数交易机器人守护进程..."
    
    if is_running; then
        local pid=$(cat "$PID_FILE")
        print_warning "守护进程已在运行 (PID: $pid)"
        return 0
    fi
    
    # 创建必要的目录
    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 启动守护进程
    nohup python run.py bot -m daemon --daemon > "$LOG_FILE" 2>&1 &
    local daemon_pid=$!
    
    # 等待启动确认
    sleep 3
    
    if ps -p $daemon_pid > /dev/null 2>&1; then
        print_success "守护进程启动成功 (PID: $daemon_pid)"
        print_status "日志文件: $LOG_FILE"
        print_status "PID文件: $PID_FILE"
        
        # 显示定时任务
        print_status "定时任务配置:"
        print_status "  - 每天15:05 自动数据拉取和提交"
        print_status "  - 每天09:30 日常交易流程"
        print_status "  - 每天01:00 系统健康检查"
        print_status "  - 每周日02:00 数据备份"
        print_status "  - 每小时性能指标收集"
        
        return 0
    else
        print_error "守护进程启动失败"
        return 1
    fi
}

# 停止守护进程
stop_daemon() {
    print_status "停止指数交易机器人守护进程..."
    
    if ! is_running; then
        print_warning "守护进程未运行"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    
    # 发送TERM信号
    kill -TERM $pid 2>/dev/null
    
    # 等待进程退出
    local count=0
    while [ $count -lt 30 ]; do
        if ! ps -p $pid > /dev/null 2>&1; then
            print_success "守护进程已停止 (PID: $pid)"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    # 强制终止
    print_warning "正在强制终止守护进程..."
    kill -KILL $pid 2>/dev/null
    rm -f "$PID_FILE"
    print_success "守护进程已强制终止"
}

# 重启守护进程
restart_daemon() {
    print_status "重启指数交易机器人守护进程..."
    stop_daemon
    sleep 2
    start_daemon
}

# 检查状态
check_status() {
    print_status "检查指数交易机器人守护进程状态..."
    
    if is_running; then
        local pid=$(cat "$PID_FILE")
        print_success "守护进程正在运行 (PID: $pid)"
        
        # 显示进程信息
        if command -v ps >/dev/null 2>&1; then
            print_status "进程信息:"
            ps -p $pid -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null | tail -n +2 | while read line; do
                print_status "  $line"
            done
        fi
        
        # 显示最近的日志
        if [ -f "$LOG_FILE" ]; then
            print_status "最近日志 (最后10行):"
            tail -n 10 "$LOG_FILE" | while read line; do
                print_status "  $line"
            done
        fi
        
        # 调用机器人状态检查
        print_status "详细状态信息:"
        cd "$PROJECT_ROOT"
        python run.py bot -m status 2>/dev/null
        
    else
        print_warning "守护进程未运行"
        
        # 检查是否有陈旧的日志
        if [ -f "$LOG_FILE" ]; then
            print_status "最后运行日志:"
            tail -n 5 "$LOG_FILE" | while read line; do
                print_status "  $line"
            done
        fi
    fi
}

# 查看日志
view_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_status "实时查看守护进程日志 (Ctrl+C退出):"
        tail -f "$LOG_FILE"
    else
        print_warning "日志文件不存在: $LOG_FILE"
    fi
}

# 执行健康检查
health_check() {
    print_status "执行系统健康检查..."
    cd "$PROJECT_ROOT"
    python run.py bot -m health
}

# 执行备份
backup_data() {
    print_status "执行手动数据备份..."
    cd "$PROJECT_ROOT"
    python run.py bot -m backup
}

# 显示帮助
show_help() {
    echo "指数交易机器人守护进程管理脚本"
    echo ""
    echo "用法: $0 {start|stop|restart|status|logs|health|backup|help}"
    echo ""
    echo "命令说明:"
    echo "  start   - 启动守护进程"
    echo "  stop    - 停止守护进程"
    echo "  restart - 重启守护进程"
    echo "  status  - 检查运行状态"
    echo "  logs    - 实时查看日志"
    echo "  health  - 执行健康检查"
    echo "  backup  - 执行数据备份"
    echo "  help    - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start         # 启动守护进程"
    echo "  $0 status        # 查看运行状态"
    echo "  $0 logs          # 实时查看日志"
    echo ""
    echo "配置文件: $PROJECT_ROOT/config/config_core.yaml"
    echo "日志文件: $LOG_FILE"
    echo "PID文件: $PID_FILE"
}

# 主逻辑
main() {
    case "$1" in
        start)
            check_venv
            check_dependencies
            start_daemon
            ;;
        stop)
            stop_daemon
            ;;
        restart)
            check_venv
            check_dependencies
            restart_daemon
            ;;
        status)
            check_status
            ;;
        logs)
            view_logs
            ;;
        health)
            check_venv
            health_check
            ;;
        backup)
            check_venv
            backup_data
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "无效命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 确保脚本在项目根目录执行
if [ ! -f "$PROJECT_ROOT/run.py" ]; then
    print_error "未找到run.py文件，请确保脚本在正确的项目目录中运行"
    exit 1
fi

# 执行主逻辑
main "$@" 