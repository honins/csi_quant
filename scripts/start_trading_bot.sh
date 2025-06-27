#!/bin/bash

# 日常交易机器人启动器 (Linux/Mac版本)

echo "================================================================"
echo "                   🚀 日常交易机器人启动器"
echo "================================================================"
echo

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查虚拟环境
if [ -f "venv/bin/activate" ]; then
    echo "✅ 发现虚拟环境，正在激活..."
    source venv/bin/activate
else
    echo "⚠️  未发现虚拟环境，使用系统Python"
fi

show_menu() {
    echo
    echo "请选择运行模式:"
    echo "[1] 单次执行 (立即执行一次交易流程)"
    echo "[2] 定时执行 (每天9:30自动执行)"
    echo "[3] 查看状态 (查看机器人运行状态)"
    echo "[4] 手动训练 (执行模型训练)"
    echo "[5] 手动预测 (执行预测)"
    echo "[0] 退出"
    echo
}

while true; do
    show_menu
    read -p "请输入选择 (0-5): " choice
    
    case $choice in
        1)
            echo
            echo "🚀 启动单次执行模式..."
            python run.py bot -m run
            echo
            read -p "按回车键继续..."
            ;;
        2)
            echo
            echo "⏰ 启动定时执行模式..."
            echo "💡 提示: 程序将在每天9:30自动执行交易流程"
            echo "💡 按 Ctrl+C 可以停止定时任务"
            python run.py bot -m schedule
            break
            ;;
        3)
            echo
            echo "📊 查看机器人状态..."
            python run.py bot -m status
            echo
            read -p "按回车键继续..."
            ;;
        4)
            echo
            echo "🤖 执行手动训练..."
            python run.py ai -m incremental
            echo
            read -p "按回车键继续..."
            ;;
        5)
            echo
            echo "🔮 执行手动预测..."
            python run.py ai -m demo
            echo
            read -p "按回车键继续..."
            ;;
        0)
            echo
            echo "👋 再见！"
            exit 0
            ;;
        *)
            echo
            echo "❌ 无效选择，请重新输入"
            sleep 1
            ;;
    esac
done

echo
echo "================================================================"
echo "执行完成！"
echo "================================================================" 