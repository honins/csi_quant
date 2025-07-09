#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式使用指导脚本
为用户提供个性化的使用建议和命令生成
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header():
    """打印头部信息"""
    print("=" * 60)
    print("🎯 中证500指数相对低点识别系统 - 交互式指导")
    print("=" * 60)
    print()

def check_environment():
    """检查运行环境"""
    print("🔍 环境检查...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✅ Python版本: {python_version.major}.{python_version.minor}")
    else:
        print(f"  ❌ Python版本过低: {python_version.major}.{python_version.minor} (需要3.8+)")
        return False
    
    # 检查虚拟环境
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("  ✅ 运行在虚拟环境中")
    else:
        print("  ⚠️  未在虚拟环境中运行 (建议使用虚拟环境)")
    
    # 检查关键文件
    key_files = [
        "run.py",
        "config/config.yaml",
        "src/ai/ai_optimizer_improved.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - 文件不存在")
            missing_files.append(file_path)
    
    print()
    return len(missing_files) == 0

def get_user_experience():
    """获取用户经验水平"""
    print("👤 请选择您的经验水平:")
    print("  1. 🆕 新手用户 (第一次使用)")
    print("  2. 📊 有经验的量化投资者")
    print("  3. 👨‍💻 开发者/研究者")
    print()
    
    while True:
        choice = input("请输入选择 (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("❌ 无效选择，请输入1、2或3")

def get_user_goal():
    """获取用户目标"""
    print("🎯 请选择您的主要目标:")
    print("  1. 🔍 快速体验系统功能")
    print("  2. 📈 进行单日预测")
    print("  3. 📊 回测历史策略表现")
    print("  4. 🤖 训练AI模型")
    print("  5. ⚙️ 优化策略参数")
    print("  6. 🤖 设置自动化交易机器人")
    print("  7. 🔧 系统故障排除")
    print()
    
    while True:
        choice = input("请输入选择 (1-7): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7']:
            return int(choice)
        print("❌ 无效选择，请输入1-7的数字")

def generate_newbie_guide():
    """生成新手指导"""
    print("🆕 新手用户指导:")
    print("建议按以下顺序进行:")
    print()
    
    steps = [
        ("1. 基础测试", "python run.py b", "验证系统基本功能是否正常"),
        ("2. AI模型训练", "python run.py ai -m full", "训练AI模型（约6-12分钟）"),
        ("3. 测试预测", "python run.py s 2024-06-15", "测试单日预测功能"),
        ("4. 查看结果", "cat results/single_predictions/prediction_*.json", "查看预测结果文件"),
        ("5. 进行回测", "python run.py r 2023-01-01 2023-12-31", "验证策略历史表现")
    ]
    
    for step_name, command, description in steps:
        print(f"  📌 {step_name}")
        print(f"     命令: {command}")
        print(f"     说明: {description}")
        print()
    
    print("📚 更多详细信息请查看: USER_GUIDE.md")

def generate_goal_specific_guide(goal):
    """根据目标生成具体指导"""
    guides = {
        1: {  # 快速体验
            "title": "🔍 快速体验系统功能",
            "commands": [
                ("基础测试", "python run.py b"),
                ("AI测试", "python run.py a"),
                ("单日预测", "python run.py s 2024-06-15")
            ],
            "time": "约5-10分钟",
            "note": "这些命令可以让您快速了解系统的主要功能"
        },
        2: {  # 单日预测
            "title": "📈 进行单日预测",
            "commands": [
                ("检查模型", "ls models/improved_model_*.pkl"),
                ("训练模型 (如需)", "python run.py ai -m full"),
                ("单日预测", "python run.py s 2024-12-01"),
                ("查看结果", "cat results/single_predictions/prediction_*.json")
            ],
            "time": "约1-15分钟 (取决于是否需要训练)",
            "note": "如果没有已训练的模型，系统会自动训练"
        },
        3: {  # 回测
            "title": "📊 回测历史策略表现",
            "commands": [
                ("最近一年回测", "python run.py r 2023-01-01 2023-12-31"),
                ("疫情期间回测", "python run.py r 2020-01-01 2021-12-31"),
                ("查看图表", "ls results/charts/rolling_backtest/")
            ],
            "time": "约2-5分钟",
            "note": "回测会生成详细的性能图表和统计报告"
        },
        4: {  # AI训练
            "title": "🤖 训练AI模型",
            "commands": [
                ("完全重训练", "python run.py ai -m full"),
                ("增量训练", "python run.py ai -m incremental"),
                ("检查模型", "ls models/"),
                ("测试预测", "python run.py s 2024-06-15")
            ],
            "time": "约6-15分钟",
            "note": "推荐使用6年数据进行训练，平衡效果和速度"
        },
        5: {  # 参数优化
            "title": "⚙️ 优化策略参数",
            "commands": [
                ("快速优化", "python run.py opt"),
                ("深度优化", "python run.py opt -i 50"),
                ("完整AI优化", "python run.py ai -m optimize"),
                ("验证效果", "python run.py s 2024-06-15")
            ],
            "time": "约10-30分钟",
            "note": "优化后的参数会自动保存到配置文件"
        },
        6: {  # 交易机器人
            "title": "🤖 设置自动化交易机器人",
            "commands": [
                ("测试运行", "python run.py bot -m run"),
                ("查看状态", "python run.py bot -m status"),
                ("定时执行", "python run.py bot -m schedule")
            ],
            "time": "约2-5分钟",
            "note": "机器人可以定时执行预测和发送通知"
        },
        7: {  # 故障排除
            "title": "🔧 系统故障排除",
            "commands": [
                ("详细测试", "python run.py b -v"),
                ("检查日志", "tail -f logs/system.log"),
                ("验证配置", "python -c \"import yaml; print('配置文件语法正确')\""),
                ("检查依赖", "pip list | grep -E '(pandas|numpy|sklearn)'")
            ],
            "time": "约3-10分钟",
            "note": "按顺序执行这些命令可以诊断大部分问题"
        }
    }
    
    guide = guides[goal]
    print(f"{guide['title']}:")
    print(f"  ⏱️  预计时间: {guide['time']}")
    print(f"  💡 说明: {guide['note']}")
    print()
    print("  📋 执行步骤:")
    
    for i, (step_name, command) in enumerate(guide['commands'], 1):
        print(f"    {i}. {step_name}")
        print(f"       {command}")
        print()

def generate_command_script(experience, goal):
    """生成可执行的命令脚本"""
    script_name = f"quick_start_{experience}_{goal}.bat" if os.name == 'nt' else f"quick_start_{experience}_{goal}.sh"
    
    commands = []
    
    if experience == 1:  # 新手
        commands = [
            "# 新手用户快速开始脚本",
            "echo 正在进行基础测试...",
            "python run.py b",
            "echo 正在训练AI模型...",
            "python run.py ai -m full",
            "echo 正在测试预测...",
            "python run.py s 2024-06-15",
            "echo 完成！请查看 results/ 目录下的结果文件"
        ]
    else:
        goal_commands = {
            1: ["python run.py b", "python run.py a", "python run.py s 2024-06-15"],
            2: ["python run.py ai -m incremental", "python run.py s 2024-12-01"],
            3: ["python run.py r 2023-01-01 2023-12-31"],
            4: ["python run.py ai -m full"],
            5: ["python run.py opt", "python run.py ai -m optimize"],
            6: ["python run.py bot -m run"],
            7: ["python run.py b -v", "tail -n 50 logs/system.log"]
        }
        commands = goal_commands.get(goal, ["python run.py b"])
    
    try:
        with open(script_name, 'w', encoding='utf-8') as f:
            if os.name == 'nt':  # Windows
                f.write("@echo off\n")
                f.write("call venv\\Scripts\\activate\n")
            else:  # Linux/Mac
                f.write("#!/bin/bash\n")
                f.write("source venv/bin/activate\n")
            
            for cmd in commands:
                f.write(f"{cmd}\n")
                
        print(f"📝 已生成执行脚本: {script_name}")
        print(f"   运行方式: {script_name}")
        
    except Exception as e:
        print(f"❌ 生成脚本失败: {e}")

def show_useful_resources():
    """显示有用的资源链接"""
    print("📚 有用的资源:")
    print("  📖 完整使用指南: USER_GUIDE.md")
    print("  ⚡ 快速开始: QUICKSTART.md")
    print("  🔧 参数重置指南: RESET_GUIDE.md")
    print("  📊 数据分析: DATA_ANALYSIS.md")
    print("  📝 详细文档: DOCS.md")
    print()
    print("📞 遇到问题时:")
    print("  1. 查看 logs/system.log 日志文件")
    print("  2. 运行 python run.py b -v 进行详细测试")
    print("  3. 检查虚拟环境是否激活")
    print("  4. 确认配置文件语法正确")

def main():
    """主函数"""
    print_header()
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请先解决环境问题")
        return
    
    # 获取用户信息
    experience = get_user_experience()
    goal = get_user_goal()
    
    print()
    print("🎯 个性化指导:")
    print("=" * 40)
    
    # 生成指导
    if experience == 1:  # 新手用户
        generate_newbie_guide()
    else:
        generate_goal_specific_guide(goal)
    
    # 询问是否生成脚本
    print()
    create_script = input("是否生成可执行脚本? (y/n): ").strip().lower()
    if create_script in ['y', 'yes', '是']:
        generate_command_script(experience, goal)
    
    print()
    show_useful_resources()
    
    print()
    print("🎉 祝您使用愉快！如有问题请查看相关文档。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查环境配置或查看 USER_GUIDE.md 获取帮助") 