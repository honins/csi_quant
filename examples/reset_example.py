#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置脚本使用示例

演示如何使用策略参数重置功能
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_command(cmd):
    """运行命令并显示结果"""
    print(f"\n🔧 执行命令: {cmd}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"命令执行失败: {e}")
        return False


def demo_reset_functions():
    """演示重置功能"""
    
    print("🎯 策略参数重置功能演示")
    print("=" * 60)
    
    # 1. 显示帮助信息
    print("\n📖 1. 查看重置脚本帮助")
    run_command("python3 reset_strategy_params.py --help")
    
    # 2. 查看当前配置
    print("\n📄 2. 查看当前策略配置")
    run_command("python3 reset_strategy_params.py --show strategy")
    
    # 3. 创建备份
    print("\n💾 3. 创建配置备份")
    run_command("python3 reset_strategy_params.py --backup")
    
    # 4. 演示选择性重置（仅演示，不实际执行）
    print("\n🔄 4. 重置选项演示（以下为示例命令，未实际执行）")
    print("\n重置所有参数:")
    print("  python3 reset_strategy_params.py --all")
    
    print("\n仅重置策略参数:")
    print("  python3 reset_strategy_params.py --strategy")
    
    print("\n仅重置置信度权重:")
    print("  python3 reset_strategy_params.py --confidence")
    
    print("\n仅重置优化参数:")
    print("  python3 reset_strategy_params.py --optimization")
    
    print("\n强制重置（跳过确认）:")
    print("  python3 reset_strategy_params.py --all --force")
    
    # 5. 快速重置演示
    print("\n⚡ 5. 快速重置演示（以下为示例命令，未实际执行）")
    print("\n快速重置核心参数:")
    print("  python3 quick_reset.py")
    
    # 6. 查看备份目录
    print("\n📁 6. 查看备份目录")
    backup_dir = project_root / "config" / "backups"
    if backup_dir.exists():
        print(f"\n备份目录: {backup_dir}")
        for item in sorted(backup_dir.iterdir(), reverse=True)[:5]:  # 显示最新的5个备份
            if item.is_dir():
                files = list(item.glob("*.yaml"))
                print(f"  📂 {item.name} ({len(files)} 个文件)")
    
    print("\n✅ 演示完成!")
    print("\n💡 使用建议:")
    print("  1. 重置前先创建备份: --backup")
    print("  2. 查看当前配置: --show strategy")
    print("  3. 选择合适的重置方式")
    print("  4. 重置后运行测试验证")


def show_reset_scenarios():
    """显示常见重置场景"""
    
    print("\n🎯 常见重置场景")
    print("=" * 40)
    
    scenarios = [
        {
            "场景": "参数优化后效果不佳",
            "建议": "python3 reset_strategy_params.py --confidence",
            "说明": "重置置信度权重，保留其他优化设置"
        },
        {
            "场景": "系统运行异常",
            "建议": "python3 reset_strategy_params.py --all",
            "说明": "完全重置到默认状态"
        },
        {
            "场景": "快速修复问题",
            "建议": "python3 quick_reset.py",
            "说明": "一键重置核心参数"
        },
        {
            "场景": "测试新策略前",
            "建议": "python3 reset_strategy_params.py --strategy",
            "说明": "重置策略参数，保留优化设置"
        },
        {
            "场景": "优化算法调试",
            "建议": "python3 reset_strategy_params.py --optimization",
            "说明": "重置优化参数，保留策略设置"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['场景']}")
        print(f"   命令: {scenario['建议']}")
        print(f"   说明: {scenario['说明']}")


def main():
    """主函数"""
    
    print("🚀 策略参数重置功能演示")
    print("=" * 60)
    
    # 检查是否在正确的目录
    if not (project_root / "reset_strategy_params.py").exists():
        print("❌ 错误: 请在项目根目录下运行此脚本")
        return
    
    try:
        # 演示重置功能
        demo_reset_functions()
        
        # 显示使用场景
        show_reset_scenarios()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示已中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()