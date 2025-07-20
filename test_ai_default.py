#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 python run.py ai 的默认行为
"""

import os
import sys
import yaml
import shutil
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def backup_config():
    """备份当前配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"config/backups/test_ai_default_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists("config/strategy.yaml"):
        shutil.copy2("config/strategy.yaml", f"{backup_dir}/strategy_backup.yaml")
        print(f"✅ 已备份策略配置")
    
    return backup_dir

def load_config(config_file):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

def test_ai_default_behavior():
    """测试AI默认行为"""
    print("🧪 测试 python run.py ai 默认行为")
    print("=" * 50)
    
    # 1. 备份配置
    backup_dir = backup_config()
    
    # 2. 记录执行前状态
    print("\n📋 记录执行前状态...")
    before_strategy = load_config("config/strategy.yaml")
    print(f"✅ 策略配置: {'已加载' if before_strategy else '不存在'}")
    
    # 3. 模拟AI默认行为（optimize模式）
    print("\n🚀 模拟 python run.py ai 默认行为...")
    print("💡 根据代码分析，默认模式是 'optimize'")
    
    try:
        # 加载配置
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        # 获取数据
        data_config = config.get('data', {})
        time_range = data_config.get('time_range', {})
        start_date = time_range.get('start_date', '2019-01-01')
        end_date = time_range.get('end_date', '2025-07-15')
        
        data = data_module.get_history_data(start_date, end_date)
        
        if data is None or data.empty:
            print("❌ 无法获取数据")
            return False
        
        print(f"✅ 获取数据成功: {len(data)} 条记录")
        
        # 运行完整AI优化（默认optimize模式）
        print("🔧 运行完整AI优化（包含策略参数优化）...")
        start_time = datetime.now()
        
        result = ai_optimizer.run_complete_optimization(data, strategy_module)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  优化耗时: {duration:.1f}秒")
        
        if result.get('success', False):
            print("✅ AI优化成功完成")
            
            # 检查策略参数是否更新
            print("\n📋 检查策略参数更新...")
            after_strategy = load_config("config/strategy.yaml")
            
            if before_strategy != after_strategy:
                print("✅ 策略参数已更新！")
                print("\n📊 变化详情:")
                
                if before_strategy and after_strategy:
                    for key in set(before_strategy.keys()) | set(after_strategy.keys()):
                        if key not in before_strategy:
                            print(f"  ➕ {key}: 新增")
                        elif key not in after_strategy:
                            print(f"  ➖ {key}: 删除")
                        elif before_strategy[key] != after_strategy[key]:
                            if isinstance(before_strategy[key], dict) and isinstance(after_strategy[key], dict):
                                print(f"  🔄 {key}: 字典内容变化")
                            else:
                                print(f"  🔄 {key}: {before_strategy[key]} → {after_strategy[key]}")
                
                return True
            else:
                print("❌ 策略参数未更新")
                return False
        else:
            print("❌ AI优化失败")
            print(f"错误: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def restore_config(backup_dir):
    """恢复配置"""
    print(f"\n🔄 恢复原始配置...")
    backup_file = f"{backup_dir}/strategy_backup.yaml"
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, "config/strategy.yaml")
        print("✅ 配置已恢复")
    else:
        print("❌ 备份文件不存在")

def main():
    """主测试函数"""
    try:
        # 运行测试
        success = test_ai_default_behavior()
        
        # 总结
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"AI默认行为测试: {'✅ 成功' if success else '❌ 失败'}")
        
        if success:
            print("\n🎉 python run.py ai 确实会更新策略参数！")
            print("💡 默认模式是 'optimize'，会进行完整的策略参数优化")
            print("📋 包括:")
            print("   - 策略参数优化（遗传算法）")
            print("   - 模型训练")
            print("   - 参数保存到config/strategy.yaml")
        else:
            print("\n💡 python run.py ai 可能不会更新策略参数")
        
        # 询问是否恢复配置
        print(f"\n💾 原始配置已备份")
        response = input("是否恢复原始配置？(y/N): ").strip().lower()
        if response == 'y':
            restore_config(backup_dir)
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 