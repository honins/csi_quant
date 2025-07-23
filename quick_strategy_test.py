#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试策略参数更新功能
"""

import os
import sys
import yaml
import shutil
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.utils.config_saver import save_strategy_config

def backup_config():
    """备份当前配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"config/backups/quick_test_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists("config/strategy.yaml"):
        shutil.copy2("config/strategy.yaml", f"{backup_dir}/strategy_backup.yaml")
        print(f"✅ 已备份策略配置到: {backup_dir}/strategy_backup.yaml")
    
    return backup_dir

def load_config(config_file):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

def test_strategy_param_save():
    """测试策略参数保存功能"""
    print("🧪 快速策略参数更新测试")
    print("=" * 50)
    
    # 1. 备份当前配置
    backup_dir = backup_config()
    
    # 2. 记录保存前的配置
    print("\n📋 记录保存前配置...")
    before_config = load_config("config/strategy.yaml")
    print(f"✅ 策略配置: {'已加载' if before_config else '不存在'}")
    
    # 3. 模拟AI优化后的参数
    print("\n🔧 模拟AI优化参数...")
    optimized_params = {
        'rise_threshold': 0.045,  # 从0.04优化到0.045
        'max_days': 18,           # 从20优化到18
        'confidence_weights': {
            'rsi_oversold_threshold': 28,  # 从30优化到28
            'rsi_low_threshold': 38,       # 从40优化到38
            # final_threshold 现在在 system.yaml 中，不在这里优化
            'dynamic_confidence_adjustment': 0.85,  # 新增参数
            'market_sentiment_weight': 1.2,         # 新增参数
            'trend_strength_weight': 1.8            # 新增参数
        }
    }
    
    print("📊 模拟优化参数:")
    for key, value in optimized_params.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # 4. 保存优化参数
    print("\n💾 保存优化参数...")
    try:
        success = save_strategy_config(optimized_params)
        
        if success:
            print("✅ 参数保存成功")
        else:
            print("❌ 参数保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 保存异常: {e}")
        return False
    
    # 5. 检查保存后的配置
    print("\n📋 检查保存后配置...")
    after_config = load_config("config/strategy.yaml")
    
    if after_config:
        print("✅ 配置文件已更新")
        
        # 6. 比较变化
        print("\n🔍 配置变化分析:")
        print("=" * 30)
        
        if before_config != after_config:
            print("✅ 策略参数已更新！")
            print("\n📊 详细变化:")
            
            # 检查主要参数
            for key in ['rise_threshold', 'max_days']:
                if key in before_config and key in after_config:
                    if before_config[key] != after_config[key]:
                        print(f"  🔄 {key}: {before_config[key]} → {after_config[key]}")
                elif key in after_config:
                    print(f"  ➕ {key}: 新增 = {after_config[key]}")
            
            # 检查置信度权重
            if 'confidence_weights' in after_config:
                print(f"  ➕ confidence_weights: 新增配置段")
                for weight_key, weight_value in after_config['confidence_weights'].items():
                    print(f"    {weight_key}: {weight_value}")
            
            return True
        else:
            print("❌ 策略参数未更新")
            return False
    else:
        print("❌ 配置文件读取失败")
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
        success = test_strategy_param_save()
        
        # 总结
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"策略参数更新: {'✅ 成功' if success else '❌ 失败'}")
        
        if success:
            print("\n🎉 AI优化确实会更新策略参数！")
            print("💡 说明: AI优化器会通过_save_optimized_parameters方法")
            print("   将优化后的参数保存到config/strategy.yaml文件")
        else:
            print("\n💡 AI优化可能不会更新策略参数，或者保存功能有问题")
        
        # 询问是否恢复配置
        print(f"\n💾 原始配置已备份")
        response = input("是否恢复原始配置？(y/N): ").strip().lower()
        if response == 'y':
            backup_dir = f"config/backups/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            restore_config(backup_dir)
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()