#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试参数范围修复
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.ai.ai_optimizer_improved import AIOptimizerImproved

def test_parameter_ranges():
    """测试参数范围修复"""
    print("🧪 测试参数范围修复")
    print("=" * 50)
    
    try:
        # 加载配置
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # 初始化AI优化器
        ai_optimizer = AIOptimizerImproved(config)
        
        # 获取参数范围
        param_ranges = ai_optimizer._get_enhanced_parameter_ranges({})
        
        print("📋 配置文件中的参数范围:")
        print("-" * 30)
        
        # 显示strategy_ranges
        strategy_ranges = config.get('strategy_ranges', {})
        print("🔧 strategy_ranges (基础参数):")
        for param_name, param_config in strategy_ranges.items():
            print(f"   {param_name}: {param_config.get('min', 'N/A')} - {param_config.get('max', 'N/A')}")
        print()
        
        # 显示optimization_ranges
        optimization_ranges = config.get('optimization_ranges', {})
        print("🤖 optimization_ranges (AI优化参数):")
        for param_name, param_config in optimization_ranges.items():
            print(f"   {param_name}: {param_config.get('min', 'N/A')} - {param_config.get('max', 'N/A')}")
        print()
        
        print("🎯 AI优化器使用的参数范围:")
        print("-" * 30)
        for param_name, param_config in param_ranges.items():
            print(f"   {param_name}: {param_config['min']} - {param_config['max']} ({param_config['type']})")
        print()
        
        # 验证参数数量
        expected_params = len(strategy_ranges) + len(optimization_ranges) - 2  # 减去固定的rise_threshold和max_days
        actual_params = len(param_ranges)
        
        print("📊 参数数量对比:")
        print(f"   配置文件中的参数数量: {len(strategy_ranges) + len(optimization_ranges)}")
        print(f"   固定参数数量: 2 (rise_threshold, max_days)")
        print(f"   预期优化参数数量: {expected_params}")
        print(f"   实际优化参数数量: {actual_params}")
        print()
        
        # 验证参数范围一致性
        print("✅ 参数范围一致性检查:")
        print("-" * 30)
        
        all_consistent = True
        
        # 检查strategy_ranges参数
        for param_name, param_config in strategy_ranges.items():
            if param_name in ['rise_threshold', 'max_days']:
                continue
                
            if param_name in param_ranges:
                config_min = param_config.get('min')
                config_max = param_config.get('max')
                optimizer_min = param_ranges[param_name]['min']
                optimizer_max = param_ranges[param_name]['max']
                
                if config_min == optimizer_min and config_max == optimizer_max:
                    print(f"   ✅ {param_name}: 范围一致")
                else:
                    print(f"   ❌ {param_name}: 范围不一致")
                    print(f"      配置文件: {config_min} - {config_max}")
                    print(f"      优化器: {optimizer_min} - {optimizer_max}")
                    all_consistent = False
            else:
                print(f"   ❌ {param_name}: 在优化器中缺失")
                all_consistent = False
        
        # 检查optimization_ranges参数
        for param_name, param_config in optimization_ranges.items():
            if param_name in param_ranges:
                config_min = param_config.get('min')
                config_max = param_config.get('max')
                optimizer_min = param_ranges[param_name]['min']
                optimizer_max = param_ranges[param_name]['max']
                
                if config_min == optimizer_min and config_max == optimizer_max:
                    print(f"   ✅ {param_name}: 范围一致")
                else:
                    print(f"   ❌ {param_name}: 范围不一致")
                    print(f"      配置文件: {config_min} - {config_max}")
                    print(f"      优化器: {optimizer_min} - {optimizer_max}")
                    all_consistent = False
            else:
                print(f"   ❌ {param_name}: 在优化器中缺失")
                all_consistent = False
        
        print()
        
        # 总结
        print("📊 测试总结:")
        print("=" * 30)
        print(f"参数数量匹配: {'✅ 是' if actual_params == expected_params else '❌ 否'}")
        print(f"参数范围一致: {'✅ 是' if all_consistent else '❌ 否'}")
        
        if actual_params == expected_params and all_consistent:
            print("\n🎉 参数范围修复成功！")
            print("💡 AI优化器现在使用配置文件中的参数范围")
        else:
            print("\n💡 参数范围修复需要进一步调整")
        
        return actual_params == expected_params and all_consistent
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    try:
        # 运行测试
        success = test_parameter_ranges()
        
        # 总结
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"参数范围修复测试: {'✅ 成功' if success else '❌ 失败'}")
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 