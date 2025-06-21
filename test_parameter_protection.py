#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试参数保护功能
验证rise_threshold和max_days是否真正保持固定
"""

import sys
import os
import yaml

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.utils import load_config
from ai.ai_optimizer import AIOptimizer
from strategy.strategy_module import StrategyModule
from data.data_module import DataModule

def test_parameter_protection():
    """测试参数保护功能"""
    print("="*60)
    print("🧪 测试参数保护功能")
    print("="*60)
    
    # 1. 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 记录原始的核心参数值
    original_rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.05)
    original_max_days = config.get('strategy', {}).get('max_days', 20)
    
    print(f"📋 原始核心参数:")
    print(f"   - rise_threshold: {original_rise_threshold}")
    print(f"   - max_days: {original_max_days}")
    
    # 2. 初始化模块
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    ai_optimizer = AIOptimizer(config)
    
    # 3. 获取测试数据
    print("\n📊 准备测试数据...")
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    raw_data = data_module.get_history_data(start_date, end_date)
    processed_data = data_module.preprocess_data(raw_data)
    print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
    
    # 4. 测试AI优化器
    print("\n🔧 测试AI优化器...")
    
    # 测试主要优化方法
    print("   测试 optimize_strategy_parameters...")
    optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
    
    print(f"   优化结果:")
    print(f"     - rise_threshold: {optimized_params.get('rise_threshold')} (应该是 {original_rise_threshold})")
    print(f"     - max_days: {optimized_params.get('max_days')} (应该是 {original_max_days})")
    
    # 验证核心参数是否保持固定
    rise_threshold_fixed = abs(optimized_params.get('rise_threshold', 0) - original_rise_threshold) < 1e-6
    max_days_fixed = optimized_params.get('max_days', 0) == original_max_days
    
    print(f"   验证结果:")
    print(f"     - rise_threshold 固定: {'✅' if rise_threshold_fixed else '❌'}")
    print(f"     - max_days 固定: {'✅' if max_days_fixed else '❌'}")
    
    # 5. 测试高级优化方法
    print("\n   测试 optimize_strategy_parameters_advanced...")
    advanced_params = ai_optimizer.optimize_strategy_parameters_advanced(strategy_module, processed_data)
    
    print(f"   高级优化结果:")
    print(f"     - rise_threshold: {advanced_params.get('rise_threshold')} (应该是 {original_rise_threshold})")
    print(f"     - max_days: {advanced_params.get('max_days')} (应该是 {original_max_days})")
    
    # 验证核心参数是否保持固定
    advanced_rise_threshold_fixed = abs(advanced_params.get('rise_threshold', 0) - original_rise_threshold) < 1e-6
    advanced_max_days_fixed = advanced_params.get('max_days', 0) == original_max_days
    
    print(f"   高级优化验证结果:")
    print(f"     - rise_threshold 固定: {'✅' if advanced_rise_threshold_fixed else '❌'}")
    print(f"     - max_days 固定: {'✅' if advanced_max_days_fixed else '❌'}")
    
    # 6. 测试遗传算法
    print("\n   测试遗传算法...")
    
    def evaluate_func(params):
        return 0.5  # 简单的评估函数
    
    genetic_params = ai_optimizer.run_genetic_algorithm(evaluate_func)
    
    print(f"   遗传算法结果:")
    print(f"     - rise_threshold: {genetic_params.get('rise_threshold')} (应该是 {original_rise_threshold})")
    print(f"     - max_days: {genetic_params.get('max_days')} (应该是 {original_max_days})")
    
    # 验证核心参数是否保持固定
    genetic_rise_threshold_fixed = abs(genetic_params.get('rise_threshold', 0) - original_rise_threshold) < 1e-6
    genetic_max_days_fixed = genetic_params.get('max_days', 0) == original_max_days
    
    print(f"   遗传算法验证结果:")
    print(f"     - rise_threshold 固定: {'✅' if genetic_rise_threshold_fixed else '❌'}")
    print(f"     - max_days 固定: {'✅' if genetic_max_days_fixed else '❌'}")
    
    # 7. 测试分层优化
    print("\n   测试分层优化...")
    hierarchical_result = ai_optimizer.hierarchical_optimization(processed_data)
    hierarchical_params = hierarchical_result['strategy_params']
    
    print(f"   分层优化结果:")
    print(f"     - rise_threshold: {hierarchical_params.get('rise_threshold')} (应该是 {original_rise_threshold})")
    print(f"     - max_days: {hierarchical_params.get('max_days')} (应该是 {original_max_days})")
    
    # 验证核心参数是否保持固定
    hierarchical_rise_threshold_fixed = abs(hierarchical_params.get('rise_threshold', 0) - original_rise_threshold) < 1e-6
    hierarchical_max_days_fixed = hierarchical_params.get('max_days', 0) == original_max_days
    
    print(f"   分层优化验证结果:")
    print(f"     - rise_threshold 固定: {'✅' if hierarchical_rise_threshold_fixed else '❌'}")
    print(f"     - max_days 固定: {'✅' if hierarchical_max_days_fixed else '❌'}")
    
    # 8. 总结测试结果
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    all_tests_passed = (
        rise_threshold_fixed and max_days_fixed and
        advanced_rise_threshold_fixed and advanced_max_days_fixed and
        genetic_rise_threshold_fixed and genetic_max_days_fixed and
        hierarchical_rise_threshold_fixed and hierarchical_max_days_fixed
    )
    
    if all_tests_passed:
        print("🎉 所有测试通过！核心参数保护功能正常工作")
        print("✅ rise_threshold 和 max_days 在所有优化方法中都保持固定")
    else:
        print("❌ 部分测试失败！核心参数保护功能存在问题")
        print("请检查AI优化器的实现")
    
    print("="*60)
    
    return all_tests_passed

if __name__ == '__main__':
    success = test_parameter_protection()
    sys.exit(0 if success else 1) 