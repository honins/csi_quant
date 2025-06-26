#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试AI优化参数功能
验证新增的3个AI优化参数：动态置信度调整、市场情绪权重、趋势强度权重
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer

def load_config():
    """加载配置文件"""
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_ai_optimization_params():
    """测试AI优化参数功能"""
    print("=" * 60)
    print("测试AI优化参数功能")
    print("=" * 60)
    
    # 1. 加载配置
    config = load_config()
    print("✓ 配置文件加载成功")
    
    # 2. 加载数据
    data_module = DataModule(config)
    # 获取历史数据
    start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
    end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
    data = data_module.get_history_data(start_date, end_date)
    # 预处理数据
    data = data_module.preprocess_data(data)
    print(f"✓ 数据加载成功，数据长度: {len(data)}")
    
    # 3. 初始化策略模块
    strategy_module = StrategyModule(config)
    print("✓ 策略模块初始化成功")
    
    # 4. 获取原始参数
    original_params = strategy_module.get_params()
    print("\n原始策略参数:")
    for key, value in original_params.items():
        print(f"  {key}: {value}")
    
    # 5. 运行基准回测
    print("\n运行基准回测...")
    baseline_results = strategy_module.backtest(data)
    baseline_evaluation = strategy_module.evaluate_strategy(baseline_results)
    print(f"基准策略评估结果:")
    print(f"  识别点数: {baseline_evaluation['total_points']}")
    print(f"  成功率: {baseline_evaluation['success_rate']:.2%}")
    print(f"  平均涨幅: {baseline_evaluation['avg_rise']:.2%}")
    print(f"  综合得分: {baseline_evaluation['score']:.4f}")
    
    # 6. 初始化AI优化器
    ai_optimizer = AIOptimizer(config)
    print("\n✓ AI优化器初始化成功")
    
    # 7. 运行AI参数优化
    print("\n开始AI参数优化...")
    optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, data)
    
    print("\n优化后的参数:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    
    # 8. 更新策略参数
    strategy_module.update_params(optimized_params)
    print("\n✓ 策略参数已更新")
    
    # 9. 运行优化后的回测
    print("\n运行优化后的回测...")
    optimized_results = strategy_module.backtest(data)
    optimized_evaluation = strategy_module.evaluate_strategy(optimized_results)
    print(f"优化后策略评估结果:")
    print(f"  识别点数: {optimized_evaluation['total_points']}")
    print(f"  成功率: {optimized_evaluation['success_rate']:.2%}")
    print(f"  平均涨幅: {optimized_evaluation['avg_rise']:.2%}")
    print(f"  综合得分: {optimized_evaluation['score']:.4f}")
    
    # 10. 对比分析
    print("\n" + "=" * 40)
    print("优化效果对比:")
    print("=" * 40)
    
    metrics = ['识别点数', '成功率', '平均涨幅', '综合得分']
    baseline_values = [
        baseline_evaluation['total_points'],
        baseline_evaluation['success_rate'],
        baseline_evaluation['avg_rise'],
        baseline_evaluation['score']
    ]
    optimized_values = [
        optimized_evaluation['total_points'],
        optimized_evaluation['success_rate'],
        optimized_evaluation['avg_rise'],
        optimized_evaluation['score']
    ]
    
    for i, metric in enumerate(metrics):
        baseline_val = baseline_values[i]
        optimized_val = optimized_values[i]
        
        if isinstance(baseline_val, int):
            change = optimized_val - baseline_val
            change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0
            print(f"{metric}: {baseline_val} → {optimized_val} ({change:+d}, {change_pct:+.1f}%)")
        else:
            change = optimized_val - baseline_val
            change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0
            print(f"{metric}: {baseline_val:.4f} → {optimized_val:.4f} ({change:+.4f}, {change_pct:+.1f}%)")
    
    # 11. 验证新参数是否生效
    print("\n" + "=" * 40)
    print("验证新AI优化参数:")
    print("=" * 40)
    
    # 检查新参数是否在优化结果中
    new_params = ['dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight']
    for param in new_params:
        if param in optimized_params:
            print(f"✓ {param}: {optimized_params[param]}")
        else:
            print(f"✗ {param}: 未找到")
    
    # 12. 测试单个识别点的详细分析
    print("\n" + "=" * 40)
    print("测试单个识别点分析:")
    print("=" * 40)
    
    # 找到最近的相对低点
    recent_low_points = optimized_results[optimized_results['is_low_point']].tail(1)
    if len(recent_low_points) > 0:
        test_date = recent_low_points.iloc[0]['date']
        test_data = data[data['date'] <= test_date].tail(30)  # 取最近30天数据
        
        identification_result = strategy_module.identify_relative_low(test_data)
        
        print(f"测试日期: {test_date}")
        print(f"识别结果: {'是' if identification_result['is_low_point'] else '否'}")
        print(f"置信度: {identification_result['confidence']:.4f}")
        print(f"识别原因: {', '.join(identification_result['reasons'])}")
        
        # 检查是否包含新参数的影响
        reasons = identification_result['reasons']
        new_param_effects = []
        for reason in reasons:
            if any(param in reason.lower() for param in ['波动率', '情绪', '趋势']):
                new_param_effects.append(reason)
        
        if new_param_effects:
            print(f"新参数影响: {', '.join(new_param_effects)}")
        else:
            print("新参数影响: 未检测到")
    else:
        print("未找到相对低点进行测试")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_ai_optimization_params() 