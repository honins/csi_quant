#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
临时测试脚本：策略模块功能测试
"""

from src.utils.config_loader import load_config
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def test_strategy_module():
    print("🎯 测试策略模块功能")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    print("✅ 配置加载成功")
    
    # 初始化模块
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    print("✅ 模块初始化成功")
    
    # 验证策略参数
    strategy_config = config.get('strategy', {})
    rise_threshold = strategy_config.get('rise_threshold')
    max_days = strategy_config.get('max_days')
    print(f"✅ 策略参数: rise_threshold={rise_threshold}, max_days={max_days}")
    
    # 获取测试数据
    data = data_module.get_history_data('2024-01-01', '2024-02-29')
    processed_data = data_module.preprocess_data(data)
    print(f"✅ 测试数据准备完成: {len(processed_data)} 条记录")
    
    # 测试相对低点识别（单次识别）
    identification_result = strategy_module.identify_relative_low(processed_data)
    print(f"✅ 相对低点识别完成: 是否低点={identification_result['is_low_point']}, 置信度={identification_result['confidence']:.3f}")
    
    # 测试回测功能
    backtest_results = strategy_module.backtest(processed_data)
    print(f"✅ 回测完成: {len(backtest_results)} 条结果")
    
    # 测试策略评估
    evaluation = strategy_module.evaluate_strategy(backtest_results)
    print(f"✅ 策略评估完成: 成功率={evaluation['success_rate']:.2%}, 识别点数={evaluation['total_points']}")
    
    # 测试参数获取
    current_params = strategy_module.get_params()
    print(f"✅ 参数获取成功: 当前有{len(current_params)}个参数")
    
    print("✅ 策略模块功能正常")

if __name__ == "__main__":
    test_strategy_module() 