#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成功率计算验证测试
验证策略模块中成功率的计算逻辑
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.config_loader import load_config
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def verify_success_rate_calculation():
    """验证成功率计算逻辑"""
    print("🔍 验证成功率计算逻辑")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置
    config = load_config()
    
    # 获取数据
    data_module = DataModule(config)
    data = data_module.get_history_data('2023-01-01', '2024-12-31')
    data = data_module.preprocess_data(data)
    
    # 策略回测
    strategy_module = StrategyModule(config)
    backtest_results = strategy_module.backtest(data)
    
    # 手动验证成功率计算
    print("\n📊 手动验证成功率计算：")
    
    # 1. 获取所有相对低点
    low_points = backtest_results[backtest_results['is_low_point']]
    total_points = len(low_points)
    print(f"✅ 总识别点数: {total_points}")
    
    # 2. 计算成功案例
    rise_threshold = strategy_module.rise_threshold
    successful_points = low_points[low_points['future_max_rise'] >= rise_threshold]
    success_count = len(successful_points)
    print(f"✅ 成功案例数: {success_count}")
    print(f"✅ 目标涨幅阈值: {rise_threshold:.1%}")
    
    # 3. 手动计算成功率
    manual_success_rate = success_count / total_points if total_points > 0 else 0
    print(f"✅ 手动计算成功率: {manual_success_rate:.1%}")
    
    # 4. 系统计算成功率
    evaluation = strategy_module.evaluate_strategy(backtest_results)
    system_success_rate = evaluation['success_rate']
    print(f"✅ 系统计算成功率: {system_success_rate:.1%}")
    
    # 5. 验证一致性
    print(f"\n🔍 验证结果:")
    if abs(manual_success_rate - system_success_rate) < 0.001:
        print("✅ 成功率计算正确！手动计算与系统计算一致")
    else:
        print("❌ 成功率计算有误！手动计算与系统计算不一致")
        
    # 6. 详细分析
    print(f"\n📈 详细分析:")
    print(f"   - 平均涨幅: {evaluation['avg_rise']:.1%}")
    print(f"   - 最大涨幅: {evaluation['max_rise']:.1%}")
    print(f"   - 最小涨幅: {evaluation['min_rise']:.1%}")
    print(f"   - 平均天数: {evaluation['avg_days']:.1f}天")
    print(f"   - 综合得分: {evaluation['score']:.4f}")
    
    # 7. 查看成功案例示例
    if len(successful_points) > 0:
        print(f"\n🎯 成功案例示例（前5个）:")
        for i, (idx, row) in enumerate(successful_points.head().iterrows()):
            print(f"   {i+1}. 日期: {row['date']}, 涨幅: {row['future_max_rise']:.1%}, "
                  f"天数: {row['days_to_rise']}")
    
    # 8. 查看失败案例示例
    failed_points = low_points[low_points['future_max_rise'] < rise_threshold]
    if len(failed_points) > 0:
        print(f"\n❌ 失败案例示例（前3个）:")
        for i, (idx, row) in enumerate(failed_points.head(3).iterrows()):
            print(f"   {i+1}. 日期: {row['date']}, 涨幅: {row['future_max_rise']:.1%}")
    
    return True

if __name__ == "__main__":
    verify_success_rate_calculation() 