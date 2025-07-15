#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
零得分诊断脚本
分析为什么AI优化过程中所有得分都是0
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.utils.config_loader import load_config

def diagnose_zero_score():
    """诊断零得分问题"""
    print("="*80)
    print("🔍 零得分问题诊断")
    print("="*80)
    
    # 1. 加载配置和数据
    print("\n📋 步骤1: 加载配置和数据")
    config = load_config()
    if not config:
        print("❌ 配置加载失败")
        return
    
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    
    # 获取测试数据
    start_date = '2023-01-01'
    end_date = '2025-06-21'
    data = data_module.get_history_data(start_date, end_date)
    data = data_module.preprocess_data(data)
    
    print(f"✅ 数据加载完成: {len(data)} 条记录")
    print(f"📅 数据范围: {data['date'].min()} 到 {data['date'].max()}")
    
    # 2. 测试不同参数组合
    print("\n📊 步骤2: 测试不同参数组合")
    
    # 测试参数组合
    test_params = [
        {
            'name': '宽松参数',
            'final_threshold': 0.3,
            'rsi_oversold_threshold': 35,
            'rsi_low_threshold': 45,
            'ma_all_below': 0.2,
            'bb_lower_near': 0.15
        },
        {
            'name': '中等参数',
            'final_threshold': 0.4,
            'rsi_oversold_threshold': 32,
            'rsi_low_threshold': 42,
            'ma_all_below': 0.25,
            'bb_lower_near': 0.18
        },
        {
            'name': '严格参数',
            'final_threshold': 0.5,
            'rsi_oversold_threshold': 30,
            'rsi_low_threshold': 40,
            'ma_all_below': 0.3,
            'bb_lower_near': 0.2
        }
    ]
    
    results = []
    
    for params in test_params:
        print(f"\n🔧 测试参数组合: {params['name']}")
        
        # 更新策略参数
        strategy_module.update_params(params)
        
        # 运行回测
        backtest_results = strategy_module.backtest(data)
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        # 记录结果
        result = {
            'params': params['name'],
            'total_points': evaluation.get('total_points', 0),
            'success_rate': evaluation.get('success_rate', 0),
            'avg_rise': evaluation.get('avg_rise', 0),
            'score': evaluation.get('score', 0)
        }
        results.append(result)
        
        print(f"   识别点数: {result['total_points']}")
        print(f"   成功率: {result['success_rate']:.2%}")
        print(f"   平均涨幅: {result['avg_rise']:.2%}")
        print(f"   得分: {result['score']:.4f}")
    
    # 3. 分析数据特征
    print("\n📈 步骤3: 分析数据特征")
    
    # 检查技术指标分布
    print("\n📊 技术指标分布:")
    print(f"   RSI范围: {data['rsi'].min():.2f} - {data['rsi'].max():.2f}")
    print(f"   RSI < 30的比例: {(data['rsi'] < 30).mean():.2%}")
    print(f"   RSI < 40的比例: {(data['rsi'] < 40).mean():.2%}")
    print(f"   MACD < 0的比例: {(data['macd'] < 0).mean():.2%}")
    
    # 检查价格相对位置
    print("\n💰 价格相对位置:")
    price_below_ma5 = (data['close'] < data['ma5']).mean()
    price_below_ma10 = (data['close'] < data['ma10']).mean()
    price_below_ma20 = (data['close'] < data['ma20']).mean()
    price_below_all_ma = ((data['close'] < data['ma5']) & 
                          (data['close'] < data['ma10']) & 
                          (data['close'] < data['ma20'])).mean()
    
    print(f"   价格 < MA5: {price_below_ma5:.2%}")
    print(f"   价格 < MA10: {price_below_ma10:.2%}")
    print(f"   价格 < MA20: {price_below_ma20:.2%}")
    print(f"   价格 < 所有均线: {price_below_all_ma:.2%}")
    
    # 检查布林带位置
    print("\n📏 布林带位置:")
    bb_position = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    near_bb_lower = (bb_position < 0.2).mean()
    print(f"   接近下轨比例: {near_bb_lower:.2%}")
    
    # 4. 详细分析单个识别点
    print("\n🔍 步骤4: 详细分析单个识别点")
    
    # 使用最宽松的参数
    strategy_module.update_params(test_params[0])
    backtest_results = strategy_module.backtest(data)
    
    # 找到所有识别点
    low_points = backtest_results[backtest_results['is_low_point']]
    
    if len(low_points) > 0:
        print(f"✅ 找到 {len(low_points)} 个识别点")
        
        # 分析前几个识别点
        for i, (idx, row) in enumerate(low_points.head(3).iterrows()):
            print(f"\n📅 识别点 {i+1}: {row['date'].strftime('%Y-%m-%d')}")
            print(f"   收盘价: {row['close']:.2f}")
            print(f"   RSI: {row['rsi']:.2f}")
            print(f"   MACD: {row['macd']:.4f}")
            print(f"   相对MA5: {((row['close'] - row['ma5']) / row['ma5']):.2%}")
            print(f"   相对MA20: {((row['close'] - row['ma20']) / row['ma20']):.2%}")
            print(f"   未来最大涨幅: {row['future_max_rise']:.2%}")
            print(f"   达到目标天数: {row['days_to_rise']}")
    else:
        print("❌ 没有找到任何识别点")
        
        # 分析为什么没有识别点
        print("\n🔍 分析可能的原因:")
        
        # 检查置信度分布
        confidence_scores = []
        for i in range(len(data)):
            historical_data = data.iloc[:i+1].copy()
            if len(historical_data) > 20:  # 确保有足够的历史数据
                result = strategy_module.identify_relative_low(historical_data)
                confidence_scores.append(result['confidence'])
        
        if confidence_scores:
            confidence_scores = np.array(confidence_scores)
            print(f"   置信度范围: {confidence_scores.min():.4f} - {confidence_scores.max():.4f}")
            print(f"   置信度平均值: {confidence_scores.mean():.4f}")
            print(f"   置信度 > 0.3的比例: {(confidence_scores > 0.3).mean():.2%}")
            print(f"   置信度 > 0.4的比例: {(confidence_scores > 0.4).mean():.2%}")
            print(f"   置信度 > 0.5的比例: {(confidence_scores > 0.5).mean():.2%}")
    
    # 5. 建议解决方案
    print("\n💡 步骤5: 建议解决方案")
    
    if all(r['total_points'] == 0 for r in results):
        print("❌ 所有参数组合都没有识别到相对低点")
        print("\n🔧 建议的解决方案:")
        print("1. 降低 final_threshold 到 0.3-0.4")
        print("2. 放宽 RSI 阈值 (oversold: 35, low: 45)")
        print("3. 降低移动平均线权重")
        print("4. 检查数据质量和市场环境")
        print("5. 考虑调整 rise_threshold 到 0.03")
    else:
        print("✅ 找到有效的参数组合")
        best_result = max(results, key=lambda x: x['total_points'])
        print(f"最佳参数: {best_result['params']}")
        print(f"识别点数: {best_result['total_points']}")
    
    return results

def main():
    """主函数"""
    try:
        results = diagnose_zero_score()
        
        # 保存诊断结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"results/zero_score_diagnosis_{timestamp}.json"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'results': results,
                'summary': '零得分问题诊断结果'
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 诊断结果已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 