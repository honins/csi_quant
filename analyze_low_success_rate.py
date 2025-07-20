#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成功率低问题分析脚本
分析训练集26.79%，测试集27.96%成功率低的原因
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_low_success_rate():
    """分析成功率低的原因"""
    print("🔍 成功率低问题分析")
    print("=" * 60)
    print("当前成功率: 训练集26.79%，测试集27.96%")
    print("目标分析: 找出导致成功率低的根本原因")
    print("=" * 60)
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.data.data_module import DataModule
        from src.strategy.strategy_module import StrategyModule
        
        # 1. 加载配置和数据
        print("📋 1. 加载配置和数据...")
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        data_module = DataModule(config)
        data_config = config.get('data', {})
        time_range = data_config.get('time_range', {})
        start_date = time_range.get('start_date', '2019-01-01')
        end_date = time_range.get('end_date', '2025-07-15')
        
        data = data_module.get_history_data(start_date, end_date)
        data = data_module.preprocess_data(data)
        
        print(f"   数据时间范围: {start_date} ~ {end_date}")
        print(f"   数据长度: {len(data)} 条")
        
        # 2. 分析策略参数设置
        print("\n🔧 2. 分析策略参数设置...")
        strategy_module = StrategyModule(config)
        
        # 关键参数
        rise_threshold = strategy_module.rise_threshold
        max_days = strategy_module.max_days
        final_threshold = config.get('strategy', {}).get('confidence_weights', {}).get('final_threshold', 0.5)
        
        print(f"   涨幅阈值: {rise_threshold:.1%}")
        print(f"   观察天数: {max_days} 天")
        print(f"   置信度阈值: {final_threshold}")
        
        # 3. 市场环境分析
        print("\n📊 3. 市场环境分析...")
        
        # 计算不同天数下的涨幅分布
        future_returns = []
        for i in range(len(data) - max_days):
            current_price = data.iloc[i]['close']
            future_prices = data.iloc[i+1:i+max_days+1]['close']
            if len(future_prices) > 0:
                max_future_price = future_prices.max()
                max_return = (max_future_price - current_price) / current_price
                future_returns.append(max_return)
        
        future_returns = np.array(future_returns)
        
        # 理论最大成功率
        theoretical_max = np.sum(future_returns >= rise_threshold) / len(future_returns)
        print(f"   理论最大成功率: {theoretical_max:.2%}")
        print(f"   (任何算法在此配置下的绝对上限)")
        
        # 涨幅分布分析
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"   涨幅分布 ({max_days}天内):")
        for p in percentiles:
            value = np.percentile(future_returns, p)
            print(f"   {p:2d}%: {value:+.1%}")
        
        # 4. 策略识别效果分析
        print("\n🎯 4. 策略识别效果分析...")
        
        backtest_results = strategy_module.backtest(data)
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        print(f"   识别点数: {evaluation.get('total_points', 0)} 个")
        print(f"   成功点数: {evaluation.get('success_count', 0)} 个")
        print(f"   实际成功率: {evaluation.get('success_rate', 0):.2%}")
        print(f"   平均涨幅: {evaluation.get('avg_rise', 0):.2%}")
        print(f"   平均天数: {evaluation.get('avg_days', 0):.1f} 天")
        
        # 选择效率
        selection_efficiency = evaluation.get('success_rate', 0) / theoretical_max if theoretical_max > 0 else 0
        print(f"   选择效率: {selection_efficiency:.1%}")
        print(f"   (实际成功率 / 理论最大成功率)")
        
        # 5. 置信度阈值分析
        print("\n📈 5. 置信度阈值分析...")
        
        # 测试不同置信度阈值下的效果
        test_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        print("   阈值  |  识别数  |  成功数  |  成功率  |  选择效率")
        print("   -----|---------|---------|---------|----------")
        
        for threshold in test_thresholds:
            # 临时修改置信度阈值
            temp_config = config.copy()
            temp_config['strategy']['confidence_weights']['final_threshold'] = threshold
            temp_strategy = StrategyModule(temp_config)
            
            temp_backtest = temp_strategy.backtest(data)
            temp_eval = temp_strategy.evaluate_strategy(temp_backtest)
            
            temp_efficiency = temp_eval.get('success_rate', 0) / theoretical_max if theoretical_max > 0 else 0
            
            print(f"   {threshold:4.2f} | {temp_eval.get('total_points', 0):7d} | {temp_eval.get('success_count', 0):7d} | {temp_eval.get('success_rate', 0):6.1%} | {temp_efficiency:7.1%}")
        
        # 6. 技术指标分析
        print("\n📊 6. 技术指标分析...")
        
        # RSI分析
        rsi_values = data['rsi'].dropna()
        print(f"   RSI统计:")
        print(f"   平均值: {rsi_values.mean():.1f}")
        print(f"   RSI < 30: {np.sum(rsi_values < 30)} 次 ({np.sum(rsi_values < 30)/len(rsi_values)*100:.1f}%)")
        print(f"   RSI < 35: {np.sum(rsi_values < 35)} 次 ({np.sum(rsi_values < 35)/len(rsi_values)*100:.1f}%)")
        
        # 成交量分析
        volume_change = data['volume_change'].dropna()
        print(f"   成交量变化:")
        print(f"   平均值: {volume_change.mean():.2f}")
        print(f"   放量(>1.5): {np.sum(volume_change > 1.5)} 次 ({np.sum(volume_change > 1.5)/len(volume_change)*100:.1f}%)")
        print(f"   缩量(<0.8): {np.sum(volume_change < 0.8)} 次 ({np.sum(volume_change < 0.8)/len(volume_change)*100:.1f}%)")
        
        # 7. 标签质量分析
        print("\n🏷️ 7. 标签质量分析...")
        
        low_points = backtest_results[backtest_results['is_low_point']]
        if len(low_points) > 0:
            successful_points = low_points[low_points['future_max_rise'] >= rise_threshold]
            
            print(f"   标签统计:")
            print(f"   总样本数: {len(backtest_results)}")
            print(f"   正样本数: {len(low_points)}")
            print(f"   正样本比例: {len(low_points)/len(backtest_results):.2%}")
            print(f"   成功正样本: {len(successful_points)}")
            print(f"   标签准确率: {len(successful_points)/len(low_points):.2%}")
            
            if len(successful_points) > 0:
                avg_rise = successful_points['future_max_rise'].mean()
                avg_days = successful_points['days_to_rise'].mean()
                print(f"   成功案例平均涨幅: {avg_rise:.2%}")
                print(f"   成功案例平均天数: {avg_days:.1f} 天")
        
        # 8. 问题诊断
        print("\n💡 8. 问题诊断...")
        
        issues = []
        recommendations = []
        
        # 检查理论上限
        if theoretical_max < 0.4:
            issues.append(f"市场环境限制：{max_days}天内{rise_threshold:.1%}涨幅的理论最大成功率仅{theoretical_max:.1%}")
            recommendations.append("考虑降低涨幅阈值到3%或2.5%")
            recommendations.append("考虑延长观察天数到25-30天")
        
        # 检查选择效率
        if selection_efficiency < 0.5:
            issues.append(f"策略选择效率低：仅达到理论最大值的{selection_efficiency:.1%}")
            recommendations.append("优化置信度阈值，提高信号识别精度")
            recommendations.append("增强技术指标权重配置")
        
        # 检查置信度阈值
        if final_threshold > 0.25:
            issues.append(f"置信度阈值可能过高：{final_threshold}")
            recommendations.append("尝试降低置信度阈值到0.15-0.25")
        
        # 检查正样本比例
        positive_ratio = len(low_points)/len(backtest_results)
        if positive_ratio < 0.05:
            issues.append(f"正样本比例过低：{positive_ratio:.2%}")
            recommendations.append("调整策略参数增加信号识别数量")
        
        print("   🔴 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n   💡 优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # 9. 快速优化方案
        print("\n🚀 9. 快速优化方案...")
        
        print("   方案A：降低涨幅阈值")
        print("   - 涨幅阈值: 4% → 3%")
        print("   - 预期提升: 成功率提升5-10%")
        
        print("   方案B：优化置信度阈值")
        print("   - 置信度阈值: 0.37 → 0.25")
        print("   - 预期提升: 识别点数增加20-30%")
        
        print("   方案C：延长观察天数")
        print("   - 观察天数: 20天 → 25天")
        print("   - 预期提升: 理论上限提升3-5%")
        
        print("   方案D：组合优化")
        print("   - 涨幅阈值: 4% → 3.5%")
        print("   - 置信度阈值: 0.37 → 0.25")
        print("   - 观察天数: 20天 → 25天")
        print("   - 预期提升: 成功率提升10-20%")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_low_success_rate() 