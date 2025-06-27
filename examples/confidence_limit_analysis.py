#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
置信度限制分析脚本
分析不同最大变化限制设置的效果
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ai.ai_optimizer_improved import ConfidenceSmoother


def create_test_config(max_daily_change: float, dynamic_enabled: bool = True) -> dict:
    """创建测试配置"""
    return {
        'ai': {
            'confidence_smoothing': {
                'enabled': True,
                'ema_alpha': 0.3,
                'max_daily_change': max_daily_change,
                'dynamic_adjustment': {
                    'enabled': dynamic_enabled,
                    'min_limit': 0.15,
                    'max_limit': 0.50,
                    'volatility_factor': {
                        'enabled': True,
                        'max_multiplier': 2.0,
                        'min_multiplier': 0.5
                    },
                    'price_factor': {
                        'enabled': True,
                        'sensitivity': 10,
                        'max_multiplier': 2.0
                    },
                    'volume_factor': {
                        'enabled': True,
                        'panic_threshold': 1.5,
                        'low_threshold': 0.7,
                        'max_multiplier': 1.8
                    },
                    'confidence_factor': {
                        'enabled': True,
                        'large_change_threshold': 0.5,
                        'max_multiplier': 1.5
                    }
                },
                'debug_mode': False,
                'log_adjustments': False
            }
        }
    }


def create_market_scenarios():
    """创建不同的市场场景"""
    scenarios = {}
    
    # 正常市场
    normal_data = pd.DataFrame({
        'close': [100, 101, 102, 101.5, 102.5],
        'volume': [1000, 1100, 1050, 1000, 1200],
        'volatility': [0.02, 0.025, 0.02, 0.022, 0.024]
    })
    scenarios['正常市场'] = normal_data
    
    # 高波动市场
    volatile_data = pd.DataFrame({
        'close': [100, 95, 105, 90, 110],
        'volume': [1000, 2000, 2500, 3000, 1800],
        'volatility': [0.05, 0.08, 0.10, 0.12, 0.09]
    })
    scenarios['高波动市场'] = volatile_data
    
    # 恐慌市场
    panic_data = pd.DataFrame({
        'close': [100, 92, 85, 80, 88],
        'volume': [1000, 5000, 8000, 6000, 3000],
        'volatility': [0.08, 0.15, 0.20, 0.18, 0.12]
    })
    scenarios['恐慌市场'] = panic_data
    
    # 用户案例：6-23到6-24的情况
    user_case_data = pd.DataFrame({
        'close': [5800, 5674.17, 5765.84],  # 模拟用户案例
        'volume': [500000, 450000, 572000],  # 成交量放大
        'volatility': [0.015, 0.025, 0.018]  # 波动性变化
    })
    scenarios['用户案例(6-23到6-24)'] = user_case_data
    
    return scenarios


def simulate_confidence_changes(scenarios: dict, configs: dict) -> dict:
    """模拟不同配置下的置信度变化"""
    results = {}
    
    for scenario_name, market_data in scenarios.items():
        results[scenario_name] = {}
        
        for config_name, config in configs.items():
            smoother = ConfidenceSmoother(config)
            
            # 模拟置信度序列
            raw_confidences = []
            smoothed_confidences = []
            
            # 模拟从高置信度到低置信度的变化（用户案例）
            if scenario_name == '用户案例(6-23到6-24)':
                raw_sequence = [1.0, 0.12]  # 用户的实际情况
            else:
                # 其他场景的模拟
                raw_sequence = [0.8, 0.2, 0.7, 0.1, 0.9]
            
            for i, raw_conf in enumerate(raw_sequence):
                date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                
                # 为每次平滑提供相应的市场数据
                if i < len(market_data):
                    current_market_data = market_data.iloc[:i+1] if i > 0 else market_data.iloc[:1]
                else:
                    current_market_data = market_data
                
                smoothed = smoother.smooth_confidence(raw_conf, date, current_market_data)
                
                raw_confidences.append(raw_conf)
                smoothed_confidences.append(smoothed)
            
            results[scenario_name][config_name] = {
                'raw': raw_confidences,
                'smoothed': smoothed_confidences,
                'changes': [abs(smoothed_confidences[i] - smoothed_confidences[i-1]) 
                           for i in range(1, len(smoothed_confidences))]
            }
    
    return results


def analyze_results(results: dict):
    """分析结果"""
    print("=" * 80)
    print("置信度变化限制分析报告")
    print("=" * 80)
    
    for scenario_name, scenario_results in results.items():
        print(f"\n📊 {scenario_name}:")
        print("-" * 50)
        
        for config_name, data in scenario_results.items():
            raw = data['raw']
            smoothed = data['smoothed']
            changes = data['changes']
            
            if len(changes) > 0:
                max_change = max(changes)
                avg_change = np.mean(changes)
                
                print(f"\n  {config_name}:")
                print(f"    原始置信度: {raw}")
                print(f"    平滑置信度: {[f'{x:.3f}' for x in smoothed]}")
                print(f"    最大日变化: {max_change:.3f}")
                print(f"    平均日变化: {avg_change:.3f}")
                
                # 特别分析用户案例
                if scenario_name == '用户案例(6-23到6-24)' and len(smoothed) >= 2:
                    original_change = abs(raw[1] - raw[0])  # 原始变化 0.88
                    smoothed_change = abs(smoothed[1] - smoothed[0])  # 平滑后变化
                    reduction = (1 - smoothed_change/original_change) * 100
                    print(f"    🎯 原始变化: {original_change:.3f} → 平滑变化: {smoothed_change:.3f}")
                    print(f"    📉 变化减少: {reduction:.1f}%")
    
    print("\n" + "=" * 80)
    print("分析总结:")
    print("=" * 80)
    
    # 建议不同的配置
    print("\n💡 配置建议:")
    print("1. 保守型 (±0.20): 适合稳定的量化策略，减少噪音干扰")
    print("2. 平衡型 (±0.25): 当前默认设置，平衡稳定性和响应性")
    print("3. 灵敏型 (±0.35): 适合需要快速响应市场变化的策略")
    print("4. 动态型 (±0.25+动态): 根据市场情况自动调整，推荐使用")


def plot_comparison(results: dict):
    """绘制对比图"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (scenario_name, scenario_results) in enumerate(results.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        for config_name, data in scenario_results.items():
            days = range(len(data['smoothed']))
            ax.plot(days, data['smoothed'], marker='o', label=config_name, linewidth=2)
        
        ax.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('交易日')
        ax.set_ylabel('置信度')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('confidence_limit_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 对比图已保存: confidence_limit_analysis.png")


def main():
    """主函数"""
    print("开始置信度变化限制分析...")
    
    # 创建不同的配置
    configs = {
        '保守型(±0.20)': create_test_config(0.20, dynamic_enabled=False),
        '当前设置(±0.25)': create_test_config(0.25, dynamic_enabled=False),
        '灵敏型(±0.35)': create_test_config(0.35, dynamic_enabled=False),
        '动态调整(±0.25+)': create_test_config(0.25, dynamic_enabled=True),
    }
    
    # 创建市场场景
    scenarios = create_market_scenarios()
    
    # 运行模拟
    results = simulate_confidence_changes(scenarios, configs)
    
    # 分析结果
    analyze_results(results)
    
    # 绘制对比图
    try:
        plot_comparison(results)
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n✅ 分析完成！")
    
    # 给出具体建议
    print("\n" + "🔧 配置建议:")
    print("-" * 50)
    print("基于分析结果，建议配置如下：")
    print()
    print("1. 如果您希望更快响应市场变化，可以将 max_daily_change 调整为 0.35-0.40")
    print("2. 如果您希望保持当前的稳定性，建议启用动态调整功能")
    print("3. 动态调整可以在正常情况下使用较小限制，在市场异常时自动放宽")
    print()
    print("修改配置文件 config/config_improved.yaml:")
    print("  max_daily_change: 0.35  # 或您希望的值")
    print("  dynamic_adjustment:")
    print("    enabled: true  # 启用动态调整")


if __name__ == "__main__":
    main() 