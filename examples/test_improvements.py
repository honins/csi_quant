#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试AI优化器改进效果
比较原版和改进版在置信度稳定性和预测准确性方面的差异
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.utils import load_config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_stability_comparison():
    """
    测试置信度稳定性比较
    """
    setup_logging()
    logger = logging.getLogger("StabilityTest")
    
    print("=== AI优化器改进效果测试 ===")
    print()
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config_improved_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
    
    config_original = load_config(config_path=config_path)
    config_improved = load_config(config_path=config_improved_path)
    
    # 初始化模块
    data_module = DataModule(config_original)
    strategy_module = StrategyModule(config_original)
    
    # 初始化AI优化器
    ai_original = AIOptimizer(config_original)
    ai_improved = AIOptimizerImproved(config_improved)
    
    # 测试日期范围（包含6-23到6-24的关键变化）
    test_dates = [
        '2025-06-20', '2025-06-21', '2025-06-23', 
        '2025-06-24', '2025-06-25', '2025-06-26', '2025-06-27'
    ]
    
    results_original = []
    results_improved = []
    
    print("📊 开始滚动测试...")
    
    for i, date_str in enumerate(test_dates):
        print(f"\n--- 测试日期: {date_str} ({i+1}/{len(test_dates)}) ---")
        
        # 获取训练数据（当前日期之前的历史数据）
        predict_date = datetime.strptime(date_str, '%Y-%m-%d')
        start_date_for_training = predict_date - timedelta(days=1000)
        
        training_data = data_module.get_history_data(
            start_date=start_date_for_training.strftime('%Y-%m-%d'),
            end_date=date_str
        )
        
        if training_data.empty:
            print(f"❌ {date_str}: 无训练数据")
            continue
            
        training_data = data_module.preprocess_data(training_data)
        print(f"✅ 获取训练数据: {len(training_data)} 条记录")
        
        # 原版AI测试
        try:
            if i == 0:  # 第一次完全训练
                train_result_orig = ai_original.train_model(training_data, strategy_module)
            else:  # 后续完全重训练
                train_result_orig = ai_original.train_model(training_data, strategy_module)
                
            pred_result_orig = ai_original.predict_low_point(training_data.iloc[-1:])
            
            results_original.append({
                'date': date_str,
                'confidence': pred_result_orig.get('confidence', 0.0),
                'prediction': pred_result_orig.get('is_low_point', False),
                'method': 'full_retrain'
            })
            print(f"🔵 原版置信度: {pred_result_orig.get('confidence', 0.0):.4f}")
            
        except Exception as e:
            print(f"❌ 原版AI预测失败: {e}")
            results_original.append({
                'date': date_str,
                'confidence': 0.0,
                'prediction': False,
                'method': 'failed'
            })
        
        # 改进版AI测试
        try:
            if i == 0:  # 第一次完全训练
                train_result_imp = ai_improved.full_train(training_data, strategy_module)
            else:  # 后续增量训练
                train_result_imp = ai_improved.incremental_train(training_data, strategy_module)
                
            pred_result_imp = ai_improved.predict_low_point(training_data.iloc[-1:], date_str)
            
            results_improved.append({
                'date': date_str,
                'confidence': pred_result_imp.get('confidence', 0.0),
                'smoothed_confidence': pred_result_imp.get('smoothed_confidence', 0.0),
                'prediction': pred_result_imp.get('is_low_point', False),
                'method': train_result_imp.get('method', 'unknown')
            })
            print(f"🟢 改进版原始置信度: {pred_result_imp.get('confidence', 0.0):.4f}")
            print(f"🟢 改进版平滑置信度: {pred_result_imp.get('smoothed_confidence', 0.0):.4f}")
            print(f"🔧 训练方法: {train_result_imp.get('method', 'unknown')}")
            
        except Exception as e:
            print(f"❌ 改进版AI预测失败: {e}")
            results_improved.append({
                'date': date_str,
                'confidence': 0.0,
                'smoothed_confidence': 0.0,
                'prediction': False,
                'method': 'failed'
            })
    
    # 分析结果
    print("\n" + "="*60)
    print("📈 结果分析")
    print("="*60)
    
    # 转换为DataFrame便于分析
    df_original = pd.DataFrame(results_original)
    df_improved = pd.DataFrame(results_improved)
    
    print("\n1. 置信度变化分析:")
    print("原版置信度变化:")
    for i in range(len(df_original)):
        row = df_original.iloc[i]
        if i > 0:
            prev_conf = df_original.iloc[i-1]['confidence']
            change = row['confidence'] - prev_conf
            print(f"  {row['date']}: {row['confidence']:.4f} (变化: {change:+.4f})")
        else:
            print(f"  {row['date']}: {row['confidence']:.4f} (基准)")
    
    print("\n改进版置信度变化:")
    for i in range(len(df_improved)):
        row = df_improved.iloc[i]
        if i > 0:
            prev_conf = df_improved.iloc[i-1]['smoothed_confidence']
            change = row['smoothed_confidence'] - prev_conf
            print(f"  {row['date']}: 原始={row['confidence']:.4f}, 平滑={row['smoothed_confidence']:.4f} (变化: {change:+.4f}) [{row['method']}]")
        else:
            print(f"  {row['date']}: 原始={row['confidence']:.4f}, 平滑={row['smoothed_confidence']:.4f} (基准) [{row['method']}]")
    
    # 计算稳定性指标
    orig_confidences = df_original['confidence'].values
    imp_raw_confidences = df_improved['confidence'].values  
    imp_smooth_confidences = df_improved['smoothed_confidence'].values
    
    # 计算变化幅度
    orig_changes = np.abs(np.diff(orig_confidences))
    imp_raw_changes = np.abs(np.diff(imp_raw_confidences))
    imp_smooth_changes = np.abs(np.diff(imp_smooth_confidences))
    
    print(f"\n2. 稳定性指标:")
    print(f"原版平均变化幅度: {np.mean(orig_changes):.4f}")
    print(f"原版最大变化幅度: {np.max(orig_changes):.4f}")
    print(f"改进版原始平均变化幅度: {np.mean(imp_raw_changes):.4f}")
    print(f"改进版平滑平均变化幅度: {np.mean(imp_smooth_changes):.4f}")
    print(f"改进版平滑最大变化幅度: {np.max(imp_smooth_changes):.4f}")
    
    # 计算6-23到6-24的关键变化
    idx_623 = df_original[df_original['date'] == '2025-06-23'].index
    idx_624 = df_original[df_original['date'] == '2025-06-24'].index
    
    if len(idx_623) > 0 and len(idx_624) > 0:
        orig_623 = df_original.loc[idx_623[0], 'confidence']
        orig_624 = df_original.loc[idx_624[0], 'confidence']
        orig_key_change = abs(orig_624 - orig_623)
        
        imp_623 = df_improved.loc[idx_623[0], 'smoothed_confidence']
        imp_624 = df_improved.loc[idx_624[0], 'smoothed_confidence'] 
        imp_key_change = abs(imp_624 - imp_623)
        
        print(f"\n3. 关键日期(6-23到6-24)变化:")
        print(f"原版: {orig_623:.4f} → {orig_624:.4f} (变化: {orig_key_change:.4f})")
        print(f"改进版: {imp_623:.4f} → {imp_624:.4f} (变化: {imp_key_change:.4f})")
        print(f"稳定性改善: {((orig_key_change - imp_key_change) / orig_key_change * 100):.1f}%")
    
    # 绘制对比图
    plot_comparison(df_original, df_improved)
    
    return df_original, df_improved

def plot_comparison(df_original, df_improved):
    """绘制对比图"""
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    dates = df_original['date'].values
    x_pos = range(len(dates))
    
    # 子图1: 置信度对比
    ax1.plot(x_pos, df_original['confidence'], 'o-', label='原版置信度', color='red', linewidth=2)
    ax1.plot(x_pos, df_improved['confidence'], 's-', label='改进版原始置信度', color='orange', linewidth=2)
    ax1.plot(x_pos, df_improved['smoothed_confidence'], '^-', label='改进版平滑置信度', color='green', linewidth=2)
    
    ax1.set_title('置信度对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('置信度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dates, rotation=45)
    
    # 标记关键变化点
    if '2025-06-23' in dates and '2025-06-24' in dates:
        idx_623 = list(dates).index('2025-06-23')
        idx_624 = list(dates).index('2025-06-24') 
        ax1.axvspan(idx_623-0.5, idx_624+0.5, alpha=0.2, color='yellow', label='关键变化期')
    
    # 子图2: 变化幅度对比
    orig_changes = [0] + list(np.abs(np.diff(df_original['confidence'])))
    imp_smooth_changes = [0] + list(np.abs(np.diff(df_improved['smoothed_confidence'])))
    
    ax2.bar([x-0.2 for x in x_pos], orig_changes, width=0.4, label='原版变化幅度', color='red', alpha=0.7)
    ax2.bar([x+0.2 for x in x_pos], imp_smooth_changes, width=0.4, label='改进版变化幅度', color='green', alpha=0.7)
    
    ax2.set_title('置信度变化幅度对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('变化幅度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dates, rotation=45)
    
    # 子图3: 累计改善效果
    orig_volatility = np.cumsum(orig_changes)
    imp_volatility = np.cumsum(imp_smooth_changes)
    improvement = orig_volatility - imp_volatility
    
    ax3.plot(x_pos, improvement, 'o-', label='累计稳定性改善', color='blue', linewidth=2)
    ax3.fill_between(x_pos, 0, improvement, alpha=0.3, color='blue')
    
    ax3.set_title('累计稳定性改善效果', fontsize=14, fontweight='bold')
    ax3.set_ylabel('改善幅度')
    ax3.set_xlabel('日期')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dates, rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'results/ai_improvement_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图表已保存: {plot_path}")
    
    plt.show()

def print_improvement_summary():
    """打印改进措施总结"""
    print("\n" + "="*60)
    print("🚀 AI优化器改进措施总结")
    print("="*60)
    
    print("""
💡 主要改进措施:

1. 📈 置信度平滑机制
   • EMA指数移动平均平滑 (α=0.3)
   • 最大日变化限制 (±0.25)
   • 自适应平滑强度调整

2. 🔄 增量学习机制  
   • 避免每日完全重训练
   • 使用warm_start增量更新
   • 智能触发完全重训练条件

3. ⚖️ 特征权重优化
   • 长期趋势指标权重 +50%~100%
   • 短期技术指标权重 -30%~50%  
   • 平衡中期指标权重

4. 📊 趋势确认指标
   • 趋势强度指标 (线性回归斜率)
   • 价格位置指标 (在均线系统中的位置)
   • 标准化波动率和成交量趋势

5. 🎯 模型参数优化
   • 增加树的数量 (100→150)
   • 调整树的深度和样本要求
   • 启用并行训练加速

🎯 预期效果:
   • 降低置信度异常波动
   • 提高模型在关键转折点的稳定性
   • 减少追涨杀跌的错误信号
   • 提升长期预测准确性
""")

if __name__ == "__main__":
    try:
        # 运行测试
        df_orig, df_imp = test_stability_comparison()
        
        # 打印改进总结
        print_improvement_summary()
        
        print("\n✅ 测试完成！请查看生成的对比图表和分析结果。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 