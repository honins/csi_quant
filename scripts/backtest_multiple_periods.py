#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多周期回测对比脚本
自动运行 1, 3, 6, 12 个月的回测，并生成对比表格。
"""

import sys
import os
import pandas as pd
from datetime import datetime
import warnings

# 添加项目根目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from examples.run_rolling_backtest import run_rolling_backtest

# 忽略部分警告以保持输出整洁
warnings.filterwarnings('ignore')

def main():
    # 定义回测周期（月）
    periods = [1, 3, 6, 12]
    
    # 结束日期设为今天
    end_date = datetime.now()
    
    results = []
    
    print(f"\n🚀 开始多周期回测对比任务")
    print(f"📅 结束基准日期: {end_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    for months in periods:
        # 计算开始日期
        start_date = end_date - pd.DateOffset(months=months)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\nRunning backtest for {months} month(s) period...")
        print(f"Range: {start_date_str} -> {end_date_str}")
        
        try:
            # 调用滚动回测函数
            # reuse_model=True: 使用已有模型，不重新训练
            # generate_report=False: 不生成详细 Markdown 报告，只获取数据
            res = run_rolling_backtest(
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                training_window_days=365,
                reuse_model=True,
                generate_report=False 
            )
            
            if res.get('success'):
                m = res['metrics']
                results.append({
                    'Period': f"{months} Month(s)",
                    'Win Rate': m.get('win_rate', 0.0),
                    'Avg Return': m.get('avg_return', 0.0),
                    'Total Return': m.get('total_return', 0.0),
                    'Max Drawdown': m.get('max_drawdown', 0.0),
                    'Trades': m.get('total_trades', 0),
                    'Start Date': start_date_str
                })
                print(f"✅ 完成: Win Rate={m.get('win_rate', 0):.1%}, Total Return={m.get('total_return', 0):.1%}")
            else:
                print(f"❌ 失败: {res.get('error')}")
                
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 输出对比表格
    if results:
        df = pd.DataFrame(results)
        
        # 格式化百分比
        format_cols = ['Win Rate', 'Avg Return', 'Total Return', 'Max Drawdown']
        for col in format_cols:
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
            
        print("\n" + "="*80)
        print("📊 多周期回测对比结果 (Comparative Backtest Results)")
        print("="*80)
        
        # 使用 to_string 输出表格
        print(df.to_string(index=False))
        print("="*80)
        
        # 保存 CSV 结果
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"multi_period_summary_{end_date.strftime('%Y%m%d')}.csv")
        df.to_csv(output_path, index=False)
        print(f"\n📝 汇总结果已保存至: {output_path}")
    else:
        print("\n⚠️ 未生成任何有效回测结果。")

if __name__ == "__main__":
    main()
