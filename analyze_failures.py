#!/usr/bin/env python3
"""
分析错判样本脚本
使用failure_analysis模块分析最新回测中的错判样本，输出改进建议
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai.failure_analysis import FailureAnalyzer
from strategy.strategy_module import StrategyModule
from data.data_module import DataModule

def load_latest_backtest_results():
    """加载最新的回测结果"""
    csv_dir = "results/csv"
    if not os.path.exists(csv_dir):
        print(f"❌ CSV目录不存在: {csv_dir}")
        return None
    
    # 找到最新的CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("daily_details_rolling_backtest_") and f.endswith(".csv")]
    if not csv_files:
        print("❌ 未找到回测CSV文件")
        return None
    
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(csv_dir, latest_csv)
    
    print(f"📄 加载最新回测结果: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功加载 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ 加载CSV文件失败: {e}")
        return None

def analyze_failed_predictions(backtest_df, data_df, config):
    """分析错判样本"""
    print("\n🔍 开始分析错判样本...")
    
    # 初始化失败分析器
    failure_analyzer = FailureAnalyzer(config)
    
    # 准备回测结果格式
    backtest_results = []
    for _, row in backtest_df.iterrows():
        result = {
            'date': row['日期'],
            'is_low_point': row['预测低点'] == 'True',  # 使用'True'字符串比较
            'prediction': row['预测低点'] == 'True',
            'confidence': row['置信度'],
            'strategy_confidence': row['策略置信度'],
            'actual_return': 0,  # CSV中没有这个字段，设为0
            'future_max_rise': row['未来最大涨幅'],
            'is_correct': row['预测正确'] == 'True',  # 使用'True'字符串比较
            'exit_date': '',  # CSV中没有这个字段
            'exit_price': 0,  # CSV中没有这个字段
            'trade_return': 0,  # CSV中没有这个字段
            'price': row['预测价格'],  # 使用预测价格作为信号价格
            'close': row['预测价格']  # 同时提供close字段
        }
        backtest_results.append(result)
    
    # 调试信息：打印样本统计
    df_results = pd.DataFrame(backtest_results)
    total_predictions = len(df_results)
    low_point_predictions = df_results['is_low_point'].sum()
    correct_predictions = df_results['is_correct'].sum()
    incorrect_predictions = total_predictions - correct_predictions
    
    print(f"   总预测数: {total_predictions}")
    print(f"   预测为低点数: {low_point_predictions}")
    print(f"   预测正确数: {correct_predictions}")
    print(f"   预测错误数: {incorrect_predictions}")
    
    # 检查rise_threshold
    rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.04)
    print(f"   涨幅阈值: {rise_threshold}")
    
    # 改进的失败案例检测逻辑：
    # 1. 预测为低点但未来涨幅未达到阈值（传统定义）
    # 2. 预测错误的案例（更广泛的失败定义）
    traditional_failed_cases = df_results[
        (df_results['is_low_point'] == True) & 
        (df_results['future_max_rise'] < rise_threshold)
    ]
    
    # 预测错误的案例（包括预测为低点但实际不是，以及预测不是低点但实际是）
    prediction_failed_cases = df_results[df_results['is_correct'] == False]
    
    print(f"   传统失败案例数（预测低点但涨幅不足）: {len(traditional_failed_cases)}")
    print(f"   预测错误案例数: {len(prediction_failed_cases)}")
    
    # 选择分析的失败案例
    if len(traditional_failed_cases) > 0:
        # 如果有传统失败案例，优先分析这些
        failed_cases_to_analyze = traditional_failed_cases
        print(f"   ✅ 使用传统失败案例进行分析: {len(failed_cases_to_analyze)} 个")
    elif len(prediction_failed_cases) > 0:
        # 否则分析预测错误的案例
        failed_cases_to_analyze = prediction_failed_cases.head(10)  # 限制数量避免过多
        print(f"   ✅ 使用预测错误案例进行分析: {len(failed_cases_to_analyze)} 个")
        
        # 为这些案例添加失败标记，便于failure_analyzer处理
        for idx in failed_cases_to_analyze.index:
            backtest_results[idx]['analysis_type'] = 'prediction_error'
            # 如果未来涨幅达到阈值但预测错误，可能是其他类型的错误
            if backtest_results[idx]['future_max_rise'] >= rise_threshold:
                backtest_results[idx]['analysis_type'] = 'false_negative'  # 漏判
            else:
                backtest_results[idx]['analysis_type'] = 'false_positive'  # 误判
    else:
        print("   ⚠️ 未检测到任何失败案例")
        return {
            'total_failures': 0,
            'failure_rate': 0.0,
            'failure_types': {},
            'detailed_analysis': [],
            'recommendations': []
        }
    
    # 执行失败分析
    failure_analysis = failure_analyzer.analyze_failures(backtest_results, data_df)
    
    return failure_analysis

def print_failure_analysis(failure_analysis):
    """打印失败分析结果"""
    print("\n" + "="*60)
    print("📊 错判样本分析报告")
    print("="*60)
    
    # 基本统计
    total_failures = failure_analysis.get('total_failures', 0)
    failure_rate = failure_analysis.get('failure_rate', 0)
    
    print(f"\n📈 基本统计:")
    print(f"   总错判数: {total_failures}")
    print(f"   错判率: {failure_rate:.2%}")
    
    # 失败类型分布
    failure_types = failure_analysis.get('failure_types', {})
    if failure_types:
        print(f"\n🏷️ 失败类型分布:")
        for failure_type, info in failure_types.items():
            count = info.get('count', 0)
            percentage = info.get('percentage', 0)
            print(f"   {failure_type}: {count}次 ({percentage:.1%})")
    
    # 详细分析
    detailed_analysis = failure_analysis.get('detailed_analysis', [])
    if detailed_analysis:
        print(f"\n🔍 详细失败案例分析 (前5个):")
        for i, analysis in enumerate(detailed_analysis[:5]):
            print(f"\n   案例 {i+1}:")
            print(f"     日期: {analysis.get('date', 'N/A')}")
            print(f"     失败类型: {analysis.get('failure_type', 'N/A')}")
            print(f"     置信度: {analysis.get('confidence', 0):.3f}")
            print(f"     策略置信度: {analysis.get('strategy_confidence', 0):.3f}")
            print(f"     实际涨幅: {analysis.get('actual_rise', 0):.2%}")
            print(f"     分析原因: {analysis.get('analysis', 'N/A')}")
    
    # 改进建议
    recommendations = failure_analysis.get('recommendations', [])
    if recommendations:
        print(f"\n💡 改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # 预期改进效果
    expected_improvements = failure_analysis.get('expected_improvements', {})
    if expected_improvements:
        print(f"\n📈 预期改进效果:")
        for metric, improvement in expected_improvements.items():
            if isinstance(improvement, (int, float)):
                print(f"   {metric}: +{improvement:.2%}")
            else:
                print(f"   {metric}: {improvement}")

def main():
    """主函数"""
    print("🚀 启动错判样本分析...")
    
    # 加载配置
    try:
        from utils.config_loader import load_config
        config = load_config()
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 加载最新回测结果
    backtest_df = load_latest_backtest_results()
    if backtest_df is None:
        return
    
    # 加载历史数据
    try:
        data_module = DataModule(config)
        # 获取回测日期范围
        start_date = backtest_df['日期'].min()
        end_date = backtest_df['日期'].max()
        print(f"📅 回测日期范围: {start_date} ~ {end_date}")
        
        # 加载更大范围的数据以便分析
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)
        extended_start = start_dt.strftime('%Y-%m-%d')
        
        data_df = data_module.get_history_data(extended_start, end_date)
        print(f"✅ 成功加载历史数据 {len(data_df)} 条记录")
    except Exception as e:
        print(f"❌ 历史数据加载失败: {e}")
        return
    
    # 分析错判样本
    try:
        failure_analysis = analyze_failed_predictions(backtest_df, data_df, config)
        
        # 打印分析结果
        print_failure_analysis(failure_analysis)
        
        # 保存分析结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/failure_analysis_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failure_analysis, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 错判分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()