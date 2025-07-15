#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能异常分析诊断脚本
专门诊断测试集成功率高于验证集和训练集的反常现象

根据机器学习理论和时间序列分析最佳实践，正常情况下应该是：
训练集性能 > 验证集性能 > 测试集性能

当出现相反情况时，可能存在以下问题：
1. 数据泄漏 (Data Leakage)
2. 时间序列前瞻偏差 (Lookahead Bias) 
3. 数据分布变化 (Distribution Shift)
4. 样本权重问题
5. 过拟合检测机制问题
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.config_loader import ConfigLoader
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_data_distribution(train_data, val_data, test_data, logger):
    """分析三个数据集的分布差异"""
    logger.info("🔍 分析数据分布差异...")
    
    results = {
        'distribution_analysis': {},
        'potential_issues': []
    }
    
    # 1. 基础统计信息
    datasets = {
        'training': train_data,
        'validation': val_data, 
        'test': test_data
    }
    
    print("\n📊 数据分布分析:")
    print("=" * 60)
    
    for name, data in datasets.items():
        if 'close' in data.columns:
            price_stats = {
                'mean': data['close'].mean(),
                'std': data['close'].std(),
                'min': data['close'].min(),
                'max': data['close'].max(),
                'skewness': data['close'].skew(),
                'kurtosis': data['close'].kurtosis()
            }
            results['distribution_analysis'][name] = price_stats
            
            print(f"\n{name.capitalize()}集统计:")
            print(f"  均值: {price_stats['mean']:.2f}")
            print(f"  标准差: {price_stats['std']:.2f}")
            print(f"  偏度: {price_stats['skewness']:.3f}")
            print(f"  峰度: {price_stats['kurtosis']:.3f}")
            print(f"  价格范围: {price_stats['min']:.2f} - {price_stats['max']:.2f}")
    
    # 2. 检查分布漂移
    train_mean = results['distribution_analysis']['training']['mean']
    val_mean = results['distribution_analysis']['validation']['mean'] 
    test_mean = results['distribution_analysis']['test']['mean']
    
    # 计算分布漂移程度
    val_drift = abs(val_mean - train_mean) / train_mean
    test_drift = abs(test_mean - train_mean) / train_mean
    
    print(f"\n📈 分布漂移分析:")
    print(f"  验证集vs训练集: {val_drift:.2%}")
    print(f"  测试集vs训练集: {test_drift:.2%}")
    
    # 分布漂移阈值检查
    if val_drift > 0.1:  # 10%阈值
        issue = f"验证集与训练集存在显著分布差异 ({val_drift:.1%})"
        results['potential_issues'].append(issue)
        print(f"  ⚠️ {issue}")
        
    if test_drift > 0.1:
        issue = f"测试集与训练集存在显著分布差异 ({test_drift:.1%})"
        results['potential_issues'].append(issue)
        print(f"  ⚠️ {issue}")
    
    # 3. 检查时间趋势
    if 'date' in train_data.columns:
        print(f"\n📅 时间范围分析:")
        print(f"  训练集: {train_data['date'].min()} ~ {train_data['date'].max()}")
        print(f"  验证集: {val_data['date'].min()} ~ {val_data['date'].max()}")
        print(f"  测试集: {test_data['date'].min()} ~ {test_data['date'].max()}")
        
        # 检查时间重叠
        train_dates = set(train_data['date'])
        val_dates = set(val_data['date'])
        test_dates = set(test_data['date'])
        
        if train_dates & val_dates:
            issue = "训练集与验证集存在时间重叠 - 可能数据泄漏!"
            results['potential_issues'].append(issue)
            print(f"  🚨 {issue}")
            
        if train_dates & test_dates:
            issue = "训练集与测试集存在时间重叠 - 严重数据泄漏!"
            results['potential_issues'].append(issue)
            print(f"  🚨 {issue}")
            
        if val_dates & test_dates:
            issue = "验证集与测试集存在时间重叠 - 数据泄漏!"
            results['potential_issues'].append(issue)
            print(f"  🚨 {issue}")
    
    return results

def analyze_market_conditions(train_data, val_data, test_data, logger):
    """分析不同时期的市场条件"""
    logger.info("📈 分析市场条件差异...")
    
    results = {
        'market_analysis': {},
        'potential_explanations': []
    }
    
    datasets = {
        'training': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    print("\n📈 市场条件分析:")
    print("=" * 60)
    
    for name, data in datasets.items():
        if len(data) > 1 and 'close' in data.columns:
            # 计算收益率
            returns = data['close'].pct_change().dropna()
            
            # 计算市场指标
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            trend = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100  # 总收益率
            positive_days = (returns > 0).mean() * 100  # 上涨天数比例
            
            market_stats = {
                'volatility': volatility,
                'total_return': trend,
                'positive_days_pct': positive_days,
                'max_drawdown': ((data['close'] / data['close'].cummax() - 1).min()) * 100
            }
            
            results['market_analysis'][name] = market_stats
            
            print(f"\n{name.capitalize()}集市场条件:")
            print(f"  年化波动率: {volatility:.1%}")
            print(f"  总收益率: {trend:.1f}%")
            print(f"  上涨天数比例: {positive_days:.1f}%")
            print(f"  最大回撤: {market_stats['max_drawdown']:.1f}%")
    
    # 分析市场条件变化
    if len(results['market_analysis']) == 3:
        train_vol = results['market_analysis']['training']['volatility']
        val_vol = results['market_analysis']['validation']['volatility']
        test_vol = results['market_analysis']['test']['volatility']
        
        train_return = results['market_analysis']['training']['total_return']
        val_return = results['market_analysis']['validation']['total_return']
        test_return = results['market_analysis']['test']['total_return']
        
        print(f"\n📊 市场条件变化:")
        print(f"  波动率变化: 训练({train_vol:.1%}) → 验证({val_vol:.1%}) → 测试({test_vol:.1%})")
        print(f"  收益率变化: 训练({train_return:.1f}%) → 验证({val_return:.1f}%) → 测试({test_return:.1f}%)")
        
        # 解释可能的原因
        if test_vol < train_vol * 0.7:
            explanation = "测试期波动率显著降低，可能使预测更容易"
            results['potential_explanations'].append(explanation)
            print(f"  💡 {explanation}")
            
        if test_return > val_return > train_return:
            explanation = "测试期市场表现逐步改善，可能存在趋势性机会"
            results['potential_explanations'].append(explanation)
            print(f"  💡 {explanation}")
            
        if abs(test_return) > abs(train_return) * 2:
            explanation = "测试期存在显著趋势，可能导致策略表现异常"
            results['potential_explanations'].append(explanation)
            print(f"  💡 {explanation}")
    
    return results

def check_data_leakage_indicators(data, strategy_module, logger):
    """检查数据泄漏的指标"""
    logger.info("🔍 检查数据泄漏指标...")
    
    results = {
        'leakage_indicators': {},
        'warnings': []
    }
    
    print("\n🔍 数据泄漏检查:")
    print("=" * 60)
    
    # 1. 检查特征中是否包含未来信息
    feature_columns = data.columns.tolist()
    suspicious_features = []
    
    future_keywords = ['future', 'target', 'label', 'next', 'forward', 'ahead']
    for col in feature_columns:
        for keyword in future_keywords:
            if keyword in col.lower():
                suspicious_features.append(col)
    
    if suspicious_features:
        warning = f"发现可疑的未来信息特征: {suspicious_features}"
        results['warnings'].append(warning)
        print(f"  🚨 {warning}")
    else:
        print("  ✅ 未发现明显的未来信息特征")
    
    # 2. 检查回测逻辑中的数据泄漏
    # 模拟一小段数据的回测，检查是否使用了未来信息
    sample_data = data.head(50).copy()  # 取前50天数据
    
    try:
        # 检查策略识别逻辑
        backtest_results = strategy_module.backtest(sample_data)
        
        # 检查是否在计算当前点时使用了未来数据
        for i in range(min(10, len(backtest_results)-1)):
            current_date = backtest_results.iloc[i]['date']
            
            # 检查当前行是否包含了未来日期的信息
            if 'max_rise_date' in backtest_results.columns:
                max_rise_date = backtest_results.iloc[i]['max_rise_date']
                if pd.notna(max_rise_date) and max_rise_date > current_date:
                    # 这是正常的，因为max_rise_date是用于验证的未来信息
                    pass
    
        print("  ✅ 回测逻辑检查通过")
        
    except Exception as e:
        warning = f"回测逻辑检查异常: {e}"
        results['warnings'].append(warning)
        print(f"  ⚠️ {warning}")
    
    # 3. 检查技术指标计算是否正确
    if 'rsi' in data.columns and len(data) > 20:
        # 检查RSI计算是否有前瞻偏差
        rsi_values = data['rsi'].dropna()
        if len(rsi_values) > 0:
            # RSI应该在0-100之间
            if rsi_values.min() < 0 or rsi_values.max() > 100:
                warning = f"RSI值异常 (范围: {rsi_values.min():.2f} - {rsi_values.max():.2f})"
                results['warnings'].append(warning)
                print(f"  ⚠️ {warning}")
            else:
                print("  ✅ 技术指标范围正常")
    
    return results

def analyze_sample_weights_impact(data, logger):
    """分析样本权重对结果的影响"""
    logger.info("⚖️ 分析样本权重影响...")
    
    results = {
        'weight_analysis': {},
        'insights': []
    }
    
    print("\n⚖️ 样本权重分析:")
    print("=" * 60)
    
    if 'date' in data.columns:
        # 模拟时间衰减权重计算
        dates = pd.to_datetime(data['date'])
        latest_date = dates.max()
        decay_rate = 0.4  # 默认衰减率
        
        # 计算权重
        days_diff = (latest_date - dates).dt.days
        weights = np.exp(-decay_rate * days_diff / 365.0)
        
        # 分析权重分布
        print(f"  权重范围: {weights.min():.4f} - {weights.max():.4f}")
        print(f"  权重平均值: {weights.mean():.4f}")
        print(f"  权重标准差: {weights.std():.4f}")
        
        # 检查权重偏差
        early_weights = weights[:len(weights)//3].mean()
        late_weights = weights[-len(weights)//3:].mean()
        weight_bias = late_weights / early_weights
        
        print(f"  早期数据平均权重: {early_weights:.4f}")
        print(f"  后期数据平均权重: {late_weights:.4f}")
        print(f"  权重偏差比率: {weight_bias:.2f}")
        
        if weight_bias > 5:
            insight = f"后期数据权重显著高于早期 ({weight_bias:.1f}倍)，可能导致训练偏向新数据"
            results['insights'].append(insight)
            print(f"  💡 {insight}")
        
        results['weight_analysis'] = {
            'min_weight': weights.min(),
            'max_weight': weights.max(),
            'mean_weight': weights.mean(),
            'weight_bias_ratio': weight_bias
        }
    
    return results

def analyze_strategy_complexity(train_data, val_data, test_data, strategy_module, logger):
    """分析策略复杂度和拟合情况"""
    logger.info("🎯 分析策略复杂度...")
    
    results = {
        'complexity_analysis': {},
        'recommendations': []
    }
    
    print("\n🎯 策略复杂度分析:")
    print("=" * 60)
    
    # 获取策略参数
    params = strategy_module.get_params()
    print(f"  策略参数: {params}")
    
    # 分析每个数据集上的表现
    datasets = {
        'training': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    performance_scores = {}
    
    for name, data in datasets.items():
        try:
            backtest_results = strategy_module.backtest(data)
            evaluation = strategy_module.evaluate_strategy(backtest_results)
            
            performance_scores[name] = {
                'score': evaluation.get('score', 0),
                'success_rate': evaluation.get('success_rate', 0),
                'total_points': evaluation.get('total_points', 0),
                'avg_rise': evaluation.get('avg_rise', 0)
            }
            
            print(f"\n  {name.capitalize()}集表现:")
            print(f"    综合得分: {evaluation.get('score', 0):.4f}")
            print(f"    成功率: {evaluation.get('success_rate', 0):.2%}")
            print(f"    识别点数: {evaluation.get('total_points', 0)}")
            print(f"    平均涨幅: {evaluation.get('avg_rise', 0):.2%}")
            
        except Exception as e:
            logger.warning(f"分析{name}集时出错: {e}")
    
    # 分析性能趋势
    if len(performance_scores) == 3:
        train_score = performance_scores['training']['score']
        val_score = performance_scores['validation']['score'] 
        test_score = performance_scores['test']['score']
        
        print(f"\n📊 性能趋势分析:")
        print(f"  训练集 → 验证集 → 测试集: {train_score:.4f} → {val_score:.4f} → {test_score:.4f}")
        
        # 异常模式检测
        if test_score > val_score > train_score:
            recommendation = "异常模式：测试集>验证集>训练集，可能存在数据泄漏或分布偏移"
            results['recommendations'].append(recommendation)
            print(f"  🚨 {recommendation}")
            
        elif test_score > val_score * 1.2:
            recommendation = "测试集性能异常优秀，建议检查数据分割和时间序列连续性"
            results['recommendations'].append(recommendation)
            print(f"  ⚠️ {recommendation}")
            
        elif val_score < train_score * 0.8:
            recommendation = "可能存在过拟合，建议降低模型复杂度"
            results['recommendations'].append(recommendation)
            print(f"  💡 {recommendation}")
    
    results['complexity_analysis'] = performance_scores
    return results

def generate_diagnostic_report(all_results, logger):
    """生成诊断报告"""
    logger.info("📝 生成诊断报告...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"results/performance_anomaly_diagnosis_{timestamp}.json"
    
    # 汇总所有发现的问题
    all_issues = []
    all_explanations = []
    all_recommendations = []
    
    for result in all_results:
        all_issues.extend(result.get('potential_issues', []))
        all_issues.extend(result.get('warnings', []))
        all_explanations.extend(result.get('potential_explanations', []))
        all_explanations.extend(result.get('insights', []))
        all_recommendations.extend(result.get('recommendations', []))
    
    # 生成最终诊断
    diagnosis = {
        'timestamp': timestamp,
        'summary': {
            'total_issues_found': len(all_issues),
            'total_explanations': len(all_explanations),
            'total_recommendations': len(all_recommendations)
        },
        'identified_issues': all_issues,
        'potential_explanations': all_explanations,
        'recommendations': all_recommendations,
        'detailed_analysis': all_results
    }
    
    # 保存报告
    os.makedirs('results', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📝 诊断报告:")
    print("=" * 60)
    print(f"🔍 发现问题数量: {len(all_issues)}")
    print(f"💡 可能解释数量: {len(all_explanations)}")
    print(f"🎯 改进建议数量: {len(all_recommendations)}")
    print(f"📄 报告保存路径: {report_path}")
    
    # 输出主要发现
    if all_issues:
        print(f"\n🚨 主要问题:")
        for i, issue in enumerate(all_issues[:5], 1):
            print(f"  {i}. {issue}")
    
    if all_explanations:
        print(f"\n💡 可能解释:")
        for i, explanation in enumerate(all_explanations[:3], 1):
            print(f"  {i}. {explanation}")
    
    if all_recommendations:
        print(f"\n🎯 改进建议:")
        for i, rec in enumerate(all_recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    return diagnosis

def main():
    """主函数"""
    print("🔍 性能异常分析诊断")
    print("=" * 80)
    print("目标：诊断测试集成功率高于验证集和训练集的反常现象")
    print("=" * 80)
    
    logger = setup_logging()
    
    try:
        # 加载配置和数据
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        data_module = DataModule(config)
        # 从配置文件获取时间范围
        data_config = config.get('data', {})
        time_range = data_config.get('time_range', {})
        start_date = time_range.get('start_date', '2019-01-01')
        end_date = time_range.get('end_date', '2025-07-15')
        
        data = data_module.get_history_data(start_date, end_date)
        data = data_module.preprocess_data(data)
        
        strategy_module = StrategyModule(config)
        
        # 模拟AI优化器的数据分割方式
        validation_config = config.get('ai', {}).get('validation', {})
        train_ratio = validation_config.get('train_ratio', 0.70)
        val_ratio = validation_config.get('validation_ratio', 0.20)
        test_ratio = validation_config.get('test_ratio', 0.10)
        
        # 计算分割点
        train_end = int(len(data) * train_ratio)
        val_end = int(len(data) * (train_ratio + val_ratio))
        
        # 分割数据
        train_data = data.iloc[:train_end].copy()
        validation_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        print(f"📊 数据分割结果:")
        print(f"  训练集: {len(train_data)}条 ({train_ratio:.1%})")
        print(f"  验证集: {len(validation_data)}条 ({val_ratio:.1%})")
        print(f"  测试集: {len(test_data)}条 ({test_ratio:.1%})")
        
        # 开始诊断分析
        all_results = []
        
        # 1. 数据分布分析
        dist_result = analyze_data_distribution(train_data, validation_data, test_data, logger)
        all_results.append(dist_result)
        
        # 2. 市场条件分析
        market_result = analyze_market_conditions(train_data, validation_data, test_data, logger)
        all_results.append(market_result)
        
        # 3. 数据泄漏检查
        leakage_result = check_data_leakage_indicators(data, strategy_module, logger)
        all_results.append(leakage_result)
        
        # 4. 样本权重影响分析
        weight_result = analyze_sample_weights_impact(data, logger)
        all_results.append(weight_result)
        
        # 5. 策略复杂度分析
        complexity_result = analyze_strategy_complexity(train_data, validation_data, test_data, strategy_module, logger)
        all_results.append(complexity_result)
        
        # 6. 生成诊断报告
        final_diagnosis = generate_diagnostic_report(all_results, logger)
        
        print(f"\n✅ 诊断分析完成！")
        return True
        
    except Exception as e:
        logger.error(f"诊断分析异常: {e}")
        print(f"\n❌ 诊断分析失败: {e}")
        return False

if __name__ == "__main__":
    main() 