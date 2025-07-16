#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号识别诊断测试
专门用于诊断策略为什么只识别出少量信号点的问题
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.config_loader import load_config
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_confidence_distribution(data, strategy_module, logger):
    """
    分析置信度分布情况
    """
    logger.info("分析置信度分布...")
    
    confidences = []
    signal_days = []
    
    # 遍历所有交易日，分析每日的置信度
    for i in range(50, len(data)):  # 从第50天开始，确保有足够的历史数据
        historical_data = data.iloc[:i+1].copy()
        result = strategy_module.identify_relative_low(historical_data)
        
        confidences.append(result.get('confidence', 0.0))
        signal_days.append({
            'date': result.get('date'),
            'confidence': result.get('confidence', 0.0),
            'is_low_point': result.get('is_low_point', False),
            'reasons': result.get('reasons', [])
        })
    
    # 统计分析
    confidences = np.array(confidences)
    logger.info(f"置信度统计:")
    logger.info(f"  平均值: {confidences.mean():.4f}")
    logger.info(f"  最大值: {confidences.max():.4f}")
    logger.info(f"  最小值: {confidences.min():.4f}")
    logger.info(f"  标准差: {confidences.std():.4f}")
    logger.info(f"  中位数: {np.median(confidences):.4f}")
    
    # 分析不同置信度阈值下的信号数量
    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
    for threshold in thresholds:
        signal_count = np.sum(confidences >= threshold)
        percentage = signal_count / len(confidences) * 100
        logger.info(f"  置信度 >= {threshold}: {signal_count} 个信号 ({percentage:.1f}%)")
    
    return signal_days

def analyze_technical_indicators(data, logger):
    """
    分析技术指标的分布情况
    """
    logger.info("分析技术指标分布...")
    
    # RSI分析
    rsi_values = data['rsi'].dropna()
    logger.info(f"RSI统计:")
    logger.info(f"  平均值: {rsi_values.mean():.2f}")
    logger.info(f"  最小值: {rsi_values.min():.2f}")
    logger.info(f"  最大值: {rsi_values.max():.2f}")
    logger.info(f"  RSI < 30: {np.sum(rsi_values < 30)} 次 ({np.sum(rsi_values < 30)/len(rsi_values)*100:.1f}%)")
    logger.info(f"  RSI < 35: {np.sum(rsi_values < 35)} 次 ({np.sum(rsi_values < 35)/len(rsi_values)*100:.1f}%)")
    logger.info(f"  RSI < 40: {np.sum(rsi_values < 40)} 次 ({np.sum(rsi_values < 40)/len(rsi_values)*100:.1f}%)")
    
    # 移动平均线分析
    logger.info(f"移动平均线分析:")
    price_below_ma5 = np.sum(data['close'] < data['ma5']) / len(data) * 100
    price_below_ma10 = np.sum(data['close'] < data['ma10']) / len(data) * 100
    price_below_ma20 = np.sum(data['close'] < data['ma20']) / len(data) * 100
    price_below_all_ma = np.sum((data['close'] < data['ma5']) & 
                               (data['close'] < data['ma10']) & 
                               (data['close'] < data['ma20'])) / len(data) * 100
    
    logger.info(f"  价格低于MA5: {price_below_ma5:.1f}%")
    logger.info(f"  价格低于MA10: {price_below_ma10:.1f}%")
    logger.info(f"  价格低于MA20: {price_below_ma20:.1f}%")
    logger.info(f"  价格低于所有均线: {price_below_all_ma:.1f}%")
    
    # 布林带分析
    if 'bb_lower' in data.columns:
        price_near_bb_lower = np.sum(data['close'] <= data['bb_lower'] * 1.02) / len(data) * 100
        logger.info(f"  价格接近布林带下轨: {price_near_bb_lower:.1f}%")

def test_different_thresholds(data, config, logger):
    """
    测试不同置信度阈值下的信号数量
    """
    logger.info("测试不同置信度阈值...")
    
    results = {}
    
    # 测试不同的final_threshold值
    test_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    for threshold in test_thresholds:
        # 创建临时配置
        temp_config = config.copy()
        
        # 确保配置结构存在
        if 'strategy' not in temp_config:
            temp_config['strategy'] = {}
        if 'confidence_weights' not in temp_config['strategy']:
            temp_config['strategy']['confidence_weights'] = {}
            
        temp_config['strategy']['confidence_weights']['final_threshold'] = threshold
        
        # 创建策略模块
        strategy_module = StrategyModule(temp_config)
        
        # 运行回测
        backtest_results = strategy_module.backtest(data)
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        results[threshold] = {
            'total_points': evaluation['total_points'],
            'success_rate': evaluation['success_rate'],
            'avg_rise': evaluation['avg_rise'],
            'score': evaluation['score']
        }
        
        logger.info(f"  阈值 {threshold}: {evaluation['total_points']} 个信号, "
                   f"成功率 {evaluation['success_rate']:.1%}, "
                   f"平均涨幅 {evaluation['avg_rise']:.1%}, "
                   f"得分 {evaluation['score']:.4f}")
    
    return results

def analyze_strategy_logic(data, config, logger):
    """
    分析策略逻辑的各个组成部分
    """
    logger.info("分析策略逻辑各组成部分...")
    
    strategy_module = StrategyModule(config)
    
    # 分析最后几天的具体逻辑
    sample_days = min(10, len(data))
    
    for i in range(len(data) - sample_days, len(data)):
        historical_data = data.iloc[:i+1].copy()
        result = strategy_module.identify_relative_low(historical_data)
        
        current_date = data.iloc[i]['date']
        logger.info(f"日期 {current_date}:")
        logger.info(f"  置信度: {result.get('confidence', 0):.4f}")
        logger.info(f"  是否低点: {result.get('is_low_point', False)}")
        logger.info(f"  理由: {result.get('reasons', [])}")
        
        # 分析技术指标
        current_data = data.iloc[i]
        logger.info(f"  技术指标:")
        logger.info(f"    RSI: {current_data.get('rsi', 'N/A')}")
        logger.info(f"    MACD: {current_data.get('macd', 'N/A')}")
        logger.info(f"    价格: {current_data['close']}")
        logger.info(f"    MA5: {current_data.get('ma5', 'N/A')}")
        logger.info(f"    MA10: {current_data.get('ma10', 'N/A')}")
        logger.info(f"    MA20: {current_data.get('ma20', 'N/A')}")
        
        if 'volume_ratio' in data.columns:
            logger.info(f"    成交量比率: {current_data.get('volume_ratio', 'N/A')}")

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始信号识别诊断测试")
    
    try:
        # 加载配置
        config = load_config()
        
        # 获取数据
        data_module = DataModule(config)
        backtest_config = config.get('backtest', {})
        start_date = backtest_config.get('start_date', '2022-01-01')
        end_date = backtest_config.get('end_date', '2024-12-31')
        
        data = data_module.get_history_data(start_date, end_date)
        
        if data is None or data.empty:
            logger.error("无法获取数据")
            return False
        
        # 🔧 关键修复：对数据进行预处理，计算技术指标
        logger.info("对数据进行预处理，计算技术指标...")
        data = data_module.preprocess_data(data)
        logger.info(f"预处理完成，数据列: {list(data.columns)}")
        
        logger.info(f"数据总长度: {len(data)} 天")
        logger.info(f"数据时间范围: {data['date'].min()} 到 {data['date'].max()}")
        
        # 1. 分析当前配置下的信号数量
        logger.info("="*50)
        logger.info("1. 当前配置分析")
        strategy_module = StrategyModule(config)
        backtest_results = strategy_module.backtest(data)
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        logger.info(f"当前配置下识别的信号点数: {evaluation['total_points']}")
        logger.info(f"成功率: {evaluation['success_rate']:.1%}")
        logger.info(f"平均涨幅: {evaluation['avg_rise']:.1%}")
        logger.info(f"得分: {evaluation['score']:.4f}")
        
        # 2. 分析置信度分布
        logger.info("="*50)
        logger.info("2. 置信度分布分析")
        signal_days = analyze_confidence_distribution(data, strategy_module, logger)
        
        # 3. 分析技术指标分布
        logger.info("="*50)
        logger.info("3. 技术指标分布分析")
        analyze_technical_indicators(data, logger)
        
        # 4. 测试不同阈值
        logger.info("="*50)
        logger.info("4. 不同置信度阈值测试")
        threshold_results = test_different_thresholds(data, config, logger)
        
        # 5. 分析策略逻辑
        logger.info("="*50)
        logger.info("5. 策略逻辑详细分析")
        analyze_strategy_logic(data, config, logger)
        
        # 6. 保存诊断结果
        diagnosis_result = {
            'timestamp': datetime.now().isoformat(),
            'data_length': len(data),
            'data_range': f"{data['date'].min()} 到 {data['date'].max()}",
            'current_signals': evaluation['total_points'],
            'current_success_rate': evaluation['success_rate'],
            'current_avg_rise': evaluation['avg_rise'],
            'current_score': evaluation['score'],
            'threshold_tests': threshold_results,
            'signal_details': signal_days[-10:]  # 只保存最后10天的详细信息
        }
        
        # 保存到文件
        results_dir = Path(project_root / 'results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        diagnosis_file = results_dir / f'signal_diagnosis_{timestamp}.json'
        
        with open(diagnosis_file, 'w', encoding='utf-8') as f:
            json.dump(diagnosis_result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"诊断结果保存到: {diagnosis_file}")
        
        # 7. 给出建议
        logger.info("="*50)
        logger.info("7. 诊断建议")
        
        if evaluation['total_points'] < 10:
            logger.info("🔴 问题: 信号点过少")
            logger.info("💡 建议:")
            logger.info("   1. 降低 final_threshold (当前可能过高)")
            logger.info("   2. 调整技术指标阈值 (RSI, 移动平均线等)")
            logger.info("   3. 检查数据质量和技术指标计算")
        
        # 找到最优阈值建议
        best_threshold = None
        best_score = 0
        for threshold, result in threshold_results.items():
            if result['total_points'] >= 5 and result['score'] > best_score:
                best_score = result['score']
                best_threshold = threshold
        
        if best_threshold:
            logger.info(f"🎯 推荐阈值: {best_threshold} (得分: {best_score:.4f}, 信号数: {threshold_results[best_threshold]['total_points']})")
        
        return True
        
    except Exception as e:
        logger.error(f"诊断测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 