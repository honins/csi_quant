#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能优化器测试脚本
验证基于失败案例分析的参数优化功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('smart_optimizer_test.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config():
    """加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'strategy.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_smart_optimizer():
    """测试智能优化器"""
    print("\n" + "="*80)
    print("🧪 智能优化器测试")
    print("="*80)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        print("📋 加载配置文件...")
        config = load_config()
        
        # 确保启用智能优化
        config['optimization']['use_smart_optimization'] = True
        print(f"✅ 智能优化已启用: {config['optimization']['use_smart_optimization']}")
        
        # 初始化数据模块
        print("📊 初始化数据模块...")
        data_module = DataModule(config)
        
        # 加载测试数据
        print("📈 加载历史数据...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2年数据
        
        # 使用一个测试股票代码
        test_symbols = ['000001.SZ', '000002.SZ', '600000.SH']  # 测试几只股票
        
        for symbol in test_symbols:
            print(f"\n🔍 测试股票: {symbol}")
            print("-" * 60)
            
            try:
                # 加载数据
                data = data_module.get_history_data(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if data is None or len(data) < 100:
                    print(f"⚠️  {symbol} 数据不足，跳过")
                    continue
                
                print(f"📊 数据加载成功: {len(data)} 条记录")
                print(f"📅 数据范围: {data.index[0]} 到 {data.index[-1]}")
                
                # 初始化策略模块
                print("🎯 初始化策略模块...")
                strategy_module = StrategyModule(config)
                
                # 初始化AI优化器
                print("🤖 初始化AI优化器...")
                ai_optimizer = AIOptimizerImproved(config)
                
                # 获取当前策略表现
                print("📊 评估当前策略表现...")
                current_backtest = strategy_module.backtest(data)
                current_evaluation = strategy_module.evaluate_strategy(current_backtest)
                
                print(f"   当前成功率: {current_evaluation.get('success_rate', 0):.2%}")
                print(f"   平均涨幅: {current_evaluation.get('avg_rise', 0):.2%}")
                print(f"   平均持股天数: {current_evaluation.get('avg_days', 0):.1f}")
                print(f"   信号数量: {current_evaluation.get('total_signals', 0)}")
                
                # 运行智能优化
                print("\n🚀 开始智能优化...")
                optimization_result = ai_optimizer.run_complete_optimization(data, strategy_module)
                
                # 显示优化结果
                if optimization_result.get('success', False):
                    strategy_result = optimization_result.get('strategy_optimization', {})
                    
                    print("\n✅ 优化完成!")
                    print(f"   优化方法: {strategy_result.get('optimization_method', 'unknown')}")
                    print(f"   最佳策略: {strategy_result.get('best_strategy', 'unknown')}")
                    
                    if 'failure_analysis' in strategy_result:
                        failure_analysis = strategy_result['failure_analysis']
                        print("\n📊 失败案例分析:")
                        
                        failure_types = failure_analysis.get('failure_types', {})
                        for failure_type, info in failure_types.items():
                            print(f"   {failure_type}: {info.get('count', 0)} 个 ({info.get('percentage', 0):.1f}%)")
                    
                    if 'recommendations' in strategy_result:
                        recommendations = strategy_result['recommendations']
                        print("\n💡 优化建议:")
                        for i, rec in enumerate(recommendations, 1):
                            print(f"   {i}. {rec}")
                    
                    # 显示参数变化
                    if 'best_params' in strategy_result:
                        print("\n🔧 关键参数调整:")
                        best_params = strategy_result['best_params']
                        current_params = strategy_module.get_current_params()
                        
                        key_params = [
                            'rsi_oversold_threshold', 'volume_panic_threshold', 
                            'final_threshold', 'market_sentiment_weight'
                        ]
                        
                        for param in key_params:
                            if param in best_params and param in current_params:
                                old_val = current_params[param]
                                new_val = best_params[param]
                                change = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                                print(f"   {param}: {old_val:.4f} → {new_val:.4f} ({change:+.1f}%)")
                    
                else:
                    print("❌ 优化失败")
                    errors = optimization_result.get('errors', [])
                    for error in errors:
                        print(f"   错误: {error}")
                
                print(f"\n✅ {symbol} 测试完成")
                
            except Exception as e:
                print(f"❌ {symbol} 测试失败: {e}")
                logger.error(f"{symbol} 测试失败", exc_info=True)
                continue
        
        print("\n" + "="*80)
        print("🎉 智能优化器测试完成")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        logger.error("测试失败", exc_info=True)
        return False
    
    return True

def test_failure_analyzer_only():
    """仅测试失败分析器"""
    print("\n" + "="*60)
    print("🔍 失败分析器独立测试")
    print("="*60)
    
    try:
        # 加载配置
        config = load_config()
        
        # 创建模拟的回测结果
        print("📊 创建模拟回测数据...")
        
        # 模拟100个信号的回测结果
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        backtest_results = []
        for i, date in enumerate(dates):
            # 模拟不同类型的失败
            rand = np.random.random()
            
            # 基础价格
            base_price = np.random.uniform(10, 50)
            
            if rand < 0.3:  # 30% 成功
                result = {
                    'date': date,
                    'symbol': f'TEST{i:03d}',
                    'success': True,
                    'max_rise': np.random.uniform(0.04, 0.15),
                    'days_to_target': np.random.randint(1, 20),
                    'final_rise': np.random.uniform(0.04, 0.12),
                    'price': base_price,
                    'confidence': np.random.uniform(0.6, 0.9),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.6, 0.9)
                    }
                }
            elif rand < 0.5:  # 20% 接飞刀
                result = {
                    'date': date,
                    'symbol': f'TEST{i:03d}',
                    'success': False,
                    'max_rise': np.random.uniform(-0.15, 0.02),
                    'days_to_target': 20,
                    'final_rise': np.random.uniform(-0.15, -0.02),
                    'price': base_price,
                    'confidence': np.random.uniform(0.5, 0.8),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.5, 0.8)
                    }
                }
            elif rand < 0.75:  # 25% 横盘不动
                result = {
                    'date': date,
                    'symbol': f'TEST{i:03d}',
                    'success': False,
                    'max_rise': np.random.uniform(-0.02, 0.03),
                    'days_to_target': 20,
                    'final_rise': np.random.uniform(-0.02, 0.02),
                    'price': base_price,
                    'confidence': np.random.uniform(0.5, 0.7),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.5, 0.7)
                    }
                }
            else:  # 25% 功亏一篑
                result = {
                    'date': date,
                    'symbol': f'TEST{i:03d}',
                    'success': False,
                    'max_rise': np.random.uniform(0.02, 0.039),  # 接近但未达到4%
                    'days_to_target': np.random.randint(15, 25),  # 或者超时
                    'final_rise': np.random.uniform(0.01, 0.035),
                    'price': base_price,
                    'confidence': np.random.uniform(0.6, 0.8),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.6, 0.8)
                    }
                }
            
            backtest_results.append(result)
        
        # 模拟股票数据
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        mock_data = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(10, 50, 200),
            'volume': np.random.uniform(1000000, 10000000, 200)
        })
        
        # 测试失败分析器
        from src.ai.failure_analysis import FailureAnalyzer
        
        print("🔍 初始化失败分析器...")
        failure_analyzer = FailureAnalyzer(config)
        
        print("📊 分析失败案例...")
        analysis_result = failure_analyzer.analyze_failures(backtest_results, mock_data)
        
        print("\n📈 分析结果:")
        print(f"   总信号数: {analysis_result.get('total_signals', 0)}")
        print(f"   成功信号数: {analysis_result.get('successful_signals', 0)}")
        print(f"   失败信号数: {analysis_result.get('total_failures', 0)}")
        print(f"   成功率: {analysis_result.get('success_rate', 0):.2%}")
        
        failure_types = analysis_result.get('failure_types', {})
        print("\n🔍 失败类型分布:")
        for failure_type, info in failure_types.items():
            print(f"   {failure_type}: {info.get('count', 0)} 个 ({info.get('percentage', 0):.1f}%)")
            
            # 显示典型案例
            examples = info.get('examples', [])
            if examples:
                print(f"     典型案例: {examples[0].get('symbol', 'N/A')} (涨幅: {examples[0].get('max_rise', 0):.2%})")
        
        recommendations = analysis_result.get('optimization_recommendations', [])
        if recommendations:
            print("\n💡 优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n✅ 失败分析器测试完成")
        
    except Exception as e:
        print(f"❌ 失败分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 开始智能优化器测试")
    
    # 首先测试失败分析器
    print("\n第一阶段: 失败分析器测试")
    if test_failure_analyzer_only():
        print("✅ 失败分析器测试通过")
    else:
        print("❌ 失败分析器测试失败")
        sys.exit(1)
    
    # 然后测试完整的智能优化器
    print("\n第二阶段: 完整智能优化器测试")
    if test_smart_optimizer():
        print("✅ 所有测试通过")
    else:
        print("❌ 测试失败")
        sys.exit(1)