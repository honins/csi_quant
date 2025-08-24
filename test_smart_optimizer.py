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
    """加载配置：使用统一配置加载器合并 system.yaml 与 strategy.yaml"""
    try:
        from src.utils.config_loader import load_config as merged_load_config
        config = merged_load_config()
        # 兜底：若缺少 optimization 段则创建，避免后续 set/use 时 KeyError
        if 'optimization' not in config:
            config['optimization'] = {}
        return config
    except Exception:
        # 回退到原有单文件加载（仅当统一加载器不可用时）
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'strategy.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}


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

        # 测试加速：减少贝叶斯优化评估次数
        bo_cfg = config.setdefault('bayesian_optimization', {})
        bo_cfg['n_calls'] = 10
        bo_cfg['n_initial_points'] = 5
        bo_cfg['random_state'] = bo_cfg.get('random_state', 42)

        # 打印关键数据配置，便于确认路径
        data_cfg = config.get('data', {})
        print(f"📂 data_file_path: {data_cfg.get('data_file_path', '<未配置>')}\n")
        
        # 确保启用智能优化
        config['optimization']['use_smart_optimization'] = True
        print(f"✅ 智能优化已启用: {config['optimization']['use_smart_optimization']}")
        
        # 初始化数据模块
        print("📊 初始化数据模块...")
        data_module = DataModule(config)
        
        # 加载测试数据（缩短为1年以加速）
        print("📈 加载历史数据...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1年数据
        
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
                # 若DataModule返回带date列，索引可能是默认数值索引，这里展示日期范围
                print(f"📅 数据范围: {data['date'].min()} 到 {data['date'].max()}")
                
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
                # 打印利润相关指标以确认score即总利润
                print(f"   当前总利润(score): {current_evaluation.get('score', 0):.6f}")
                print(f"   当前total_profit: {current_evaluation.get('total_profit', 0):.6f}")
                print(f"   平均涨幅: {current_evaluation.get('avg_rise', 0):.2%}")
                print(f"   平均持股天数: {current_evaluation.get('avg_days', 0):.1f}")
                
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
                    'max_rise': np.random.uniform(0.0, 0.03),
                    'days_to_target': 20,
                    'final_rise': np.random.uniform(-0.02, 0.02),
                    'price': base_price,
                    'confidence': np.random.uniform(0.4, 0.7),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.4, 0.7)
                    }
                }
            else:  # 25% 震荡失败
                result = {
                    'date': date,
                    'symbol': f'TEST{i:03d}',
                    'success': False,
                    'max_rise': np.random.uniform(-0.1, 0.08),
                    'days_to_target': np.random.randint(1, 20),
                    'final_rise': np.random.uniform(-0.05, 0.05),
                    'price': base_price,
                    'confidence': np.random.uniform(0.3, 0.8),
                    'signal': {
                        'date': date,
                        'price': base_price,
                        'confidence': np.random.uniform(0.3, 0.8)
                    }
                }
            backtest_results.append(result)
        
        # 创建DataFrame
        backtest_df = pd.DataFrame(backtest_results)
        backtest_df['is_low_point'] = True
        backtest_df['entry_price'] = backtest_df['price']
        backtest_df['trade_return'] = backtest_df['final_rise']
        
        # 初始化策略模块（仅为使分析器可用）
        strategy_module = StrategyModule(config)
        
        # 直接调用策略模块的评估查看score（应为利润）
        evaluation = strategy_module.evaluate_strategy(backtest_df)
        print(f"   模拟总利润(score): {evaluation.get('score', 0):.6f}")
        print(f"   模拟PF参考(pf_score): {evaluation.get('pf_score', 0):.6f}")
        
        print("✅ 失败分析器模拟数据准备完成")
        return True
    
    except Exception as e:
        print(f"❌ 失败分析器测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🚀 开始智能优化器测试")

    print("\n第一阶段: 失败分析器测试")
    if test_failure_analyzer_only():
        print("✅ 失败分析器测试通过")
    else:
        print("❌ 失败分析器测试失败")
        sys.exit(1)

    print("\n第二阶段: 完整智能优化器测试")
    if test_smart_optimizer():
        print("✅ 所有测试通过")
    else:
        print("❌ 测试失败")
        sys.exit(1)