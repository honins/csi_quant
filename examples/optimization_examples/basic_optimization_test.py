#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础优化测试示例
演示如何使用重构后的AI优化功能
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.utils import setup_logging, load_config, Timer
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from ai.ai_optimizer_refactored import AIOptimizerRefactored


def test_basic_optimization():
    """测试基础优化功能"""
    print("🔧 测试基础优化功能...")
    
    # 设置日志
    setup_logging('INFO')
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    try:
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerRefactored(config)
        
        # 准备数据
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        
        print(f"数据准备完成: {len(processed_data)} 条记录")
        
        # 基础策略测试
        print("\n1. 基础策略测试...")
        timer = Timer()
        timer.start()
        
        backtest_results = strategy_module.backtest(processed_data)
        baseline_evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        timer.stop()
        print(f"基础策略得分: {baseline_evaluation['score']:.4f} (耗时: {timer.elapsed_str()})")
        
        # 参数优化测试
        print("\n2. 参数优化测试...")
        timer.start()
        
        optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
        
        timer.stop()
        print(f"参数优化完成 (耗时: {timer.elapsed_str()})")
        print(f"优化后参数: {optimized_params}")
        
        # 测试优化效果
        strategy_module.update_params(optimized_params)
        optimized_backtest = strategy_module.backtest(processed_data)
        optimized_evaluation = strategy_module.evaluate_strategy(optimized_backtest)
        
        print(f"优化后得分: {optimized_evaluation['score']:.4f}")
        
        # 计算改进幅度
        improvement = (optimized_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        print(f"改进幅度: {improvement:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_validation():
    """测试数据验证功能"""
    print("🔒 测试数据验证功能...")
    
    # 设置日志
    setup_logging('INFO')
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    try:
        # 初始化模块
        data_module = DataModule(config)
        ai_optimizer = AIOptimizerRefactored(config)
        
        # 准备数据
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        
        # 测试数据分割
        print("\n数据分割测试...")
        data_splits = ai_optimizer.strict_data_split(processed_data)
        
        print(f"训练集: {len(data_splits['train'])} 条")
        print(f"验证集: {len(data_splits['validation'])} 条")
        print(f"测试集: {len(data_splits['test'])} 条")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("基础优化测试示例")
    print("="*60)
    
    success = True
    
    # 测试1: 基础优化
    print("\n📋 测试1: 基础优化功能")
    success &= test_basic_optimization()
    
    # 测试2: 数据验证
    print("\n📋 测试2: 数据验证功能")
    success &= test_data_validation()
    
    print("\n" + "="*60)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败。")
    print("="*60)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 