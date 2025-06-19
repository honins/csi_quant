#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级优化演示脚本
演示分层优化、时间序列交叉验证等高级功能
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.utils import setup_logging, load_config, Timer
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from ai.ai_optimizer import AIOptimizer

def demo_basic_optimization(data_module, strategy_module, ai_optimizer, data):
    """演示基础优化"""
    print("\n" + "="*50)
    print("🔧 基础参数优化演示")
    print("="*50)
    
    timer = Timer()
    timer.start()
    
    # 基础优化
    optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, data)
    
    timer.stop()
    print(f"✅ 基础优化完成 (耗时: {timer.elapsed_str()})")
    print(f"   - 优化参数: {optimized_params}")
    
    # 测试优化效果
    strategy_module.update_params(optimized_params)
    backtest_results = strategy_module.backtest(data)
    evaluation = strategy_module.evaluate_strategy(backtest_results)
    
    print(f"   - 优化后得分: {evaluation['score']:.4f}")
    print(f"   - 成功率: {evaluation['success_rate']:.2%}")
    
    return optimized_params, evaluation

def demo_advanced_optimization(data_module, strategy_module, ai_optimizer, data):
    """演示高级优化"""
    print("\n" + "="*50)
    print("🚀 高级优化演示")
    print("="*50)
    
    timer = Timer()
    timer.start()
    
    # 高级优化
    advanced_params = ai_optimizer.optimize_strategy_parameters_advanced(strategy_module, data)
    
    timer.stop()
    print(f"✅ 高级优化完成 (耗时: {timer.elapsed_str()})")
    print(f"   - 优化参数: {advanced_params}")
    
    # 测试优化效果
    strategy_module.update_params(advanced_params)
    backtest_results = strategy_module.backtest(data)
    evaluation = strategy_module.evaluate_strategy(backtest_results)
    
    print(f"   - 优化后得分: {evaluation['score']:.4f}")
    print(f"   - 成功率: {evaluation['success_rate']:.2%}")
    
    return advanced_params, evaluation

def demo_time_series_cv(data_module, strategy_module, ai_optimizer, data):
    """演示时间序列交叉验证"""
    print("\n" + "="*50)
    print("⏰ 时间序列交叉验证演示")
    print("="*50)
    
    timer = Timer()
    timer.start()
    
    # 时间序列交叉验证
    cv_score = ai_optimizer.time_series_cv_evaluation(data, strategy_module)
    
    timer.stop()
    print(f"✅ 时间序列交叉验证完成 (耗时: {timer.elapsed_str()})")
    print(f"   - 平均得分: {cv_score:.4f}")
    
    return cv_score

def demo_hierarchical_optimization(data_module, strategy_module, ai_optimizer, data):
    """演示分层优化"""
    print("\n" + "="*50)
    print("🏗️ 分层优化演示")
    print("="*50)
    
    timer = Timer()
    timer.start()
    
    # 分层优化
    result = ai_optimizer.hierarchical_optimization(data)
    
    timer.stop()
    print(f"✅ 分层优化完成 (耗时: {timer.elapsed_str()})")
    print(f"   - 最终参数: {result['strategy_params']}")
    print(f"   - 交叉验证得分: {result['cv_score']:.4f}")
    print(f"   - 最终得分: {result['final_score']:.4f}")
    print(f"   - 优化方法: {result['optimization_method']}")
    
    # 测试最终效果
    strategy_module.update_params(result['strategy_params'])
    backtest_results = strategy_module.backtest(data)
    evaluation = strategy_module.evaluate_strategy(backtest_results)
    
    print(f"   - 实际回测得分: {evaluation['score']:.4f}")
    print(f"   - 成功率: {evaluation['success_rate']:.2%}")
    
    return result, evaluation

def demo_comparison(data_module, strategy_module, ai_optimizer, data):
    """演示优化方法对比"""
    print("\n" + "="*50)
    print("📊 优化方法对比")
    print("="*50)
    
    # 基准策略
    baseline_backtest = strategy_module.backtest(data)
    baseline_evaluation = strategy_module.evaluate_strategy(baseline_backtest)
    
    # 基础优化
    basic_params, basic_evaluation = demo_basic_optimization(data_module, strategy_module, ai_optimizer, data)
    
    # 高级优化
    advanced_params, advanced_evaluation = demo_advanced_optimization(data_module, strategy_module, ai_optimizer, data)
    
    # 分层优化
    hierarchical_result, hierarchical_evaluation = demo_hierarchical_optimization(data_module, strategy_module, ai_optimizer, data)
    
    # 对比结果
    print("\n" + "="*50)
    print("🏆 最终对比结果")
    print("="*50)
    
    methods = [
        ("基准策略", baseline_evaluation['score'], baseline_evaluation['success_rate']),
        ("基础优化", basic_evaluation['score'], basic_evaluation['success_rate']),
        ("高级优化", advanced_evaluation['score'], advanced_evaluation['success_rate']),
        ("分层优化", hierarchical_evaluation['score'], hierarchical_evaluation['success_rate'])
    ]
    
    print(f"{'方法':<12} {'得分':<10} {'成功率':<10} {'改进幅度':<12}")
    print("-" * 50)
    
    baseline_score = baseline_evaluation['score']
    for name, score, success_rate in methods:
        improvement = (score - baseline_score) / baseline_score * 100
        print(f"{name:<12} {score:<10.4f} {success_rate:<10.2%} {improvement:<+12.2f}%")
    
    # 找出最佳方法
    best_method = max(methods, key=lambda x: x[1])
    print(f"\n🏆 最佳方法: {best_method[0]} (得分: {best_method[1]:.4f})")

def main():
    """主函数"""
    print("="*60)
    print("中证1000指数相对低点识别系统 - 高级优化演示")
    print("="*60)
    
    # 设置日志
    setup_logging('INFO')
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    if not config:
        print("❌ 配置文件加载失败")
        return False
    
    try:
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        
        # 获取数据
        print("\n📊 准备数据...")
        start_date = '2022-01-01'
        end_date = '2025-06-19'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 运行各种优化演示
        demo_comparison(data_module, strategy_module, ai_optimizer, processed_data)
        
        print("\n🎉 高级优化演示完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 