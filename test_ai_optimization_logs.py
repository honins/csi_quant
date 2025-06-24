#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试AI优化日志功能
展示AI策略优化时的详细进度日志
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer

def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai_optimization.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config():
    """加载配置文件"""
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_basic_optimization_logs():
    """测试基础优化的日志输出"""
    print("=" * 80)
    print("🧪 测试基础AI优化日志功能")
    print("=" * 80)
    
    # 1. 加载配置
    config = load_config()
    print("✓ 配置文件加载成功")
    
    # 2. 加载数据
    data_module = DataModule(config)
    start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
    end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
    data = data_module.get_history_data(start_date, end_date)
    data = data_module.preprocess_data(data)
    print(f"✓ 数据加载成功，数据长度: {len(data)}")
    
    # 3. 初始化策略模块
    strategy_module = StrategyModule(config)
    print("✓ 策略模块初始化成功")
    
    # 4. 初始化AI优化器
    ai_optimizer = AIOptimizer(config)
    print("✓ AI优化器初始化成功")
    
    # 5. 运行基础优化（这里会显示详细的进度日志）
    print("\n" + "🚀 开始运行基础AI优化...")
    print("注意观察下面的详细进度日志输出：")
    print("-" * 80)
    
    optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, data)
    
    print("-" * 80)
    print("✅ 基础优化完成！")
    print("\n最终优化结果:")
    for key, value in optimized_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

def test_hierarchical_optimization_logs():
    """测试分层优化的日志输出"""
    print("\n" + "=" * 80)
    print("🧪 测试分层AI优化日志功能")
    print("=" * 80)
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 加载数据
    data_module = DataModule(config)
    start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
    end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
    data = data_module.get_history_data(start_date, end_date)
    data = data_module.preprocess_data(data)
    
    # 3. 初始化AI优化器
    ai_optimizer = AIOptimizer(config)
    
    # 4. 运行分层优化（这里会显示详细的进度日志）
    print("\n🚀 开始运行分层AI优化...")
    print("注意观察下面的详细进度日志输出：")
    print("-" * 80)
    
    result = ai_optimizer.hierarchical_optimization(data)
    
    print("-" * 80)
    print("✅ 分层优化完成！")
    print("\n分层优化结果:")
    print(f"  最佳得分: {result['best_score']:.4f}")
    print(f"  交叉验证得分: {result['cv_score']:.4f}")
    print(f"  高级优化得分: {result['advanced_score']:.4f}")
    print(f"  总耗时: {result['total_time']:.1f}秒")

def test_time_series_cv_logs():
    """测试时间序列交叉验证的日志输出"""
    print("\n" + "=" * 80)
    print("🧪 测试时间序列交叉验证日志功能")
    print("=" * 80)
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 加载数据
    data_module = DataModule(config)
    start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
    end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
    data = data_module.get_history_data(start_date, end_date)
    data = data_module.preprocess_data(data)
    
    # 3. 初始化策略模块和AI优化器
    strategy_module = StrategyModule(config)
    ai_optimizer = AIOptimizer(config)
    
    # 4. 运行时间序列交叉验证（这里会显示详细的进度日志）
    print("\n🚀 开始运行时间序列交叉验证...")
    print("注意观察下面的详细进度日志输出：")
    print("-" * 80)
    
    cv_score = ai_optimizer.time_series_cv_evaluation(data, strategy_module)
    
    print("-" * 80)
    print("✅ 时间序列交叉验证完成！")
    print(f"最终交叉验证得分: {cv_score:.4f}")

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    print("🧪 AI优化日志功能测试")
    print("本测试将展示AI策略优化时的详细进度日志")
    print("日志将同时输出到控制台和logs/ai_optimization.log文件")
    
    try:
        # 测试基础优化日志
        test_basic_optimization_logs()
        
        # 测试分层优化日志
        test_hierarchical_optimization_logs()
        
        # 测试时间序列交叉验证日志
        test_time_series_cv_logs()
        
        print("\n" + "=" * 80)
        print("🎉 所有日志测试完成！")
        print("=" * 80)
        print("📝 详细日志已保存到: logs/ai_optimization.log")
        print("🔍 你可以查看该文件来了解完整的优化过程")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        logging.error(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main() 