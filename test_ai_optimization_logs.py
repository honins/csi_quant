#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试AI优化器的详细进度日志功能
"""

import os
import sys
import logging
import yaml
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.ai_optimizer import AIOptimizer
from src.data.data_module import DataModule

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ai_optimization_test.log', encoding='utf-8')
        ]
    )

def load_config():
    """加载配置文件"""
    config_path = 'config/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_ai_optimization_logs():
    """测试AI优化器的详细进度日志"""
    print("=" * 60)
    print("🧪 测试AI优化器的详细进度日志功能")
    print("=" * 60)
    
    try:
        # 1. 设置日志
        setup_logging()
        logger = logging.getLogger('TestAILogs')
        
        # 2. 加载配置
        logger.info("📋 加载配置文件...")
        config = load_config()
        logger.info("✅ 配置文件加载完成")
        
        # 3. 准备数据
        logger.info("📊 准备测试数据...")
        data_module = DataModule(config)
        
        # 获取回测日期范围
        start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
        end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
        
        # 加载历史数据
        data = data_module.get_history_data(start_date, end_date)
        data = data_module.preprocess_data(data)
        logger.info(f"✅ 数据加载完成，共 {len(data)} 条记录")
        
        # 4. 创建AI优化器
        logger.info("🤖 创建AI优化器...")
        ai_optimizer = AIOptimizer(config)
        logger.info("✅ AI优化器创建完成")
        
        # 5. 测试分层优化（包含详细进度日志）
        logger.info("🚀 开始测试分层优化...")
        start_time = datetime.now()
        
        result = ai_optimizer.hierarchical_optimization(data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 6. 输出结果
        logger.info("=" * 60)
        logger.info("🎯 测试完成!")
        logger.info("=" * 60)
        logger.info(f"📊 测试统计:")
        logger.info(f"   - 总耗时: {duration:.1f}秒")
        logger.info(f"   - 最佳得分: {result.get('best_score', 0):.4f}")
        logger.info(f"   - 交叉验证得分: {result.get('cv_score', 0):.4f}")
        logger.info(f"   - 高级优化得分: {result.get('advanced_score', 0):.4f}")
        
        if 'error' in result:
            logger.error(f"❌ 测试失败: {result['error']}")
            return False
        
        # 7. 验证参数
        params = result.get('params', {})
        logger.info("🔧 优化后的参数:")
        for key, value in params.items():
            if isinstance(value, float):
                logger.info(f"   - {key}: {value:.4f}")
            else:
                logger.info(f"   - {key}: {value}")
        
        logger.info("✅ 测试成功完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        return False

def test_parameter_optimization_logs():
    """测试参数优化的详细进度日志"""
    print("=" * 60)
    print("🧪 测试参数优化的详细进度日志")
    print("=" * 60)
    
    try:
        # 1. 设置日志
        setup_logging()
        logger = logging.getLogger('TestParamLogs')
        
        # 2. 加载配置和数据
        config = load_config()
        data_module = DataModule(config)
        
        # 获取回测日期范围
        start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
        end_date = config.get('backtest', {}).get('end_date', '2025-06-21')
        
        # 加载历史数据
        data = data_module.get_history_data(start_date, end_date)
        data = data_module.preprocess_data(data)
        
        # 3. 创建AI优化器和策略模块
        ai_optimizer = AIOptimizer(config)
        from src.strategy.strategy_module import StrategyModule
        strategy_module = StrategyModule(config)
        
        # 4. 测试参数优化
        logger.info("🎯 开始测试参数优化...")
        start_time = datetime.now()
        
        optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 5. 输出结果
        logger.info("=" * 60)
        logger.info("🎯 参数优化测试完成!")
        logger.info("=" * 60)
        logger.info(f"📊 优化统计:")
        logger.info(f"   - 总耗时: {duration:.1f}秒")
        
        logger.info("🔧 优化后的参数:")
        for key, value in optimized_params.items():
            if isinstance(value, float):
                logger.info(f"   - {key}: {value:.4f}")
            else:
                logger.info(f"   - {key}: {value}")
        
        logger.info("✅ 参数优化测试成功完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 参数优化测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 AI优化器进度日志测试")
    print("=" * 60)
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 测试1: 分层优化日志
    success1 = test_ai_optimization_logs()
    
    print("\n" + "=" * 60)
    
    # 测试2: 参数优化日志
    success2 = test_parameter_optimization_logs()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"   - 分层优化测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"   - 参数优化测试: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，请检查日志文件")
    
    print("=" * 60) 