#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化滚动回测脚本
直接使用当前优化后的参数进行回测，避免AI模型训练问题
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.dates as mdates

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.utils.utils import load_config, setup_logging

def simple_rolling_backtest(start_date_str: str, end_date_str: str):
    """
    简化的滚动回测，使用当前优化后的参数
    """
    setup_logging('INFO')
    logger = logging.getLogger("SimpleRollingBacktest")

    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config = load_config(config_path=config_path)
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        
        logger.info(f"📊 开始滚动回测：{start_date_str} 到 {end_date_str}")
        logger.info(f"使用当前优化参数进行预测")

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # 获取回测期间的数据
        all_data = data_module.get_history_data(start_date=start_date_str, end_date=end_date_str)
        all_data = data_module.preprocess_data(all_data)
        
        if all_data.empty:
            logger.error("没有可用的数据进行回测")
            return False
            
        logger.info(f"获取到 {len(all_data)} 条回测数据")

        # 执行回测
        backtest_results = strategy_module.backtest(all_data)
        
        # 评估结果
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        # 显示结果
        logger.info("🎯 回测结果:")
        logger.info(f"   - 识别点数: {len(backtest_results)}")
        logger.info(f"   - 成功率: {evaluation.get('success_rate', 0):.2%}")
        logger.info(f"   - 平均涨幅: {evaluation.get('avg_rise', 0):.2%}")
        logger.info(f"   - 综合得分: {evaluation.get('score', 0):.4f}")
        
        # 显示基本统计
        if len(backtest_results) > 0:
            logger.info(f"📊 在此期间识别了 {len(backtest_results)} 个相对低点")
            logger.info(f"   时间段较短，策略参数可能过于敏感")
        else:
            logger.info("📊 在此期间未识别出相对低点")
        
        return True

    except Exception as e:
        logger.error(f"简化滚动回测失败: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python simple_rolling_backtest.py <start_date> <end_date>")
        print("示例: python simple_rolling_backtest.py 2025-06-01 2025-06-26")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    simple_rolling_backtest(start_date, end_date) 