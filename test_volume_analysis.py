#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试成交量分析逻辑的脚本
验证均线跌破+成交量分析的组合判断
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.utils.utils import load_config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def test_volume_analysis():
    """测试成交量分析逻辑"""
    logger = logging.getLogger(__name__)
    
    # 初始化模块
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    
    # 获取测试数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    logger.info(f"获取测试数据: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    data = data_module.get_history_data(start_date, end_date)
    
    if data is None or len(data) == 0:
        logger.error("无法获取测试数据")
        return
    
    # 测试最近几个交易日
    test_dates = data.tail(10).index.tolist()
    
    logger.info("=== 成交量分析测试结果 ===")
    
    for date in test_dates:
        # 获取到该日期的数据
        date_data = data.loc[:date]
        if len(date_data) < 20:  # 需要足够的历史数据
            continue
            
        # 计算技术指标
        date_data = data_module._calculate_technical_indicators(date_data)
        
        # 获取最新数据
        latest_data = date_data.iloc[-1]
        
        # 检查均线跌破条件
        ma5 = latest_data.get('ma5')
        ma10 = latest_data.get('ma10')
        ma20 = latest_data.get('ma20')
        latest_price = latest_data['close']
        
        if ma5 is not None and ma10 is not None and ma20 is not None:
            if latest_price < ma5 and latest_price < ma10 and latest_price < ma20:
                # 价格跌破所有均线
                volume_ratio = latest_data.get('volume_ratio', 1.0)
                price_change = latest_data.get('price_change', 0.0)
                
                logger.info(f"\n📅 {date.strftime('%Y-%m-%d')}")
                logger.info(f"   收盘价: {latest_price:.2f}")
                logger.info(f"   MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f}")
                logger.info(f"   成交量比率: {volume_ratio:.2f}")
                logger.info(f"   价格变化: {price_change:.2%}")
                
                # 判断成交量状态
                if volume_ratio > 1.4 and price_change < -0.02:
                    logger.info(f"   🔴 恐慌性抛售 - 可能是见底信号")
                elif volume_ratio > 1.2:
                    logger.info(f"   🟡 温和放量 - 可能是见底信号")
                elif volume_ratio < 0.8:
                    logger.info(f"   🔵 成交量萎缩 - 可能是下跌通道")
                else:
                    logger.info(f"   ⚪ 正常成交量")
            else:
                logger.info(f"\n📅 {date.strftime('%Y-%m-%d')} - 未跌破所有均线")

if __name__ == "__main__":
    setup_logging()
    test_volume_analysis() 