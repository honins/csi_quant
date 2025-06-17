#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单日相对低点预测脚本
允许用户输入日期，预测该日期是否为相对低点，并验证结果。
"""

import sys
import os
import logging
from datetime import datetime

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day

def predict_single_day(predict_date_str: str):
    setup_logging()
    logger = logging.getLogger("SingleDayPredictor")

    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        config = load_config(config_path=config_path)
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)

        predict_date = datetime.strptime(predict_date_str, "%Y-%m-%d")
        if not is_trading_day(predict_date.date()):
            logger.warning(f"{predict_date_str} 不是A股交易日，跳过预测。")
            return
        logger.info(f"开始预测日期: {predict_date.strftime('%Y-%m-%d')} 是否为相对低点")

        # 使用公共模块进行预测和验证
        result = predict_and_validate(
            predict_date=predict_date,
            data_module=data_module,
            strategy_module=strategy_module,
            ai_optimizer=ai_optimizer,
            config=config,
            logger=logger
        )

        if result is None:
            logger.error("预测和验证过程失败")
            return

        if result['prediction_correct'] is not None:
            if result['prediction_correct']:
                logger.info("预测与实际相符！")
            else:
                logger.warning("预测与实际不符！")

    except Exception as e:
        logger.error(f"单日预测脚本运行失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python predict_single_day.py <YYYY-MM-DD>")
        sys.exit(1)
    
    predict_date_str = sys.argv[1]
    predict_single_day(predict_date_str)


