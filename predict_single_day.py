#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单日相对低点预测脚本
允许用户输入日期，预测该日期是否为相对低点，并验证结果。
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer
from src.utils.utils import load_config

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

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
        logger.info(f"开始预测日期: {predict_date.strftime('%Y-%m-%d')} 是否为相对低点")

        # 1. 获取训练数据 (预测日期之前的所有数据)
        # 为了确保有足够的历史数据进行特征计算，我们获取 predict_date 之前足够长的数据
        # 假设我们需要至少 history_days 的数据来计算特征ß
        history_days_needed = config["data"]["history_days"] # 额外多一些天数确保计算指标
        start_date_for_training = predict_date - timedelta(days=history_days_needed)
        
        logger.info(f"获取训练数据从 {start_date_for_training.strftime('%Y-%m-%d')} 到 {predict_date.strftime('%Y-%m-%d')}")
        training_data = data_module.get_history_data(
            start_date=start_date_for_training.strftime('%Y-%m-%d'),
            end_date=predict_date.strftime('%Y-%m-%d')
        )
        
        if training_data.empty:
            logger.error("训练数据为空，无法进行预测。")
            return

        # 预处理数据 (计算技术指标等)
        training_data = data_module.preprocess_data(training_data)

        # 2. 训练AI模型
        logger.info("开始训练AI模型...")
        train_result = ai_optimizer.train_prediction_model(training_data, strategy_module)
        
        if not train_result["success"]:
            logger.error(f"AI模型训练失败: {train_result.get('error', '未知错误')}")
            return
        logger.info("AI模型训练成功，测试集准确率: %.2f%%", train_result["accuracy"] * 100)

        # 3. 预测输入日期是否为相对低点
        # 提取预测日期的数据点，确保是预处理后的训练数据中的最后一行
        # 预测时，我们只关心预测日期当天的数据，所以从训练数据中提取最后一行
        predict_day_data = training_data.iloc[-1:].copy()

        prediction_result = ai_optimizer.predict_low_point(predict_day_data)
        is_predicted_low_point = prediction_result["is_low_point"]
        confidence = prediction_result["confidence"]

        logger.info(f"预测结果: {predict_date_str} {'是' if is_predicted_low_point else '否'} 相对低点，置信度: {confidence:.2f}")

        # 4. 验证预测结果 (获取预测日期之后的数据)
        # 获取预测日期后 max_days + 10 个交易日的数据，用于验证
        end_date_for_validation = predict_date + timedelta(days=config["strategy"]["max_days"] + 10) # 额外多一些天数
        # 为了确保backtest能够正确评估predict_date，我们需要predict_date之前的数据
        # 并且backtest方法会从每个点开始向后看，所以验证数据需要包含predict_date之前的数据
        start_date_for_validation = predict_date - timedelta(days=config["strategy"]["max_days"] + 10) # 额外多一些天数
        
        start_date_str = start_date_for_validation.strftime('%Y-%m-%d')
        end_date_str = end_date_for_validation.strftime('%Y-%m-%d')
        logger.info(f"获取验证数据从 {start_date_str} 到 {end_date_str}")

        validation_data = data_module.get_history_data(
            start_date=start_date_str,
            end_date=end_date_str
        )

        if validation_data.empty:
            logger.warning("验证数据为空，无法验证预测结果。")
            return

        # 预处理验证数据
        full_validation_set = data_module.preprocess_data(validation_data)

        # 直接计算 predict_date 这一天的“实际是否为相对低点”、“未来最大涨幅”和“达到目标涨幅所需天数”
        predict_date_data = full_validation_set[full_validation_set['date'] == predict_date]
        if predict_date_data.empty:
            logger.warning(f"无法在验证数据中找到 {predict_date_str} 的记录，无法验证预测结果。")
            return

        predict_price = predict_date_data.iloc[0]['close']
        future_data = full_validation_set[full_validation_set['date'] > predict_date]
        if future_data.empty:
            logger.warning(f"无法获取 {predict_date_str} 之后的数据，无法验证预测结果。")
            return

        max_rise = 0.0
        days_to_rise = 0
        for i, row in future_data.iterrows():
            rise_rate = (row['close'] - predict_price) / predict_price
            if rise_rate > max_rise:
                max_rise = rise_rate
                days_to_rise = (row['date'] - predict_date).days

        actual_is_low_point = max_rise >= config["strategy"]["rise_threshold"]

        logger.info(f"\n--- 验证结果 --- ")
        logger.info(f"日期: {predict_date_str}")
        logger.info(f"实际是否为相对低点: {'是' if actual_is_low_point else '否'}")
        logger.info(f"未来最大涨幅: {max_rise:.2%}")
        logger.info(f"达到目标涨幅所需天数: {days_to_rise} 天")

        if is_predicted_low_point == actual_is_low_point:
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


