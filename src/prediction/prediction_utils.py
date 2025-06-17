#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测工具模块
包含预测和验证的公共逻辑
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def predict_and_validate(
    predict_date: datetime,
    data_module,
    strategy_module,
    ai_optimizer,
    config,
    logger
):
    """
    预测指定日期是否为相对低点并验证结果
    
    Args:
        predict_date: 预测日期
        data_module: 数据模块实例
        strategy_module: 策略模块实例
        ai_optimizer: AI优化器实例
        config: 配置信息
        logger: 日志记录器
    
    Returns:
        dict: 包含预测和验证结果的字典
    """
    try:
        # 1. 获取训练数据
        history_days_needed = config["data"]["history_days"]
        start_date_for_training = predict_date - timedelta(days=history_days_needed)
        
        logger.info(f"获取训练数据从 {start_date_for_training.strftime('%Y-%m-%d')} 到 {predict_date.strftime('%Y-%m-%d')}")
        training_data = data_module.get_history_data(
            start_date=start_date_for_training.strftime('%Y-%m-%d'),
            end_date=predict_date.strftime('%Y-%m-%d')
        )
        
        if training_data.empty:
            logger.error("训练数据为空，无法进行预测。")
            return {
                'date': predict_date,
                'predicted_low_point': None,
                'actual_low_point': None,
                'confidence': None,
                'future_max_rise': None,
                'days_to_rise': None,
                'prediction_correct': None
            }

        # 预处理数据
        training_data = data_module.preprocess_data(training_data)

        # 2. 训练AI模型
        logger.info("开始训练AI模型...")
        train_result = ai_optimizer.train_model(training_data, strategy_module)
        validate_result = ai_optimizer.validate_model(training_data, strategy_module)
        print('训练结果:', train_result)
        print('验证结果:', validate_result)
        
        if not train_result.get("success"):
            logger.error(f"AI模型训练失败: {train_result.get('error', '未知错误')}")
            return {
                'date': predict_date,
                'predicted_low_point': None,
                'actual_low_point': None,
                'confidence': None,
                'future_max_rise': None,
                'days_to_rise': None,
                'prediction_correct': None
            }
        if not validate_result.get("success"):
            logger.error(f"AI模型验证失败: {validate_result.get('error', '未知错误')}")
        # 训练成功后再输出验证集准确率
        logger.info("AI模型训练成功，验证集准确率: %.2f%%", (validate_result.get("accuracy") or 0) * 100)

        # 3. 预测输入日期是否为相对低点
        predict_day_data = training_data.iloc[-1:].copy()
        prediction_result = ai_optimizer.predict_low_point(predict_day_data)
        is_predicted_low_point = prediction_result.get("is_low_point")
        confidence = prediction_result.get("confidence")

        logger.info(f"预测结果: {predict_date.strftime('%Y-%m-%d')} {'是' if is_predicted_low_point else '否'} 相对低点，置信度: {confidence:.2f}")

        # 4. 验证预测结果
        end_date_for_validation = predict_date + timedelta(days=config["strategy"]["max_days"] + 10)
        start_date_for_validation = predict_date - timedelta(days=config["strategy"]["max_days"] + 10)
        
        validation_data = data_module.get_history_data(
            start_date=start_date_for_validation.strftime('%Y-%m-%d'),
            end_date=end_date_for_validation.strftime('%Y-%m-%d')
        )

        if validation_data.empty:
            logger.warning("验证数据为空，无法验证预测结果。")
            return {
                'date': predict_date,
                'predicted_low_point': is_predicted_low_point,
                'actual_low_point': None,
                'confidence': confidence,
                'future_max_rise': None,
                'days_to_rise': None,
                'prediction_correct': None
            }

        # 预处理验证数据
        full_validation_set = data_module.preprocess_data(validation_data)
        predict_date_data = full_validation_set[full_validation_set['date'] == predict_date]
        
        if predict_date_data.empty:
            logger.warning(f"无法在验证数据中找到 {predict_date.strftime('%Y-%m-%d')} 的记录，无法验证预测结果。")
            return {
                'date': predict_date,
                'predicted_low_point': is_predicted_low_point,
                'actual_low_point': None,
                'confidence': confidence,
                'future_max_rise': None,
                'days_to_rise': None,
                'prediction_correct': None
            }

        predict_price = predict_date_data.iloc[0]['close']
        future_data = full_validation_set[full_validation_set['date'] > predict_date]
        
        if future_data.empty:
            logger.warning(f"无法获取 {predict_date.strftime('%Y-%m-%d')} 之后的数据，无法验证预测结果。")
            return {
                'date': predict_date,
                'predicted_low_point': is_predicted_low_point,
                'actual_low_point': None,
                'confidence': confidence,
                'future_max_rise': None,
                'days_to_rise': None,
                'prediction_correct': None
            }

        max_rise = 0.0
        days_to_rise = 0
        for i, row in future_data.iterrows():
            rise_rate = (row['close'] - predict_price) / predict_price
            if rise_rate > max_rise:
                max_rise = rise_rate
                days_to_rise = (row['date'] - predict_date).days

        actual_is_low_point = max_rise >= config["strategy"]["rise_threshold"]

        logger.info(f"\n--- 验证结果 --- ")
        logger.info(f"日期: {predict_date.strftime('%Y-%m-%d')}")
        logger.info(f"实际是否为相对低点: {'是' if actual_is_low_point else '否'}")
        logger.info(f"未来最大涨幅: {max_rise:.2%}")
        logger.info(f"达到目标涨幅所需天数: {days_to_rise} 天")

        return {
            'date': predict_date,
            'predicted_low_point': is_predicted_low_point,
            'actual_low_point': actual_is_low_point,
            'confidence': confidence,
            'future_max_rise': max_rise,
            'days_to_rise': days_to_rise,
            'prediction_correct': is_predicted_low_point == actual_is_low_point
        }

    except Exception as e:
        logger.error(f"预测和验证过程发生错误: {e}")
        return {
            'date': predict_date if 'predict_date' in locals() else None,
            'predicted_low_point': None,
            'actual_low_point': None,
            'confidence': None,
            'future_max_rise': None,
            'days_to_rise': None,
            'prediction_correct': None
        } 