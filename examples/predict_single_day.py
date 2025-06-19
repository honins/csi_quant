#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单日相对低点预测脚本
允许用户输入日期，预测该日期是否为相对低点，并验证结果。
支持使用已训练好的AI模型进行预测。
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

def predict_single_day(predict_date_str: str, use_trained_model: bool = True):
    """
    预测单日相对低点
    
    Args:
        predict_date_str: 预测日期字符串 (YYYY-MM-DD)
        use_trained_model: 是否使用已训练好的模型 (默认True)
    
    Returns:
        bool: 预测是否成功
    """
    setup_logging()
    logger = logging.getLogger("SingleDayPredictor")

    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config = load_config(config_path=config_path)
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)

        predict_date = datetime.strptime(predict_date_str, "%Y-%m-%d")
        if not is_trading_day(predict_date.date()):
            logger.warning(f"{predict_date_str} 不是A股交易日，跳过预测。")
            return False
            
        logger.info(f"开始预测日期: {predict_date.strftime('%Y-%m-%d')} 是否为相对低点")
        
        if use_trained_model:
            logger.info("使用已训练好的AI模型进行预测...")
            # 尝试加载已训练的模型
            if not ai_optimizer._load_model():
                logger.warning("未找到已训练的模型，将重新训练模型...")
                use_trained_model = False

        if use_trained_model:
            # 使用已训练模型进行预测
            result = predict_with_trained_model(
                predict_date=predict_date,
                data_module=data_module,
                strategy_module=strategy_module,
                ai_optimizer=ai_optimizer,
                config=config,
                logger=logger
            )
        else:
            # 使用原有方法（重新训练模型）
            logger.info("重新训练AI模型...")
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
            return False

        if result.prediction_correct is not None:
            if result.prediction_correct:
                logger.info("✅ 预测与实际相符！")
            else:
                logger.warning("❌ 预测与实际不符！")
        return True
        
    except Exception as e:
        logger.error(f"单日预测脚本运行失败: {e}")
        return False

def predict_with_trained_model(
    predict_date: datetime,
    data_module,
    strategy_module,
    ai_optimizer,
    config,
    logger
):
    """
    使用已训练模型进行预测
    
    Args:
        predict_date: 预测日期
        data_module: 数据模块实例
        strategy_module: 策略模块实例
        ai_optimizer: AI优化器实例（已加载模型）
        config: 配置信息
        logger: 日志记录器
    
    Returns:
        PredictionResult: 预测结果
    """
    from src.prediction.prediction_utils import PredictionResult
    from datetime import timedelta
    
    try:
        # 1. 获取预测所需的历史数据（用于特征提取）
        history_days_needed = config["data"]["history_days"]
        start_date_for_prediction = predict_date - timedelta(days=history_days_needed)
        
        logger.info(f"获取预测数据从 {start_date_for_prediction.strftime('%Y-%m-%d')} 到 {predict_date.strftime('%Y-%m-%d')}")
        prediction_data = data_module.get_history_data(
            start_date=start_date_for_prediction.strftime('%Y-%m-%d'),
            end_date=predict_date.strftime('%Y-%m-%d')
        )
        
        if prediction_data.empty:
            logger.error("预测数据为空，无法进行预测。")
            return None

        # 预处理数据
        prediction_data = data_module.preprocess_data(prediction_data)
        
        # 2. 使用已训练模型进行预测
        predict_day_data = prediction_data.iloc[-1:].copy()
        prediction_result = ai_optimizer.predict_low_point(predict_day_data)
        
        is_predicted_low_point = prediction_result.get("is_low_point")
        confidence = prediction_result.get("confidence")
        
        logger.info(f"AI预测结果: {predict_date.strftime('%Y-%m-%d')} {'是' if is_predicted_low_point else '否'} 相对低点，置信度: {confidence:.4f}")

        # 3. 验证预测结果（如果需要）
        end_date_for_validation = predict_date + timedelta(days=config["strategy"]["max_days"] + 10)
        start_date_for_validation = predict_date - timedelta(days=config["strategy"]["max_days"] + 10)
        
        validation_data = data_module.get_history_data(
            start_date=start_date_for_validation.strftime('%Y-%m-%d'),
            end_date=end_date_for_validation.strftime('%Y-%m-%d')
        )

        if validation_data.empty:
            logger.warning("验证数据为空，无法验证预测结果。")
            return PredictionResult(
                date=predict_date,
                predicted_low_point=is_predicted_low_point,
                actual_low_point=None,
                confidence=confidence,
                future_max_rise=None,
                days_to_rise=None,
                prediction_correct=None,
                predict_price=None
            )

        # 预处理验证数据
        full_validation_set = data_module.preprocess_data(validation_data)
        predict_date_data = full_validation_set[full_validation_set['date'] == predict_date]
        
        if predict_date_data.empty:
            logger.warning(f"无法在验证数据中找到 {predict_date.strftime('%Y-%m-%d')} 的记录，无法验证预测结果。")
            return PredictionResult(
                date=predict_date,
                predicted_low_point=is_predicted_low_point,
                actual_low_point=None,
                confidence=confidence,
                future_max_rise=None,
                days_to_rise=None,
                prediction_correct=None,
                predict_price=None
            )

        predict_price = predict_date_data.iloc[0]['close']
        future_data = full_validation_set[full_validation_set['date'] > predict_date]
        
        if future_data.empty:
            logger.warning(f"无法获取 {predict_date.strftime('%Y-%m-%d')} 之后的数据，无法验证预测结果。")
            return PredictionResult(
                date=predict_date,
                predicted_low_point=is_predicted_low_point,
                actual_low_point=None,
                confidence=confidence,
                future_max_rise=None,
                days_to_rise=None,
                prediction_correct=None,
                predict_price=predict_price
            )

        # 获取预测日的index
        predict_index = predict_date_data.iloc[0]['index']
        max_rise = 0.0
        days_to_rise = 0
        
        # 计算未来最大涨幅和达到目标涨幅所需天数
        for i, row in future_data.iterrows():
            rise_rate = (row['close'] - predict_price) / predict_price
            if rise_rate > max_rise:
                max_rise = rise_rate
                days_to_rise = row['index'] - predict_index

        actual_is_low_point = max_rise >= config["strategy"]["rise_threshold"]

        logger.info(f"\n--- 验证结果 --- ")
        logger.info(f"日期: {predict_date.strftime('%Y-%m-%d')}")
        logger.info(f"实际是否为相对低点: {'是' if actual_is_low_point else '否'}")
        logger.info(f"未来最大涨幅: {max_rise:.2%}")
        logger.info(f"达到目标涨幅所需天数: {days_to_rise} 天")

        return PredictionResult(
            date=predict_date,
            predicted_low_point=is_predicted_low_point,
            actual_low_point=actual_is_low_point,
            confidence=confidence,
            future_max_rise=max_rise,
            days_to_rise=days_to_rise,
            prediction_correct=is_predicted_low_point == actual_is_low_point,
            predict_price=predict_price
        )

    except Exception as e:
        logger.error(f"使用已训练模型预测失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict_single_day.py <YYYY-MM-DD> [--retrain]")
        print("示例: python predict_single_day.py 2024-06-01")
        print("示例: python predict_single_day.py 2024-06-01 --retrain")
        sys.exit(1)
    
    predict_date_str = sys.argv[1]
    use_trained_model = "--retrain" not in sys.argv
    
    if use_trained_model:
        print("🔮 使用已训练模型进行预测...")
    else:
        print("🔄 重新训练模型进行预测...")
    
    success = predict_single_day(predict_date_str, use_trained_model)
    sys.exit(0 if success else 1)


