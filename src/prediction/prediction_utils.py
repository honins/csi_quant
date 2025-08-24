#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测工具模块
包含预测和验证的公共逻辑
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.utils.utils import resolve_confidence_param

@dataclass
class PredictionResult:
    date: datetime
    predicted_low_point: Optional[bool]
    actual_low_point: Optional[bool]
    confidence: Optional[float]
    final_confidence: Optional[float]
    future_max_rise: Optional[float]
    days_to_rise: Optional[int]
    prediction_correct: Optional[bool]
    predict_price: Optional[float]

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
    logger,
    force_retrain: bool = False,
    only_use_trained_model: bool = False
) -> PredictionResult:
    """
    预测指定日期是否为相对低点并验证结果
    
    Args:
        predict_date: 预测日期
        data_module: 数据模块实例
        strategy_module: 策略模块实例
        ai_optimizer: AI优化器实例
        config: 配置信息
        logger: 日志记录器
        force_retrain: 是否强制重新训练模型
        only_use_trained_model: 是否只允许使用已训练模型，禁止任何训练和保存
    Returns:
        PredictionResult: 包含预测和验证结果的对象
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
            return PredictionResult(
                date=predict_date,
                predicted_low_point=None,
                actual_low_point=None,
                confidence=None,
                final_confidence=None,
                future_max_rise=None,
                days_to_rise=None,
                prediction_correct=None,
                predict_price=None
            )

        # 预处理数据
        training_data = data_module.preprocess_data(training_data)

        # 2. 智能训练策略：避免重复训练
        need_retrain = force_retrain
        # 检查是否需要重新训练
        if not need_retrain:
            if hasattr(ai_optimizer, 'model') and ai_optimizer.model is not None:
                logger.info("检测到已训练的模型")
                need_retrain = False
            else:
                # 没有模型，需要训练
                logger.info("未检测到训练模型，需要首次训练")
                need_retrain = True

        # 只允许用已训练模型，禁止任何训练和保存
        if only_use_trained_model:
            if need_retrain:
                logger.error("❌ 只允许使用已训练模型，当前无可用模型或模型已过期！")
                logger.error("💡 请先运行 'python run.py ai -m optimize' 训练模型！")
                return PredictionResult(
                    date=predict_date,
                    predicted_low_point=None,
                    actual_low_point=None,
                    confidence=None,
                    final_confidence=None,
                    future_max_rise=None,
                    days_to_rise=None,
                    prediction_correct=None,
                    predict_price=None
                )
            else:
                logger.info("使用现有AI模型进行预测...")
        else:
            # 3. 根据需要训练或使用现有模型（使用严格三层数据分割）
            if need_retrain:
                logger.info("开始训练AI模型（使用严格三层数据分割）...")
                
                # === 严格三层数据分割实现 ===
                # 确保训练数据足够大，至少100条记录
                if len(training_data) < 100:
                    logger.warning(f"训练数据量不足({len(training_data)}条)，跳过三层分割")
                    # 数据不足时直接使用全部数据
                    if hasattr(ai_optimizer, 'full_train'):
                        train_result = ai_optimizer.full_train(training_data, strategy_module)
                        validate_result = train_result
                    else:
                        train_result = ai_optimizer.train_model(training_data, strategy_module)
                        validate_result = ai_optimizer.validate_model(training_data, strategy_module)
                else:
                    # 获取数据分割比例
                    validation_config = config.get('ai', {}).get('validation', {})
                    train_ratio = validation_config.get('train_ratio', 0.6)
                    val_ratio = validation_config.get('validation_ratio', 0.25)
                    test_ratio = validation_config.get('test_ratio', 0.15)
                    
                    # 验证比例总和
                    total_ratio = train_ratio + val_ratio + test_ratio
                    if abs(total_ratio - 1.0) > 0.01:
                        logger.warning(f"数据分割比例总和不等于1.0: {total_ratio:.3f}，自动调整")
                        train_ratio = train_ratio / total_ratio
                        val_ratio = val_ratio / total_ratio
                        test_ratio = test_ratio / total_ratio
                    
                    # 时间序列数据分割（严格按时间顺序）
                    n = len(training_data)
                    train_end = int(n * train_ratio)
                    val_end = int(n * (train_ratio + val_ratio))
                    
                    train_data = training_data.iloc[:train_end].copy()
                    validation_data = training_data.iloc[train_end:val_end].copy()
                    test_data = training_data.iloc[val_end:].copy()
                    
                    logger.info(f"严格三层数据分割:")
                    logger.info(f"  训练集: {len(train_data)}条 ({len(train_data)/n:.1%})")
                    logger.info(f"  验证集: {len(validation_data)}条 ({len(validation_data)/n:.1%})")
                    logger.info(f"  测试集: {len(test_data)}条 ({len(test_data)/n:.1%})")
                    
                    # 过拟合检测：只在训练集上训练，在验证集上评估
                    if hasattr(ai_optimizer, 'full_train'):
                        train_result = ai_optimizer.full_train(train_data, strategy_module)
                        
                        # 在验证集上评估模型性能并进行过拟合检测
                        if len(validation_data) > 0:
                            from src.ai.overfitting_detector import OverfittingDetector, validate_data_split
                            
                            # 验证数据分割的正确性
                            split_validation = validate_data_split(train_data, validation_data, test_data)
                            if not split_validation['valid']:
                                for issue in split_validation['issues']:
                                    logger.error(f"数据分割问题: {issue}")
                            
                            # 收集验证集预测结果
                            val_prediction_results = []
                            for _, row in validation_data.iterrows():
                                single_row_df = pd.DataFrame([row])
                                pred_result = ai_optimizer.predict_low_point(single_row_df)
                                val_prediction_results.append(pred_result.get('confidence', 0))
                            
                            # 收集训练集预测结果（用于对比）
                            train_prediction_results = []
                            train_sample_size = min(50, len(train_data))  # 最多采样50个训练样本
                            for _, row in train_data.sample(n=train_sample_size).iterrows():
                                single_row_df = pd.DataFrame([row])
                                pred_result = ai_optimizer.predict_low_point(single_row_df)
                                train_prediction_results.append(pred_result.get('confidence', 0))
                            
                            # 使用专业的过拟合检测器
                            detector = OverfittingDetector(config)
                            
                            # 计算训练集和验证集得分
                            train_score = train_result.get('training_score', 0.8)  # 默认值
                            val_strategy_results = strategy_module.backtest(validation_data)
                            val_evaluation = strategy_module.evaluate_strategy(val_strategy_results)
                            val_score = val_evaluation.get('score', 0)
                            
                            # 执行综合过拟合检测
                            overfitting_result = detector.detect_overfitting(
                                train_score=train_score,
                                val_score=val_score,
                                val_predictions=val_prediction_results,
                                train_predictions=train_prediction_results
                            )
                            
                            # 如果检测到过拟合，记录详细信息
                            if overfitting_result['overfitting_detected']:
                                logger.error("🚨 检测到严重过拟合问题!")
                                for warning in overfitting_result['warnings']:
                                    logger.error(f"   ⚠️ {warning}")
                                logger.info("💡 建议采取以下措施:")
                                for recommendation in overfitting_result['recommendations']:
                                    logger.info(f"   📝 {recommendation}")
                        
                        validate_result = train_result
                    else:
                        train_result = ai_optimizer.train_model(train_data, strategy_module)
                        validate_result = ai_optimizer.validate_model(train_data, strategy_module)
                
                print('训练结果:', train_result)
                print('验证结果:', validate_result)
                
                if not train_result.get("success"):
                    logger.error(f"AI模型训练失败: {train_result.get('error', '未知错误')}")
                    return PredictionResult(
                        date=predict_date,
                        predicted_low_point=None,
                        actual_low_point=None,
                        confidence=None,
                        final_confidence=None,
                        future_max_rise=None,
                        days_to_rise=None,
                        prediction_correct=None,
                        predict_price=None
                    )
                if not validate_result.get("success"):
                    logger.error(f"AI模型验证失败: {validate_result.get('error', '未知错误')}")
                
                # 记录训练时间
                ai_optimizer._last_training_date = predict_date
                # 训练成功后再输出验证集准确率
                logger.info("AI模型训练成功，验证集准确率: %.2f%%", (validate_result.get("accuracy") or 0) * 100)
            else:
                logger.info("使用现有AI模型进行预测...")

        # 4. 预测输入日期是否为相对低点
        predict_day_data = training_data.iloc[-1:].copy()
        
        # 4a. 使用AI模型预测
        ai_prediction_result = ai_optimizer.predict_low_point(predict_day_data, predict_date.strftime('%Y-%m-%d'))
        ai_is_predicted_low_point = ai_prediction_result.get("is_low_point")
        ai_confidence = ai_prediction_result.get("confidence")
        ai_final_confidence = ai_prediction_result.get("final_confidence", ai_confidence)
        
        # 4b. 使用策略模块预测（获取策略置信度）
        strategy_prediction_result = strategy_module.identify_relative_low(predict_day_data)
        strategy_is_predicted_low_point = strategy_prediction_result.get("is_low_point")
        strategy_confidence = strategy_prediction_result.get("confidence")

        # === 置信度融合：AI 与 策略 ===
        # 读取权重与阈值（优先顶层confidence_weights，其次strategy.confidence_weights，最后default_strategy）
        conf_top = config.get('confidence_weights', {}) or {}
        conf_strategy = (config.get('strategy', {}) or {}).get('confidence_weights', {}) or {}
        conf_default = (config.get('default_strategy', {}) or {}).get('confidence_weights', {}) or {}

        ai_w = conf_top.get('ai_weight', conf_strategy.get('ai_weight', conf_default.get('ai_weight', 0.5)))
        stg_w = conf_top.get('strategy_weight', conf_strategy.get('strategy_weight', conf_default.get('strategy_weight', 0.5)))
        denom = (ai_w or 0) + (stg_w or 0)
        if denom <= 0:
            ai_w, stg_w = 0.5, 0.5
            denom = 1.0

        fused_final_confidence = (ai_final_confidence * ai_w + strategy_confidence * stg_w) / denom

        # 门槛读取：从配置读取固定阈值（不进行动态调整）
        final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)
        used_threshold = final_threshold

        # 基于RSI与20日波动率的“谨慎型”动态阈值（小幅、可控、尊重市场）
        try:
            # 读取动态阈值配置（优先级：顶层 -> strategy -> default_strategy）
            dyn_top = (config.get('confidence_weights', {}) or {}).get('dynamic_threshold', {}) or {}
            dyn_stg = ((config.get('strategy', {}) or {}).get('confidence_weights', {}) or {}).get('dynamic_threshold', {}) or {}
            dyn_def = ((config.get('default_strategy', {}) or {}).get('confidence_weights', {}) or {}).get('dynamic_threshold', {}) or {}
            dyn_cfg = dyn_top if dyn_top else (dyn_stg if dyn_stg else dyn_def)

            enabled = dyn_cfg.get('enabled', True)
            if enabled:
                max_adjust = float(dyn_cfg.get('max_adjust', 0.03))  # 单侧最多±0.03
                # 读取当前RSI与波动率
                latest_row = predict_day_data.iloc[-1] if len(predict_day_data) > 0 else None
                current_rsi = float(latest_row.get('rsi')) if (latest_row is not None and 'rsi' in latest_row and not pd.isna(latest_row.get('rsi'))) else None
                current_vol = float(latest_row.get('volatility')) if (latest_row is not None and 'volatility' in latest_row and not pd.isna(latest_row.get('volatility'))) else None

                adj = 0.0

                # RSI调整（默认：RSI<=oversold 小幅放宽阈值；RSI>=upper 小幅收紧阈值）
                rsi_cfg = dyn_cfg.get('rsi', {}) or {}
                # 使用策略层的RSI阈值作为基准，若未配置则回退到默认值
                conf_top = config.get('confidence_weights', {}) or {}
                oversold_base = conf_top.get('rsi_oversold_threshold', 30)
                rsi_oversold = float(rsi_cfg.get('oversold', oversold_base))
                rsi_upper = float(rsi_cfg.get('upper', 65))
                rsi_lower_adjust = float(rsi_cfg.get('lower_adjust', 0.015))  # 阈值下调幅度（更容易触发）
                rsi_upper_adjust = float(rsi_cfg.get('upper_adjust', 0.015))  # 阈值上调幅度（更严格）

                if current_rsi is not None:
                    if current_rsi <= rsi_oversold:
                        adj -= rsi_lower_adjust
                    elif current_rsi >= rsi_upper:
                        adj += rsi_upper_adjust

                # 波动率调整（默认：低波动→收紧；高波动→谨慎小幅放宽）
                vol_cfg = dyn_cfg.get('volatility', {}) or {}
                lookback = int(vol_cfg.get('lookback_mean', 60))
                low_ratio = float(vol_cfg.get('low_ratio', 0.90))
                high_ratio = float(vol_cfg.get('high_ratio', 1.10))
                vol_low_adjust = float(vol_cfg.get('low_adjust', 0.015))   # 低波动：上调阈值
                vol_high_adjust = float(vol_cfg.get('high_adjust', -0.010)) # 高波动：下调阈值（允许更宽松，但幅度更小）

                if current_vol is not None and 'volatility' in training_data.columns and training_data['volatility'].notna().any():
                    vol_mean = float(training_data['volatility'].tail(lookback).mean()) if len(training_data) >= 1 else None
                    if vol_mean and vol_mean > 0:
                        vol_ratio = current_vol / vol_mean
                        if vol_ratio <= low_ratio:
                            adj += vol_low_adjust
                        elif vol_ratio >= high_ratio:
                            adj += vol_high_adjust

                # 限幅并与基础阈值合成
                if adj > max_adjust:
                    adj = max_adjust
                if adj < -max_adjust:
                    adj = -max_adjust
                used_threshold = float(final_threshold + adj)
                # 物理边界（避免极端阈值）
                used_threshold = max(0.10, min(0.90, used_threshold))

                logger.info(f"动态阈值: base={final_threshold:.3f}, adj={adj:+.3f} -> used={used_threshold:.3f} (rsi={current_rsi if current_rsi is not None else 'N/A'}, vol={current_vol if current_vol is not None else 'N/A'})")
        except Exception as _e:
            # 动态阈值失败时回退到基础阈值，确保稳健
            used_threshold = final_threshold
            logger.warning(f"动态阈值计算失败，回退到固定阈值: {final_threshold:.3f}，原因: {_e}")

        # 使用融合后的final_confidence进行门控（固定阈值）
        is_predicted_low_point = fused_final_confidence >= used_threshold
        confidence = strategy_confidence  # 保留策略置信度作为基础显示
        final_confidence = fused_final_confidence

        logger.info(f"AI预测结果: {predict_date.strftime('%Y-%m-%d')} {'是' if ai_is_predicted_low_point else '否'} 相对低点，AI置信度: {ai_confidence:.2f}")
        logger.info(f"策略预测结果: {predict_date.strftime('%Y-%m-%d')} {'是' if strategy_is_predicted_low_point else '否'} 相对低点，策略置信度: {strategy_confidence:.2f}")
        # 详细融合日志：显示各自贡献与权重
        logger.info(f"融合细节: AI贡献={ai_final_confidence:.4f}×{ai_w:.2f}={(ai_final_confidence*ai_w):.4f}, 策略贡献={strategy_confidence:.4f}×{stg_w:.2f}={(strategy_confidence*stg_w):.4f}, 归一化分母={denom:.2f}")
        logger.info(f"融合结果: ai_weight={ai_w:.2f}, strategy_weight={stg_w:.2f}, final_confidence={final_confidence:.2f}, 阈值={used_threshold:.2f} → 预测{'是' if is_predicted_low_point else '否'}相对低点")

        # 5. 验证预测结果
        # 统一读取策略参数自 config['strategy']，若无则回退到 default_strategy
        strategy_section = config.get("strategy", {})
        if not strategy_section and "default_strategy" in config:
            # 兼容旧版配置
            strategy_section = {
                'max_days': config['default_strategy'].get('max_days', 20),
                'rise_threshold': config['default_strategy'].get('rise_threshold', 0.04)
            }
        max_days = strategy_section.get("max_days", 20)
        end_date_for_validation = predict_date + timedelta(days=max_days + 10)
        start_date_for_validation = predict_date - timedelta(days=max_days + 10)
        
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
                final_confidence=final_confidence,
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
                final_confidence=final_confidence,
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
                final_confidence=final_confidence,
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
                days_to_rise = row['index'] - predict_index  # 用index相减，代表交易日天数

        # 使用统一的策略段读取 rise_threshold，兼容 default_strategy
        rise_threshold = strategy_section.get('rise_threshold', 0.04)
        actual_is_low_point = max_rise >= rise_threshold

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
            final_confidence=final_confidence,
            future_max_rise=max_rise,
            days_to_rise=days_to_rise,
            prediction_correct=is_predicted_low_point == actual_is_low_point,
            predict_price=predict_price
        )

    except Exception as e:
        logger.error(f"预测和验证过程发生错误: {e}")
        # 使用传入的predict_date，确保不为None
        error_date = predict_date if predict_date is not None else datetime.now()
        return PredictionResult(
            date=error_date,
            predicted_low_point=None,
            actual_low_point=None,
            confidence=None,
            final_confidence=None,
            future_max_rise=None,
            days_to_rise=None,
            prediction_correct=None,
            predict_price=None
        )