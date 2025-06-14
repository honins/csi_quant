#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI预测模型滚动回测脚本

该脚本实现AI预测模型的滚动回测，模拟模型在不同时间点进行训练和预测，
并统计预测的成功率，以更真实地评估模型的长期表现。
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

def run_rolling_backtest(start_date_str: str, end_date_str: str, training_window_days: int = 365):
    setup_logging()
    logger = logging.getLogger("RollingBacktest")

    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        config = load_config(config_path=config_path)
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        results = []
        current_date = start_date

        while current_date <= end_date:
            logger.info(f"\n--- 滚动回测日期: {current_date.strftime('%Y-%m-%d')} ---")

            # 1. 获取训练数据 (当前日期之前的训练窗口数据)
            training_start_date = current_date - timedelta(days=training_window_days)
            training_end_date = current_date - timedelta(days=1) # 训练数据截止到预测日期的前一天
            
            logger.info(f"获取训练数据从 {training_start_date.strftime('%Y-%m-%d')} 到 {training_end_date.strftime('%Y-%m-%d')}")
            training_data = data_module.get_history_data(
                start_date=training_start_date.strftime('%Y-%m-%d'),
                end_date=training_end_date.strftime('%Y-%m-%d')
            )
            
            if training_data.empty:
                logger.warning(f"训练数据为空，跳过 {current_date.strftime('%Y-%m-%d')} 的预测。")
                current_date += timedelta(days=1)
                continue

            training_data = data_module.preprocess_data(training_data)

            # 2. 训练AI模型
            logger.info("开始训练AI模型...")
            train_result = ai_optimizer.train_prediction_model(training_data, strategy_module)
            
            if not train_result["success"]:
                logger.error(f"AI模型训练失败: {train_result.get('error', '未知错误')}，跳过 {current_date.strftime('%Y-%m-%d')} 的预测。")
                current_date += timedelta(days=1)
                continue
            logger.info("AI模型训练成功，测试集准确率: %.2f%%", train_result["accuracy"] * 100)

            # 3. 获取预测日期的数据
            predict_day_data = data_module.get_history_data(
                start_date=current_date.strftime('%Y-%m-%d'),
                end_date=current_date.strftime('%Y-%m-%d')
            )
            
            if predict_day_data.empty:
                logger.warning(f"预测日期 {current_date.strftime('%Y-%m-%d')} 数据为空，跳过。")
                current_date += timedelta(days=1)
                continue

            predict_day_data = data_module.preprocess_data(predict_day_data)

            # 4. 预测当前日期是否为相对低点
            prediction_result = ai_optimizer.predict_low_point(predict_day_data)
            is_predicted_low_point = prediction_result["is_low_point"]
            confidence = prediction_result["confidence"]

            logger.info(f"预测结果: {current_date.strftime('%Y-%m-%d')} {'是' if is_predicted_low_point else '否'} 相对低点，置信度: {confidence:.2f}")

            # 5. 验证预测结果 (获取预测日期之后的数据)
            # 获取预测日期后 max_days + 10 个交易日的数据，用于验证
            validation_end_date = current_date + timedelta(days=config["strategy"]["max_days"] + 10) # 额外多一些天数
            
            validation_data = data_module.get_history_data(
                start_date=current_date.strftime('%Y-%m-%d'),
                end_date=validation_end_date.strftime('%Y-%m-%d')
            )
            
            if validation_data.empty:
                logger.warning(f"验证数据为空，无法验证 {current_date.strftime('%Y-%m-%d')} 的预测结果。")
                actual_is_low_point = None
                future_max_rise = None
                days_to_rise = None
            else:
                full_validation_set = data_module.preprocess_data(validation_data)
                # 仅对当前预测日期进行回测验证
                backtest_validation_results = strategy_module.backtest(full_validation_set)
                
                predicted_day_validation = backtest_validation_results[
                    backtest_validation_results['date'] == pd.to_datetime(current_date.strftime('%Y-%m-%d'))
                ]

                if not predicted_day_validation.empty:
                    actual_is_low_point = predicted_day_validation.iloc[0]['is_low_point']
                    future_max_rise = predicted_day_validation.iloc[0]['future_max_rise']
                    days_to_rise = predicted_day_validation.iloc[0]['days_to_rise']
                    logger.info(f"实际是否为相对低点: {'是' if actual_is_low_point else '否'}")
                    logger.info(f"未来最大涨幅: {future_max_rise:.2%}")
                    logger.info(f"达到目标涨幅所需天数: {days_to_rise} 天")
                else:
                    logger.warning(f"无法在验证数据中找到 {current_date.strftime('%Y-%m-%d')} 的记录，无法验证预测结果。")
                    actual_is_low_point = None
                    future_max_rise = None
                    days_to_rise = None

            results.append({
                'date': current_date,
                'predicted_low_point': is_predicted_low_point,
                'actual_low_point': actual_is_low_point,
                'confidence': confidence,
                'future_max_rise': future_max_rise,
                'days_to_rise': days_to_rise,
                'prediction_correct': (is_predicted_low_point == actual_is_low_point) if actual_is_low_point is not None else None
            })

            current_date += timedelta(days=1) # 移动到下一个日期

        # 统计和可视化结果
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)

        # 过滤掉无法验证的行
        results_df_validated = results_df.dropna(subset=['prediction_correct'])

        if not results_df_validated.empty:
            total_predictions = len(results_df_validated)
            correct_predictions = results_df_validated['prediction_correct'].sum()
            success_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

            logger.info(f"\n--- 滚动回测统计结果 ---")
            logger.info(f"总预测日期数 (可验证): {total_predictions}")
            logger.info(f"正确预测数: {correct_predictions}")
            logger.info(f"预测成功率: {success_rate:.2f}%")

            # 生成统计图
            plt.figure(figsize=(15, 7))
            plt.plot(results_df.index, results_df['predicted_low_point'], 'o', label='Predicted Low Point', alpha=0.7)
            plt.plot(results_df.index, results_df['actual_low_point'], 'x', label='Actual Low Point', alpha=0.7)
            
            # 标记正确和错误的预测
            correct_dates = results_df_validated[results_df_validated['prediction_correct']].index
            incorrect_dates = results_df_validated[~results_df_validated['prediction_correct']].index
            
            plt.plot(correct_dates, results_df.loc[correct_dates, 'predicted_low_point'], 'go', markersize=8, label='Correct Prediction')
            plt.plot(incorrect_dates, results_df.loc[incorrect_dates, 'predicted_low_point'], 'ro', markersize=8, label='Incorrect Prediction')

            plt.title('AI Prediction Model Rolling Backtest Results')
            plt.xlabel('Date')
            plt.ylabel('Is Low Point (True/False)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: 'True' if x > 0.5 else 'False'))
            plt.gca().yaxis.set_major_locator(plt.FixedLocator([0, 1]))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # 设置日期格式
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.gcf().autofmt_xdate() # 自动调整日期倾斜

            chart_path = os.path.join(os.path.dirname(__file__), 'results', 'rolling_backtest_results.png')
            plt.savefig(chart_path)
            logger.info(f"滚动回测结果图已保存至: {chart_path}")
        else:
            logger.warning("没有可验证的预测结果来生成统计图。")

    except Exception as e:
        logger.error(f"滚动回测脚本运行失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python run_rolling_backtest.py <start_date> <end_date> <training_window_days>")
        print("示例: python run_rolling_backtest.py 2023-01-01 2023-03-31 365")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    training_window_days = int(sys.argv[3])
    run_rolling_backtest(start_date, end_date, training_window_days)


