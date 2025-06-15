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
            history_days_needed = config["data"]["history_days"]
            training_start_date = current_date - timedelta(days=history_days_needed)
            training_end_date = current_date  # 包含预测当天
            
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

            # 3. 预测当前日期是否为相对低点
            # 从训练数据中提取最后一行作为预测数据
            predict_day_data = training_data.iloc[-1:].copy()

            prediction_result = ai_optimizer.predict_low_point(predict_day_data)
            is_predicted_low_point = prediction_result["is_low_point"]
            confidence = prediction_result["confidence"]

            logger.info(f"预测结果: {current_date.strftime('%Y-%m-%d')} {'是' if is_predicted_low_point else '否'} 相对低点，置信度: {confidence:.2f}")

            # 4. 验证预测结果 (获取预测日期之后的数据)
            # 获取预测日期后 max_days + 10 个交易日的数据，用于验证
            validation_end_date = current_date + timedelta(days=config["strategy"]["max_days"] + 10) # 额外多一些天数
            # 为了确保backtest能够正确评估current_date，我们需要current_date之前的数据
            # 并且backtest方法会从每个点开始向后看，所以验证数据需要包含current_date之前的数据
            validation_start_date = current_date - timedelta(days=config["strategy"]["max_days"] + 10) # 额外多一些天数
            
            validation_data = data_module.get_history_data(
                start_date=validation_start_date.strftime('%Y-%m-%d'),
                end_date=validation_end_date.strftime('%Y-%m-%d')
            )
            
            if validation_data.empty:
                logger.warning(f"验证数据为空，无法验证 {current_date.strftime('%Y-%m-%d')} 的预测结果。")
                actual_is_low_point = None
                future_max_rise = None
                days_to_rise = None
            else:
                full_validation_set = data_module.preprocess_data(validation_data)
                # 直接计算当前预测日期的"实际是否为相对低点"、"未来最大涨幅"和"达到目标涨幅所需天数"
                predict_date_data = full_validation_set[full_validation_set['date'] == pd.to_datetime(current_date.strftime('%Y-%m-%d'))]
                if predict_date_data.empty:
                    logger.warning(f"无法在验证数据中找到 {current_date.strftime('%Y-%m-%d')} 的记录，无法验证预测结果。")
                    actual_is_low_point = None
                    future_max_rise = None
                    days_to_rise = None
                else:
                    predict_price = predict_date_data.iloc[0]['close']
                    future_data = full_validation_set[full_validation_set['date'] > pd.to_datetime(current_date.strftime('%Y-%m-%d'))]
                    if future_data.empty:
                        logger.warning(f"无法获取 {current_date.strftime('%Y-%m-%d')} 之后的数据，无法验证预测结果。")
                        actual_is_low_point = None
                        future_max_rise = None
                        days_to_rise = None
                    else:
                        max_rise = 0.0
                        days_to_rise = 0
                        for i, row in future_data.iterrows():
                            rise_rate = (row['close'] - predict_price) / predict_price
                            if rise_rate > max_rise:
                                max_rise = rise_rate
                                days_to_rise = (row['date'] - pd.to_datetime(current_date.strftime('%Y-%m-%d'))).days
                        actual_is_low_point = max_rise >= config["strategy"]["rise_threshold"]
                        future_max_rise = max_rise
                        logger.info(f"实际是否为相对低点: {'是' if actual_is_low_point else '否'}")
                        logger.info(f"未来最大涨幅: {future_max_rise:.2%}")
                        logger.info(f"达到目标涨幅所需天数: {days_to_rise} 天")

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
            
            # 在图片下方添加判定记录文本
            plt.figtext(0.5, 0.01, f"总预测日期数 (可验证): {total_predictions}\n正确预测数: {correct_predictions}\n预测成功率: {success_rate:.2f}%", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # 设置日期格式
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate() # 自动调整日期倾斜

            chart_path = os.path.join(os.path.dirname(__file__), 'results', f'rolling_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path)
            logger.info(f"滚动回测结果图已保存至: {chart_path}")

            # 生成第二张图片，记录每次预测情况
            plt.figure(figsize=(15, 10))
            plt.axis('off')
            table_data = []
            for date, row in results_df.iterrows():
                table_data.append([
                    date.strftime('%Y-%m-%d'),
                    '是' if row['predicted_low_point'] else '否',
                    '是' if row['actual_low_point'] else '否',
                    f"{row['confidence']:.2f}",
                    f"{row['future_max_rise']:.2%}" if pd.notna(row['future_max_rise']) else 'N/A',
                    f"{row['days_to_rise']}" if pd.notna(row['days_to_rise']) else 'N/A',
                    '是' if row['prediction_correct'] else '否'
                ])
            table = plt.table(cellText=table_data, colLabels=['日期', '预测结果', '实际结果', '置信度', '未来最大涨幅', '达到目标涨幅所需天数', '预测是否成功'], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            # 设置单元格颜色
            for i, row in enumerate(table_data):
                cell = table.get_celld()[(i+1, 6)]
                if row[-1] == '是':
                    cell.set_facecolor('#b6fcb6')  # 绿色
                else:
                    cell.set_facecolor('#ffb6b6')  # 红色
            plt.title('每次预测情况记录', fontsize=12)
            plt.tight_layout()
            chart_path2 = os.path.join(os.path.dirname(__file__), 'results', f'prediction_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path2, bbox_inches='tight')
            logger.info(f"每次预测情况记录图已保存至: {chart_path2}")
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


