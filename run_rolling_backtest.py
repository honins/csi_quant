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
from matplotlib import font_manager
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import matplotlib.dates as mdates

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day

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
            if not is_trading_day(current_date.date()):
                current_date += timedelta(days=1)
                continue

            logger.info(f"\n--- 滚动回测日期: {current_date.strftime('%Y-%m-%d')} ---")

            # 使用公共模块进行预测和验证
            result = predict_and_validate(
                predict_date=current_date,
                data_module=data_module,
                strategy_module=strategy_module,
                ai_optimizer=ai_optimizer,
                config=config,
                logger=logger
            )

            if result is not None:
                results.append(result)

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
            
            # Mark correct and incorrect predictions
            correct_dates = results_df_validated[results_df_validated['prediction_correct']].index
            incorrect_dates = results_df_validated[~results_df_validated['prediction_correct']].index
            
            plt.plot(correct_dates, results_df.reindex(correct_dates)['predicted_low_point'], 'go', markersize=8, label='Correct Prediction')
            plt.plot(incorrect_dates, results_df.reindex(incorrect_dates)['predicted_low_point'], 'ro', markersize=8, label='Incorrect Prediction')

            plt.title('AI Prediction Model Rolling Backtest Results')
            plt.xlabel('Date')
            plt.ylabel('Is Low Point (True/False)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: 'True' if x > 0.5 else 'False'))
            plt.gca().yaxis.set_major_locator(plt.FixedLocator([0, 1]))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Add summary text below the chart
            plt.figtext(0.5, 0.01, f"Total validated predictions: {total_predictions}\nCorrect predictions: {correct_predictions}\nSuccess rate: {success_rate:.2f}%", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Set date format
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate() # Auto rotate date labels

            chart_path = os.path.join(os.path.dirname(__file__), 'results', f'rolling_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path)
            logger.info(f"Rolling backtest result chart saved to: {chart_path}")

            # Second chart: record each prediction
            plt.figure(figsize=(15, 10))
            plt.axis('off')
            table_data = []
            for date, row in results_df.iterrows():
                # 获取当日收盘价
                close_value = None
                if 'close' in results_df.columns:
                    close_value = row['close']
                elif 'close' in row:
                    close_value = row['close']
                else:
                    close_value = 'N/A'
                table_data.append([
                    date.strftime('%Y-%m-%d'),
                    f"{close_value:.2f}" if close_value != 'N/A' and close_value is not None else 'N/A',
                    'Yes' if row['predicted_low_point'] else 'No',
                    'Yes' if row['actual_low_point'] else 'No',
                    f"{row['confidence']:.2f}",
                    f"{row['future_max_rise']:.2%}" if pd.notna(row['future_max_rise']) else 'N/A',
                    f"{row['days_to_rise']}" if pd.notna(row['days_to_rise']) else 'N/A',
                    'Yes' if row['prediction_correct'] else 'No'
                ])
            table = plt.table(cellText=table_data, colLabels=['Date', 'Close', 'Predicted', 'Actual', 'Confidence', 'Max Future Rise', 'Days to Target Rise', 'Prediction Correct'], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            # Set cell color
            for i, row in enumerate(table_data):
                cell = table.get_celld()[(i+1, 7)]
                if row[-1] == 'Yes':
                    cell.set_facecolor('#b6fcb6')  # Green
                else:
                    cell.set_facecolor('#ffb6b6')  # Red
            plt.title('Prediction Details', fontsize=12)
            plt.tight_layout()
            chart_path2 = os.path.join(os.path.dirname(__file__), 'results', f'prediction_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path2, bbox_inches='tight')
            logger.info(f"Prediction details chart saved to: {chart_path2}")
        else:
            logger.warning("没有可验证的预测结果来生成统计图。")

    except Exception as e:
        logger.error(f"滚动回测脚本运行失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python run_rolling_backtest.py <start_date> <end_date>")
        print("示例: python run_rolling_backtest.py 2023-01-01 2023-03-31")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    run_rolling_backtest(start_date, end_date)


