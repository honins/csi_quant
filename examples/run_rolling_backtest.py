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
from src.ai.ai_optimizer_improved import AIOptimizerImproved as AIOptimizer
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day

def run_rolling_backtest(start_date_str: str, end_date_str: str, training_window_days: int = 365, 
                         reuse_model: bool = True, retrain_interval_days: int = None):
    setup_logging()
    logger = logging.getLogger("RollingBacktest")

    try:
        # 使用标准配置加载器（自动合并所有配置文件）
        from src.utils.config_loader import load_config as load_config_improved
        config = load_config_improved()
        
        # 应用训练策略配置
        if retrain_interval_days is not None:
            config.setdefault('ai', {})['retrain_interval_days'] = retrain_interval_days
        config.setdefault('ai', {})['enable_model_reuse'] = reuse_model
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        # 使用AI优化器
        from src.ai.ai_optimizer_improved import AIOptimizerImproved
        ai_optimizer = AIOptimizerImproved(config)

        # 🔒 禁止回测过程中自动训练模型，只允许使用已训练模型
        if not ai_optimizer._load_model():
            logger.error("❌ 未找到已训练的模型！")
            logger.error("💡 请先运行以下命令训练模型：")
            logger.error("   python run.py ai -m optimize  # AI优化+训练")
            logger.error("   python run.py ai -m full      # 完整重训练")
            return {
                'success': False,
                'error': '未找到已训练模型，请先训练！'
            }

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        results = []
        current_date = start_date
        training_count = 0  # 记录实际训练次数

        # 预先获取所有可用交易日
        all_data = data_module.get_history_data(start_date=start_date, end_date=end_date)
        all_data = data_module.preprocess_data(all_data)
        available_dates = set(pd.to_datetime(all_data['date']).dt.date)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 滚动回测配置")
        logger.info(f"{'='*60}")
        logger.info(f"📅 回测期间: {start_date_str} 至 {end_date_str}")
        logger.info(f"🤖 只使用已训练模型: 启用")
        logger.info(f"📊 可用交易日: {len(available_dates)} 天")
        logger.info(f"{'='*60}")

        while current_date <= end_date:
            # 新增：判断该日期是否在交易数据源中
            if current_date.date() not in available_dates:
                current_date += timedelta(days=1)
                continue
            if not is_trading_day(current_date.date()):
                current_date += timedelta(days=1)
                continue

            logger.info(f"\n--- 滚动回测日期: {current_date.strftime('%Y-%m-%d')} ---")

            # 只用已训练模型进行预测和验证
            result = predict_and_validate(
                predict_date=current_date,
                data_module=data_module,
                strategy_module=strategy_module,
                ai_optimizer=ai_optimizer,
                config=config,
                logger=logger,
                force_retrain=False,  # 禁止自动训练
                only_use_trained_model=True  # 禁止任何训练和保存
            )
            # 统计训练次数（理论上应为0）
            if hasattr(ai_optimizer, '_last_training_date') and ai_optimizer._last_training_date == current_date:
                training_count += 1

            if result is not None and getattr(result, 'date', None) is not None:
                results.append(result)

            current_date += timedelta(days=1) # 移动到下一个日期

        # 统计和可视化结果
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 回测效率统计")
        logger.info(f"{'='*60}")
        logger.info(f"🎯 总回测天数: {len(results)}")
        logger.info(f"🔄 实际训练次数: {training_count}")
        logger.info(f"⚡ 效率提升: {((len(results) - training_count) / len(results) * 100):.1f}%")
        logger.info(f"{'='*60}")
        
        results_df = pd.DataFrame([vars(r) for r in results])
        if 'date' not in results_df.columns:
            logger.error(f"结果DataFrame缺少date列，实际列: {results_df.columns.tolist()}")
            raise ValueError("结果DataFrame缺少date列")
        # 确保date为datetime类型
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df.set_index('date', inplace=True)
        # 确保prediction_correct为bool类型 - 修复FutureWarning
        results_df['prediction_correct'] = results_df['prediction_correct'].fillna(False).infer_objects(copy=False).astype(bool)

        # 过滤掉无法验证的行
        results_df_validated = results_df.dropna(subset=['prediction_correct'])

        if not results_df_validated.empty:
            total_predictions = len(results_df_validated)
            correct_predictions = results_df_validated['prediction_correct'].sum()
            success_rate = (correct_predictions / total_predictions) if total_predictions > 0 else 0

            logger.info("\n--- 滚动回测统计结果 ---")
            logger.info(f"总预测日期数 (可验证): {total_predictions}")
            logger.info(f"正确预测数: {correct_predictions}")
            logger.info(f"预测成功率: {success_rate:.2%}")

            # 生成统计图
            plt.figure(figsize=(15, 8))  # 增加高度为参数信息留出空间
            plt.plot(results_df.index, results_df['predicted_low_point'], 'o', label='Predicted Low Point', alpha=0.7)
            plt.plot(results_df.index, results_df['actual_low_point'], 'x', label='Actual Low Point', alpha=0.7)
            
            # Mark correct and incorrect predictions
            correct_dates = results_df_validated[results_df_validated['prediction_correct']].index
            incorrect_dates = results_df_validated[~results_df_validated['prediction_correct']].index
            
            plt.plot(correct_dates, results_df.loc[correct_dates, 'predicted_low_point'], 'go', markersize=8, label='Correct Prediction')
            plt.plot(incorrect_dates, results_df.loc[incorrect_dates, 'predicted_low_point'], 'ro', markersize=8, label='Incorrect Prediction')

            # 获取策略参数
            rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.04)
            max_days = config.get('strategy', {}).get('max_days', 20)
            confidence_weights = config.get('strategy', {}).get('confidence_weights', {})
            rsi_oversold = confidence_weights.get('rsi_oversold_threshold', 30)
            rsi_low = confidence_weights.get('rsi_low_threshold', 40)
            # 从 system.yaml 读取 final_threshold
            final_threshold = config.get('final_threshold', 0.5)

            # 更新标题以包含训练效率信息
            efficiency_str = f"训练次数: {training_count}/{len(results)} (节省 {((len(results) - training_count) / len(results) * 100):.1f}%)"
            plt.title(f'AI Prediction Model Rolling Backtest Results\n(Rise Threshold: {rise_threshold:.1%}, Max Days: {max_days}, {efficiency_str})', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel('Is Low Point (True/False)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: 'True' if x > 0.5 else 'False'))
            plt.gca().yaxis.set_major_locator(plt.FixedLocator([0, 1]))
            plt.legend()
            plt.grid(True)
            
            # Set date format
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate() # Auto rotate date labels
            
            # 调整布局，为底部参数信息留出空间
            plt.subplots_adjust(bottom=0.15)
            
            # Add summary text and strategy parameters below the chart
            last_data_date = results_df.index.max().strftime('%Y-%m-%d')
            summary_text = f"Total Predictions: {total_predictions} | Correct: {correct_predictions} | Success Rate: {success_rate:.2f}% | Data Until: {last_data_date}"
            param_text = f"Strategy Params: Rise Threshold={rise_threshold:.1%}, Max Days={max_days}, RSI Oversold={rsi_oversold}, RSI Low={rsi_low}, Confidence Threshold={final_threshold:.2f}"
            efficiency_text = f"Training Efficiency: {training_count}/{len(results)} trainings (Saved {((len(results) - training_count) / len(results) * 100):.1f}% computing time)"
            
            plt.figtext(0.5, 0.10, summary_text, ha='center', fontsize=11, 
                       bbox=dict(facecolor='lightblue', alpha=0.8))
            plt.figtext(0.5, 0.06, param_text, ha='center', fontsize=10, 
                       bbox=dict(facecolor='lightgray', alpha=0.8))
            plt.figtext(0.5, 0.02, efficiency_text, ha='center', fontsize=10, 
                       bbox=dict(facecolor='lightgreen', alpha=0.8))
            
            # 确保results目录存在
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            
            # 创建子目录结构
            charts_dir = os.path.join(results_dir, 'charts')
            backtest_dir = os.path.join(charts_dir, 'rolling_backtest')
            
            for directory in [results_dir, charts_dir, backtest_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    
            chart_path = os.path.join(backtest_dir, f'rolling_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path)
            logger.info(f"Rolling backtest result chart saved to: {os.path.relpath(chart_path)}")

            # Second chart: record each prediction
            plt.figure(figsize=(15, 12))  # 增加高度为参数信息留出空间
            plt.axis('off')
            table_data = []
            def safe_str(val, float_fmt="{:.2f}"):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "N/A"
                if isinstance(val, float):
                    return float_fmt.format(val)
                return str(val)

            for date, row in results_df.iterrows():
                predict_price = safe_str(row['predict_price'])
                predicted = "Yes" if row['predicted_low_point'] else "No"
                confidence = safe_str(row.get('final_confidence', row.get('confidence', 0)))
                actual = "Yes" if row['actual_low_point'] else "No"
                max_rise = safe_str(row['future_max_rise'], "{:.2%}")
                days_to_rise = safe_str(row['days_to_rise'], "{:.0f}")
                prediction_correct = "Yes" if row['prediction_correct'] else "No"
         
                # 判断是否验证数据不足
                if pd.isna(row['actual_low_point']):
                    actual = 'Insufficient Data'
                    max_rise = 'Insufficient Data'
                    days_to_rise = 'Insufficient Data'
                    prediction_correct = 'Insufficient Data'
                table_data.append([
                    date.strftime('%Y-%m-%d'),
                    predict_price,
                    predicted,
                    confidence,
                    actual,
                    max_rise,
                    days_to_rise,
                    prediction_correct
                ])
            table = plt.table(cellText=table_data, colLabels=['Date', 'Predict Price', 'Predicted', 'Final Confidence', 'Actual', 'Max Future Rise', 'Days to Target Rise', 'Prediction Correct'], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            # Set cell color
            for i, row in enumerate(table_data):
                    # Predict Price
                table.get_celld()[(i+1, 1)].set_facecolor('#e3f2fd')
                # Predicted
                table.get_celld()[(i+1, 2)].set_facecolor('#fff9c4')
                # Confidence
                table.get_celld()[(i+1, 3)].set_facecolor('#ede7f6')
                # Actual
                table.get_celld()[(i+1, 4)].set_facecolor('#ffe0b2')
                # Max Future Rise
                table.get_celld()[(i+1, 5)].set_facecolor('#e8f5e9')
                # Days to Target Rise
                table.get_celld()[(i+1, 6)].set_facecolor('#f5f5f5')

                for j, cell_val in enumerate(row):
                    cell = table.get_celld()[(i+1, j)]
                    val = str(cell_val).strip().lower()
                    if val == 'yes':
                        cell.set_facecolor('#e8f5e9')  # 淡绿
                    elif val == 'no':
                        cell.set_facecolor('#ffebee')  # 淡红

            # 获取策略参数
            rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.04)
            max_days = config.get('strategy', {}).get('max_days', 20)
            confidence_weights = config.get('strategy', {}).get('confidence_weights', {})
            rsi_oversold = confidence_weights.get('rsi_oversold_threshold', 30)
            rsi_low = confidence_weights.get('rsi_low_threshold', 40)
            # 从 system.yaml 读取 final_threshold
            final_threshold = config.get('final_threshold', 0.5)

            plt.title(f'Prediction Details\n(Rise Threshold: {rise_threshold:.1%}, Max Days: {max_days})', fontsize=14, pad=20)
            
            # 调整布局，为底部参数信息留出空间
            plt.subplots_adjust(bottom=0.12)
            
            # 在表格下方添加策略参数信息
            param_text = f"Strategy Params: Rise Threshold={rise_threshold:.1%}, Max Days={max_days}, RSI Oversold={rsi_oversold}, RSI Low={rsi_low}, Confidence Threshold={final_threshold:.2f} | Data Until: {last_data_date}"
            plt.figtext(0.5, 0.05, param_text, ha='center', fontsize=11, 
                       bbox=dict(facecolor='lightgray', alpha=0.8))
            
            # 保存到同一个子目录
            chart_path2 = os.path.join(backtest_dir, f'prediction_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path2, bbox_inches='tight')
            logger.info(f"Prediction details chart saved to: {os.path.relpath(chart_path2)}")
            return {
                'success': True,
                'success_rate': success_rate,
                'total_signals': total_predictions,
                'correct_predictions': correct_predictions,
                'training_efficiency': f"{((len(results) - training_count) / len(results) * 100):.1f}%"
            }
        else:
            logger.warning("没有可验证的预测结果来生成统计图。")
            return {
                'success': False,
                'error': '没有可验证的预测结果'
            }

    except Exception as e:
        logger.error(f"滚动回测脚本运行失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python run_rolling_backtest.py <start_date> <end_date>")
        print("示例: python run_rolling_backtest.py 2023-01-01 2023-03-31")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    run_rolling_backtest(start_date, end_date)


