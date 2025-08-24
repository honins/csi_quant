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
# from matplotlib import font_manager
import pandas as pd
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# import matplotlib.dates as mdates
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved as AIOptimizer
from src.utils.utils import load_config, resolve_confidence_param
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day

def run_rolling_backtest(start_date_str: str, end_date_str: str, training_window_days: int = 365, 
                         reuse_model: bool = True, retrain_interval_days: int = None,
                         generate_report: bool = True, report_dir: str = None):
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
        
        # 新增：控制前N天的详细日志输出（默认前5天，可通过环境变量覆盖）
        try:
            first_n_days = int(os.environ.get('RB_FIRST_N_DAYS', '5'))
        except Exception:
            first_n_days = 5
        detailed_days_counter = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 滚动回测配置")
        logger.info(f"{'='*60}")
        logger.info(f"📅 回测期间: {start_date_str} 至 {end_date_str}")
        logger.info(f"🤖 只使用已训练模型: 启用")
        logger.info(f"📊 可用交易日: {len(available_dates)} 天")
        logger.info(f"📝 将输出前 {first_n_days} 天的详细预测日志（confidence 与 final_confidence 分布）")
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

            # 前N天详细日志：输出 confidence 和 final_confidence 及阈值
            if result is not None and getattr(result, 'date', None) is not None and detailed_days_counter < first_n_days:
                try:
                    # 固定阈值：从配置读取
                    final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)
                    logger.info("[详细日志-阈值与置信度]")
                    logger.info(f"  阈值(final_threshold): {final_threshold:.4f}")
                    logger.info(f"  confidence: {getattr(result, 'confidence', None):.4f}  | final_confidence: {getattr(result, 'final_confidence', None):.4f}")
                    logger.info(f"  预测结果: {'是低点' if getattr(result, 'predicted_low_point', False) else '非低点'}  | 实际: {('是低点' if getattr(result, 'actual_low_point', False) else '非/未知')}  | 是否正确: {getattr(result, 'prediction_correct', None)}")
                except Exception as e:
                    logger.warning(f"输出详细日志时出错: {e}")
                detailed_days_counter += 1

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

            # ========== 分类指标（Precision / Recall / F1 / Specificity / Balanced Accuracy）==========
            # 仅使用可验证的数据行计算混淆矩阵
            y_pred = results_df_validated['predicted_low_point'].astype(bool)
            y_true = results_df_validated['actual_low_point'].astype(bool)

            tp = int(((y_pred == True) & (y_true == True)).sum())
            fp = int(((y_pred == True) & (y_true == False)).sum())
            tn = int(((y_pred == False) & (y_true == False)).sum())
            fn = int(((y_pred == False) & (y_true == True)).sum())

            pred_pos = tp + fp
            pred_neg = tn + fn
            actual_pos = tp + fn
            actual_neg = tn + fp

            success_rate = (tp + tn) / max((tp + tn + fp + fn), 1)
            precision = tp / max((tp + fp), 1) if (tp + fp) > 0 else 0.0
            recall = tp / max((tp + fn), 1) if (tp + fn) > 0 else 0.0
            specificity = tn / max((tn + fp), 1) if (tn + fp) > 0 else 0.0
            balanced_acc = (recall + specificity) / 2.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            # 统一使用“可验证样本数”
            total_predictions_validated = int((tp + tn + fp + fn))
        
            # 新增：先初始化概率校准相关变量，避免上方诊断异常时未定义
            brier_value = None
            logloss_value = None
            ece_value = None
            calib_bin_rows = []
            reliability_points = []
        
            # 新增：置信度分布诊断（confidence 与 final_confidence）
            try:
                final_series = results_df['final_confidence'].astype(float).dropna()
                conf_series = results_df['confidence'].astype(float).dropna()

                def _safe_stat(s: pd.Series):
                    if s.empty:
                        return {
                            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                            'q10': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 0.0, 'q90': 0.0
                        }
                    qs = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                    return {
                        'mean': float(s.mean()),
                        'std': float(s.std(ddof=0)),
                        'min': float(s.min()),
                        'max': float(s.max()),
                        'q10': float(qs.loc[0.1]),
                        'q25': float(qs.loc[0.25]),
                        'q50': float(qs.loc[0.5]),
                        'q75': float(qs.loc[0.75]),
                        'q90': float(qs.loc[0.9]),
                    }

                conf_stat = _safe_stat(conf_series)
                final_stat = _safe_stat(final_series)

                # 直方分布（final_confidence）：含阈值附近的细分
                final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)
                bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
                bin_labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
                bin_counts = {lbl: 0 for lbl in bin_labels}
                if len(final_series) > 0:
                    counts, _ = np.histogram(final_series, bins=bins)
                    for i, lbl in enumerate(bin_labels):
                        bin_counts[lbl] = int(counts[i])
                bin_perc = {k: (v / len(final_series) * 100 if len(final_series) > 0 else 0.0) for k, v in bin_counts.items()}

                # 灰区
                gray_width = 0.05
                gray_lower = max(0.0, final_threshold - gray_width)
                gray_upper = min(1.0, final_threshold + gray_width)
                gray_mask = (results_df['final_confidence'] >= gray_lower) & (results_df['final_confidence'] <= gray_upper)
                gray_df = results_df[gray_mask]
                gray_total = len(gray_df)
                gray_pos = int((gray_df['predicted_low_point'] == True).sum()) if gray_total > 0 else 0
                gray_correct = int((gray_df['prediction_correct'] == True).sum()) if gray_total > 0 else 0

                # 相关性：final_confidence 与 future_max_rise、prediction_correct
                corr_final_rise = 0.0
                corr_final_correct = 0.0
                try:
                    tmp = results_df[['final_confidence', 'future_max_rise']].dropna()
                    if not tmp.empty and tmp['final_confidence'].nunique() > 1 and tmp['future_max_rise'].nunique() > 1:
                        corr_final_rise = float(tmp['final_confidence'].corr(tmp['future_max_rise']))
                except Exception:
                    pass
                try:
                    tmp2 = results_df[['final_confidence', 'prediction_correct']].dropna()
                    if not tmp2.empty and tmp2['final_confidence'].nunique() > 1 and tmp2['prediction_correct'].nunique() > 1:
                        corr_final_correct = float(tmp2['final_confidence'].corr(tmp2['prediction_correct'].astype(float)))
                except Exception:
                    pass

                logger.info("\n--- 置信度分布诊断 ---")
                logger.info(f"final_confidence: mean={final_stat['mean']:.4f}, std={final_stat['std']:.4f}, min={final_stat['min']:.4f}, max={final_stat['max']:.4f}")
                logger.info(f"quantiles(10/25/50/75/90): {final_stat['q10']:.4f}/{final_stat['q25']:.4f}/{final_stat['q50']:.4f}/{final_stat['q75']:.4f}/{final_stat['q90']:.4f}")
                logger.info(f"confidence: mean={conf_stat['mean']:.4f}, std={conf_stat['std']:.4f}, min={conf_stat['min']:.4f}, max={conf_stat['max']:.4f}")
                logger.info(f"分箱(final_confidence)：" + ", ".join([f"{lbl}: {bin_counts[lbl]} ({bin_perc[lbl]:.2f}%)" for lbl in bin_labels]))
                logger.info(f"阈值(final_threshold)={final_threshold:.2f}，灰区[{gray_lower:.2f}, {gray_upper:.2f}] 覆盖: {gray_total} 条，占比 {(gray_total/len(results_df)*100 if len(results_df)>0 else 0):.2f}%；灰区中预测正类 {gray_pos} 条，正确 {gray_correct} 条")
                logger.info(f"相关性：final_confidence vs future_max_rise = {corr_final_rise:.3f}，final_confidence vs prediction_correct = {corr_final_correct:.3f}")
            except Exception as e:
                logger.warning(f"置信度分布诊断失败: {e}")

            # 新增：概率校准评估（Brier / LogLoss / ECE & 可靠性表）
            try:
                if not results_df_validated.empty and 'final_confidence' in results_df_validated.columns and 'actual_low_point' in results_df_validated.columns:
                    y_true_arr = results_df_validated['actual_low_point'].astype(int).values
                    y_prob_arr = results_df_validated['final_confidence'].astype(float).values
                    mask = ~np.isnan(y_prob_arr)
                    y_true_f = y_true_arr[mask]
                    y_prob_f = np.clip(y_prob_arr[mask], 1e-6, 1 - 1e-6)

                    if y_true_f.size > 0 and np.unique(y_true_f).size > 1:
                        brier_value = float(brier_score_loss(y_true_f, y_prob_f))
                        logloss_value = float(log_loss(y_true_f, y_prob_f))

                        # 计算 ECE 与可靠性表（10 等宽分箱）
                        n_bins = 10
                        bins = np.linspace(0.0, 1.0, n_bins + 1)
                        # 将 1.0 放入最后一个箱
                        bin_ids = np.digitize(y_prob_f, bins, right=True) - 1
                        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

                        calib_bin_rows = []
                        reliability_points = []
                        ece_sum = 0.0
                        total_cnt = y_prob_f.size
                        for k in range(n_bins):
                            lo, hi = bins[k], bins[k+1]
                            idx = (bin_ids == k)
                            cnt = int(idx.sum())
                            if cnt > 0:
                                avg_conf = float(y_prob_f[idx].mean())
                                acc = float(y_true_f[idx].mean())
                                gap = abs(acc - avg_conf)
                                weight = cnt / max(total_cnt, 1)
                                ece_sum += weight * gap
                                # 区间右开（最后一段右闭）
                                right_bracket = ')' if k < n_bins - 1 else ']'
                                calib_bin_rows.append({
                                    'range': f"[{lo:.1f},{hi:.1f}{right_bracket}",
                                    'count': cnt,
                                    'avg_conf': avg_conf,
                                    'acc': acc,
                                    'gap': gap,
                                })
                                reliability_points.append({'mean_pred': avg_conf, 'frac_pos': acc})
                        ece_value = float(ece_sum)
                        logger.info(f"概率校准评估：Brier={brier_value:.4f}, LogLoss={logloss_value:.4f}, ECE(10)={ece_value:.4f}")
                    else:
                        logger.info("概率校准评估：类别单一或样本不足，跳过 Brier/LogLoss/ECE 计算")
                else:
                    logger.info("概率校准评估：结果为空或缺少必要列，跳过计算")
            except Exception as e:
                logger.warning(f"概率校准评估失败: {e}")

            # 获取策略参数
            rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.04)
            max_days = config.get('strategy', {}).get('max_days', 20)
            confidence_weights = config.get('strategy', {}).get('confidence_weights', {})
            rsi_oversold = confidence_weights.get('rsi_oversold_threshold', 30)
            rsi_low = confidence_weights.get('rsi_low_threshold', 40)
            # 从多个候选位置推断 final_threshold（用于报告展示）
            final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)

            # 生成报告
            report_path = None
            if generate_report:
                # 确保报告目录
                base_results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                reports_dir = report_dir or os.path.join(base_results_dir, 'reports')
                if not os.path.exists(reports_dir):
                    os.makedirs(reports_dir)
                report_path = os.path.join(reports_dir, f"report_rolling_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

                # 选取关键信号（预测为正类）按最终置信度排序
                pos_signals = []
                try:
                    pos_df = results_df_validated[results_df_validated['predicted_low_point'] == True].copy()
                    pos_df = pos_df.sort_values(by=['final_confidence', 'confidence'], ascending=False)
                    for idx, (dt, row) in enumerate(pos_df.head(15).iterrows(), start=1):
                        pos_signals.append({
                            'index': idx,
                            'date': dt.strftime('%Y-%m-%d'),
                            'predicted': '是' if row.get('predicted_low_point') else '否',
                            'actual': '是' if row.get('actual_low_point') else '否',
                            'confidence': row.get('confidence', 0),
                            'final_confidence': row.get('final_confidence', 0),
                            'future_max_rise': row.get('future_max_rise', 0),
                            'actual': '是' if row.get('actual_low_point') else '否',
                            'max_rise': f"{float(row.get('future_max_rise', 0)):.2%}" if not pd.isna(row.get('future_max_rise')) else "N/A",
                            'days_to_rise': f"{int(row.get('days_to_rise', 0))}" if not pd.isna(row.get('days_to_rise')) else "N/A",
                            'predict_price': row.get('predict_price') if row.get('predict_price') is not None else 'N/A',
                            'correct': '✅' if row.get('prediction_correct') else '❌'
                        })
                except Exception as e:
                    pos_signals = [{'error': f"生成样例行时出现异常: {e}"}]

                # 新增：全区间 Top-N（按 final_confidence 降序，包含未达阈值）
                top_all_signals = []
                try:
                    all_df = results_df.copy()
                    all_df = all_df.sort_values(by=['final_confidence', 'confidence'], ascending=False)
                    for idx, (dt, row) in enumerate(all_df.head(15).iterrows(), start=1):
                        top_all_signals.append({
                            'index': idx,
                            'date': dt.strftime('%Y-%m-%d'),
                            'predicted': '是' if row.get('predicted_low_point') else '否',
                            'actual': '是' if row.get('actual_low_point') else '否',
                            'confidence': row.get('confidence', 0),
                            'final_confidence': row.get('final_confidence', 0),
                            'future_max_rise': row.get('future_max_rise', 0),
                            'days_to_rise': int(row.get('days_to_rise') or 0) if row.get('days_to_rise') is not None else 0,
                            'predict_price': row.get('predict_price') if row.get('predict_price') is not None else 'N/A',
                            'correct': ('✅' if row.get('prediction_correct') else ('❌' if row.get('prediction_correct') is not None else 'N/A'))
                        })
                except Exception as e:
                    top_all_signals = [{'error': f"生成Top-N时出现异常: {e}"}]

                # 生成预测分布表格（类似原图表的数据展示）
                correct_dates = results_df_validated[results_df_validated['prediction_correct'] == True]
                incorrect_dates = results_df_validated[results_df_validated['prediction_correct'] == False]
                
                # 按月份统计预测分布（替代原图表的时间轴信息）
                monthly_stats = []
                try:
                    monthly_groups = results_df_validated.groupby(results_df_validated.index.to_period('M'))
                    for period, group in monthly_groups:
                        month_total = len(group)
                        month_correct = group['prediction_correct'].sum()
                        month_pos_pred = (group['predicted_low_point'] == True).sum()
                        month_pos_actual = (group['actual_low_point'] == True).sum()
                        monthly_stats.append({
                            'month': str(period),
                            'total': month_total,
                            'correct': month_correct,
                            'success_rate': month_correct / month_total if month_total > 0 else 0,
                            'pred_positive': month_pos_pred,
                            'actual_positive': month_pos_actual
                        })
                except Exception as e:
                    monthly_stats = [{'error': f"月度统计异常: {e}"}]

                # 构建完整报告
                report_lines = []
                report_lines.append(f"# AI滚动回测报告")
                report_lines.append("")
                report_lines.append(f"## 基本信息")
                report_lines.append(f"- **回测期间**: {start_date_str} 至 {end_date_str}")
                report_lines.append(f"- **使用模型**: 已训练模型（禁止回测训练）")
                report_lines.append(f"- **策略参数**: rise_threshold={resolve_confidence_param(config, 'rise_threshold', 0.04):.1%}, max_days={config.get('strategy', {}).get('max_days', 20)}")
                report_lines.append(f"- **置信度阈值(final_threshold)**: {final_threshold:.2f}")
                report_lines.append(f"- **训练效率**: {training_count}/{len(results)} (节省 {((len(results) - training_count) / len(results) * 100):.1f}%)")
                report_lines.append("")
                
                report_lines.append("## 总体指标")
                report_lines.append(f"- **总预测日期数(可验证)**: {total_predictions_validated}")
                report_lines.append(f"- **正确预测数**: {correct_predictions}")
                report_lines.append(f"- **准确率(Accuracy)**: {success_rate:.2%}")
                report_lines.append(f"- **Precision**: {precision:.2%}")
                report_lines.append(f"- **Recall**: {recall:.2%}")
                report_lines.append(f"- **F1 Score**: {(2*precision*recall/max(precision+recall, 1e-12)):.2%}")
                report_lines.append(f"- **Specificity**: {specificity:.2%}")
                report_lines.append(f"- **Balanced Accuracy**: {balanced_acc:.2%}")
                report_lines.append("")

                # 新增：置信度分布诊断（写入报告）
                try:
                    report_lines.append("## 置信度分布诊断")
                    report_lines.append(f"- final_confidence: 均值={final_stat['mean']:.4f}, 标准差={final_stat['std']:.4f}, 最小={final_stat['min']:.4f}, 最大={final_stat['max']:.4f}")
                    report_lines.append(f"- 分位数(10/25/50/75/90): {final_stat['q10']:.4f} / {final_stat['q25']:.4f} / {final_stat['q50']:.4f} / {final_stat['q75']:.4f} / {final_stat['q90']:.4f}")
                    report_lines.append(f"- confidence: 均值={conf_stat['mean']:.4f}, 标准差={conf_stat['std']:.4f}, 最小={conf_stat['min']:.4f}, 最大={conf_stat['max']:.4f}")
                    report_lines.append("")
                    report_lines.append("### final_confidence 直方分布")
                    report_lines.append("| 区间 | 数量 | 占比 |")
                    report_lines.append("|------|------|------|")
                    for lbl in bin_labels:
                        report_lines.append(f"| {lbl} | {int(bin_counts[lbl])} | {bin_perc[lbl]:.2f}% |")
                    report_lines.append("")
                    report_lines.append(f"- 阈值={final_threshold:.2f}，灰区[{gray_lower:.2f}, {gray_upper:.2f}] 覆盖: {gray_total} 条，占比 {(gray_total/len(results_df)*100 if len(results_df)>0 else 0):.2f}%；灰区中预测正类 {gray_pos} 条，正确 {gray_correct} 条")
                    report_lines.append(f"- 相关性：final_confidence vs future_max_rise = {corr_final_rise:.3f}，final_confidence vs prediction_correct = {corr_final_correct:.3f}")
                    report_lines.append("")
                except Exception as e:
                    report_lines.append(f"(置信度分布诊断生成失败: {e})")
                    report_lines.append("")

                report_lines.append("## 概率校准评估")
                if brier_value is not None:
                    report_lines.append(f"- Brier Score: {brier_value:.4f}（越低越好）")
                else:
                    report_lines.append(f"- Brier Score: N/A")
                if logloss_value is not None:
                    report_lines.append(f"- Log Loss: {logloss_value:.4f}（越低越好）")
                else:
                    report_lines.append(f"- Log Loss: N/A")
                if ece_value is not None:
                    report_lines.append(f"- ECE(10 bins): {ece_value:.4f}（越低越好）")
                    report_lines.append("")
                    report_lines.append("### 可靠性表（分箱统计）")
                    report_lines.append("| 置信度区间 | 数量 | 平均置信度 | 实际正率 | 偏差 |")
                    report_lines.append("|------------|------|------------|----------|------|")
                    for row in calib_bin_rows:
                        report_lines.append(f"| {row['range']} | {row['count']} | {row['avg_conf']:.3f} | {row['acc']:.3f} | {row['gap']:.3f} |")
                if len(reliability_points) > 0:
                    report_lines.append("")
                    report_lines.append("- 可靠性曲线点(mean_pred → frac_pos)：" + ", ".join([f"{pt['mean_pred']:.2f}→{pt['frac_pos']:.2f}" for pt in reliability_points]))
                else:
                    report_lines.append(f"- ECE(10 bins): N/A")
                report_lines.append("")

                report_lines.append("## 预测分布与混淆矩阵")
                report_lines.append(f"- **预测为低点(正类)**: {pred_pos} ({(pred_pos/max(total_predictions_validated,1)*100):.2f}%)")
                report_lines.append(f"- **预测为非低点(负类)**: {pred_neg} ({(pred_neg/max(total_predictions_validated,1)*100):.2f}%)")
                report_lines.append(f"- **实际为低点(正类)**: {actual_pos} ({(actual_pos/max(total_predictions_validated,1)*100):.2f}%)")
                report_lines.append(f"- **实际为非低点(负类)**: {actual_neg} ({(actual_neg/max(total_predictions_validated,1)*100):.2f}%)")
                report_lines.append("")
                report_lines.append("### 混淆矩阵")
                report_lines.append("|       | 预测正类 | 预测负类 |")
                report_lines.append("|-------|---------|---------|")
                report_lines.append(f"| **实际正类** | TP: {tp} | FN: {fn} |")
                report_lines.append(f"| **实际负类** | FP: {fp} | TN: {tn} |")
                report_lines.append("")

                # 月度统计（按索引日期分组）
                month_group = results_df_validated.groupby(results_df_validated.index.to_period('M').astype(str))
                month_stats = []
                for month, group in month_group:
                    total = len(group)
                    month_correct = group['prediction_correct'].sum()
                    month_success_rate = month_correct / total if total > 0 else 0.0
                    month_pred_pos = (group['predicted_low_point'] == True).sum()
                    month_pos_actual = (group['actual_low_point'] == True).sum()
                    month_stats.append({
                        'month': month,
                        'total': int(total),
                        'correct': int(month_correct),
                        'success_rate': float(month_success_rate),
                        'pred_positive': int(month_pred_pos),
                        'actual_positive': int(month_pos_actual)
                    })

                report_lines.append("## 月度预测分布")
                report_lines.append("| 月份 | 总预测 | 正确数 | 成功率 | 预测正类 | 实际正类 |")
                report_lines.append("|------|--------|--------|--------|----------|----------|")
                for stat in month_stats:
                    report_lines.append(f"| {stat['month']} | {stat['total']} | {stat['correct']} | {stat['success_rate']:.1%} | {stat['pred_positive']} | {stat['actual_positive']} |")
                report_lines.append("")

                report_lines.append("## 每日预测明细")
                report_lines.append("| 日期 | 预测价格 | 预测结果 | 置信度 | 最终置信度 | 实际结果 | 未来最大涨幅 | 达标用时(天) | 预测正确 |")
                report_lines.append("|------|----------|----------|--------|------------|----------|-------------|-------------|----------|")
                for dt, row in results_df.iterrows():
                    date_str = pd.to_datetime(dt).strftime('%Y-%m-%d') if not pd.isna(dt) else ''
                    predict_price = f"{row.get('predict_price', '')}"
                    predicted = "是" if row.get('predicted_low_point') else "否"
                    confidence = f"{row.get('confidence', 0):.2f}"
                    final_confidence = f"{row.get('final_confidence', 0):.2f}"
                    actual = "是" if row.get('actual_low_point') else "否"
                    max_rise = f"{float(row.get('future_max_rise', 0)):.2%}" if not pd.isna(row.get('future_max_rise')) else "N/A"
                    days_to_rise = f"{int(row.get('days_to_rise', 0))}" if not pd.isna(row.get('days_to_rise')) else "N/A"
                    prediction_correct = "是" if row.get('prediction_correct') else "否"
                    if pd.isna(row.get('actual_low_point')):
                        actual = '数据不足'
                    if pd.isna(row.get('prediction_correct')):
                        prediction_correct = '数据不足'
                    report_lines.append(f"| {date_str} | {predict_price} | {predicted} | {confidence} | {final_confidence} | {actual} | {max_rise} | {days_to_rise} | {prediction_correct} |")
                report_lines.append("")

                report_lines.append(f"**策略参数**: 涨幅阈值={resolve_confidence_param(config, 'rise_threshold', 0.04):.1%}, 最大观察天数={config.get('strategy', {}).get('max_days', 20)}, RSI超卖={config.get('strategy', {}).get('rsi_oversold', 30)}, RSI偏低={config.get('strategy', {}).get('rsi_low', 40)}, 置信度阈值={final_threshold:.2f}")
                report_lines.append("")

                report_lines.append("## 关键信号详情（按最终置信度降序，最多15条）")
                report_lines.append("| 序号 | 日期 | 预测 | 实际 | 置信度 | 最终置信度 | 未来最大涨幅 | 用时天数 | 预测价 | 结果 |")
                report_lines.append("|------|------|------|------|--------|------------|-------------|----------|---------|------|")
                if len(pos_signals) > 0:
                    for signal in pos_signals:
                        report_lines.append(f"| {signal['index']} | {signal['date']} | {signal['predicted']} | {signal['actual']} | {signal['confidence']:.2f} | {signal['final_confidence']:.2f} | {signal['future_max_rise']:.2%} | {signal['days_to_rise']} | {signal['predict_price']} | {signal['correct']} |")
                else:
                    report_lines.append("- (本次无正类信号或无法生成样例)")
                report_lines.append("")

                # 新增：全区间 Top-N final_confidence（包含未达阈值）
                report_lines.append("## 全区间 Top-N final_confidence（包含未达阈值）")
                report_lines.append("| 序号 | 日期 | 预测 | 实际 | 置信度 | 最终置信度 | 未来最大涨幅 | 用时天数 | 预测价 | 结果 |")
                report_lines.append("|------|------|------|------|--------|------------|-------------|----------|---------|------|")
                if len(top_all_signals) > 0:
                    for signal in top_all_signals:
                        report_lines.append(f"| {signal['index']} | {signal['date']} | {signal['predicted']} | {signal['actual']} | {signal['confidence']:.2f} | {signal['final_confidence']:.2f} | {signal['future_max_rise']:.2%} | {signal['days_to_rise']} | {signal['predict_price']} | {signal['correct']} |")
                else:
                    report_lines.append("- (无法生成Top-N列表)")
                report_lines.append("")

                report_lines.append("## 策略参数详情")
                report_lines.append(f"- **涨幅阈值**: {resolve_confidence_param(config, 'rise_threshold', 0.04):.1%}")
                report_lines.append(f"- **最大观察天数**: {config.get('strategy', {}).get('max_days', 20)}")
                report_lines.append(f"- **RSI超卖阈值**: {config.get('strategy', {}).get('rsi_oversold', 30)}")
                report_lines.append(f"- **RSI偏低阈值**: {config.get('strategy', {}).get('rsi_low', 40)}")
                report_lines.append(f"- **最终置信度阈值**: {final_threshold:.2f}")
                report_lines.append("")

                report_lines.append("> **免责声明**: 本报告由脚本自动生成，仅用于策略与模型评估，不构成投资建议。")

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(report_lines))
                logger.info(f"📄 回测报告已生成: {os.path.relpath(report_path)}")

            # 返回计算好的指标（供网格测试/报告使用）
            return {
                'success': True,
                'metrics': {
                    'total_predictions': total_predictions,
                    'correct_predictions': int(correct_predictions),
                    'success_rate': success_rate,
                    'pred_positive': pred_pos,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'specificity': specificity,
                    'balanced_accuracy': balanced_acc
                },
                'report_path': report_path
            }
        else:
            logger.warning("没有有效的预测结果用于统计分析")
            return {
                'success': False,
                'error': '没有有效的预测结果'
            }

    except Exception as e:
        logger.error(f"滚动回测过程中发生错误: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def run_rolling_backtest_with_return(start_date_str: str, end_date_str: str, training_window_days: int = 365, 
                                     reuse_model: bool = True, retrain_interval_days: int = None,
                                     generate_report: bool = True, report_dir: str = None):
    """
    带返回值的滚动回测函数（供网格测试/报告使用）
    
    Args:
        start_date_str: 开始日期字符串
        end_date_str: 结束日期字符串
        training_window_days: 训练窗口天数
        reuse_model: 是否重用模型
        retrain_interval_days: 重训练间隔天数
        generate_report: 是否生成报告文档（Markdown）
        report_dir: 报告输出目录（可选）
    
    Returns:
        dict: 包含 success 标志和 metrics 的结果字典
    """
    return run_rolling_backtest(start_date_str, end_date_str, training_window_days, reuse_model, retrain_interval_days,
                                generate_report=generate_report, report_dir=report_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python run_rolling_backtest.py <start_date> <end_date>")
        print("示例: python run_rolling_backtest.py 2023-01-01 2023-03-31")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    run_rolling_backtest(start_date, end_date)


