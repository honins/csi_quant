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
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
# 新增：用于离线概率标定
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import json

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

from src.utils.utils import resolve_confidence_param
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day
from src.utils.reporting import format_backtest_summary

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
        logger.info(f"📝 将输出前 {first_n_days} 天的详细预测日志（confidence 分布）")
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

            # 前N天详细日志：输出 confidence 及阈值
            if result is not None and getattr(result, 'date', None) is not None and detailed_days_counter < first_n_days:
                try:
                    # 固定阈值：从配置读取
                    final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)
                    logger.info("[详细日志-阈值与置信度]")
                    logger.info(f"  阈值(final_threshold): {final_threshold:.4f}")
                    logger.info(f"  confidence: {getattr(result, 'confidence', None):.4f}")
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
            calib_compare = []
            calib_bins_map = {}
        
            # 新增：置信度分布诊断（confidence）
            try:
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

                # 直方分布（confidence）：含阈值附近的细分
                final_threshold = resolve_confidence_param(config, 'final_threshold', 0.5)
                bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
                bin_labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
                bin_counts = {lbl: 0 for lbl in bin_labels}
                if len(conf_series) > 0:
                    counts, _ = np.histogram(conf_series, bins=bins)
                    for i, lbl in enumerate(bin_labels):
                        bin_counts[lbl] = int(counts[i])
                bin_perc = {k: (v / len(conf_series) * 100 if len(conf_series) > 0 else 0.0) for k, v in bin_counts.items()}

                # 灰区（决策带）
                band_top = (config.get('confidence_weights', {}) or {}).get('decision_band', {}) or {}
                band_stg = ((config.get('strategy', {}) or {}).get('confidence_weights', {}) or {}).get('decision_band', {}) or {}
                band_def = ((config.get('default_strategy', {}) or {}).get('confidence_weights', {}) or {}).get('decision_band', {}) or {}
                band_cfg = band_top if band_top else (band_stg if band_stg else band_def)
                abstain_thr = band_cfg.get('abstain_threshold')
                enter_thr = band_cfg.get('enter_threshold')
                if isinstance(abstain_thr, (int, float)) and isinstance(enter_thr, (int, float)):
                    gray_lower = float(abstain_thr)
                    gray_upper = float(enter_thr)
                else:
                    # 回退：若未配置决策带，使用固定±0.05范围
                    gray_width = 0.05
                    gray_lower = max(0.0, final_threshold - gray_width)
                    gray_upper = min(1.0, final_threshold + gray_width)
                if gray_lower > gray_upper:
                    gray_lower, gray_upper = gray_upper, gray_lower
                gray_lower = max(0.0, min(gray_lower, 1.0))
                gray_upper = max(0.0, min(gray_upper, 1.0))
                gray_mask = (results_df['confidence'] >= gray_lower) & (results_df['confidence'] <= gray_upper)
                gray_df = results_df[gray_mask]
                gray_total = len(gray_df)
                gray_pos = int((gray_df['predicted_low_point'] == True).sum()) if gray_total > 0 else 0
                gray_correct = int((gray_df['prediction_correct'] == True).sum()) if gray_total > 0 else 0

                # 相关性：confidence 与 future_max_rise、prediction_correct
                corr_final_rise = 0.0
                corr_final_correct = 0.0
                try:
                    tmp = results_df[['confidence', 'future_max_rise']].dropna()
                    if not tmp.empty and tmp['confidence'].nunique() > 1 and tmp['future_max_rise'].nunique() > 1:
                        corr_final_rise = float(tmp['confidence'].corr(tmp['future_max_rise']))
                except Exception:
                    pass
                try:
                    tmp2 = results_df[['confidence', 'prediction_correct']].dropna()
                    if not tmp2.empty and tmp2['confidence'].nunique() > 1 and tmp2['prediction_correct'].nunique() > 1:
                        corr_final_correct = float(tmp2['confidence'].corr(tmp2['prediction_correct'].astype(float)))
                except Exception:
                    pass

                logger.info("\n--- 置信度分布诊断 ---")
                logger.info(f"confidence: mean={conf_stat['mean']:.4f}, std={conf_stat['std']:.4f}, min={conf_stat['min']:.4f}, max={conf_stat['max']:.4f}")
                logger.info(f"quantiles(10/25/50/75/90): {conf_stat['q10']:.4f} / {conf_stat['q25']:.4f} / {conf_stat['q50']:.4f} / {conf_stat['q75']:.4f} / {conf_stat['q90']:.4f}")
                logger.info(f"分箱(confidence)：" + ", ".join([f"{lbl}: {bin_counts[lbl]} ({bin_perc[lbl]:.2f}%)" for lbl in bin_labels]))
                logger.info(f"阈值(final_threshold)={final_threshold:.2f}，灰区[{gray_lower:.2f}, {gray_upper:.2f}] 覆盖: {gray_total} 条，占比 {(gray_total/len(results_df)*100 if len(results_df)>0 else 0):.2f}%；灰区中预测正类 {gray_pos} 条，正确 {gray_correct} 条")
                logger.info(f"相关性：confidence vs future_max_rise = {corr_final_rise:.3f}，confidence vs prediction_correct = {corr_final_correct:.3f}")
            except Exception as e:
                logger.warning(f"置信度分布诊断失败: {e}")

            # 新增：概率校准评估（Brier / LogLoss / ECE & 可靠性表）
            try:
                if not results_df_validated.empty and 'confidence' in results_df_validated.columns and 'actual_low_point' in results_df_validated.columns:
                    y_true_arr = results_df_validated['actual_low_point'].astype(int).values
                    y_prob_arr = results_df_validated['confidence'].astype(float).values
                    mask = ~np.isnan(y_prob_arr)
                    y_true_f = y_true_arr[mask]
                    y_prob_f = np.clip(y_prob_arr[mask], 1e-6, 1 - 1e-6)

                    if y_true_f.size > 0 and np.unique(y_true_f).size > 1:
                        brier_value = float(brier_score_loss(y_true_f, y_prob_f))
                        logloss_value = float(log_loss(y_true_f, y_prob_f))

                        # 计算 ECE 与可靠性表（10 等宽分箱）
                        def _ece_and_bins(probs: np.ndarray, truths: np.ndarray, n_bins: int = 10):
                            probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
                            truths = np.asarray(truths, dtype=int)
                            bins = np.linspace(0.0, 1.0, n_bins + 1)
                            bin_ids = np.digitize(probs, bins, right=True) - 1
                            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

                            total_cnt = probs.size
                            ece_sum = 0.0
                            bins_rows = []
                            for k in range(n_bins):
                                lo, hi = bins[k], bins[k+1]
                                idx = (bin_ids == k)
                                cnt = int(idx.sum())
                                if cnt > 0:
                                    avg_conf = float(probs[idx].mean())
                                    acc = float(truths[idx].mean())
                                    gap = abs(acc - avg_conf)
                                    weight = cnt / max(total_cnt, 1)
                                    ece_sum += weight * gap
                                    right_bracket = ')' if k < n_bins - 1 else ']'
                                    bins_rows.append({
                                        'range': f"[{lo:.1f},{hi:.1f}{right_bracket}",
                                        'count': cnt,
                                        'avg_conf': avg_conf,
                                        'acc': acc,
                                        'gap': gap,
                                    })
                                else:
                                    right_bracket = ')' if k < n_bins - 1 else ']'
                                    bins_rows.append({
                                        'range': f"[{lo:.1f},{hi:.1f}{right_bracket}",
                                        'count': 0,
                                        'avg_conf': 0.0,
                                        'acc': 0.0,
                                        'gap': 0.0,
                                    })
                            return float(ece_sum), bins_rows

                        # Original 直接作为基线
                        ece_orig2, bins_orig2 = _ece_and_bins(y_prob_f, y_true_f)
                        ece_value = ece_orig2
                        calib_compare.append({'method': '原始(Original)', 'brier': brier_value, 'logloss': logloss_value, 'ece': ece_orig2})
                        calib_bins_map['original'] = bins_orig2

                        # Platt scaling（用概率作为单特征进行逻辑回归校准）
                        try:
                            X = y_prob_f.reshape(-1, 1)
                            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
                            lr.fit(X, y_true_f)
                            p_platt = lr.predict_proba(X)[:, 1]
                            brier_platt = float(brier_score_loss(y_true_f, p_platt))
                            logloss_platt = float(log_loss(y_true_f, p_platt))
                            ece_platt, bins_platt = _ece_and_bins(p_platt, y_true_f)
                            calib_compare.append({'method': 'Platt(逻辑回归)', 'brier': brier_platt, 'logloss': logloss_platt, 'ece': ece_platt})
                            calib_bins_map['platt'] = bins_platt
                            logger.info(f"Platt校准：Brier={brier_platt:.4f}, LogLoss={logloss_platt:.4f}, ECE(10)={ece_platt:.4f}")
                        except Exception as ce:
                            logger.warning(f"Platt 标定失败: {ce}")

                        # Isotonic Regression
                        try:
                            iso = IsotonicRegression(out_of_bounds='clip')
                            p_iso = iso.fit_transform(y_prob_f, y_true_f)
                            brier_iso = float(brier_score_loss(y_true_f, p_iso))
                            logloss_iso = float(log_loss(y_true_f, p_iso))
                            ece_iso, bins_iso = _ece_and_bins(p_iso, y_true_f)
                            calib_compare.append({'method': 'Isotonic(保序回归)', 'brier': brier_iso, 'logloss': logloss_iso, 'ece': ece_iso})
                            calib_bins_map['isotonic'] = bins_iso
                            logger.info(f"Isotonic校准：Brier={brier_iso:.4f}, LogLoss={logloss_iso:.4f}, ECE(10)={ece_iso:.4f}")
                        except Exception as ie:
                            logger.warning(f"Isotonic 标定失败: {ie}")
                    else:
                        logger.info("概率校准评估：类别单一或样本不足，跳过 Brier/LogLoss/ECE 计算")
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
                ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = os.path.join(reports_dir, f"report_rolling_backtest_{ts_str}.md")

                # 选取关键信号（预测为正类）按置信度排序
                pos_signals = []
                try:
                    pos_df = results_df_validated[results_df_validated['predicted_low_point'] == True].copy()
                    pos_df = pos_df.sort_values(by=['confidence'], ascending=False)
                    for idx, (dt, row) in enumerate(pos_df.head(15).iterrows(), start=1):
                        pos_signals.append({
                            'index': idx,
                            'date': dt.strftime('%Y-%m-%d'),
                            'predicted': '是' if row.get('predicted_low_point') else '否',
                            'actual': '是' if row.get('actual_low_point') else '否',
                            'confidence': row.get('confidence', 0),
                            'future_max_rise': row.get('future_max_rise', 0),
                            'days_to_rise': f"{int(row.get('days_to_target', 0))}" if not pd.isna(row.get('days_to_target')) else "N/A",
                            'predict_price': row.get('predict_price') if row.get('predict_price') is not None else 'N/A',
                            'correct': '✅' if row.get('prediction_correct') else '❌'
                        })
                except Exception as e:
                    pos_signals = [{'index': 1, 'error': f"生成样例行时出现异常: {e}"}]

                # 新增：全区间 Top-N（按 confidence 降序，包含未达阈值）
                top_all_signals = []
                try:
                    all_df = results_df.copy()
                    all_df = all_df.sort_values(by=['confidence'], ascending=False)
                    for idx, (dt, row) in enumerate(all_df.head(15).iterrows(), start=1):
                        top_all_signals.append({
                            'index': idx,
                            'date': dt.strftime('%Y-%m-%d'),
                            'predicted': '是' if row.get('predicted_low_point') else '否',
                            'actual': '是' if row.get('actual_low_point') else '否',
                            'confidence': row.get('confidence', 0),
                            'future_max_rise': row.get('future_max_rise', 0),
                            'days_to_rise': int(row.get('days_to_target') or 0) if row.get('days_to_target') is not None else 0,
                            'predict_price': row.get('predict_price') if row.get('predict_price') is not None else 'N/A',
                            'correct': ('✅' if row.get('prediction_correct') else ('❌' if row.get('prediction_correct') is not None else 'N/A'))
                        })
                except Exception as e:
                    top_all_signals = [{'index': 1, 'error': f"生成Top-N时出现异常: {e}"}]

     
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
                report_lines.append(f"- **策略参数**: rise_threshold={resolve_confidence_param(config, 'rise_threshold', 0.04):.3%}, max_days={config.get('strategy', {}).get('max_days', 20)}")
                report_lines.append(f"- **置信度阈值**: {final_threshold:.3f}")
                report_lines.append(f"- **训练效率**: {training_count}/{len(results)} (节省 {((len(results) - training_count) / len(results) * 100):.1f}%)")
                report_lines.append("")
                
                report_lines.append("## 总体指标")
                report_lines.append(f"- **总预测日期数(可验证)**: {total_predictions_validated}")
                report_lines.append(f"- **正确预测数**: {correct_predictions}")
                report_lines.append(f"- **准确率(Accuracy)**: {success_rate:.3%}")
                report_lines.append(f"- **精确率(Precision)**: {precision:.3%}")
                report_lines.append(f"- **召回率(Recall)**: {recall:.3%}")
                report_lines.append(f"- **F1分数(F1 Score)**: {(2*precision*recall/max(precision+recall, 1e-12)):.3%}")
                report_lines.append(f"- **特异性(Specificity)**: {specificity:.3%}")
                report_lines.append(f"- **平衡准确率(Balanced Accuracy)**: {balanced_acc:.3%}")
                report_lines.append("")

                # 新增：置信度分布诊断（写入报告）
                try:
                    report_lines.append("## 置信度分布诊断")
                    report_lines.append(f"- confidence: 均值={conf_stat['mean']:.3f}, 标准差={conf_stat['std']:.3f}, 最小={conf_stat['min']:.3f}, 最大={conf_stat['max']:.3f}")
                    report_lines.append(f"- 分位数(10/25/50/75/90): {conf_stat['q10']:.3f} / {conf_stat['q25']:.3f} / {conf_stat['q50']:.3f} / {conf_stat['q75']:.3f} / {conf_stat['q90']:.3f}")
                    report_lines.append("")
                    report_lines.append("### confidence 直方分布")
                    report_lines.append("| 区间 | 数量 | 占比 |")
                    report_lines.append("|------|------|------|")
                    for lbl in bin_labels:
                        report_lines.append(f"| {lbl} | {int(bin_counts[lbl])} | {bin_perc[lbl]:.2f}% |")
                    report_lines.append("")
                    report_lines.append(f"- 阈值={final_threshold:.2f}，灰区[{gray_lower:.2f}, {gray_upper:.2f}] 覆盖: {gray_total} 条，占比 {(gray_total/len(results_df)*100 if len(results_df)>0 else 0):.2f}%；灰区中预测正类 {gray_pos} 条，正确 {gray_correct} 条")
                    report_lines.append(f"- 相关性：confidence vs future_max_rise = {corr_final_rise:.3f}，confidence vs prediction_correct = {corr_final_correct:.3f}")
                    report_lines.append("")
                except Exception as e:
                    report_lines.append(f"(置信度分布诊断生成失败: {e})")
                    report_lines.append("")

                report_lines.append("## 概率校准评估")
                if brier_value is not None:
                    report_lines.append(f"- 布里尔分数(Brier Score): {brier_value:.4f}（越低越好）")
                else:
                    report_lines.append(f"- 布里尔分数(Brier Score): N/A")
                if logloss_value is not None:
                    report_lines.append(f"- 对数损失(Log Loss): {logloss_value:.4f}（越低越好）")
                else:
                    report_lines.append(f"- 对数损失(Log Loss): N/A")
                if ece_value is not None:
                    report_lines.append(f"- 期望校准误差(ECE,10 bins): {ece_value:.4f}（越低越好）")
                else:
                    report_lines.append(f"- 期望校准误差(ECE,10 bins): N/A")
                report_lines.append("")

                # 离线概率标定对比实验（不改主逻辑，仅输出对比表格）
                if calib_compare:
                    report_lines.append("### 离线概率标定对比实验（Original vs Platt vs Isotonic）")
                    report_lines.append("| 方法 | 布里尔分数(Brier) | 对数损失(LogLoss) | 期望校准误差(ECE,10) |")
                    report_lines.append("|------|------:|--------:|--------:|")
                    for row in calib_compare:
                        report_lines.append(f"| {row['method']} | {row['brier']:.4f} | {row['logloss']:.4f} | {row['ece']:.4f} |")
                    report_lines.append("")
                    # 可靠性（分箱）表（仅展示每种方法的10个分箱）
                    for key, title in [( 'original','原始(Original)' ), ( 'platt','Platt(逻辑回归)' ), ( 'isotonic','Isotonic(保序回归)' )]:
                        bins_rows = calib_bins_map.get(key)
                        if bins_rows:
                            report_lines.append(f"#### 可靠性分箱 - {title}")
                            report_lines.append("| 置信度区间 | 样本数 | 平均置信度 | 实际比例 | 绝对差 |")
                            report_lines.append("|-----------:|------:|-----------:|--------:|------:|")
                            for br in bins_rows:
                                report_lines.append(f"| {br['range']} | {br['count']} | {br['avg_conf']:.3f} | {br['acc']:.3f} | {br['gap']:.3f} |")
                            report_lines.append("")

                report_lines.append("## 每日预测明细")
                report_lines.append("")
                report_lines.append("**字段说明：**")
                report_lines.append("- **置信度**: AI模型输出的预测概率 (0-1)")
                report_lines.append("- **阈值(used)**: 实际使用的决策阈值，当AI置信度≥此值时做出正向预测")
                report_lines.append("- **策略详情**: 文本合并，包含策略原因与策略指标")
                report_lines.append("")
                report_lines.append("| 日期 | 预测价格 | 预测结果 | 置信度 | 阈值(used) | 实际结果 | 趋势 | 未来最大涨幅 | 达标用时(天) | 预测正确 | 策略详情 |")
                report_lines.append("|------|----------|----------|--------|------------|------|-------------|-------------|----------|----------|------------|")
                for dt, row in results_df.iterrows():
                    date_str = pd.to_datetime(dt).strftime('%Y-%m-%d') if not pd.isna(dt) else ''
                    pp_val = row.get('predict_price')
                    predict_price = f"{float(pp_val):.3f}" if pp_val is not None and not pd.isna(pp_val) else ''
                    predicted = "是" if row.get('predicted_low_point') else "否"
                    confidence = f"{row.get('confidence', 0):.3f}"
                    used_threshold = row.get('used_threshold')
                    used_threshold_str = f"{float(used_threshold):.3f}" if used_threshold is not None and not pd.isna(used_threshold) else "N/A"
                    actual = "是" if row.get('actual_low_point') else "否"
                    # 新增：提取趋势状态
                    trend_str = ''
                    ind = row.get('strategy_indicators')
                    if isinstance(ind, dict):
                        trend_str = ind.get('trend_regime', '')
                    elif isinstance(ind, str):
                        s = ind.strip()
                        if s.startswith('{') and s.endswith('}'):
                            try:
                                d = json.loads(s)
                                if isinstance(d, dict):
                                    trend_str = d.get('trend_regime', '')
                            except Exception:
                                trend_str = ''
                    max_rise = f"{float(row.get('future_max_rise', 0)):.3%}" if not pd.isna(row.get('future_max_rise')) else "N/A"
                    days_to_target = f"{int(row.get('days_to_target', 0))}" if not pd.isna(row.get('days_to_target')) else "N/A"
                    prediction_correct = "是" if row.get('prediction_correct') else "否"
                    if pd.isna(row.get('actual_low_point')):
                        actual = '数据不足'
                    if pd.isna(row.get('prediction_correct')):
                        prediction_correct = '数据不足'
                    # 合并策略原因与策略指标为文本
                    reasons_val = row.get('strategy_reasons')
                    if isinstance(reasons_val, (list, tuple)):
                        reasons_str = ' | '.join(map(str, reasons_val))
                    elif isinstance(reasons_val, str):
                        reasons_str = reasons_val
                    else:
                        reasons_str = ''
                    indicators_val = row.get('strategy_indicators')
                    indicators_kv = ''
                    if isinstance(indicators_val, dict):
                        try:
                            indicators_kv = ', '.join([f"{k}={v}" for k, v in indicators_val.items()])
                        except Exception:
                            indicators_kv = json.dumps(indicators_val, ensure_ascii=False)
                    elif isinstance(indicators_val, str):
                        s = indicators_val.strip()
                        if s.startswith('{') and s.endswith('}'):
                            try:
                                d = json.loads(s)
                                if isinstance(d, dict):
                                    indicators_kv = ', '.join([f"{k}={v}" for k, v in d.items()])
                                else:
                                    indicators_kv = s
                            except Exception:
                                indicators_kv = s
                        else:
                            indicators_kv = s
                    details_parts = []
                    if reasons_str:
                        details_parts.append(f"原因: {reasons_str}")
                    if indicators_kv:
                        details_parts.append(f"指标: {indicators_kv}")
                    details_str = '； '.join(details_parts)
                    report_lines.append(f"| {date_str} | {predict_price} | {predicted} | {confidence} | {used_threshold_str} | {actual} | {trend_str} | {max_rise} | {days_to_target} | {prediction_correct} | {details_str} |")
                report_lines.append("")

                # 新增：趋势分布与命中率（含震荡区间）
                try:
                    if 'strategy_indicators' in results_df.columns and 'prediction_correct' in results_df.columns:
                        def _extract_trend_for_group(v):
                            if isinstance(v, dict):
                                return v.get('trend_regime', '')
                            if isinstance(v, str):
                                s = v.strip()
                                if s.startswith('{') and s.endswith('}'):
                                    try:
                                        d = json.loads(s)
                                        if isinstance(d, dict):
                                            return d.get('trend_regime', '')
                                    except Exception:
                                        return ''
                            return ''
                        trend_series = results_df['strategy_indicators'].apply(_extract_trend_for_group)
                        correct_series = results_df['prediction_correct'].fillna(False).astype(bool)
                        # 统计
                        stats = {}
                        for tr in ['bull', 'sideways', 'bear', '']:
                            mask = (trend_series == tr)
                            cnt = int(mask.sum())
                            if cnt > 0:
                                hit = int(correct_series[mask].sum())
                                rate = hit / cnt if cnt > 0 else 0.0
                                stats[tr if tr else 'unknown'] = (cnt, hit, rate)
                        if stats:
                            report_lines.append("## 趋势分布与命中率（含震荡区间sideways）")
                            report_lines.append("| 趋势 | 样本数 | 命中数 | 命中率 |")
                            report_lines.append("|------|------:|------:|------:|")
                            for k in ['bull', 'sideways', 'bear', 'unknown']:
                                if k in stats:
                                    c, h, r = stats[k]
                                    report_lines.append(f"| {k} | {c} | {h} | {r:.3%} |")
                            report_lines.append("")

                            # 新增：分趋势的置信度分布统计
                            try:
                                if 'confidence' in results_df.columns and 'strategy_confidence' in results_df.columns:
                                    report_lines.append("### 分趋势置信度分布统计")
                                    report_lines.append("| 趋势 | 样本数 | AI置信度均值 | AI置信度标准差 | 策略置信度均值 | 策略置信度标准差 | 命中率 |")
                                    report_lines.append("|------|------:|------------:|-------------:|---------------:|----------------:|------:|")
                                    
                                    ai_conf_series = pd.to_numeric(results_df['confidence'], errors='coerce')
                                    strategy_conf_series = pd.to_numeric(results_df['strategy_confidence'], errors='coerce')
                                    
                                    for k in ['bull', 'sideways', 'bear', 'unknown']:
                                        if k in stats:
                                            mask = (trend_series == ('' if k=='unknown' else k))
                                            cnt = int(mask.sum())
                                            if cnt > 0:
                                                ai_mean = float(ai_conf_series[mask].mean()) if not ai_conf_series[mask].isna().all() else 0.0
                                                ai_std = float(ai_conf_series[mask].std()) if not ai_conf_series[mask].isna().all() else 0.0
                                                strategy_mean = float(strategy_conf_series[mask].mean()) if not strategy_conf_series[mask].isna().all() else 0.0
                                                strategy_std = float(strategy_conf_series[mask].std()) if not strategy_conf_series[mask].isna().all() else 0.0
                                                _, _, hit_rate = stats[k]
                                                report_lines.append(f"| {k} | {cnt} | {ai_mean:.3f} | {ai_std:.3f} | {strategy_mean:.3f} | {strategy_std:.3f} | {hit_rate:.2%} |")
                                    report_lines.append("")
                            except Exception:
                                pass

                            # 震荡区间(sideways)有效性验证
                            try:
                                # 提取价格和MA20
                                price_series = results_df.get('predict_price')
                                def _extract_ma20(v):
                                    if isinstance(v, dict):
                                        return v.get('ma20', float('nan'))
                                    if isinstance(v, str):
                                        s = v.strip()
                                        if s.startswith('{') and s.endswith('}'):
                                            try:
                                                d = json.loads(s)
                                                if isinstance(d, dict):
                                                    return d.get('ma20', float('nan'))
                                            except Exception:
                                                return float('nan')
                                    return float('nan')
                                ma20_series = results_df['strategy_indicators'].apply(_extract_ma20) if 'strategy_indicators' in results_df.columns else None

                                # 计算日内波动(|收益|)与相对MA20偏离
                                vol_series = None
                                if isinstance(price_series, pd.Series):
                                    # 按索引排序，避免错乱
                                    price_series = price_series.sort_index()
                                    vol_series = price_series.pct_change().abs()

                                dev_series = None
                                if isinstance(price_series, pd.Series) and isinstance(ma20_series, pd.Series):
                                    with pd.option_context('mode.use_inf_as_na', True):
                                        dev_series = ((price_series - ma20_series).abs() / ma20_series.replace(0, pd.NA)).astype(float)

                                # 分组统计
                                if isinstance(trend_series, pd.Series):
                                    report_lines.append("## 震荡区间有效性验证")
                                    report_lines.append("- 指标解释：|日收益|中位数用于衡量波动强度；MA20相对偏离中位数用于测量价格是否围绕均线波动；近均线占比(|偏离|≤1%)用于判断贴近均线的天数占比。")
                                    report_lines.append("| 趋势 | 样本数 | |日收益|中位数 | MA20相对偏离中位数 | 近均线占比(|偏离|≤1%) |")
                                    report_lines.append("|------|------:|-------------:|--------------------:|-----------------------:|")
                                    for k in ['bull', 'sideways', 'bear', 'unknown']:
                                        mask = (trend_series == ('' if k=='unknown' else k))
                                        cnt = int(mask.sum())
                                        if cnt == 0:
                                            continue
                                        vol_med = float(vol_series[mask].median()) if isinstance(vol_series, pd.Series) else float('nan')
                                        dev_med = float(dev_series[mask].median()) if isinstance(dev_series, pd.Series) else float('nan')
                                        near_ma = None
                                        if isinstance(dev_series, pd.Series):
                                            near_ma = float((dev_series[mask] <= 0.01).mean())
                                        report_lines.append(f"| {k} | {cnt} | {vol_med:.3%} | {dev_med:.3%} | {near_ma:.3%} |")
                                    report_lines.append("")
                            except Exception:
                                pass
                except Exception:
                    pass
                report_lines.append("")
                report_lines.append(f"**策略参数**: 涨幅阈值={resolve_confidence_param(config, 'rise_threshold', 0.04):.3%}, 最大观察天数={config.get('strategy', {}).get('max_days', 20)}, RSI超卖={config.get('strategy', {}).get('rsi_oversold', 30)}, RSI偏低={config.get('strategy', {}).get('rsi_low', 40)}, 置信度阈值={final_threshold:.3f}")
                report_lines.append("")

                report_lines.append("## 关键信号详情（按置信度降序，最多15条）")
                report_lines.append("| 序号 | 日期 | 预测 | 实际 | 置信度 | 未来最大涨幅 | 用时天数 | 预测价 | 结果 |")
                report_lines.append("|------|------|------|------|--------|------------|----------|----------|---------|")
                if len(pos_signals) > 0:
                    for signal in pos_signals:
                        if 'error' in signal:
                            report_lines.append(f"| {signal.get('index', 1)} | - | - | - | - | - | - | - | {signal['error']} |")
                        else:
                            pp_str = (f"{float(signal['predict_price']):.3f}" if isinstance(signal.get('predict_price'), (int, float)) else str(signal.get('predict_price')))
                            report_lines.append(f"| {signal['index']} | {signal['date']} | {signal['predicted']} | {signal['actual']} | {signal['confidence']:.3f} | {signal['future_max_rise']:.3%} | {signal['days_to_rise']} | {pp_str} | {signal['correct']} |")
                else:
                    report_lines.append("- (本次无正类信号或无法生成样例)")
                report_lines.append("")

                # 新增：全区间 Top-N confidence（包含未达阈值）")
                report_lines.append("## 全区间 Top-N confidence（包含未达阈值）")
                report_lines.append("| 序号 | 日期 | 预测 | 实际 | 置信度 | 未来最大涨幅 | 用时天数 | 预测价 | 结果 |")
                report_lines.append("|------|------|------|------|--------|-------------|----------|---------|------|")
                if len(top_all_signals) > 0:
                    for signal in top_all_signals:
                        if 'error' in signal:
                            report_lines.append(f"| {signal.get('index', 1)} | - | - | - | - | - | - | - | {signal['error']} |")
                        else:
                            pp_str = (f"{float(signal['predict_price']):.3f}" if isinstance(signal.get('predict_price'), (int, float)) else str(signal.get('predict_price')))
                            report_lines.append(f"| {signal['index']} | {signal['date']} | {signal['predicted']} | {signal['actual']} | {signal['confidence']:.3f} | {signal['future_max_rise']:.3%} | {signal['days_to_rise']} | {pp_str} | {signal['correct']} |")
                else:
                    report_lines.append("- (无法生成Top-N列表)")
                report_lines.append("")

                report_lines.append("## 策略参数详情")
                report_lines.append(f"- **涨幅阈值**: {resolve_confidence_param(config, 'rise_threshold', 0.04):.3%}")
                report_lines.append(f"- **最大观察天数**: {config.get('strategy', {}).get('max_days', 20)}")
                report_lines.append(f"- **RSI超卖阈值**: {config.get('strategy', {}).get('rsi_oversold', 30)}")
                report_lines.append(f"- **RSI偏低阈值**: {config.get('strategy', {}).get('rsi_low', 40)}")
                report_lines.append(f"- **最终置信度阈值**: {final_threshold:.3f}")
                report_lines.append("")

                report_lines.append("> **免责声明**: 本报告由脚本自动生成，仅用于策略与模型评估，不构成投资建议。")

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(report_lines))
                logger.info(f"📄 回测报告已生成: {os.path.relpath(report_path)}")

                # 新增：导出每日明细CSV（包含 used_threshold）
                try:
                    csv_dir = os.path.join(base_results_dir, 'csv')
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_path = os.path.join(csv_dir, f"daily_details_rolling_backtest_{ts_str}.csv")

                    csv_df = results_df.copy()
                    # 插入日期列（格式化）
                    try:
                        dates = pd.to_datetime(csv_df.index)
                        csv_df.insert(0, 'date', dates.strftime('%Y-%m-%d'))
                    except Exception:
                        csv_df.insert(0, 'date', csv_df.index.astype(str))

                    # 转换策略原因与指标为字符串，便于CSV阅读
                    if 'strategy_reasons' in csv_df.columns:
                        csv_df['strategy_reasons'] = csv_df['strategy_reasons'].apply(
                            lambda x: ' | '.join(x) if isinstance(x, (list, tuple)) else ('' if x is None else str(x))
                        )
                    # 先从 strategy_indicators 提取 trend_regime（在转字符串之前进行展平）
                    try:
                        if 'strategy_indicators' in csv_df.columns:
                            def _extract_trend_regime(v):
                                if isinstance(v, dict):
                                    return v.get('trend_regime', '')
                                # 某些情况下可能已是JSON字符串
                                if isinstance(v, str):
                                    s = v.strip()
                                    if s.startswith('{') and s.endswith('}'):
                                        try:
                                            d = json.loads(s)
                                            if isinstance(d, dict):
                                                return d.get('trend_regime', '')
                                        except Exception:
                                            return ''
                                return ''
                            csv_df['trend_regime'] = csv_df['strategy_indicators'].apply(_extract_trend_regime)
                    except Exception:
                        # 出错时保持列为空，避免中断导出
                        csv_df['trend_regime'] = ''
                    if 'strategy_indicators' in csv_df.columns:
                        csv_df['strategy_indicators'] = csv_df['strategy_indicators'].apply(
                            lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else ('' if d is None else str(d))
                        )

                    # 格式化置信度保留两位小数
                    if 'confidence' in csv_df.columns:
                        try:
                            csv_df['confidence'] = csv_df['confidence'].apply(
                                lambda v: (f"{float(v):.2f}" if (v is not None and not pd.isna(v)) else '')
                            )
                        except Exception:
                            pass

                    # 将未来最大涨幅格式化为两位小数的百分比
                    if 'future_max_rise' in csv_df.columns:
                        try:
                            csv_df['future_max_rise'] = csv_df['future_max_rise'].apply(
                                lambda v: (f"{float(v) * 100:.2f}%" if (v is not None and not pd.isna(v)) else '')
                            )
                        except Exception:
                            pass

                    # 仅保留关心的列（若缺失则自动跳过）
                    preferred_cols = ['date', 'predict_price', 'predicted_low_point', 'confidence',
                                      'used_threshold', 'actual_low_point', 'trend_regime', 'future_max_rise', 'days_to_target', 'prediction_correct']
                    cols = [c for c in preferred_cols if c in csv_df.columns]
                    
                    # 创建中文列名映射（移除策略原因与策略指标）
                    chinese_column_mapping = {
                        'date': '日期',
                        'predict_price': '预测价格',
                        'predicted_low_point': '预测低点',
                        'confidence': '置信度',
                        'used_threshold': '使用阈值',
                        'actual_low_point': '实际低点',
                        'trend_regime': '趋势状态',
                        'future_max_rise': '未来最大涨幅',
                        'days_to_target': '达标用时(天)',
                        'prediction_correct': '预测正确'
                    }
                    
                    # 重命名列名为中文
                    csv_df_chinese = csv_df[cols].copy()
                    csv_df_chinese.columns = [chinese_column_mapping.get(col, col) for col in csv_df_chinese.columns]
                    csv_df_chinese.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    logger.info(f"🧾 每日明细已导出CSV: {os.path.relpath(csv_path)}")
                except Exception as e:
                    logger.warning(f"导出每日明细CSV失败: {e}")

                # 准备返回结果与输出
                metrics = {
                    'success_rate': locals().get('success_rate', 0.0),
                    'precision': locals().get('precision', 0.0),
                    'recall': locals().get('recall', 0.0),
                    'specificity': locals().get('specificity', 0.0),
                    'balanced_acc': locals().get('balanced_acc', 0.0),
                    'f1': locals().get('f1', 0.0),
                    'tp': int(locals().get('tp', 0)),
                    'tn': int(locals().get('tn', 0)),
                    'fp': int(locals().get('fp', 0)),
                    'fn': int(locals().get('fn', 0)),
                    'total_predictions': int(locals().get('total_predictions_validated', 0)),
                    'training_count': int(locals().get('training_count', 0))
                }
                return {
                    'success': True,
                    'metrics': metrics,
                    'report_path': report_path
                }
            else:
                # 当不生成报告时，仍需要返回结果
                metrics = {
                    'success_rate': locals().get('success_rate', 0.0),
                    'precision': locals().get('precision', 0.0),
                    'recall': locals().get('recall', 0.0),
                    'specificity': locals().get('specificity', 0.0),
                    'balanced_acc': locals().get('balanced_acc', 0.0),
                    'f1': locals().get('f1', 0.0),
                    'tp': int(locals().get('tp', 0)),
                    'tn': int(locals().get('tn', 0)),
                    'fp': int(locals().get('fp', 0)),
                    'fn': int(locals().get('fn', 0)),
                    'total_predictions': int(locals().get('total_predictions_validated', 0)),
                    'training_count': int(locals().get('training_count', 0))
                }
                return {
                    'success': True,
                    'metrics': metrics
                }
    except Exception as e:
        logger.error(f"滚动回测发生异常: {e}")
        return {
            'success': False,
            'error': str(e)
        }



def run_rolling_backtest_with_return(start_date_str: str, end_date_str: str, training_window_days: int = 365,
                                     reuse_model: bool = True, retrain_interval_days: int = None,
                                     generate_report: bool = True, report_dir: str = None):
    """
    兼容入口：与 run_rolling_backtest 相同，只是显式返回其结果，供网格测试脚本调用。
    """
    return run_rolling_backtest(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        training_window_days=training_window_days,
        reuse_model=reuse_model,
        retrain_interval_days=retrain_interval_days,
        generate_report=generate_report,
        report_dir=report_dir,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行滚动回测')
    parser.add_argument('--start_date', required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--training_window_days', type=int, default=365, help='训练窗口天数')
    parser.add_argument('--reuse_model', action='store_true', default=True, help='是否重用模型')
    parser.add_argument('--retrain_interval_days', type=int, help='重训练间隔天数')
    parser.add_argument('--no_report', action='store_true', help='不生成报告')
    parser.add_argument('--report_dir', help='报告目录')

    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    result = run_rolling_backtest(
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        training_window_days=args.training_window_days,
        reuse_model=args.reuse_model,
        retrain_interval_days=args.retrain_interval_days,
        generate_report=not args.no_report,
        report_dir=args.report_dir,
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if result['success']:
        print(format_backtest_summary(result, project_root=project_root))
    else:
        print(f"回测失败: {result['error']}")
        sys.exit(1)


