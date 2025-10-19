#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最近一个月预测脚本
批量预测最近一个月的交易日数据
"""

import sys
import os
import logging
import pandas as pd
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved as AIOptimizer
from src.utils.config_loader import load_config
from src.prediction.prediction_utils import setup_logging, PredictionResult, predict_and_validate

# 设置日志
setup_logging()
logger = logging.getLogger("RecentMonthPredictor")

def get_recent_trading_days(data_file, days=30):
    """获取最近的交易日"""
    try:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 获取最近的交易日
        recent_days = df.tail(days)['date'].dt.strftime('%Y-%m-%d').tolist()
        return recent_days
    except Exception as e:
        logger.error(f"获取交易日失败: {e}")
        return []

def predict_single_date(predict_date_str, config, data_module, strategy_module, ai_optimizer):
    """预测单个日期（包含验证）"""
    try:
        predict_date = datetime.strptime(predict_date_str, '%Y-%m-%d')
        
        # 使用统一的预测+验证流程（不在此处训练，使用已训练模型）
        pr: PredictionResult = predict_and_validate(
            predict_date=predict_date,
            data_module=data_module,
            strategy_module=strategy_module,
            ai_optimizer=ai_optimizer,
            config=config,
            logger=logger,
            force_retrain=False,
            only_use_trained_model=True
        )
        
        # 组装字典结果，便于后续生成报告/CSV
        return {
            'date': predict_date_str,
            'predicted_low_point': bool(pr.predicted_low_point) if pr.predicted_low_point is not None else False,
            'actual_low_point': pr.actual_low_point,
            'prediction_correct': pr.prediction_correct,
            'confidence': float(pr.confidence) if pr.confidence is not None else 0.0,
            'predict_price': pr.predict_price,
            'used_threshold': pr.used_threshold,
            'future_max_rise': pr.future_max_rise,
            'days_to_rise': pr.days_to_rise,
            'days_to_target': pr.days_to_target,
            'strategy_indicators': pr.strategy_indicators
        }
    except Exception as e:
        logger.error(f"预测 {predict_date_str} 失败: {e}")
        return None

def generate_prediction_report(results, start_date, end_date, config):
    """生成预测报告"""
    try:
        # 创建报告目录
        base_results_dir = os.path.join(project_root, 'results')
        reports_dir = os.path.join(base_results_dir, 'reports')
        csv_dir = os.path.join(base_results_dir, 'csv')
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 报告文件路径
        report_filename = f'report_recent_month_prediction_{timestamp}.md'
        report_path = os.path.join(reports_dir, report_filename)
        
        # CSV文件路径
        csv_filename = f'recent_month_prediction_{timestamp}.csv'
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # 统计数据
        total_predictions = len(results)
        low_point_predictions = sum(1 for r in results if r['predicted_low_point'])
        high_confidence_predictions = sum(1 for r in results if r['confidence'] > 0.5)
        avg_confidence = sum(r['confidence'] for r in results) / total_predictions if total_predictions > 0 else 0
        
        # 置信度分布统计
        confidences = [r['confidence'] for r in results]
        confidence_stats = {
            'mean': sum(confidences) / len(confidences) if confidences else 0,
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0,
            'std': (sum((x - avg_confidence) ** 2 for x in confidences) / len(confidences)) ** 0.5 if confidences else 0
        }
        
        # 生成Markdown报告
        report_lines = []
        report_lines.append("# 📈 最近一个月预测报告")
        report_lines.append("")
        report_lines.append("## 📊 基本信息")
        report_lines.append(f"- **预测期间**: {start_date} 至 {end_date}")
        report_lines.append(f"- **生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report_lines.append(f"- **报告编号**: `{timestamp}`")
        report_lines.append(f"- **使用模型**: AI优化模型")
        report_lines.append("")
        
        
        # 预测详情
        if low_point_predictions > 0:
            report_lines.append("## 🎯 预测的相对低点")
            report_lines.append("| 日期 | 置信度 | 备注 |")
            report_lines.append("| --- | --- | --- |")
            for r in results:
                if r['predicted_low_point']:
                    report_lines.append(f"| {r['date']} | {r['confidence']:.2f} | 预测为相对低点 |")
            report_lines.append("")
        else:
            report_lines.append("## 📊 预测结果")
            report_lines.append("**最近一个月未发现明显的相对低点**")
            report_lines.append("")
        
        # 置信度最高的预测
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        report_lines.append("## 🔝 置信度最高的预测")
        report_lines.append("| 排名 | 日期 | 预测结果 | 置信度 |")
        report_lines.append("| --- | --- | --- | --- |")
        for i, r in enumerate(sorted_results[:10], 1):
            prediction_text = "相对低点" if r['predicted_low_point'] else "非相对低点"
            report_lines.append(f"| {i} | {r['date']} | {prediction_text} | {r['confidence']:.2f} |")
        report_lines.append("")
        
        # 每日预测明细 - 使用与历史回测报告一致的字段格式
        report_lines.append("## 每日预测明细")
        report_lines.append("| 日期 | 预测价格 | 预测结果 | 置信度 | 阈值(used) | 实际结果 | 趋势 | 未来最大涨幅 | 达标用时(天) | 预测正确 |")
        report_lines.append("|------|----------|----------|--------|------------|----------|------|-------------|-------------|----------|")
        for r in results:
            prediction_text = "是" if r['predicted_low_point'] else "否"
            predict_price = r.get('predict_price', 'N/A')
            used_thr = r.get('used_threshold')
            used_thr = used_thr if isinstance(used_thr, (int, float)) else 0.50
            actual_val = r.get('actual_low_point')
            actual_text = "是" if actual_val else "否" if actual_val is not None else "数据不足"
            pc = r.get('prediction_correct')
            prediction_success_text = "√" if pc is True else "×" if pc is False else "N/A"
            ind = r.get('strategy_indicators') or {}
            regime = ind.get('trend_regime')
            trend_text = (
                '上升' if regime == 'bull' else '下降' if regime == 'bear' else '横盘' if regime == 'sideways' else 'N/A'
            )
            max_rise = r.get('future_max_rise')
            max_rise_text = f"{max_rise*100:.2f}%" if isinstance(max_rise, (int, float)) else "N/A"
            dtt = r.get('days_to_target')
            if dtt is None:
                days_to_target_text = "N/A"
            elif dtt == 0:
                days_to_target_text = "未达标"
            else:
                days_to_target_text = str(int(dtt))
            report_lines.append(f"| {r['date']} | {predict_price} | {prediction_text} | {r['confidence']:.2f} | {used_thr:.2f} | {actual_text} | {trend_text} | {max_rise_text} | {days_to_target_text} | {prediction_success_text} |")
        report_lines.append("")
        
        report_lines.append("> **免责声明**: 本报告由AI模型自动生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。")
        
        # 写入Markdown报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"📄 预测报告已生成: {os.path.relpath(report_path)}")
        
        # 生成CSV文件
        csv_data = []
        for r in results:
            used_threshold = r.get('used_threshold')
            used_threshold_str = f"{float(used_threshold):.2f}" if used_threshold is not None and not pd.isna(used_threshold) else "N/A"
            pc = r.get('prediction_correct')
            csv_data.append({
                '日期': r['date'],
                '预测为低点': r['predicted_low_point'],
                '置信度': f"{r['confidence']:.2f}",
                '置信度阈值': used_threshold_str,
                '实际结果': '是' if r.get('actual_low_point') else '否',
                '预测成功': '是' if pc is True else ('否' if pc is False else 'N/A')
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"🧾 预测明细已导出CSV: {os.path.relpath(csv_path)}")
        
        return {
            'report_path': report_path,
            'csv_path': csv_path,
            'total_predictions': total_predictions,
            'low_point_predictions': low_point_predictions,
            'high_confidence_predictions': high_confidence_predictions,
            'avg_confidence': avg_confidence
        }
        
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        return None

def main():
    """主函数"""
    try:
        logger.info("开始最近一个月预测...")
        
        # 加载合并配置（system.yaml + strategy.yaml + optimized_params.yaml）
        config = load_config()
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        
        # 尝试预加载已保存模型，避免仅用已训练模型时提前返回
        try:
            if getattr(ai_optimizer, 'model', None) is None:
                loaded = ai_optimizer._load_model()
                logger.info(f"预加载模型: {'成功' if loaded else '失败'}")
        except Exception as _e:
            logger.warning(f"预加载模型异常: {_e}")
        
        # 获取最近30个交易日
        data_file = "data/SHSE.000905_1d.csv"
        recent_days = get_recent_trading_days(data_file, 30)
        
        if not recent_days:
            logger.error("无法获取最近的交易日")
            return False
            
        logger.info(f"将预测最近 {len(recent_days)} 个交易日")
        
        # 批量预测
        results = []
        for i, date_str in enumerate(recent_days, 1):
            logger.info(f"预测进度: {i}/{len(recent_days)} - {date_str}")
            
            result = predict_single_date(date_str, config, data_module, strategy_module, ai_optimizer)
            if result:
                results.append(result)
                
                # 输出预测结果
                is_low = "是" if result['predicted_low_point'] else "否"
                confidence = result['confidence'] * 100
                actual_text = "是" if result.get('actual_low_point') else "否" if result.get('actual_low_point') is not None else "数据不足"
                success_text = "是" if (result.get('prediction_correct') is True) else "否"
                logger.info(f"  📅 {date_str}: {is_low}相对低点 (置信度: {confidence:.2f}%) ｜ 实际: {actual_text} ｜ 预测成功: {success_text}")
            else:
                logger.warning(f"  ❌ {date_str}: 预测失败")
        
        # 汇总结果
        if results:
            logger.info("\n" + "="*60)
            logger.info("📊 最近一个月预测汇总")
            logger.info("="*60)
            
            low_points = [r for r in results if r['predicted_low_point']]
            high_confidence = [r for r in results if r['confidence'] > 0.5]
            
            logger.info(f"总预测天数: {len(results)}")
            logger.info(f"预测为相对低点: {len(low_points)} 天")
            logger.info(f"高置信度预测: {len(high_confidence)} 天")
            
            if low_points:
                logger.info("\n🎯 预测的相对低点日期:")
                for lp in low_points:
                    logger.info(f"  📅 {lp['date']}: 置信度 {lp['confidence']*100:.2f}%")
            else:
                logger.info("\n📈 最近一个月未发现明显的相对低点")
                
            # 显示最高置信度的几个预测
            sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            logger.info("\n🔝 置信度最高的5个预测:")
            for i, r in enumerate(sorted_results[:5], 1):
                is_low = "相对低点" if r['predicted_low_point'] else "非低点"
                logger.info(f"  {i}. {r['date']}: {is_low} (置信度: {r['confidence']*100:.2f}%)")
            
            # 生成预测报告
            logger.info("\n📄 正在生成预测报告...")
            start_date_str = recent_days[0] if recent_days else ""
            end_date_str = recent_days[-1] if recent_days else ""
            report_info = generate_prediction_report(results, start_date_str, end_date_str, config)
            
            if report_info:
                logger.info("\n📋 报告生成完成:")
                logger.info(f"  📄 Markdown报告: {os.path.relpath(report_info['report_path'])}")
                logger.info(f"  🧾 CSV明细: {os.path.relpath(report_info['csv_path'])}")
                logger.info(f"  📊 统计信息: {report_info['total_predictions']}天预测, {report_info['low_point_predictions']}个低点, 平均置信度{report_info['avg_confidence']:.4f}")
            else:
                logger.warning("⚠️ 报告生成失败")
                
        else:
            logger.error("没有成功的预测结果")
            return False
            
        logger.info("\n✅ 最近一个月预测完成")
        return True
        
    except Exception as e:
        logger.error(f"预测过程发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)