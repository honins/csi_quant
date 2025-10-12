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
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging, PredictionResult

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
    """预测单个日期"""
    try:
        predict_date = datetime.strptime(predict_date_str, '%Y-%m-%d')
        
        # 获取预测所需的数据（获取最近1年数据用于预测）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        market_data = data_module.get_history_data(start_date, end_date)
        if market_data is None or market_data.empty:
            logger.error("无法获取市场数据")
            return None
            
        # 处理日期格式
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # 🔧 修复：正确获取预测日期对应的数据
        # 首先找到预测日期在数据中的位置
        predict_date_data = market_data[market_data['date'] == predict_date]
        if predict_date_data.empty:
            logger.warning(f"无法找到 {predict_date_str} 的数据")
            return None
            
        # 获取预测日期之前的数据用于构建特征（包含预测日期当天）
        train_data = market_data[market_data['date'] <= predict_date].copy()
        if train_data.empty:
            logger.warning(f"无法获取 {predict_date_str} 之前的数据")
            return None
            
        if len(train_data) < 100:  # 确保有足够的训练数据
            logger.warning(f"训练数据不足 ({len(train_data)} 条)")
            return None
            
        # 🔧 修复：获取预测日期的收盘价作为预测价格
        predict_date_data = market_data[market_data['date'] == predict_date]
        predict_price = predict_date_data.iloc[0]['close'] if not predict_date_data.empty else None
        
        # 🎯 关键修复：使用包含预测日期的完整数据进行特征计算
        # 这样每个日期都会有不同的技术指标和特征
        predict_day_data = train_data.copy()
        prediction_result = ai_optimizer.predict_low_point(
            predict_day_data, predict_date_str
        )
        
        if prediction_result:
            return {
                'date': predict_date_str,
                'predicted_low_point': prediction_result.get('is_low_point', False),
                'confidence': prediction_result.get('confidence', 0.0),
                'predict_price': predict_price  # 使用实际的收盘价
            }
        else:
            logger.error(f"预测 {predict_date_str} 返回空结果")
            return None
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
        
        report_lines.append("## 🎯 预测汇总")
        report_lines.append(f"- **总预测天数**: {total_predictions}")
        report_lines.append(f"- **预测为相对低点**: {low_point_predictions} 天 ({low_point_predictions/total_predictions*100:.1f}%)")
        report_lines.append(f"- **高置信度预测**: {high_confidence_predictions} 天 ({high_confidence_predictions/total_predictions*100:.1f}%)")
        report_lines.append(f"- **平均置信度**: {avg_confidence:.4f}")
        report_lines.append("")
        
        report_lines.append("## 📈 置信度分析")
        report_lines.append(f"- **置信度均值**: {confidence_stats['mean']:.4f}")
        report_lines.append(f"- **置信度标准差**: {confidence_stats['std']:.4f}")
        report_lines.append(f"- **最低置信度**: {confidence_stats['min']:.4f}")
        report_lines.append(f"- **最高置信度**: {confidence_stats['max']:.4f}")
        report_lines.append("")
        
        # 预测详情
        if low_point_predictions > 0:
            report_lines.append("## 🎯 预测的相对低点")
            report_lines.append("| 日期 | 置信度 | 备注 |")
            report_lines.append("| --- | --- | --- |")
            for r in results:
                if r['predicted_low_point']:
                    report_lines.append(f"| {r['date']} | {r['confidence']:.4f} | 预测为相对低点 |")
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
            report_lines.append(f"| {i} | {r['date']} | {prediction_text} | {r['confidence']:.4f} |")
        report_lines.append("")
        
        # 每日预测明细 - 使用与历史回测报告一致的字段格式
        report_lines.append("## 每日预测明细")
        report_lines.append("| 日期 | 预测价格 | 预测结果 | 置信度 | 阈值(used) | 实际结果 | 趋势 | 未来最大涨幅 | 达标用时(天) | 预测正确 |")
        report_lines.append("|------|----------|----------|--------|------------|----------|------|-------------|-------------|----------|")
        for r in results:
            prediction_text = "是" if r['predicted_low_point'] else "否"
            predict_price = r.get('predict_price', 'N/A')
            # 由于是最近预测，实际结果等字段暂时显示为待验证
            report_lines.append(f"| {r['date']} | {predict_price} | {prediction_text} | {r['confidence']:.2f} | 0.50 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |")
        report_lines.append("")
        
        report_lines.append("> **免责声明**: 本报告由AI模型自动生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。")
        
        # 写入Markdown报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"📄 预测报告已生成: {os.path.relpath(report_path)}")
        
        # 生成CSV文件
        csv_data = []
        for r in results:
            csv_data.append({
                '日期': r['date'],
                '预测为低点': r['predicted_low_point'],
                '置信度': r['confidence'],
                '预测结果': '相对低点' if r['predicted_low_point'] else '非相对低点',
                '置信度等级': '高' if r['confidence'] > 0.5 else '中' if r['confidence'] > 0.3 else '低'
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
        
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'system.yaml')
        config = load_config(config_path=config_path)
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        
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
                logger.info(f"  📅 {date_str}: {is_low}相对低点 (置信度: {confidence:.2f}%)")
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