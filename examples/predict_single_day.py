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
import json
from datetime import datetime

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved as AIOptimizer
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging, predict_and_validate
from src.utils.trade_date import is_trading_day

def save_prediction_results(prediction_result, predict_date_str, config, market_data=None, technical_indicators=None, model_analysis=None, detailed_analysis=None):
    """
    保存预测结果到文件
    
    Args:
        prediction_result: 预测结果对象
        predict_date_str: 预测日期字符串
        config: 配置信息
        market_data: 市场数据字典
        technical_indicators: 技术指标字典
        model_analysis: 模型分析结果字典
        detailed_analysis: 详细分析数据字典
    """
    try:
        # 确保results目录及其子目录存在
        results_path = config.get('results', {}).get('save_path', 'results')
        
        # 创建子目录结构
        single_predictions_dir = os.path.join(results_path, 'single_predictions')
        reports_dir = os.path.join(results_path, 'reports')
        history_dir = os.path.join(results_path, 'history')
        
        for directory in [results_path, single_predictions_dir, reports_dir, history_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 准备完整的结果数据
        full_results = {
            'timestamp': timestamp,
            'prediction_date': predict_date_str,
            'prediction_time': datetime.now().isoformat(),
            'model_info': {
                'model_file': model_analysis.get('model_file', '') if model_analysis else '',
                'model_age_hours': model_analysis.get('model_age_hours', 0) if model_analysis else 0,
                'feature_count': model_analysis.get('feature_count', 0) if model_analysis else 0,
                'model_type': model_analysis.get('model_type', '') if model_analysis else ''
            },
            'market_data': market_data or {},
            'technical_indicators': technical_indicators or {},
            'prediction_results': {
                'is_predicted_low_point': prediction_result.predicted_low_point if prediction_result else False,
                'confidence': prediction_result.confidence if prediction_result else 0.0,
                'actual_low_point': prediction_result.actual_low_point if prediction_result else None,
                'prediction_correct': prediction_result.prediction_correct if prediction_result else None,
                'future_max_rise': prediction_result.future_max_rise if prediction_result else None,
                'days_to_rise': prediction_result.days_to_rise if prediction_result else None,
                'predict_price': prediction_result.predict_price if prediction_result else None
            },
            'model_analysis': model_analysis or {}
        }
        
        # 保存JSON格式结果到单独预测目录
        json_filename = f'prediction_{predict_date_str}_{timestamp}.json'
        json_path = os.path.join(single_predictions_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存可读Markdown报告到报告目录
        md_filename = f'report_{predict_date_str}_{timestamp}.md'
        md_path = os.path.join(reports_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 📋 单日相对低点预测报告\n\n")
            f.write(f"## 📊 基本信息\n\n")
            f.write(f"- **🎯 预测日期**: {predict_date_str}\n")
            f.write(f"- **🕐 生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"- **📊 报告编号**: `{timestamp}`\n\n")
            
            # 模型信息
            if model_analysis:
                f.write("## 🤖 模型信息\n\n")
                f.write("| 项目 | 值 |\n")
                f.write("| --- | --- |\n")
                f.write(f"| 模型文件 | `{model_analysis.get('model_file', 'N/A')}` |\n")
                f.write(f"| 模型类型 | {model_analysis.get('model_type', 'N/A')} |\n")
                f.write(f"| 特征维度 | {model_analysis.get('feature_count', 'N/A')} |\n")
                f.write(f"| 模型年龄 | {model_analysis.get('model_age_description', 'N/A')} |\n\n")
            
            # 市场数据
            if market_data:
                f.write("## 📊 市场数据\n\n")
                f.write("| 指标 | 数值 |\n")
                f.write("| --- | --- |\n")
                f.write(f"| 收盘价 | **{market_data.get('close_price', 'N/A')}** |\n")
                f.write(f"| 涨跌幅 | {market_data.get('price_change', 'N/A')} |\n")
                f.write(f"| 成交量变化 | {market_data.get('volume_change', 'N/A')} |\n")
                f.write(f"| 波动率 | {market_data.get('volatility', 'N/A')} |\n\n")
            
            # 技术指标
            if technical_indicators:
                f.write("## 📈 技术指标\n\n")
                f.write("| 指标 | 数值 | 状态 |\n")
                f.write("| --- | --- | --- |\n")
                f.write(f"| RSI(14) | {technical_indicators.get('rsi', 'N/A')} | {technical_indicators.get('rsi_status', '')} |\n")
                f.write(f"| MACD | {technical_indicators.get('macd', 'N/A')} | {technical_indicators.get('macd_trend', '')} |\n")
                f.write(f"| MACD信号 | {technical_indicators.get('signal', 'N/A')} | - |\n")
                f.write(f"| MACD柱状 | {technical_indicators.get('hist', 'N/A')} | - |\n")
                f.write(f"| 布林带位置 | {technical_indicators.get('bb_position', 'N/A')} | {technical_indicators.get('bb_status', '')} |\n\n")
                
                # 详细的布林带分析
                if technical_indicators.get('bb_upper') and technical_indicators.get('bb_lower'):
                    f.write("### 📏 布林带详细分析\n\n")
                    f.write(f"- **上轨**: {technical_indicators.get('bb_upper', 'N/A')}\n")
                    f.write(f"- **下轨**: {technical_indicators.get('bb_lower', 'N/A')}\n")
                    f.write(f"- **相对位置**: {technical_indicators.get('bb_position', 'N/A')}\n")
                    f.write(f"- **市场状态**: {technical_indicators.get('bb_status', 'N/A')}\n\n")
            
            # 特征重要性分析
            if model_analysis and model_analysis.get('feature_importance'):
                f.write("## 🔬 特征重要性分析\n\n")
                f.write("| 排名 | 特征名称 | 重要性 |\n")
                f.write("| --- | --- | --- |\n")
                feature_importance = model_analysis.get('feature_importance', {})
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                    f.write(f"| {i+1} | {feature} | {importance:.4f} |\n")
                f.write("\n")
            
            # 详细分析章节
            if detailed_analysis:
                # 均线分析
                if detailed_analysis.get('ma_analysis'):
                    f.write("## 📊 均线详细分析\n\n")
                    ma_analysis = detailed_analysis['ma_analysis']
                    f.write("| 均线 | 数值 | 距离 | 趋势 |\n")
                    f.write("| --- | --- | --- | --- |\n")
                    for ma_info in ma_analysis:
                        trend_icon = "📈" if ma_info['distance'] > 0 else "📉" if ma_info['distance'] < -1 else "➖"
                        f.write(f"| {ma_info['name']} | {ma_info['value']:.2f} | {ma_info['distance']:+.2f}% | {trend_icon} |\n")
                    f.write("\n")
                
                # AI模型决策分析
                if detailed_analysis.get('ai_decision'):
                    f.write("## 🤖 AI模型决策分析\n\n")
                    ai_decision = detailed_analysis['ai_decision']
                    f.write(f"### 预测概率分布\n")
                    f.write(f"- **非低点概率**: {ai_decision.get('non_low_prob', 0):.4f} ({ai_decision.get('non_low_prob', 0):.2%})\n")
                    f.write(f"- **低点概率**: {ai_decision.get('low_prob', 0):.4f} ({ai_decision.get('low_prob', 0):.2%})\n")
                    f.write(f"- **置信度评级**: {ai_decision.get('confidence_level', 'N/A')}\n\n")
                
                # 决策依据分析
                if detailed_analysis.get('decision_basis'):
                    f.write("## 🧠 决策依据分析\n\n")
                    decision_basis = detailed_analysis['decision_basis']
                    for basis in decision_basis:
                        icon = "✅" if basis['support'] == 'strong' else "⚡" if basis['support'] == 'partial' else "❌"
                        f.write(f"- {icon} **{basis['indicator']}**: {basis['description']}\n")
                    f.write("\n")
            
            # 预测结果
            f.write("## 🎯 预测结果\n\n")
            if prediction_result:
                result_text = "**是相对低点** ✅" if prediction_result.predicted_low_point else "**不是相对低点** ❌"
                f.write(f"### AI预测\n{result_text}\n\n")
                f.write(f"**置信度**: `{prediction_result.confidence:.4f}` ({prediction_result.confidence:.2%})\n\n")
                
                if prediction_result.actual_low_point is not None:
                    actual_text = "**是相对低点** ✅" if prediction_result.actual_low_point else "**不是相对低点** ❌"
                    f.write(f"### 验证结果\n")
                    f.write(f"- **实际结果**: {actual_text}\n")
                    
                    if prediction_result.prediction_correct is not None:
                        correct_text = "**正确** ✅" if prediction_result.prediction_correct else "**错误** ❌"
                        f.write(f"- **预测准确性**: {correct_text}\n")
                    
                    if prediction_result.future_max_rise is not None:
                        f.write(f"- **未来最大涨幅**: `{prediction_result.future_max_rise:.2%}`\n")
                    
                    if prediction_result.days_to_rise is not None and prediction_result.days_to_rise > 0:
                        rise_threshold = config.get('strategy', {}).get('rise_threshold', 0.04)
                        f.write(f"- **达到{rise_threshold:.1%}涨幅用时**: {prediction_result.days_to_rise}天\n")
                    
                    f.write("\n")
                else:
                    f.write("> ⚠️ 无法获取验证数据\n\n")
            else:
                f.write("> ❌ 预测失败\n\n")
            
            f.write("---\n")
            f.write("*📝 报告结束*\n")
        
        # 更新预测历史记录（保存在history子目录）
        update_prediction_history(predict_date_str, full_results, history_dir)
        
        logging.getLogger("SingleDayPredictor").info(f"💾 预测结果已保存:")
        logging.getLogger("SingleDayPredictor").info(f"   📄 JSON数据: {os.path.relpath(json_path)}")
        logging.getLogger("SingleDayPredictor").info(f"   📋 Markdown报告: {os.path.relpath(md_path)}")
        logging.getLogger("SingleDayPredictor").info(f"   📊 历史记录: {os.path.relpath(os.path.join(history_dir, 'prediction_history.json'))}")
        
        return json_path, md_path
        
    except Exception as e:
        logging.getLogger("SingleDayPredictor").error(f"❌ 保存预测结果失败: {e}")
        return None, None

def update_prediction_history(predict_date_str, results_data, history_dir):
    """
    更新预测历史记录
    
    Args:
        predict_date_str: 预测日期字符串
        results_data: 结果数据
        history_dir: 历史记录保存目录
    """
    try:
        history_file = os.path.join(history_dir, 'prediction_history.json')
        
        # 读取现有历史记录
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        # 添加新记录
        history_entry = {
            'date': predict_date_str,
            'timestamp': results_data['timestamp'],
            'prediction_time': results_data['prediction_time'],
            'predicted_low_point': results_data['prediction_results']['is_predicted_low_point'],
            'confidence': results_data['prediction_results']['confidence'],
            'actual_low_point': results_data['prediction_results']['actual_low_point'],
            'prediction_correct': results_data['prediction_results']['prediction_correct'],
            'model_file': results_data['model_info']['model_file']
        }
        
        # 检查是否已存在相同日期的记录，如果存在则替换
        existing_index = None
        for i, entry in enumerate(history):
            if entry.get('date') == predict_date_str:
                existing_index = i
                break
        
        if existing_index is not None:
            history[existing_index] = history_entry
        else:
            history.append(history_entry)
        
        # 按日期排序
        history.sort(key=lambda x: x['date'], reverse=True)
        
        # 保存历史记录
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
            
    except Exception as e:
        logging.getLogger("SingleDayPredictor").error(f"❌ 更新预测历史记录失败: {e}")

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
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system.yaml')
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
                logger.error("❌ 未找到已训练的模型！")
                logger.error("💡 请先运行以下命令训练模型：")
                logger.error("   python run.py ai -m optimize  # AI优化+训练")
                logger.error("   python run.py ai -m full      # 完整重训练")
                return False

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
    import os
    
    try:
        # 输出模型信息
        logger.info("="*80)
        logger.info("📋 模型信息与依据分析")
        logger.info("="*80)
        
        # 初始化数据收集字典
        market_data = {}
        technical_indicators = {}
        model_analysis = {}
        detailed_analysis = {}
        
        # 1. 检查并输出模型版本信息
        latest_model_path = os.path.join(ai_optimizer.models_dir, 'latest_improved_model.txt')
        if os.path.exists(latest_model_path):
            with open(latest_model_path, 'r') as f:
                model_path = f.read().strip()
                
                # 如果是相对路径，转换为绝对路径（相对于项目根目录）
                if not os.path.isabs(model_path):
                    # 获取项目根目录（从当前文件位置向上两级：examples/predict_single_day.py -> 项目根目录）
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    model_path = os.path.join(project_root, model_path)
                
                model_file = os.path.basename(model_path)
                model_analysis['model_file'] = model_file
                # 从文件名提取时间戳
                if 'model_' in model_file:
                        timestamp_str = model_file.replace('model_', '').replace('.pkl', '')
                        try:
                            from datetime import datetime
                            model_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                            logger.info(f"🤖 使用模型: {model_file}")
                            logger.info(f"🕐 训练时间: {model_time.strftime('%Y年%m月%d日 %H:%M:%S')}")
                            
                            # 计算模型年龄
                            model_age = datetime.now() - model_time
                            model_analysis['model_time'] = model_time.isoformat()
                            model_analysis['model_age_hours'] = (model_age.total_seconds() / 3600)
                            
                            if model_age.days == 0:
                                age_description = f"{model_age.seconds // 3600}小时{(model_age.seconds % 3600) // 60}分钟 (非常新鲜)"
                                logger.info(f"📅 模型年龄: {age_description}")
                            else:
                                age_description = f"{model_age.days}天 {'(较新)' if model_age.days < 7 else '(需考虑更新)' if model_age.days < 30 else '(建议重新训练)'}"
                                logger.info(f"📅 模型年龄: {age_description}")
                            
                            model_analysis['model_age_description'] = age_description
                        except:
                            logger.info(f"🤖 使用模型: {model_file}")
        
        # 2. 获取特征重要性
        feature_importance = ai_optimizer.get_feature_importance()
        if feature_importance:
            logger.info("\n📊 特征重要性排序 (Top 10):")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                logger.info(f"   {i+1:2d}. {feature:<20}: {importance:.4f}")
            model_analysis['feature_importance'] = feature_importance
        
        logger.info("\n" + "="*80)
        logger.info("🔍 预测过程详细分析")
        logger.info("="*80)
        
        # 1. 获取预测所需的历史数据（用于特征提取）
        history_days_needed = config["data"]["history_days"]
        start_date_for_prediction = predict_date - timedelta(days=history_days_needed)
        
        logger.info(f"📈 数据获取范围: {start_date_for_prediction.strftime('%Y-%m-%d')} 至 {predict_date.strftime('%Y-%m-%d')}")
        prediction_data = data_module.get_history_data(
            start_date=start_date_for_prediction.strftime('%Y-%m-%d'),
            end_date=predict_date.strftime('%Y-%m-%d')
        )
        
        if prediction_data.empty:
            logger.error("❌ 预测数据为空，无法进行预测。")
            return None

        # 预处理数据
        prediction_data = data_module.preprocess_data(prediction_data)
        logger.info(f"✅ 成功获取并预处理 {len(prediction_data)} 条历史数据")
        
        # 获取当天的具体数据
        predict_day_data = prediction_data.iloc[-1:].copy()
        current_row = predict_day_data.iloc[0]
        
        # 收集市场数据
        market_data.update({
            'close_price': f"{current_row.get('close', 'N/A'):.2f}",
            'price_change': f"{current_row.get('price_change', 0):.2%}",
            'volume_change': f"{current_row.get('volume_change', 0):.2%}",
            'volatility': f"{current_row.get('volatility', 0):.4f}"
        })
        
        # 输出当日关键市场数据
        logger.info(f"\n📊 {predict_date.strftime('%Y年%m月%d日')} 关键市场数据:")
        logger.info(f"   收盘价: {market_data['close_price']}")
        logger.info(f"   涨跌幅: {market_data['price_change']}")
        logger.info(f"   成交量变化: {market_data['volume_change']}")
        logger.info(f"   波动率: {market_data['volatility']}")
        
        # 收集技术指标数据
        rsi_val = current_row.get('rsi', 50)
        rsi_status = '[超卖]' if rsi_val < 30 else '[偏弱]' if rsi_val < 40 else '[中性]' if rsi_val < 60 else '[偏强]'
        macd_trend = '[金叉趋势]' if current_row.get('hist', 0) > 0 else '[死叉趋势]'
        
        technical_indicators.update({
            'rsi': f"{rsi_val:.2f}",
            'rsi_status': rsi_status,
            'macd': f"{current_row.get('macd', 'N/A'):.4f}",
            'signal': f"{current_row.get('signal', 'N/A'):.4f}",
            'hist': f"{current_row.get('hist', 'N/A'):.4f}",
            'macd_trend': macd_trend
        })
        
        # 输出技术指标
        logger.info(f"\n📈 技术指标分析:")
        logger.info(f"   RSI(14): {technical_indicators['rsi']} {rsi_status}")
        logger.info(f"   MACD: {technical_indicators['macd']}")
        logger.info(f"   MACD信号: {technical_indicators['signal']}")
        logger.info(f"   MACD柱状: {technical_indicators['hist']} {macd_trend}")
        
        # 输出均线情况
        ma5 = current_row.get('ma5', 0)
        ma10 = current_row.get('ma10', 0)
        ma20 = current_row.get('ma20', 0)
        ma60 = current_row.get('ma60', 0)
        close_price = current_row.get('close', 0)
        
        # 收集均线分析数据
        ma_analysis = [
            {'name': 'MA5', 'value': ma5, 'distance': ((close_price - ma5) / ma5 * 100) if ma5 > 0 else 0},
            {'name': 'MA10', 'value': ma10, 'distance': ((close_price - ma10) / ma10 * 100) if ma10 > 0 else 0},
            {'name': 'MA20', 'value': ma20, 'distance': ((close_price - ma20) / ma20 * 100) if ma20 > 0 else 0},
            {'name': 'MA60', 'value': ma60, 'distance': ((close_price - ma60) / ma60 * 100) if ma60 > 0 else 0}
        ]
        detailed_analysis['ma_analysis'] = ma_analysis
        
        logger.info(f"\n📊 均线分析:")
        logger.info(f"   MA5:  {ma5:.2f} (距离: {((close_price - ma5) / ma5 * 100):+.2f}%)")
        logger.info(f"   MA10: {ma10:.2f} (距离: {((close_price - ma10) / ma10 * 100):+.2f}%)")
        logger.info(f"   MA20: {ma20:.2f} (距离: {((close_price - ma20) / ma20 * 100):+.2f}%)")
        logger.info(f"   MA60: {ma60:.2f} (距离: {((close_price - ma60) / ma60 * 100):+.2f}%)")
        
        # 布林带分析
        bb_upper = current_row.get('bb_upper', 0)
        bb_lower = current_row.get('bb_lower', 0)
        if bb_upper > 0 and bb_lower > 0:
            bb_position = (close_price - bb_lower) / (bb_upper - bb_lower)
            bb_status = '[超卖区域]' if bb_position < 0.2 else '[偏弱区域]' if bb_position < 0.4 else '[中性区域]' if bb_position < 0.6 else '[偏强区域]' if bb_position < 0.8 else '[超买区域]'
            
            technical_indicators.update({
                'bb_upper': f"{bb_upper:.2f}",
                'bb_lower': f"{bb_lower:.2f}",
                'bb_position': f"{bb_position:.2%}",
                'bb_status': bb_status
            })
            
            logger.info(f"\n📏 布林带分析:")
            logger.info(f"   上轨: {bb_upper:.2f}")
            logger.info(f"   下轨: {bb_lower:.2f}")
            logger.info(f"   位置: {bb_position:.2%} {bb_status}")
        
        # 2. 使用已训练模型进行预测
        logger.info(f"\n🔮 AI模型预测分析:")
        prediction_result = ai_optimizer.predict_low_point(predict_day_data)
        
        # 检查预测结果是否包含错误
        if prediction_result is None or prediction_result.get("error"):
            error_msg = prediction_result.get("error", "预测失败") if prediction_result else "预测返回None"
            logger.error(f"❌ AI模型预测失败: {error_msg}")
            return None
        
        is_predicted_low_point = prediction_result.get("is_low_point")
        confidence = prediction_result.get("confidence")
        # final_confidence 已废弃，统一使用 confidence
        prediction_proba = prediction_result.get("prediction_proba", [])
        
        # 收集模型分析数据
        model_analysis.update({
            'model_type': prediction_result.get('model_type', ''),
            'feature_count': prediction_result.get('feature_count', 0),
            'prediction_proba': prediction_proba
        })
        
        # 输出预测概率分布
        if len(prediction_proba) >= 2:
            logger.info(f"   非低点概率: {prediction_proba[0]:.4f} ({prediction_proba[0]:.2%})")
            logger.info(f"   低点概率:   {prediction_proba[1]:.4f} ({prediction_proba[1]:.2%})")
        
        # 置信度评级
        confidence_level = "极低" if confidence < 0.3 else "较低" if confidence < 0.5 else "中等" if confidence < 0.7 else "较高" if confidence < 0.85 else "很高"
        logger.info(f"   置信度评级: {confidence_level} ({confidence:.2%})")
        
        # 收集AI决策分析数据
        ai_decision = {
            'non_low_prob': prediction_proba[0] if len(prediction_proba) >= 2 else 0,
            'low_prob': prediction_proba[1] if len(prediction_proba) >= 2 else 0,
            'confidence_level': confidence_level,
            'confidence_value': confidence
        }
        detailed_analysis['ai_decision'] = ai_decision
        
        # 预测结果总结
        logger.info(f"\n🎯 AI预测结果: \033[1;{'32' if is_predicted_low_point else '31'}m{predict_date.strftime('%Y-%m-%d')} {'是' if is_predicted_low_point else '不是'} 相对低点\033[0m")
        logger.info(f"   整体置信度: \033[1m{confidence:.4f}\033[0m ({confidence:.2%})")
        
        # 决策依据分析
        logger.info(f"\n🧠 决策依据分析:")
        
        # 收集决策依据数据
        decision_basis = []
        
        # RSI依据
        rsi_val = current_row.get('rsi', 50)
        if rsi_val < 30:
            logger.info(f"   ✅ RSI超卖 ({rsi_val:.1f} < 30) - 支持低点判断")
            decision_basis.append({
                'indicator': 'RSI',
                'description': f'RSI超卖 ({rsi_val:.1f} < 30) - 支持低点判断',
                'support': 'strong'
            })
        elif rsi_val < 40:
            logger.info(f"   ⚡ RSI偏弱 ({rsi_val:.1f} < 40) - 部分支持低点判断")
            decision_basis.append({
                'indicator': 'RSI',
                'description': f'RSI偏弱 ({rsi_val:.1f} < 40) - 部分支持低点判断',
                'support': 'partial'
            })
        else:
            logger.info(f"   ❌ RSI非超卖 ({rsi_val:.1f} ≥ 40) - 不支持低点判断")
            decision_basis.append({
                'indicator': 'RSI',
                'description': f'RSI非超卖 ({rsi_val:.1f} ≥ 40) - 不支持低点判断',
                'support': 'none'
            })
        
        # MACD依据
        macd_val = current_row.get('macd', 0)
        hist_val = current_row.get('hist', 0)
        if macd_val < 0 and hist_val > current_row.get('hist_prev', hist_val - 0.001):
            logger.info(f"   ✅ MACD负值但柱状线改善 - 支持反转信号")
            decision_basis.append({
                'indicator': 'MACD',
                'description': 'MACD负值但柱状线改善 - 支持反转信号',
                'support': 'strong'
            })
        elif macd_val < 0:
            logger.info(f"   ⚡ MACD为负 ({macd_val:.4f}) - 趋势偏弱")
            decision_basis.append({
                'indicator': 'MACD',
                'description': f'MACD为负 ({macd_val:.4f}) - 趋势偏弱',
                'support': 'partial'
            })
        else:
            logger.info(f"   ❌ MACD为正 ({macd_val:.4f}) - 趋势向上")
            decision_basis.append({
                'indicator': 'MACD',
                'description': f'MACD为正 ({macd_val:.4f}) - 趋势向上',
                'support': 'none'
            })
        
        # 均线依据  
        ma_below_count = sum([close_price < ma5, close_price < ma10, close_price < ma20, close_price < ma60])
        if ma_below_count >= 3:
            logger.info(f"   ✅ 价格低于{ma_below_count}/4条均线 - 强烈支持低点")
            decision_basis.append({
                'indicator': '均线',
                'description': f'价格低于{ma_below_count}/4条均线 - 强烈支持低点',
                'support': 'strong'
            })
        elif ma_below_count >= 2:
            logger.info(f"   ⚡ 价格低于{ma_below_count}/4条均线 - 部分支持低点")
            decision_basis.append({
                'indicator': '均线',
                'description': f'价格低于{ma_below_count}/4条均线 - 部分支持低点',
                'support': 'partial'
            })
        else:
            logger.info(f"   ❌ 价格高于多数均线 ({4-ma_below_count}/4条) - 不支持低点")
            decision_basis.append({
                'indicator': '均线',
                'description': f'价格高于多数均线 ({4-ma_below_count}/4条) - 不支持低点',
                'support': 'none'
            })
        
        # 布林带依据
        if bb_upper > 0 and bb_lower > 0:
            if close_price <= bb_lower * 1.02:  # 接近下轨
                logger.info(f"   ✅ 价格接近布林带下轨 - 支持低点判断")
                decision_basis.append({
                    'indicator': '布林带',
                    'description': '价格接近布林带下轨 - 支持低点判断',
                    'support': 'strong'
                })
            elif bb_position < 0.3:
                logger.info(f"   ⚡ 价格在布林带下方区域 - 部分支持低点")
                decision_basis.append({
                    'indicator': '布林带',
                    'description': '价格在布林带下方区域 - 部分支持低点',
                    'support': 'partial'
                })
            else:
                logger.info(f"   ❌ 价格远离布林带下轨 - 不支持低点判断")
                decision_basis.append({
                    'indicator': '布林带',
                    'description': '价格远离布林带下轨 - 不支持低点判断',
                    'support': 'none'
                })
        
        # 成交量依据
        volume_change = current_row.get('volume_change', 0)
        if volume_change > 0.4:  # 放量下跌
            logger.info(f"   ✅ 成交量放大 ({volume_change:+.1%}) - 可能恐慌性抛售")
            decision_basis.append({
                'indicator': '成交量',
                'description': f'成交量放大 ({volume_change:+.1%}) - 可能恐慌性抛售',
                'support': 'strong'
            })
        elif volume_change < -0.2:  # 缩量下跌
            logger.info(f"   ⚡ 成交量萎缩 ({volume_change:+.1%}) - 抛压减轻")
            decision_basis.append({
                'indicator': '成交量',
                'description': f'成交量萎缩 ({volume_change:+.1%}) - 抛压减轻',
                'support': 'partial'
            })
        else:
            logger.info(f"   ➖ 成交量变化适中 ({volume_change:+.1%}) - 中性信号")
            decision_basis.append({
                'indicator': '成交量',
                'description': f'成交量变化适中 ({volume_change:+.1%}) - 中性信号',
                'support': 'none'
            })
        
        # 保存决策依据数据
        detailed_analysis['decision_basis'] = decision_basis

        # 3. 验证预测结果（如果需要）
        logger.info(f"\n" + "="*80)
        logger.info("📊 历史验证分析")
        logger.info("="*80)
        
        end_date_for_validation = predict_date + timedelta(days=config["default_strategy"]["max_days"] + 10)
        start_date_for_validation = predict_date - timedelta(days=config["default_strategy"]["max_days"] + 10)
        
        validation_data = data_module.get_history_data(
            start_date=start_date_for_validation.strftime('%Y-%m-%d'),
            end_date=end_date_for_validation.strftime('%Y-%m-%d')
        )

        if validation_data.empty:
            logger.warning("⚠️  验证数据为空，无法验证预测结果。")
            prediction_result_obj = PredictionResult(
                date=predict_date,
                predicted_low_point=is_predicted_low_point,
                actual_low_point=None,
                confidence=confidence,
                future_max_rise=None,
                days_to_rise=None,
                prediction_correct=None,
                predict_price=None
            )
            
            # 保存结果
            save_prediction_results(
                prediction_result_obj,
                predict_date.strftime('%Y-%m-%d'),
                config,
                market_data,
                technical_indicators,
                model_analysis,
                detailed_analysis
            )
            
            return prediction_result_obj

        # 预处理验证数据
        full_validation_set = data_module.preprocess_data(validation_data)
        predict_date_data = full_validation_set[full_validation_set['date'] == predict_date]
        
        if predict_date_data.empty:
            logger.warning(f"⚠️  无法在验证数据中找到 {predict_date.strftime('%Y-%m-%d')} 的记录，无法验证预测结果。")
            prediction_result_obj = PredictionResult(
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
            
            # 保存结果
            save_prediction_results(
                prediction_result_obj,
                predict_date.strftime('%Y-%m-%d'),
                config,
                market_data,
                technical_indicators,
                model_analysis,
                detailed_analysis
            )
            
            return prediction_result_obj

        predict_price = predict_date_data.iloc[0]['close']
        future_data = full_validation_set[full_validation_set['date'] > predict_date]
        
        if future_data.empty:
            logger.warning(f"⚠️  无法获取 {predict_date.strftime('%Y-%m-%d')} 之后的数据，无法验证预测结果。")
            prediction_result_obj = PredictionResult(
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
            
            # 保存结果
            save_prediction_results(
                prediction_result_obj,
                predict_date.strftime('%Y-%m-%d'),
                config,
                market_data,
                technical_indicators,
                model_analysis,
                detailed_analysis
            )
            
            return prediction_result_obj

        # 获取预测日的index
        predict_index = predict_date_data.iloc[0]['index']
        max_rise = 0.0
        days_to_rise = 0
        rise_threshold = config["default_strategy"]["rise_threshold"]
        max_days = config["default_strategy"]["max_days"]
        
        logger.info(f"📈 未来{max_days}天表现追踪:")
        logger.info(f"   预测日价格: {predict_price:.2f}")
        logger.info(f"   目标涨幅: {rise_threshold:.1%}")
        
        # 计算未来最大涨幅和达到目标涨幅所需天数
        daily_performance = []
        for i, row in future_data.iterrows():
            rise_rate = (row['close'] - predict_price) / predict_price
            days_elapsed = row['index'] - predict_index
            
            if days_elapsed <= max_days:
                daily_performance.append({
                    'day': days_elapsed,
                    'date': row['date'],
                    'price': row['close'],
                    'rise': rise_rate
                })
            
            if rise_rate > max_rise:
                max_rise = rise_rate
                
            if rise_rate >= rise_threshold and days_to_rise == 0:
                days_to_rise = days_elapsed

        # 显示前几天的详细表现
        for i, perf in enumerate(daily_performance[:min(10, len(daily_performance))]):
            status = "✅达标" if perf['rise'] >= rise_threshold else "📈上涨" if perf['rise'] > 0 else "📉下跌"
            logger.info(f"   第{perf['day']}天 ({perf['date']}): {perf['price']:.2f} ({perf['rise']:+.2%}) {status}")

        actual_is_low_point = max_rise >= rise_threshold

        logger.info(f"\n📋 验证结果总结:")
        logger.info(f"   实际是否为相对低点: {'✅ 是' if actual_is_low_point else '❌ 否'}")
        logger.info(f"   未来{max_days}天最大涨幅: {max_rise:.2%}")
        if days_to_rise > 0:
            logger.info(f"   达到{rise_threshold:.1%}涨幅用时: {days_to_rise}天")
        else:
            logger.info(f"   未在{max_days}天内达到{rise_threshold:.1%}涨幅")
        
        # 预测准确性分析
        prediction_correct = is_predicted_low_point == actual_is_low_point
        logger.info(f"\n🎯 预测准确性: {'✅ 正确' if prediction_correct else '❌ 错误'}")
        
        if prediction_correct:
            if is_predicted_low_point:
                logger.info(f"   🎉 成功识别出相对低点！未来最大收益{max_rise:.2%}")
            else:
                logger.info(f"   ✅ 正确避开非低点，避免了可能的损失")
        else:
            if is_predicted_low_point and not actual_is_low_point:
                logger.info(f"   ⚠️  误判为低点（假阳性），可能导致过早入场")
            else:
                logger.info(f"   😔 错过真正的低点（假阴性），错失了{max_rise:.2%}的收益机会")

        logger.info("="*80)

        prediction_result_obj = PredictionResult(
            date=predict_date,
            predicted_low_point=is_predicted_low_point,
            actual_low_point=actual_is_low_point,
            confidence=confidence,
            future_max_rise=max_rise,
            days_to_rise=days_to_rise,
            prediction_correct=prediction_correct,
            predict_price=predict_price
        )
        
        # 保存结果
        save_prediction_results(
            prediction_result_obj,
            predict_date.strftime('%Y-%m-%d'),
            config,
            market_data,
            technical_indicators,
            model_analysis,
            detailed_analysis
        )

        return prediction_result_obj

    except Exception as e:
        logger.error(f"❌ 使用已训练模型预测失败: {e}")
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


