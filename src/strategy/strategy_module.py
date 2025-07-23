#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略执行模块
实现相对低点识别算法和回测功能
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

class StrategyModule:
    """策略执行模块类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略模块
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger('StrategyModule')
        self.config = config
        
        # 策略参数
        strategy_config = config.get('strategy', {})
        self.rise_threshold = strategy_config.get('rise_threshold', 0.04)
        self.max_days = strategy_config.get('max_days', 20)
        
        # 创建结果目录
        self.results_dir = strategy_config.get('results_dir', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.logger.info("策略模块初始化完成，参数: rise_threshold=%.4f, max_days=%d", 
                        self.rise_threshold, self.max_days)
        
    def identify_relative_low(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        识别相对低点
        
        参数:
        data: 市场数据
        
        返回:
        dict: 识别结果
        """
        # self.logger.info("识别相对低点")
        
        try:
            if len(data) == 0:
                return {
                    'date': None,
                    'price': None,
                    'is_low_point': False,
                    'confidence': 0.0,
                    'reason': '数据为空'
                }
            
            # 获取最新日期的数据
            latest_data = data.iloc[-1]
            latest_date = latest_data['date']
            latest_price = latest_data['close']
            
            # 获取技术指标
            ma5 = latest_data.get('ma5', None)
            ma10 = latest_data.get('ma10', None)
            ma20 = latest_data.get('ma20', None)
            rsi = latest_data.get('rsi', None)
            macd = latest_data.get('macd', None)
            bb_lower = latest_data.get('bb_lower', None)
            
            # 判断是否可能是相对低点
            is_low_point = False
            confidence = 0.0
            reasons = []
            
            # 从配置文件获取置信度权重
            strategy_config = self.config.get('strategy', {})
            confidence_config = strategy_config.get('confidence_weights', {})
            
            # 条件1: 价格低于多条移动平均线
            if ma5 is not None and ma10 is not None and ma20 is not None:
                if latest_price < ma5 and latest_price < ma10 and latest_price < ma20:
                    # 价格跌破所有均线 - 基础条件满足
                    base_confidence = confidence_config.get('ma_all_below', 0.3)
                    
                    # 成交量分析 - 区分是下跌通道还是见底信号
                    volume_ratio = data.iloc[-1]['volume_ratio'] if 'volume_ratio' in data.columns else 1.0
                    price_decline = data.iloc[-1]['price_change'] if 'price_change' in data.columns else 0.0
                    
                    # 获取成交量相关阈值
                    volume_panic_threshold = confidence_config.get('volume_panic_threshold', 1.4)
                    volume_surge_threshold = confidence_config.get('volume_surge_threshold', 1.2)
                    volume_shrink_threshold = confidence_config.get('volume_shrink_threshold', 0.8)
                    price_decline_threshold = confidence_config.get('price_decline_threshold', -0.02)
                    
                    # 判断成交量状态
                    if volume_ratio > volume_panic_threshold and price_decline < price_decline_threshold:
                        # 恐慌性抛售 - 可能是见底信号
                        panic_bonus = confidence_config.get('volume_panic_bonus', 0.1)
                        confidence += base_confidence + panic_bonus
                        is_low_point = True
                        reasons.append(f"价格跌破所有均线+恐慌性抛售(成交量放大{volume_ratio:.1f}倍)")
                    elif volume_ratio > volume_surge_threshold:
                        # 温和放量 - 可能是见底信号
                        surge_bonus = confidence_config.get('volume_surge_bonus', 0.05)
                        confidence += base_confidence + surge_bonus
                        is_low_point = True
                        reasons.append(f"价格跌破所有均线+温和放量(成交量放大{volume_ratio:.1f}倍)")
                    elif volume_ratio < volume_shrink_threshold:
                        # 成交量萎缩 - 可能是下跌通道中
                        shrink_penalty = confidence_config.get('volume_shrink_penalty', 0.7)
                        confidence += base_confidence * shrink_penalty
                        reasons.append(f"价格跌破所有均线+成交量萎缩(可能是下跌通道)")
                    else:
                        # 正常成交量 - 保持原有逻辑
                        confidence += base_confidence
                        is_low_point = True
                        reasons.append("价格低于MA5/MA10/MA20")
                elif latest_price < ma10 and latest_price < ma20:
                    confidence += confidence_config.get('ma_partial_below', 0.2)
                    reasons.append("价格低于MA10/MA20")
                    
            # 条件2: RSI超卖
            if rsi is not None:
                rsi_oversold_threshold = confidence_config.get('rsi_oversold_threshold', 30)
                rsi_low_threshold = confidence_config.get('rsi_low_threshold', 40)
                if rsi < rsi_oversold_threshold:
                    is_low_point = True
                    confidence += confidence_config.get('rsi_oversold', 0.3)
                    reasons.append(f"RSI超卖({rsi:.2f})")
                elif rsi < rsi_low_threshold:
                    confidence += confidence_config.get('rsi_low', 0.2)
                    reasons.append(f"RSI偏低({rsi:.2f})")
                    
            # 🆕 条件2B: RSI上升阶段的回调识别（新增逻辑）
            if rsi is not None and len(data) >= 10:
                # 获取RSI历史数据用于回调分析
                rsi_series = data['rsi'].tail(10) if 'rsi' in data.columns else None
                if rsi_series is not None and not rsi_series.isna().all():
                    # 🔥 RSI上升阶段参数（大幅放宽条件）
                    rsi_uptrend_min = confidence_config.get('rsi_uptrend_min', 35)  # 大幅降低门槛
                    rsi_uptrend_max = confidence_config.get('rsi_uptrend_max', 85)  # 扩大范围
                    rsi_pullback_threshold = confidence_config.get('rsi_pullback_threshold', 3)  # 降低回调要求
                    
                    # 🎯 更宽松的RSI阶段识别（适应更多上升阶段情况）
                    if rsi_uptrend_min <= rsi <= rsi_uptrend_max:
                        # 计算RSI短期变化
                        rsi_recent_high = rsi_series.tail(5).max()  # 近5日RSI最高值
                        rsi_recent_low = rsi_series.tail(5).min()   # 近5日RSI最低值
                        rsi_pullback = rsi_recent_high - rsi  # RSI回调幅度
                        
                        # 条件1: 任何程度的健康回调都给予奖励
                        if rsi_pullback >= rsi_pullback_threshold:
                            # 🚀 不再要求严格的价格回调条件
                            uptrend_pullback_weight = confidence_config.get('rsi_uptrend_pullback', 0.35)
                            confidence += uptrend_pullback_weight
                            is_low_point = True
                            reasons.append(f"上升趋势中健康回调(RSI:{rsi:.1f}, 回调{rsi_pullback:.1f}点)")
                        
                        # 条件2: RSI在中高位（40-70）也给予支持
                        elif 40 <= rsi <= 70:
                            # 任何在中高位的RSI都可能是相对低点
                            moderate_rsi_weight = confidence_config.get('moderate_rsi_bonus', 0.20)
                            confidence += moderate_rsi_weight
                            reasons.append(f"RSI中高位支撑({rsi:.1f})")
                        
                        # 条件3: RSI从任何高位回落（更宽松）
                        elif rsi_recent_high >= 60 and rsi >= 45:
                            # 从中高位回落也算修正机会
                            overbought_correction_weight = confidence_config.get('rsi_overbought_correction', 0.25)
                            confidence += overbought_correction_weight
                            reasons.append(f"RSI超买修正({rsi:.1f}, 从{rsi_recent_high:.1f}回落)")
                    
            # 条件3: MACD负值
            if macd is not None and macd < 0:
                confidence += confidence_config.get('macd_negative', 0.1)
                reasons.append("MACD负值")
                
            # 条件4: 价格接近布林带下轨
            if bb_lower is not None:
                bb_near_threshold = confidence_config.get('bb_near_threshold', 1.02)
                if latest_price <= bb_lower * bb_near_threshold:
                    is_low_point = True
                    confidence += confidence_config.get('bb_lower_near', 0.2)
                    reasons.append("价格接近布林带下轨")
                
            # 条件5: 近期大幅下跌
            if len(data) >= 5:
                price_5d_ago = data.iloc[-6]['close'] if len(data) >= 6 else data.iloc[0]['close']
                decline_5d = (latest_price - price_5d_ago) / price_5d_ago
                decline_threshold = confidence_config.get('decline_threshold', -0.05)  # 5%下跌阈值
                if decline_5d < decline_threshold:
                    confidence += confidence_config.get('recent_decline', 0.2)
                    reasons.append(f"近5日大幅下跌({decline_5d:.2%})")
            
            # 条件6: AI优化参数调整
            # 动态置信度调整 - 根据市场波动性调整
            dynamic_confidence_adjustment = confidence_config.get('dynamic_confidence_adjustment', 0.1)
            if len(data) >= 20:
                # 计算20日波动率
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()
                # 高波动率时降低置信度要求，低波动率时提高要求
                if volatility > 0.03:  # 高波动率
                    confidence += dynamic_confidence_adjustment * 0.5
                    reasons.append(f"高波动率调整(+{dynamic_confidence_adjustment * 0.5:.3f})")
                elif volatility < 0.015:  # 低波动率
                    confidence -= dynamic_confidence_adjustment * 0.3
                    reasons.append(f"低波动率调整(-{dynamic_confidence_adjustment * 0.3:.3f})")
            
            # 市场情绪权重 - 基于成交量变化判断市场情绪
            market_sentiment_weight = confidence_config.get('market_sentiment_weight', 0.15)
            if len(data) >= 10:
                # 计算近期成交量变化
                recent_volume_avg = data['volume'].tail(5).mean()
                historical_volume_avg = data['volume'].tail(20).mean()
                volume_ratio = recent_volume_avg / historical_volume_avg
                
                if volume_ratio > 1.5:  # 放量 - 可能是恐慌性抛售或抄底
                    if latest_price < data['close'].tail(10).mean():  # 价格下跌时放量
                        confidence += market_sentiment_weight
                        reasons.append(f"恐慌性抛售情绪(+{market_sentiment_weight:.3f})")
                elif volume_ratio < 0.7:  # 缩量 - 可能是观望情绪
                    confidence += market_sentiment_weight * 0.3
                    reasons.append(f"观望情绪(+{market_sentiment_weight * 0.3:.3f})")
            
            # 🆕 上升趋势中的成交量配合分析（新增逻辑）
            if len(data) >= 20:
                # 判断是否处于上升趋势
                ma20_current = latest_data.get('ma20', None)
                ma20_prev = data.iloc[-5]['ma20'] if len(data) >= 5 and 'ma20' in data.columns else None
                
                if ma20_current and ma20_prev and ma20_current > ma20_prev:
                    # 确认在上升趋势中
                    price_vs_ma20 = (latest_price - ma20_current) / ma20_current
                    
                    # 价格回调但仍在均线附近（健康调整）
                    if -0.02 <= price_vs_ma20 <= 0.03:  # 价格在MA20的-2%到+3%范围内
                        volume_ratio = data.iloc[-1]['volume_ratio'] if 'volume_ratio' in data.columns else 1.0
                        
                        # 缩量回调（健康的洗盘）
                        if volume_ratio < 0.8:
                            uptrend_volume_pullback = confidence_config.get('uptrend_volume_pullback', 0.15)
                            confidence += uptrend_volume_pullback
                            is_low_point = True
                            reasons.append(f"上升趋势中缩量回调(+{uptrend_volume_pullback:.3f})")
                        
                        # 温和放量（可能是支撑位抄底）
                        elif 1.0 <= volume_ratio <= 1.3:
                            uptrend_support_volume = confidence_config.get('uptrend_support_volume', 0.12)
                            confidence += uptrend_support_volume
                            reasons.append(f"上升趋势中支撑位放量(+{uptrend_support_volume:.3f})")
                    
                    # 价格接近或略低于重要均线（强支撑位）
                    elif -0.05 <= price_vs_ma20 < -0.02:  # 价格在MA20下方2-5%
                        ma_support_weight = confidence_config.get('uptrend_ma_support', 0.18)
                        confidence += ma_support_weight
                        is_low_point = True
                        reasons.append(f"上升趋势中均线支撑(+{ma_support_weight:.3f})")
            
            # 🔄 趋势强度权重 - 智能趋势内回调识别（修改后的逻辑）
            trend_strength_weight = confidence_config.get('trend_strength_weight', 0.12)
            if len(data) >= 20:
                # 计算多时间框架趋势
                # 长期趋势（20日）
                x_long = np.arange(20)
                y_long = data['close'].tail(20).values
                slope_long = np.polyfit(x_long, y_long, 1)[0]
                trend_strength_long = abs(slope_long) / y_long.mean()
                
                # 短期趋势（5日）
                if len(data) >= 5:
                    x_short = np.arange(5)
                    y_short = data['close'].tail(5).values
                    slope_short = np.polyfit(x_short, y_short, 1)[0]
                    trend_strength_short = abs(slope_short) / y_short.mean()
                else:
                    slope_short = slope_long
                    trend_strength_short = trend_strength_long
                
                # 智能趋势分析
                if trend_strength_long > 0.01:  # 长期强趋势
                    if slope_long > 0:  # 长期上涨趋势
                        # 🆕 上升趋势中的智能回调识别
                        if slope_short < 0 and trend_strength_short > 0.005:
                            # 短期回调但长期向上 - 这是好的买入机会！
                            uptrend_pullback_bonus = confidence_config.get('uptrend_pullback_bonus', 0.18)
                            confidence += uptrend_pullback_bonus
                            is_low_point = True
                            reasons.append(f"上升趋势中回调机会(+{uptrend_pullback_bonus:.3f})")
                        elif abs(slope_short) < 0.002:
                            # 上升趋势中的横盘整理
                            uptrend_consolidation_bonus = confidence_config.get('uptrend_consolidation_bonus', 0.12)
                            confidence += uptrend_consolidation_bonus
                            reasons.append(f"上升趋势中横盘整理(+{uptrend_consolidation_bonus:.3f})")
                        else:
                            # 继续上涨，适度降低权重但不大幅减分
                            confidence -= trend_strength_weight * 0.1
                            reasons.append(f"强上涨趋势延续(-{trend_strength_weight * 0.1:.3f})")
                    else:  # 长期下跌趋势
                        confidence += trend_strength_weight
                        reasons.append(f"强下跌趋势(+{trend_strength_weight:.3f})")
                elif trend_strength_long < 0.002:  # 弱趋势
                    confidence += trend_strength_weight * 0.2
                    reasons.append(f"弱趋势调整(+{trend_strength_weight * 0.2:.3f})")
                    
            # 最终判断 - 从 system.yaml 读取 final_threshold
            confidence_threshold = self.config.get('final_threshold', 0.5)
            if confidence >= confidence_threshold:
                is_low_point = True
                
            # 限制置信度在0-1之间
            confidence = min(confidence, 1.0)
            
            # 构建结果
            result = {
                'date': latest_date,
                'price': latest_price,
                'is_low_point': is_low_point,
                'confidence': confidence,
                'reasons': reasons,
                'technical_indicators': {
                    'ma5': ma5,
                    'ma10': ma10,
                    'ma20': ma20,
                    'rsi': rsi,
                    'macd': macd,
                    'bb_lower': bb_lower
                }
            }
            
            # self.logger.info("相对低点识别结果: %s, 置信度: %.2f", 
            #                    "是" if is_low_point else "否", confidence)
            return result
            
        except Exception as e:
            self.logger.error("识别相对低点失败: %s", str(e))
            return {
                'date': None,
                'price': None,
                'is_low_point': False,
                'confidence': 0.0,
                'error': str(e)
            }
            
    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        回测策略
        
        参数:
        data: 历史数据
        
        返回:
        pandas.DataFrame: 回测结果
        """
        self.logger.info("开始回测，数据长度: %d", len(data))
        
        try:
            # 复制数据避免修改原数据
            backtest_data = data.copy()
            
            # 添加回测结果列
            backtest_data['is_low_point'] = False
            backtest_data['future_max_rise'] = 0.0
            backtest_data['days_to_rise'] = 0
            backtest_data['max_rise_date'] = None

            # index为交易日序号，date为实际交易日，未出现的日期视为非交易日
            # 只遍历到倒数max_days个交易日，保证未来最多只看max_days个交易日
            for i in range(len(backtest_data) - self.max_days):
                current_price = backtest_data.iloc[i]['close']
                current_date = backtest_data.iloc[i]['date']
                # 当前index可用：backtest_data.iloc[i]['index']

                max_rise = 0.0
                days_to_rise = 0
                max_rise_date = None

                # 只统计未来max_days个交易日（严格以index为步进，date为实际交易日）
                for j in range(1, self.max_days + 1):
                    if i + j >= len(backtest_data):
                        break  # 超出数据范围
                    future_price = backtest_data.iloc[i + j]['close']
                    future_date = backtest_data.iloc[i + j]['date']
                    rise_rate = (future_price - current_price) / current_price

                    if rise_rate > max_rise:
                        max_rise = rise_rate
                        max_rise_date = future_date

                    if rise_rate >= self.rise_threshold and days_to_rise == 0:
                        days_to_rise = j  # j即为x个交易日后

                # 更新数据
                backtest_data.loc[i, 'future_max_rise'] = max_rise
                backtest_data.loc[i, 'days_to_rise'] = days_to_rise
                backtest_data.loc[i, 'max_rise_date'] = max_rise_date

                # 使用策略识别相对低点（基于技术指标，而不是未来结果）
                # 传递从开始到当前位置的所有历史数据，让算法基于历史数据判断当前时点
                historical_data = backtest_data.iloc[:i+1].copy()
                identification_result = self.identify_relative_low(historical_data)
                backtest_data.loc[i, 'is_low_point'] = identification_result['is_low_point']
                
            self.logger.info("回测完成")
            return backtest_data
            
        except Exception as e:
            self.logger.error("回测失败: %s", str(e))
            raise
            
    def evaluate_strategy(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        评估策略
        
        参数:
        backtest_results: 回测结果
        
        返回:
        dict: 评估结果
        """
        self.logger.info("评估策略")
        
        try:
            # 获取相对低点
            low_points = backtest_results[backtest_results['is_low_point']]
            total_points = len(low_points)
            
            if total_points == 0:
                return {
                    'total_points': 0,
                    'success_rate': 0.0,
                    'avg_rise': 0.0,
                    'avg_days': 0.0,
                    'max_rise': 0.0,
                    'min_rise': 0.0,
                    'score': 0.0
                }
            
            # 计算统计数据
            avg_rise = low_points['future_max_rise'].mean()
            avg_days = low_points['days_to_rise'].mean()
            max_rise = low_points['future_max_rise'].max()
            min_rise = low_points['future_max_rise'].min()
            
            # 计算成功率（未来实际涨幅超过阈值的比例）
            successful_points = low_points[low_points['future_max_rise'] >= self.rise_threshold]
            success_rate = len(successful_points) / total_points
            
            # 计算综合得分
            score = self._calculate_score(success_rate, avg_rise, avg_days)
            
            # 构建评估结果
            evaluation = {
                'total_points': total_points,
                'success_rate': success_rate,
                'avg_rise': avg_rise,
                'avg_days': avg_days,
                'max_rise': max_rise,
                'min_rise': min_rise,
                'score': score,
                'rise_threshold': self.rise_threshold,
                'max_days': self.max_days
            }
            
            self.logger.info("策略评估完成: 识别点数=%d, 成功率=%.2f%%, 平均涨幅=%.2f%%, 得分=%.4f", 
                           total_points, success_rate * 100, avg_rise * 100, score)
            
            return evaluation
            
        except Exception as e:
            self.logger.error("评估策略失败: %s", str(e))
            raise
            
    def _calculate_score(self, success_rate: float, avg_rise: float, avg_days: float) -> float:
        """
        计算策略得分
        
        参数:
        success_rate: 成功率
        avg_rise: 平均涨幅
        avg_days: 平均天数
        
        返回:
        float: 策略得分
        """
        # 从配置文件获取统一的评分参数
        scoring_config = self.config.get('strategy_scoring', {})
        
        # 成功率权重：50%
        success_weight = scoring_config.get('success_weight', 0.5)
        success_score = success_rate * success_weight
        
        # 平均涨幅权重：30%（相对于基准涨幅）
        rise_weight = scoring_config.get('rise_weight', 0.3)
        rise_benchmark = scoring_config.get('rise_benchmark', 0.1)  # 10%基准
        rise_score = min(avg_rise / rise_benchmark, 1.0) * rise_weight
        
        # 平均天数权重：20%（天数越少越好，以基准天数为准）
        days_weight = scoring_config.get('days_weight', 0.2)
        days_benchmark = scoring_config.get('days_benchmark', 10.0)  # 10天基准
        if avg_days > 0:
            days_score = min(days_benchmark / avg_days, 1.0) * days_weight
        else:
            days_score = 0.0
            
        total_score = success_score + rise_score + days_score
        return total_score
        
    def visualize_backtest(self, backtest_results: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        可视化回测结果
        
        参数:
        backtest_results: 回测结果
        save_path: 保存路径，如果为None则自动生成
        
        返回:
        str: 图表文件路径
        """
        self.logger.info("可视化回测结果")
        
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('回测结果分析', fontsize=16, fontweight='bold')
            
            # 1. 价格曲线和相对低点
            ax1 = axes[0, 0]
            ax1.plot(backtest_results['date'], backtest_results['close'], 
                    label='收盘价', linewidth=1, alpha=0.8)
            
            # 标记相对低点
            low_points = backtest_results[backtest_results['is_low_point']]
            if len(low_points) > 0:
                ax1.scatter(low_points['date'], low_points['close'], 
                          color='red', marker='^', s=50, label='相对低点', zorder=5)
            
            ax1.set_title(f'价格走势与相对低点\n(涨幅阈值: {self.rise_threshold:.1%}, 最大观察天数: {self.max_days}天)')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 涨幅分布
            ax2 = axes[0, 1]
            if len(low_points) > 0:
                rises = low_points['future_max_rise'] * 100
                ax2.hist(rises, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(x=self.rise_threshold * 100, color='red', linestyle='--', 
                          label=f'目标涨幅: {self.rise_threshold:.1%}')
                ax2.set_title(f'相对低点后的最大涨幅分布\n(目标: {self.rise_threshold:.1%}, 最大观察: {self.max_days}天)')
                ax2.set_xlabel('涨幅 (%)')
                ax2.set_ylabel('频次')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '无相对低点数据', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title(f'相对低点后的最大涨幅分布\n(目标: {self.rise_threshold:.1%}, 最大观察: {self.max_days}天)')
            
            # 3. 达到目标涨幅的天数分布
            ax3 = axes[1, 0]
            if len(low_points) > 0:
                successful_points = low_points[low_points['days_to_rise'] > 0]
                if len(successful_points) > 0:
                    days = successful_points['days_to_rise']
                    ax3.hist(days, bins=range(1, self.max_days + 2), alpha=0.7, 
                           color='lightgreen', edgecolor='black')
                    ax3.axvline(x=self.max_days, color='orange', linestyle='--', 
                              label=f'最大观察天数: {self.max_days}天')
                    ax3.set_title(f'达到目标涨幅所需天数分布\n(目标涨幅: {self.rise_threshold:.1%})')
                    ax3.set_xlabel('天数')
                    ax3.set_ylabel('频次')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, '无成功案例', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=14)
                    ax3.set_title(f'达到目标涨幅所需天数分布\n(目标涨幅: {self.rise_threshold:.1%})')
            else:
                ax3.text(0.5, 0.5, '无相对低点数据', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title(f'达到目标涨幅所需天数分布\n(目标涨幅: {self.rise_threshold:.1%})')
            
            # 4. 策略评估指标
            ax4 = axes[1, 1]
            evaluation = self.evaluate_strategy(backtest_results)
            
            metrics = ['成功率', '平均涨幅', '平均天数', '综合得分']
            values = [
                evaluation['success_rate'],
                evaluation['avg_rise'],
                evaluation['avg_days'] / self.max_days,  # 标准化
                evaluation['score']
            ]
            
            bars = ax4.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            ax4.set_title(f'策略评估指标\n(涨幅阈值: {self.rise_threshold:.1%}, 最大天数: {self.max_days}天)')
            ax4.set_ylabel('数值')
            ax4.set_ylim(0, 1)
            
            # 在柱状图上添加数值标签
            for bar, value, metric in zip(bars, values, metrics):
                height = bar.get_height()
                if metric == '平均天数':
                    label = f'{evaluation["avg_days"]:.1f}天'
                elif metric == '平均涨幅':
                    label = f'{value:.1%}'
                elif metric == '成功率':
                    label = f'{value:.1%}'
                else:
                    label = f'{value:.3f}'
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        label, ha='center', va='bottom')
            
            ax4.grid(True, alpha=0.3)
            
            # 在图表底部添加策略参数信息
            confidence_weights = self.config.get('strategy', {}).get('confidence_weights', {})
            param_info = f"策略参数: 涨幅阈值={self.rise_threshold:.1%}, 最大观察天数={self.max_days}天, RSI超卖阈值={confidence_weights.get('rsi_oversold_threshold', 30)}, RSI偏低阈值={confidence_weights.get('rsi_low_threshold', 40)}, 置信度阈值={self.config.get('final_threshold', 0.5):.2f}"
            plt.figtext(0.5, 0.02, param_info, ha='center', fontsize=10, 
                       bbox=dict(facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08)  # 为底部参数信息留出空间
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 创建子目录结构
                charts_dir = os.path.join(self.results_dir, 'charts')
                strategy_dir = os.path.join(charts_dir, 'strategy_analysis')
                
                for directory in [self.results_dir, charts_dir, strategy_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                save_path = os.path.join(strategy_dir, f'backtest_analysis_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("回测结果可视化完成，保存到: %s", save_path)
            return save_path
            
        except Exception as e:
            self.logger.error("可视化回测结果失败: %s", str(e))
            raise
            
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        更新策略参数
        
        参数:
        params: 新参数
        """
        # 🔧 修复：添加参数更新的详细日志追踪
        param_count = len(params)
        # self.logger.info(f"更新策略参数({param_count}个): {', '.join(params)}")
        
        # 更新基础参数
        if 'rise_threshold' in params:
            self.rise_threshold = params['rise_threshold']
            
        if 'max_days' in params:
            self.max_days = params['max_days']
        
        # 确保confidence_weights存在
        if 'confidence_weights' not in self.config['strategy']:
            self.config['strategy']['confidence_weights'] = {}
        
        # 定义所有可能的参数及其存储位置
        confidence_weight_params = [
            'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
            'dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight',
            # 🆕 新增的confidence_weights参数
            'volume_panic_bonus', 'volume_surge_bonus', 'volume_shrink_penalty',
            'bb_lower_near', 'price_decline_threshold', 'decline_threshold',
            'rsi_uptrend_min', 'rsi_uptrend_max', 'rsi_pullback_threshold',
            'rsi_uptrend_pullback', 'rsi_overbought_correction'
        ]
        
        strategy_level_params = [
            'volume_weight', 'price_momentum_weight', 'bb_near_threshold',
            'volume_panic_threshold', 'volume_surge_threshold', 'volume_shrink_threshold'
        ]
        
        # 更新confidence_weights中的参数
        for param in confidence_weight_params:
            if param in params:
                self.config['strategy']['confidence_weights'][param] = params[param]
        
        # 更新strategy级别的参数
        for param in strategy_level_params:
            if param in params:
                self.config['strategy'][param] = params[param]
        
        # 参数更新完成
        self.logger.debug("策略参数更新完成")
                        
    def get_params(self) -> Dict[str, Any]:
        """
        获取当前策略参数
        
        返回:
        dict: 当前参数
        """
        confidence_weights = self.config.get('strategy', {}).get('confidence_weights', {})
        
        # 获取所有可用的参数，包括新增的AI优化参数
        params = {
            'rise_threshold': self.rise_threshold,
            'max_days': self.max_days,
            'rsi_oversold_threshold': confidence_weights.get('rsi_oversold_threshold', 30),
            'rsi_low_threshold': confidence_weights.get('rsi_low_threshold', 40),
            'final_threshold': self.config.get('final_threshold', 0.5),
            # 原有AI优化参数
            'dynamic_confidence_adjustment': confidence_weights.get('dynamic_confidence_adjustment', 0.1),
            'market_sentiment_weight': confidence_weights.get('market_sentiment_weight', 0.15),
            'trend_strength_weight': confidence_weights.get('trend_strength_weight', 0.12),
            # 🆕 新增高重要度参数
            'volume_panic_threshold': confidence_weights.get('volume_panic_threshold', 1.45),
            'volume_surge_threshold': confidence_weights.get('volume_surge_threshold', 1.25),
            'volume_shrink_threshold': confidence_weights.get('volume_shrink_threshold', 0.78),
            'bb_near_threshold': confidence_weights.get('bb_near_threshold', 1.018),
            'rsi_uptrend_min': confidence_weights.get('rsi_uptrend_min', 35),
            'rsi_uptrend_max': confidence_weights.get('rsi_uptrend_max', 85),
            # 🆕 新增中重要度参数
            'volume_panic_bonus': confidence_weights.get('volume_panic_bonus', 0.12),
            'volume_surge_bonus': confidence_weights.get('volume_surge_bonus', 0.06),
            'volume_shrink_penalty': confidence_weights.get('volume_shrink_penalty', 0.68),
            'bb_lower_near': confidence_weights.get('bb_lower_near', 0.22),
            'price_decline_threshold': confidence_weights.get('price_decline_threshold', -0.018),
            'decline_threshold': confidence_weights.get('decline_threshold', -0.048)
        }
        
        # 添加其他可能存在的参数
        additional_params = [
            'volume_weight', 'price_momentum_weight', 'bb_near_threshold',
            'volume_panic_threshold', 'volume_surge_threshold', 'volume_shrink_threshold'
        ]
        
        for param in additional_params:
            if param in confidence_weights:
                params[param] = confidence_weights[param]
        
        # 检查strategy级别是否有这些参数（某些参数可能存储在strategy级别而不是confidence_weights中）
        strategy_config = self.config.get('strategy', {})
        for param in additional_params:
            if param in strategy_config:
                params[param] = strategy_config[param]
        
        return params

    def get_current_params(self) -> Dict[str, Any]:
        """
        获取当前策略参数（别名方法，与get_params()相同）
        
        返回:
        dict: 当前参数
        """
        return self.get_params()

