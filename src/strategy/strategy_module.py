#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略执行模块
实现相对低点识别算法和回测功能
"""

import os
import logging
import pandas as pd

from typing import Dict, Any

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
            trend_regime = 'sideways'
            
            # 从配置文件获取置信度权重
            strategy_config = self.config.get('strategy', {})
            # 优化参数现在在根级别的confidence_weights中
            confidence_config = self.config.get('confidence_weights', {})
            
            # 统计触发的关键信号数量用于最终确认（提高精准度）
            signal_count = 0
            
            # 条件1: 价格低于多条移动平均线
            if ma5 is not None and ma10 is not None and ma20 is not None:
                if latest_price < ma5 and latest_price < ma10 and latest_price < ma20:
                    base_confidence = confidence_config.get('ma_all_below', 0.3)
                    volume_ratio = data.iloc[-1]['volume_ratio'] if 'volume_ratio' in data.columns else 1.0
                    price_decline = data.iloc[-1]['price_change'] if 'price_change' in data.columns else 0.0
                    volume_panic_threshold = confidence_config.get('volume_panic_threshold', 1.4)
                    volume_surge_threshold = confidence_config.get('volume_surge_threshold', 1.2)
                    volume_shrink_threshold = confidence_config.get('volume_shrink_threshold', 0.8)
                    price_decline_threshold = confidence_config.get('price_decline_threshold', -0.02)

                    if volume_ratio > volume_panic_threshold and price_decline < price_decline_threshold:
                        panic_bonus = confidence_config.get('volume_panic_bonus', 0.1)
                        confidence += base_confidence + panic_bonus
                        is_low_point = True
                        signal_count += 1
                        reasons.append(f"价格跌破所有均线+恐慌性抛售(成交量放大{volume_ratio:.1f}倍)")
                    elif volume_ratio > volume_surge_threshold:
                        surge_bonus = confidence_config.get('volume_surge_bonus', 0.05)
                        confidence += base_confidence + surge_bonus
                        is_low_point = True
                        signal_count += 1
                        reasons.append(f"价格跌破所有均线+温和放量(成交量放大{volume_ratio:.1f}倍)")
                    elif volume_ratio < volume_shrink_threshold:
                        shrink_penalty = confidence_config.get('volume_shrink_penalty', 0.7)
                        confidence += base_confidence * shrink_penalty
                        reasons.append(f"价格跌破所有均线+成交量萎缩(可能是下跌通道)")
                    else:
                        confidence += base_confidence
                        is_low_point = True
                        signal_count += 1
                        reasons.append("价格低于MA5/MA10/MA20")
                elif latest_price < ma10 and latest_price < ma20:
                    confidence += confidence_config.get('ma_partial_below', 0.2)
                    reasons.append("价格低于MA10/MA20")

            # 🔥 成交量分析 - 恐慌性抛售检测
            volume_panic_bonus = confidence_config.get('volume_panic_bonus', 0.15)
            if len(data) >= 20:
                avg_volume_20 = data['volume'].tail(20).mean()
                avg_volume_5 = data['volume'].tail(5).mean()
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
                volume_ratio_5d = avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1
                
                # 动态判断涨跌方向，避免文案误导
                price_change_val = 0.0
                if len(data) >= 2:
                    price_change_val = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                direction_str = "下跌" if price_change_val < 0 else "上涨"

                if volume_ratio > 2.5:
                    confidence += volume_panic_bonus * 2.5
                    reasons.append(f"极度放量{direction_str}(量比{volume_ratio:.1f})")
                elif volume_ratio > 2.0:
                    confidence += volume_panic_bonus * 2.0
                    reasons.append(f"大幅放量{direction_str}(量比{volume_ratio:.1f})")
                elif volume_ratio > 1.5:
                    confidence += volume_panic_bonus * 1.2
                    reasons.append(f"放量{direction_str}(量比{volume_ratio:.1f})")
                elif volume_ratio > 1.2:
                    confidence += volume_panic_bonus * 0.8
                    reasons.append(f"温和放量{direction_str}(量比{volume_ratio:.1f})")
                elif volume_ratio < 0.5:
                    confidence += volume_panic_bonus * 0.4
                    reasons.append(f"缩量{direction_str}(量比{volume_ratio:.1f})")
                elif volume_ratio < 0.8:
                    confidence += volume_panic_bonus * 0.2
                    reasons.append(f"成交量偏低(量比{volume_ratio:.1f})")
                
                # 5日平均成交量分析
                if volume_ratio_5d > 1.3:
                    confidence += volume_panic_bonus * 0.6
                    reasons.append(f"近期成交活跃(5日量比{volume_ratio_5d:.1f})")
                elif volume_ratio_5d < 0.7:
                    confidence += volume_panic_bonus * 0.3
                    reasons.append(f"近期成交低迷(5日量比{volume_ratio_5d:.1f})")

            # 条件2: RSI超卖
            if rsi is not None:
                rsi_oversold_threshold = confidence_config.get('rsi_oversold_threshold', 30)
                rsi_low_threshold = confidence_config.get('rsi_low_threshold', 40)
                rsi_moderate_threshold = confidence_config.get('rsi_moderate_threshold', 50)
                if rsi < rsi_oversold_threshold:
                    is_low_point = True
                    signal_count += 1
                    confidence += confidence_config.get('rsi_oversold', 0.35)
                    reasons.append(f"RSI超卖({rsi:.2f})")
                elif rsi < rsi_low_threshold:
                    confidence += confidence_config.get('rsi_low', 0.2)
                    reasons.append(f"RSI偏低({rsi:.2f})")
                elif rsi < rsi_moderate_threshold:
                    confidence += confidence_config.get('rsi_moderate', 0.1)
                    reasons.append(f"RSI中性偏低({rsi:.2f})")

            # 条件3: MACD信号
            if macd is not None:
                macd_negative_threshold = confidence_config.get('macd_negative_threshold', -0.01)
                macd_weak_negative_threshold = confidence_config.get('macd_weak_negative_threshold', 0.005)
                if macd < macd_negative_threshold:
                    confidence += confidence_config.get('macd_negative', 0.15)
                    reasons.append(f"MACD负值({macd:.4f})")
                elif macd < macd_weak_negative_threshold:
                    confidence += confidence_config.get('macd_weak_negative', 0.08)
                    reasons.append(f"MACD弱负值({macd:.4f})")

            # 条件4: 价格接近布林带下轨
            if bb_lower is not None:
                bb_near_threshold = confidence_config.get('bb_near_threshold', 1.02)
                if latest_price <= bb_lower * bb_near_threshold:
                    is_low_point = True
                    signal_count += 1
                    confidence += confidence_config.get('bb_lower_near', 0.25)
                    reasons.append("价格接近布林带下轨")

            # 条件5: 价格动量分析
            if len(data) >= 5:
                price_momentum_weight = confidence_config.get('price_momentum_weight', 0.1)
                recent_prices = data['close'].tail(5).values
                price_change_1d = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
                price_change_3d = (recent_prices[-1] - recent_prices[-4]) / recent_prices[-4] if len(recent_prices) >= 4 else 0
                
                if price_change_1d < -0.02:  # 单日跌幅超过2%
                    confidence += price_momentum_weight * 1.5
                    reasons.append(f"单日大跌({price_change_1d:.2%})")
                elif price_change_1d < -0.01:  # 单日跌幅超过1%
                    confidence += price_momentum_weight * 1.0
                    reasons.append(f"单日下跌({price_change_1d:.2%})")
                elif price_change_1d < 0:  # 单日下跌
                    confidence += price_momentum_weight * 0.5
                    reasons.append(f"单日微跌({price_change_1d:.2%})")
                
                if price_change_3d < -0.05:  # 3日跌幅超过5%
                    confidence += price_momentum_weight * 1.0
                    reasons.append(f"3日累计大跌({price_change_3d:.2%})")
                elif price_change_3d < -0.02:  # 3日跌幅超过2%
                    confidence += price_momentum_weight * 0.6
                    reasons.append(f"3日累计下跌({price_change_3d:.2%})")

            # 🔄 趋势强度权重 - 减弱上升趋势惩罚，突出下跌趋势加分
            trend_strength_weight = confidence_config.get('trend_strength_weight', 0.12)
            if len(data) >= 20:
                x_long = np.arange(20)
                y_long = data['close'].tail(20).values
                slope_long = np.polyfit(x_long, y_long, 1)[0]
                trend_strength_long = abs(slope_long) / y_long.mean()

                # 简单趋势判别：结合斜率方向与价格相对MA20位置
                price_above_ma20 = ma20 is not None and latest_price >= ma20
                price_below_ma20 = ma20 is not None and latest_price < ma20
                if slope_long > 0 and price_above_ma20:
                    trend_regime = 'bull'
                elif slope_long < 0 and price_below_ma20:
                    trend_regime = 'bear'
                else:
                    trend_regime = 'sideways'

                if trend_strength_long > 0.01:
                    if slope_long > 0:
                        # 上升趋势中不给惩罚，避免与相对低点冲突，仅轻微降低置信度
                        confidence -= trend_strength_weight * 0.03
                        reasons.append(f"强上涨趋势轻微扣分(-{trend_strength_weight * 0.03:.3f})")
                    else:
                        confidence += trend_strength_weight * 2.0
                        reasons.append(f"强下跌趋势加分(+{trend_strength_weight * 2.0:.3f})")
                elif trend_strength_long < 0.002:
                    confidence += trend_strength_weight * 0.8
                    reasons.append(f"弱趋势调整(+{trend_strength_weight * 0.8:.3f})")

            # 🟢 多头趋势的回撤买点识别（与原有相对低点逻辑兼容，可作为加分项）
            if trend_regime == 'bull':
                try:
                    up_ma_support_w = confidence_config.get('uptrend_ma_support', 0.3)
                    up_pullback_w = confidence_config.get('uptrend_pullback_bonus', 0.3)
                    up_vol_pullback_w = confidence_config.get('uptrend_volume_pullback', 0.2)
                    rsi_min = confidence_config.get('rsi_uptrend_min', 30)
                    rsi_max = confidence_config.get('rsi_uptrend_max', 80)
                    rsi_pb_th = confidence_config.get('rsi_pullback_threshold', 3)

                    ma_support = False
                    if ma10 is not None and ma20 is not None:
                        ma_support = (ma10 >= ma20) and (latest_price >= ma20)
                    elif ma20 is not None:
                        ma_support = latest_price >= ma20

                    if ma_support:
                        confidence += up_ma_support_w
                        reasons.append("多头趋势: 均线支撑(MA10≥MA20且价在MA20上方)")

                    # RSI健康区间内的回撤
                    rsi_valid = rsi is not None and not pd.isna(rsi)
                    rsi_prev = None
                    if 'rsi' in data.columns and len(data) >= 2:
                        rsi_prev = data['rsi'].iloc[-2]
                    rsi_prev_valid = rsi_prev is not None and not pd.isna(rsi_prev)

                    if rsi_valid and rsi_prev_valid:
                        rsi_in_trend = (rsi_min <= rsi <= rsi_max)
                        rsi_drop_ok = (rsi_prev - rsi) >= rsi_pb_th
                        if rsi_in_trend and rsi_drop_ok:
                            confidence += up_pullback_w
                            reasons.append(f"多头趋势: RSI健康回撤({rsi_prev:.1f}→{rsi:.1f}, -{(rsi_prev - rsi):.1f})")

                    # 回撤期缩量更优
                    vol_ratio = data.iloc[-1]['volume_ratio'] if 'volume_ratio' in data.columns else None
                    if vol_ratio is not None and pd.notna(vol_ratio) and vol_ratio < 1.0:
                        confidence += up_vol_pullback_w
                        reasons.append(f"多头趋势: 回撤缩量(量比{vol_ratio:.2f})")
                except Exception as _e:
                    # 保守处理，出现异常不影响主流程
                    pass
            
            # 🆕 动态置信度调整：多信号共振时额外加分
            dynamic_adj_weight = confidence_config.get('dynamic_confidence_adjustment', 0.15)
            # 移除最少信号数(min_signals)逻辑，仅根据置信度阈值进行判断
            if signal_count >= 3:
                confidence += dynamic_adj_weight * 1.0
                reasons.append(f"多信号确认奖励(+{dynamic_adj_weight * 1.0:.3f})")
            elif signal_count >= 2:
                confidence += dynamic_adj_weight * 0.3
                reasons.append(f"最低信号确认(+{dynamic_adj_weight * 0.3:.3f})")

            # 最终判断 - 仅门控（去除多信号数量限制）
            confidence_threshold = confidence_config.get('final_threshold', 0.5)
            if confidence >= confidence_threshold:
                is_low_point = True
                reasons.append(f"置信度达阈值({confidence:.2f} ≥ {confidence_threshold:.2f})")
            else:
                if is_low_point:
                    reasons.append(f"门控未通过: 置信度不足(置信度{confidence:.2f}/{confidence_threshold:.2f})")
                is_low_point = False
            
            # 基础置信度调整：确保即使没有明显信号也有最小置信度
            min_base_confidence = confidence_config.get('min_base_confidence', 0.10)
            base_confidence_ratio = confidence_config.get('base_confidence_ratio', 0.6)
            
            # 调试信息：检查参数是否正确读取
            # self.logger.debug(f"min_base_confidence: {min_base_confidence}, confidence before adjustment: {confidence}")
            
            if confidence <= 0.01:  # 几乎没有信号
                # 给予最小基础置信度
                confidence = min_base_confidence
                reasons.append(f"最小基础置信度({min_base_confidence:.2f})")
            elif confidence > 0 and not is_low_point:
                # 对于有一定信号但未达到交易阈值的情况，保留部分置信度
                confidence = max(confidence * base_confidence_ratio, min_base_confidence)
                reasons.append(f"基础置信度保留({base_confidence_ratio:.1f}倍)")
            
            # 确保最终置信度不低于最小值
            if confidence < min_base_confidence:
                confidence = min_base_confidence
                if "最小基础置信度" not in str(reasons):
                    reasons.append(f"最小置信度保障({min_base_confidence:.2f})")
            
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
                    'bb_lower': bb_lower,
                    'trend_regime': trend_regime
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
        回测策略 - 使用T+1开盘价买入的真实逻辑
        
        参数:
        data: 历史数据
        
        返回:
        pandas.DataFrame: 回测结果
        """
        self.logger.info("开始T+1真实回测，数据长度: %d", len(data))
        
        try:
            # 复制数据避免修改原数据
            backtest_data = data.copy()
            
            # 确保数据有open列用于T+1买入
            if 'open' not in backtest_data.columns:
                self.logger.error("数据缺少open列，无法进行T+1开盘价回测")
                raise ValueError("数据必须包含open列用于T+1开盘价买入")
            
            # 添加回测结果列
            backtest_data['is_low_point'] = False
            backtest_data['entry_price'] = 0.0  # T+1开盘价
            backtest_data['exit_price'] = 0.0
            backtest_data['exit_date'] = None
            backtest_data['trade_return'] = 0.0
            backtest_data['days_to_target'] = 0
            
            # 信号列表用于收集所有识别的低点
            signals = []

            # index为交易日序号，date为实际交易日，未出现的日期视为非交易日
            # 只遍历到倒数max_days-1个交易日，保证T+1买入后还有足够的持有期
            for i in range(len(backtest_data) - self.max_days - 1):
                # 使用策略识别相对低点（基于技术指标，而不是未来结果）
                # 传递从开始到当前位置的所有历史数据，让算法基于历史数据判断当前时点
                historical_data = backtest_data.iloc[:i+1].copy()
                identification_result = self.identify_relative_low(historical_data)
                
                # 记录识别结果
                backtest_data.loc[i, 'is_low_point'] = identification_result['is_low_point']
                
                # 如果识别为相对低点，进行T+1交易模拟
                if identification_result['is_low_point']:
                    signal_date = backtest_data.iloc[i]['date']
                    
                    # T+1买入：使用次日开盘价
                    if i + 1 < len(backtest_data):
                        entry_price = backtest_data.iloc[i + 1]['open']
                        entry_date = backtest_data.iloc[i + 1]['date']
                        
                        # 记录买入价格
                        backtest_data.loc[i, 'entry_price'] = entry_price
                        
                        # 寻找退出点：从买入日开始，最多持有max_days天
                        exit_price = None
                        exit_date = None
                        days_to_target = 0
                        
                        # 检查未来max_days天的表现
                        for j in range(1, self.max_days + 1):
                            if i + 1 + j >= len(backtest_data):
                                break  # 超出数据范围
                            
                            future_high = backtest_data.iloc[i + 1 + j]['high']
                            future_close = backtest_data.iloc[i + 1 + j]['close']
                            future_date = backtest_data.iloc[i + 1 + j]['date']
                            
                            # 检查是否在当日达到目标涨幅
                            if future_high >= entry_price * (1 + self.rise_threshold):
                                # 按目标价格卖出
                                exit_price = entry_price * (1 + self.rise_threshold)
                                exit_date = future_date
                                days_to_target = j
                                break
                        
                        # 如果没有达到目标，在最后一天收盘价卖出
                        if exit_price is None:
                            max_check_idx = min(i + 1 + self.max_days, len(backtest_data) - 1)
                            exit_price = backtest_data.iloc[max_check_idx]['close']
                            exit_date = backtest_data.iloc[max_check_idx]['date']
                            days_to_target = 0  # 未达到目标
                        
                        # 计算交易收益率
                        trade_return = (exit_price / entry_price) - 1
                        
                        # 记录交易结果
                        backtest_data.loc[i, 'exit_price'] = exit_price
                        backtest_data.loc[i, 'exit_date'] = exit_date
                        backtest_data.loc[i, 'trade_return'] = trade_return
                        backtest_data.loc[i, 'days_to_target'] = days_to_target
                        
                        # 收集信号用于收益率计算
                        signals.append({
                            'signal_date': signal_date,
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'exit_price': exit_price,
                            'trade_return': trade_return,
                            'days_to_target': days_to_target
                        })
                        
            self.logger.info("T+1真实回测完成，识别信号数: %d", len(signals))
            return backtest_data
            
        except Exception as e:
            self.logger.error("T+1真实回测失败: %s", str(e))
            raise
            
    def evaluate_strategy(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        评估策略性能 - 基于T+1真实交易结果
        
        参数:
        backtest_data: 回测结果数据
        
        返回:
        dict: 包含各种性能指标的字典
        """
        low_points = backtest_data[backtest_data['is_low_point'] == True]
        total_low_points = len(low_points)
        
        if total_low_points == 0:
            result = {
                'total_signals': 0,
                'total_trades': 0,
                'success_count': 0,
                'success_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_holding_days': 0.0,
                'profit_factor': 0.0,
                'score': 0.0
            }
            self._last_evaluation = result
            return result
        
        # 有效交易统计（成功买入的交易）
        valid_trades = low_points[low_points['entry_price'] > 0]
        total_trades = len(valid_trades)
        
        if total_trades == 0:
            result = {
                'total_signals': total_low_points,
                'total_trades': 0,
                'success_count': 0,
                'success_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_holding_days': 0.0,
                'profit_factor': 0.0,
                'score': 0.0
            }
            self._last_evaluation = result
            return result
        
        # 成功交易统计（达到目标涨幅）
        successful_trades = valid_trades[valid_trades['days_to_target'] > 0]
        success_count = len(successful_trades)
        success_rate = success_count / total_trades if total_trades > 0 else 0.0
        
        # 收益率统计
        trade_returns = valid_trades['trade_return']
        avg_return = trade_returns.mean()
        total_return = (1 + trade_returns).prod() - 1  # 复合收益率
        
        # 胜率统计（正收益交易比例）
        positive_trades = valid_trades[valid_trades['trade_return'] > 0]
        win_rate = len(positive_trades) / total_trades if total_trades > 0 else 0.0
        
        # 计算总利润值（替代夏普比率）
        total_profit = trade_returns.sum()
        
        # 计算最大回撤
        cumulative_returns = (1 + trade_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 平均持有天数统计
        all_holding_days = []
        for _, trade in valid_trades.iterrows():
            if trade['days_to_target'] > 0:
                all_holding_days.append(trade['days_to_target'])
            else:
                # 未达到目标的交易，使用max_days作为持有期
                all_holding_days.append(self.max_days)
        
        avg_holding_days = np.mean(all_holding_days) if all_holding_days else 0.0
        
        # 新增：利润因子（Profit Factor）
        trades_nonzero = trade_returns[trade_returns != 0]
        total_gains = trades_nonzero[trades_nonzero > 0].sum()
        total_losses = abs(trades_nonzero[trades_nonzero < 0].sum())
        if total_gains == 0 and total_losses == 0:
            profit_factor = 0.0
        elif total_losses == 0:
            profit_factor = 999.0
        else:
            profit_factor = float(total_gains / total_losses)
        
        # 基于PF与交易次数的统一打分（供AI优化与报告使用）
        min_trades_threshold = int(self.config.get('optimization_constraints', {}).get('min_trades_threshold', 10))
        if total_trades < min_trades_threshold:
            pf_score = 0.0
        else:
            pf_score = float(profit_factor * np.log1p(total_trades))
        
        self.logger.info(f"策略评估完成 - 信号数: {total_low_points}, 交易数: {total_trades}, 成功率: {success_rate:.2%}")
        
        result = {
            'total_signals': total_low_points,       # 总信号数
            'total_trades': total_trades,            # 总交易数
            'success_count': success_count,          # 成功交易数
            'success_rate': success_rate,            # 成功率
            'avg_return': avg_return,                # 平均收益率
            'total_return': total_return,            # 总收益率
            'total_profit': total_profit,            # 总利润值
            'max_drawdown': max_drawdown,            # 最大回撤
            'win_rate': win_rate,                    # 胜率
            'avg_holding_days': avg_holding_days,    # 平均持有天数
            'profit_factor': profit_factor,          # 新增：利润因子
            'pf_score': pf_score,                    # 保留：PF×log(交易数) 作为参考
            'score': float(total_profit)             # 统一得分改为：总利润值（按用户要求）
        }
        # 缓存以供打分函数使用
        self._last_evaluation = result
        return result
        
    # 已移除：visualize_backtest 方法（主类不再提供图表可视化）

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
        
        # 确保根级别 confidence_weights 存在
        if 'confidence_weights' not in self.config:
            self.config['confidence_weights'] = {}
        
        # 定义所有可能的参数及其存储位置（统一写入根级别 confidence_weights）
        confidence_weight_params = [
            'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
            'dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight',
            # 🆕 新增的confidence_weights参数
            'volume_panic_bonus', 'volume_surge_bonus', 'volume_shrink_penalty',
            'bb_lower_near', 'price_decline_threshold', 'decline_threshold',
            'rsi_uptrend_min', 'rsi_uptrend_max', 'rsi_pullback_threshold',
            'rsi_uptrend_pullback', 'rsi_overbought_correction',
            # 补全缺失的参数
            'ma_all_below', 'ma_partial_below', 'macd_negative', 'macd_weak_negative',
            'rsi_oversold', 'rsi_low', 'rsi_moderate',
            # 阈值与权重（在 strategy.yaml 中也位于 confidence_weights 下）
            'bb_near_threshold', 'volume_panic_threshold', 'volume_surge_threshold', 'volume_shrink_threshold',
            'volume_weight', 'price_momentum_weight'
        ]
        
        # 将上述参数全部更新到根级别 confidence_weights
        for param in confidence_weight_params:
            if param in params:
                self.config['confidence_weights'][param] = params[param]
        
        # 参数更新完成
        self.logger.debug("策略参数更新完成")
                        
    def get_params(self) -> Dict[str, Any]:
        """
        获取当前策略参数
        
        返回:
        dict: 当前参数
        """
        confidence_weights = self.config.get('confidence_weights', {})
        
        # 获取所有可用的参数，包括新增的AI优化参数
        params = {
            'rise_threshold': self.rise_threshold,
            'max_days': self.max_days,
            'rsi_oversold_threshold': confidence_weights.get('rsi_oversold_threshold', 30),
            'rsi_low_threshold': confidence_weights.get('rsi_low_threshold', 40),
            'final_threshold': confidence_weights.get('final_threshold', 0.5),
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

