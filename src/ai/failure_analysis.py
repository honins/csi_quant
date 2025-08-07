#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
失败案例分析模块
用于诊断策略失败的原因，并提供针对性的优化建议
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

class FailureAnalyzer:
    """失败案例分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化失败分析器
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 获取策略配置
        strategy_config = config.get('strategy', {})
        self.rise_threshold = strategy_config.get('rise_threshold', 0.04)
        self.max_days = strategy_config.get('max_days', 20)
        
        self.logger.info("失败案例分析器初始化完成")
    
    def analyze_failures(self, backtest_results, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析失败案例
        
        参数:
        backtest_results: 回测结果 (DataFrame或list)
        data: 原始数据
        
        返回:
        dict: 分析结果
        """
        self.logger.info("开始分析失败案例")
        
        # 转换为DataFrame格式
        if isinstance(backtest_results, list):
            backtest_results = pd.DataFrame(backtest_results)
        
        # 根据实际回测结果格式筛选失败案例
        # 失败定义：预测为相对低点但未来涨幅未达到阈值
        if 'success' in backtest_results.columns:
            failed_signals = backtest_results[backtest_results['success'] == False].copy()
        else:
            # 使用实际的回测结果字段
            rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            failed_signals = backtest_results[
                (backtest_results['is_low_point'] == True) & 
                (backtest_results['future_max_rise'] < rise_threshold)
            ].copy()
            # 添加success字段以保持兼容性
            failed_signals['success'] = False
        
        if len(failed_signals) == 0:
            return {
                'total_failures': 0,
                'failure_types': {},
                'recommendations': []
            }
        
        # 分析每个失败案例
        failure_analysis = []
        for idx, signal in failed_signals.iterrows():
            analysis = self._analyze_single_failure(signal, data)
            failure_analysis.append(analysis)
        
        # 按类型分类
        failure_types = self._categorize_failures(failure_analysis)
        
        # 生成优化建议
        recommendations = self._generate_recommendations(failure_types)
        
        result = {
            'total_failures': len(failed_signals),
            'failure_rate': len(failed_signals) / len(backtest_results) if len(backtest_results) > 0 else 0,
            'failure_types': failure_types,
            'detailed_analysis': failure_analysis,
            'recommendations': recommendations
        }
        
        self.logger.info(f"失败案例分析完成，共分析{len(failed_signals)}个失败案例")
        return result
    
    def _analyze_single_failure(self, signal: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析单个失败案例
        
        参数:
        signal: 信号记录
        data: 原始数据
        
        返回:
        dict: 单个失败案例的分析结果
        """
        # 适配不同的数据格式
        if 'date' in signal:
            signal_date = signal['date']
        elif 'timestamp' in signal:
            signal_date = signal['timestamp']
        else:
            # 使用索引作为日期
            signal_date = signal.name if hasattr(signal, 'name') else None
            
        if 'price' in signal:
            signal_price = signal['price']
        elif 'close' in signal:
            signal_price = signal['close']
        else:
            return {'type': 'data_error', 'reason': '无法找到价格信息'}
            
        if 'confidence' in signal:
            confidence = signal['confidence']
        else:
            confidence = 0.5  # 默认置信度
        
        # 获取信号后的价格数据
        signal_idx = data[data['date'] == signal_date].index
        if len(signal_idx) == 0:
            return {'type': 'data_error', 'reason': '无法找到信号日期对应的数据'}
        
        signal_idx = signal_idx[0]
        future_data = data.iloc[signal_idx:signal_idx + self.max_days + 5]  # 多取5天用于分析
        
        if len(future_data) < 2:
            return {'type': 'data_insufficient', 'reason': '信号后数据不足'}
        
        # 计算实际表现
        max_rise = 0
        max_rise_day = 0
        final_rise = 0
        
        for i in range(1, len(future_data)):  # 从第二天开始
            row = future_data.iloc[i]
            current_rise = (row['close'] - signal_price) / signal_price
            if current_rise > max_rise:
                max_rise = current_rise
                max_rise_day = i
            
            if i == min(self.max_days, len(future_data) - 1):
                final_rise = current_rise
        
        # 判断失败类型
        failure_type = self._classify_failure_type(max_rise, max_rise_day, final_rise, future_data, signal_price)
        
        return {
            'signal_date': signal_date,
            'signal_price': signal_price,
            'confidence': confidence,
            'max_rise': max_rise,
            'max_rise_day': max_rise_day,
            'final_rise': final_rise,
            'type': failure_type['type'],
            'reason': failure_type['reason'],
            'details': failure_type['details']
        }
    
    def _classify_failure_type(self, max_rise: float, max_rise_day: int, final_rise: float, 
                              future_data: pd.DataFrame, signal_price: float) -> Dict[str, Any]:
        """
        根据用户提出的三种类型对失败案例进行分类
        
        参数:
        max_rise: 最大涨幅
        max_rise_day: 最大涨幅出现的天数
        final_rise: 最终涨幅
        future_data: 信号后的数据
        signal_price: 信号价格
        
        返回:
        dict: 失败类型分析
        """
        
        # 类型1: "功亏一篑"型失败
        if max_rise >= self.rise_threshold * 0.75:  # 达到成功阈值的75%以上
            if max_rise_day > self.max_days:  # 在观察期外才达到目标
                return {
                    'type': 'near_miss_timing',
                    'reason': f'在第{max_rise_day}天达到{max_rise:.2%}涨幅，超出{self.max_days}天观察期',
                    'details': {
                        'max_rise_achieved': max_rise,
                        'days_to_target': max_rise_day,
                        'target_threshold': self.rise_threshold,
                        'observation_period': self.max_days
                    }
                }
            elif max_rise < self.rise_threshold:
                return {
                    'type': 'near_miss_magnitude',
                    'reason': f'最高涨幅{max_rise:.2%}，接近但未达到{self.rise_threshold:.2%}目标',
                    'details': {
                        'max_rise_achieved': max_rise,
                        'target_threshold': self.rise_threshold,
                        'shortfall': self.rise_threshold - max_rise
                    }
                }
        
        # 类型2: "横盘不动"型失败
        if abs(max_rise) < 0.02 and abs(final_rise) < 0.02:  # 最大波动小于2%
            # 分析成交量情况
            avg_volume = future_data['volume'].mean() if 'volume' in future_data.columns else 0
            signal_volume = future_data.iloc[0]['volume'] if 'volume' in future_data.columns else 0
            volume_ratio = avg_volume / signal_volume if signal_volume > 0 else 1
            
            return {
                'type': 'sideways_stagnation',
                'reason': f'信号后横盘整理，最大涨幅仅{max_rise:.2%}，缺乏上涨动能',
                'details': {
                    'max_rise': max_rise,
                    'final_rise': final_rise,
                    'volume_ratio': volume_ratio,
                    'price_volatility': future_data['close'].std() / signal_price if len(future_data) > 1 else 0
                }
            }
        
        # 类型3: "接飞刀"型失败
        if final_rise < -0.05:  # 最终下跌超过5%
            # 分析下跌的严重程度
            min_price = future_data['close'].min()
            max_decline = (min_price - signal_price) / signal_price
            
            return {
                'type': 'catching_falling_knife',
                'reason': f'信号后继续大幅下跌，最大跌幅{max_decline:.2%}，最终跌幅{final_rise:.2%}',
                'details': {
                    'max_decline': max_decline,
                    'final_decline': final_rise,
                    'continued_downtrend': True,
                    'risk_level': 'high'
                }
            }
        
        # 其他类型的失败
        return {
            'type': 'other_failure',
            'reason': f'其他类型失败，最大涨幅{max_rise:.2%}，最终涨幅{final_rise:.2%}',
            'details': {
                'max_rise': max_rise,
                'final_rise': final_rise,
                'classification': 'unclassified'
            }
        }
    
    def _categorize_failures(self, failure_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对失败案例进行分类统计
        
        参数:
        failure_analysis: 失败案例分析列表
        
        返回:
        dict: 分类统计结果
        """
        categories = {
            'near_miss_timing': [],      # 功亏一篑-时机问题
            'near_miss_magnitude': [],   # 功亏一篑-幅度问题
            'sideways_stagnation': [],   # 横盘不动
            'catching_falling_knife': [], # 接飞刀
            'other_failure': []          # 其他
        }
        
        for analysis in failure_analysis:
            failure_type = analysis['type']
            if failure_type in categories:
                categories[failure_type].append(analysis)
            else:
                categories['other_failure'].append(analysis)
        
        # 计算统计信息
        total_failures = len(failure_analysis)
        result = {}
        
        for category, cases in categories.items():
            count = len(cases)
            percentage = count / total_failures * 100 if total_failures > 0 else 0
            
            result[category] = {
                'count': count,
                'percentage': percentage,
                'cases': cases
            }
        
        return result
    
    def _generate_recommendations(self, failure_types: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据失败类型分析生成优化建议
        
        参数:
        failure_types: 失败类型统计
        
        返回:
        list: 优化建议列表
        """
        recommendations = []
        
        # 针对"功亏一篑"型失败的建议
        near_miss_total = (failure_types.get('near_miss_timing', {}).get('count', 0) + 
                          failure_types.get('near_miss_magnitude', {}).get('count', 0))
        
        if near_miss_total > 0:
            near_miss_percentage = (near_miss_total / sum(ft.get('count', 0) for ft in failure_types.values())) * 100
            
            if near_miss_percentage > 30:  # 如果功亏一篑型失败超过30%
                recommendations.append({
                    'type': 'parameter_adjustment',
                    'priority': 'high',
                    'target': 'near_miss_failures',
                    'description': '针对功亏一篑型失败的优化',
                    'actions': [
                        {
                            'parameter': 'rise_threshold',
                            'current_value': self.rise_threshold,
                            'suggested_value': self.rise_threshold * 0.9,  # 降低成功阈值10%
                            'reason': '降低成功阈值，将接近成功的案例转为成功'
                        },
                        {
                            'parameter': 'max_days',
                            'current_value': self.max_days,
                            'suggested_value': self.max_days + 5,  # 延长观察期
                            'reason': '延长观察期，给予更多时间达到目标涨幅'
                        }
                    ]
                })
        
        # 针对"横盘不动"型失败的建议
        sideways_count = failure_types.get('sideways_stagnation', {}).get('count', 0)
        if sideways_count > 0:
            sideways_percentage = (sideways_count / sum(ft.get('count', 0) for ft in failure_types.values())) * 100
            
            if sideways_percentage > 25:  # 如果横盘型失败超过25%
                recommendations.append({
                    'type': 'signal_enhancement',
                    'priority': 'high',
                    'target': 'sideways_stagnation',
                    'description': '针对横盘不动型失败的优化',
                    'actions': [
                        {
                            'parameter': 'volume_shrink_penalty',
                            'current_value': self.config.get('confidence_weights', {}).get('volume_shrink_penalty', 0.7),
                            'suggested_value': 0.5,  # 加强成交量萎缩的惩罚
                            'reason': '加强对成交量萎缩的惩罚，过滤缺乏买盘支撑的信号'
                        },
                        {
                            'parameter': 'market_sentiment_weight',
                            'current_value': self.config.get('confidence_weights', {}).get('market_sentiment_weight', 0.15),
                            'suggested_value': 0.25,  # 提高市场情绪权重
                            'reason': '提高市场情绪权重，更重视有催化剂的信号'
                        }
                    ]
                })
        
        # 针对"接飞刀"型失败的建议
        falling_knife_count = failure_types.get('catching_falling_knife', {}).get('count', 0)
        if falling_knife_count > 0:
            falling_knife_percentage = (falling_knife_count / sum(ft.get('count', 0) for ft in failure_types.values())) * 100
            
            if falling_knife_percentage > 20:  # 如果接飞刀型失败超过20%
                recommendations.append({
                    'type': 'risk_control',
                    'priority': 'critical',
                    'target': 'catching_falling_knife',
                    'description': '针对接飞刀型失败的风险控制',
                    'actions': [
                        {
                            'parameter': 'rsi_oversold_threshold',
                            'current_value': self.config.get('confidence_weights', {}).get('rsi_oversold_threshold', 30),
                            'suggested_value': 25,  # 降低RSI超卖阈值
                            'reason': '要求更极端的超卖条件，避免在下跌趋势中过早入场'
                        },
                        {
                            'parameter': 'volume_panic_threshold',
                            'current_value': self.config.get('confidence_weights', {}).get('volume_panic_threshold', 1.5),
                            'suggested_value': 1.8,  # 提高恐慌性抛售阈值
                            'reason': '要求更显著的恐慌性抛售，确保是真正的底部信号'
                        },
                        {
                            'parameter': 'final_threshold',
                            'current_value': self.config.get('confidence_weights', {}).get('final_threshold', 0.5),
                            'suggested_value': 0.6,  # 提高最终置信度阈值
                            'reason': '提高入场门槛，只选择最高质量的信号'
                        }
                    ]
                })
        
        return recommendations
    
    def generate_optimization_strategy(self, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于失败分析生成具体的优化策略
        
        参数:
        failure_analysis: 失败分析结果
        
        返回:
        dict: 优化策略
        """
        recommendations = failure_analysis.get('recommendations', [])
        
        # 按优先级排序建议
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        # 生成参数调整方案
        parameter_adjustments = {}
        for rec in recommendations:
            for action in rec.get('actions', []):
                param_name = action['parameter']
                parameter_adjustments[param_name] = {
                    'current': action['current_value'],
                    'suggested': action['suggested_value'],
                    'reason': action['reason'],
                    'priority': rec['priority']
                }
        
        return {
            'strategy_name': 'failure_driven_optimization',
            'description': '基于失败案例分析的参数优化策略',
            'parameter_adjustments': parameter_adjustments,
            'implementation_order': [rec['type'] for rec in recommendations],
            'expected_improvements': self._estimate_improvements(failure_analysis)
        }
    
    def _estimate_improvements(self, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        估算优化后的预期改进
        
        参数:
        failure_analysis: 失败分析结果
        
        返回:
        dict: 预期改进估算
        """
        total_failures = failure_analysis.get('total_failures', 0)
        failure_types = failure_analysis.get('failure_types', {})
        
        # 估算可挽救的失败案例
        recoverable_failures = 0
        
        # 功亏一篑型失败 - 预计可挽救80%
        near_miss_total = (failure_types.get('near_miss_timing', {}).get('count', 0) + 
                          failure_types.get('near_miss_magnitude', {}).get('count', 0))
        recoverable_failures += near_miss_total * 0.8
        
        # 横盘不动型失败 - 预计可挽救40%
        sideways_count = failure_types.get('sideways_stagnation', {}).get('count', 0)
        recoverable_failures += sideways_count * 0.4
        
        # 接飞刀型失败 - 预计可避免60%
        falling_knife_count = failure_types.get('catching_falling_knife', {}).get('count', 0)
        recoverable_failures += falling_knife_count * 0.6
        
        # 计算预期成功率改进
        current_failure_rate = failure_analysis.get('failure_rate', 0)
        current_success_rate = 1 - current_failure_rate
        
        if total_failures > 0:
            improvement_rate = recoverable_failures / total_failures
            expected_success_rate = current_success_rate + (current_failure_rate * improvement_rate)
        else:
            expected_success_rate = current_success_rate
        
        return {
            'current_success_rate': current_success_rate,
            'expected_success_rate': min(expected_success_rate, 0.95),  # 最高不超过95%
            'potential_improvement': expected_success_rate - current_success_rate,
            'recoverable_failures': int(recoverable_failures),
            'confidence_level': 'medium'  # 基于历史经验的中等置信度
        }