#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略评估模块
负责策略性能评估、得分计算等功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any


class StrategyEvaluator:
    """策略评估器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略评估器
        
        参数:
        config: 配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate_on_test_set_only(self, strategy_module, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在测试集上评估策略
        
        参数:
        strategy_module: 策略模块
        test_data: 测试数据
        
        返回:
        dict: 评估结果
        """
        self.logger.info("🎯 开始测试集评估...")
        
        try:
            # 在测试集上运行回测
            test_backtest = strategy_module.backtest(test_data)
            test_evaluation = strategy_module.evaluate_strategy(test_backtest)
            
            test_score = test_evaluation['score']
            
            self.logger.info(f"✅ 测试集评估完成")
            self.logger.info(f"   - 测试集得分: {test_score:.4f}")
            self.logger.info(f"   - 识别点数: {test_evaluation.get('total_points', 0)}")
            self.logger.info(f"   - 成功率: {test_evaluation.get('success_rate', 0):.2%}")
            
            return {
                'success': True,
                'test_score': test_score,
                'test_evaluation': test_evaluation,
                'test_size': len(test_data),
                'success_rate': test_evaluation.get('success_rate', 0),
                'total_points': test_evaluation.get('total_points', 0),
                'avg_rise': test_evaluation.get('avg_rise', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 测试集评估失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def evaluate_params_with_fixed_labels(self, data: pd.DataFrame, fixed_labels: np.ndarray, 
                                         rise_threshold: float, max_days: int) -> float:
        """
        使用固定标签评估策略参数
        
        参数:
        data: 历史数据
        fixed_labels: 固定的标签（相对低点标识）
        rise_threshold: 涨幅阈值
        max_days: 最大天数
        
        返回:
        float: 策略得分
        """
        try:
            scores = []
            low_point_indices = np.where(fixed_labels == 1)[0]
            
            for idx in low_point_indices:
                if idx >= len(data) - max_days:
                    continue
                    
                current_price = data.iloc[idx]['close']
                max_rise = 0.0
                days_to_rise = 0
                
                # 计算未来max_days内的最大涨幅
                for j in range(1, max_days + 1):
                    if idx + j >= len(data):
                        break
                    future_price = data.iloc[idx + j]['close']
                    rise_rate = (future_price - current_price) / current_price
                    
                    if rise_rate > max_rise:
                        max_rise = rise_rate
                        
                    if rise_rate >= rise_threshold and days_to_rise == 0:
                        days_to_rise = j
                
                # 计算单个点的得分
                success = max_rise >= rise_threshold
                point_score = self.calculate_point_score(success, max_rise, days_to_rise, max_days)
                scores.append(point_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error("评估参数失败: %s", str(e))
            return 0.0

    def calculate_point_score(self, success: bool, max_rise: float, days_to_rise: int, max_days: int) -> float:
        """
        计算单个识别点的得分
        
        参数:
        success: 是否成功达到目标涨幅
        max_rise: 最大涨幅
        days_to_rise: 达到目标涨幅的天数
        max_days: 最大观察天数
        
        返回:
        float: 单个点得分
        """
        # 成功率权重：60%
        success_score = 1.0 if success else 0.0
        
        # 涨幅权重：30%
        rise_score = min(max_rise / 0.1, 1.0)  # 以10%为基准
        
        # 速度权重：10%
        if days_to_rise > 0:
            speed_score = min(max_days / days_to_rise, 1.0)
        else:
            speed_score = 0.0
        
        total_score = success_score * 0.6 + rise_score * 0.3 + speed_score * 0.1
        return total_score

    def calculate_strategy_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """
        计算策略综合指标
        
        参数:
        backtest_results: 回测结果
        
        返回:
        dict: 策略指标字典
        """
        try:
            # 获取相对低点
            low_points = backtest_results[backtest_results['is_low_point'] == True]
            
            if len(low_points) == 0:
                return {
                    'total_points': 0,
                    'success_rate': 0.0,
                    'avg_rise': 0.0,
                    'avg_days': 0.0,
                    'max_rise': 0.0,
                    'min_rise': 0.0,
                    'score': 0.0
                }
            
            # 计算基本指标
            total_points = len(low_points)
            successful_points = len(low_points[low_points['rise_achieved'] == True])
            success_rate = successful_points / total_points if total_points > 0 else 0.0
            
            # 计算涨幅指标
            rises = low_points['max_rise'].values
            avg_rise = np.mean(rises) if len(rises) > 0 else 0.0
            max_rise = np.max(rises) if len(rises) > 0 else 0.0
            min_rise = np.min(rises) if len(rises) > 0 else 0.0
            
            # 计算时间指标
            days = low_points['days_to_rise'].values
            avg_days = np.mean(days[days > 0]) if len(days[days > 0]) > 0 else 0.0
            
            # 计算综合得分
            scoring_config = self.config.get('ai', {}).get('scoring', {})
            success_weight = scoring_config.get('success_weight', 0.4)
            rise_weight = scoring_config.get('rise_weight', 0.3)
            speed_weight = scoring_config.get('speed_weight', 0.2)
            risk_weight = scoring_config.get('risk_weight', 0.1)
            
            rise_benchmark = scoring_config.get('rise_benchmark', 0.1)
            risk_benchmark = scoring_config.get('risk_benchmark', 0.2)
            
            # 归一化得分
            success_score = success_rate
            rise_score = min(avg_rise / rise_benchmark, 1.0) if rise_benchmark > 0 else 0.0
            speed_score = min(20 / avg_days, 1.0) if avg_days > 0 else 0.0
            risk_score = max(1.0 - (min_rise / (-risk_benchmark)), 0.0) if risk_benchmark > 0 else 1.0
            
            total_score = (
                success_score * success_weight +
                rise_score * rise_weight +
                speed_score * speed_weight +
                risk_score * risk_weight
            )
            
            return {
                'total_points': total_points,
                'success_rate': success_rate,
                'avg_rise': avg_rise,
                'avg_days': avg_days,
                'max_rise': max_rise,
                'min_rise': min_rise,
                'score': total_score,
                'success_score': success_score,
                'rise_score': rise_score,
                'speed_score': speed_score,
                'risk_score': risk_score
            }
            
        except Exception as e:
            self.logger.error(f"计算策略指标失败: {str(e)}")
            return {
                'total_points': 0,
                'success_rate': 0.0,
                'avg_rise': 0.0,
                'avg_days': 0.0,
                'max_rise': 0.0,
                'min_rise': 0.0,
                'score': 0.0
            }

    def compare_strategies(self, baseline_results: Dict[str, float], 
                          optimized_results: Dict[str, float]) -> Dict[str, Any]:
        """
        比较两个策略的性能
        
        参数:
        baseline_results: 基准策略结果
        optimized_results: 优化策略结果
        
        返回:
        dict: 比较结果
        """
        try:
            improvements = {}
            
            # 计算改进幅度
            for metric in ['score', 'success_rate', 'avg_rise']:
                baseline_value = baseline_results.get(metric, 0.0)
                optimized_value = optimized_results.get(metric, 0.0)
                
                if baseline_value > 0:
                    improvement = (optimized_value - baseline_value) / baseline_value * 100
                else:
                    improvement = 0.0
                
                improvements[f'{metric}_improvement'] = improvement
            
            # 判断是否有显著改进
            score_improvement = improvements.get('score_improvement', 0.0)
            is_significant = score_improvement > 5.0  # 5%以上改进认为是显著的
            
            return {
                'baseline': baseline_results,
                'optimized': optimized_results,
                'improvements': improvements,
                'is_significant': is_significant,
                'best_strategy': 'optimized' if score_improvement > 0 else 'baseline'
            }
            
        except Exception as e:
            self.logger.error(f"策略比较失败: {str(e)}")
            return {
                'baseline': baseline_results,
                'optimized': optimized_results,
                'improvements': {},
                'is_significant': False,
                'best_strategy': 'baseline'
            } 