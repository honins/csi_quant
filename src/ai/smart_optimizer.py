#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能参数优化器
基于失败案例分析实现针对性的参数优化策略
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import copy

from .failure_analysis import FailureAnalyzer

class SmartOptimizer:
    """智能参数优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化智能优化器
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.failure_analyzer = FailureAnalyzer(config)
        
        self.logger.info("智能参数优化器初始化完成")
    
    def optimize_based_on_failures(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        基于失败案例分析进行参数优化
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        self.logger.info("开始基于失败案例的智能优化")
        
        # 步骤1: 获取当前参数的回测结果
        current_backtest = strategy_module.backtest(data)
        current_evaluation = strategy_module.evaluate_strategy(current_backtest)
        current_success_rate = current_evaluation.get('success_rate', 0)
        
        print(f"    📊 当前成功率: {current_success_rate:.2%}")
        self.logger.info(f"当前成功率: {current_success_rate:.2%}")
        
        # 步骤2: 分析失败案例
        print(f"    🔍 分析失败案例...")
        failure_analysis = self.failure_analyzer.analyze_failures(current_backtest, data)
        
        # 步骤3: 生成优化策略
        optimization_strategies = self._generate_optimization_strategies(failure_analysis)
        
        # 步骤4: 依次测试优化策略
        best_params = strategy_module.get_current_params()
        best_score = self._calculate_score(current_evaluation)
        best_strategy = 'baseline'
        
        optimization_results = []
        
        for strategy_name, strategy_config in optimization_strategies.items():
            print(f"    🧪 测试优化策略: {strategy_name}")
            self.logger.info(f"测试优化策略: {strategy_name}")
            
            # 应用策略参数
            test_params = self._apply_strategy_params(best_params, strategy_config)
            strategy_module.update_params(test_params)
            
            # 回测评估
            test_backtest = strategy_module.backtest(data)
            test_evaluation = strategy_module.evaluate_strategy(test_backtest)
            test_score = self._calculate_score(test_evaluation)
            
            result = {
                'strategy_name': strategy_name,
                'params': test_params,
                'evaluation': test_evaluation,
                'score': test_score,
                'improvement': test_score - best_score
            }
            
            optimization_results.append(result)
            
            print(f"       成功率: {test_evaluation.get('success_rate', 0):.2%} "
                  f"(改进: {(test_evaluation.get('success_rate', 0) - current_success_rate):.2%})")
            print(f"       综合得分: {test_score:.4f} (改进: {test_score - best_score:.4f})")
            
            # 更新最佳结果
            if test_score > best_score:
                best_params = test_params.copy()
                best_score = test_score
                best_strategy = strategy_name
                print(f"       ✅ 发现更优策略!")
        
        # 恢复最佳参数
        strategy_module.update_params(best_params)
        
        return {
            'success': True,
            'optimization_method': 'failure_driven',
            'best_strategy': best_strategy,
            'best_params': best_params,
            'best_score': best_score,
            'improvement': best_score - self._calculate_score(current_evaluation),
            'failure_analysis': failure_analysis,
            'optimization_results': optimization_results,
            'recommendations': self._generate_final_recommendations(optimization_results)
        }
    
    def _generate_optimization_strategies(self, failure_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        基于失败分析生成三种优化策略
        
        参数:
        failure_analysis: 失败分析结果
        
        返回:
        dict: 优化策略字典
        """
        failure_types = failure_analysis.get('failure_types', {})
        total_failures = failure_analysis.get('total_failures', 0)
        
        strategies = {}
        
        # 计算各类型失败的占比
        falling_knife_pct = failure_types.get('catching_falling_knife', {}).get('percentage', 0)
        sideways_pct = failure_types.get('sideways_stagnation', {}).get('percentage', 0)
        near_miss_pct = (failure_types.get('near_miss_timing', {}).get('percentage', 0) + 
                        failure_types.get('near_miss_magnitude', {}).get('percentage', 0))
        
        print(f"    📈 失败类型分析:")
        print(f"       接飞刀型: {falling_knife_pct:.1f}%")
        print(f"       横盘不动型: {sideways_pct:.1f}%")
        print(f"       功亏一篑型: {near_miss_pct:.1f}%")
        
        # 策略一：主攻"接飞刀"型失败（提高成功率，宁可错过不可做错）
        if falling_knife_pct > 15:  # 如果接飞刀型失败超过15%
            strategies['conservative_filter'] = {
                'description': '保守过滤策略 - 主攻接飞刀型失败',
                'target': 'catching_falling_knife',
                'parameter_adjustments': {
                    # 提高核心阈值
                    'rsi_oversold_threshold': {
                        'adjustment': 'decrease',
                        'factor': 0.9,  # 从30降到27
                        'reason': '要求更极端的超卖条件'
                    },
                    'volume_panic_threshold': {
                        'adjustment': 'increase',
                        'factor': 1.2,  # 从1.5提高到1.8
                        'reason': '要求更显著的恐慌放量'
                    },
                    # 加强惩罚和验证
                    'volume_shrink_penalty': {
                        'adjustment': 'decrease',
                        'factor': 0.8,  # 加大惩罚效果
                        'reason': '加强对缺乏成交量支撑的惩罚'
                    },
                    'ma_all_below': {
                        'adjustment': 'increase',
                        'factor': 1.15,  # 提高权重
                        'reason': '更重视技术破位信号'
                    },
                    # 收紧"大门"
                    'final_threshold': {
                        'adjustment': 'increase',
                        'factor': 1.2,  # 从0.5提高到0.6
                        'reason': '提高入场门槛，只选择最高质量信号'
                    }
                }
            }
        
        # 策略二：主攻"横盘不动"型失败（提高资金效率）
        if sideways_pct > 20:  # 如果横盘型失败超过20%
            strategies['momentum_confirmation'] = {
                'description': '动能确认策略 - 主攻横盘不动型失败',
                'target': 'sideways_stagnation',
                'parameter_adjustments': {
                    # 加强成交量要求
                    'volume_surge_bonus': {
                        'adjustment': 'increase',
                        'factor': 1.5,  # 提高温和放量奖励
                        'reason': '更重视资金关注的早期信号'
                    },
                    'volume_shrink_penalty': {
                        'adjustment': 'decrease',
                        'factor': 0.6,  # 进一步加强萎缩惩罚
                        'reason': '严厉惩罚缺乏买盘支撑的信号'
                    },
                    # 提高市场情绪权重
                    'market_sentiment_weight': {
                        'adjustment': 'increase',
                        'factor': 1.4,  # 提高市场情绪权重
                        'reason': '更重视有催化剂的信号'
                    },
                    # 加强MACD动能确认
                    'macd_negative': {
                        'adjustment': 'increase',
                        'factor': 1.3,  # 提高MACD权重
                        'reason': '更重视动量指标确认'
                    },
                    # 降低RSI要求，但要求配合其他指标
                    'rsi_low_threshold': {
                        'adjustment': 'increase',
                        'factor': 1.1,  # 稍微放宽RSI要求
                        'reason': '在有其他确认信号时放宽RSI要求'
                    }
                }
            }
        
        # 策略三：拥抱波动性（提高整体收益，而非单纯成功率）
        if near_miss_pct > 25:  # 如果功亏一篑型失败超过25%
            strategies['volatility_embrace'] = {
                'description': '拥抱波动策略 - 重奖极端事件',
                'target': 'near_miss_failures',
                'parameter_adjustments': {
                    # 重奖极端事件
                    'volume_panic_bonus': {
                        'adjustment': 'increase',
                        'factor': 1.6,  # 大幅提高恐慌性抛售奖励
                        'reason': '极端恐慌往往伴随最强反弹'
                    },
                    'recent_decline': {
                        'adjustment': 'increase',
                        'factor': 1.4,  # 大幅提高近期下跌权重
                        'reason': '非理性暴跌往往有最快反弹'
                    },
                    # 利用市场情绪
                    'market_sentiment_weight': {
                        'adjustment': 'increase',
                        'factor': 1.5,  # 大幅提高市场情绪权重
                        'reason': '恐慌性抛售是反向投资的精髓'
                    },
                    # 适度放宽基础条件
                    'rsi_oversold_threshold': {
                        'adjustment': 'increase',
                        'factor': 1.05,  # 稍微放宽RSI超卖条件
                        'reason': '在极端事件中适度放宽条件'
                    },
                    # 降低最终阈值，捕捉更多机会
                    'final_threshold': {
                        'adjustment': 'decrease',
                        'factor': 0.9,  # 从0.5降到0.45
                        'reason': '降低门槛，捕捉更多极端机会'
                    }
                }
            }
        
        # 如果没有明显的失败模式，使用平衡策略
        if not strategies:
            strategies['balanced_improvement'] = {
                'description': '平衡改进策略 - 全面优化',
                'target': 'overall_improvement',
                'parameter_adjustments': {
                    'rsi_oversold_threshold': {
                        'adjustment': 'decrease',
                        'factor': 0.95,  # 稍微降低RSI阈值
                        'reason': '适度提高信号质量'
                    },
                    'volume_panic_threshold': {
                        'adjustment': 'increase',
                        'factor': 1.1,  # 稍微提高恐慌阈值
                        'reason': '提高恐慌信号质量'
                    },
                    'market_sentiment_weight': {
                        'adjustment': 'increase',
                        'factor': 1.2,  # 提高市场情绪权重
                        'reason': '更重视市场情绪信号'
                    }
                }
            }
        
        return strategies
    
    def _apply_strategy_params(self, base_params: Dict[str, Any], strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用策略参数调整
        
        参数:
        base_params: 基础参数
        strategy_config: 策略配置
        
        返回:
        dict: 调整后的参数
        """
        adjusted_params = copy.deepcopy(base_params)
        adjustments = strategy_config.get('parameter_adjustments', {})
        
        for param_name, adjustment_config in adjustments.items():
            if param_name in adjusted_params:
                current_value = adjusted_params[param_name]
                adjustment_type = adjustment_config['adjustment']
                factor = adjustment_config['factor']
                
                if adjustment_type == 'increase':
                    new_value = current_value * factor
                elif adjustment_type == 'decrease':
                    new_value = current_value * factor
                else:
                    new_value = current_value
                
                # 应用合理的边界限制
                new_value = self._apply_parameter_bounds(param_name, new_value)
                adjusted_params[param_name] = new_value
                
                self.logger.info(f"参数调整: {param_name} {current_value:.4f} -> {new_value:.4f}")
        
        return adjusted_params
    
    def _apply_parameter_bounds(self, param_name: str, value: float) -> float:
        """
        应用参数边界限制
        
        参数:
        param_name: 参数名称
        value: 参数值
        
        返回:
        float: 限制后的参数值
        """
        # 从配置中获取参数范围
        optimization_ranges = self.config.get('optimization_ranges', {})
        
        if param_name in optimization_ranges:
            param_range = optimization_ranges[param_name]
            min_val = param_range.get('min', 0)
            max_val = param_range.get('max', 1)
            return max(min_val, min(max_val, value))
        
        # 默认边界
        bounds = {
            'rsi_oversold_threshold': (20, 35),
            'rsi_low_threshold': (35, 55),
            'volume_panic_threshold': (1.2, 2.0),
            'volume_shrink_penalty': (0.3, 0.9),
            'final_threshold': (0.3, 0.8),
            'ma_all_below': (0.1, 0.6),
            'volume_surge_bonus': (0.01, 0.3),
            'market_sentiment_weight': (0.05, 0.4),
            'macd_negative': (0.02, 0.3),
            'volume_panic_bonus': (0.02, 0.4),
            'recent_decline': (0.05, 0.5)
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            return max(min_val, min(max_val, value))
        
        return value
    
    def _calculate_score(self, evaluation: Dict[str, Any]) -> float:
        """
        计算综合评分
        
        参数:
        evaluation: 评估结果
        
        返回:
        float: 综合评分
        """
        success_rate = evaluation.get('success_rate', 0)
        avg_rise = evaluation.get('avg_rise', 0)
        avg_days = evaluation.get('avg_days', 0)
        
        # 使用与AI优化器相同的评分方法
        scoring_config = self.config.get('strategy_scoring', {})
        
        success_weight = scoring_config.get('success_weight', 0.5)
        rise_weight = scoring_config.get('rise_weight', 0.3)
        days_weight = scoring_config.get('days_weight', 0.2)
        
        rise_benchmark = scoring_config.get('rise_benchmark', 0.1)
        days_benchmark = scoring_config.get('days_benchmark', 10.0)
        
        success_score = success_rate * success_weight
        rise_score = min(avg_rise / rise_benchmark, 1.0) * rise_weight if avg_rise > 0 else 0
        days_score = min(days_benchmark / avg_days, 1.0) * days_weight if avg_days > 0 else 0
        
        return success_score + rise_score + days_score
    
    def _generate_final_recommendations(self, optimization_results: List[Dict[str, Any]]) -> List[str]:
        """
        生成最终优化建议
        
        参数:
        optimization_results: 优化结果列表
        
        返回:
        list: 建议列表
        """
        recommendations = []
        
        # 找到最佳策略
        best_result = max(optimization_results, key=lambda x: x['score']) if optimization_results else None
        
        if best_result:
            improvement = best_result['improvement']
            strategy_name = best_result['strategy_name']
            
            if improvement > 0.05:  # 显著改进
                recommendations.append(f"强烈建议采用'{strategy_name}'策略，预期综合得分提升{improvement:.4f}")
            elif improvement > 0.01:  # 轻微改进
                recommendations.append(f"建议考虑采用'{strategy_name}'策略，有轻微改进")
            else:
                recommendations.append("当前参数已相对优化，建议保持现状或进行微调")
        
        # 基于结果给出具体建议
        for result in optimization_results:
            if result['improvement'] > 0:
                strategy_desc = result.get('strategy_name', '未知策略')
                success_rate = result['evaluation'].get('success_rate', 0)
                recommendations.append(f"{strategy_desc}: 成功率可达{success_rate:.2%}")
        
        return recommendations
    
    def run_comprehensive_optimization(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行综合优化流程
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 综合优化结果
        """
        self.logger.info("开始综合优化流程")
        
        print("\n" + "="*60)
        print("🚀 智能参数优化器 - 基于失败案例分析")
        print("="*60)
        
        # 阶段1: 基于失败案例的优化
        print("\n📊 阶段1: 失败案例分析与针对性优化")
        failure_driven_result = self.optimize_based_on_failures(strategy_module, data)
        
        # 阶段2: 如果改进不明显，尝试传统贝叶斯优化
        improvement = failure_driven_result.get('improvement', 0)
        
        if improvement < 0.02:  # 如果改进小于2%
            print("\n🔬 阶段2: 传统贝叶斯优化补充")
            print("    💡 失败驱动优化改进有限，启动贝叶斯优化")
            
            # 这里可以调用原有的贝叶斯优化
            # bayesian_result = self.run_bayesian_optimization(...)
            # 暂时跳过，专注于失败驱动优化
            
        print("\n" + "="*60)
        print("✅ 智能优化完成")
        print("="*60)
        
        return failure_driven_result