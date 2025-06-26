#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
贝叶斯优化模块
负责使用贝叶斯优化进行智能参数搜索
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# 贝叶斯优化相关导入
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class BayesianOptimizer:
    """贝叶斯优化器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化贝叶斯优化器
        
        参数:
        config: 配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """
        检查贝叶斯优化是否可用
        
        返回:
        bool: 是否可用
        """
        return BAYESIAN_AVAILABLE

    def optimize_parameters(self, data: pd.DataFrame, objective_func, 
                          current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用贝叶斯优化进行参数搜索
        
        参数:
        data: 历史数据
        objective_func: 目标函数，接受参数字典并返回得分
        current_params: 当前参数，用于构建智能搜索范围
        
        返回:
        dict: 优化结果
        """
        self.logger.info("🔍 开始贝叶斯优化参数搜索...")
        
        if not BAYESIAN_AVAILABLE:
            self.logger.error("❌ scikit-optimize未安装，无法使用贝叶斯优化")
            return {'success': False, 'error': 'scikit-optimize未安装'}
        
        try:
            # 获取贝叶斯优化配置
            ai_config = self.config.get('ai', {})
            bayesian_config = ai_config.get('bayesian_optimization', {})
            
            if not bayesian_config.get('enabled', True):
                self.logger.info("贝叶斯优化已禁用，跳过")
                return {'success': False, 'error': '贝叶斯优化已禁用'}
            
            # 配置参数
            n_calls = bayesian_config.get('n_calls', 100)
            n_initial_points = bayesian_config.get('n_initial_points', 20)
            acq_func = bayesian_config.get('acq_func', 'EI')
            xi = bayesian_config.get('xi', 0.01)
            kappa = bayesian_config.get('kappa', 1.96)
            n_jobs = bayesian_config.get('n_jobs', 1)
            random_state = bayesian_config.get('random_state', 42)
            
            self.logger.info(f"贝叶斯优化配置:")
            self.logger.info(f"  - 调用次数: {n_calls}")
            self.logger.info(f"  - 初始点数: {n_initial_points}")
            self.logger.info(f"  - 采集函数: {acq_func}")
            
            # 定义基于当前参数的智能搜索空间
            dimensions, param_names = self._build_adaptive_parameter_space(current_params)
            
            if len(dimensions) == 0:
                self.logger.error("❌ 未定义优化参数空间")
                return {'success': False, 'error': '未定义优化参数空间'}
            
            self.logger.info(f"参数空间维度: {len(dimensions)}")
            for i, dim in enumerate(dimensions):
                self.logger.info(f"  - {param_names[i]}: [{dim.low}, {dim.high}]")
            
            # 记录评估历史
            evaluation_history = []
            
            @use_named_args(dimensions)
            def objective(**params):
                """目标函数：最小化负得分（因为gp_minimize是最小化）"""
                try:
                    # 调用外部目标函数
                    score = objective_func(params)
                    
                    # 记录评估历史
                    evaluation_history.append({
                        'params': params.copy(),
                        'score': score,
                        'iteration': len(evaluation_history) + 1
                    })
                    
                    if len(evaluation_history) % 10 == 0:
                        self.logger.info(f"贝叶斯优化进度: {len(evaluation_history)}/{n_calls}, 当前得分: {score:.4f}")
                    
                    # 返回负得分（因为要最小化）
                    return -score
                    
                except Exception as e:
                    self.logger.error(f"目标函数评估失败: {str(e)}")
                    return 1.0  # 返回最差得分
            
            # 运行贝叶斯优化
            self.logger.info("🚀 开始贝叶斯优化...")
            
            # 根据采集函数设置参数
            gp_kwargs = {
                'func': objective,
                'dimensions': dimensions,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'acq_func': acq_func,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'verbose': False
            }
            
            # 根据采集函数类型添加特定参数
            if acq_func == 'EI':
                gp_kwargs['xi'] = xi
            elif acq_func == 'LCB':
                gp_kwargs['kappa'] = kappa
            
            result = gp_minimize(**gp_kwargs)
            
            # 提取最优参数
            best_params = {}
            for i, param_name in enumerate(param_names):
                best_params[param_name] = result.x[i]
            
            best_score = -result.fun  # 转换回正得分
            
            self.logger.info("✅ 贝叶斯优化完成")
            self.logger.info(f"   - 最优得分: {best_score:.4f}")
            self.logger.info(f"   - 总评估次数: {len(evaluation_history)}")
            self.logger.info(f"   - 最优参数:")
            for param, value in best_params.items():
                self.logger.info(f"     - {param}: {value:.4f}")
            
            # 分析收敛情况
            scores = [eval_record['score'] for eval_record in evaluation_history]
            best_scores = np.maximum.accumulate(scores)
            improvement_rate = (best_scores[-1] - best_scores[n_initial_points]) / max(best_scores[n_initial_points], 0.001)
            
            self.logger.info(f"   - 改进率: {improvement_rate:.2%}")
            self.logger.info(f"   - 最终收敛得分: {best_scores[-1]:.4f}")
            
            return {
                'success': True,
                'best_params': best_params,
                'best_score': best_score,
                'n_evaluations': len(evaluation_history),
                'improvement_rate': improvement_rate,
                'evaluation_history': evaluation_history,
                'convergence_scores': best_scores.tolist(),
                'optimization_result': result
            }
            
        except Exception as e:
            self.logger.error(f"❌ 贝叶斯优化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _build_adaptive_parameter_space(self, current_params: Dict[str, Any]) -> tuple:
        """
        基于当前参数构建自适应参数空间
        
        参数:
        current_params: 当前最优参数
        
        返回:
        tuple: (dimensions列表, 参数名列表)
        """
        dimensions = []
        param_names = []
        
        self.logger.info("🎯 构建基于当前参数的自适应搜索空间...")
        
        # 从配置中读取基础参数范围
        optimization_ranges = self.config.get('ai', {}).get('optimization_ranges', {})
        
        # 智能搜索半径
        search_factor = self.config.get('ai', {}).get('bayesian_optimization', {}).get('search_factor', 0.3)
        
        for param_name, param_range in optimization_ranges.items():
            base_min = param_range.get('min', 0.0)
            base_max = param_range.get('max', 1.0)
            current_value = current_params.get(param_name, (base_min + base_max) / 2)
            
            # 基于当前值动态调整搜索范围
            range_width = base_max - base_min
            adaptive_radius = range_width * search_factor
            
            adaptive_min = max(base_min, current_value - adaptive_radius)
            adaptive_max = min(base_max, current_value + adaptive_radius)
            
            dimensions.append(Real(adaptive_min, adaptive_max, name=param_name))
            param_names.append(param_name)
            
            self.logger.info(f"   - {param_name}: 当前值 {current_value:.3f}, 搜索范围 [{adaptive_min:.3f}, {adaptive_max:.3f}]")
        
        # RSI相关参数的自适应范围
        base_rsi_oversold = current_params.get('rsi_oversold_threshold', 30)
        base_rsi_low = current_params.get('rsi_low_threshold', 40)
        base_final_threshold = current_params.get('final_threshold', 0.5)
        
        # RSI参数搜索半径
        rsi_radius = 4  # RSI参数的搜索半径
        threshold_radius = 0.15  # final_threshold的搜索半径
        
        # 自适应RSI oversold范围
        rsi_oversold_min = max(25, base_rsi_oversold - rsi_radius)
        rsi_oversold_max = min(35, base_rsi_oversold + rsi_radius)
        
        # 自适应RSI low范围
        rsi_low_min = max(35, base_rsi_low - rsi_radius)
        rsi_low_max = min(45, base_rsi_low + rsi_radius)
        
        # 自适应final_threshold范围
        final_threshold_min = max(0.3, base_final_threshold - threshold_radius)
        final_threshold_max = min(0.7, base_final_threshold + threshold_radius)
        
        dimensions.extend([
            Integer(rsi_oversold_min, rsi_oversold_max, name='rsi_oversold_threshold'),
            Integer(rsi_low_min, rsi_low_max, name='rsi_low_threshold'),
            Real(final_threshold_min, final_threshold_max, name='final_threshold')
        ])
        param_names.extend(['rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold'])
        
        self.logger.info(f"   - rsi_oversold_threshold: 当前值 {base_rsi_oversold}, 搜索范围 [{rsi_oversold_min}, {rsi_oversold_max}]")
        self.logger.info(f"   - rsi_low_threshold: 当前值 {base_rsi_low}, 搜索范围 [{rsi_low_min}, {rsi_low_max}]")
        self.logger.info(f"   - final_threshold: 当前值 {base_final_threshold:.3f}, 搜索范围 [{final_threshold_min:.3f}, {final_threshold_max:.3f}]")
        
        return dimensions, param_names

    def _build_parameter_space(self, param_ranges: Dict[str, Dict[str, Any]]) -> tuple:
        """
        构建传统的固定参数空间（保持向后兼容）
        
        参数:
        param_ranges: 参数范围配置
        
        返回:
        tuple: (dimensions列表, 参数名列表)
        """
        dimensions = []
        param_names = []
        
        # 从配置中读取参数范围
        optimization_ranges = self.config.get('ai', {}).get('optimization_ranges', {})
        
        for param_name, param_range in optimization_ranges.items():
            min_val = param_range.get('min', 0.0)
            max_val = param_range.get('max', 1.0)
            
            dimensions.append(Real(min_val, max_val, name=param_name))
            param_names.append(param_name)
        
        # 添加RSI相关参数
        dimensions.extend([
            Integer(25, 35, name='rsi_oversold_threshold'),
            Integer(35, 45, name='rsi_low_threshold'),
            Real(0.3, 0.7, name='final_threshold')
        ])
        param_names.extend(['rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold'])
        
        return dimensions, param_names 