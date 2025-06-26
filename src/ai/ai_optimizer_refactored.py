#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重构后的AI优化器模块
主控制器，集成数据验证、参数优化、模型管理和策略评估等功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

# 导入各个子模块
from .data_validator import DataValidator
from .bayesian_optimizer import BayesianOptimizer
from .model_manager import ModelManager
from .strategy_evaluator import StrategyEvaluator


class AIOptimizerRefactored:
    """重构后的AI优化器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化AI优化器
        
        参数:
        config: 配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个子模块
        self.data_validator = DataValidator(config)
        self.bayesian_optimizer = BayesianOptimizer(config)
        self.model_manager = ModelManager(config)
        self.strategy_evaluator = StrategyEvaluator(config)
        
        self.logger.info("重构后的AI优化器初始化完成")

    # ==================== 数据验证和分割 ====================
    
    def strict_data_split(self, data: pd.DataFrame, preserve_test_set: bool = True) -> Dict[str, pd.DataFrame]:
        """
        严格的数据分割，防止过拟合
        
        参数:
        data: 输入数据
        preserve_test_set: 是否保护测试集
        
        返回:
        dict: 包含训练集、验证集、测试集的字典
        """
        return self.data_validator.strict_data_split(data, preserve_test_set)

    def walk_forward_validation(self, data: pd.DataFrame, strategy_module, 
                              window_size: int = 252, step_size: int = 63) -> Dict[str, Any]:
        """
        走前验证，模拟真实交易环境
        
        参数:
        data: 历史数据
        strategy_module: 策略模块
        window_size: 训练窗口大小（交易日数）
        step_size: 步进大小（交易日数）
        
        返回:
        dict: 验证结果
        """
        return self.data_validator.walk_forward_validation(data, strategy_module, window_size, step_size)

    # ==================== 参数优化 ====================
    
    def optimize_strategy_parameters(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        智能参数优化：根据配置选择最佳优化策略
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("🎯 开始智能参数优化...")
        
        try:
            # 获取优化配置
            ai_config = self.config.get('ai', {})
            bayesian_config = ai_config.get('bayesian_optimization', {})
            advanced_config = ai_config.get('advanced_optimization', {})
            
            # 检查是否启用贝叶斯优化
            if bayesian_config.get('enabled', False) and self.bayesian_optimizer.is_available():
                self.logger.info("🔍 使用贝叶斯优化策略")
                
                # 使用严格数据分割进行贝叶斯优化
                if advanced_config.get('use_hierarchical', True):
                    data_splits = self.strict_data_split(data, preserve_test_set=True)
                    train_data = data_splits['train']
                    
                    # 在训练集上进行贝叶斯优化
                    bayesian_result = self.bayesian_optimize_parameters(strategy_module, train_data)
                    
                    if bayesian_result['success']:
                        return bayesian_result['best_params']
                    else:
                        self.logger.warning(f"贝叶斯优化失败: {bayesian_result.get('error')}")
                        self.logger.info("回退到传统优化方法")
                        
                else:
                    # 在全部数据上进行贝叶斯优化
                    bayesian_result = self.bayesian_optimize_parameters(strategy_module, data)
                    
                    if bayesian_result['success']:
                        return bayesian_result['best_params']
                    else:
                        self.logger.warning(f"贝叶斯优化失败: {bayesian_result.get('error')}")
                        self.logger.info("回退到传统优化方法")
            
            # 回退到传统优化方法
            self.logger.info("🔧 使用传统参数优化策略")
            return self._traditional_parameter_optimization(strategy_module, data)
            
        except Exception as e:
            self.logger.error(f"❌ 智能参数优化失败: {str(e)}")
            # 返回默认参数
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                'max_days': self.config.get('strategy', {}).get('max_days', 20),
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5
            }

    def bayesian_optimize_parameters(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用贝叶斯优化进行参数搜索
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        self.logger.info("🔍 开始贝叶斯优化参数搜索...")
        
        try:
            # 固定核心参数
            fixed_params = {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                'max_days': self.config.get('strategy', {}).get('max_days', 20)
            }
            
            # 获取基准策略结果用于标签固定
            baseline_backtest = strategy_module.backtest(data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            
            # 定义目标函数
            def objective_func(params):
                """目标函数：评估参数组合的得分"""
                # 合并固定参数和优化参数
                full_params = fixed_params.copy()
                full_params.update(params)
                
                # 评估参数
                score = self.strategy_evaluator.evaluate_params_with_fixed_labels(
                    data, fixed_labels, 
                    full_params['rise_threshold'], 
                    full_params['max_days']
                )
                
                return score
            
            # 获取当前策略参数
            current_params = strategy_module.get_current_params()
            
            # 调用贝叶斯优化器
            optimization_result = self.bayesian_optimizer.optimize_parameters(
                data, objective_func, current_params
            )
            
            if optimization_result['success']:
                # 合并固定参数和优化后的参数
                best_params = fixed_params.copy()
                best_params.update(optimization_result['best_params'])
                optimization_result['best_params'] = best_params
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ 贝叶斯优化失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _traditional_parameter_optimization(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        传统参数优化方法（网格搜索/随机搜索）
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("🔧 执行传统参数优化...")
        
        try:
            # 固定核心参数
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            # 获取基准策略识别结果
            baseline_backtest = strategy_module.backtest(data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            
            # 参数搜索范围
            param_ranges = {
                'rsi_oversold_threshold': np.arange(25, 36, 1),
                'rsi_low_threshold': np.arange(35, 46, 1),
                'final_threshold': np.arange(0.3, 0.71, 0.05)
            }
            
            best_score = -1
            best_params = None
            
            # 获取优化配置
            ai_config = self.config.get('ai', {})
            optimization_config = ai_config.get('optimization', {})
            max_iterations = optimization_config.get('global_iterations', 200)
            
            for i in range(max_iterations):
                params = {
                    'rise_threshold': fixed_rise_threshold,
                    'max_days': fixed_max_days,
                    'rsi_oversold_threshold': int(np.random.choice(param_ranges['rsi_oversold_threshold'])),
                    'rsi_low_threshold': int(np.random.choice(param_ranges['rsi_low_threshold'])),
                    'final_threshold': np.random.choice(param_ranges['final_threshold'])
                }
                
                score = self.strategy_evaluator.evaluate_params_with_fixed_labels(
                    data, fixed_labels, 
                    params['rise_threshold'], params['max_days']
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"传统优化进度: {i + 1}/{max_iterations}, 当前最佳得分: {best_score:.4f}")
            
            self.logger.info(f"✅ 传统优化完成，最佳得分: {best_score:.4f}")
            
            return best_params if best_params else {
                'rise_threshold': fixed_rise_threshold,
                'max_days': fixed_max_days,
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5
            }
            
        except Exception as e:
            self.logger.error(f"❌ 传统参数优化失败: {str(e)}")
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                'max_days': self.config.get('strategy', {}).get('max_days', 20),
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5
            }

    # ==================== 策略评估 ====================
    
    def evaluate_on_test_set_only(self, strategy_module, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在测试集上评估策略
        
        参数:
        strategy_module: 策略模块
        test_data: 测试数据
        
        返回:
        dict: 评估结果
        """
        return self.strategy_evaluator.evaluate_on_test_set_only(strategy_module, test_data)

    # ==================== 机器学习模型 ====================
    
    def train_model(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        训练机器学习模型
        
        参数:
        data: 训练数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
        """
        return self.model_manager.train_model(data, strategy_module)

    def validate_model(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        验证模型性能
        
        参数:
        data: 验证数据
        strategy_module: 策略模块
        
        返回:
        dict: 验证结果
        """
        return self.model_manager.validate_model(data, strategy_module)

    def predict_low_point(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测相对低点
        
        参数:
        data: 市场数据
        
        返回:
        dict: 预测结果
        """
        return self.model_manager.predict_low_point(data)

    # ==================== 兼容性方法 ====================
    
    def optimize_strategy_parameters_on_train_only(self, strategy_module, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在训练集上优化策略参数（兼容性方法）
        
        参数:
        strategy_module: 策略模块
        train_data: 训练数据
        
        返回:
        dict: 优化后的参数
        """
        return self._traditional_parameter_optimization(strategy_module, train_data)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取模型的特征重要性
        
        返回:
        dict: 特征名称和重要性的字典，按重要性降序排列
        """
        return self.model_manager.get_feature_importance() 