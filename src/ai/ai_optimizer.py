#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化模块
使用机器学习方法优化策略参数和预测相对低点
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pickle
import json
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.strategy_module import StrategyModule

# 机器学习相关
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class AIOptimizer:
    """AI优化器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化AI优化器
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger('AIOptimizer')
        self.config = config
        
        # AI配置
        ai_config = config.get('ai', {})
        self.model_type = ai_config.get('model_type', 'machine_learning')
        self.optimization_interval = ai_config.get('optimization_interval', 30)
        
        # 创建模型保存目录
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # 初始化模型
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # 参数历史记录
        self.parameter_history_file = os.path.join(self.models_dir, 'parameter_history.json')
        self.best_parameters_file = os.path.join(self.models_dir, 'best_parameters.json')
        
        # 新增：严格数据分割配置
        validation_config = ai_config.get('validation', {})
        self.train_ratio = validation_config.get('train_ratio', 0.6)
        self.validation_ratio = validation_config.get('validation_ratio', 0.2)
        self.test_ratio = validation_config.get('test_ratio', 0.2)
        
        # 确保比例总和为1
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            self.logger.warning(f"数据分割比例总和 {total_ratio:.3f} 不等于1.0，自动调整")
            self.train_ratio = self.train_ratio / total_ratio
            self.validation_ratio = self.validation_ratio / total_ratio
            self.test_ratio = self.test_ratio / total_ratio
        
        # 数据泄露保护
        self._test_set_locked = False
        self._test_set_indices = None
        
        self.logger.info("AI优化器初始化完成，模型类型: %s", self.model_type)
        self.logger.info(f"数据分割比例 - 训练: {self.train_ratio:.1%}, 验证: {self.validation_ratio:.1%}, 测试: {self.test_ratio:.1%}")
        
    def strict_data_split(self, data: pd.DataFrame, preserve_test_set: bool = True) -> Dict[str, pd.DataFrame]:
        """
        严格的时间序列数据分割，防止数据泄露
        
        参数:
        data: 原始数据
        preserve_test_set: 是否保护测试集（一旦分割，测试集永远不参与优化）
        
        返回:
        dict: 包含 'train', 'validation', 'test' 键的数据字典
        """
        self.logger.info("🔒 开始严格数据分割...")
        
        # 按时间序列分割数据
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.validation_ratio))
        
        # 分割数据
        train_data = data.iloc[:train_end].copy()
        validation_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        # 保护测试集
        if preserve_test_set:
            if self._test_set_locked and self._test_set_indices is not None:
                # 检查测试集是否被篡改
                current_test_indices = test_data.index.tolist()
                if current_test_indices != self._test_set_indices:
                    self.logger.error("❌ 检测到测试集数据泄露风险！")
                    raise ValueError("测试集数据已被篡改，存在数据泄露风险")
                self.logger.info("🔒 测试集完整性验证通过")
            else:
                # 首次锁定测试集
                self._test_set_indices = test_data.index.tolist()
                self._test_set_locked = True
                self.logger.info("🔒 测试集已锁定，防止数据泄露")
        
        self.logger.info(f"✅ 数据分割完成:")
        self.logger.info(f"   - 训练集: {len(train_data)} 条 ({len(train_data)/n:.1%})")
        self.logger.info(f"   - 验证集: {len(validation_data)} 条 ({len(validation_data)/n:.1%})")
        self.logger.info(f"   - 测试集: {len(test_data)} 条 ({len(test_data)/n:.1%})")
        self.logger.info(f"   - 时间范围:")
        self.logger.info(f"     训练: {train_data.iloc[0]['date']} ~ {train_data.iloc[-1]['date']}")
        self.logger.info(f"     验证: {validation_data.iloc[0]['date']} ~ {validation_data.iloc[-1]['date']}")
        self.logger.info(f"     测试: {test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}")
        
        return {
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        }
    
    def walk_forward_validation(self, data: pd.DataFrame, strategy_module, 
                              window_size: int = 252, step_size: int = 63) -> Dict[str, Any]:
        """
        走前验证：模拟真实交易环境的严格验证方法
        
        参数:
        data: 历史数据
        strategy_module: 策略模块实例
        window_size: 训练窗口大小（交易日）
        step_size: 步进大小（交易日）
        
        返回:
        dict: 验证结果
        """
        self.logger.info("🚶 开始走前验证...")
        self.logger.info(f"   - 训练窗口: {window_size} 天")
        self.logger.info(f"   - 步进大小: {step_size} 天")
        
        scores = []
        fold_results = []
        start_time = time.time()
        
        # 计算总的验证折数
        total_folds = max(0, (len(data) - window_size) // step_size)
        if total_folds == 0:
            self.logger.error("❌ 数据不足以进行走前验证")
            return {'success': False, 'error': '数据不足'}
        
        self.logger.info(f"📊 总验证折数: {total_folds}")
        
        for fold in range(total_folds):
            fold_start_time = time.time()
            
            # 计算数据窗口
            start_idx = fold * step_size
            train_end_idx = start_idx + window_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + step_size, len(data))
            
            # 检查测试窗口是否足够
            if test_end_idx - test_start_idx < 20:
                self.logger.info(f"   ⏭️ 第{fold+1}折：测试窗口不足，跳过")
                continue
            
            # 获取训练和测试数据
            train_data = data.iloc[start_idx:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()
            
            self.logger.info(f"🔄 第{fold+1}/{total_folds}折:")
            self.logger.info(f"   - 训练数据: {len(train_data)} 条")
            self.logger.info(f"   - 测试数据: {len(test_data)} 条")
            self.logger.info(f"   - 训练期间: {train_data.iloc[0]['date']} ~ {train_data.iloc[-1]['date']}")
            self.logger.info(f"   - 测试期间: {test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}")
            
            try:
                # 在训练数据上优化参数（严格隔离）
                temp_strategy = StrategyModule(self.config)
                optimized_params = self.optimize_strategy_parameters_on_train_only(
                    temp_strategy, train_data
                )
                temp_strategy.update_params(optimized_params)
                
                # 在测试数据上评估（绝对不参与优化）
                backtest_results = temp_strategy.backtest(test_data)
                evaluation = temp_strategy.evaluate_strategy(backtest_results)
                score = evaluation['score']
                
                scores.append(score)
                fold_results.append({
                    'fold': fold + 1,
                    'score': score,
                    'train_period': f"{train_data.iloc[0]['date']} ~ {train_data.iloc[-1]['date']}",
                    'test_period': f"{test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}",
                    'optimized_params': optimized_params,
                    'evaluation': evaluation
                })
                
                fold_time = time.time() - fold_start_time
                self.logger.info(f"   ✅ 得分: {score:.4f}，耗时: {fold_time:.1f}秒")
                
            except Exception as e:
                self.logger.error(f"   ❌ 第{fold+1}折失败: {str(e)}")
                continue
        
        if len(scores) == 0:
            self.logger.error("❌ 走前验证失败，没有有效结果")
            return {'success': False, 'error': '没有有效的验证结果'}
        
        # 统计结果
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_time = time.time() - start_time
        
        self.logger.info("✅ 走前验证完成!")
        self.logger.info(f"📊 验证统计:")
        self.logger.info(f"   - 有效折数: {len(scores)}/{total_folds}")
        self.logger.info(f"   - 平均得分: {avg_score:.4f} ± {std_score:.4f}")
        self.logger.info(f"   - 得分范围: [{min_score:.4f}, {max_score:.4f}]")
        self.logger.info(f"   - 总耗时: {total_time:.1f}秒")
        
        return {
            'success': True,
            'avg_score': avg_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'valid_folds': len(scores),
            'total_folds': total_folds,
            'fold_results': fold_results,
            'total_time': total_time
        }
    
    def optimize_strategy_parameters_on_train_only(self, strategy_module, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在训练数据上进行参数优化，绝对不使用验证/测试数据
        
        参数:
        strategy_module: 策略模块实例
        train_data: 严格的训练数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("🔧 开始训练集参数优化（数据泄露保护）...")
        
        try:
            # 1. 验证这是纯训练数据
            if self._test_set_locked and self._test_set_indices:
                train_indices = train_data.index.tolist()
                test_indices_set = set(self._test_set_indices)
                if any(idx in test_indices_set for idx in train_indices):
                    raise ValueError("❌ 检测到数据泄露：训练数据包含测试集数据！")
            
            # 2. 获取基准策略的识别结果作为固定标签
            baseline_backtest = strategy_module.backtest(train_data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            
            # 3. 固定核心参数
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            # 4. 获取搜索范围
            ai_config = self.config.get('ai', {})
            optimization_ranges = ai_config.get('optimization_ranges', {})
            
            # 直接构建参数网格（避免方法顺序依赖问题）
            param_grid = {}
            param_configs = {
                'rsi_oversold_threshold': {'type': 'int', 'default_min': 25, 'default_max': 35, 'default_step': 1},
                'rsi_low_threshold': {'type': 'int', 'default_min': 35, 'default_max': 45, 'default_step': 1},
                'final_threshold': {'type': 'float', 'default_min': 0.3, 'default_max': 0.7, 'default_step': 0.05},
                'dynamic_confidence_adjustment': {'type': 'float', 'default_min': 0.05, 'default_max': 0.25, 'default_step': 0.02},
                'market_sentiment_weight': {'type': 'float', 'default_min': 0.08, 'default_max': 0.25, 'default_step': 0.02},
                'trend_strength_weight': {'type': 'float', 'default_min': 0.06, 'default_max': 0.20, 'default_step': 0.02},
                'volume_weight': {'type': 'float', 'default_min': 0.15, 'default_max': 0.35, 'default_step': 0.02},
                'price_momentum_weight': {'type': 'float', 'default_min': 0.12, 'default_max': 0.30, 'default_step': 0.02}
            }
            
            for param_name, config in param_configs.items():
                param_range = optimization_ranges.get(param_name, {})
                min_val = param_range.get('min', config['default_min'])
                max_val = param_range.get('max', config['default_max'])
                step = param_range.get('step', config['default_step'])
                
                if config['type'] == 'int':
                    param_grid[param_name] = np.arange(min_val, max_val + 1, step)
                else:
                    param_grid[param_name] = np.arange(min_val, max_val + step, step)
            
            # 5. 早停机制
            early_stopping = EarlyStopping(
                patience=ai_config.get('early_stopping', {}).get('patience', 50),
                min_delta=ai_config.get('early_stopping', {}).get('min_delta', 0.001)
            )
            
            # 6. 参数优化（仅使用训练数据）
            best_score = -1
            best_params = None
            max_iterations = 200  # 减少迭代次数以提高效率
            
            for iteration in range(max_iterations):
                # 随机生成参数组合
                params = {
                    'rise_threshold': fixed_rise_threshold,
                    'max_days': fixed_max_days,
                    'rsi_oversold_threshold': int(np.random.choice(param_grid['rsi_oversold_threshold'])),
                    'rsi_low_threshold': int(np.random.choice(param_grid['rsi_low_threshold'])),
                    'final_threshold': np.random.choice(param_grid['final_threshold']),
                    'dynamic_confidence_adjustment': np.random.choice(param_grid['dynamic_confidence_adjustment']),
                    'market_sentiment_weight': np.random.choice(param_grid['market_sentiment_weight']),
                    'trend_strength_weight': np.random.choice(param_grid['trend_strength_weight']),
                    'volume_weight': np.random.choice(param_grid['volume_weight']),
                    'price_momentum_weight': np.random.choice(param_grid['price_momentum_weight'])
                }
                
                # 评估参数（仅使用训练数据）
                # 直接实现评估逻辑避免方法顺序依赖
                try:
                    scores = []
                    low_point_indices = np.where(fixed_labels == 1)[0]
                    
                    rise_threshold = params['rise_threshold']
                    max_days = params['max_days']
                    
                    for idx in low_point_indices:
                        if idx >= len(train_data) - max_days:
                            continue
                            
                        current_price = train_data.iloc[idx]['close']
                        max_rise = 0.0
                        days_to_rise = 0
                        
                        # 计算未来max_days内的最大涨幅
                        for j in range(1, max_days + 1):
                            if idx + j >= len(train_data):
                                break
                            future_price = train_data.iloc[idx + j]['close']
                            rise_rate = (future_price - current_price) / current_price
                            
                            if rise_rate > max_rise:
                                max_rise = rise_rate
                                
                            if rise_rate >= rise_threshold and days_to_rise == 0:
                                days_to_rise = j
                        
                        # 计算单个点的得分
                        success = max_rise >= rise_threshold
                        
                        # 简化的得分计算
                        if success:
                            speed_factor = max_days / max(days_to_rise, 1) if days_to_rise > 0 else 0
                            point_score = 0.6 + 0.3 * min(max_rise / 0.1, 1.0) + 0.1 * min(speed_factor, 1.0)
                        else:
                            point_score = 0.1 * min(max_rise / 0.05, 1.0)  # 部分分数
                        
                        scores.append(point_score)
                    
                    score = np.mean(scores) if len(scores) > 0 else 0.0
                    
                except Exception as eval_error:
                    self.logger.warning(f"参数评估失败: {str(eval_error)}")
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # 早停检查
                if early_stopping(score):
                    self.logger.info(f"🛑 早停触发，停止优化 (迭代: {iteration+1})")
                    break
            
            self.logger.info(f"✅ 训练集优化完成，最佳得分: {best_score:.4f}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"❌ 训练集参数优化失败: {str(e)}")
            # 返回默认参数
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                'max_days': self.config.get('strategy', {}).get('max_days', 20),
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5,
                'dynamic_confidence_adjustment': 0.15,
                'market_sentiment_weight': 0.15,
                'trend_strength_weight': 0.12,
                'volume_weight': 0.25,
                'price_momentum_weight': 0.20
            }
    
    def evaluate_on_test_set_only(self, strategy_module, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在测试集上进行最终评估，绝对不影响优化过程
        
        参数:
        strategy_module: 已优化的策略模块
        test_data: 严格保护的测试数据
        
        返回:
        dict: 测试集评估结果
        """
        self.logger.info("🎯 开始测试集最终评估...")
        
        try:
            # 验证测试集完整性
            if self._test_set_locked and self._test_set_indices:
                test_indices = test_data.index.tolist()
                if test_indices != self._test_set_indices:
                    raise ValueError("❌ 测试集数据不完整或被篡改！")
            
            # 在测试集上运行回测
            self.logger.info("📊 在测试集上运行回测...")
            backtest_results = strategy_module.backtest(test_data)
            evaluation = strategy_module.evaluate_strategy(backtest_results)
            
            # 详细统计
            test_score = evaluation['score']
            success_rate = evaluation['success_rate']
            total_points = evaluation['total_points']
            avg_rise = evaluation['avg_rise']
            
            self.logger.info("✅ 测试集评估完成!")
            self.logger.info(f"📊 测试集性能:")
            self.logger.info(f"   - 综合得分: {test_score:.4f}")
            self.logger.info(f"   - 成功率: {success_rate:.2%}")
            self.logger.info(f"   - 识别点数: {total_points}")
            self.logger.info(f"   - 平均涨幅: {avg_rise:.2%}")
            self.logger.info(f"   - 测试期间: {test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}")
            
            return {
                'success': True,
                'test_score': test_score,
                'success_rate': success_rate,
                'total_points': total_points,
                'avg_rise': avg_rise,
                'test_period': f"{test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}",
                'backtest_results': backtest_results,
                'evaluation': evaluation
            }
            
        except Exception as e:
            self.logger.error(f"❌ 测试集评估失败: {str(e)}")
            return {'success': False, 'error': str(e)}


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        """
        初始化早停机制
        
        参数:
        patience: 耐心值，连续多少次无改进后停止
        min_delta: 最小改进幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        
    def __call__(self, val_score: float) -> bool:
        """
        检查是否应该早停
        
        参数:
        val_score: 当前验证得分
        
        返回:
        bool: 是否应该停止
        """
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

    def optimize_strategy_parameters(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化策略参数（rise_threshold和max_days保持固定）
        
        参数:
        strategy_module: 策略模块实例
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始AI策略参数优化")
        self.logger.info("=" * 60)
        
        try:
            # 1. 获取基准策略的识别结果作为固定标签
            self.logger.info("📊 阶段1: 获取基准策略识别结果...")
            baseline_backtest = strategy_module.backtest(data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            self.logger.info(f"✅ 基准策略识别点数: {np.sum(fixed_labels)}")
            
            # 2. 固定核心参数，不允许优化
            self.logger.info("🔧 阶段2: 设置固定参数...")
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            self.logger.info(f"✅ 固定参数设置完成:")
            self.logger.info(f"   - rise_threshold: {fixed_rise_threshold}")
            self.logger.info(f"   - max_days: {fixed_max_days}")
            
            # 3. 加载历史最优参数，决定是否进行增量优化
            self.logger.info("📋 阶段3: 检查历史参数...")
            historical_best_params = self._load_best_parameters()
            
            if historical_best_params:
                self.logger.info("🔄 发现历史最优参数，启用增量优化模式")
                use_incremental = True
                base_params = historical_best_params
            else:
                self.logger.info("🆕 没有历史参数，使用全局搜索模式")
                use_incremental = False
                base_params = None
            
            # 4. 从配置文件获取可优化参数的搜索范围
            self.logger.info("📋 阶段4: 配置参数搜索范围...")
            ai_config = self.config.get('ai', {})
            optimization_ranges = ai_config.get('optimization_ranges', {})
            
            # 验证配置
            if not self._validate_optimization_config(optimization_ranges):
                self.logger.error("❌ 优化配置验证失败，使用默认配置")
                optimization_ranges = {}
            
            # 根据是否增量优化选择搜索范围
            if use_incremental:
                self.logger.info("🎯 使用增量搜索范围（基于历史最优参数）:")
                param_grid = self._get_incremental_search_ranges(base_params, optimization_ranges)
            else:
                self.logger.info("🌐 使用全局搜索范围:")
                param_grid = self._build_parameter_grid(optimization_ranges)
            
            self.logger.info("✅ 可优化参数搜索范围:")
            for param, values in param_grid.items():
                self.logger.info(f"   - {param}: {values[0]} - {values[-1]}, 步长: {values[1]-values[0] if len(values)>1 else 'N/A'}")
            
            # 5. 设置初始最佳参数和得分
            if use_incremental and base_params:
                # 增量优化：以历史最优参数为起点
                best_score = self._evaluate_params_with_fixed_labels_advanced(data, fixed_labels, base_params)
                best_params = base_params.copy()
                self.logger.info(f"🎯 历史最优参数作为起点，得分: {best_score:.4f}")
            else:
                # 全局优化：从零开始
                best_score = -1
                best_params = None
            
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            self.logger.info(f"📈 总搜索组合数: {total_combinations:,}")
            
            # 6. 基于固定标签优化可调参数
            # 从配置文件获取迭代次数配置
            optimization_config = ai_config.get('optimization', {})
            global_iterations = optimization_config.get('global_iterations', 150)
            incremental_iterations = optimization_config.get('incremental_iterations', 100)
            enable_incremental = optimization_config.get('enable_incremental', True)
            
            # 根据是否增量优化调整迭代次数
            if use_incremental and enable_incremental:
                max_iterations = min(incremental_iterations, total_combinations)  # 增量优化使用较少迭代
                self.logger.info(f"🎯 增量优化模式，最大迭代次数: {max_iterations} (配置值: {incremental_iterations})")
            else:
                max_iterations = min(global_iterations, total_combinations)  # 全局优化使用更多迭代
                self.logger.info(f"🌐 全局优化模式，最大迭代次数: {max_iterations} (配置值: {global_iterations})")
            
            # 预生成参数组合以提高效率
            self.logger.info("⚡ 预生成参数组合...")
            param_combinations = []
            for _ in range(max_iterations):
                params = {
                    'rise_threshold': fixed_rise_threshold,  # 固定不变
                    'max_days': fixed_max_days,              # 固定不变
                    'rsi_oversold_threshold': int(np.random.choice(param_grid['rsi_oversold_threshold'])),
                    'rsi_low_threshold': int(np.random.choice(param_grid['rsi_low_threshold'])),
                    'final_threshold': np.random.choice(param_grid['final_threshold']),
                    # 新增AI优化参数
                    'dynamic_confidence_adjustment': np.random.choice(param_grid['dynamic_confidence_adjustment']),
                    'market_sentiment_weight': np.random.choice(param_grid['market_sentiment_weight']),
                    'trend_strength_weight': np.random.choice(param_grid['trend_strength_weight']),
                    # 新增2个高重要度参数
                    'volume_weight': np.random.choice(param_grid['volume_weight']),
                    'price_momentum_weight': np.random.choice(param_grid['price_momentum_weight'])
                }
                param_combinations.append(params)
            
            # 记录优化开始时间
            import time
            start_time = time.time()
            
            self.logger.info("🔄 阶段5: 开始参数优化迭代...")
            self.logger.info("-" * 50)
            
            # 记录改进次数
            improvement_count = 0
            
            for iteration in range(max_iterations):
                # 计算进度
                progress = (iteration + 1) / max_iterations * 100
                
                # 每5次迭代或第一次迭代时打印进度
                if iteration == 0 or (iteration + 1) % 5 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / (iteration + 1)
                    remaining_iter = max_iterations - (iteration + 1)
                    estimated_remaining_time = remaining_iter * avg_time_per_iter
                    
                    self.logger.info(f"📊 进度: {progress:.1f}% ({iteration+1}/{max_iterations})")
                    self.logger.info(f"⏱️  已用时间: {elapsed_time:.1f}s, 预计剩余: {estimated_remaining_time:.1f}s")
                    self.logger.info(f"🏆 当前最佳得分: {best_score:.4f}")
                    if best_params:
                        self.logger.info(f"🎯 当前最佳参数:")
                        self.logger.info(f"   - RSI超卖阈值: {best_params['rsi_oversold_threshold']}")
                        self.logger.info(f"   - RSI低值阈值: {best_params['rsi_low_threshold']}")
                        self.logger.info(f"   - 最终置信度: {best_params['final_threshold']:.3f}")
                        self.logger.info(f"   - 动态调整系数: {best_params['dynamic_confidence_adjustment']:.3f}")
                        self.logger.info(f"   - 市场情绪权重: {best_params['market_sentiment_weight']:.3f}")
                        self.logger.info(f"   - 趋势强度权重: {best_params['trend_strength_weight']:.3f}")
                        self.logger.info(f"   - 成交量权重: {best_params['volume_weight']:.3f}")
                        self.logger.info(f"   - 价格动量权重: {best_params['price_momentum_weight']:.3f}")
                    self.logger.info("-" * 30)
                
                # 使用预生成的参数组合
                params = param_combinations[iteration]
                
                # 使用固定标签评估参数
                score = self._evaluate_params_with_fixed_labels_advanced(
                    data, fixed_labels, params
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    improvement_count += 1
                    
                    # 计算改进幅度
                    improvement = score - best_score if best_score != -1 else 0
                    
                    self.logger.info(f"🎉 发现更好的参数组合 (第{improvement_count}次改进, 迭代{iteration+1}):")
                    self.logger.info(f"   📈 得分提升: {improvement:.4f} → {best_score:.4f}")
                    self.logger.info(f"   🔧 参数详情:")
                    self.logger.info(f"      - RSI超卖阈值: {best_params['rsi_oversold_threshold']}")
                    self.logger.info(f"      - RSI低值阈值: {best_params['rsi_low_threshold']}")
                    self.logger.info(f"      - 最终置信度: {best_params['final_threshold']:.3f}")
                    self.logger.info(f"      - 动态调整系数: {best_params['dynamic_confidence_adjustment']:.3f}")
                    self.logger.info(f"      - 市场情绪权重: {best_params['market_sentiment_weight']:.3f}")
                    self.logger.info(f"      - 趋势强度权重: {best_params['trend_strength_weight']:.3f}")
                    self.logger.info(f"      - 成交量权重: {best_params['volume_weight']:.3f}")
                    self.logger.info(f"      - 价格动量权重: {best_params['price_momentum_weight']:.3f}")
                    self.logger.info("-" * 50)
            
            # 优化完成统计
            total_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info("🎯 AI策略参数优化完成!")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 优化统计:")
            self.logger.info(f"   - 优化模式: {'增量优化' if use_incremental else '全局优化'}")
            self.logger.info(f"   - 总迭代次数: {max_iterations}")
            self.logger.info(f"   - 总耗时: {total_time:.1f}秒")
            self.logger.info(f"   - 平均每次迭代: {total_time/max_iterations:.3f}秒")
            self.logger.info(f"   - 改进次数: {improvement_count}")
            self.logger.info(f"   - 最终最佳得分: {best_score:.4f}")
            self.logger.info("")
            self.logger.info(f"🏆 最终最佳参数:")
            for key, value in best_params.items():
                if isinstance(value, float):
                    self.logger.info(f"   - {key}: {value:.4f}")
                else:
                    self.logger.info(f"   - {key}: {value}")
            
            # 保存优化结果到历史记录
            self.logger.info("💾 保存优化结果...")
            self._save_parameter_history(best_params, best_score)
            self._save_best_parameters(best_params, best_score)
            
            return best_params
            
        except Exception as e:
            self.logger.error("AI策略参数优化失败: %s", str(e))
            self.logger.error("错误详情:", exc_info=True)
            
            # 尝试返回默认参数
            try:
                default_params = {
                    'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                    'max_days': self.config.get('strategy', {}).get('max_days', 20),
                    'rsi_oversold_threshold': 30,
                    'rsi_low_threshold': 40,
                    'final_threshold': 0.5,
                    'dynamic_confidence_adjustment': 0.15,
                    'market_sentiment_weight': 0.15,
                    'trend_strength_weight': 0.12,
                    'volume_weight': 0.25,
                    'price_momentum_weight': 0.20
                }
                self.logger.warning("返回默认参数作为备选方案")
                return default_params
            except Exception as fallback_error:
                self.logger.error("备选方案也失败: %s", str(fallback_error))
                return {}
    
    def _evaluate_params_with_fixed_labels(self, data: pd.DataFrame, fixed_labels: np.ndarray, 
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
            # 1. 计算每个识别点的未来表现
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
                point_score = self._calculate_point_score(success, max_rise, days_to_rise, max_days)
                scores.append(point_score)
            
            # 2. 计算总体得分
            if len(scores) == 0:
                return 0.0
                
            return np.mean(scores)
            
        except Exception as e:
            self.logger.error("评估参数失败: %s", str(e))
            return 0.0
    
    def _calculate_point_score(self, success: bool, max_rise: float, days_to_rise: int, max_days: int) -> float:
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
        # 从配置文件获取评分参数
        ai_config = self.config.get('ai', {})
        scoring_config = ai_config.get('scoring', {})
        
        # 成功率权重：40%
        success_weight = scoring_config.get('success_weight', 0.4)
        success_score = 1.0 if success else 0.0
        
        # 涨幅权重：30%（相对于基准涨幅）
        rise_weight = scoring_config.get('rise_weight', 0.3)
        rise_benchmark = scoring_config.get('rise_benchmark', 0.1)  # 10%基准
        rise_score = min(max_rise / rise_benchmark, 1.0)
        
        # 速度权重：20%（天数越少越好）
        speed_weight = scoring_config.get('speed_weight', 0.2)
        if days_to_rise > 0:
            speed_score = min(max_days / days_to_rise, 1.0)
        else:
            speed_score = 0.0
        
        # 风险调整：10%（避免过度冒险）
        risk_weight = scoring_config.get('risk_weight', 0.1)
        risk_benchmark = scoring_config.get('risk_benchmark', 0.2)  # 20%风险阈值
        risk_score = min(max_rise / risk_benchmark, 1.0)  # 超过风险阈值的涨幅给予风险惩罚
        
        # 综合得分
        total_score = (
            success_score * success_weight +
            rise_score * rise_weight +
            speed_score * speed_weight +
            risk_score * risk_weight
        )
        
        return total_score
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        准备机器学习特征
        
        参数:
        data: 历史数据
        
        返回:
        tuple: (特征矩阵, 特征名称列表)
        """
        self.logger.info("准备机器学习特征")
        
        # 选择特征列
        feature_columns = [
            'ma5', 'ma10', 'ma20', 'ma60',
            'rsi', 'macd', 'signal', 'hist',
            'bb_upper', 'bb_lower',
            'dist_ma5', 'dist_ma10', 'dist_ma20',
            'volume_change', 'volatility',
            'price_change', 'price_change_5d', 'price_change_10d'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if len(available_columns) == 0:
            self.logger.warning("没有可用的特征列")
            return np.array([]), []
            
        # 提取特征
        features = data[available_columns].fillna(0).values
        
        self.logger.info("特征准备完成，特征数量: %d, 样本数量: %d", 
                        len(available_columns), len(features))
        
        return features, available_columns
        
    def prepare_labels(self, data: pd.DataFrame, strategy_module) -> np.ndarray:
        """
        准备机器学习标签
        
        参数:
        data: 历史数据
        strategy_module: 策略模块实例
        
        返回:
        numpy.ndarray: 标签数组
        """
        self.logger.info("准备机器学习标签")
        
        # 运行回测获取真实的相对低点标签
        backtest_results = strategy_module.backtest(data)
        labels = backtest_results['is_low_point'].astype(int).values
        
        positive_count = np.sum(labels)
        total_count = len(labels)
        
        self.logger.info("标签准备完成，正样本: %d, 总样本: %d, 正样本比例: %.2f%%", 
                        positive_count, total_count, positive_count / total_count * 100)
        
        return labels
        
    def train_model(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        只负责训练模型并保存，不做评估。
        """
        self.logger.info("开始训练模型（不做验证评估）")
        try:
            features, feature_names = self.prepare_features(data)
            labels = self.prepare_labels(data, strategy_module)
            if len(features) == 0 or len(labels) == 0:
                self.logger.error("特征或标签为空，无法训练模型")
                return {'success': False, 'error': '特征或标签为空'}
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            aligned_data = data.iloc[:min_length].copy()
            split_ratio = self.config.get("ai", {}).get("train_test_split_ratio", 0.8)
            split_index = int(len(features) * split_ratio)
            X_train = features[:split_index]
            y_train = labels[:split_index]
            train_dates = aligned_data["date"].iloc[:split_index]
            sample_weights = self._calculate_sample_weights(train_dates)
            if self.model_type == 'machine_learning':
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight='balanced'
                    ))
                ])
            else:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight='balanced'
                    ))
                ])
            self.logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, sample_weights shape: {sample_weights.shape}")
            model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
            self.model = model
            self.feature_names = feature_names
            self._save_model()
            self.logger.info("模型训练完成")
            return {'success': True, 'train_samples': len(X_train), 'feature_count': len(feature_names)}
        except Exception as e:
            self.logger.error("训练模型失败: %s", str(e))
            return {'success': False, 'error': str(e)}

    def validate_model(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        只负责评估模型在验证集上的表现。
        """
        self.logger.info("开始验证模型（只做评估，不训练）")
        try:
            if self.model is None:
                self.logger.warning("模型未训练，尝试加载已保存的模型")
                if not self._load_model():
                    return {'success': False, 'error': '模型未训练且无法加载已保存的模型'}
            features, feature_names = self.prepare_features(data)
            labels = self.prepare_labels(data, strategy_module)
            if len(features) == 0 or len(labels) == 0:
                self.logger.error("特征或标签为空，无法验证模型")
                return {'success': False, 'error': '特征或标签为空'}
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            split_ratio = self.config.get("ai", {}).get("train_test_split_ratio", 0.8)
            split_index = int(len(features) * split_ratio)
            X_test = features[split_index:]
            y_test = labels[split_index:]
            if len(X_test) == 0 or len(y_test) == 0:
                self.logger.warning("验证集为空，无法评估模型")
                return {'success': False, 'error': '验证集为空'}
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            positive_count_test = np.sum(y_test)
            self.logger.info("模型在验证集上评估完成，准确率: %.4f, 精确率: %.4f, 召回率: %.4f, F1: %.4f", accuracy, precision, recall, f1)
            return {
                'success': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_samples': len(X_test),
                'positive_samples_test': positive_count_test
            }
        except Exception as e:
            self.logger.error("验证模型失败: %s", str(e))
            return {'success': False, 'error': str(e)}

    def predict_low_point(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测相对低点
        
        参数:
        data: 市场数据
        
        返回:
        dict: 预测结果
        """
        self.logger.info("预测相对低点")
        
        try:
            if self.model is None:
                self.logger.warning("模型未训练，尝试加载已保存的模型")
                if not self._load_model():
                    return {
                        'is_low_point': False,
                        'confidence': 0.0,
                        'error': '模型未训练且无法加载已保存的模型'
                    }
                    
            if len(data) == 0:
                return {
                    'is_low_point': False,
                    'confidence': 0.0,
                    'error': '数据为空'
                }
                
            # 准备特征
            features, _ = self.prepare_features(data)
            
            if len(features) == 0:
                return {
                    'is_low_point': False,
                    'confidence': 0.0,
                    'error': '无法提取特征'
                }
                
            # 使用最新数据进行预测
            latest_features = features[-1:].reshape(1, -1)
            
            # 预测
            prediction = self.model.predict(latest_features)[0]
            prediction_proba = self.model.predict_proba(latest_features)[0]
            
            # 获取置信度
            confidence = prediction_proba[1] if len(prediction_proba) > 1 else 0.0
            
            result = {
                'is_low_point': bool(prediction),
                'confidence': float(confidence),
                'prediction_proba': prediction_proba.tolist()
            }
            
            self.logger.info("----------------------------------------------------");
            self.logger.info("AI预测结果: \033[1m%s\033[0m, 置信度: \033[1m%.4f\033[0m", 
                           "相对低点" if prediction else "非相对低点", confidence)
            self.logger.info("----------------------------------------------------");
            return result
            
        except Exception as e:
            self.logger.error("预测相对低点失败: %s", str(e))
            return {
                'is_low_point': False,
                'confidence': 0.0,
                'error': str(e)
            }
            
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        返回:
        dict: 特征重要性，如果模型未训练返回None
        """
        if self.model is None or self.feature_names is None:
            return None
            
        try:
            # 获取分类器
            classifier = self.model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                
                # 按重要性排序
                sorted_importance = dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True))
                
                self.logger.info("特征重要性获取成功")
                return sorted_importance
            else:
                self.logger.warning("模型不支持特征重要性")
                return None
                
        except Exception as e:
            self.logger.error("获取特征重要性失败: %s", str(e))
            return None
            
    def _save_model(self) -> bool:
        """
        保存模型
        
        返回:
        bool: 是否保存成功
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存模型
            model_path = os.path.join(self.models_dir, f'model_{timestamp}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            # 保存特征名称
            features_path = os.path.join(self.models_dir, f'features_{timestamp}.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)
                
            # 保存最新模型的路径
            latest_path = os.path.join(self.models_dir, 'latest_model.txt')
            with open(latest_path, 'w') as f:
                f.write(f'{model_path}\n{features_path}')
                
            self.logger.info("模型保存成功: %s", model_path)
            return True
            
        except Exception as e:
            self.logger.error("保存模型失败: %s", str(e))
            return False
            
    def _load_model(self) -> bool:
        """
        加载模型
        
        返回:
        bool: 是否加载成功
        """
        try:
            latest_path = os.path.join(self.models_dir, 'latest_model.txt')
            
            if not os.path.exists(latest_path):
                self.logger.warning("没有找到已保存的模型")
                return False
                
            # 读取最新模型路径
            with open(latest_path, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) < 2:
                    self.logger.error("模型路径文件格式错误")
                    return False
                    
                model_path = lines[0]
                features_path = lines[1]
                
            # 加载模型
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # 加载特征名称
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
                
            self.logger.info("模型加载成功: %s", model_path)
            return True
            
        except Exception as e:
            self.logger.error("加载模型失败: %s", str(e))
            return False
            
    def run_genetic_algorithm(self, evaluate_func, population_size: int = 20, 
                            generations: int = 10) -> Dict[str, Any]:
        """
        运行遗传算法优化（rise_threshold和max_days保持固定）
        
        参数:
        evaluate_func: 评估函数
        population_size: 种群大小
        generations: 迭代代数
        
        返回:
        dict: 最优参数
        """
        self.logger.info("运行遗传算法优化（rise_threshold和max_days保持固定），种群大小: %d, 迭代代数: %d", 
                        population_size, generations)
        
        try:
            # 固定核心参数，不允许优化
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            self.logger.info(f"固定参数 - rise_threshold: {fixed_rise_threshold}, max_days: {fixed_max_days}")
            
            # 由于核心参数已固定，遗传算法不需要进行
            # 直接返回固定参数
            self.logger.info("核心参数已固定，跳过遗传算法优化")
            
            return {
                'rise_threshold': fixed_rise_threshold,
                'max_days': fixed_max_days
            }
            
        except Exception as e:
            self.logger.error("遗传算法优化失败: %s", str(e))
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04), 
                'max_days': self.config.get('strategy', {}).get('max_days', 20)
            }
            
    def _genetic_operations(self, population: List[Dict], scores: List[float]) -> List[Dict]:
        """
        遗传算法操作（选择、交叉、变异）
        
        参数:
        population: 当前种群
        scores: 适应度得分
        
        返回:
        list: 新种群
        """
        # 选择（轮盘赌选择）
        total_score = sum(scores)
        if total_score <= 0:
            # 如果所有得分都是负数或零，随机选择
            selected = np.random.choice(len(population), size=len(population), replace=True)
        else:
            probabilities = [score / total_score for score in scores]
            selected = np.random.choice(len(population), size=len(population), 
                                      replace=True, p=probabilities)
            
        new_population = []
        
        for i in range(0, len(population), 2):
            parent1 = population[selected[i]]
            parent2 = population[selected[min(i + 1, len(population) - 1)]]
            
            # 交叉
            child1, child2 = self._crossover(parent1, parent2)
            
            # 变异
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
            
        return new_population[:len(population)]
        
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        交叉操作（rise_threshold和max_days保持固定）
        
        参数:
        parent1: 父代1
        parent2: 父代2
        
        返回:
        tuple: (子代1, 子代2)
        """
        # 固定核心参数
        fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
        fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
        
        child1 = {
            'rise_threshold': fixed_rise_threshold,  # 固定不变
            'max_days': fixed_max_days              # 固定不变
        }
        
        child2 = {
            'rise_threshold': fixed_rise_threshold,  # 固定不变
            'max_days': fixed_max_days              # 固定不变
        }
        
        return child1, child2
        
    def _mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """
        变异操作（rise_threshold和max_days保持固定）
        
        参数:
        individual: 个体
        mutation_rate: 变异率
        
        返回:
        dict: 变异后的个体
        """
        mutated = individual.copy()
        
        # 固定核心参数，不允许变异
        fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
        fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
        
        # 确保核心参数保持固定
        mutated['rise_threshold'] = fixed_rise_threshold
        mutated['max_days'] = fixed_max_days
            
        return mutated

    def _calculate_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """
        根据数据日期计算样本权重，越近的数据权重越高。
        权重衰减模型：V(t) = V₀ × e^(-λt)
        其中λ是衰减系数，根据分析报告，λ约为0.3-0.5。
        这里我们取λ=0.4，并根据时间差计算权重。
        
        参数:
        dates: 训练集数据的日期序列
        
        返回:
        numpy.ndarray: 样本权重数组
        """
        self.logger.info("计算样本权重")
        
        weights = np.ones(len(dates))
        if len(dates) == 0: # Handle empty dates series
            return weights

        latest_date = dates.max()
        
        for i, date in enumerate(dates):
            time_diff = (latest_date - date).days / 365.25  # 年为单位
            # 衰减系数λ，可以根据config配置
            decay_rate = self.config.get("ai", {}).get("data_decay_rate", 0.4)
            weight = np.exp(-decay_rate * time_diff)
            weights[i] = weight
            
        # 归一化权重，使其和为1，或者保持原始比例
        # 这里选择保持原始比例，因为RandomForestClassifier的sample_weight参数是乘法关系
        # 也可以选择归一化到某个范围，例如0-1
        
        self.logger.info("样本权重计算完成，最大权重: %.4f, 最小权重: %.4f", 
                        np.max(weights), np.min(weights))
        
        return weights

    def optimize_strategy_parameters_advanced(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        高级策略参数优化 - 使用多目标优化（rise_threshold保持固定）
        
        参数:
        strategy_module: 策略模块实例
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("开始高级策略参数优化（rise_threshold保持固定）")
        
        try:
            # 固定核心参数，不允许优化
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            self.logger.info(f"固定参数 - rise_threshold: {fixed_rise_threshold}, max_days: {fixed_max_days}")
            
            # 由于rise_threshold和max_days都是固定的，高级优化实际上不需要进行
            # 直接返回固定参数，只优化其他参数
            self.logger.info("核心参数已固定，跳过高级优化，使用基础优化方法")
            return self.optimize_strategy_parameters(strategy_module, data)
                
        except Exception as e:
            self.logger.error("高级优化失败: %s", str(e))
            return self.optimize_strategy_parameters(strategy_module, data)
    
    def time_series_cv_evaluation(self, data: pd.DataFrame, strategy_module) -> float:
        """
        时间序列交叉验证评估
        
        参数:
        data: 历史数据
        strategy_module: 策略模块实例
        
        返回:
        float: 平均得分
        """
        self.logger.info("🔄 开始时间序列交叉验证评估")
        
        try:
            total_score = 0
            cv_folds = 5
            fold_scores = []
            
            self.logger.info(f"📊 将数据分为 {cv_folds} 折进行验证...")
            
            # 记录开始时间
            import time
            cv_start_time = time.time()
            
            for i in range(cv_folds):
                fold_start_time = time.time()
                fold_progress = (i + 1) / cv_folds * 100
                
                self.logger.info(f"🔄 第{i+1}/{cv_folds}折 ({fold_progress:.1f}%) - 开始处理...")
                
                # 按时间分割数据
                split_point = int(len(data) * (i + 1) / cv_folds)
                train_data = data.iloc[:split_point]
                test_data = data.iloc[split_point:min(split_point + 100, len(data))]  # 测试窗口
                
                if len(test_data) < 20:  # 测试数据太少，跳过
                    self.logger.info(f"   ⏭️ 第{i+1}折：测试数据不足，跳过")
                    continue
                
                self.logger.info(f"   📋 数据分割完成：训练数据 {len(train_data)} 条，测试数据 {len(test_data)} 条")
                
                # 在训练数据上优化策略参数
                self.logger.info(f"   🔧 第{i+1}折：开始参数优化...")
                temp_strategy = StrategyModule(self.config)
                optimized_params = self.optimize_strategy_parameters(temp_strategy, train_data)
                temp_strategy.update_params(optimized_params)
                self.logger.info(f"   ✅ 第{i+1}折：参数优化完成")
                
                # 在测试数据上评估
                self.logger.info(f"   📊 第{i+1}折：开始回测评估...")
                backtest_results = temp_strategy.backtest(test_data)
                evaluation = temp_strategy.evaluate_strategy(backtest_results)
                score = evaluation['score']
                
                fold_scores.append(score)
                total_score += score
                
                fold_time = time.time() - fold_start_time
                self.logger.info(f"   ✅ 第{i+1}折完成：得分 {score:.4f}，耗时 {fold_time:.1f}秒")
                
                # 显示整体进度
                elapsed_time = time.time() - cv_start_time
                avg_time_per_fold = elapsed_time / (i + 1)
                remaining_folds = cv_folds - (i + 1)
                estimated_remaining_time = remaining_folds * avg_time_per_fold
                
                self.logger.info(f"   📈 整体进度：{fold_progress:.1f}%，已用时间：{elapsed_time:.1f}秒，预计剩余：{estimated_remaining_time:.1f}秒")
                self.logger.info("-" * 40)
            
            if len(fold_scores) == 0:
                self.logger.warning("⚠️ 没有有效的交叉验证结果")
                return 0.0
                
            avg_score = total_score / len(fold_scores)
            total_cv_time = time.time() - cv_start_time
            self.logger.info(f"📊 交叉验证完成，平均得分: {avg_score:.4f} (共{len(fold_scores)}折)")
            self.logger.info(f"⏱️ 总耗时: {total_cv_time:.1f}秒，平均每折: {total_cv_time/len(fold_scores):.1f}秒")
            
            return avg_score
            
        except Exception as e:
            self.logger.error("❌ 时间序列交叉验证失败: %s", str(e))
            return 0.0
    
    def hierarchical_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分层优化策略（使用严格数据分割防止过拟合）
        
        参数:
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        self.logger.info("=" * 60)
        self.logger.info("🏗️ 开始分层优化策略（严格数据分割版本）")
        self.logger.info("=" * 60)
        
        try:
            start_time = time.time()
            
            # 步骤1：严格数据分割
            self.logger.info("🔒 第一步：严格数据分割...")
            data_splits = self.strict_data_split(data, preserve_test_set=True)
            train_data = data_splits['train']
            validation_data = data_splits['validation']
            test_data = data_splits['test']
            
            # 第一层：仅在训练集上优化策略参数
            self.logger.info("📊 第一层：训练集策略参数优化...")
            layer1_start = time.time()
            strategy_module = StrategyModule(self.config)
            
            strategy_params = self.optimize_strategy_parameters_on_train_only(
                strategy_module, train_data
            )
            strategy_module.update_params(strategy_params)
            layer1_time = time.time() - layer1_start
            
            self.logger.info("✅ 第一层完成")
            self.logger.info(f"   - 优化参数: {strategy_params}")
            self.logger.info(f"   - 耗时: {layer1_time:.1f}秒")
            
            # 第二层：在训练集上训练AI模型，在验证集上评估
            self.logger.info("🤖 第二层：AI模型训练（训练集）+ 评估（验证集）...")
            layer2_start = time.time()
            
            # 使用训练集训练AI模型
            training_result = self.train_model(train_data, strategy_module)
            
            # 使用验证集评估AI模型
            validation_result = self.validate_model(validation_data, strategy_module)
            
            layer2_time = time.time() - layer2_start
            self.logger.info("✅ 第二层完成")
            self.logger.info(f"   - 训练准确率: {training_result.get('accuracy', 0):.4f}")
            self.logger.info(f"   - 验证准确率: {validation_result.get('accuracy', 0):.4f}")
            self.logger.info(f"   - 耗时: {layer2_time:.1f}秒")
            
            # 第三层：走前验证（使用训练+验证数据）
            self.logger.info("🚶 第三层：走前验证...")
            layer3_start = time.time()
            
            # 合并训练和验证数据用于走前验证
            train_val_data = pd.concat([train_data, validation_data]).reset_index(drop=True)
            
            # 获取走前验证配置
            ai_config = self.config.get('ai', {})
            validation_config = ai_config.get('validation', {})
            walk_forward_config = validation_config.get('walk_forward', {})
            
            if walk_forward_config.get('enabled', True):
                wf_result = self.walk_forward_validation(
                    train_val_data, 
                    strategy_module,
                    window_size=walk_forward_config.get('window_size', 252),
                    step_size=walk_forward_config.get('step_size', 63)
                )
                cv_score = wf_result.get('avg_score', 0.0) if wf_result['success'] else 0.0
            else:
                # 如果禁用走前验证，使用简单验证集评估
                val_backtest = strategy_module.backtest(validation_data)
                val_evaluation = strategy_module.evaluate_strategy(val_backtest)
                cv_score = val_evaluation['score']
            
            layer3_time = time.time() - layer3_start
            self.logger.info("✅ 第三层完成")
            self.logger.info(f"   - 验证得分: {cv_score:.4f}")
            self.logger.info(f"   - 耗时: {layer3_time:.1f}秒")
            
            # 第四层：最终测试集评估（严格保护）
            self.logger.info("🎯 第四层：测试集最终评估...")
            layer4_start = time.time()
            
            test_result = self.evaluate_on_test_set_only(strategy_module, test_data)
            test_score = test_result.get('test_score', 0.0) if test_result['success'] else 0.0
            
            layer4_time = time.time() - layer4_start
            self.logger.info("✅ 第四层完成")
            self.logger.info(f"   - 测试集得分: {test_score:.4f}")
            self.logger.info(f"   - 耗时: {layer4_time:.1f}秒")
            
            # 最终结果统计
            total_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info("🎯 分层优化完成！（严格数据分割版本）")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 优化统计:")
            self.logger.info(f"   - 总耗时: {total_time:.1f}秒")
            self.logger.info(f"   - 第一层（训练集优化）: {layer1_time:.1f}秒 ({layer1_time/total_time*100:.1f}%)")
            self.logger.info(f"   - 第二层（AI训练）: {layer2_time:.1f}秒 ({layer2_time/total_time*100:.1f}%)")
            self.logger.info(f"   - 第三层（走前验证）: {layer3_time:.1f}秒 ({layer3_time/total_time*100:.1f}%)")
            self.logger.info(f"   - 第四层（测试评估）: {layer4_time:.1f}秒 ({layer4_time/total_time*100:.1f}%)")
            self.logger.info("")
            self.logger.info(f"🏆 最终结果:")
            self.logger.info(f"   - 验证集得分: {cv_score:.4f}")
            self.logger.info(f"   - 测试集得分: {test_score:.4f}")
            self.logger.info(f"   - 过拟合检测: {'通过' if test_score >= cv_score * 0.8 else '警告'}")
            
            # 计算过拟合程度
            if cv_score > 0:
                overfitting_ratio = (cv_score - test_score) / cv_score
                if overfitting_ratio > 0.2:
                    self.logger.warning(f"⚠️ 检测到可能的过拟合，验证-测试得分差异: {overfitting_ratio:.1%}")
                else:
                    self.logger.info(f"✅ 过拟合风险较低，验证-测试得分差异: {overfitting_ratio:.1%}")
            
            return {
                'params': strategy_params,
                'cv_score': cv_score,
                'test_score': test_score,
                'best_score': cv_score,  # 使用验证集得分作为最佳得分
                'total_time': total_time,
                'layer_times': {
                    'layer1': layer1_time,
                    'layer2': layer2_time,
                    'layer3': layer3_time,
                    'layer4': layer4_time
                },
                'data_splits': {
                    'train_size': len(train_data),
                    'validation_size': len(validation_data),
                    'test_size': len(test_data)
                },
                'overfitting_check': {
                    'passed': test_score >= cv_score * 0.8,
                    'validation_score': cv_score,
                    'test_score': test_score,
                    'difference_ratio': (cv_score - test_score) / cv_score if cv_score > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error("❌ 分层优化失败: %s", str(e))
            return {
                'params': self.config.get('strategy', {}),
                'cv_score': 0.0,
                'test_score': 0.0,
                'best_score': 0.0,
                'error': str(e)
            }

    def _evaluate_params_with_fixed_labels_advanced(self, data: pd.DataFrame, fixed_labels: np.ndarray, 
                                                  params: Dict[str, Any]) -> float:
        """
        使用固定标签评估多参数策略
        
        参数:
        data: 历史数据
        fixed_labels: 固定的标签（相对低点标识）
        params: 参数字典，包含rise_threshold, max_days, rsi_oversold_threshold, rsi_low_threshold, final_threshold
        
        返回:
        float: 策略得分
        """
        try:
            # 1. 计算每个识别点的未来表现
            scores = []
            low_point_indices = np.where(fixed_labels == 1)[0]
            
            rise_threshold = params['rise_threshold']
            max_days = params['max_days']
            
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
                point_score = self._calculate_point_score(success, max_rise, days_to_rise, max_days)
                scores.append(point_score)
            
            # 2. 计算总体得分
            if len(scores) == 0:
                return 0.0
                
            return np.mean(scores)
            
        except Exception as e:
            self.logger.error("评估多参数失败: %s", str(e))
            return 0.0

    def _save_parameter_history(self, params: Dict[str, Any], score: float) -> bool:
        """
        保存参数历史记录
        
        参数:
        params: 参数字典
        score: 对应的得分
        
        返回:
        bool: 是否保存成功
        """
        try:
            # 读取现有历史记录
            history = []
            if os.path.exists(self.parameter_history_file):
                with open(self.parameter_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加新记录
            record = {
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'score': score
            }
            history.append(record)
            
            # 从配置文件获取最大记录数
            ai_config = self.config.get('ai', {})
            optimization_config = ai_config.get('optimization', {})
            max_history_records = optimization_config.get('max_history_records', 100)
            enable_history = optimization_config.get('enable_history', True)
            
            # 只保留最近N条记录
            if len(history) > max_history_records:
                history = history[-max_history_records:]
            
            # 保存历史记录
            if enable_history:
                with open(self.parameter_history_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"参数历史记录保存成功 (共{len(history)}条记录)")
            else:
                self.logger.info("参数历史记录功能已禁用")
            
            return True
            
        except Exception as e:
            self.logger.error("保存参数历史记录失败: %s", str(e))
            return False
    
    def _load_best_parameters(self) -> Optional[Dict[str, Any]]:
        """
        加载历史最优参数
        
        返回:
        dict: 历史最优参数，如果没有则返回None
        """
        try:
            if not os.path.exists(self.best_parameters_file):
                self.logger.info("没有找到历史最优参数文件")
                return None
            
            with open(self.best_parameters_file, 'r', encoding='utf-8') as f:
                best_record = json.load(f)
            
            self.logger.info("加载历史最优参数成功")
            self.logger.info(f"   - 历史最优得分: {best_record.get('score', 0):.4f}")
            self.logger.info(f"   - 历史最优参数: {best_record.get('parameters', {})}")
            
            return best_record.get('parameters')
            
        except Exception as e:
            self.logger.error("加载历史最优参数失败: %s", str(e))
            return None
    
    def _save_best_parameters(self, params: Dict[str, Any], score: float) -> bool:
        """
        保存当前最优参数
        
        参数:
        params: 参数字典
        score: 对应的得分
        
        返回:
        bool: 是否保存成功
        """
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'score': score
            }
            
            with open(self.best_parameters_file, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            
            self.logger.info("当前最优参数保存成功")
            return True
            
        except Exception as e:
            self.logger.error("保存当前最优参数失败: %s", str(e))
            return False
    
    def _get_incremental_search_ranges(self, base_params: Dict[str, Any], 
                                     optimization_ranges: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        基于历史最优参数生成增量搜索范围
        
        参数:
        base_params: 基础参数（历史最优参数）
        optimization_ranges: 完整搜索范围配置
        
        返回:
        dict: 增量搜索范围
        """
        try:
            incremental_ranges = {}
            
            # 从配置文件获取收缩比例
            ai_config = self.config.get('ai', {})
            optimization_config = ai_config.get('optimization', {})
            contraction_factor = optimization_config.get('incremental_contraction_factor', 0.3)
            
            self.logger.info(f"📊 增量搜索收缩比例: {contraction_factor}")
            
            # 定义所有必需的参数及其默认值
            required_params = {
                'rsi_oversold_threshold': {'type': 'int', 'default': 30},
                'rsi_low_threshold': {'type': 'int', 'default': 40},
                'final_threshold': {'type': 'float', 'default': 0.5},
                'dynamic_confidence_adjustment': {'type': 'float', 'default': 0.15},
                'market_sentiment_weight': {'type': 'float', 'default': 0.15},
                'trend_strength_weight': {'type': 'float', 'default': 0.12},
                'volume_weight': {'type': 'float', 'default': 0.25},
                'price_momentum_weight': {'type': 'float', 'default': 0.20}
            }
            
            for param_name, param_info in required_params.items():
                # 跳过核心参数
                if param_name in ['rise_threshold', 'max_days']:
                    continue
                
                # 获取基础值（从历史参数或默认值）
                base_value = base_params.get(param_name, param_info['default'])
                
                # 获取参数范围配置
                param_range = optimization_ranges.get(param_name, {})
                
                # 设置默认范围
                if param_name == 'rsi_oversold_threshold':
                    min_val = param_range.get('min', 25)
                    max_val = param_range.get('max', 35)
                    step = param_range.get('step', 1)
                elif param_name == 'rsi_low_threshold':
                    min_val = param_range.get('min', 35)
                    max_val = param_range.get('max', 45)
                    step = param_range.get('step', 1)
                elif param_name == 'final_threshold':
                    min_val = param_range.get('min', 0.3)
                    max_val = param_range.get('max', 0.7)
                    step = param_range.get('step', 0.05)
                elif param_name == 'dynamic_confidence_adjustment':
                    min_val = param_range.get('min', 0.05)
                    max_val = param_range.get('max', 0.25)
                    step = param_range.get('step', 0.02)
                elif param_name == 'market_sentiment_weight':
                    min_val = param_range.get('min', 0.08)
                    max_val = param_range.get('max', 0.25)
                    step = param_range.get('step', 0.02)
                elif param_name == 'trend_strength_weight':
                    min_val = param_range.get('min', 0.06)
                    max_val = param_range.get('max', 0.20)
                    step = param_range.get('step', 0.02)
                elif param_name == 'volume_weight':
                    min_val = param_range.get('min', 0.15)
                    max_val = param_range.get('max', 0.35)
                    step = param_range.get('step', 0.02)
                elif param_name == 'price_momentum_weight':
                    min_val = param_range.get('min', 0.12)
                    max_val = param_range.get('max', 0.30)
                    step = param_range.get('step', 0.02)
                else:
                    # 使用通用默认值
                    min_val = param_range.get('min', 0)
                    max_val = param_range.get('max', 1)
                    step = param_range.get('step', 0.01)
                
                # 计算增量搜索范围
                range_width = max_val - min_val
                incremental_width = range_width * contraction_factor
                
                # 以基础值为中心，向两边扩展
                new_min = max(min_val, base_value - incremental_width / 2)
                new_max = min(max_val, base_value + incremental_width / 2)
                
                # 确保至少有一个值
                if new_min >= new_max:
                    new_min = max(min_val, base_value - step)
                    new_max = min(max_val, base_value + step)
                
                # 生成搜索数组
                if param_info['type'] == 'int':
                    # 整数参数
                    incremental_ranges[param_name] = np.arange(
                        int(new_min), int(new_max) + 1, max(1, int(step))
                    )
                else:
                    # 浮点数参数
                    incremental_ranges[param_name] = np.arange(
                        new_min, new_max + step, step
                    )
                
                # 确保数组不为空
                if len(incremental_ranges[param_name]) == 0:
                    incremental_ranges[param_name] = np.array([base_value])
                
                self.logger.info(f"   - {param_name}: {new_min:.4f} - {new_max:.4f} (基于 {base_value:.4f})")
            
            return incremental_ranges
            
        except Exception as e:
            self.logger.error("生成增量搜索范围失败: %s", str(e))
            # 返回默认参数网格作为备选
            return self._build_parameter_grid(optimization_ranges)

    def _build_parameter_grid(self, optimization_ranges: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        构建参数搜索网格
        
        参数:
        optimization_ranges: 参数搜索范围配置
        
        返回:
        dict: 参数搜索网格
        """
        param_grid = {}
        
        # 定义参数配置
        param_configs = {
            'rsi_oversold_threshold': {'type': 'int', 'default_min': 25, 'default_max': 35, 'default_step': 1},
            'rsi_low_threshold': {'type': 'int', 'default_min': 35, 'default_max': 45, 'default_step': 1},
            'final_threshold': {'type': 'float', 'default_min': 0.3, 'default_max': 0.7, 'default_step': 0.05},
            'dynamic_confidence_adjustment': {'type': 'float', 'default_min': 0.05, 'default_max': 0.25, 'default_step': 0.02},
            'market_sentiment_weight': {'type': 'float', 'default_min': 0.08, 'default_max': 0.25, 'default_step': 0.02},
            'trend_strength_weight': {'type': 'float', 'default_min': 0.06, 'default_max': 0.20, 'default_step': 0.02},
            'volume_weight': {'type': 'float', 'default_min': 0.15, 'default_max': 0.35, 'default_step': 0.02},
            'price_momentum_weight': {'type': 'float', 'default_min': 0.12, 'default_max': 0.30, 'default_step': 0.02}
        }
        
        for param_name, config in param_configs.items():
            param_range = optimization_ranges.get(param_name, {})
            min_val = param_range.get('min', config['default_min'])
            max_val = param_range.get('max', config['default_max'])
            step = param_range.get('step', config['default_step'])
            
            if config['type'] == 'int':
                param_grid[param_name] = np.arange(min_val, max_val + 1, step)
            else:
                param_grid[param_name] = np.arange(min_val, max_val + step, step)
        
        return param_grid

    def _validate_optimization_config(self, optimization_ranges: Dict[str, Any]) -> bool:
        """
        验证优化配置的合理性
        
        参数:
        optimization_ranges: 参数搜索范围配置
        
        返回:
        bool: 配置是否有效
        """
        try:
            required_params = [
                'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
                'dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight',
                'volume_weight', 'price_momentum_weight'
            ]
            
            for param in required_params:
                if param not in optimization_ranges:
                    self.logger.warning(f"参数 {param} 未在配置中定义，将使用默认值")
                    continue
                
                param_range = optimization_ranges[param]
                min_val = param_range.get('min')
                max_val = param_range.get('max')
                step = param_range.get('step')
                
                if min_val is None or max_val is None or step is None:
                    self.logger.error(f"参数 {param} 配置不完整，缺少 min/max/step")
                    return False
                
                if min_val >= max_val:
                    self.logger.error(f"参数 {param} 范围配置错误: min({min_val}) >= max({max_val})")
                    return False
                
                if step <= 0:
                    self.logger.error(f"参数 {param} 步长配置错误: step({step}) <= 0")
                    return False
            
            self.logger.info("✅ 优化配置验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            return False


