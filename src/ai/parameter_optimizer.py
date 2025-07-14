#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参数优化器模块
专门负责策略参数的优化，从原AI优化器中分离出来

功能：
- 策略参数搜索和优化
- 网格搜索和随机搜索
- 参数范围管理
- 评分函数计算
- 优化结果保存
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from itertools import product

from ..utils.base_module import AIModule
from ..utils.common import (
    PerformanceMonitor, DataValidator, MathUtils,
    safe_execute, error_context, FileManager
)


class ParameterOptimizer(AIModule):
    """
    参数优化器
    
    专门负责策略参数的搜索和优化
    """
    
    def _initialize_module(self):
        """初始化参数优化器"""
        # 获取优化配置
        self.optimization_config = self.get_config_section('optimization', {})
        
        # 优化历史记录（带大小限制）
        self.optimization_history = []
        self.max_history_size = self.optimization_config.get('max_history_size', 100)
        
        # 当前最佳参数
        self.best_params = None
        self.best_score = -np.inf
        
        # 评分权重配置
        self.scoring_weights = self._load_scoring_weights()
        
        self.logger.info("参数优化器初始化完成")
    
    def _validate_module_config(self):
        """验证参数优化器配置"""
        # 基础AI模块验证
        super()._validate_module_config()
        
        # 检查优化配置
        if 'optimization' not in self.config:
            self.logger.warning("缺少optimization配置部分，使用默认配置")
    
    def _get_module_directories(self) -> List:
        """参数优化器特定目录"""
        base_dirs = super()._get_module_directories()
        return base_dirs + [
            self.project_root / 'results' / 'optimization',
            self.project_root / 'cache' / 'parameters'
        ]
    
    def _load_scoring_weights(self) -> Dict[str, float]:
        """加载评分权重配置"""
        default_weights = {
            'success_rate': 0.4,    # 成功率权重
            'avg_rise': 0.3,        # 平均涨幅权重
            'avg_days': 0.2,        # 平均天数权重（负权重，天数越少越好）
            'risk_penalty': 0.1     # 风险惩罚权重
        }
        
        scoring_config = self.get_config_section('ai', {}).get('scoring', {})
        weights = scoring_config.get('strategy_scoring', default_weights)
        
        # 验证权重总和
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"评分权重总和不等于1.0: {total_weight}，进行归一化")
            # 归一化权重
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def optimize_parameters(self, 
                           strategy_module,
                           data,
                           param_ranges: Dict[str, Any],
                           method: str = 'random',
                           max_iterations: int = 100) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            strategy_module: 策略模块实例
            data: 训练数据
            param_ranges: 参数搜索范围
            method: 优化方法 ('grid', 'random', 'bayesian')
            max_iterations: 最大迭代次数
        
        返回:
            Dict[str, Any]: 优化结果
        """
        with PerformanceMonitor(f"参数优化-{method}"):
            try:
                self.logger.info(f"开始参数优化，方法: {method}，最大迭代: {max_iterations}")
                
                # 验证输入数据
                valid, errors = self._validate_optimization_inputs(data, param_ranges)
                if not valid:
                    return {'success': False, 'error': f"输入验证失败: {errors}"}
                
                # 根据方法选择优化算法
                if method == 'grid':
                    result = self._grid_search_optimization(strategy_module, data, param_ranges)
                elif method == 'random':
                    result = self._random_search_optimization(strategy_module, data, param_ranges, max_iterations)
                elif method == 'bayesian':
                    result = self._bayesian_optimization(strategy_module, data, param_ranges, max_iterations)
                else:
                    return {'success': False, 'error': f"不支持的优化方法: {method}"}
                
                # 保存优化结果
                self._save_optimization_result(result, method)
                
                return result
                
            except Exception as e:
                self.logger.error(f"参数优化异常: {e}")
                return {'success': False, 'error': str(e)}
    
    def _validate_optimization_inputs(self, data, param_ranges) -> Tuple[bool, List[str]]:
        """验证优化输入"""
        errors = []
        
        # 验证数据
        if data is None or data.empty:
            errors.append("数据为空")
        elif len(data) < 100:
            errors.append(f"数据量不足，需要至少100条，实际{len(data)}条")
        
        # 验证参数范围
        if not param_ranges:
            errors.append("参数范围为空")
        
        for param_name, param_config in param_ranges.items():
            if not isinstance(param_config, dict):
                errors.append(f"参数 {param_name} 配置格式错误")
                continue
            
            required_keys = ['min', 'max']
            for key in required_keys:
                if key not in param_config:
                    errors.append(f"参数 {param_name} 缺少 {key} 配置")
        
        return len(errors) == 0, errors
    
    def _grid_search_optimization(self, strategy_module, data, param_ranges) -> Dict[str, Any]:
        """网格搜索优化"""
        self.logger.info("执行网格搜索优化")
        
        start_time = time.time()
        
        # 生成参数网格
        param_combinations = self._generate_parameter_grid(param_ranges)
        total_combinations = len(param_combinations)
        
        if total_combinations > 10000:
            self.logger.warning(f"参数组合数量过大: {total_combinations}，建议使用随机搜索")
            # 随机采样
            import random
            param_combinations = random.sample(param_combinations, 10000)
            total_combinations = len(param_combinations)
        
        self.logger.info(f"网格搜索组合数: {total_combinations}")
        
        best_params = None
        best_score = -np.inf
        best_metrics = None
        
        # 遍历所有参数组合
        for i, params in enumerate(param_combinations):
            try:
                # 评估参数组合
                score, metrics = self._evaluate_parameters(strategy_module, data, params)
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    
                    self.logger.info(f"发现更好参数 (第{i+1}次): 得分={score:.4f}")
                
                # 进度报告
                if (i + 1) % max(1, total_combinations // 10) == 0:
                    progress = (i + 1) / total_combinations * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"网格搜索进度: {progress:.1f}% ({i+1}/{total_combinations}), 已用时: {elapsed:.1f}秒")
                
            except Exception as e:
                self.logger.warning(f"评估参数组合失败 {params}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'grid_search',
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'total_combinations': total_combinations,
            'optimization_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _random_search_optimization(self, strategy_module, data, param_ranges, max_iterations) -> Dict[str, Any]:
        """随机搜索优化"""
        self.logger.info(f"执行随机搜索优化，最大迭代: {max_iterations}")
        
        start_time = time.time()
        
        best_params = None
        best_score = -np.inf
        best_metrics = None
        improvements = 0
        
        # 随机搜索
        for i in range(max_iterations):
            try:
                # 生成随机参数
                params = self._generate_random_parameters(param_ranges)
                
                # 评估参数
                score, metrics = self._evaluate_parameters(strategy_module, data, params)
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    improvements += 1
                    
                    self.logger.info(f"发现更好参数 (第{improvements}次改进, 迭代{i+1}): 得分={score:.4f}")
                
                # 进度报告
                if (i + 1) % max(1, max_iterations // 10) == 0:
                    progress = (i + 1) / max_iterations * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"随机搜索进度: {progress:.1f}% ({i+1}/{max_iterations}), 已用时: {elapsed:.1f}秒")
                
            except Exception as e:
                self.logger.warning(f"评估随机参数失败 {params}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'random_search',
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'iterations': max_iterations,
            'improvements': improvements,
            'optimization_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _bayesian_optimization(self, strategy_module, data, param_ranges, max_iterations) -> Dict[str, Any]:
        """贝叶斯优化（简化实现）"""
        # 这里可以集成 scikit-optimize 或其他贝叶斯优化库
        # 目前使用改进的随机搜索作为替代
        self.logger.info("执行贝叶斯优化（当前使用改进随机搜索）")
        
        # 使用自适应的随机搜索
        return self._adaptive_random_search(strategy_module, data, param_ranges, max_iterations)
    
    def _adaptive_random_search(self, strategy_module, data, param_ranges, max_iterations) -> Dict[str, Any]:
        """自适应随机搜索"""
        start_time = time.time()
        
        best_params = None
        best_score = -np.inf
        best_metrics = None
        improvements = 0
        
        # 历史参数和得分（带内存管理）
        param_history = []
        score_history = []
        max_history_items = min(max_iterations, 1000)  # 限制历史记录大小
        
        for i in range(max_iterations):
            try:
                # 自适应生成参数
                if i < max_iterations * 0.3:
                    # 前30%使用纯随机搜索
                    params = self._generate_random_parameters(param_ranges)
                else:
                    # 后70%基于历史结果调整搜索
                    params = self._generate_adaptive_parameters(param_ranges, param_history, score_history)
                
                # 评估参数
                score, metrics = self._evaluate_parameters(strategy_module, data, params)
                
                # 记录历史（保持大小限制）
                param_history.append(params)
                score_history.append(score)
                
                # 清理过多的历史记录
                if len(param_history) > max_history_items:
                    # 保留最好的一半和最近的一半
                    keep_count = max_history_items // 2
                    
                    # 找到最好的参数索引
                    best_indices = np.argsort(score_history)[-keep_count:]
                    recent_indices = list(range(len(param_history) - keep_count, len(param_history)))
                    
                    # 合并并去重
                    keep_indices = sorted(set(best_indices.tolist() + recent_indices))
                    
                    param_history = [param_history[i] for i in keep_indices]
                    score_history = [score_history[i] for i in keep_indices]
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    improvements += 1
                    
                    self.logger.info(f"发现更好参数 (第{improvements}次改进, 迭代{i+1}): 得分={score:.4f}")
                
                # 进度报告
                if (i + 1) % max(1, max_iterations // 10) == 0:
                    progress = (i + 1) / max_iterations * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (max_iterations - i - 1)
                    self.logger.info(f"自适应搜索进度: {progress:.1f}% ({i+1}/{max_iterations}), 已用时: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")
                
            except Exception as e:
                self.logger.warning(f"评估自适应参数失败: {e}")
                continue
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'adaptive_search',
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'iterations': max_iterations,
            'improvements': improvements,
            'optimization_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_parameter_grid(self, param_ranges) -> List[Dict[str, Any]]:
        """生成参数网格"""
        param_names = []
        param_values = []
        
        for name, config in param_ranges.items():
            param_names.append(name)
            
            # 生成参数值列表
            if 'values' in config:
                # 预定义的值列表
                values = config['values']
            else:
                # 根据范围生成值
                min_val = config['min']
                max_val = config['max']
                step = config.get('step', (max_val - min_val) / 10)
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # 整数范围
                    values = list(range(int(min_val), int(max_val) + 1, max(1, int(step))))
                else:
                    # 浮点数范围
                    values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                    values = [v for v in values if v <= max_val]
            
            param_values.append(values)
        
        # 生成所有组合
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_random_parameters(self, param_ranges) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        
        for name, config in param_ranges.items():
            if 'values' in config:
                # 从预定义值中随机选择
                params[name] = np.random.choice(config['values'])
            else:
                # 在范围内随机生成
                min_val = config['min']
                max_val = config['max']
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # 整数类型
                    params[name] = np.random.randint(min_val, max_val + 1)
                else:
                    # 浮点数类型
                    params[name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def _generate_adaptive_parameters(self, param_ranges, param_history, score_history) -> Dict[str, Any]:
        """基于历史结果生成自适应参数"""
        if not param_history:
            return self._generate_random_parameters(param_ranges)
        
        # 找到最好的几个参数组合
        sorted_indices = np.argsort(score_history)[-5:]  # 取最好的5个
        best_params = [param_history[i] for i in sorted_indices]
        
        # 在最好参数附近搜索
        base_params = best_params[-1]  # 使用最好的参数作为基础
        params = {}
        
        for name, config in param_ranges.items():
            base_value = base_params.get(name, (config['min'] + config['max']) / 2)
            
            # 在基础值附近添加噪声
            if 'values' in config:
                # 预定义值：随机选择或保持原值
                if np.random.random() < 0.7:
                    params[name] = base_value
                else:
                    params[name] = np.random.choice(config['values'])
            else:
                # 连续值：在基础值附近添加高斯噪声
                min_val = config['min']
                max_val = config['max']
                
                noise_scale = (max_val - min_val) * 0.1  # 10%的范围作为噪声
                new_value = base_value + np.random.normal(0, noise_scale)
                
                # 确保在有效范围内
                new_value = np.clip(new_value, min_val, max_val)
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = int(round(new_value))
                else:
                    params[name] = new_value
        
        return params
    
    def _evaluate_parameters(self, strategy_module, data, params) -> Tuple[float, Dict[str, Any]]:
        """评估参数组合"""
        # 更新策略模块参数
        strategy_module.update_params(params)
        
        # 运行回测
        backtest_results = strategy_module.backtest(data)
        
        # 计算评估指标
        metrics = strategy_module.evaluate_strategy(backtest_results)
        
        # 计算综合得分
        score = self._calculate_score(metrics)
        
        return score, metrics
    
    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合得分"""
        try:
            # 获取基础指标
            success_rate = metrics.get('success_rate', 0.0)
            avg_rise = metrics.get('avg_rise', 0.0)
            avg_days = metrics.get('avg_days', 20.0)
            total_signals = metrics.get('total_signals', 0)
            
            # 数值安全检查
            success_rate = max(0.0, min(1.0, success_rate))  # 限制在[0,1]范围
            avg_rise = max(0.0, avg_rise)  # 确保非负
            avg_days = max(1.0, avg_days)  # 确保至少1天
            total_signals = max(0, total_signals)  # 确保非负
            
            # 基础得分计算
            success_score = success_rate * self.scoring_weights['success_rate']
            
            # 涨幅得分（相对于基准4%）
            base_rise = 0.04
            if base_rise > 0:
                rise_score = min(avg_rise / base_rise, 2.0) * self.scoring_weights['avg_rise']
            else:
                rise_score = 0.0
            
            # 天数得分（天数越少越好，最大20天）
            max_days = 20.0
            days_score = max(0, (max_days - avg_days) / max_days) * self.scoring_weights['avg_days']
            
            # 风险惩罚（信号数过少惩罚）
            risk_penalty = 0
            min_signals = 10
            if total_signals < min_signals:
                risk_penalty = (min_signals - total_signals) / min_signals * self.scoring_weights['risk_penalty']
            
            # 综合得分
            total_score = success_score + rise_score + days_score - risk_penalty
            
            # 确保得分在合理范围内
            total_score = max(0.0, min(10.0, total_score))
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"计算得分失败: {e}")
            return 0.0
    
    def _save_optimization_result(self, result: Dict[str, Any], method: str):
        """保存优化结果"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{method}_{timestamp}.json"
            file_path = self.get_file_path('results', filename)
            
            # 保存结果
            success = FileManager.safe_json_save(result, file_path)
            
            if success:
                self.logger.info(f"优化结果已保存: {file_path}")
            else:
                self.logger.error(f"保存优化结果失败: {file_path}")
                
            # 更新历史记录（带大小限制）
            self.optimization_history.append(result)
            
            # 清理过多的历史记录
            if len(self.optimization_history) > self.max_history_size:
                # 保留最近的记录
                self.optimization_history = self.optimization_history[-self.max_history_size:]
                self.logger.debug(f"清理优化历史记录，保留最近 {self.max_history_size} 条")
            
            # 更新最佳参数
            if result.get('success') and result.get('best_score', -np.inf) > self.best_score:
                self.best_params = result['best_params']
                self.best_score = result['best_score']
                
        except Exception as e:
            self.logger.error(f"保存优化结果异常: {e}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_history.copy()
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """获取最佳参数"""
        return self.best_params.copy() if self.best_params else None
    
    def get_parameter_ranges(self) -> Dict[str, Any]:
        """获取默认参数范围配置"""
        return self.optimization_config.get('parameter_ranges', {
            'rsi_oversold_threshold': {'min': 25, 'max': 35, 'step': 1},
            'rsi_low_threshold': {'min': 35, 'max': 45, 'step': 1},
            'confidence_threshold': {'min': 0.3, 'max': 0.7, 'step': 0.05},
            'dynamic_confidence_adjustment': {'min': 0.1, 'max': 0.5, 'step': 0.05},
            'market_sentiment_weight': {'min': 0.1, 'max': 0.3, 'step': 0.05},
            'trend_strength_weight': {'min': 0.1, 'max': 0.3, 'step': 0.05}
        })


# 模块导出
__all__ = ['ParameterOptimizer'] 