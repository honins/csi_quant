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
        
        self.logger.info("AI优化器初始化完成，模型类型: %s", self.model_type)
        
    def optimize_strategy_parameters(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
        strategy_module: 策略模块实例
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("开始优化策略参数")
        
        try:
            # 1. 获取基准策略的识别结果作为固定标签
            baseline_backtest = strategy_module.backtest(data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            self.logger.info(f"基准策略识别点数: {np.sum(fixed_labels)}")
            
            # 2. 从配置文件获取参数搜索范围
            optimization_config = self.config.get('optimization', {})
            param_ranges = optimization_config.get('param_ranges', {})
            
            # 获取各个参数的搜索范围
            rise_threshold_range = param_ranges.get('rise_threshold', {})
            max_days_range = param_ranges.get('max_days', {})
            rsi_oversold_range = param_ranges.get('rsi_oversold_threshold', {})
            rsi_low_range = param_ranges.get('rsi_low_threshold', {})
            final_threshold_range = param_ranges.get('final_threshold', {})
            
            # 定义参数搜索空间
            param_grid = {
                'rise_threshold': np.arange(
                    rise_threshold_range.get('min', 0.03),
                    rise_threshold_range.get('max', 0.08) + rise_threshold_range.get('step', 0.005),
                    rise_threshold_range.get('step', 0.005)
                ),
                'max_days': np.arange(
                    max_days_range.get('min', 10),
                    max_days_range.get('max', 30) + max_days_range.get('step', 1),
                    max_days_range.get('step', 1)
                ),
                'rsi_oversold_threshold': np.arange(
                    rsi_oversold_range.get('min', 25),
                    rsi_oversold_range.get('max', 35) + rsi_oversold_range.get('step', 1),
                    rsi_oversold_range.get('step', 1)
                ),
                'rsi_low_threshold': np.arange(
                    rsi_low_range.get('min', 35),
                    rsi_low_range.get('max', 45) + rsi_low_range.get('step', 1),
                    rsi_low_range.get('step', 1)
                ),
                'final_threshold': np.arange(
                    final_threshold_range.get('min', 0.3),
                    final_threshold_range.get('max', 0.7) + final_threshold_range.get('step', 0.05),
                    final_threshold_range.get('step', 0.05)
                )
            }
            
            self.logger.info(f"参数搜索范围:")
            for param, values in param_grid.items():
                self.logger.info(f"  {param}: {values[0]} - {values[-1]}, 步长: {values[1]-values[0] if len(values)>1 else 'N/A'}")
            
            best_score = -1
            best_params = None
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            self.logger.info(f"总搜索组合数: {total_combinations}")
            
            # 3. 基于固定标签优化策略参数
            # 为了减少计算量，我们使用随机采样而不是全网格搜索
            max_iterations = min(100, total_combinations)  # 最多100次迭代
            self.logger.info(f"使用随机采样，最大迭代次数: {max_iterations}")
            
            for iteration in range(max_iterations):
                # 随机选择参数组合
                params = {
                    'rise_threshold': np.random.choice(param_grid['rise_threshold']),
                    'max_days': int(np.random.choice(param_grid['max_days'])),
                    'rsi_oversold_threshold': int(np.random.choice(param_grid['rsi_oversold_threshold'])),
                    'rsi_low_threshold': int(np.random.choice(param_grid['rsi_low_threshold'])),
                    'final_threshold': np.random.choice(param_grid['final_threshold'])
                }
                
                # 使用固定标签评估参数
                score = self._evaluate_params_with_fixed_labels_advanced(
                    data, fixed_labels, params
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    self.logger.info(f"发现更好的参数组合 (迭代 {iteration+1}): {best_params}, 得分: {best_score:.4f}")
                        
            self.logger.info("参数优化完成，最佳参数: %s, 得分: %.4f", best_params, best_score)
            return best_params
            
        except Exception as e:
            self.logger.error("优化策略参数失败: %s", str(e))
            # 返回默认参数
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.05),
                'max_days': self.config.get('strategy', {}).get('max_days', 20),
                'rsi_oversold_threshold': self.config.get('strategy', {}).get('confidence_weights', {}).get('rsi_oversold_threshold', 30),
                'rsi_low_threshold': self.config.get('strategy', {}).get('confidence_weights', {}).get('rsi_low_threshold', 40),
                'final_threshold': self.config.get('strategy', {}).get('confidence_weights', {}).get('final_threshold', 0.5)
            }
    
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
        运行遗传算法优化
        
        参数:
        evaluate_func: 评估函数
        population_size: 种群大小
        generations: 迭代代数
        
        返回:
        dict: 最优参数
        """
        self.logger.info("运行遗传算法优化，种群大小: %d, 迭代代数: %d", 
                        population_size, generations)
        
        try:
            # 从配置文件获取参数范围
            optimization_config = self.config.get('optimization', {})
            param_ranges = optimization_config.get('param_ranges', {})
            rise_threshold_range = param_ranges.get('rise_threshold', {})
            max_days_range = param_ranges.get('max_days', {})
            
            # 获取rise_threshold的范围
            min_threshold = rise_threshold_range.get('min', 0.03)
            max_threshold = rise_threshold_range.get('max', 0.08)
            
            # 获取max_days的范围
            min_days = max_days_range.get('min', 10)
            max_days = max_days_range.get('max', 30)
            
            # 初始化种群
            population = []
            for _ in range(population_size):
                individual = {
                    'rise_threshold': np.random.uniform(min_threshold, max_threshold),
                    'max_days': np.random.randint(min_days, max_days + 1)
                }
                population.append(individual)
                
            best_individual = None
            best_score = -1
            
            for generation in range(generations):
                # 评估种群
                scores = []
                for individual in population:
                    score = evaluate_func(individual)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_individual = individual.copy()
                        
                # 选择、交叉、变异
                population = self._genetic_operations(population, scores)
                
                self.logger.info("第 %d 代完成，最佳得分: %.4f", generation + 1, best_score)
                
            self.logger.info("遗传算法优化完成，最佳参数: %s, 得分: %.4f", 
                           best_individual, best_score)
            
            return best_individual
            
        except Exception as e:
            self.logger.error("遗传算法优化失败: %s", str(e))
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.05), 
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
        交叉操作
        
        参数:
        parent1: 父代1
        parent2: 父代2
        
        返回:
        tuple: (子代1, 子代2)
        """
        child1 = {
            'rise_threshold': parent1['rise_threshold'],
            'max_days': parent2['max_days']
        }
        
        child2 = {
            'rise_threshold': parent2['rise_threshold'],
            'max_days': parent1['max_days']
        }
        
        return child1, child2
        
    def _mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """
        变异操作
        
        参数:
        individual: 个体
        mutation_rate: 变异率
        
        返回:
        dict: 变异后的个体
        """
        mutated = individual.copy()
        
        # 从配置文件获取参数范围
        optimization_config = self.config.get('optimization', {})
        param_ranges = optimization_config.get('param_ranges', {})
        rise_threshold_range = param_ranges.get('rise_threshold', {})
        max_days_range = param_ranges.get('max_days', {})
        
        # 获取rise_threshold的范围
        min_threshold = rise_threshold_range.get('min', 0.03)
        max_threshold = rise_threshold_range.get('max', 0.08)
        
        # 获取max_days的范围
        min_days = max_days_range.get('min', 10)
        max_days = max_days_range.get('max', 30)
        
        if np.random.random() < mutation_rate:
            mutated['rise_threshold'] = np.clip(
                mutated['rise_threshold'] + np.random.normal(0, 0.005),
                min_threshold, max_threshold
            )
            
        if np.random.random() < mutation_rate:
            mutated['max_days'] = np.clip(
                int(mutated['max_days'] + np.random.randint(-2, 3)),
                min_days, max_days
            )
            
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
        高级策略参数优化 - 使用多目标优化
        
        参数:
        strategy_module: 策略模块实例
        data: 历史数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("开始高级策略参数优化")
        
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                rise_threshold = params[0]
                
                # 使用固定标签评估
                baseline_strategy = StrategyModule(self.config)
                baseline_backtest = baseline_strategy.backtest(data)
                fixed_labels = baseline_backtest['is_low_point'].astype(int).values
                
                # 从config读取max_days，不参与优化
                max_days = self.config.get('strategy', {}).get('max_days', 20)
                
                score = self._evaluate_params_with_fixed_labels(
                    data, fixed_labels, rise_threshold, max_days
                )
                
                return -score  # 最小化负得分 = 最大化得分
            
            # 约束条件（只优化rise_threshold）
            # 从配置文件获取rise_threshold的范围
            optimization_config = self.config.get('optimization', {})
            param_ranges = optimization_config.get('param_ranges', {})
            rise_threshold_range = param_ranges.get('rise_threshold', {})
            
            min_threshold = rise_threshold_range.get('min', 0.03)
            max_threshold = rise_threshold_range.get('max', 0.08)
            
            bounds = [(min_threshold, max_threshold)]
            
            # 初始值
            x0 = [self.config.get('strategy', {}).get('rise_threshold', 0.05)]
            
            # 优化
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                # 从config读取max_days
                max_days = self.config.get('strategy', {}).get('max_days', 20)
                
                best_params = {
                    'rise_threshold': result.x[0],
                    'max_days': max_days
                }
                best_score = -result.fun
                
                self.logger.info("高级优化完成，最佳参数: %s, 得分: %.4f", best_params, best_score)
                return best_params
            else:
                self.logger.warning("高级优化失败，使用网格搜索")
                return self.optimize_strategy_parameters(strategy_module, data)
                
        except ImportError:
            self.logger.warning("scipy未安装，使用网格搜索")
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
            
            for i in range(cv_folds):
                # 按时间分割数据
                split_point = int(len(data) * (i + 1) / cv_folds)
                train_data = data.iloc[:split_point]
                test_data = data.iloc[split_point:min(split_point + 100, len(data))]  # 测试窗口
                
                if len(test_data) < 20:  # 测试数据太少，跳过
                    self.logger.info(f"   ⏭️ 第{i+1}折：测试数据不足，跳过")
                    continue
                
                self.logger.info(f"   🔄 第{i+1}折：训练数据 {len(train_data)} 条，测试数据 {len(test_data)} 条")
                
                # 在训练数据上优化策略参数
                temp_strategy = StrategyModule(self.config)
                optimized_params = self.optimize_strategy_parameters(temp_strategy, train_data)
                temp_strategy.update_params(optimized_params)
                
                # 在测试数据上评估
                backtest_results = temp_strategy.backtest(test_data)
                evaluation = temp_strategy.evaluate_strategy(backtest_results)
                score = evaluation['score']
                
                fold_scores.append(score)
                total_score += score
                
                self.logger.info(f"   ✅ 第{i+1}折得分: {score:.4f}")
            
            if len(fold_scores) == 0:
                self.logger.warning("⚠️ 没有有效的交叉验证结果")
                return 0.0
                
            avg_score = total_score / len(fold_scores)
            self.logger.info(f"📊 交叉验证完成，平均得分: {avg_score:.4f} (共{len(fold_scores)}折)")
            
            return avg_score
            
        except Exception as e:
            self.logger.error("❌ 时间序列交叉验证失败: %s", str(e))
            return 0.0
    
    def hierarchical_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分层优化策略
        
        参数:
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        self.logger.info("=" * 60)
        self.logger.info("🏗️ 开始分层优化策略")
        self.logger.info("=" * 60)
        
        try:
            # 第一层：策略参数优化
            self.logger.info("📊 第一层：策略参数优化...")
            strategy_module = StrategyModule(self.config)
            strategy_params = self.optimize_strategy_parameters(strategy_module, data)
            self.logger.info("✅ 策略参数优化完成")
            self.logger.info(f"   - 涨幅阈值: {strategy_params['rise_threshold']:.3f}")
            self.logger.info(f"   - 最大观察天数: {strategy_params['max_days']}")
            
            # 第二层：基于优化后的策略训练AI模型
            self.logger.info("🤖 第二层：更新策略参数并准备AI训练...")
            strategy_module.update_params(strategy_params)
            self.logger.info("✅ 策略参数更新完成")
            
            # 第三层：时间序列交叉验证
            self.logger.info("🔄 第三层：时间序列交叉验证...")
            cv_score = self.time_series_cv_evaluation(data, strategy_module)
            self.logger.info(f"✅ 交叉验证完成，平均得分: {cv_score:.4f}")
            
            # 第四层：高级优化（如果可用）
            self.logger.info("🚀 第四层：高级优化...")
            try:
                advanced_params = self.optimize_strategy_parameters_advanced(strategy_module, data)
                advanced_score = self._evaluate_params_with_fixed_labels(
                    data, 
                    strategy_module.backtest(data)['is_low_point'].astype(int).values,
                    advanced_params['rise_threshold'],
                    advanced_params['max_days']
                )
                
                self.logger.info(f"   - 高级优化得分: {advanced_score:.4f}")
                self.logger.info(f"   - 交叉验证得分: {cv_score:.4f}")
                
                # 选择更好的参数
                if advanced_score > cv_score:
                    final_params = advanced_params
                    final_score = advanced_score
                    self.logger.info("✅ 选择高级优化参数（得分更高）")
                else:
                    final_params = strategy_params
                    final_score = cv_score
                    self.logger.info("✅ 选择交叉验证参数（得分更高）")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 高级优化失败，使用基础优化结果: {str(e)}")
                final_params = strategy_params
                final_score = cv_score
                self.logger.info("✅ 使用基础优化参数")
            
            result = {
                'strategy_params': final_params,
                'cv_score': cv_score,
                'final_score': final_score,
                'optimization_method': 'hierarchical'
            }
            
            self.logger.info("=" * 60)
            self.logger.info("🎉 分层优化完成")
            self.logger.info("=" * 60)
            self.logger.info(f"📈 最终参数:")
            self.logger.info(f"   - 涨幅阈值: {final_params['rise_threshold']:.3f}")
            self.logger.info(f"   - 最大观察天数: {final_params['max_days']}")
            self.logger.info(f"   - RSI超卖阈值: {final_params.get('rsi_oversold_threshold', 'N/A')}")
            self.logger.info(f"   - RSI偏低阈值: {final_params.get('rsi_low_threshold', 'N/A')}")
            self.logger.info(f"   - 置信度阈值: {final_params.get('final_threshold', 'N/A')}")
            self.logger.info(f"📊 最终得分: {final_score:.4f}")
            self.logger.info(f"🔧 优化方法: {result['optimization_method']}")
            self.logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.logger.error("❌ 分层优化失败: %s", str(e))
            self.logger.info("🔄 使用默认参数作为备选方案")
            return {
                'strategy_params': {
                    'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.05), 
                    'max_days': self.config.get('strategy', {}).get('max_days', 20)
                },
                'cv_score': 0.0,
                'final_score': 0.0,
                'optimization_method': 'fallback'
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


