#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化器模块
负责使用AI技术优化策略参数
"""

import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 导入策略模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategy'))
from strategy_module import StrategyModule


class AIOptimizer:
    """AI优化器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化AI优化器
        
        参数:
        config: 配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置模型目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(project_root, 'models')
        
        # 确保模型目录存在
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # 设置参数历史记录文件路径
        cache_dir = os.path.join(project_root, 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.parameter_history_file = os.path.join(cache_dir, 'parameter_history.json')
        self.best_parameters_file = os.path.join(cache_dir, 'best_parameters.json')
        
        # 初始化模型相关属性
        self.model = None
        self.feature_names = None
        
        # 从配置获取模型类型
        ai_config = config.get('ai', {})
        self.model_type = ai_config.get('model_type', 'machine_learning')
        
        self.logger.info("AI优化器初始化完成，模型类型: %s", self.model_type)

    def strict_data_split(self, data: pd.DataFrame, preserve_test_set: bool = True) -> Dict[str, pd.DataFrame]:
        """
        严格的数据分割，防止过拟合
        
        参数:
        data: 输入数据
        preserve_test_set: 是否保护测试集
        
        返回:
        dict: 包含训练集、验证集、测试集的字典
        """
        self.logger.info("🔒 开始严格数据分割...")
        
        try:
            # 从配置获取分割比例
            ai_config = self.config.get('ai', {})
            validation_config = ai_config.get('validation', {})
            
            train_ratio = validation_config.get('train_ratio', 0.65)
            validation_ratio = validation_config.get('validation_ratio', 0.20)
            test_ratio = validation_config.get('test_ratio', 0.15)
            
            # 确保比例和为1
            total_ratio = train_ratio + validation_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.001:
                self.logger.warning(f"分割比例总和不为1: {total_ratio:.3f}，进行归一化")
                train_ratio /= total_ratio
                validation_ratio /= total_ratio
                test_ratio /= total_ratio
            
            # 计算分割点
            n = len(data)
            train_end = int(n * train_ratio)
            validation_end = int(n * (train_ratio + validation_ratio))
            
            # 时间序列分割（保持时间顺序）
            train_data = data.iloc[:train_end].copy()
            validation_data = data.iloc[train_end:validation_end].copy()
            test_data = data.iloc[validation_end:].copy()
            
            # 检测数据泄露
            train_dates = set(pd.to_datetime(train_data['date']).dt.date)
            test_dates = set(pd.to_datetime(test_data['date']).dt.date)
            overlap = train_dates.intersection(test_dates)
            
            if overlap:
                self.logger.warning(f"❌ 检测到数据泄露：{len(overlap)}个重复日期")
                for date in list(overlap)[:5]:  # 只显示前5个
                    self.logger.warning(f"   重复日期: {date}")
            else:
                self.logger.info("✅ 数据泄露检测通过，无重复日期")
            
            self.logger.info("📊 数据分割完成:")
            self.logger.info(f"   - 训练集: {len(train_data)} 条 ({len(train_data)/n:.1%})")
            self.logger.info(f"   - 验证集: {len(validation_data)} 条 ({len(validation_data)/n:.1%})")
            self.logger.info(f"   - 测试集: {len(test_data)} 条 ({len(test_data)/n:.1%})")
            
            return {
                'train': train_data,
                'validation': validation_data,
                'test': test_data
            }
            
        except Exception as e:
            self.logger.error(f"❌ 严格数据分割失败: {str(e)}")
            raise

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
        self.logger.info("🚶 开始走前验证...")
        
        try:
            scores = []
            windows = []
            
            # 计算总窗口数
            total_windows = max(1, (len(data) - window_size) // step_size)
            self.logger.info(f"总验证窗口数: {total_windows}")
            
            for i in range(total_windows):
                start_idx = i * step_size
                train_end_idx = start_idx + window_size
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + step_size, len(data))
                
                if test_end_idx <= test_start_idx:
                    continue
                
                # 分割数据
                train_window = data.iloc[start_idx:train_end_idx].copy()
                test_window = data.iloc[test_start_idx:test_end_idx].copy()
                
                self.logger.info(f"窗口 {i+1}/{total_windows}: 训练 {len(train_window)} 条, 测试 {len(test_window)} 条")
                
                # 在训练窗口上优化参数
                optimized_params = self.optimize_strategy_parameters_on_train_only(strategy_module, train_window)
                
                # 更新策略参数
                temp_strategy = StrategyModule(self.config)
                temp_strategy.update_params(optimized_params)
                
                # 在测试窗口上评估
                test_results = temp_strategy.backtest(test_window)
                evaluation = temp_strategy.evaluate_strategy(test_results)
                
                score = evaluation['score']
                scores.append(score)
                windows.append({
                    'window': i + 1,
                    'train_start': train_window.iloc[0]['date'],
                    'train_end': train_window.iloc[-1]['date'],
                    'test_start': test_window.iloc[0]['date'],
                    'test_end': test_window.iloc[-1]['date'],
                    'score': score
                })
                
                self.logger.info(f"窗口 {i+1} 得分: {score:.4f}")
            
            if not scores:
                self.logger.warning("没有有效的验证窗口")
                return {'success': False, 'error': '没有有效的验证窗口'}
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            self.logger.info("✅ 走前验证完成")
            self.logger.info(f"平均得分: {avg_score:.4f} ± {std_score:.4f}")
            
            return {
                'success': True,
                'avg_score': avg_score,
                'std_score': std_score,
                'all_scores': scores,
                'windows': windows
            }
            
        except Exception as e:
            self.logger.error(f"❌ 走前验证失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def optimize_strategy_parameters_on_train_only(self, strategy_module, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        仅在训练集上优化策略参数
        
        参数:
        strategy_module: 策略模块
        train_data: 训练数据
        
        返回:
        dict: 优化后的参数
        """
        self.logger.info("🎯 开始训练集策略参数优化...")
        
        try:
            # 固定核心参数
            fixed_rise_threshold = self.config.get('strategy', {}).get('rise_threshold', 0.04)
            fixed_max_days = self.config.get('strategy', {}).get('max_days', 20)
            
            # 获取基准策略识别结果
            baseline_backtest = strategy_module.backtest(train_data)
            fixed_labels = baseline_backtest['is_low_point'].astype(int).values
            
            # 参数搜索范围
            param_ranges = {
                'rsi_oversold_threshold': np.arange(25, 36, 1),
                'rsi_low_threshold': np.arange(35, 46, 1),
                'final_threshold': np.arange(0.3, 0.71, 0.05)
            }
            
            best_score = -1
            best_params = None
            
            # 简化搜索（仅针对训练集）
            max_iterations = 50
            
            for i in range(max_iterations):
                params = {
                    'rise_threshold': fixed_rise_threshold,
                    'max_days': fixed_max_days,
                    'rsi_oversold_threshold': int(np.random.choice(param_ranges['rsi_oversold_threshold'])),
                    'rsi_low_threshold': int(np.random.choice(param_ranges['rsi_low_threshold'])),
                    'final_threshold': np.random.choice(param_ranges['final_threshold'])
                }
                
                score = self._evaluate_params_with_fixed_labels(
                    train_data, fixed_labels, 
                    params['rise_threshold'], params['max_days']
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            self.logger.info(f"✅ 训练集优化完成，最佳得分: {best_score:.4f}")
            
            return best_params if best_params else {
                'rise_threshold': fixed_rise_threshold,
                'max_days': fixed_max_days,
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5
            }
            
        except Exception as e:
            self.logger.error(f"❌ 训练集优化失败: {str(e)}")
            return {
                'rise_threshold': self.config.get('strategy', {}).get('rise_threshold', 0.04),
                'max_days': self.config.get('strategy', {}).get('max_days', 20),
                'rsi_oversold_threshold': 30,
                'rsi_low_threshold': 40,
                'final_threshold': 0.5
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
            
            return np.mean(scores) if scores else 0.0
            
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
                'test_size': len(test_data)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 测试集评估失败: {str(e)}")
            return {'success': False, 'error': str(e)}

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