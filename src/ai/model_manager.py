#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习模型管理模块
负责模型训练、验证、预测、保存和加载等功能
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


class ModelManager:
    """机器学习模型管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型管理器
        
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
        
        # 初始化模型相关属性
        self.model = None
        self.feature_names = None
        
        # 从配置获取模型类型
        ai_config = config.get('ai', {})
        self.model_type = ai_config.get('model_type', 'machine_learning')

    def train_model(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        训练机器学习模型
        
        参数:
        data: 训练数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
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
        验证模型性能
        
        参数:
        data: 验证数据
        strategy_module: 策略模块
        
        返回:
        dict: 验证结果
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
            
            self.logger.info("模型在验证集上评估完成，准确率: %.4f, 精确率: %.4f, 召回率: %.4f, F1: %.4f", 
                           accuracy, precision, recall, f1)
            
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
        根据数据日期计算样本权重，越近的数据权重越高
        
        参数:
        dates: 训练集数据的日期序列
        
        返回:
        numpy.ndarray: 样本权重数组
        """
        self.logger.info("计算样本权重")
        
        weights = np.ones(len(dates))
        if len(dates) == 0:  # Handle empty dates series
            return weights

        latest_date = dates.max()
        
        for i, date in enumerate(dates):
            time_diff = (latest_date - date).days / 365.25  # 年为单位
            # 衰减系数λ，可以根据config配置
            decay_rate = self.config.get("ai", {}).get("data_decay_rate", 0.4)
            weight = np.exp(-decay_rate * time_diff)
            weights[i] = weight
            
        self.logger.info("样本权重计算完成，最大权重: %.4f, 最小权重: %.4f", 
                        np.max(weights), np.min(weights))
        
        return weights

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取模型的特征重要性
        
        返回:
        dict: 特征名称和重要性的字典，按重要性降序排列
        """
        try:
            if self.model is None:
                self.logger.warning("模型未训练，尝试加载已保存的模型")
                if not self._load_model():
                    self.logger.error("无法获取特征重要性：模型未训练且无法加载已保存的模型")
                    return {}
            
            if self.feature_names is None:
                self.logger.error("特征名称未设置，无法获取特征重要性")
                return {}
            
            # 从Pipeline中获取RandomForestClassifier
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    feature_importances = classifier.feature_importances_
                else:
                    self.logger.error("分类器没有feature_importances_属性")
                    return {}
            else:
                self.logger.error("模型没有预期的Pipeline结构")
                return {}
            
            # 创建特征重要性字典
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(feature_importances):
                    importance_dict[feature_name] = float(feature_importances[i])
            
            # 按重要性降序排列
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            self.logger.info("特征重要性获取成功，共 %d 个特征", len(importance_dict))
            
            return importance_dict
            
        except Exception as e:
            self.logger.error("获取特征重要性失败: %s", str(e))
            return {} 