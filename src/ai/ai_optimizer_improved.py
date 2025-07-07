#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进版AI优化器
集成增量学习、特征权重优化和趋势确认指标
已废弃置信度平滑功能，直接使用AI模型原始输出
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List, Optional
import json
import yaml

# 注释：以下ConfidenceSmoother类已废弃，不再使用平滑处理
# 现在直接使用模型的原始输出，保持信息完整性


class AIOptimizerImproved:
    """改进版AI优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型相关属性
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(project_root, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = config.get('ai', {}).get('model_type', 'machine_learning')
        
        # 增量学习配置
        incremental_config = config.get('ai', {}).get('incremental_learning', {})
        self.incremental_enabled = incremental_config.get('enabled', True)
        self.retrain_threshold = incremental_config.get('retrain_threshold', 0.1)  # 模型性能下降阈值
        self.max_incremental_updates = incremental_config.get('max_updates', 10)  # 最大增量更新次数
        self.incremental_count = 0
        
        # 移除置信度平滑器 - 使用模型原始输出
        # self.confidence_smoother = ConfidenceSmoother(config)
        
        self.logger.info("改进版AI优化器初始化完成")
    
    def prepare_features_improved(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        改进的特征准备，调整权重和增加趋势确认指标
        
        参数:
        data: 历史数据
        
        返回:
        tuple: (特征矩阵, 特征名称列表)
        """
        self.logger.info("准备改进的机器学习特征")
        
        # 计算额外的趋势确认指标
        data = self._calculate_trend_indicators(data)
        
        # 重新设计特征列，降低短期指标权重，增加趋势确认指标
        feature_columns = [
            # 长期趋势指标（高权重）
            'ma20', 'ma60',  # 长期均线
            'trend_strength_20', 'trend_strength_60',  # 趋势强度
            'price_position_20', 'price_position_60',  # 价格在均线系统中的位置
            
            # 中期趋势指标（中等权重）
            'ma10', 'dist_ma10', 'dist_ma20',
            'rsi', 'macd', 'signal',
            'bb_upper', 'bb_lower',
            'volatility_normalized',  # 标准化波动率
            
            # 短期指标（降低权重）
            'ma5', 'dist_ma5', 'hist',
            'price_change_5d', 'price_change_10d',
            
            # 成交量和波动率（平衡权重）
            'volume_trend', 'volume_strength',  # 成交量趋势
            'volatility'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if len(available_columns) == 0:
            self.logger.warning("没有可用的特征列")
            return np.array([]), []
        
        # 提取特征并应用权重
        features_df = data[available_columns].fillna(0).copy()
        features = self._apply_feature_weights(features_df, available_columns)
        
        self.logger.info("改进特征准备完成，特征数量: %d, 样本数量: %d", 
                        len(available_columns), len(features))
        
        return features, available_columns
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势确认指标
        
        参数:
        data: 原始数据
        
        返回:
        pd.DataFrame: 添加了趋势指标的数据
        """
        # 趋势强度指标（基于线性回归斜率）
        for period in [20, 60]:
            slopes = []
            for i in range(len(data)):
                if i >= period - 1:
                    prices = data['close'].iloc[i-period+1:i+1].values
                    x = np.arange(period)
                    slope = np.polyfit(x, prices, 1)[0]
                    # 标准化斜率
                    normalized_slope = slope / prices.mean()
                    slopes.append(normalized_slope)
                else:
                    slopes.append(0)
            data[f'trend_strength_{period}'] = slopes
        
        # 价格在均线系统中的位置
        data['price_position_20'] = (data['close'] - data['ma20']) / data['ma20']
        data['price_position_60'] = (data['close'] - data['ma60']) / data['ma60']
        
        # 标准化波动率
        data['volatility_normalized'] = data['volatility'] / data['volatility'].rolling(60).mean()
        
        # 成交量趋势指标
        data['volume_ma20'] = data['volume'].rolling(20).mean()
        data['volume_trend'] = (data['volume'] - data['volume_ma20']) / data['volume_ma20']
        
        # 成交量强度（相对于历史）
        data['volume_strength'] = data['volume'] / data['volume'].rolling(60).mean()
        
        return data
    
    def _apply_feature_weights(self, features_df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """
        应用特征权重，降低短期指标影响
        
        参数:
        features_df: 特征数据框
        feature_names: 特征名称列表
        
        返回:
        np.ndarray: 加权后的特征矩阵
        """
        # 定义特征权重
        feature_weights = {
            # 长期趋势指标（高权重）
            'ma20': 1.5, 'ma60': 1.5,
            'trend_strength_20': 2.0, 'trend_strength_60': 2.0,
            'price_position_20': 1.8, 'price_position_60': 1.8,
            
            # 中期指标（正常权重）
            'ma10': 1.0, 'dist_ma10': 1.2, 'dist_ma20': 1.2,
            'rsi': 1.0, 'macd': 1.0, 'signal': 1.0,
            'bb_upper': 1.0, 'bb_lower': 1.0,
            'volatility_normalized': 1.0,
            
            # 短期指标（降低权重）
            'ma5': 0.6, 'dist_ma5': 0.6, 'hist': 0.7,
            'price_change_5d': 0.5, 'price_change_10d': 0.7,
            
            # 成交量指标（平衡权重）
            'volume_trend': 1.1, 'volume_strength': 1.1,
            'volatility': 0.9
        }
        
        # 应用权重
        weighted_features = features_df.copy()
        for feature in feature_names:
            weight = feature_weights.get(feature, 1.0)
            weighted_features[feature] = weighted_features[feature] * weight
        
        return weighted_features.values
    
    def incremental_train(self, new_data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        增量训练模型
        
        参数:
        new_data: 新增数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
        """
        self.logger.info("开始增量训练模型")
        
        try:
            # 检查是否需要完全重训练
            if self.model is None or self.incremental_count >= self.max_incremental_updates:
                self.logger.info("触发完全重训练条件")
                return self.full_train(new_data, strategy_module)
            
            # 准备新数据的特征和标签
            new_features, feature_names = self.prepare_features_improved(new_data)
            new_labels = self._prepare_labels(new_data, strategy_module)
            
            if len(new_features) == 0 or len(new_labels) == 0:
                self.logger.warning("新数据为空，跳过增量训练")
                return {'success': False, 'error': '新数据为空'}
            
            # 检查特征一致性
            if self.feature_names and feature_names != self.feature_names:
                self.logger.warning("特征不一致，进行完全重训练")
                return self.full_train(new_data, strategy_module)
            
            # 使用最近的数据进行增量更新
            recent_features = new_features[-10:]  # 最近10天的数据
            recent_labels = new_labels[-10:]
            
            if len(recent_features) > 0:
                # 对新数据进行标准化（使用已有的scaler）
                if self.scaler is not None:
                    recent_features_scaled = self.scaler.transform(recent_features)
                else:
                    self.logger.warning("缺少scaler，进行完全重训练")
                    return self.full_train(new_data, strategy_module)
                
                # 使用warm_start进行增量学习
                if hasattr(self.model.named_steps['classifier'], 'n_estimators'):
                    classifier = self.model.named_steps['classifier']
                    classifier.n_estimators += 10  # 增加树的数量
                    classifier.warm_start = True
                    
                    # 重新训练（这里实际上是增量的）
                    self.model.named_steps['classifier'].fit(recent_features_scaled, recent_labels)
                    
                    self.incremental_count += 1
                    self.logger.info(f"增量训练完成，更新次数: {self.incremental_count}")
                    
                    return {
                        'success': True,
                        'method': 'incremental',
                        'update_count': self.incremental_count,
                        'new_samples': len(recent_features)
                    }
                else:
                    self.logger.warning("模型不支持增量学习，进行完全重训练")
                    return self.full_train(new_data, strategy_module)
            
            return {'success': False, 'error': '没有足够的新数据进行增量训练'}
            
        except Exception as e:
            self.logger.error(f"增量训练失败: {e}")
            # 失败时进行完全重训练
            return self.full_train(new_data, strategy_module)
    
    def full_train(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        完全重训练模型
        
        参数:
        data: 训练数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
        """
        self.logger.info("开始完全重训练模型")
        
        try:
            # 准备特征和标签
            features, feature_names = self.prepare_features_improved(data)
            labels = self._prepare_labels(data, strategy_module)
            
            if len(features) == 0 or len(labels) == 0:
                self.logger.error("特征或标签为空，无法训练模型")
                return {'success': False, 'error': '特征或标签为空'}
            
            # 数据分割
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            aligned_data = data.iloc[:min_length].copy()
            
            split_ratio = self.config.get("ai", {}).get("train_test_split_ratio", 0.8)
            split_index = int(len(features) * split_ratio)
            
            X_train = features[:split_index]
            y_train = labels[:split_index]
            train_dates = aligned_data["date"].iloc[:split_index]
            
            # 计算样本权重
            sample_weights = self._calculate_sample_weights(train_dates)
            
            # 创建新模型
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # 使用改进的RandomForest参数
            classifier = RandomForestClassifier(
                n_estimators=150,  # 增加树的数量
                max_depth=12,      # 适当增加深度
                min_samples_split=8,  # 增加分割样本数
                min_samples_leaf=3,   # 增加叶子节点样本数
                random_state=42,
                class_weight='balanced',
                warm_start=True,  # 支持增量学习
                n_jobs=-1         # 并行训练
            )
            
            # 训练分类器
            classifier.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            # 创建完整的pipeline
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('classifier', classifier)
            ])
            
            self.feature_names = feature_names
            self.incremental_count = 0  # 重置增量计数
            
            # 保存模型
            self._save_model()
            
            self.logger.info("完全重训练完成")
            return {
                'success': True,
                'method': 'full_retrain',
                'train_samples': len(X_train),
                'feature_count': len(feature_names)
            }
            
        except Exception as e:
            self.logger.error(f"完全重训练失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_low_point(self, data: pd.DataFrame, prediction_date: str = None) -> Dict[str, Any]:
        """
        预测相对低点（带置信度平滑）
        
        参数:
        data: 市场数据
        prediction_date: 预测日期（用于置信度平滑）
        
        返回:
        dict: 预测结果
        """
        self.logger.info("预测相对低点（改进版）")
        
        try:
            # 加载模型（如果未加载）
            if self.model is None:
                if not self._load_model():
                    return {
                        'is_low_point': False,
                        'confidence': 0.0,
                        'smoothed_confidence': 0.0,
                        'error': '模型未训练且无法加载已保存的模型'
                    }
            
            if len(data) == 0:
                return {
                    'is_low_point': False,
                    'confidence': 0.0,
                    'smoothed_confidence': 0.0,
                    'error': '数据为空'
                }
            
            # 准备特征
            features, feature_names = self.prepare_features_improved(data)
            
            if len(features) == 0:
                return {
                    'is_low_point': False,
                    'confidence': 0.0,
                    'final_confidence': 0.0,
                    'error': '无法提取特征'
                }
            
            # 使用最新数据进行预测
            latest_features = features[-1:].reshape(1, -1)
            
            # 获取预测概率（不使用predict方法，避免内置阈值影响）
            prediction_proba = self.model.predict_proba(latest_features)[0]
            
            # 获取原始置信度（不再进行平滑处理）
            raw_confidence = prediction_proba[1] if len(prediction_proba) > 1 else 0.0
            
            # 直接使用原始置信度，不进行平滑处理
            final_confidence = raw_confidence
            
            # 使用配置的阈值和原始置信度进行最终预测
            confidence_config = self.config.get('strategy', {}).get('confidence_weights', {})
            final_threshold = confidence_config.get('final_threshold', 0.5)
            
            # 基于原始置信度和配置阈值进行预测
            is_low_point = final_confidence >= final_threshold
            
            result = {
                'is_low_point': bool(is_low_point),
                'confidence': float(raw_confidence),
                'final_confidence': float(final_confidence),  # 现在等于原始置信度
                'prediction_proba': prediction_proba.tolist(),
                'feature_count': len(feature_names),
                'model_type': type(self.model.named_steps['classifier']).__name__,
                'threshold_used': final_threshold
            }
            
            # 输出预测结果
            self.logger.info("----------------------------------------------------")
            self.logger.info("AI预测结果（无平滑）: \033[1m%s\033[0m", 
                           "相对低点" if is_low_point else "非相对低点")
            self.logger.info("原始置信度: \033[1m%.4f\033[0m, 阈值: \033[1m%.2f\033[0m", 
                           raw_confidence, final_threshold)
            self.logger.info("----------------------------------------------------")
            
            return result
            
        except Exception as e:
            self.logger.error(f"预测相对低点失败: {e}")
            return {
                'is_low_point': False,
                'confidence': 0.0,
                'final_confidence': 0.0,
                'error': str(e)
            }
    
    def _prepare_labels(self, data: pd.DataFrame, strategy_module) -> np.ndarray:
        """准备标签"""
        backtest_results = strategy_module.backtest(data)
        return backtest_results['is_low_point'].astype(int).values
    
    def _calculate_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """计算样本权重"""
        weights = np.ones(len(dates))
        if len(dates) == 0:
            return weights
        
        latest_date = dates.max()
        decay_rate = self.config.get("ai", {}).get("data_decay_rate", 0.4)
        
        for i, date in enumerate(dates):
            time_diff = (latest_date - date).days / 365.25
            weight = np.exp(-decay_rate * time_diff)
            weights[i] = weight
        
        return weights
    
    def _save_model(self) -> bool:
        """保存模型"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存模型
            model_path = os.path.join(self.models_dir, f'improved_model_{timestamp}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'incremental_count': self.incremental_count,
                    'scaler': self.scaler
                }, f)
            
            # 保存最新模型路径
            latest_path = os.path.join(self.models_dir, 'latest_improved_model.txt')
            with open(latest_path, 'w') as f:
                f.write(model_path)
            
            self.logger.info(f"改进模型保存成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存改进模型失败: {e}")
            return False
    
    def _load_model(self) -> bool:
        """加载模型（安全版本）"""
        try:
            latest_path = os.path.join(self.models_dir, 'latest_improved_model.txt')
            
            if not os.path.exists(latest_path):
                self.logger.warning("没有找到已保存的改进模型")
                return False
            
            with open(latest_path, 'r') as f:
                model_path = f.read().strip()
            
            # 安全检查：验证模型文件路径
            if not os.path.abspath(model_path).startswith(os.path.abspath(self.models_dir)):
                self.logger.error(f"模型文件路径不安全: {model_path}")
                return False
            
            # 安全检查：验证文件大小（防止过大的恶意文件）
            max_file_size = 500 * 1024 * 1024  # 500MB限制
            if os.path.getsize(model_path) > max_file_size:
                self.logger.error(f"模型文件过大，可能存在安全风险: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                # 使用受限的pickle加载器（限制可导入的模块）
                import pickle
                import builtins
                
                # 创建安全的unpickler
                logger = self.logger  # 保存logger引用
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # 只允许加载特定的安全模块
                        safe_modules = {
                            'sklearn.ensemble._forest',
                            'sklearn.ensemble',
                            'sklearn.pipeline',
                            'sklearn.preprocessing._data',
                            'sklearn.preprocessing',
                            'numpy',
                            'pandas.core.frame',
                            'pandas',
                            'builtins'
                        }
                        
                        if module in safe_modules or module.startswith('numpy') or module.startswith('sklearn'):
                            return getattr(__import__(module, fromlist=[name]), name)
                        else:
                            logger.warning(f"拒绝加载不安全的模块: {module}.{name}")
                            raise pickle.PicklingError(f"Unsafe module: {module}")
                
                # 使用安全的unpickler加载数据
                safe_unpickler = SafeUnpickler(f)
                data = safe_unpickler.load()
                
                # 验证加载的数据结构
                required_keys = ['model', 'feature_names']
                if not isinstance(data, dict) or not all(key in data for key in required_keys):
                    self.logger.error("模型文件格式不正确")
                    return False
                
                self.model = data['model']
                self.feature_names = data['feature_names']
                self.incremental_count = data.get('incremental_count', 0)
                self.scaler = data.get('scaler')
            
            self.logger.info(f"改进模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载改进模型失败: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        返回:
        dict: 特征重要性字典，按重要性降序排列
        """
        try:
            if self.model is None:
                self.logger.warning("模型未训练，尝试加载已保存的模型")
                if not self._load_model():
                    self.logger.error("无法获取特征重要性：模型未训练且无法加载")
                    return {}
            
            if self.feature_names is None:
                self.logger.error("特征名称未设置，无法获取特征重要性")
                return {}
            
            # 从Pipeline中获取分类器
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
            else:
                # 如果模型不是Pipeline，直接使用
                classifier = self.model
            
            # 检查分类器是否有feature_importances_属性
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # 创建特征重要性字典
                feature_importance = dict(zip(self.feature_names, importances))
                
                # 按重要性降序排列
                sorted_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
                self.logger.info(f"成功获取 {len(sorted_importance)} 个特征的重要性")
                return sorted_importance
            else:
                self.logger.warning(f"分类器 {type(classifier).__name__} 不支持特征重要性")
                return {}
                
        except Exception as e:
            self.logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def run_complete_optimization(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        运行完整的AI优化流程（包含参数优化 + 模型训练）
        
        参数:
        data: 历史数据
        strategy_module: 策略模块
        
        返回:
        dict: 优化结果
        """
        self.logger.info("🚀 开始完整的AI优化流程（改进版）")
        
        try:
            optimization_result = {
                'success': False,
                'strategy_optimization': {},
                'model_training': {},
                'final_evaluation': {},
                'errors': []
            }
            
            # 1. 策略参数优化
            self.logger.info("🔧 步骤1: 策略参数优化")
            strategy_result = self.optimize_strategy_parameters_improved(strategy_module, data)
            optimization_result['strategy_optimization'] = strategy_result
            
            if strategy_result['success']:
                # 更新策略模块参数
                strategy_module.update_params(strategy_result['best_params'])
                self.logger.info(f"✅ 策略参数优化完成: {strategy_result['best_params']}")
            else:
                self.logger.warning("⚠️ 策略参数优化失败，使用默认参数继续")
                optimization_result['errors'].append("策略参数优化失败")
            
            # 2. 改进版模型训练
            self.logger.info("🤖 步骤2: 改进版模型训练")
            model_result = self.full_train(data, strategy_module)
            optimization_result['model_training'] = model_result
            
            if not model_result['success']:
                self.logger.error("❌ 模型训练失败")
                optimization_result['errors'].append("模型训练失败")
                return optimization_result
            
            # 3. 最终评估
            self.logger.info("📊 步骤3: 最终性能评估")
            evaluation_result = self.evaluate_optimized_system(data, strategy_module)
            optimization_result['final_evaluation'] = evaluation_result
            
            # 4. 保存优化结果
            if strategy_result['success']:
                self.save_optimized_params(strategy_result['best_params'])
            
            optimization_result['success'] = model_result['success']
            
            self.logger.info("🎉 完整AI优化流程完成")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"完整AI优化流程失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_optimization': {},
                'model_training': {},
                'final_evaluation': {}
            }
    
    def optimize_strategy_parameters_improved(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        改进版策略参数优化（使用严格三层数据分割）
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        self.logger.info("开始改进版策略参数优化（严格三层分割）")
        
        try:
            # 检查是否有足够的数据
            if len(data) < 100:
                return {
                    'success': False,
                    'error': '数据量不足，无法进行参数优化'
                }
            
            # 从配置文件获取三层数据分割比例
            validation_config = self.config.get('ai', {}).get('validation', {})
            train_ratio = validation_config.get('train_ratio', 0.65)
            val_ratio = validation_config.get('validation_ratio', 0.2) 
            test_ratio = validation_config.get('test_ratio', 0.15)
            
            # 验证比例总和
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                self.logger.warning(f"数据分割比例总和不等于1.0: {total_ratio:.3f}，自动调整")
                # 重新归一化
                train_ratio = train_ratio / total_ratio
                val_ratio = val_ratio / total_ratio 
                test_ratio = test_ratio / total_ratio
            
            # 计算分割点
            train_end = int(len(data) * train_ratio)
            val_end = int(len(data) * (train_ratio + val_ratio))
            
            # 严格三层数据分割
            train_data = data.iloc[:train_end].copy()
            validation_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
            
            self.logger.info(f"严格三层数据分割:")
            self.logger.info(f"   📊 训练集: {len(train_data)}条 ({train_ratio:.1%}) - 仅用于参数优化")
            self.logger.info(f"   📈 验证集: {len(validation_data)}条 ({val_ratio:.1%}) - 用于模型验证和过拟合检测")
            self.logger.info(f"   🔒 测试集: {len(test_data)}条 ({test_ratio:.1%}) - 完全锁定，仅最终评估")
            
            # 定义优化参数范围
            param_ranges = self.config.get('optimization', {}).get('param_ranges', {})
            
            # 使用网格搜索进行参数优化（仅在训练集上）
            self.logger.info("🔧 步骤1: 在训练集上进行参数搜索...")
            best_params, best_score = self._grid_search_optimization(
                strategy_module, train_data, param_ranges
            )
            
            # 在验证集上验证最佳参数
            self.logger.info("📈 步骤2: 在验证集上验证最佳参数...")
            strategy_module.update_params(best_params)
            val_backtest = strategy_module.backtest(validation_data)
            val_evaluation = strategy_module.evaluate_strategy(val_backtest)
            val_score = val_evaluation['score']
            val_success_rate = val_evaluation.get('success_rate', 0)
            val_total_points = val_evaluation.get('total_points', 0)
            val_avg_rise = val_evaluation.get('avg_rise', 0)
            
            # 检查过拟合
            overfitting_threshold = 0.8  # 验证集得分应该至少是训练集得分的80%
            overfitting_passed = val_score >= best_score * overfitting_threshold
            
            # 在完全锁定的测试集上进行最终评估
            self.logger.info("🔒 步骤3: 在测试集上进行最终评估...")
            test_backtest = strategy_module.backtest(test_data)
            test_evaluation = strategy_module.evaluate_strategy(test_backtest)
            test_score = test_evaluation['score']
            test_success_rate = test_evaluation.get('success_rate', 0)
            test_total_points = test_evaluation.get('total_points', 0)
            test_avg_rise = test_evaluation.get('avg_rise', 0)
            
            # 评估模型泛化能力
            generalization_ratio = test_score / val_score if val_score > 0 else 0
            generalization_passed = generalization_ratio >= 0.85  # 测试集得分应该接近验证集
            
            self.logger.info(f"✅ 三层验证结果:")
            self.logger.info(f"   📊 训练集得分: {best_score:.4f}")
            self.logger.info(f"   📈 验证集得分: {val_score:.4f} | 成功率: {val_success_rate:.2%} | 识别点数: {val_total_points} | 平均涨幅: {val_avg_rise:.2%}")
            self.logger.info(f"   🔒 测试集得分: {test_score:.4f} | 成功率: {test_success_rate:.2%} | 识别点数: {test_total_points} | 平均涨幅: {test_avg_rise:.2%}")
            self.logger.info(f"   🛡️ 过拟合检测: {'✅ 通过' if overfitting_passed else '⚠️ 警告'}")
            self.logger.info(f"   🎯 泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")
            
            return {
                'success': True,
                'best_params': best_params,
                'best_score': best_score,
                'validation_score': val_score,
                'validation_success_rate': val_success_rate,
                'validation_total_points': val_total_points,
                'validation_avg_rise': val_avg_rise,
                'test_score': test_score,
                'test_success_rate': test_success_rate,
                'test_total_points': test_total_points,
                'test_avg_rise': test_avg_rise,
                'overfitting_passed': overfitting_passed,
                'generalization_passed': generalization_passed,
                'generalization_ratio': generalization_ratio,
                'optimization_method': 'grid_search_improved_3layer',
                'data_split': {
                    'train_ratio': train_ratio,
                    'validation_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'train_samples': len(train_data),
                    'validation_samples': len(validation_data),
                    'test_samples': len(test_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"策略参数优化失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _grid_search_optimization(self, strategy_module, train_data: pd.DataFrame, param_ranges: dict) -> tuple:
        """
        网格搜索优化
        
        参数:
        strategy_module: 策略模块
        train_data: 训练数据
        param_ranges: 参数范围
        
        返回:
        tuple: (最佳参数, 最佳得分)
        """
        self.logger.info("开始网格搜索优化")
        
        # 定义默认参数范围
        default_ranges = {
            'rsi_oversold_threshold': {'min': 25, 'max': 35, 'step': 2},
            'rsi_low_threshold': {'min': 35, 'max': 45, 'step': 2},
            'final_threshold': {'min': 0.3, 'max': 0.7, 'step': 0.1}
        }
        
        # 合并用户配置和默认配置
        search_ranges = {**default_ranges, **param_ranges}
        
        # 生成参数组合
        param_combinations = []
        param_names = list(search_ranges.keys())
        
        def generate_range(param_config):
            start = param_config['min']
            end = param_config['max']
            step = param_config['step']
            return [start + i * step for i in range(int((end - start) / step) + 1)]
        
        # 生成所有参数组合（限制数量以避免过长时间）
        from itertools import product
        
        ranges = [generate_range(search_ranges[param]) for param in param_names]
        all_combinations = list(product(*ranges))
        
        # 限制搜索数量
        max_combinations = 50
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            all_combinations = random.sample(all_combinations, max_combinations)
        
        self.logger.info(f"将测试 {len(all_combinations)} 个参数组合")
        
        best_score = -float('inf')
        best_params = {}
        
        for i, combination in enumerate(all_combinations):
            # 构建参数字典
            params = dict(zip(param_names, combination))
            
            try:
                # 更新参数并测试
                strategy_module.update_params(params)
                backtest_results = strategy_module.backtest(train_data)
                evaluation = strategy_module.evaluate_strategy(backtest_results)
                score = evaluation['score']
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已测试 {i + 1}/{len(all_combinations)} 个组合，当前最佳得分: {best_score:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"参数组合 {params} 测试失败: {e}")
                continue
        
        self.logger.info(f"网格搜索完成，最佳得分: {best_score:.4f}")
        return best_params, best_score
    
    def evaluate_optimized_system(self, data: pd.DataFrame, strategy_module) -> Dict[str, Any]:
        """
        评估优化后的系统
        
        参数:
        data: 测试数据
        strategy_module: 策略模块
        
        返回:
        dict: 评估结果
        """
        self.logger.info("评估优化后的系统")
        
        try:
            # 策略评估
            backtest_results = strategy_module.backtest(data)
            strategy_evaluation = strategy_module.evaluate_strategy(backtest_results)
            
            # AI模型预测评估
            prediction_result = self.predict_low_point(data)
            
            return {
                'success': True,
                'strategy_score': strategy_evaluation['score'],
                'strategy_success_rate': strategy_evaluation['success_rate'],
                'identified_points': strategy_evaluation['total_points'],
                'avg_rise': strategy_evaluation['avg_rise'],
                'ai_confidence': prediction_result.get('final_confidence', 0),
                'ai_prediction': prediction_result.get('is_low_point', False)
            }
            
        except Exception as e:
            self.logger.error(f"系统评估失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_optimized_params(self, params: dict):
        """
        保存优化后的参数到配置文件（保留注释版）
        
        参数:
        params: 优化后的参数
        """
        try:
            # 尝试使用保留注释的保存器
            try:
                from src.utils.config_saver import ConfigSaver
                saver = ConfigSaver()
                saver.save_optimized_params(params)
                self.logger.info("参数已保存（保留注释版本）")
                return
            except Exception as e:
                self.logger.warning(f"保留注释版本保存失败，使用传统方式: {e}")
                self._save_params_fallback(params)
        except Exception as e:
            self.logger.error(f"保存优化参数失败: {e}")
    
    def _save_params_fallback(self, params: dict):
        """
        传统的参数保存方式（备用方案）
        
        参数:
        params: 优化后的参数
        """
        try:
            config_path = 'config/config_improved.yaml'
            
            # 读取现有配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 更新参数
            for param_name, param_value in params.items():
                if param_name == 'final_threshold':
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    if 'confidence_weights' not in config['strategy']:
                        config['strategy']['confidence_weights'] = {}
                    config['strategy']['confidence_weights']['final_threshold'] = param_value
                elif param_name in ['rsi_oversold_threshold', 'rsi_low_threshold']:
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    if 'confidence_weights' not in config['strategy']:
                        config['strategy']['confidence_weights'] = {}
                    config['strategy']['confidence_weights'][param_name] = param_value
                else:
                    # 其他参数按原有逻辑处理
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    config['strategy'][param_name] = param_value
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info("参数已保存到配置文件（传统方式）")
            
        except Exception as e:
            self.logger.error(f"传统方式保存参数失败: {e}") 