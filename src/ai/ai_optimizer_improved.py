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
from itertools import product
import sys
import time

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
        完整训练改进版AI模型
        
        参数:
        data: 历史数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
        """
        train_start_time = time.time()
        self.logger.info("🤖 开始改进版AI模型完整训练")
        self.logger.info("=" * 80)
        
        try:
            # 步骤1: 特征工程
            self.logger.info("⚙️ 步骤1: 特征工程...")
            feature_start_time = time.time()
            
            features, feature_names = self.prepare_features_improved(data)
            
            if len(features) == 0:
                return {
                    'success': False,
                    'error': '无法提取特征'
                }
            
            self.feature_names = feature_names
            feature_time = time.time() - feature_start_time
            
            self.logger.info(f"✅ 特征工程完成 (耗时: {feature_time:.2f}s)")
            self.logger.info(f"   特征数量: {len(feature_names)}")
            self.logger.info(f"   样本数量: {len(features)}")
            self.logger.info(f"   特征列表: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")
            self.logger.info("-" * 60)
            
            # 步骤2: 标签准备
            self.logger.info("🏷️ 步骤2: 标签准备...")
            label_start_time = time.time()
            
            labels = self._prepare_labels(data, strategy_module)
            
            if len(labels) != len(features):
                return {
                    'success': False,
                    'error': f'特征数量({len(features)})与标签数量({len(labels)})不匹配'
                }
            
            label_time = time.time() - label_start_time
            positive_ratio = np.mean(labels)
            
            self.logger.info(f"✅ 标签准备完成 (耗时: {label_time:.2f}s)")
            self.logger.info(f"   正样本比例: {positive_ratio:.2%}")
            self.logger.info(f"   正样本数量: {np.sum(labels)} / {len(labels)}")
            self.logger.info("-" * 60)
            
            # 步骤3: 样本权重计算
            self.logger.info("⚖️ 步骤3: 样本权重计算...")
            weight_start_time = time.time()
            
            # 严格要求数据包含正确的日期信息
            if 'date' in data.columns:
                date_series = data['date']
                if not pd.api.types.is_datetime64_any_dtype(date_series):
                    raise ValueError(f"data['date']列不是datetime类型，实际类型: {date_series.dtype}")
            elif pd.api.types.is_datetime64_any_dtype(data.index):
                date_series = data.index.to_series()
            else:
                raise ValueError(
                    "数据缺少有效的日期信息。要求：\n"
                    "1. 包含datetime类型的'date'列，或\n"
                    "2. 使用datetime类型的索引\n"
                    f"实际情况：\n"
                    f"  - 'date'列: {'存在' if 'date' in data.columns else '不存在'}\n"
                    f"  - 索引类型: {type(data.index).__name__}\n"
                    f"  - 索引dtype: {data.index.dtype}"
                )
            
            sample_weights = self._calculate_sample_weights(date_series)
            weight_time = time.time() - weight_start_time
            
            self.logger.info(f"✅ 样本权重计算完成 (耗时: {weight_time:.2f}s)")
            self.logger.info(f"   权重范围: {np.min(sample_weights):.4f} - {np.max(sample_weights):.4f}")
            self.logger.info(f"   平均权重: {np.mean(sample_weights):.4f}")
            self.logger.info("-" * 60)
            
            # 步骤4: 模型训练
            self.logger.info("🏋️ 步骤4: 模型训练...")
            model_start_time = time.time()
            
            # 创建改进的模型pipeline（降低复杂度防止过拟合）
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,      # 从150降到100
                    max_depth=8,           # 从12降到8
                    min_samples_split=15,  # 从8提高到15
                    min_samples_leaf=8,    # 从3提高到8
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1  # 启用训练进度输出
                ))
            ])
            
            self.logger.info("🌲 RandomForest模型配置（防过拟合）:")
            self.logger.info("   n_estimators: 100 (决策树数量) - 降低复杂度")
            self.logger.info("   max_depth: 8 (最大深度) - 减少过拟合")
            self.logger.info("   min_samples_split: 15 (最小分割样本数) - 增加稳定性")
            self.logger.info("   min_samples_leaf: 8 (最小叶子节点样本数) - 增加稳定性")
            self.logger.info("   class_weight: balanced (自动平衡类别权重)")
            self.logger.info("   n_jobs: -1 (并行训练)")
            
            self.logger.info("🚀 开始模型训练...")
            # 训练模型
            model.fit(features, labels, classifier__sample_weight=sample_weights)
            
            model_time = time.time() - model_start_time
            self.model = model
            
            self.logger.info(f"✅ 模型训练完成 (耗时: {model_time:.2f}s)")
            self.logger.info("-" * 60)
            
            # 步骤5: 模型保存
            self.logger.info("💾 步骤5: 模型保存...")
            save_start_time = time.time()
            
            save_success = self._save_model()
            save_time = time.time() - save_start_time
            
            if save_success:
                self.logger.info(f"✅ 模型保存成功 (耗时: {save_time:.2f}s)")
            else:
                self.logger.warning(f"⚠️ 模型保存失败 (耗时: {save_time:.2f}s)")
            
            # 训练总结
            total_train_time = time.time() - train_start_time
            self.logger.info("=" * 80)
            self.logger.info("🎉 改进版AI模型训练完成!")
            self.logger.info(f"⏱️ 总耗时: {total_train_time:.2f}s ({total_train_time/60:.1f}分钟)")
            self.logger.info(f"📊 训练统计:")
            self.logger.info(f"   特征工程: {feature_time:.2f}s")
            self.logger.info(f"   标签准备: {label_time:.2f}s")
            self.logger.info(f"   权重计算: {weight_time:.2f}s")
            self.logger.info(f"   模型训练: {model_time:.2f}s")
            self.logger.info(f"   模型保存: {save_time:.2f}s")
            self.logger.info(f"🎯 训练结果:")
            self.logger.info(f"   样本数量: {len(features)}")
            self.logger.info(f"   特征数量: {len(feature_names)}")
            self.logger.info(f"   正样本比例: {positive_ratio:.2%}")
            self.logger.info(f"   模型保存: {'成功' if save_success else '失败'}")
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'method': 'improved_full_training',
                'train_samples': len(features),
                'feature_count': len(feature_names),
                'positive_ratio': positive_ratio,
                'training_time': total_train_time,
                'feature_time': feature_time,
                'model_time': model_time,
                'save_time': save_time,
                'save_success': save_success,
                'feature_names': feature_names
            }
            
        except Exception as e:
            self.logger.error(f"改进版模型训练失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
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
        """
        计算基于时间衰减的样本权重
        
        参数:
        dates: 日期序列，必须是datetime类型
        
        返回:
        np.ndarray: 样本权重数组，越新的数据权重越高
        
        异常:
        ValueError: 当dates不是datetime类型或为空时
        """
        if len(dates) == 0:
            raise ValueError("日期序列为空，无法计算样本权重")
        
        # 严格检查数据类型：只接受datetime类型
        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError(f"样本权重计算要求datetime类型的日期数据，实际类型: {dates.dtype}")
        
        # 检查是否有空值
        if dates.isnull().any():
            raise ValueError("日期序列包含空值，无法计算准确的样本权重")
        
        # 计算时间衰减权重
        latest_date = dates.max()
        decay_rate = self.config.get("ai", {}).get("data_decay_rate", 0.4)
        
        weights = np.zeros(len(dates))
        for i, date in enumerate(dates):
            time_diff_days = (latest_date - date).days
            if time_diff_days < 0:
                raise ValueError(f"发现未来日期: {date} > {latest_date}")
            
            time_diff_years = time_diff_days / 365.25
            weight = np.exp(-decay_rate * time_diff_years)
            weights[i] = weight
        
        # 验证权重计算结果
        if np.any(weights <= 0) or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("样本权重计算结果异常，包含非正值或NaN/Inf")
        
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
        from datetime import datetime
        complete_start_time = time.time()
        
        # 同时使用print和logger确保输出可见
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"🚀 开始完整的AI优化流程（改进版） [{current_time}]")
        print("=" * 80)
        self.logger.info("🚀 开始完整的AI优化流程（改进版）")
        self.logger.info("=" * 80)
        
        try:
            optimization_result = {
                'success': False,
                'strategy_optimization': {},
                'model_training': {},
                'final_evaluation': {},
                'errors': []
            }
            
            # 步骤预览
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"📋 优化流程概览: [{current_time}]")
            print("   🔧 步骤A: 策略参数优化 (遗传算法/网格搜索)")
            print("   🤖 步骤B: 改进版模型训练")  
            print("   📊 步骤C: 最终性能评估")
            print("   💾 步骤D: 结果保存")
            print("-" * 80)
            
            self.logger.info("📋 优化流程概览:")
            self.logger.info("   🔧 步骤A: 策略参数优化 (遗传算法/网格搜索)")
            self.logger.info("   🤖 步骤B: 改进版模型训练")  
            self.logger.info("   📊 步骤C: 最终性能评估")
            self.logger.info("   💾 步骤D: 结果保存")
            self.logger.info("-" * 80)
            
            # 步骤A: 策略参数优化
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🔧 步骤A: 策略参数优化 [{current_time}]")
            print("   🎯 目标: 寻找最优策略参数组合")
            print("   📊 方法: 遗传算法高精度优化")
            print("   🔒 固定参数: rise_threshold=0.04, max_days=20")
            
            self.logger.info("🔧 步骤A: 策略参数优化")
            self.logger.info("   🎯 目标: 寻找最优策略参数组合")
            self.logger.info("   📊 方法: 遗传算法高精度优化")
            self.logger.info("   🔒 固定参数: rise_threshold=0.04, max_days=20")
            step_a_start = time.time()
            
            strategy_result = self.optimize_strategy_parameters_improved(strategy_module, data)
            step_a_time = time.time() - step_a_start
            optimization_result['strategy_optimization'] = strategy_result
            
            if strategy_result['success']:
                # 更新策略模块参数
                strategy_module.update_params(strategy_result['best_params'])
                
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"✅ 步骤A完成 (耗时: {step_a_time:.2f}s) [{current_time}]")
                print(f"   🎯 优化方法: {strategy_result.get('optimization_method', 'unknown')}")
                print(f"   📈 最优得分: {strategy_result.get('best_score', 0):.6f}")
                print(f"   📊 测试集成功率: {strategy_result.get('test_success_rate', 0):.2%}")
                
                self.logger.info(f"✅ 步骤A完成 (耗时: {step_a_time:.2f}s)")
                self.logger.info(f"   🎯 优化方法: {strategy_result.get('optimization_method', 'unknown')}")
                self.logger.info(f"   📈 最优得分: {strategy_result.get('best_score', 0):.6f}")
                self.logger.info(f"   📊 测试集成功率: {strategy_result.get('test_success_rate', 0):.2%}")
                
                # 简要显示优化参数
                best_params = strategy_result.get('best_params', {})
                if len(best_params) <= 5:
                    print(f"   🔧 最优参数: {best_params}")
                    self.logger.info(f"   🔧 最优参数: {best_params}")
                else:
                    # 显示前5个重要参数
                    key_params = dict(list(best_params.items())[:5])
                    print(f"   🔧 关键参数: {key_params}... (共{len(best_params)}个)")
                    self.logger.info(f"   🔧 关键参数: {key_params}... (共{len(best_params)}个)")
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"⚠️ 步骤A失败 (耗时: {step_a_time:.2f}s) - 使用默认参数继续 [{current_time}]")
                self.logger.warning(f"⚠️ 步骤A失败 (耗时: {step_a_time:.2f}s) - 使用默认参数继续")
                optimization_result['errors'].append("策略参数优化失败")
            
            print("-" * 80)
            self.logger.info("-" * 80)
            
            # 步骤B: 改进版模型训练
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🤖 步骤B: 改进版模型训练 [{current_time}]")
            print("   🎯 目标: 训练RandomForest分类模型")
            print("   ⚙️ 配置: 150棵树, 深度12, 平衡权重")
            print("   📊 数据: 特征工程 + 样本权重 + 标准化")
            
            self.logger.info("🤖 步骤B: 改进版模型训练")
            self.logger.info("   🎯 目标: 训练RandomForest分类模型")
            self.logger.info("   ⚙️ 配置: 150棵树, 深度12, 平衡权重")
            self.logger.info("   📊 数据: 特征工程 + 样本权重 + 标准化")
            step_b_start = time.time()
            
            model_result = self.full_train(data, strategy_module)
            step_b_time = time.time() - step_b_start
            optimization_result['model_training'] = model_result
            
            if model_result['success']:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"✅ 步骤B完成 (耗时: {step_b_time:.2f}s) [{current_time}]")
                print(f"   📊 训练样本: {model_result.get('train_samples', 0):,}条")
                print(f"   📈 特征数量: {model_result.get('feature_count', 0)}个")
                print(f"   📊 正样本比例: {model_result.get('positive_ratio', 0):.2%}")
                print(f"   💾 模型保存: {'成功' if model_result.get('save_success', False) else '失败'}")
                
                self.logger.info(f"✅ 步骤B完成 (耗时: {step_b_time:.2f}s)")
                self.logger.info(f"   📊 训练样本: {model_result.get('train_samples', 0):,}条")
                self.logger.info(f"   📈 特征数量: {model_result.get('feature_count', 0)}个")
                self.logger.info(f"   📊 正样本比例: {model_result.get('positive_ratio', 0):.2%}")
                self.logger.info(f"   💾 模型保存: {'成功' if model_result.get('save_success', False) else '失败'}")
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"❌ 步骤B失败 (耗时: {step_b_time:.2f}s) [{current_time}]")
                self.logger.error(f"❌ 步骤B失败 (耗时: {step_b_time:.2f}s)")
                optimization_result['errors'].append("模型训练失败")
                
                # 计算已耗时间
                elapsed_time = time.time() - complete_start_time
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"💔 优化流程中断 (已运行: {elapsed_time:.2f}s) [{current_time}]")
                self.logger.error(f"💔 优化流程中断 (已运行: {elapsed_time:.2f}s)")
                return optimization_result
            
            print("-" * 80)
            self.logger.info("-" * 80)
            
            # 步骤C: 最终性能评估
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"📊 步骤C: 最终性能评估 [{current_time}]")
            print("   🎯 目标: 验证整体系统性能")
            print("   📊 指标: 策略得分 + AI置信度 + 识别效果")
            
            self.logger.info("📊 步骤C: 最终性能评估")
            self.logger.info("   🎯 目标: 验证整体系统性能")
            self.logger.info("   📊 指标: 策略得分 + AI置信度 + 识别效果")
            step_c_start = time.time()
            
            evaluation_result = self.evaluate_optimized_system(data, strategy_module)
            step_c_time = time.time() - step_c_start
            optimization_result['final_evaluation'] = evaluation_result
            
            if evaluation_result.get('success'):
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"✅ 步骤C完成 (耗时: {step_c_time:.2f}s) [{current_time}]")
                print(f"   🎯 策略得分: {evaluation_result.get('strategy_score', 0):.4f}")
                print(f"   📊 成功率: {evaluation_result.get('strategy_success_rate', 0):.2%}")
                print(f"   🔍 识别点数: {evaluation_result.get('identified_points', 0)}")
                print(f"   🤖 AI置信度: {evaluation_result.get('ai_confidence', 0):.4f}")
                
                self.logger.info(f"✅ 步骤C完成 (耗时: {step_c_time:.2f}s)")
                self.logger.info(f"   🎯 策略得分: {evaluation_result.get('strategy_score', 0):.4f}")
                self.logger.info(f"   📊 成功率: {evaluation_result.get('strategy_success_rate', 0):.2%}")
                self.logger.info(f"   🔍 识别点数: {evaluation_result.get('identified_points', 0)}")
                self.logger.info(f"   🤖 AI置信度: {evaluation_result.get('ai_confidence', 0):.4f}")
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"⚠️ 步骤C部分失败 (耗时: {step_c_time:.2f}s) [{current_time}]")
                self.logger.warning(f"⚠️ 步骤C部分失败 (耗时: {step_c_time:.2f}s)")
                optimization_result['errors'].append("最终评估部分失败")
            
            print("-" * 80)
            self.logger.info("-" * 80)
            
            # 步骤D: 保存优化结果
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"💾 步骤D: 保存优化结果 [{current_time}]")
            self.logger.info("💾 步骤D: 保存优化结果")
            step_d_start = time.time()
            
            if strategy_result['success']:
                print("   📝 保存最优参数到配置文件...")
                self.logger.info("   📝 保存最优参数到配置文件...")
                self.save_optimized_params(strategy_result['best_params'])
                print("   ✅ 参数保存完成")
                self.logger.info("   ✅ 参数保存完成")
            else:
                print("   ⚠️ 跳过参数保存 (策略优化失败)")
                self.logger.info("   ⚠️ 跳过参数保存 (策略优化失败)")
            
            step_d_time = time.time() - step_d_start
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"✅ 步骤D完成 (耗时: {step_d_time:.2f}s) [{current_time}]")
            self.logger.info(f"✅ 步骤D完成 (耗时: {step_d_time:.2f}s)")
            
            # 设置最终成功状态
            optimization_result['success'] = model_result['success']
            
            # 总结报告
            total_time = time.time() - complete_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print("=" * 80)
            print(f"🎉 完整AI优化流程完成! [{current_time}]")
            print(f"⏱️ 总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
            print("📊 各步骤耗时分析:")
            print(f"   🔧 策略优化: {step_a_time:.2f}s ({(step_a_time/total_time)*100:.1f}%)")
            print(f"   🤖 模型训练: {step_b_time:.2f}s ({(step_b_time/total_time)*100:.1f}%)")
            print(f"   📊 性能评估: {step_c_time:.2f}s ({(step_c_time/total_time)*100:.1f}%)")
            print(f"   💾 结果保存: {step_d_time:.2f}s ({(step_d_time/total_time)*100:.1f}%)")
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 完整AI优化流程完成!")
            self.logger.info(f"⏱️ 总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
            self.logger.info("📊 各步骤耗时分析:")
            self.logger.info(f"   🔧 策略优化: {step_a_time:.2f}s ({(step_a_time/total_time)*100:.1f}%)")
            self.logger.info(f"   🤖 模型训练: {step_b_time:.2f}s ({(step_b_time/total_time)*100:.1f}%)")
            self.logger.info(f"   📊 性能评估: {step_c_time:.2f}s ({(step_c_time/total_time)*100:.1f}%)")
            self.logger.info(f"   💾 结果保存: {step_d_time:.2f}s ({(step_d_time/total_time)*100:.1f}%)")
            
            success_steps = sum([
                1 if strategy_result['success'] else 0,
                1 if model_result['success'] else 0,
                1 if evaluation_result.get('success', False) else 0
            ])
            print(f"✅ 成功步骤: {success_steps}/3")
            self.logger.info(f"✅ 成功步骤: {success_steps}/3")
            
            if optimization_result['errors']:
                print("⚠️ 遇到的问题:")
                self.logger.warning("⚠️ 遇到的问题:")
                for error in optimization_result['errors']:
                    print(f"   - {error}")
                    self.logger.warning(f"   - {error}")
            
            print("=" * 80)
            self.logger.info("=" * 80)
            return optimization_result
            
        except Exception as e:
            elapsed_time = time.time() - complete_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"💥 完整AI优化流程异常失败 (已运行: {elapsed_time:.2f}s): {e} [{current_time}]")
            self.logger.error(f"💥 完整AI优化流程异常失败 (已运行: {elapsed_time:.2f}s): {e}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"异常详情: {traceback_str}")
            self.logger.error(f"异常详情: {traceback_str}")
            return {
                'success': False,
                'error': str(e),
                'strategy_optimization': {},
                'model_training': {},
                'final_evaluation': {},
                'elapsed_time': elapsed_time
            }
    
    def optimize_strategy_parameters_improved(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        改进版策略参数优化（集成遗传算法的高精度模式）
        
        参数:
        strategy_module: 策略模块
        data: 历史数据
        
        返回:
        dict: 优化结果
        """
        from datetime import datetime
        optimization_start_time = time.time()
        
        # 同时使用print和logger确保输出可见
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"    🚀 启动策略参数优化子流程 [{current_time}]")
        print(f"    📊 数据规模: {len(data)} 条记录")
        
        self.logger.info("🚀 开始改进版策略参数优化（集成遗传算法）")
        self.logger.info("=" * 80)
        
        try:
            # 检查是否有足够的数据
            if len(data) < 100:
                print(f"    ❌ 数据量不足 (仅{len(data)}条)")
                return {
                    'success': False,
                    'error': '数据量不足，无法进行参数优化'
                }
            
            # 步骤1: 数据分割准备
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    📊 子步骤1: 数据分割与验证 [{current_time}]")
            self.logger.info("📊 步骤1: 数据分割与验证...")
            data_prep_start = time.time()
            
            # 从配置文件获取三层数据分割比例
            validation_config = self.config.get('ai', {}).get('validation', {})
            train_ratio = validation_config.get('train_ratio', 0.65)
            val_ratio = validation_config.get('validation_ratio', 0.2) 
            test_ratio = validation_config.get('test_ratio', 0.15)
            
            # 验证比例总和
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                print(f"    ⚠️ 数据分割比例调整: {total_ratio:.3f} → 1.0")
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
            
            data_prep_time = time.time() - data_prep_start
            
            print(f"    ✅ 数据分割完成 (耗时: {data_prep_time:.2f}s)")
            print(f"       📊 训练集: {len(train_data)}条 ({train_ratio:.1%})")
            print(f"       📈 验证集: {len(validation_data)}条 ({val_ratio:.1%})")
            print(f"       🔒 测试集: {len(test_data)}条 ({test_ratio:.1%})")
            
            self.logger.info(f"✅ 数据分割完成 (耗时: {data_prep_time:.2f}s):")
            self.logger.info(f"   📊 训练集: {len(train_data)}条 ({train_ratio:.1%}) - 仅用于参数优化")
            self.logger.info(f"   📈 验证集: {len(validation_data)}条 ({val_ratio:.1%}) - 用于模型验证和过拟合检测")
            self.logger.info(f"   🔒 测试集: {len(test_data)}条 ({test_ratio:.1%}) - 完全锁定，仅最终评估")
            self.logger.info("-" * 60)
            
            # 步骤2: 参数优化方法选择
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    🔧 子步骤2: 参数优化方法选择 [{current_time}]")
            self.logger.info("🔧 步骤2: 参数优化方法选择...")
            
            # 检查是否启用遗传算法（带默认值）
            genetic_config = self.config.get('genetic_algorithm', {})
            advanced_config = self.config.get('advanced_optimization', {})
            
            genetic_enabled = genetic_config.get('enabled', True)
            advanced_enabled = advanced_config.get('enabled', True)
            
            # 如果配置不完整，记录警告但继续执行
            if not genetic_config:
                print("    ⚠️ genetic_algorithm配置缺失，使用默认值")
                self.logger.warning("genetic_algorithm配置缺失，使用默认值")
            if not advanced_config:
                print("    ⚠️ advanced_optimization配置缺失，使用默认值")
                self.logger.warning("advanced_optimization配置缺失，使用默认值")
            
            best_params = {}
            best_score = -float('inf')
            optimization_method = 'unknown'
            
            if genetic_enabled and advanced_enabled:
                print("    🧬 选择遗传算法进行高精度参数优化")
                print("    🎯 配置参数: 50个体 × 30代 = 1500次评估")
                print("    ⏳ 预计耗时: 15-30分钟（取决于数据量）")
                
                self.logger.info("🧬 选择遗传算法进行高精度参数优化")
                genetic_start_time = time.time()
                
                # 定义评估函数
                def evaluate_strategy_params(params):
                    try:
                        # 🚨 重要：添加固定参数
                        # 确保固定参数始终使用预设值
                        complete_params = params.copy()
                        complete_params['rise_threshold'] = 0.04  # 固定涨幅阈值
                        complete_params['max_days'] = 20          # 固定交易日期
                        
                        # 保存原始参数以便恢复
                        original_params = strategy_module.get_current_params() if hasattr(strategy_module, 'get_current_params') else None
                        
                        # 应用完整参数（包含固定参数）
                        strategy_module.update_params(complete_params)
                        
                        # 在训练集上评估
                        backtest_results = strategy_module.backtest(train_data)
                        evaluation = strategy_module.evaluate_strategy(backtest_results)
                        
                        # 恢复原始参数（可选，但更安全）
                        if original_params is not None:
                            strategy_module.update_params(original_params)
                        
                        # 综合评分（重点关注准确度）
                        score = evaluation.get('score', 0)
                        success_rate = evaluation.get('success_rate', 0)
                        avg_rise = evaluation.get('avg_rise', 0)
                        
                        # 高精度评分：更重视成功率，添加安全检查
                        final_score = (
                            success_rate * 0.7 +          # 70%权重给成功率
                            min(avg_rise / 0.1, 1.0) * 0.2 +  # 20%权重给涨幅（最高10%）
                            score * 0.1                     # 10%权重给综合分
                        )
                        
                        # 确保返回值在合理范围内
                        return max(0.0, min(1.0, final_score))
                        
                    except Exception as e:
                        self.logger.warning(f"参数评估失败: {e}")
                        return -1.0
                
                # 运行遗传算法
                print(f"    🔬 开始遗传算法参数搜索... [{datetime.now().strftime('%H:%M:%S')}]")
                self.logger.info("🔬 开始遗传算法参数搜索...")
                genetic_params = self.run_genetic_algorithm(evaluate_strategy_params)
                genetic_time = time.time() - genetic_start_time
                
                if genetic_params:
                    # 评估遗传算法结果
                    strategy_module.update_params(genetic_params)
                    genetic_backtest = strategy_module.backtest(train_data)
                    genetic_evaluation = strategy_module.evaluate_strategy(genetic_backtest)
                    genetic_score = genetic_evaluation.get('score', 0)
                    
                    best_params = genetic_params
                    best_score = genetic_score
                    optimization_method = 'genetic_algorithm_high_precision'
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"    🧬 遗传算法完成 (耗时: {genetic_time:.2f}s) [{current_time}]")
                    print(f"       📈 最优得分: {genetic_score:.6f}")
                    print(f"       📊 成功率: {genetic_evaluation.get('success_rate', 0):.2%}")
                    print(f"       🔍 识别点数: {genetic_evaluation.get('total_points', 0)}")
                    print(f"       📈 平均涨幅: {genetic_evaluation.get('avg_rise', 0):.2%}")
                    
                    self.logger.info(f"🧬 遗传算法完成 (耗时: {genetic_time:.2f}s)")
                    self.logger.info(f"   最优得分: {genetic_score:.6f}")
                    self.logger.info(f"   成功率: {genetic_evaluation.get('success_rate', 0):.2%}")
                    self.logger.info(f"   识别点数: {genetic_evaluation.get('total_points', 0)}")
                    self.logger.info(f"   平均涨幅: {genetic_evaluation.get('avg_rise', 0):.2%}")
                else:
                    print("    ⚠️ 遗传算法未找到有效解，回退到网格搜索")
                    self.logger.warning("⚠️ 遗传算法未找到有效解，回退到网格搜索")
                    genetic_enabled = False
            
            # 如果遗传算法未启用或失败，使用网格搜索作为备选
            if not genetic_enabled or not best_params:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"    🔧 使用网格搜索进行参数优化 [{current_time}]")
                self.logger.info("🔧 使用网格搜索进行参数优化")
                grid_start_time = time.time()
                
                param_ranges = self.config.get('optimization', {}).get('strategy_ranges', {})
                grid_params, grid_score = self._grid_search_optimization(
                    strategy_module, train_data, param_ranges
                )
                
                grid_time = time.time() - grid_start_time
                
                if grid_score > best_score:
                    best_params = grid_params
                    best_score = grid_score
                    optimization_method = 'grid_search_fallback'
                
                print(f"    🔧 网格搜索完成 (耗时: {grid_time:.2f}s)")
                print(f"       最优得分: {grid_score:.6f}")
                self.logger.info(f"🔧 网格搜索完成 (耗时: {grid_time:.2f}s)")
                self.logger.info(f"   最优得分: {grid_score:.6f}")
            
            # 验证最佳参数
            if not best_params:
                print("    ❌ 所有优化方法都未找到有效参数")
                return {
                    'success': False,
                    'error': '所有优化方法都未找到有效参数'
                }
            
            self.logger.info("-" * 60)
            
            # 步骤3: 验证集验证
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    📈 子步骤3: 验证集验证 [{current_time}]")
            self.logger.info("📈 步骤3: 在验证集上验证最佳参数...")
            validation_start_time = time.time()
            
            strategy_module.update_params(best_params)
            val_backtest = strategy_module.backtest(validation_data)
            val_evaluation = strategy_module.evaluate_strategy(val_backtest)
            val_score = val_evaluation['score']
            val_success_rate = val_evaluation.get('success_rate', 0)
            val_total_points = val_evaluation.get('total_points', 0)
            val_avg_rise = val_evaluation.get('avg_rise', 0)
            
            validation_time = time.time() - validation_start_time
            
            # 检查过拟合
            overfitting_threshold = 0.8  # 验证集得分应该至少是训练集得分的80%
            overfitting_passed = val_score >= best_score * overfitting_threshold
            
            print(f"    ✅ 验证集评估完成 (耗时: {validation_time:.2f}s)")
            print(f"       得分: {val_score:.6f}")
            print(f"       成功率: {val_success_rate:.2%}")
            print(f"       识别点数: {val_total_points}")
            print(f"       平均涨幅: {val_avg_rise:.2%}")
            print(f"       过拟合检测: {'✅ 通过' if overfitting_passed else '⚠️ 警告'}")
            
            self.logger.info(f"✅ 验证集评估完成 (耗时: {validation_time:.2f}s)")
            self.logger.info(f"   得分: {val_score:.6f}")
            self.logger.info(f"   成功率: {val_success_rate:.2%}")
            self.logger.info(f"   识别点数: {val_total_points}")
            self.logger.info(f"   平均涨幅: {val_avg_rise:.2%}")
            self.logger.info(f"   过拟合检测: {'✅ 通过' if overfitting_passed else '⚠️ 警告'}")
            self.logger.info("-" * 60)
            
            # 步骤4: 测试集最终评估
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    🔒 子步骤4: 测试集最终评估 [{current_time}]")
            self.logger.info("🔒 步骤4: 在测试集上进行最终评估...")
            test_start_time = time.time()
            
            test_backtest = strategy_module.backtest(test_data)
            test_evaluation = strategy_module.evaluate_strategy(test_backtest)
            test_score = test_evaluation['score']
            test_success_rate = test_evaluation.get('success_rate', 0)
            test_total_points = test_evaluation.get('total_points', 0)
            test_avg_rise = test_evaluation.get('avg_rise', 0)
            
            test_time = time.time() - test_start_time
            
            # 评估模型泛化能力（添加安全检查）
            if val_score > 0.001:  # 避免除零错误
                generalization_ratio = test_score / val_score
            else:
                generalization_ratio = 0.0
                print("    ⚠️ 验证集得分过低，无法计算泛化比率")
                self.logger.warning("验证集得分过低，无法计算泛化比率")
            
            generalization_passed = generalization_ratio >= 0.85  # 测试集得分应该接近验证集
            
            print(f"    ✅ 测试集评估完成 (耗时: {test_time:.2f}s)")
            print(f"       得分: {test_score:.6f}")
            print(f"       成功率: {test_success_rate:.2%}")
            print(f"       识别点数: {test_total_points}")
            print(f"       平均涨幅: {test_avg_rise:.2%}")
            print(f"       泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")
            
            self.logger.info(f"✅ 测试集评估完成 (耗时: {test_time:.2f}s)")
            self.logger.info(f"   得分: {test_score:.6f}")
            self.logger.info(f"   成功率: {test_success_rate:.2%}")
            self.logger.info(f"   识别点数: {test_total_points}")
            self.logger.info(f"   平均涨幅: {test_avg_rise:.2%}")
            self.logger.info(f"   泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")
            
            # 总结
            optimization_total_time = time.time() - optimization_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    🎉 策略参数优化子流程完成! [{current_time}]")
            print(f"    ⏱️ 总耗时: {optimization_total_time:.2f}s ({optimization_total_time/60:.1f}分钟)")
            print(f"    🔧 优化方法: {optimization_method}")
            print(f"    📊 三层验证结果:")
            print(f"       训练集得分: {best_score:.6f}")
            print(f"       验证集得分: {val_score:.6f} | 成功率: {val_success_rate:.2%}")
            print(f"       测试集得分: {test_score:.6f} | 成功率: {test_success_rate:.2%}")
            print(f"       🛡️ 过拟合检测: {'通过' if overfitting_passed else '警告'}")
            print(f"       🎯 泛化能力: {'良好' if generalization_passed else '一般'}")
            
            self.logger.info("=" * 80)
            self.logger.info(f"🎉 策略参数优化完成!")
            self.logger.info(f"⏱️ 总耗时: {optimization_total_time:.2f}s ({optimization_total_time/60:.1f}分钟)")
            self.logger.info(f"🔧 优化方法: {optimization_method}")
            self.logger.info(f"📊 三层验证结果:")
            self.logger.info(f"   训练集得分: {best_score:.6f}")
            self.logger.info(f"   验证集得分: {val_score:.6f} | 成功率: {val_success_rate:.2%}")
            self.logger.info(f"   测试集得分: {test_score:.6f} | 成功率: {test_success_rate:.2%}")
            self.logger.info(f"   🛡️ 过拟合检测: {'通过' if overfitting_passed else '警告'}")
            self.logger.info(f"   🎯 泛化能力: {'良好' if generalization_passed else '一般'}")
            
            # 如果使用了遗传算法，输出详细的参数信息
            if optimization_method == 'genetic_algorithm_high_precision':
                print(f"    🧬 遗传算法最优参数详情:")
                self.logger.info(f"\n🧬 遗传算法最优参数详情:")
                for param_name, param_value in best_params.items():
                    print(f"       {param_name}: {param_value}")
                    self.logger.info(f"   {param_name}: {param_value}")
            
            self.logger.info("=" * 80)
            
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
                'optimization_method': optimization_method,
                'optimization_time': optimization_total_time,
                'genetic_algorithm_used': optimization_method == 'genetic_algorithm_high_precision'
            }
            
        except Exception as e:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    ❌ 策略参数优化失败: {e} [{current_time}]")
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
        传统的参数保存方式（原子性写入）
        
        参数:
        params: 优化后的参数
        """
        import tempfile
        import shutil
        
        try:
            config_path = 'config/config_improved.yaml'
            backup_path = f"{config_path}.backup"
            
            # 创建备份
            if os.path.exists(config_path):
                shutil.copy2(config_path, backup_path)
                self.logger.info(f"已创建配置文件备份: {backup_path}")
            
            # 读取现有配置
            config = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    self.logger.error(f"配置文件格式错误: {e}")
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, config_path)
                        self.logger.info("已从备份恢复配置文件")
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
            
            # 更新参数
            for param_name, param_value in params.items():
                if param_name == 'final_threshold':
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    if 'confidence_weights' not in config['strategy']:
                        config['strategy']['confidence_weights'] = {}
                    config['strategy']['confidence_weights']['final_threshold'] = float(param_value)
                elif param_name in ['rsi_oversold_threshold', 'rsi_low_threshold']:
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    if 'confidence_weights' not in config['strategy']:
                        config['strategy']['confidence_weights'] = {}
                    config['strategy']['confidence_weights'][param_name] = float(param_value)
                else:
                    # 其他参数按原有逻辑处理
                    if 'strategy' not in config:
                        config['strategy'] = {}
                    
                    # 类型转换
                    if isinstance(param_value, (int, float)):
                        config['strategy'][param_name] = float(param_value)
                    else:
                        config['strategy'][param_name] = param_value
            
            # 原子性写入：先写入临时文件，再移动
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                           dir=os.path.dirname(config_path), 
                                           delete=False) as temp_file:
                yaml.dump(config, temp_file, default_flow_style=False, allow_unicode=True)
                temp_path = temp_file.name
            
            # 移动临时文件到目标位置
            shutil.move(temp_path, config_path)
            
            self.logger.info(f"参数已安全保存到配置文件: {len(params)} 个参数")
            
            # 清理旧备份
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
        except Exception as e:
            self.logger.error(f"传统方式保存参数失败: {e}")
            # 尝试从备份恢复
            if os.path.exists(backup_path) and os.path.exists(config_path):
                try:
                    shutil.copy2(backup_path, config_path)
                    self.logger.info("已从备份恢复配置文件")
                except Exception as restore_error:
                    self.logger.error(f"备份恢复失败: {restore_error}")

    def run_genetic_algorithm(self, evaluate_func, param_ranges=None) -> Dict[str, Any]:
        """
        遗传算法参数优化（高精度版本）
        
        专为高准确度设计，不考虑执行时间限制
        
        参数:
        evaluate_func: 评估函数，接收参数字典，返回评分
        param_ranges: 参数范围字典，如果为None则使用配置文件
        
        返回:
        dict: 最优参数字典
        """
        from datetime import datetime
        
        print(f"        🧬 初始化遗传算法 [{datetime.now().strftime('%H:%M:%S')}]")
        self.logger.info("🧬 启动遗传算法优化（高精度模式）")
        start_time = time.time()
        
        try:
            # 获取遗传算法配置（针对高精度调整）
            genetic_config = self.config.get('genetic_algorithm', {})
            
            # 高精度配置：增加种群和代数
            population_size = genetic_config.get('population_size', 50)  # 增加到50
            generations = genetic_config.get('generations', 30)          # 增加到30
            crossover_rate = genetic_config.get('crossover_rate', 0.8)
            mutation_rate = genetic_config.get('mutation_rate', 0.15)    # 稍微提高变异率
            elite_ratio = genetic_config.get('elite_ratio', 0.1)         # 保留10%精英
            
            print(f"        📊 遗传算法配置:")
            print(f"           种群大小: {population_size} 个体")
            print(f"           进化代数: {generations} 代")
            print(f"           交叉概率: {crossover_rate:.1%}")
            print(f"           变异概率: {mutation_rate:.1%}")
            print(f"           精英比例: {elite_ratio:.1%}")
            
            self.logger.info(f"高精度遗传算法配置: 种群{population_size}, 代数{generations}")
            
            # 获取或生成参数范围
            if param_ranges is None:
                param_ranges = self._get_enhanced_parameter_ranges({})
            
            print(f"        🎯 优化参数数量: {len(param_ranges)} 个")
            
            # 初始化种群
            print(f"        🌱 生成初始种群... [{datetime.now().strftime('%H:%M:%S')}]")
            population = self._initialize_population(param_ranges, population_size)
            
            best_individual = None
            best_score = -float('inf')
            best_generation = 0
            stagnation_count = 0
            recent_generations = []
            
            print(f"        🚀 开始进化过程 (总计 {population_size * generations} 次评估)")
            total_evaluations = population_size * generations
            
            # 进化主循环
            for generation in range(generations):
                generation_start_time = time.time()
                current_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"        🧬 第 {generation + 1}/{generations} 代进化 ({((generation + 1)/generations)*100:.1f}% 完成) [{current_time}]")
                self.logger.info(f"\n🧬 第 {generation + 1}/{generations} 代进化 "
                               f"({((generation + 1)/generations)*100:.1f}% 完成)")
                self.logger.info("------------------------------------------------------------")
                
                # 评估当前种群
                scores = []
                valid_evaluations = 0
                failed_evaluations = 0
                
                # 每10个个体显示一次进度，避免过多输出
                for i, individual in enumerate(population):
                    if (i + 1) % 10 == 0 or i == len(population) - 1:
                        current_progress = ((generation * population_size + i + 1) / total_evaluations) * 100
                        print(f"        🔍 评估进度: {i+1}/{population_size} 个体 | 总进度: {current_progress:.1f}% | 第{generation + 1}代")
                        self.logger.info(f"🔍 评估进度: {i+1}/{population_size} 个体 | "
                                       f"总进度: {current_progress:.1f}% | 第{generation + 1}代")
                    
                    try:
                        score = evaluate_func(individual)
                        if score is not None and score >= 0:
                            scores.append(score)
                            valid_evaluations += 1
                        else:
                            scores.append(0.0)
                            failed_evaluations += 1
                    except Exception as e:
                        self.logger.warning(f"个体评估失败: {e}")
                        scores.append(0.0)
                        failed_evaluations += 1
                
                # 更新全局最优
                generation_best_idx = np.argmax(scores)
                generation_best_score = scores[generation_best_idx]
                
                if generation_best_score > best_score:
                    best_score = generation_best_score
                    best_individual = population[generation_best_idx].copy()
                    best_generation = generation + 1
                    stagnation_count = 0
                    print(f"        🎉 发现新最佳解! 得分: {best_score:.6f}")
                    self.logger.info(f"🎉 发现新最佳解! 得分: {best_score:.6f}")
                else:
                    stagnation_count += 1
                
                # 统计信息
                if len(scores) > 0:
                    max_score = max(scores)
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    std_score = np.std(scores)
                    
                    generation_time = time.time() - generation_start_time
                    
                    # 预计剩余时间
                    remaining_generations = generations - generation - 1
                    if generation > 0:
                        avg_generation_time = (time.time() - start_time) / (generation + 1)
                        estimated_remaining = remaining_generations * avg_generation_time
                    else:
                        estimated_remaining = remaining_generations * generation_time
                    
                    print(f"        📊 第{generation + 1}代统计:")
                    print(f"           ✅ 有效个体: {valid_evaluations}/{population_size}")
                    print(f"           ❌ 失败个体: {failed_evaluations}")
                    print(f"           📈 最高分: {max_score:.6f}")
                    print(f"           📊 平均分: {avg_score:.6f}")
                    print(f"           📉 最低分: {min_score:.6f}")
                    print(f"           📏 标准差: {std_score:.6f}")
                    print(f"           🏆 历史最优: {best_score:.6f}")
                    print(f"           ⏱️ 本代耗时: {generation_time:.2f}s")
                    print(f"           ⏳ 预计剩余时间: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}分钟)")
                    
                    self.logger.info(f"📊 第{generation + 1}代统计:")
                    self.logger.info(f"   ✅ 有效个体: {valid_evaluations}/{population_size}")
                    self.logger.info(f"   ❌ 失败个体: {failed_evaluations}")
                    self.logger.info(f"   📈 最高分: {max_score:.6f}")
                    self.logger.info(f"   📊 平均分: {avg_score:.6f}")
                    self.logger.info(f"   📉 最低分: {min_score:.6f}")
                    self.logger.info(f"   📏 标准差: {std_score:.6f}")
                    self.logger.info(f"   🏆 历史最优: {best_score:.6f}")
                    self.logger.info(f"   ⏱️ 本代耗时: {generation_time:.2f}s")
                    self.logger.info(f"   ⏳ 预计剩余时间: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}分钟)")
                    
                    # 保存最近几代的统计信息
                    recent_generations.append({
                        'generation': generation + 1,
                        'max_score': max_score,
                        'avg_score': avg_score,
                        'std_score': std_score,
                        'best_score': best_score
                    })
                
                # 每5代分析一次收敛趋势
                if (generation + 1) % 5 == 0 and len(recent_generations) >= 5:
                    self._log_genetic_statistics(recent_generations[-5:])
                
                # 如果不是最后一代，进行进化操作
                if generation < generations - 1:
                    evolution_start = time.time()
                    print(f"        🔄 开始第{generation + 1}代进化操作... [{datetime.now().strftime('%H:%M:%S')}]")
                    self.logger.info(f"🔄 开始第{generation + 1}代进化操作...")
                    
                    # 进化种群
                    population = self._evolve_population(
                        population, scores, param_ranges, population_size,
                        crossover_rate, mutation_rate, elite_ratio
                    )
                    
                    evolution_time = time.time() - evolution_start
                    print(f"        ✅ 进化操作完成 (耗时: {evolution_time:.2f}s)")
                    self.logger.info(f"✅ 进化操作完成 (耗时: {evolution_time:.2f}s)")
                
                # 提前停止条件：连续多代无改善
                if stagnation_count >= 10:
                    print(f"        🛑 连续{stagnation_count}代无改善，提前停止")
                    self.logger.info(f"连续{stagnation_count}代无改善，提前停止")
                    break
                
                print(f"        {'='*60}")
                self.logger.info("------------------------------------------------------------")
            
            total_time = time.time() - start_time
            current_time = datetime.now().strftime('%H:%M:%S')
            
            print(f"        🎉 遗传算法优化完成! [{current_time}]")
            print(f"        ⏱️ 总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
            print(f"        🏆 最优得分: {best_score:.6f}")
            print(f"        📍 最佳代数: 第{best_generation}代")
            
            if best_individual:
                print(f"        🔧 最优参数:")
                for param_name, param_value in best_individual.items():
                    print(f"           {param_name}: {param_value}")
            
            self.logger.info("🎉 遗传算法优化完成!")
            self.logger.info(f"总耗时: {total_time:.2f}s, 最优得分: {best_score:.6f}")
            self.logger.info(f"最佳解在第{best_generation}代发现")
            
            return best_individual if best_individual else {}
            
        except Exception as e:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"        ❌ 遗传算法执行失败: {e} [{current_time}]")
            self.logger.error(f"遗传算法执行失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return {}
    
    def _get_enhanced_parameter_ranges(self, base_ranges: dict) -> dict:
        """
        获取增强的参数范围（更精细的搜索）
        
        参数:
        base_ranges: 基础参数范围
        
        返回:
        dict: 增强的参数范围
        """
        # 🚨 重要：固定参数，不参与遗传算法优化
        # rise_threshold: 0.04 和 max_days: 20 是用户预设的固定值
        
        enhanced_ranges = {
            # ❌ 移除固定参数，不参与优化
            # 'rise_threshold': 用户指定固定为0.04
            # 'max_days': 用户指定固定为20
            
            # RSI参数（扩展范围）
            'rsi_oversold_threshold': {
                'min': 20, 'max': 40, 'type': 'int'
            },
            'rsi_low_threshold': {
                'min': 30, 'max': 50, 'type': 'int'
            },
            
            # 置信度阈值（高精度）
            'final_threshold': {
                'min': 0.25, 'max': 0.85, 'type': 'float', 'precision': 4
            },
            
            # 高级权重参数（新增）
            'dynamic_confidence_adjustment': {
                'min': 0.02, 'max': 0.30, 'type': 'float', 'precision': 4
            },
            'market_sentiment_weight': {
                'min': 0.05, 'max': 0.35, 'type': 'float', 'precision': 4
            },
            'trend_strength_weight': {
                'min': 0.03, 'max': 0.25, 'type': 'float', 'precision': 4
            },
            'volume_weight': {
                'min': 0.10, 'max': 0.45, 'type': 'float', 'precision': 4
            },
            'price_momentum_weight': {
                'min': 0.08, 'max': 0.35, 'type': 'float', 'precision': 4
            },
            
            # 布林带参数
            'bb_near_threshold': {
                'min': 1.005, 'max': 1.05, 'type': 'float', 'precision': 5
            },
            
            # 成交量参数
            'volume_panic_threshold': {
                'min': 1.2, 'max': 2.0, 'type': 'float', 'precision': 3
            },
            'volume_surge_threshold': {
                'min': 1.1, 'max': 1.5, 'type': 'float', 'precision': 3
            },
            'volume_shrink_threshold': {
                'min': 0.5, 'max': 0.9, 'type': 'float', 'precision': 3
            }
        }
        
        # 合并用户配置的范围（但排除固定参数）
        for param_name, param_config in base_ranges.items():
            # 🚨 跳过固定参数，不允许优化
            if param_name in ['rise_threshold', 'max_days']:
                self.logger.info(f"⚠️ 跳过固定参数 {param_name}，此参数不参与优化")
                continue
                
            if param_name in enhanced_ranges:
                # 更新现有参数范围
                enhanced_ranges[param_name].update(param_config)
            else:
                # 添加新参数
                enhanced_ranges[param_name] = param_config.copy()
                enhanced_ranges[param_name]['type'] = 'float'  # 默认为浮点数
                enhanced_ranges[param_name]['precision'] = 4
        
        self.logger.info(f"🎯 增强参数搜索空间: {len(enhanced_ranges)} 个参数")
        self.logger.info(f"🔒 固定参数: rise_threshold=0.04, max_days=20 (不参与优化)")
        return enhanced_ranges
    
    def _initialize_population(self, param_ranges: dict, population_size: int) -> List[Dict]:
        """
        初始化种群
        
        参数:
        param_ranges: 参数范围
        population_size: 种群大小
        
        返回:
        List[Dict]: 初始种群
        """
        if not param_ranges:
            raise ValueError("参数范围不能为空")
        
        if population_size <= 0:
            raise ValueError(f"种群大小必须大于0，当前值: {population_size}")
        
        population = []
        
        # 验证参数范围的有效性
        for param_name, param_config in param_ranges.items():
            if 'min' not in param_config or 'max' not in param_config:
                raise ValueError(f"参数 {param_name} 缺少 min 或 max 配置")
            
            min_val = param_config['min']
            max_val = param_config['max']
            
            if min_val >= max_val:
                raise ValueError(f"参数 {param_name} 的最小值({min_val})必须小于最大值({max_val})")
        
        for _ in range(population_size):
            individual = {}
            for param_name, param_config in param_ranges.items():
                min_val = param_config['min']
                max_val = param_config['max']
                param_type = param_config.get('type', 'float')
                
                try:
                    if param_type == 'int':
                        # 确保整数范围有效
                        if max_val - min_val < 1:
                            individual[param_name] = min_val
                        else:
                            individual[param_name] = np.random.randint(min_val, max_val + 1)
                    else:  # float
                        individual[param_name] = np.random.uniform(min_val, max_val)
                        precision = param_config.get('precision', 4)
                        individual[param_name] = round(individual[param_name], precision)
                        
                except Exception as e:
                    self.logger.error(f"初始化参数 {param_name} 失败: {e}")
                    # 使用中间值作为默认值
                    if param_type == 'int':
                        individual[param_name] = int((min_val + max_val) / 2)
                    else:
                        individual[param_name] = round((min_val + max_val) / 2, 
                                                     param_config.get('precision', 4))
            
            population.append(individual)
        
        self.logger.info(f"✅ 初始化种群: {population_size} 个个体，包含 {len(param_ranges)} 个参数")
        return population
    
    def _evolve_population(self, population: List[Dict], scores: List[float], 
                          param_ranges: dict, population_size: int,
                          crossover_rate: float, mutation_rate: float, 
                          elite_ratio: float) -> List[Dict]:
        """
        进化种群
        
        参数:
        population: 当前种群
        scores: 评分列表
        param_ranges: 参数范围
        population_size: 种群大小
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        elite_ratio: 精英保留比例
        
        返回:
        List[Dict]: 新种群
        """
        # 排序个体（按得分降序）
        sorted_indices = np.argsort(scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # 精英保留（深拷贝以避免引用问题）
        elite_count = int(population_size * elite_ratio)
        new_population = [individual.copy() for individual in sorted_population[:elite_count]]
        
        # 生成剩余个体
        remaining_count = population_size - elite_count
        children_needed = remaining_count // 2 * 2  # 确保偶数个子代
        
        for _ in range(children_needed // 2):
            # 选择父母（锦标赛选择）
            parent1 = self._tournament_selection(sorted_population, sorted_scores)
            parent2 = self._tournament_selection(sorted_population, sorted_scores)
            
            # 交叉
            if np.random.random() < crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, param_ranges)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异（每个子代独立决定是否变异）
            if np.random.random() < mutation_rate:
                child1 = self._mutate(child1, param_ranges)
            if np.random.random() < mutation_rate:
                child2 = self._mutate(child2, param_ranges)
            
            # 添加到新种群
            new_population.extend([child1, child2])
        
        # 如果还需要补充个体（奇数情况）
        while len(new_population) < population_size:
            parent = self._tournament_selection(sorted_population, sorted_scores)
            child = self._mutate(parent.copy(), param_ranges) if np.random.random() < mutation_rate else parent.copy()
            new_population.append(child)
        
        # 确保精确的种群大小
        return new_population[:population_size]
    
    def _tournament_selection(self, population: List[Dict], scores: List[float], 
                             tournament_size: int = 5) -> Dict:
        """
        锦标赛选择
        
        参数:
        population: 种群
        scores: 评分
        tournament_size: 锦标赛大小
        
        返回:
        Dict: 选中的个体
        """
        if not population or not scores:
            raise ValueError("种群或评分列表为空")
        
        if len(population) != len(scores):
            raise ValueError(f"种群大小({len(population)})与评分数量({len(scores)})不匹配")
        
        # 过滤有效的个体（评分不是负无穷或NaN）
        valid_indices = [i for i, score in enumerate(scores) 
                        if score != -float('inf') and not np.isnan(score)]
        
        if not valid_indices:
            # 如果没有有效个体，随机选择一个
            self.logger.warning("锦标赛选择：没有有效个体，随机选择")
            return population[np.random.randint(len(population))].copy()
        
        # 从有效个体中进行锦标赛选择
        available_size = len(valid_indices)
        actual_tournament_size = min(tournament_size, available_size)
        
        tournament_indices = np.random.choice(valid_indices, 
                                            size=actual_tournament_size, 
                                            replace=False)
        
        # 找到最佳个体
        best_idx = max(tournament_indices, key=lambda i: scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict, param_ranges: dict) -> Tuple[Dict, Dict]:
        """
        交叉操作（统一交叉 + 算术交叉）
        
        参数:
        parent1: 父母1
        parent2: 父母2
        param_ranges: 参数范围
        
        返回:
        Tuple[Dict, Dict]: 两个子代
        """
        # 参数一致性检查
        if set(parent1.keys()) != set(parent2.keys()):
            self.logger.warning("父母个体参数不一致，使用交集")
            common_params = set(parent1.keys()) & set(parent2.keys())
        else:
            common_params = set(parent1.keys())
        
        child1, child2 = {}, {}
        
        for param_name in common_params:
            # 确保参数在两个父母中都存在
            if param_name not in parent2:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent1[param_name]
                continue
            
            if np.random.random() < 0.5:
                # 交换基因
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
            else:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            
            # 算术交叉（用于数值参数）
            if np.random.random() < 0.3:  # 30%概率进行算术交叉
                alpha = np.random.random()
                val1 = parent1[param_name]
                val2 = parent2[param_name]
                
                new_val1 = alpha * val1 + (1 - alpha) * val2
                new_val2 = (1 - alpha) * val1 + alpha * val2
                
                # 确保在范围内
                param_config = param_ranges.get(param_name, {})
                min_val = param_config.get('min', 0)
                max_val = param_config.get('max', 1)
                param_type = param_config.get('type', 'float')
                
                new_val1 = np.clip(new_val1, min_val, max_val)
                new_val2 = np.clip(new_val2, min_val, max_val)
                
                if param_type == 'int':
                    new_val1 = int(round(new_val1))
                    new_val2 = int(round(new_val2))
                else:
                    precision = param_config.get('precision', 4)
                    new_val1 = round(new_val1, precision)
                    new_val2 = round(new_val2, precision)
                
                child1[param_name] = new_val1
                child2[param_name] = new_val2
        
        return child1, child2
    
    def _mutate(self, individual: Dict, param_ranges: dict, 
               mutation_strength: float = 0.1) -> Dict:
        """
        变异操作
        
        参数:
        individual: 个体
        param_ranges: 参数范围
        mutation_strength: 变异强度
        
        返回:
        Dict: 变异后的个体
        """
        mutated = individual.copy()
        
        for param_name, param_value in individual.items():
            if np.random.random() < 0.3:  # 每个基因30%概率变异
                param_config = param_ranges.get(param_name, {})
                min_val = param_config.get('min', 0)
                max_val = param_config.get('max', 1)
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    # 整数变异：随机选择邻近值
                    mutation_range = max(1, int((max_val - min_val) * mutation_strength))
                    delta = np.random.randint(-mutation_range, mutation_range + 1)
                    new_val = param_value + delta
                    new_val = np.clip(new_val, min_val, max_val)
                    mutated[param_name] = int(new_val)
                else:
                    # 浮点数变异：高斯变异
                    mutation_range = (max_val - min_val) * mutation_strength
                    delta = np.random.normal(0, mutation_range)
                    new_val = param_value + delta
                    new_val = np.clip(new_val, min_val, max_val)
                    
                    precision = param_config.get('precision', 4)
                    mutated[param_name] = round(new_val, precision)
        
        return mutated
    
    def _validate_and_fix_parameters(self, params: Dict, param_ranges: dict) -> Dict:
        """
        验证并修复参数
        
        参数:
        params: 参数字典
        param_ranges: 参数范围
        
        返回:
        Dict: 修复后的参数
        """
        fixed_params = {}
        
        for param_name, param_value in params.items():
            if param_name in param_ranges:
                param_config = param_ranges[param_name]
                min_val = param_config.get('min', 0)
                max_val = param_config.get('max', 1)
                param_type = param_config.get('type', 'float')
                
                # 确保在范围内
                fixed_value = np.clip(param_value, min_val, max_val)
                
                if param_type == 'int':
                    fixed_value = int(round(fixed_value))
                else:
                    precision = param_config.get('precision', 4)
                    fixed_value = round(fixed_value, precision)
                
                fixed_params[param_name] = fixed_value
            else:
                fixed_params[param_name] = param_value
        
        return fixed_params
    
    def _log_genetic_statistics(self, recent_generations: List[Dict]):
        """
        记录遗传算法统计信息
        
        参数:
        recent_generations: 最近几代的统计信息
        """
        if not recent_generations or len(recent_generations) == 0:
            self.logger.warning("没有可用的代数统计信息")
            return
        
        try:
            avg_scores = [gen.get('avg_score', 0) for gen in recent_generations if gen.get('avg_score') is not None and gen.get('avg_score') != -1.0]
            max_scores = [gen.get('max_score', 0) for gen in recent_generations if gen.get('max_score') is not None and gen.get('max_score') != -1.0]
            generation_times = [gen.get('generation_time', 0) for gen in recent_generations if gen.get('generation_time') is not None]
            
            if not avg_scores or not max_scores:
                self.logger.warning("统计数据不完整，跳过统计报告")
                return
            
            self.logger.info(f"📊 最近 {len(recent_generations)} 代性能分析:")
            self.logger.info(f"   📈 平均分趋势: {avg_scores[0]:.4f} → {avg_scores[-1]:.4f} "
                           f"(变化: {avg_scores[-1] - avg_scores[0]:+.4f})")
            self.logger.info(f"   🎯 最高分趋势: {max_scores[0]:.4f} → {max_scores[-1]:.4f} "
                           f"(改善: {max_scores[-1] - max_scores[0]:+.4f})")
            
            if generation_times:
                avg_time = np.mean(generation_times)
                self.logger.info(f"   ⏱️ 平均代耗时: {avg_time:.2f}s")
            
            # 收敛性分析
            if len(max_scores) >= 3:
                recent_improvement = max_scores[-1] - max_scores[-3]
                if abs(recent_improvement) < 0.001:
                    self.logger.info(f"   🎯 收敛状态: 稳定 (近期改善: {recent_improvement:+.6f})")
                else:
                    self.logger.info(f"   🚀 收敛状态: 优化中 (近期改善: {recent_improvement:+.6f})")
            
        except Exception as e:
            self.logger.warning(f"统计信息生成失败: {e}")