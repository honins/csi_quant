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

        # 检查数据中已有的技术指标
        self.logger.info(f"输入数据列: {list(data.columns)}")

        # 计算额外的趋势确认指标
        data = self._calculate_trend_indicators(data)

        # 🔧 确保重要技术指标存在且有效（如果DataModule已计算则保留，否则重新计算）
        data = self._ensure_technical_indicators(data)

        # 🎯 优化：使用精选的高效特征，保留重要的技术指标包括RSI
        # 基于特征重要性分析 + 保留关键技术指标
        optimized_feature_columns = [
            # 🔥 核心趋势指标（最高重要性：0.21 + 0.11 = 32%）
            'trend_strength_60', 'trend_strength_20',

            # 🔥 成交量指标（高重要性：0.10 + 0.07 = 17%）
            'volume_strength', 'volume_trend',

            # ⚡ 均线系统（中高重要性：0.06 + 0.06 + 0.05 = 17%）
            'ma5', 'ma10', 'ma20',

            # ⚡ 价格动量和均线距离（中等重要性：0.05 + 0.05 + 0.05 = 15%）
            'price_change_5d', 'dist_ma20', 'macd',

            # ⚡ 补充特征（较低但有效：0.05 + 0.05 + 0.04 + 0.04 = 18%）
            'dist_ma10', 'dist_ma5', 'ma60', 'price_change_10d',

            # 📊 重要技术指标（保留用于策略一致性）
            'rsi',  # RSI必须保留
            'bb_upper', 'bb_lower',  # 布林带上下轨
            'signal', 'hist'  # MACD信号和柱状线
        ]

        # ❌ 移除的真正噪音特征：
        # 'price_position_20', 'price_position_60', 'volatility', 'volatility_normalized'

        # 🚨 重要：确保所有特征都存在，如果不存在则填充合理的默认值
        for col in optimized_feature_columns:
            if col not in data.columns:
                self.logger.warning(f"缺少特征 {col}，将填充默认值")
                # 根据特征类型填充合理的默认值
                if 'ma' in col or 'price' in col.lower():
                    # 均线和价格相关：使用收盘价
                    data[col] = data['close'] if 'close' in data.columns else 0.0
                elif col in ['rsi']:
                    # RSI：填充中性值50
                    data[col] = 50.0
                elif 'dist_' in col:
                    # 距离相关：填充0（表示在均线上）
                    data[col] = 0.0
                elif 'volume' in col.lower():
                    # 成交量相关：填充1.0（表示正常）
                    data[col] = 1.0
                elif 'volatility' in col.lower():
                    # 波动率相关：填充适中值
                    data[col] = 0.02 if 'normalized' not in col else 1.0
                else:
                    # 其他特征：填充0
                    data[col] = 0.0

        # 使用精选特征集合
        available_columns = optimized_feature_columns.copy()

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
                    prices = data['close'].iloc[i - period + 1:i + 1].values
                    x = np.arange(period)
                    slope = np.polyfit(x, prices, 1)[0]
                    # 标准化斜率
                    normalized_slope = slope / prices.mean()
                    slopes.append(normalized_slope)
                else:
                    slopes.append(0)
            data[f'trend_strength_{period}'] = slopes

        # 价格在均线系统中的位置
        # 确保ma20和ma60列存在且不为NaN
        if 'ma20' in data.columns and data['ma20'].notna().any():
            data['price_position_20'] = (data['close'] - data['ma20']) / data['ma20']
        else:
            data['price_position_20'] = 0

        if 'ma60' in data.columns and data['ma60'].notna().any():
            data['price_position_60'] = (data['close'] - data['ma60']) / data['ma60']
        else:
            data['price_position_60'] = 0

        # 标准化波动率
        if 'volatility' in data.columns and data['volatility'].notna().any():
            volatility_mean = data['volatility'].rolling(60).mean()
            data['volatility_normalized'] = data['volatility'] / volatility_mean
            # 处理除零情况
            data['volatility_normalized'] = data['volatility_normalized'].fillna(1.0)
        else:
            data['volatility_normalized'] = 1.0

        # 成交量趋势指标
        volume_ma20 = data['volume'].rolling(20).mean()
        data['volume_trend'] = (data['volume'] - volume_ma20) / volume_ma20
        # 处理除零情况
        data['volume_trend'] = data['volume_trend'].fillna(0)

        # 成交量强度（相对于历史）
        volume_ma60 = data['volume'].rolling(60).mean()
        data['volume_strength'] = data['volume'] / volume_ma60
        # 处理除零情况
        data['volume_strength'] = data['volume_strength'].fillna(1.0)

        return data

    def _ensure_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        确保重要技术指标存在且有效
        
        参数:
        data: 原始数据
        
        返回:
        pd.DataFrame: 包含所有必要技术指标的数据
        """
        self.logger.info("🔧 确保技术指标完整性")

        # 检查并计算RSI
        if 'rsi' not in data.columns or data['rsi'].isna().all():
            self.logger.info("重新计算RSI指标")
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            # 修复除零错误
            rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
            rs = np.where(avg_gain == 0, 0, rs)
            data['rsi'] = 100 - (100 / (1 + rs))

        # 检查并计算MACD
        if 'macd' not in data.columns or data['macd'].isna().all():
            self.logger.info("重新计算MACD指标")
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['hist'] = data['macd'] - data['signal']

        # 检查并计算布林带
        if 'bb_upper' not in data.columns or data['bb_upper'].isna().all():
            self.logger.info("重新计算布林带指标")
            # 确保ma20存在
            if 'ma20' not in data.columns:
                data['ma20'] = data['close'].rolling(20).mean()
            data['bb_upper'] = data['ma20'] + (data['close'].rolling(20).std() * 2)
            data['bb_lower'] = data['ma20'] - (data['close'].rolling(20).std() * 2)

        # 检查移动平均线
        if 'ma5' not in data.columns:
            data['ma5'] = data['close'].rolling(5).mean()
        if 'ma10' not in data.columns:
            data['ma10'] = data['close'].rolling(10).mean()
        if 'ma20' not in data.columns:
            data['ma20'] = data['close'].rolling(20).mean()
        if 'ma60' not in data.columns:
            data['ma60'] = data['close'].rolling(60).mean()

        # 检查价格与均线距离
        if 'dist_ma5' not in data.columns:
            data['dist_ma5'] = (data['close'] - data['ma5']) / data['ma5']
        if 'dist_ma10' not in data.columns:
            data['dist_ma10'] = (data['close'] - data['ma10']) / data['ma10']
        if 'dist_ma20' not in data.columns:
            data['dist_ma20'] = (data['close'] - data['ma20']) / data['ma20']

        # 检查价格变化率
        if 'price_change_5d' not in data.columns:
            data['price_change_5d'] = data['close'].pct_change(5)
        if 'price_change_10d' not in data.columns:
            data['price_change_10d'] = data['close'].pct_change(10)

        # 验证关键指标
        key_indicators = ['rsi', 'macd', 'signal', 'hist', 'bb_upper', 'bb_lower']
        for indicator in key_indicators:
            if indicator in data.columns:
                valid_count = data[indicator].notna().sum()
                total_count = len(data)
                self.logger.info(f"✅ {indicator}: {valid_count}/{total_count} 有效值")
            else:
                self.logger.warning(f"❌ {indicator} 不存在")

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
                    n_estimators=100,  # 从150降到100
                    max_depth=8,  # 从12降到8
                    min_samples_split=15,  # 从8提高到15
                    min_samples_leaf=8,  # 从3提高到8
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
            self.logger.info(f"⏱️ 总耗时: {total_train_time:.2f}s ({total_train_time / 60:.1f}分钟)")
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
                'feature_names': feature_names,
                'training_time_breakdown': {
                    'feature_engineering': feature_time,
                    'label_preparation': label_time,
                    'weight_calculation': weight_time,
                    'model_training': model_time,
                    'model_saving': save_time
                }
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
                        'final_confidence': 0.0,
                        'error': '模型未训练且无法加载已保存的模型'
                    }

            if len(data) == 0:
                return {
                    'is_low_point': False,
                    'confidence': 0.0,
                    'final_confidence': 0.0,
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
        # 🔧 修复：确保数据包含技术指标
        if 'rsi' not in data.columns or 'macd' not in data.columns:
            self.logger.warning("数据缺少技术指标，跳过预处理...")
            # 注意：这里我们假设外部已经处理了数据预处理
            # 如果确实需要在这里处理，可以添加数据模块调用

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

            # 步骤A: 策略参数优化
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🔧 步骤A: 策略参数优化 [{current_time}]")
            print("   🎯 目标: 寻找最优策略参数组合")
            print("   📊 方法: 遗传算法高精度优化")
            
            # 🔧 从配置文件中读取固定参数值
            strategy_config = self.config.get('strategy', {})
            fixed_rise_threshold = strategy_config.get('rise_threshold', 0.04)
            fixed_max_days = strategy_config.get('max_days', 20)
            print(f"   🔒 固定参数: rise_threshold={fixed_rise_threshold}, max_days={fixed_max_days}")

            # 🔧 获取当前策略基准得分
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"   📊 评估当前策略基准得分... [{current_time}]")

            current_backtest = strategy_module.backtest(data)
            current_evaluation = strategy_module.evaluate_strategy(current_backtest)
            baseline_score = current_evaluation.get('score', 0)

            print(f"   📈 当前策略基准得分: {baseline_score:.6f}")
            self.logger.info(f"当前策略基准得分: {baseline_score:.6f}")

            step_a_start = time.time()

            strategy_result = self.optimize_strategy_parameters_improved(strategy_module, data)
            step_a_time = time.time() - step_a_start
            optimization_result['strategy_optimization'] = strategy_result

            if strategy_result['success']:
                optimized_score = strategy_result.get('best_score', 0)

                # 🎯 关键修复：只有新得分更高才更新策略参数
                if optimized_score > baseline_score:
                    # 更新策略模块参数
                    strategy_module.update_params(strategy_result['best_params'])
                    score_improvement = optimized_score - baseline_score

                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"✅ 步骤A完成 - 策略得分提升! (耗时: {step_a_time:.2f}s) [{current_time}]")
                    print(f"   🎯 优化方法: {strategy_result.get('optimization_method', 'unknown')}")
                    print(f"   📈 优化前得分: {baseline_score:.6f}")
                    print(f"   📈 优化后得分: {optimized_score:.6f}")
                    print(
                        f"   🚀 得分提升: +{score_improvement:.6f} ({(score_improvement / baseline_score * 100):.2f}%)")
                    print(f"   📊 测试集成功率: {strategy_result.get('test_success_rate', 0):.2%}")

                    self.logger.info(f"✅ 步骤A完成 - 策略得分提升! (耗时: {step_a_time:.2f}s)")
                    self.logger.info(f"   📈 优化前得分: {baseline_score:.6f}")
                    self.logger.info(f"   📈 优化后得分: {optimized_score:.6f}")
                    self.logger.info(f"   🚀 得分提升: +{score_improvement:.6f}")
                else:
                    # 优化得分未超过基准，保持原参数不变
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"⚠️ 步骤A完成但无改进 (耗时: {step_a_time:.2f}s) - 保持原参数 [{current_time}]")
                    print(f"   🎯 优化方法: {strategy_result.get('optimization_method', 'unknown')}")
                    print(f"   📈 当前得分: {baseline_score:.6f}")
                    print(f"   📈 优化得分: {optimized_score:.6f}")
                    print(f"   📉 未达到改进阈值，保持原策略参数")

                    self.logger.info(f"⚠️ 步骤A完成但无改进 (耗时: {step_a_time:.2f}s)")
                    self.logger.info(f"   📈 当前得分: {baseline_score:.6f} > 优化得分: {optimized_score:.6f}")
                    optimization_result['errors'].append("优化后得分未超过基准，保持原参数")

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

            # 设置最终成功状态和最佳得分
            optimization_result['success'] = model_result['success']

            # 设置最佳得分 - 优先使用策略优化的得分
            if strategy_result.get('success') and 'best_score' in strategy_result:
                optimization_result['best_score'] = strategy_result['best_score']
            elif model_result.get('success') and 'score' in model_result:
                optimization_result['best_score'] = model_result['score']
            elif evaluation_result.get('success') and 'score' in evaluation_result:
                optimization_result['best_score'] = evaluation_result['score']
            else:
                optimization_result['best_score'] = 0.0

            # 总结报告
            total_time = time.time() - complete_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print("=" * 80)
            print(f"🎉 完整AI优化流程完成! [{current_time}]")
            print(f"⏱️ 总耗时: {total_time:.2f}s ({total_time / 60:.1f}分钟)")
            print("📊 各步骤耗时分析:")
            print(f"   🔧 策略优化: {step_a_time:.2f}s ({(step_a_time / total_time) * 100:.1f}%)")
            print(f"   🤖 模型训练: {step_b_time:.2f}s ({(step_b_time / total_time) * 100:.1f}%)")
            print(f"   📊 性能评估: {step_c_time:.2f}s ({(step_c_time / total_time) * 100:.1f}%)")
            print(f"   💾 结果保存: {step_d_time:.2f}s ({(step_d_time / total_time) * 100:.1f}%)")

            self.logger.info("=" * 80)
            self.logger.info("🎉 完整AI优化流程完成!")
            self.logger.info(f"⏱️ 总耗时: {total_time:.2f}s ({total_time / 60:.1f}分钟)")
            self.logger.info("📊 各步骤耗时分析:")
            self.logger.info(f"   🔧 策略优化: {step_a_time:.2f}s ({(step_a_time / total_time) * 100:.1f}%)")
            self.logger.info(f"   🤖 模型训练: {step_b_time:.2f}s ({(step_b_time / total_time) * 100:.1f}%)")
            self.logger.info(f"   📊 性能评估: {step_c_time:.2f}s ({(step_c_time / total_time) * 100:.1f}%)")
            self.logger.info(f"   💾 结果保存: {step_d_time:.2f}s ({(step_d_time / total_time) * 100:.1f}%)")

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
                'elapsed_time': elapsed_time,
                'best_score': 0.0
            }

    def optimize_strategy_parameters_improved(self, strategy_module, data: pd.DataFrame) -> Dict[str, Any]:
        """
        改进版策略参数优化（集成贝叶斯优化和遗传算法的高精度模式）
        
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

        self.logger.info("🚀 开始改进版策略参数优化（集成贝叶斯优化和遗传算法）")
        self.logger.info("=" * 80)

        try:
            # 步骤1: 数据分割与验证
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    📊 子步骤1: 数据分割与验证 [{current_time}]")
            self.logger.info("📊 步骤1: 数据分割与验证...")
            split_start_time = time.time()

            # 使用配置中的数据分割比例
            train_ratio = self.config.get('validation', {}).get('train_ratio', 0.7)
            validation_ratio = self.config.get('validation', {}).get('validation_ratio', 0.2)
            test_ratio = self.config.get('validation', {}).get('test_ratio', 0.1)

            # 数据分割
            train_size = int(len(data) * train_ratio)
            val_size = int(len(data) * validation_ratio)

            train_data = data.iloc[:train_size].copy()
            validation_data = data.iloc[train_size:train_size + val_size].copy()
            test_data = data.iloc[train_size + val_size:].copy()

            split_time = time.time() - split_start_time

            print(f"    ✅ 数据分割完成 (耗时: {split_time:.2f}s)")
            print(f"       📊 训练集: {len(train_data)}条 ({train_ratio * 100:.1f}%)")
            print(f"       📈 验证集: {len(validation_data)}条 ({validation_ratio * 100:.1f}%)")
            print(f"       🔒 测试集: {len(test_data)}条 ({test_ratio * 100:.1f}%)")

            self.logger.info(f"✅ 数据分割完成 (耗时: {split_time:.2f}s):")
            self.logger.info(f"   📊 训练集: {len(train_data)}条 ({train_ratio * 100:.1f}%) - 仅用于参数优化")
            self.logger.info(
                f"   📈 验证集: {len(validation_data)}条 ({validation_ratio * 100:.1f}%) - 用于模型验证和过拟合检测")
            self.logger.info(f"   🔒 测试集: {len(test_data)}条 ({test_ratio * 100:.1f}%) - 完全锁定，仅最终评估")
            self.logger.info("-" * 50)

            # 步骤2: 选择优化方法（优先贝叶斯优化）
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    🔧 子步骤2: 参数优化方法选择 [{current_time}]")
            self.logger.info("🔧 步骤2: 参数优化方法选择...")

            # 检查配置
            bayesian_config = self.config.get('bayesian_optimization', {})
            bayesian_enabled = bayesian_config.get('enabled', True)
            genetic_config = self.config.get('genetic_algorithm', {})
            genetic_enabled = genetic_config.get('enabled', True)
            advanced_config = self.config.get('advanced_optimization', {})
            advanced_enabled = advanced_config.get('enabled', True)

            if not advanced_config:
                print("    ⚠️ advanced_optimization配置缺失，使用默认值")
                self.logger.warning("advanced_optimization配置缺失，使用默认值")

            best_params = {}
            best_score = -float('inf')
            optimization_method = 'unknown'

            # 🔧 修复：保存初始策略参数作为基准
            initial_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                             'get_current_params') else {}
            if initial_params:
                # 评估初始参数作为基准
                initial_backtest = strategy_module.backtest(train_data)
                initial_evaluation = strategy_module.evaluate_strategy(initial_backtest)
                initial_score = initial_evaluation.get('score', 0)

                best_params = initial_params.copy()
                best_score = initial_score

                print(f"    📊 初始参数基准得分: {initial_score:.6f}")
                self.logger.info(f"📊 初始参数基准得分: {initial_score:.6f}")

            # 🔧 从配置文件中读取固定参数值
            strategy_config = self.config.get('strategy', {})
            fixed_rise_threshold = strategy_config.get('rise_threshold', 0.04)
            fixed_max_days = strategy_config.get('max_days', 20)
            
            print(f"   🔒 固定参数: rise_threshold={fixed_rise_threshold}, max_days={fixed_max_days}")
            self.logger.info(f"🔒 固定参数: rise_threshold={fixed_rise_threshold}, max_days={fixed_max_days}")

            # 优先尝试贝叶斯优化
            if bayesian_enabled and advanced_enabled:
                print("    🔬 选择贝叶斯优化进行高精度参数优化")
                print("    🎯 配置参数: 高斯过程回归 + 期望改进采集函数")
                print("    ⏳ 预计耗时: 10-20分钟（智能搜索）")

                self.logger.info("🔬 选择贝叶斯优化进行高精度参数优化")
                bayesian_start_time = time.time()

                try:
                    # 使用参数优化器进行贝叶斯优化
                    from .parameter_optimizer import ParameterOptimizer
                    param_optimizer = ParameterOptimizer(self.config)

                    # 🔧 启用外部参数管理，避免parameter_optimizer内部管理参数
                    param_optimizer._external_best_management = True

                    # 获取参数范围
                    param_ranges = self._get_enhanced_parameter_ranges({})

                    # 🔧 为贝叶斯优化实现相同的参数管理逻辑
                    current_best_params_in_bayesian = initial_params.copy()
                    current_best_score_in_bayesian = best_score

                    # 定义贝叶斯优化的评估包装函数
                    original_evaluate = param_optimizer._evaluate_parameters

                    def bayesian_evaluate_wrapper(strategy_module, data, params):
                        nonlocal current_best_params_in_bayesian, current_best_score_in_bayesian

                        # 调用原始评估（现在会恢复原始参数）
                        score, metrics = original_evaluate(strategy_module, data, params)

                        # 🎯 修复后的参数管理逻辑：只有更优参数才保留
                        if score > current_best_score_in_bayesian:
                            # 新参数更优，应用到策略模块
                            prev_score = current_best_score_in_bayesian
                            strategy_module.update_params(params)
                            current_best_params_in_bayesian = params.copy()
                            current_best_score_in_bayesian = score
                            self.logger.info(f"贝叶斯优化发现更优参数: 得分 {score:.6f} > {prev_score:.6f}")
                        else:
                            # 新参数较差，确保策略模块恢复到当前最佳参数
                            if current_best_params_in_bayesian:
                                strategy_module.update_params(current_best_params_in_bayesian)

                        return score, metrics

                    # 替换评估函数
                    param_optimizer._evaluate_parameters = bayesian_evaluate_wrapper

                    # 运行贝叶斯优化
                    print(f"    🚀 开始贝叶斯优化参数搜索... [{datetime.now().strftime('%H:%M:%S')}]")
                    self.logger.info("🚀 开始贝叶斯优化参数搜索...")

                    bayesian_result = param_optimizer.optimize_parameters(
                        strategy_module, train_data, param_ranges,
                        method='bayesian', max_iterations=120
                    )

                    bayesian_time = time.time() - bayesian_start_time

                    if bayesian_result.get('success') and bayesian_result.get('best_params'):
                        # 🔧 修复：贝叶斯优化已经通过包装函数管理了最佳参数
                        # 获取当前策略模块中的参数（应该是最佳的）
                        final_bayesian_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                                                'get_current_params') else \
                        bayesian_result['best_params']

                        # 最终评估贝叶斯优化结果（策略模块已经是最佳状态）
                        bayesian_backtest = strategy_module.backtest(train_data)
                        bayesian_evaluation = strategy_module.evaluate_strategy(bayesian_backtest)
                        bayesian_score = bayesian_evaluation.get('score', 0)

                        best_params = final_bayesian_params.copy()
                        best_score = bayesian_score
                        optimization_method = 'bayesian_optimization'

                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"    🔬 贝叶斯优化完成 (耗时: {bayesian_time:.2f}s) [{current_time}]")
                        print(f"       📈 最优得分: {bayesian_score:.6f}")
                        print(f"       📊 成功率: {bayesian_evaluation.get('success_rate', 0):.2%}")
                        print(f"       🔍 识别点数: {bayesian_evaluation.get('total_points', 0)}")
                        print(f"       📈 平均涨幅: {bayesian_evaluation.get('avg_rise', 0):.2%}")
                        print(
                            f"       🔧 收敛信息: {bayesian_result.get('convergence_info', {}).get('n_calls', 0)} 次函数调用")

                        self.logger.info(f"🔬 贝叶斯优化完成 (耗时: {bayesian_time:.2f}s)")
                        self.logger.info(f"   最优得分: {bayesian_score:.6f}")
                        self.logger.info(f"   成功率: {bayesian_evaluation.get('success_rate', 0):.2%}")
                        self.logger.info(f"   识别点数: {bayesian_evaluation.get('total_points', 0)}")
                        self.logger.info(f"   平均涨幅: {bayesian_evaluation.get('avg_rise', 0):.2%}")
                    else:
                        print("    ⚠️ 贝叶斯优化未找到有效解，回退到遗传算法")
                        self.logger.warning("⚠️ 贝叶斯优化未找到有效解，回退到遗传算法")
                        bayesian_enabled = False

                except Exception as e:
                    print(f"    ❌ 贝叶斯优化失败: {e}，回退到遗传算法")
                    self.logger.error(f"❌ 贝叶斯优化失败: {e}，回退到遗传算法")
                    bayesian_enabled = False

            # 如果贝叶斯优化失败或未启用，使用遗传算法
            if (not bayesian_enabled or not best_params) and genetic_enabled and advanced_enabled:
                print("    🧬 使用遗传算法进行参数优化")
                print("    🎯 配置参数: 200个体 × 20代 = 4000次评估")
                print("    ⏳ 预计耗时: 15-30分钟（进化搜索）")

                self.logger.info("🧬 使用遗传算法进行参数优化")
                genetic_start_time = time.time()

                # 🔧 关键修复：定义不影响策略模块状态的评估函数
                current_best_params_in_genetic = initial_params.copy()
                current_best_score_in_genetic = best_score

                def evaluate_strategy_params(params):
                    nonlocal current_best_params_in_genetic, current_best_score_in_genetic

                    try:
                        # 🚨 重要：添加固定参数（从配置文件读取）
                        complete_params = params.copy()
                        complete_params['rise_threshold'] = fixed_rise_threshold  # 从配置文件读取
                        complete_params['max_days'] = fixed_max_days  # 从配置文件读取

                        # 🔧 关键修复：保存当前策略模块状态
                        original_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                                          'get_current_params') else None

                        # 临时应用参数进行评估
                        print("    临时应用参数进行评估")
                        strategy_module.update_params(complete_params)

                        # 在训练集上评估
                        backtest_results = strategy_module.backtest(train_data)
                        evaluation = strategy_module.evaluate_strategy(backtest_results)

                        # 计算评分
                        score = evaluation.get('score', 0)
                        success_rate = evaluation.get('success_rate', 0)
                        avg_rise = evaluation.get('avg_rise', 0)

                        # 高精度评分：更重视成功率
                        final_score = (
                                success_rate * 0.7 +  # 70%权重给成功率
                                min(avg_rise / 0.1, 1.0) * 0.2 +  # 20%权重给涨幅（最高10%）
                                score * 0.1  # 10%权重给综合分
                        )
                        final_score = max(0.0, min(1.0, final_score))

                        # 🎯 修复后的参数管理逻辑：只有更好的参数才保留在策略模块中
                        if final_score > current_best_score_in_genetic:
                            # 新参数更好，保留在策略模块中
                            prev_score = current_best_score_in_genetic
                            current_best_params_in_genetic = complete_params.copy()
                            current_best_score_in_genetic = final_score
                            # 策略模块已经更新为新参数，不需要额外操作
                            self.logger.info(f"遗传算法发现更优参数: 得分 {final_score:.6f} > {prev_score:.6f}")
                        else:
                            # 新参数较差，必须恢复到之前的最佳参数
                            if current_best_params_in_genetic:
                                strategy_module.update_params(current_best_params_in_genetic)
                            else:
                                # 如果没有最佳参数，恢复到原始参数
                                if original_params:
                                    strategy_module.update_params(original_params)

                        return final_score

                    except Exception as e:
                        self.logger.warning(f"参数评估失败: {e}")
                        # 出错时恢复到最佳参数或原始参数
                        if current_best_params_in_genetic:
                            strategy_module.update_params(current_best_params_in_genetic)
                        elif original_params:
                            strategy_module.update_params(original_params)
                        return -1.0

                # 运行遗传算法
                print(f"    🔬 开始遗传算法参数搜索... [{datetime.now().strftime('%H:%M:%S')}]")
                self.logger.info("🔬 开始遗传算法参数搜索...")
                genetic_params = self.run_genetic_algorithm(evaluate_strategy_params)
                genetic_time = time.time() - genetic_start_time

                if genetic_params:
                    # 🔧 修复：遗传算法已经通过评估函数管理了最佳参数
                    # 获取遗传算法过程中找到的最佳参数（已经在策略模块中）
                    final_genetic_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                                           'get_current_params') else genetic_params

                    # 最终评估遗传算法结果（此时策略模块已经是最佳状态）
                    genetic_backtest = strategy_module.backtest(train_data)
                    genetic_evaluation = strategy_module.evaluate_strategy(genetic_backtest)
                    genetic_score = genetic_evaluation.get('score', 0)

                    # 如果遗传算法结果更好，更新全局最佳参数
                    if genetic_score > best_score:
                        best_params = final_genetic_params.copy()  # 使用遗传算法管理的最佳参数
                        best_score = genetic_score
                        optimization_method = 'genetic_algorithm'

                        print(f"    ✅ 遗传算法找到更优参数! 得分提升: {best_score:.6f}")
                        self.logger.info(f"✅ 遗传算法找到更优参数! 得分提升: {best_score:.6f}")
                    else:
                        print(f"    ⚠️ 遗传算法结果未超过当前最优，恢复之前最佳参数")
                        self.logger.info(f"⚠️ 遗传算法结果未超过当前最优，恢复之前最佳参数")
                        # 恢复到之前的最佳参数
                        strategy_module.update_params(best_params)

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
                    print("    ⚠️ 遗传算法未找到有效解")
                    self.logger.warning("⚠️ 遗传算法未找到有效解")

            # 验证最佳参数
            if not best_params:
                print("    ❌ 所有优化方法都未找到有效参数")
                return {
                    'success': False,
                    'error': '所有优化方法都未找到有效参数'
                }

            # 🔧 关键修复：确保策略模块应用最终的最佳参数
            print(f"    🎯 应用最佳参数到策略模块 (得分: {best_score:.6f})")
            self.logger.info(f"🎯 应用最佳参数到策略模块 (得分: {best_score:.6f})")
            strategy_module.update_params(best_params)

            self.logger.info("-" * 60)

            # 步骤3: 验证集验证
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    📈 子步骤3: 验证集验证 [{current_time}]")
            self.logger.info("📈 步骤3: 在验证集上验证最佳参数...")
            validation_start_time = time.time()

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
            print(
                f"       泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")

            self.logger.info(f"✅ 测试集评估完成 (耗时: {test_time:.2f}s)")
            self.logger.info(f"   得分: {test_score:.6f}")
            self.logger.info(f"   成功率: {test_success_rate:.2%}")
            self.logger.info(f"   识别点数: {test_total_points}")
            self.logger.info(f"   平均涨幅: {test_avg_rise:.2%}")
            self.logger.info(
                f"   泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")

            # 总结
            optimization_total_time = time.time() - optimization_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    🎉 策略参数优化子流程完成! [{current_time}]")
            print(f"    ⏱️ 总耗时: {optimization_total_time:.2f}s ({optimization_total_time / 60:.1f}分钟)")
            print(f"    🔧 优化方法: {optimization_method}")
            print(f"    📊 三层验证结果:")
            print(f"       训练集得分: {best_score:.6f}")
            print(f"       验证集得分: {val_score:.6f} | 成功率: {val_success_rate:.2%}")
            print(f"       测试集得分: {test_score:.6f} | 成功率: {test_success_rate:.2%}")
            print(f"       🛡️ 过拟合检测: {'通过' if overfitting_passed else '警告'}")
            print(f"       🎯 泛化能力: {'良好' if generalization_passed else '一般'}")

            self.logger.info("=" * 80)
            self.logger.info(f"🎉 策略参数优化完成!")
            self.logger.info(f"⏱️ 总耗时: {optimization_total_time:.2f}s ({optimization_total_time / 60:.1f}分钟)")
            self.logger.info(f"🔧 优化方法: {optimization_method}")
            self.logger.info(f"📊 三层验证结果:")
            self.logger.info(f"   训练集得分: {best_score:.6f}")
            self.logger.info(f"   验证集得分: {val_score:.6f} | 成功率: {val_success_rate:.2%}")
            self.logger.info(f"   测试集得分: {test_score:.6f} | 成功率: {test_success_rate:.2%}")
            self.logger.info(f"   🛡️ 过拟合检测: {'通过' if overfitting_passed else '警告'}")
            self.logger.info(f"   🎯 泛化能力: {'良好' if generalization_passed else '一般'}")

            # 如果使用了遗传算法，输出详细的参数信息
            if optimization_method == 'genetic_algorithm':
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
                'genetic_algorithm_used': optimization_method == 'genetic_algorithm'
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

        # 🔧 从配置文件中读取参数范围，而不是硬编码
        optimization_ranges = self.config.get('optimization_ranges', {})
        
        # 转换配置文件格式为搜索格式
        default_ranges = {}
        for param_name, param_config in optimization_ranges.items():
            default_ranges[param_name] = {
                'min': param_config.get('min', 0),
                'max': param_config.get('max', 1),
                'step': param_config.get('step', 0.01)
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
                from src.utils.config_saver import CommentPreservingConfigSaver
                saver = CommentPreservingConfigSaver()
                saver.save_optimized_parameters(params)
                self.logger.info("参数已保存（保留注释版本）")
                return
            except ImportError as e:
                self.logger.warning(f"ruamel.yaml模块未安装，使用传统保存方式: {e}")
            except Exception as e:
                self.logger.warning(f"保留注释版本保存失败，使用传统方式: {e}")

            # 使用传统方式保存
            self._save_params_fallback(params)

        except Exception as e:
            self.logger.error(f"保存优化参数失败: {e}")
            raise

    def _save_params_fallback(self, params: dict):
        """
        传统的参数保存方式（原子性写入）
        
        参数:
        params: 优化后的参数
        """
        import tempfile
        import shutil

        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj

        # 转换参数
        converted_params = convert_numpy_types(params)

        try:
            config_path = 'config/strategy.yaml'
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
            for param_name, param_value in converted_params.items():
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

            self.logger.info(f"参数已安全保存到配置文件: {len(converted_params)} 个参数")

            # 验证保存是否成功
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        saved_config = yaml.safe_load(f)
                    # 验证参数是否正确保存
                    saved_count = 0
                    for param_name in converted_params.keys():
                        if param_name == 'final_threshold':
                            if saved_config.get('strategy', {}).get('confidence_weights', {}).get(
                                    'final_threshold') is not None:
                                saved_count += 1
                        elif param_name in ['rsi_oversold_threshold', 'rsi_low_threshold']:
                            if saved_config.get('strategy', {}).get('confidence_weights', {}).get(
                                    param_name) is not None:
                                saved_count += 1
                        else:
                            if saved_config.get('strategy', {}).get(param_name) is not None:
                                saved_count += 1

                    self.logger.info(f"验证成功: {saved_count}/{len(converted_params)} 个参数已正确保存")

                    # 清理旧备份
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                except Exception as verify_error:
                    self.logger.warning(f"参数保存验证失败: {verify_error}")
            else:
                self.logger.error("配置文件保存后不存在")

        except Exception as e:
            self.logger.error(f"传统方式保存参数失败: {e}")
            # 尝试从备份恢复
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, config_path)
                    self.logger.info("已从备份恢复配置文件")
                except Exception as restore_error:
                    self.logger.error(f"备份恢复失败: {restore_error}")
            raise

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
            generations = genetic_config.get('generations', 30)  # 增加到30
            crossover_rate = genetic_config.get('crossover_rate', 0.8)
            mutation_rate = genetic_config.get('mutation_rate', 0.15)  # 稍微提高变异率
            elite_ratio = genetic_config.get('elite_ratio', 0.1)  # 保留10%精英

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

            # 收敛检测相关变量
            convergence_history = []  # 记录最近几代的收敛信息
            convergence_threshold = 0.001  # 收敛阈值
            convergence_generations = 3  # 连续收敛代数要求

            # 进化主循环
            for generation in range(generations):
                generation_start_time = time.time()
                current_time = datetime.now().strftime("%H:%M:%S")

                print(
                    f"        🧬 第 {generation + 1}/{generations} 代进化 ({((generation + 1) / generations) * 100:.1f}% 完成) [{current_time}]")
                self.logger.info(f"\n🧬 第 {generation + 1}/{generations} 代进化 "
                                 f"({((generation + 1) / generations) * 100:.1f}% 完成)")
                self.logger.info("------------------------------------------------------------")

                # 评估当前种群
                scores = []
                valid_evaluations = 0
                failed_evaluations = 0

                # 每10个个体显示一次进度，避免过多输出
                for i, individual in enumerate(population):
                    if (i + 1) % 10 == 0 or i == len(population) - 1:
                        current_progress = ((generation * population_size + i + 1) / total_evaluations) * 100
                        print(
                            f"        🔍 评估进度: {i + 1}/{population_size} 个体 | 总进度: {current_progress:.1f}% | 第{generation + 1}代")
                        self.logger.info(f"🔍 评估进度: {i + 1}/{population_size} 个体 | "
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
                    print(f"           ⏳ 预计剩余时间: {estimated_remaining:.1f}s ({estimated_remaining / 60:.1f}分钟)")

                    self.logger.info(f"📊 第{generation + 1}代统计:")
                    self.logger.info(f"   ✅ 有效个体: {valid_evaluations}/{population_size}")
                    self.logger.info(f"   ❌ 失败个体: {failed_evaluations}")
                    self.logger.info(f"   📈 最高分: {max_score:.6f}")
                    self.logger.info(f"   📊 平均分: {avg_score:.6f}")
                    self.logger.info(f"   📉 最低分: {min_score:.6f}")
                    self.logger.info(f"   📏 标准差: {std_score:.6f}")
                    self.logger.info(f"   🏆 历史最优: {best_score:.6f}")
                    self.logger.info(f"   ⏱️ 本代耗时: {generation_time:.2f}s")
                    self.logger.info(
                        f"   ⏳ 预计剩余时间: {estimated_remaining:.1f}s ({estimated_remaining / 60:.1f}分钟)")

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

                # 🔧 新增：连续3代收敛检测
                if len(recent_generations) >= 2:
                    # 记录当前代的收敛信息
                    current_convergence = {
                        'generation': generation + 1,
                        'best_score': best_score,
                        'max_score': max_score,
                        'std_score': std_score
                    }
                    convergence_history.append(current_convergence)

                    # 保持最近的convergence_generations代记录
                    if len(convergence_history) > convergence_generations:
                        convergence_history = convergence_history[-convergence_generations:]

                    # 检测是否连续收敛
                    if len(convergence_history) >= convergence_generations:
                        is_converged = self._check_convergence(convergence_history, convergence_threshold)

                        if is_converged:
                            print(f"        🎯 检测到连续{convergence_generations}代收敛，提前停止优化")
                            print(f"        📊 收敛阈值: {convergence_threshold:.6f}")
                            print(f"        🏆 最终得分: {best_score:.6f}")

                            self.logger.info(f"🎯 检测到连续{convergence_generations}代收敛，提前停止优化")
                            self.logger.info(f"📊 收敛阈值: {convergence_threshold:.6f}")
                            self.logger.info(f"🏆 最终得分: {best_score:.6f}")
                            break

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

                print(f"        {'=' * 60}")
                self.logger.info("------------------------------------------------------------")

            total_time = time.time() - start_time
            current_time = datetime.now().strftime('%H:%M:%S')

            print(f"        🎉 遗传算法优化完成! [{current_time}]")
            print(f"        ⏱️ 总耗时: {total_time:.2f}s ({total_time / 60:.1f}分钟)")
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
        获取增强的参数范围（使用配置文件中的范围）
        
        参数:
        base_ranges: 基础参数范围
        
        返回:
        dict: 增强的参数范围
        """
        # 🚨 重要：固定参数，不参与遗传算法优化
        # 从配置文件中读取固定参数值
        strategy_config = self.config.get('strategy', {})
        fixed_rise_threshold = strategy_config.get('rise_threshold', 0.04)
        fixed_max_days = strategy_config.get('max_days', 20)

        # 从配置文件中获取参数范围
        config = self.config
        strategy_ranges = config.get('strategy_ranges', {})
        optimization_ranges = config.get('optimization_ranges', {})

        enhanced_ranges = {}

        # 添加strategy_ranges中的参数
        for param_name, param_config in strategy_ranges.items():
            # 🚨 跳过固定参数，不允许优化
            if param_name in ['rise_threshold', 'max_days']:
                self.logger.info(f"⚠️ 跳过固定参数 {param_name}，此参数不参与优化")
                continue

            # 转换配置格式
            enhanced_ranges[param_name] = {
                'min': param_config.get('min', 0),
                'max': param_config.get('max', 1),
                'type': 'int' if param_name.endswith('_threshold') and 'rsi' in param_name else 'float',
                'precision': 4
            }

        # 添加optimization_ranges中的参数
        for param_name, param_config in optimization_ranges.items():
            enhanced_ranges[param_name] = {
                'min': param_config.get('min', 0),
                'max': param_config.get('max', 1),
                'type': 'float',
                'precision': 4
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

        self.logger.info(f"🎯 参数搜索空间: {len(enhanced_ranges)} 个参数")
        self.logger.info(f"🔒 固定参数: rise_threshold={fixed_rise_threshold}, max_days={fixed_max_days} (不参与优化)")

        # 记录参数范围
        for param_name, param_config in enhanced_ranges.items():
            self.logger.info(f"   {param_name}: {param_config['min']} - {param_config['max']} ({param_config['type']})")

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
            avg_scores = [gen.get('avg_score', 0) for gen in recent_generations if
                          gen.get('avg_score') is not None and gen.get('avg_score') != -1.0]
            max_scores = [gen.get('max_score', 0) for gen in recent_generations if
                          gen.get('max_score') is not None and gen.get('max_score') != -1.0]
            generation_times = [gen.get('generation_time', 0) for gen in recent_generations if
                                gen.get('generation_time') is not None]

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

    def _check_convergence(self, convergence_history: List[Dict], threshold: float) -> bool:
        """
        检测遗传算法是否收敛
        
        参数:
        convergence_history: 最近几代的收敛信息
        threshold: 收敛阈值
        
        返回:
        bool: 是否收敛
        """
        try:
            if len(convergence_history) < 3:
                return False

            # 提取最近3代的得分
            scores = [gen['best_score'] for gen in convergence_history[-3:]]
            std_scores = [gen['std_score'] for gen in convergence_history[-3:]]

            # 条件1：最佳得分变化小于阈值
            score_changes = [abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))]
            score_stable = all(change < threshold for change in score_changes)

            # 条件2：种群标准差都很小（表示种群收敛）
            std_threshold = 0.01  # 标准差阈值
            std_stable = all(std < std_threshold for std in std_scores)

            # 条件3：连续改善幅度很小
            improvements = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
            improvement_stable = all(abs(imp) < threshold for imp in improvements)

            # 收敛判断：得分稳定且种群收敛，或者改善幅度很小
            is_converged = (score_stable and std_stable) or improvement_stable

            if is_converged:
                self.logger.info(f"收敛检测详情:")
                self.logger.info(f"  得分变化: {score_changes}")
                self.logger.info(f"  标准差: {std_scores}")
                self.logger.info(f"  改善幅度: {improvements}")
                self.logger.info(f"  得分稳定: {score_stable}, 种群收敛: {std_stable}, 改善微小: {improvement_stable}")

            return is_converged

        except Exception as e:
            self.logger.warning(f"收敛检测失败: {e}")
            return False

    def _save_optimized_parameters(self, best_params: Dict[str, Any]) -> bool:
        """
        保存优化后的参数到配置文件
        
        参数:
        best_params: 最优参数字典
        
        返回:
        bool: 是否保存成功
        """
        try:
            from src.utils.config_saver import save_strategy_config

            # 构建策略参数字典
            strategy_params = {}

            # 基础参数
            if 'rise_threshold' in best_params:
                strategy_params['rise_threshold'] = best_params['rise_threshold']
            if 'max_days' in best_params:
                strategy_params['max_days'] = best_params['max_days']

            # 置信度权重参数
            confidence_weights = {}
            confidence_weight_keys = [
                'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
                'dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight'
            ]

            for key in confidence_weight_keys:
                if key in best_params:
                    confidence_weights[key] = best_params[key]

            if confidence_weights:
                strategy_params['confidence_weights'] = confidence_weights

            # 保存到配置文件
            if strategy_params:
                success = save_strategy_config(strategy_params)

                if success:
                    print(f"   💾 最优参数已保存到配置文件")
                    self.logger.info("💾 最优参数已保存到配置文件")
                    self.logger.info(f"   保存的参数: {strategy_params}")
                    return True
                else:
                    print(f"   ⚠️ 参数保存失败，但优化结果仍然有效")
                    self.logger.warning("参数保存失败，但优化结果仍然有效")
                    return False
            else:
                self.logger.info("没有需要保存的策略参数")
                return True

        except ImportError as e:
            self.logger.warning(f"配置保存模块不可用: {e}")
            print(f"   ⚠️ 配置保存模块不可用，参数未持久化")
            return False
        except Exception as e:
            self.logger.error(f"保存优化参数失败: {e}")
            print(f"   ❌ 参数保存失败: {e}")
            return False
