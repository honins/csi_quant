#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化器
集成增量学习、特征权重优化和趋势确认指标
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from typing import Dict, Any, Tuple, List, Optional
import json
import yaml
from itertools import product
import sys
import time

# 贝叶斯优化相关导入

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
BAYESIAN_AVAILABLE = True

# 导入工具函数
from src.utils.utils import resolve_confidence_param




class AIOptimizerImproved:
    """AI优化器"""

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

        # 移除置信度处理器 - 使用模型原始输出
        # self.confidence_processor = ConfidenceProcessor(config)

        self.logger.info("AI优化器初始化完成")

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

                # 使用warm_start进行增量学习（仅当为Pipeline且底层分类器支持）
                classifier = None
                if hasattr(self.model, 'named_steps') and 'classifier' in getattr(self.model, 'named_steps', {}):
                    classifier = self.model.named_steps['classifier']
                else:
                    self.logger.warning("当前模型不是Pipeline（可能为校准后的模型），回退到完全重训练")
                    return self.full_train(new_data, strategy_module)

                if hasattr(classifier, 'n_estimators'):
                    classifier.n_estimators += 10  # 增加树的数量
                    classifier.warm_start = True

                    # 重新训练（这里实际上是增量的）
                    classifier.fit(recent_features_scaled, recent_labels)

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
        完整训练AI模型
        
        参数:
        data: 历史数据
        strategy_module: 策略模块
        
        返回:
        dict: 训练结果
        """
        train_start_time = time.time()
        self.logger.info("🤖 开始AI模型完整训练")
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
            # 保存scaler供增量训练复用
            try:
                self.scaler = model.named_steps['scaler'] if hasattr(model, 'named_steps') and 'scaler' in model.named_steps else None
            except Exception:
                self.scaler = None

            self.logger.info(f"✅ 模型训练完成 (耗时: {model_time:.2f}s)")
            self.logger.info("-" * 60)

            # === 概率校准（Calibration）开始 ===
            try:
                calib_cfg = {}
                if isinstance(self.config, dict):
                    calib_cfg = (self.config.get('calibration')
                                 or self.config.get('ai', {}).get('calibration')
                                 or {})
                calib_enabled = bool(calib_cfg.get('enabled', True))
                calib_method = calib_cfg.get('method', 'isotonic')
                calib_cv = calib_cfg.get('cv', 'prefit')
                holdout_size = int(calib_cfg.get('holdout_size', max(30, int(len(features) * 0.2))))
                holdout_size = min(holdout_size, 120)

                if calib_enabled:
                    self.logger.info("📏 启用概率校准: method=%s, cv=%s", calib_method, str(calib_cv))
                    if calib_cv == 'prefit':
                        if len(features) - holdout_size < 50:
                            self.logger.warning("样本较少，prefit 留出集不足，自动退化为 cv=3")
                            calib_cv = 3
                        else:
                            split_idx = len(features) - holdout_size
                            X_train_base = features[:split_idx]
                            y_train_base = labels[:split_idx]
                            X_calib = features[split_idx:]
                            y_calib = labels[split_idx:]
                            w_train_base = sample_weights[:split_idx] if sample_weights is not None else None

                            base_model = Pipeline([
                                ('scaler', StandardScaler()),
                                ('classifier', RandomForestClassifier(
                                    n_estimators=100,
                                    max_depth=8,
                                    min_samples_split=15,
                                    min_samples_leaf=8,
                                    class_weight='balanced',
                                    n_jobs=-1,
                                    random_state=42,
                                    verbose=1
                                ))
                            ])
                            base_model.fit(X_train_base, y_train_base,
                                           classifier__sample_weight=w_train_base)

                            calibrated = CalibratedClassifierCV(
                                estimator=base_model, method=calib_method, cv='prefit'
                            )
                            calibrated.fit(X_calib, y_calib)

                            self.model = calibrated
                            self.logger.info("✅ 概率校准完成（prefit + 留出集=%d）", len(X_calib))
                    if isinstance(calib_cv, int) and calib_cv >= 2:
                        calibrated = CalibratedClassifierCV(
                            estimator=self.model, method=calib_method, cv=calib_cv
                        )
                        try:
                            calibrated.fit(features, labels, sample_weight=sample_weights)
                        except TypeError:
                            self.logger.warning("CalibratedClassifierCV.fit 不支持 sample_weight，已回退为无权重拟合")
                            calibrated.fit(features, labels)
                        self.model = calibrated
                        self.logger.info("✅ 概率校准完成（%d 折CV）", calib_cv)
            except Exception as e_calib:
                self.logger.warning(f"概率校准阶段出现问题，已跳过校准: {e_calib}")
            # === 概率校准结束 ===

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
            self.logger.info("🎉 AI模型训练完成!")
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
            self.logger.error(f"模型训练失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict_low_point(self, data: pd.DataFrame, prediction_date: str = None) -> Dict[str, Any]:
        """
        预测相对低点
        
        参数:
        data: 市场数据
        prediction_date: 预测日期
        
        返回:
        dict: 预测结果
        """
        self.logger.info("预测相对低点")

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

            # 获取原始置信度（不再进行处理）
            raw_confidence = prediction_proba[1] if len(prediction_proba) > 1 else 0.0

            # 直接使用原始置信度，不进行处理
            final_confidence = raw_confidence

            # 使用固定阈值（从配置读取，不进行动态调整）
            final_threshold = resolve_confidence_param(self.config, 'final_threshold', 0.5)

            # 基于原始置信度和配置阈值进行预测
            is_low_point = final_confidence >= final_threshold

            # 安全获取模型类型（兼容 Pipeline 或 CalibratedClassifierCV）
            model_type = type(self.model).__name__
            try:
                if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                    model_type = type(self.model.named_steps['classifier']).__name__
                else:
                    base_est = getattr(self.model, 'base_estimator', getattr(self.model, 'estimator', None))
                    if base_est is not None:
                        if hasattr(base_est, 'named_steps') and 'classifier' in getattr(base_est, 'named_steps', {}):
                            model_type = type(base_est.named_steps['classifier']).__name__
                        else:
                            model_type = type(base_est).__name__
            except Exception:
                pass

            result = {
                'is_low_point': bool(is_low_point),
                'confidence': float(raw_confidence),
                'final_confidence': float(final_confidence),  # 现在等于原始置信度
                'prediction_proba': prediction_proba.tolist(),
                'feature_count': len(feature_names),
                'model_type': model_type,
                'threshold_used': final_threshold
            }

            # 输出预测结果
            self.logger.info("----------------------------------------------------")
            self.logger.info("AI预测结果: \033[1m%s\033[0m",
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

            # 如果是相对路径，转换为绝对路径（相对于项目根目录）
            if not os.path.isabs(model_path):
                # 获取项目根目录（从当前文件位置向上三级：src/ai/ai_optimizer_improved.py -> 项目根目录）
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(project_root, model_path)
            
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

            # 兼容 CalibratedClassifierCV，优先提取底层基学习器
            model_obj = self.model
            base_est = getattr(model_obj, 'base_estimator', getattr(model_obj, 'estimator', None))
            if base_est is not None:
                model_obj = base_est

            # 从Pipeline中获取分类器
            if hasattr(model_obj, 'named_steps') and 'classifier' in getattr(model_obj, 'named_steps', {}):
                classifier = model_obj.named_steps['classifier']
            else:
                # 如果模型不是Pipeline，直接使用
                classifier = model_obj

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
        print(f"🚀 开始完整的AI优化流程 [{current_time}]")
        print("=" * 80)
        self.logger.info("🚀 开始完整的AI优化流程")
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
            print("   🔧 步骤A: 策略参数优化 (贝叶斯优化)")
            print("   🤖 步骤B: 模型训练")
            print("   📊 步骤C: 最终性能评估")
            print("   💾 步骤D: 结果保存")
            print("-" * 80)

            # 步骤A: 策略参数优化
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🔧 步骤A: 策略参数优化 [{current_time}]")
            print("   🎯 目标: 寻找最优策略参数组合")
            print("   📊 方法: 贝叶斯优化高精度搜索")
            
            # 🔧 从配置文件中读取固定参数值
            strategy_config = self.config.get('strategy', {})

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

            # 步骤B: 模型训练
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🤖 步骤B: 模型训练 [{current_time}]")
            print("   🎯 目标: 训练RandomForest分类模型")
            print("   ⚙️ 配置: 150棵树, 深度12, 平衡权重")
            print("   📊 数据: 特征工程 + 样本权重 + 标准化")

            self.logger.info("🤖 步骤B: 模型训练")
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
                print(f"   🔍 交易数: {evaluation_result.get('identified_points', 0)}")
                print(f"   🤖 AI置信度: {evaluation_result.get('ai_confidence', 0):.4f}")

                self.logger.info(f"✅ 步骤C完成 (耗时: {step_c_time:.2f}s)")
                self.logger.info(f"   🎯 策略得分: {evaluation_result.get('strategy_score', 0):.4f}")
                self.logger.info(f"   📊 成功率: {evaluation_result.get('strategy_success_rate', 0):.2%}")
                self.logger.info(f"   🔍 交易数: {evaluation_result.get('identified_points', 0)}")
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
                self._save_optimized_parameters(strategy_result['best_params'])
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
        策略参数优化（贝叶斯优化高精度模式）
        
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

        self.logger.info("🚀 开始策略参数优化（贝叶斯优化）")
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
            # 贝叶斯优化为主要优化方法
            optimization_method_config = self.config.get('optimization_method', 'bayesian')
            advanced_config = self.config.get('advanced_optimization', {})
            advanced_enabled = advanced_config.get('enabled', True)

            if not advanced_config:
                print("    ⚠️ advanced_optimization配置缺失，使用默认值")
                self.logger.warning("advanced_optimization配置缺失，使用默认值")

            best_params = {}
            best_score = -float('inf')
            optimization_method = 'initial_params'

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

       
            # 直接进入贝叶斯优化参数优化流程
            print("    🔬 使用贝叶斯优化进行参数优化")
            print("    🎯 配置参数: 100次评估 (智能搜索)")
            print("    ⏳ 预计耗时: 5-10分钟（高效搜索）")

            self.logger.info("🔬 使用贝叶斯优化进行参数优化")
            bayesian_start_time = time.time()

            # 🔧 关键修复：定义不影响策略模块状态的评估函数
            current_best_params_in_bayesian = initial_params.copy()
            
            # 🔧 修复：计算初始参数的统一评分作为基准
            if initial_params:
                # 评估初始参数
                initial_backtest = strategy_module.backtest(train_data)
                initial_evaluation = strategy_module.evaluate_strategy(initial_backtest)
                
                # 使用统一的评分方法计算初始得分
                initial_unified_score = self._calculate_unified_score(initial_evaluation)
                
                current_best_score_in_bayesian = initial_unified_score
                print(f"    📊 贝叶斯优化初始统一得分: {initial_unified_score:.6f}")
                self.logger.info(f"📊 贝叶斯优化初始统一得分: {initial_unified_score:.6f}")
            else:
                current_best_score_in_bayesian = 0.0

            def evaluate_strategy_params(params):
                nonlocal current_best_params_in_bayesian, current_best_score_in_bayesian

                try:
                    # 🔧 修复：rise_threshold和max_days是固定参数，不应该参与优化
                    # 直接使用优化算法生成的参数，不进行强制覆盖
                    complete_params = params.copy()

                    # 🔧 关键修复：保存当前策略模块状态
                    original_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                                      'get_current_params') else None

                    # 临时应用参数进行评估
                    # print("    临时应用参数进行评估")
                    strategy_module.update_params(complete_params)

                    # 在训练集上评估（使用配置中的固定 final_threshold，仅在优化期间生效）
                    orig_ft = None
                    cw = strategy_module.config.setdefault('confidence_weights', {})
                    try:
                        orig_ft = cw.get('final_threshold', None)
                        # 从配置解析固定 final_threshold（不参与优化）
                        cw['final_threshold'] = resolve_confidence_param(strategy_module.config, 'final_threshold', 0.5)
                        backtest_results = strategy_module.backtest(train_data)
                        evaluation = strategy_module.evaluate_strategy(backtest_results)
                    finally:
                        # 恢复final_threshold，避免污染全局配置
                        if orig_ft is None:
                            if 'final_threshold' in cw:
                                del cw['final_threshold']
                        else:
                            cw['final_threshold'] = orig_ft

                    # 使用统一的评分方法
                    final_score = self._calculate_unified_score(evaluation)

                    # 🎯 修复后的参数管理逻辑：只有更好的参数才保留在策略模块中
                    if final_score > current_best_score_in_bayesian:
                        # 新参数更好，保留在策略模块中
                        prev_score = current_best_score_in_bayesian
                        current_best_params_in_bayesian = complete_params.copy()
                        current_best_score_in_bayesian = final_score
                        # 策略模块已经更新为新参数，不需要额外操作
                        self.logger.info(f"贝叶斯优化发现更优参数: 得分 {final_score:.6f} > {prev_score:.6f}")
                        self.logger.info(f"参数详情: ")
                        for param_name, param_value in complete_params.items():
                            print(f"          {param_name}: {param_value}")
                    else:
                        # 新参数较差，必须恢复到之前的最佳参数
                        if current_best_params_in_bayesian:
                            strategy_module.update_params(current_best_params_in_bayesian)
                        else:
                            # 如果没有最佳参数，恢复到原始参数
                            if original_params:
                                strategy_module.update_params(original_params)

                    # 贝叶斯优化需要返回负值（因为它是最小化算法）
                    return -final_score

                except Exception as e:
                    self.logger.warning(f"参数评估失败: {e}")
                    # 出错时恢复到最佳参数或原始参数
                    if current_best_params_in_bayesian:
                        strategy_module.update_params(current_best_params_in_bayesian)
                    elif original_params:
                        strategy_module.update_params(original_params)
                    return 1.0  # 返回正值表示失败（贝叶斯优化会避免这个区域）

            # 运行贝叶斯优化
            print(f"    🔬 开始贝叶斯优化参数搜索... [{datetime.now().strftime('%H:%M:%S')}]")
            self.logger.info("🔬 开始贝叶斯优化参数搜索...")
            bayesian_params = self.run_bayesian_optimization(evaluate_strategy_params)
            bayesian_time = time.time() - bayesian_start_time

            if bayesian_params:
                # 🔧 修复：贝叶斯优化已经通过评估函数管理了最佳参数
                # 获取贝叶斯优化过程中找到的最佳参数（已经在策略模块中）
                final_bayesian_params = strategy_module.get_current_params() if hasattr(strategy_module,
                                                                                        'get_current_params') else bayesian_params

                # 最终评估贝叶斯优化结果（此时策略模块已经是最佳状态）
                bayesian_backtest = strategy_module.backtest(train_data)
                bayesian_evaluation = strategy_module.evaluate_strategy(bayesian_backtest)
                bayesian_unified_score = self._calculate_unified_score(bayesian_evaluation)

                # 如果贝叶斯优化结果更好，更新全局最佳参数
                if bayesian_unified_score > best_score:
                    best_params = final_bayesian_params.copy()  # 使用贝叶斯优化管理的最佳参数
                    best_score = bayesian_unified_score
                    optimization_method = 'bayesian_optimization'

                    print(f"    ✅ 贝叶斯优化找到更优参数! 得分提升: {best_score:.6f}")
                    self.logger.info(f"✅ 贝叶斯优化找到更优参数! 得分提升: {best_score:.6f}")
                else:
                    print(f"    ⚠️ 贝叶斯优化结果未超过当前最优，恢复之前最佳参数")
                    self.logger.info(f"⚠️ 贝叶斯优化结果未超过当前最优，恢复之前最佳参数")
                    # 恢复到之前的最佳参数
                    strategy_module.update_params(best_params)
                    # 如果使用的是初始参数，保持optimization_method为initial_params
                    if optimization_method == 'initial_params':
                        optimization_method = 'initial_params_retained'

                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"    🔬 贝叶斯优化完成 (耗时: {bayesian_time:.2f}s) [{current_time}]")
                print(f"       📈 最优得分: {bayesian_unified_score:.6f}")
                print(f"       📊 成功率: {bayesian_evaluation.get('success_rate', 0):.2%}")
                print(f"       🔍 交易数: {bayesian_evaluation.get('total_trades', 0)}")
                print(f"       📈 平均收益: {bayesian_evaluation.get('avg_return', 0):.2%}")

                self.logger.info(f"🔬 贝叶斯优化完成 (耗时: {bayesian_time:.2f}s)")
                self.logger.info(f"   最优得分: {bayesian_unified_score:.6f}")
                self.logger.info(f"   成功率: {bayesian_evaluation.get('success_rate', 0):.2%}")
                self.logger.info(f"   交易数: {bayesian_evaluation.get('total_trades', 0)}")
                self.logger.info(f"   平均收益: {bayesian_evaluation.get('avg_return', 0):.2%}")
            else:
                print("    ⚠️ 贝叶斯优化未找到有效解")
                self.logger.warning("⚠️ 贝叶斯优化未找到有效解")
                # 如果贝叶斯优化失败，使用初始参数
                if optimization_method == 'initial_params':
                    optimization_method = 'initial_params_fallback'

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
            val_total_points = val_evaluation.get('total_trades', val_evaluation.get('total_points', 0))
            val_avg_rise = val_evaluation.get('avg_return', val_evaluation.get('avg_rise', 0))

            validation_time = time.time() - validation_start_time

            # 检查过拟合
            overfitting_threshold = 0.8  # 验证集得分应该至少是训练集得分的80%
            overfitting_passed = val_score >= best_score * overfitting_threshold

            print(f"    ✅ 验证集评估完成 (耗时: {validation_time:.2f}s)")
            print(f"       得分: {val_score:.6f}")
            print(f"       成功率: {val_success_rate:.2%}")
            print(f"       交易数: {val_total_points}")
            print(f"       平均收益: {val_avg_rise:.2%}")
            print(f"       过拟合检测: {'✅ 通过' if overfitting_passed else '⚠️ 警告'}")

            self.logger.info(f"✅ 验证集评估完成 (耗时: {validation_time:.2f}s)")
            self.logger.info(f"   得分: {val_score:.6f}")
            self.logger.info(f"   成功率: {val_success_rate:.2%}")
            self.logger.info(f"   交易数: {val_total_points}")
            self.logger.info(f"   平均收益: {val_avg_rise:.2%}")
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
            test_total_points = test_evaluation.get('total_trades', test_evaluation.get('total_points', 0))
            test_avg_rise = test_evaluation.get('avg_return', test_evaluation.get('avg_rise', 0))

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
            print(f"       交易数: {test_total_points}")
            print(f"       平均收益: {test_avg_rise:.2%}")
            print(
                f"       泛化能力: {'✅ 良好' if generalization_passed else '⚠️ 一般'} (比率: {generalization_ratio:.3f})")

            self.logger.info(f"✅ 测试集评估完成 (耗时: {test_time:.2f}s)")
            self.logger.info(f"   得分: {test_score:.6f}")
            self.logger.info(f"   成功率: {test_success_rate:.2%}")
            self.logger.info(f"   交易数: {test_total_points}")
            self.logger.info(f"   平均收益: {test_avg_rise:.2%}")
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

            # 如果使用了贝叶斯优化，输出详细的参数信息
            if optimization_method == 'bayesian_optimization':
                print(f"    🔬 贝叶斯优化最优参数详情:")
                self.logger.info(f"\n🔬 贝叶斯优化最优参数详情:")
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
                # 兼容命名：仍保留旧字段，但含义改为交易数/平均收益
                'validation_total_points': val_total_points,
                'validation_avg_rise': val_avg_rise,
                # 新字段
                'validation_total_trades': val_total_points,
                'validation_avg_return': val_avg_rise,
                'test_score': test_score,
                'test_success_rate': test_success_rate,
                'test_total_points': test_total_points,
                'test_avg_rise': test_avg_rise,
                'test_total_trades': test_total_points,
                'test_avg_return': test_avg_rise,
                'overfitting_passed': overfitting_passed,
                'generalization_passed': generalization_passed,
                'generalization_ratio': generalization_ratio,
                'optimization_method': optimization_method,
                'optimization_time': optimization_total_time,
                'bayesian_optimization_used': optimization_method == 'bayesian_optimization'
            }

        except Exception as e:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"    ❌ 策略参数优化失败: {e} [{current_time}]")
            self.logger.error(f"策略参数优化失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

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

            # 新增：记录用于预测的日期和概率向量，便于诊断
            try:
                prediction_date = None
                if 'date' in getattr(data, 'columns', []):
                    try:
                        prediction_date = pd.to_datetime(data['date'].iloc[-1]).strftime('%Y-%m-%d')
                    except Exception:
                        prediction_date = str(data['date'].iloc[-1])
                elif hasattr(data, 'index') and len(data.index) > 0:
                    try:
                        prediction_date = pd.to_datetime(data.index[-1]).strftime('%Y-%m-%d')
                    except Exception:
                        prediction_date = str(data.index[-1])

                proba = prediction_result.get('prediction_proba')

                if isinstance(proba, (list, tuple, np.ndarray)):
                    proba_iter = proba.tolist() if hasattr(proba, 'tolist') else list(proba)
                    proba_str = '[' + ', '.join(f'{float(p):.4f}' for p in proba_iter) + ']'
                else:
                    proba_str = str(proba)

                print(f"    🔎 评估-预测日期: {prediction_date} | 概率向量: {proba_str}")
                self.logger.info(f"评估-预测日期: {prediction_date} | 概率向量: {proba_str}")

                if prediction_result.get('error'):
                    print(f"    ⚠️ 评估-预测错误: {prediction_result.get('error')}")
                    self.logger.warning(f"评估-预测错误: {prediction_result.get('error')}")
            except Exception as log_ex:
                self.logger.warning(f"记录预测细节时发生异常: {log_ex}")

            return {
                'success': True,
                'strategy_score': strategy_evaluation['score'],
                'strategy_success_rate': strategy_evaluation.get('success_rate', 0),
                'identified_points': strategy_evaluation.get('total_trades', strategy_evaluation.get('total_points', 0)),
                'avg_return': strategy_evaluation.get('avg_return', strategy_evaluation.get('avg_rise', 0)),
                'total_profit': strategy_evaluation.get('total_profit', 0),
                'ai_confidence': prediction_result.get('final_confidence', 0),
                'ai_prediction': prediction_result.get('is_low_point', False)
            }

        except Exception as e:
            self.logger.error(f"系统评估失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _save_optimized_parameters(self, best_params: Dict[str, Any]) -> bool:
        """
        保存优化后的参数到optimized_params.yaml文件
        
        参数:
        best_params: 优化后的参数
        """
        try:
            from src.utils.optimized_params_saver import save_optimized_params
            
            # 添加优化信息
            optimization_info = {
                'method': 'genetic_algorithm',
                'timestamp': datetime.now().isoformat(),
                'param_count': len(best_params)
            }
            
            # 保存参数到optimized_params.yaml
            success = save_optimized_params(
                params=best_params,
                optimization_info=optimization_info
            )
            
            if success:
                self.logger.info(f"优化参数已保存到optimized_params.yaml，共{len(best_params)}个参数")
                return True
            else:
                self.logger.error("保存优化参数失败")
                return False
                
        except Exception as e:
             self.logger.error(f"保存优化参数时发生错误: {e}")
             return False

    # 已移除_save_params_fallback方法，现在统一使用optimized_params_saver

    def run_bayesian_optimization(self, evaluate_func, param_ranges=None) -> Dict[str, Any]:
        """
        贝叶斯优化参数优化（高精度版本）
        
        专为高准确度设计，使用scikit-optimize进行贝叶斯优化
        
        参数:
        evaluate_func: 评估函数，接收参数字典，返回评分
        param_ranges: 参数范围字典，如果为None则使用配置文件
        
        返回:
        dict: 最优参数字典
        """
        from datetime import datetime

        print(f"        🔬 初始化贝叶斯优化 [{datetime.now().strftime('%H:%M:%S')}]")
        self.logger.info("🔬 启动贝叶斯优化（高精度模式）")
        start_time = time.time()

        try:
            # 获取贝叶斯优化配置
            bayesian_config = self.config.get('bayesian_optimization', {})
            
            # 高精度配置
            n_calls = bayesian_config.get('n_calls', 120)  # 总调用次数
            n_initial_points = bayesian_config.get('n_initial_points', 25)  # 初始随机点
            acq_func = bayesian_config.get('acq_func', 'EI')  # 采集函数
            xi = bayesian_config.get('xi', 0.01)  # 探索参数
            kappa = bayesian_config.get('kappa', 1.96)  # UCB参数
            random_state = bayesian_config.get('random_state', 42)

            print(f"        📊 贝叶斯优化配置:")
            print(f"           总调用次数: {n_calls}")
            print(f"           初始随机点: {n_initial_points}")
            print(f"           采集函数: {acq_func}")
            print(f"           探索参数xi: {xi}")
            print(f"           UCB参数kappa: {kappa}")

            self.logger.info(f"高精度贝叶斯优化配置: 调用{n_calls}次, 初始点{n_initial_points}个")

            # 获取或生成参数范围
            if param_ranges is None:
                param_ranges = self._get_enhanced_parameter_ranges({})

            print(f"        🎯 优化参数数量: {len(param_ranges)} 个")

            # 若搜索空间为空，直接跳过并给出清晰提示
            if not param_ranges:
                msg = "参数搜索空间为空，跳过贝叶斯优化（请检查配置optimization_ranges或固定参数设置）"
                print(f"        ⚠️ {msg}")
                self.logger.warning(msg)
                return {}

            # 构建搜索空间
            dimensions = []
            param_names = []
            
            for param_name, param_range in param_ranges.items():
                param_names.append(param_name)
                
                if param_range['type'] == 'int':
                    dimensions.append(Integer(param_range['min'], param_range['max'], name=param_name))
                else:  # float
                    dimensions.append(Real(param_range['min'], param_range['max'], name=param_name))

            print(f"        🌱 构建搜索空间完成")
            self.logger.info(f"搜索空间维度: {len(dimensions)}")

            # 若维度为0，无法执行优化
            if len(dimensions) == 0:
                msg = "搜索空间维度为0，无法执行贝叶斯优化（可能所有参数被固定或范围缺失）"
                print(f"        ⚠️ {msg}")
                self.logger.warning(msg)
                return {}

            # 定义目标函数（贝叶斯优化需要最小化，所以返回负值）
            best_score = -float('inf')
            best_params = None
            evaluation_count = 0
            
            @use_named_args(dimensions)
            def objective(**params):
                nonlocal best_score, best_params, evaluation_count
                evaluation_count += 1
                
                try:
                    # 验证并修复参数
                    validated_params = self._validate_and_fix_parameters(params, param_ranges)
                    
                    # 🔍 详细日志：记录每次评估的参数取值
                    param_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                          for k, v in validated_params.items()])
                    print(f"        🧪 试验 #{evaluation_count}: {param_str}")
                    self.logger.info(f"🧪 试验 #{evaluation_count}: {param_str}")
                    
                    # 评估参数
                    score = evaluate_func(validated_params)
                    
                    # 🔍 详细日志：记录每次评估的得分
                    # 修复：evaluate_func 返回的是用于最小化的"负得分"，这里转换为正得分显示与比较
                    if score is None or not isinstance(score, (int, float)) or (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
                        print(f"           ❌ 得分: 无效 (score={score})")
                        self.logger.info(f"           ❌ 得分: 无效 (score={score})")
                        return 1.0  # 返回正值表示差的结果
                    
                    actual_score = -score  # 转回正的统一评分用于展示与比较
                    print(f"           📊 得分: {actual_score:.6f}")
                    self.logger.info(f"           📊 得分: {actual_score:.6f}")
                    
                    # 更新最佳结果（使用正值进行比较）
                    if actual_score > best_score:
                        improvement = actual_score - best_score
                        best_score = actual_score
                        best_params = validated_params.copy()
                        print(f"        🎉 发现新最佳解! 得分: {best_score:.6f} (改进: +{improvement:.6f}) (第{evaluation_count}次评估)")
                        self.logger.info(f"🎉 发现新最佳解! 得分: {best_score:.6f} (改进: +{improvement:.6f}) (第{evaluation_count}次评估)")
                    else:
                        deficit = best_score - actual_score
                        print(f"           🔍 当前解劣于最佳: -{deficit:.6f}")
                        self.logger.info(f"           🔍 当前解劣于最佳: -{deficit:.6f}")
                    
                    # 每5次评估显示进度（更频繁的反馈）
                    if evaluation_count % 5 == 0:
                        progress = (evaluation_count / n_calls) * 100
                        print(f"        📈 评估进度: {evaluation_count}/{n_calls} ({progress:.1f}%) | 当前最佳: {best_score:.6f}")
                        self.logger.info(f"📈 评估进度: {evaluation_count}/{n_calls} ({progress:.1f}%) | 当前最佳: {best_score:.6f}")
                    
                    return score  # 直接返回用于最小化的值（负得分）
                    
                except Exception as e:
                    print(f"           ❌ 参数评估异常: {e}")
                    self.logger.warning(f"参数评估失败: {e}")
                    return 1.0  # 返回正值表示差的结果

            print(f"        🚀 开始贝叶斯优化 (总计 {n_calls} 次评估)")
            
            # 执行贝叶斯优化
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=acq_func,
                xi=xi,
                kappa=kappa,
                random_state=random_state
            )
            
            optimization_time = time.time() - start_time
            
            print(f"        ✅ 贝叶斯优化完成!")
            print(f"        🏆 最佳得分: {best_score:.6f}")
            print(f"        ⏱️ 总耗时: {optimization_time:.2f}s ({optimization_time/60:.1f}分钟)")
            print(f"        📊 总评估次数: {evaluation_count}")
            
            self.logger.info(f"✅ 贝叶斯优化完成!")
            self.logger.info(f"🏆 最佳得分: {best_score:.6f}")
            self.logger.info(f"⏱️ 总耗时: {optimization_time:.2f}s ({optimization_time/60:.1f}分钟)")
            self.logger.info(f"📊 总评估次数: {evaluation_count}")
            
            if best_params is None:
                self.logger.error("贝叶斯优化未找到有效参数")
                return {}
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"贝叶斯优化执行失败: {e}")
            print(f"        ❌ 贝叶斯优化失败: {e}")
            return {}


    def _get_enhanced_parameter_ranges(self, base_ranges: dict) -> dict:
        """
        获取增强的参数范围（使用配置文件中的范围）
        
        参数:
        base_ranges: 基础参数范围
        
        返回:
        dict: 增强的参数范围
        """
        # 🚨 重要：固定参数，不参与贝叶斯优化
        # 从配置文件中读取固定参数值
        strategy_config = self.config.get('strategy', {})
        fixed_rise_threshold = strategy_config.get('rise_threshold', 0.04)
        fixed_max_days = strategy_config.get('max_days', 20)

        # 导入参数配置
        from src.utils.param_config import FIXED_PARAMS, get_all_optimizable_params

        # 从配置文件中获取参数范围
        config = self.config
        strategy_ranges = config.get('strategy_ranges', {})
        optimization_ranges = config.get('optimization_ranges', {})

        enhanced_ranges = {}

        # 添加strategy_ranges中的参数
        for param_name, param_config in strategy_ranges.items():
            # 🚨 跳过固定参数，不允许优化
            if param_name in FIXED_PARAMS:
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
            # 优先使用配置中的类型定义；若缺省则按名称进行推断（仅对RSI阈值使用整数）
            cfg_type = param_config.get('type') if isinstance(param_config, dict) else None
            inferred_type = 'int' if (param_name.endswith('_threshold') and 'rsi' in param_name) else 'float'
            final_type = cfg_type if cfg_type in ('int', 'float') else inferred_type
            precision = param_config.get('precision', 0 if final_type == 'int' else 4) if isinstance(param_config, dict) else (0 if final_type == 'int' else 4)

            enhanced_ranges[param_name] = {
                'min': param_config.get('min', 0),
                'max': param_config.get('max', 1),
                'type': final_type,
                'precision': precision
            }
            # 保留 step 信息，便于遗传算法或网格搜索使用
            if isinstance(param_config, dict) and 'step' in param_config:
                enhanced_ranges[param_name]['step'] = param_config['step']

        # 合并用户配置的范围（但排除固定参数）
        for param_name, param_config in base_ranges.items():
            # 🚨 跳过固定参数，不允许优化
            if param_name in FIXED_PARAMS:
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
        self.logger.info(f"🔒 固定参数: {', '.join(FIXED_PARAMS)} (不参与优化)")
        self.logger.info(f"🔧 可优化参数: {len(get_all_optimizable_params())} 个（14个有效参数，已移除final_threshold）")

        # 记录参数范围
        for param_name, param_config in enhanced_ranges.items():
            self.logger.info(f"   {param_name}: {param_config['min']} - {param_config['max']} ({param_config['type']})")

        return enhanced_ranges

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



    # 已移除重复的_save_optimized_parameters方法定义

    def _calculate_unified_score(self, evaluation: Dict[str, Any]) -> float:
        """
        统一评分（按利润目标）：
        使用 利润因子 × log1p(交易次数) 作为优化目标；若交易次数过少则强惩罚。
        若evaluation已包含符合规则的score，则直接使用以保持一致性。
        """
        # 优先使用evaluation中的score（允许为负，按利润直接优化）
        if isinstance(evaluation, dict):
            existing_score = evaluation.get('score', None)
            if isinstance(existing_score, (int, float)) and np.isfinite(existing_score):
                return float(existing_score)
        
        # 否则按PF与交易次数计算参考分
        profit_factor = float(evaluation.get('profit_factor', 0.0) or 0.0)
        num_trades = int(evaluation.get('total_trades', evaluation.get('total_points', 0)) or 0)

        # 最少交易次数门槛（可由配置覆盖）
        min_trades_threshold = int(self.config.get('optimization_constraints', {}).get('min_trades_threshold', 10))

        # 交易过少返回0（参考）
        if num_trades < min_trades_threshold:
            return 0.0

        # 计算复合分数：PF * log1p(N)
        score = profit_factor * float(np.log1p(num_trades))
        if not np.isfinite(score):
            return 0.0
        return float(score)
