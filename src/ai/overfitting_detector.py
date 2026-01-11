#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
过拟合检测模块
提供多种过拟合检测方法，确保模型泛化能力
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional



class OverfittingDetector:
    """过拟合检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 过拟合检测阈值
        self.validation_threshold = config.get('validation', {}).get('overfitting_threshold', 0.9)
        self.confidence_std_threshold = config.get('validation', {}).get('confidence_std_threshold', 0.05)
        self.zero_confidence_threshold = config.get('validation', {}).get('zero_confidence_threshold', 0.5)
        
    def detect_overfitting(self, 
                          train_score: float,
                          val_score: float,
                          test_score: Optional[float] = None,
                          val_predictions: List[float] = None,
                          train_predictions: List[float] = None) -> Dict:
        """
        综合过拟合检测
        
        参数:
        train_score: 训练集得分
        val_score: 验证集得分
        test_score: 测试集得分（可选）
        val_predictions: 验证集预测置信度列表
        train_predictions: 训练集预测置信度列表
        
        返回:
        dict: 检测结果
        """
        self.logger.info("🔍 开始综合过拟合检测...")
        
        results = {
            'overfitting_detected': False,
            'warnings': [],
            'metrics': {},
            'recommendations': []
        }
        
        # 1. 基础得分差异检测
        score_check = self._check_score_degradation(train_score, val_score, test_score)
        results['metrics'].update(score_check['metrics'])
        
        if score_check['overfitting']:
            results['overfitting_detected'] = True
            results['warnings'].extend(score_check['warnings'])
            results['recommendations'].extend(score_check['recommendations'])
        
        # 2. 置信度分布检测
        if val_predictions is not None:
            confidence_check = self._check_confidence_distribution(val_predictions, train_predictions)
            results['metrics'].update(confidence_check['metrics'])
            
            if confidence_check['overfitting']:
                results['overfitting_detected'] = True
                results['warnings'].extend(confidence_check['warnings'])
                results['recommendations'].extend(confidence_check['recommendations'])
        
        # 3. 学习曲线检测（如果有历史数据）
        learning_curve_check = self._check_learning_curve_pattern()
        if learning_curve_check['overfitting']:
            results['overfitting_detected'] = True
            results['warnings'].extend(learning_curve_check['warnings'])
            results['recommendations'].extend(learning_curve_check['recommendations'])
        
        # 输出检测结果
        self._log_detection_results(results)
        
        return results
    
    def _check_score_degradation(self, train_score: float, val_score: float, test_score: Optional[float] = None) -> Dict:
        """检测得分退化"""
        results = {
            'overfitting': False,
            'warnings': [],
            'recommendations': [],
            'metrics': {
                'train_score': train_score,
                'val_score': val_score,
                'score_ratio': val_score / train_score if train_score > 0 else 0
            }
        }
        
        # 验证集vs训练集得分比率
        score_ratio = val_score / train_score if train_score > 0 else 0
        
        if score_ratio < self.validation_threshold:
            results['overfitting'] = True
            results['warnings'].append(f"验证集得分明显低于训练集: {val_score:.4f} vs {train_score:.4f} (比率: {score_ratio:.3f})")
            results['recommendations'].append("减少模型复杂度，增加正则化")
        
        # 如果有测试集得分，进行额外检测
        if test_score is not None:
            results['metrics']['test_score'] = test_score
            val_test_ratio = test_score / val_score if val_score > 0 else 0
            results['metrics']['val_test_ratio'] = val_test_ratio
            
            if val_test_ratio < 0.85:  # 测试集得分应该接近验证集
                results['overfitting'] = True
                results['warnings'].append(f"测试集得分显著低于验证集: {test_score:.4f} vs {val_score:.4f}")
                results['recommendations'].append("模型可能过拟合验证集，建议重新设计验证策略")
        
        return results
    
    def _check_confidence_distribution(self, val_predictions: List[float], train_predictions: List[float] = None) -> Dict:
        """检测置信度分布异常"""
        results = {
            'overfitting': False,
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        val_array = np.array(val_predictions)
        
        # 计算置信度统计
        val_std = np.std(val_array)
        val_mean = np.mean(val_array)
        zero_ratio = np.sum(val_array == 0.0) / len(val_array)
        extreme_ratio = np.sum((val_array == 0.0) | (val_array == 1.0)) / len(val_array)
        
        results['metrics'].update({
            'val_confidence_std': val_std,
            'val_confidence_mean': val_mean,
            'zero_confidence_ratio': zero_ratio,
            'extreme_confidence_ratio': extreme_ratio
        })
        
        # 检测置信度标准差过小
        if val_std < self.confidence_std_threshold:
            results['overfitting'] = True
            results['warnings'].append(f"验证集置信度标准差过小: {val_std:.4f}")
            results['recommendations'].append("模型输出过于极端，增加模型正则化或减少特征数量")
        
        # 检测零置信度比例过高
        if zero_ratio > self.zero_confidence_threshold:
            results['overfitting'] = True
            results['warnings'].append(f"验证集零置信度比例过高: {zero_ratio:.1%}")
            results['recommendations'].append("模型可能过度拟合训练数据模式，建议重新训练")
        
        # 检测极端置信度比例
        if extreme_ratio > 0.8:
            results['overfitting'] = True
            results['warnings'].append(f"验证集极端置信度(0或1)比例过高: {extreme_ratio:.1%}")
            results['recommendations'].append("模型过于自信，缺乏泛化能力")
        
        # 如果有训练集置信度，进行对比分析
        if train_predictions is not None:
            train_array = np.array(train_predictions)
            train_std = np.std(train_array)
            train_mean = np.mean(train_array)
            
            results['metrics'].update({
                'train_confidence_std': train_std,
                'train_confidence_mean': train_mean,
                'std_ratio': val_std / train_std if train_std > 0 else 0
            })
            
            # 训练集和验证集置信度分布差异
            if train_std > 0 and val_std / train_std < 0.5:
                results['overfitting'] = True
                results['warnings'].append(f"验证集置信度方差显著小于训练集: {val_std:.4f} vs {train_std:.4f}")
                results['recommendations'].append("模型在验证集上表现异常一致，可能过拟合")
        
        return results
    
    def _check_learning_curve_pattern(self) -> Dict:
        """检测学习曲线模式（简化版）"""
        results = {
            'overfitting': False,
            'warnings': [],
            'recommendations': []
        }
        
        # 这里可以添加学习曲线分析
        # 暂时返回空结果
        return results
    
    def _log_detection_results(self, results: Dict):
        """记录检测结果"""
        if results['overfitting_detected']:
            self.logger.warning("🚨 检测到过拟合风险！")
            for warning in results['warnings']:
                self.logger.warning(f"   ⚠️ {warning}")
            
            self.logger.info("💡 建议措施:")
            for rec in results['recommendations']:
                self.logger.info(f"   📝 {rec}")
        else:
            self.logger.info("✅ 未检测到明显过拟合")
        
        # 输出关键指标
        metrics = results['metrics']
        self.logger.info("📊 检测指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.4f}")
            else:
                self.logger.info(f"   {key}: {value}")


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_score: float, epoch: int = 0) -> bool:
        """
        检查是否应该早停
        
        参数:
        val_score: 当前验证集得分
        epoch: 当前轮次
        
        返回:
        bool: 是否应该停止训练
        """
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
            return False
    
    def get_best_score(self) -> float:
        """获取最佳得分"""
        return self.best_score
    
    def get_stopped_epoch(self) -> int:
        """获取停止的轮次"""
        return self.stopped_epoch


def validate_data_split(train_data: pd.DataFrame, 
                       val_data: pd.DataFrame, 
                       test_data: pd.DataFrame,
                       date_column: str = 'date') -> Dict:
    """
    验证数据分割的正确性
    
    参数:
    train_data: 训练数据
    val_data: 验证数据
    test_data: 测试数据
    date_column: 日期列名
    
    返回:
    dict: 验证结果
    """
    results = {
        'valid': True,
        'issues': []
    }
    
    # 检查数据重叠
    train_indices = set(train_data.index)
    val_indices = set(val_data.index)
    test_indices = set(test_data.index)
    
    if train_indices & val_indices:
        results['valid'] = False
        results['issues'].append("训练集和验证集存在数据重叠")
    
    if train_indices & test_indices:
        results['valid'] = False
        results['issues'].append("训练集和测试集存在数据重叠")
    
    if val_indices & test_indices:
        results['valid'] = False
        results['issues'].append("验证集和测试集存在数据重叠")
    
    # 检查时间顺序（如果有日期列）
    if date_column in train_data.columns:
        train_max_date = train_data[date_column].max()
        val_min_date = val_data[date_column].min()
        val_max_date = val_data[date_column].max()
        test_min_date = test_data[date_column].min()
        
        if train_max_date >= val_min_date:
            results['valid'] = False
            results['issues'].append(f"时间序列顺序错误: 训练集最新日期({train_max_date}) >= 验证集最早日期({val_min_date})")
        
        if val_max_date >= test_min_date:
            results['valid'] = False
            results['issues'].append(f"时间序列顺序错误: 验证集最新日期({val_max_date}) >= 测试集最早日期({test_min_date})")
    
    return results