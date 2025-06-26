#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据验证和分割模块
负责数据的严格分割、走前验证等功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any


class DataValidator:
    """数据验证和分割类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据验证器
        
        参数:
        config: 配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

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
                
                # 在训练窗口上优化参数（这里需要传入优化器）
                # 在测试窗口上评估
                temp_strategy = strategy_module.__class__(self.config)
                
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