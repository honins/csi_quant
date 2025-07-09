#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略模块单元测试
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.strategy_module import StrategyModule

class TestStrategyModule(unittest.TestCase):
    """策略模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'strategy': {
                'rise_threshold': 0.04,
                'max_days': 20
            },
            'optimization': {
                'param_ranges': {
                    'rise_threshold': {
                        'min': 0.03,
                        'max': 0.08,
                        'step': 0.005
                    },
                    'max_days': {
                        'min': 10,
                        'max': 30,
                        'step': 1
                    }
                }
            }
        }
        self.strategy_module = StrategyModule(self.config)
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'ma5': prices,  # 简化的移动平均
            'ma10': prices,
            'ma20': prices,
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 1, 100),
            'bb_lower': prices * 0.95
        })
        
    def test_identify_relative_low(self):
        """测试相对低点识别"""
        result = self.strategy_module.identify_relative_low(self.test_data)
        
        # 检查返回结果的结构
        required_keys = ['date', 'price', 'is_low_point', 'confidence']
        for key in required_keys:
            self.assertIn(key, result)
            
        # 检查数据类型
        self.assertIsInstance(result['is_low_point'], bool)
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
        
    def test_backtest(self):
        """测试回测功能"""
        backtest_results = self.strategy_module.backtest(self.test_data)
        
        # 检查返回的数据类型
        self.assertIsInstance(backtest_results, pd.DataFrame)
        
        # 检查回测结果包含必要的列
        required_columns = ['is_low_point', 'future_max_rise', 'days_to_rise']
        for col in required_columns:
            self.assertIn(col, backtest_results.columns)
            
        # 检查数据长度一致
        self.assertEqual(len(backtest_results), len(self.test_data))
        
    def test_evaluate_strategy(self):
        """测试策略评估"""
        backtest_results = self.strategy_module.backtest(self.test_data)
        evaluation = self.strategy_module.evaluate_strategy(backtest_results)
        
        # 检查评估结果的结构
        required_keys = ['total_points', 'success_rate', 'avg_rise', 'avg_days', 'score']
        for key in required_keys:
            self.assertIn(key, evaluation)
            
        # 检查数据类型和范围
        self.assertIsInstance(evaluation['total_points'], int)
        self.assertGreaterEqual(evaluation['total_points'], 0)
        
        self.assertIsInstance(evaluation['success_rate'], (int, float))
        self.assertGreaterEqual(evaluation['success_rate'], 0)
        self.assertLessEqual(evaluation['success_rate'], 1)
        
        self.assertIsInstance(evaluation['score'], (int, float))
        self.assertGreaterEqual(evaluation['score'], 0)
        
    def test_update_params(self):
        """测试参数更新"""
        new_params = {
            'rise_threshold': 0.06,
            'max_days': 25
        }
        
        self.strategy_module.update_params(new_params)
        
        # 检查参数是否更新
        self.assertEqual(self.strategy_module.rise_threshold, 0.06)
        self.assertEqual(self.strategy_module.max_days, 25)
        
    def test_get_params(self):
        """测试获取参数"""
        params = self.strategy_module.get_params()
        
        # 检查返回的参数
        self.assertIn('rise_threshold', params)
        self.assertIn('max_days', params)
        self.assertEqual(params['rise_threshold'], 0.04)
        self.assertEqual(params['max_days'], 20)
        
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        # 测试相对低点识别
        result = self.strategy_module.identify_relative_low(empty_data)
        self.assertFalse(result['is_low_point'])
        self.assertEqual(result['confidence'], 0.0)
        
        # 测试回测
        backtest_results = self.strategy_module.backtest(empty_data)
        self.assertEqual(len(backtest_results), 0)

if __name__ == '__main__':
    unittest.main()

