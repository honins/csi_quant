#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据模块单元测试
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_module import DataModule

class TestDataModule(unittest.TestCase):
    """数据模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'data': {
                'index_code': 'SHSE.000852',
                'frequency': '1d'
            }
        }
        self.data_module = DataModule(self.config)
        
    def test_get_history_data(self):
        """测试获取历史数据"""
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        data = self.data_module.get_history_data(start_date, end_date)
        
        # 检查返回的数据类型
        self.assertIsInstance(data, pd.DataFrame)
        
        # 检查数据不为空
        self.assertGreater(len(data), 0)
        
        # 检查必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
            
    def test_preprocess_data(self):
        """测试数据预处理"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'open': [100 + i for i in range(30)],
            'high': [105 + i for i in range(30)],
            'low': [95 + i for i in range(30)],
            'close': [102 + i for i in range(30)],
            'volume': [1000000 + i * 10000 for i in range(30)]
        })
        
        processed_data = self.data_module.preprocess_data(test_data)
        
        # 检查技术指标是否被添加
        technical_indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd']
        for indicator in technical_indicators:
            self.assertIn(indicator, processed_data.columns)
            
    def test_validate_data(self):
        """测试数据验证"""
        # 创建有效的测试数据
        valid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000000] * 10
        })
        
        # 测试有效数据
        self.assertTrue(self.data_module.validate_data(valid_data))
        
        # 测试无效数据（缺少必要列）
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'close': [100] * 10
        })
        
        self.assertFalse(self.data_module.validate_data(invalid_data))

if __name__ == '__main__':
    unittest.main()

