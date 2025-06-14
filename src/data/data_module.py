#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块
负责获取中证1000指数的历史数据和实时数据
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests
import time

class DataModule:
    """数据获取模块类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据模块
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger('DataModule')
        self.config = config
        self.index_code = config.get('data', {}).get('index_code', 'SHSE.000852')
        self.frequency = config.get('data', {}).get('frequency', '1d')
        
        # 创建缓存目录
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.logger.info("数据模块初始化完成")
        
    def get_history_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据
        
        参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
        返回:
        pandas.DataFrame: 历史数据
        """
        self.logger.info("获取历史数据: %s 到 %s", start_date, end_date)
        
        data_file_path = self.config.get("data", {}).get("data_file_path")
        if not data_file_path:
            self.logger.error("配置文件中未指定历史数据文件路径 (data.data_file_path)。")
            return pd.DataFrame()

        full_data_path = os.path.join(os.path.dirname(__file__), '..', '..', data_file_path)
        if not os.path.exists(full_data_path):
            self.logger.error("历史数据文件不存在: %s", full_data_path)
            return pd.DataFrame()

        try:
            # 读取CSV文件
            # 假设日期列名为 'date'，并且需要解析为datetime对象
            df = pd.read_csv(full_data_path, parse_dates=['date'], infer_datetime_format=True)
            
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # 过滤日期范围
            df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            
            # 按照日期排序
            df_filtered = df_filtered.sort_values(by='date').reset_index(drop=True)

            self.logger.info("从文件 %s 获取历史数据成功，共 %d 条记录。", data_file_path, len(df_filtered))
            return df_filtered
        except Exception as e:
            self.logger.error("从文件 %s 读取历史数据失败: %s", data_file_path, str(e))
            return pd.DataFrame()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        参数:
        data: 原始数据
        
        返回:
        pandas.DataFrame: 预处理后的数据
        """
        self.logger.info("预处理数据")
        
        try:
            # 确保日期列是datetime类型
            if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
                
            # 按日期排序
            data = data.sort_values('date').reset_index(drop=True)
            
            # 计算技术指标
            data = self._calculate_technical_indicators(data)
            
            self.logger.info("数据预处理完成，共 %d 条记录", len(data))
            return data
        except Exception as e:
            self.logger.error("数据预处理失败: %s", str(e))
            raise
            
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
        data: 原始数据
        
        返回:
        pandas.DataFrame: 添加了技术指标的数据
        """
        self.logger.info("计算技术指标")
        
        # 计算移动平均线
        data['ma5'] = data['close'].rolling(5).mean()
        data['ma10'] = data['close'].rolling(10).mean()
        data['ma20'] = data['close'].rolling(20).mean()
        data['ma60'] = data['close'].rolling(60).mean()
        
        # 计算相对强弱指标RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['hist'] = data['macd'] - data['signal']
        
        # 计算布林带
        data['bb_upper'] = data['ma20'] + (data['close'].rolling(20).std() * 2)
        data['bb_lower'] = data['ma20'] - (data['close'].rolling(20).std() * 2)
        
        # 计算价格与移动平均线的距离
        data['dist_ma5'] = (data['close'] - data['ma5']) / data['ma5']
        data['dist_ma10'] = (data['close'] - data['ma10']) / data['ma10']
        data['dist_ma20'] = (data['close'] - data['ma20']) / data['ma20']
        
        # 计算成交量变化
        data['volume_change'] = data['volume'].pct_change()
        
        # 计算波动率
        data['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        
        # 计算价格变化率
        data['price_change'] = data['close'].pct_change()
        data['price_change_5d'] = data['close'].pct_change(5)
        data['price_change_10d'] = data['close'].pct_change(10)
        
        return data
        
    def get_latest_data(self) -> Optional[pd.Series]:
        """
        获取最新数据
        
        返回:
        pandas.Series: 最新数据，如果获取失败返回None
        """
        self.logger.info("获取最新数据")
        
        try:
            # 获取最近30天的数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            data = self.get_history_data(start_date, end_date)
            
            if len(data) > 0:
                latest = data.iloc[-1]
                self.logger.info("获取最新数据成功，日期: %s", latest['date'])
                return latest
            else:
                self.logger.warning("没有获取到最新数据")
                return None
                
        except Exception as e:
            self.logger.error("获取最新数据失败: %s", str(e))
            return None
            
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据质量
        
        参数:
        data: 数据
        
        返回:
        bool: 数据是否有效
        """
        self.logger.info("验证数据质量")
        
        try:
            # 检查必要的列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error("缺少必要的列: %s", col)
                    return False
                    
            # 检查数据类型
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                self.logger.error("日期列类型不正确")
                return False
                
            # 检查价格数据的逻辑性
            for i in range(len(data)):
                row = data.iloc[i]
                if not (row['low'] <= row['open'] <= row['high'] and 
                       row['low'] <= row['close'] <= row['high']):
                    self.logger.warning("第 %d 行价格数据逻辑不正确", i)
                    
            # 检查缺失值
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                self.logger.warning("数据中有 %d 个缺失值", missing_count)
                
            self.logger.info("数据验证完成")
            return True
            
        except Exception as e:
            self.logger.error("数据验证失败: %s", str(e))
            return False

