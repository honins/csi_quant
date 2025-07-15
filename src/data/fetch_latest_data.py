#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
获取最新数据脚本
获取000852和000905的最新数据并保存到data目录下的CSV文件中
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import yaml
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    import akshare as ak
except ImportError:
    print("错误: 未安装akshare库，请运行: pip install akshare")
    sys.exit(1)

class DataFetcher:
    """数据获取器类"""
    
    def __init__(self, config_path: str = "config/config_core.yaml"):
        """
        初始化数据获取器
        
        参数:
        config_path: 配置文件路径
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.data_dir = os.path.join(project_root, "data")
        
        # 确保data目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.logger.info("数据获取器初始化完成")
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('DataFetcher')
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
        
        return logger
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info("配置文件加载成功")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
            
    def fetch_index_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取指数数据
        
        参数:
        symbol: 指数代码 (如 '000852', '000905')
        start_date: 开始日期 (YYYY-MM-DD)，默认为一年前
        end_date: 结束日期 (YYYY-MM-DD)，默认为今天
        
        返回:
        pandas.DataFrame: 指数数据
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"获取指数 {symbol} 数据: {start_date} 到 {end_date}")
        
        try:
            # 使用akshare获取指数数据
            # 000852: 中证1000指数
            # 000905: 中证500指数
            if symbol == '000852':
                # 中证1000指数
                df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
            elif symbol == '000905':
                # 中证500指数
                df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
            else:
                self.logger.error(f"不支持的指数代码: {symbol}")
                return None
                
            # 重命名列以匹配项目标准格式
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 过滤日期范围
            df = df[(df['date'] >= pd.to_datetime(start_date)) & 
                   (df['date'] <= pd.to_datetime(end_date))]
            
            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)
            
            # 添加amount列（如果没有的话）
            if 'amount' not in df.columns:
                df['amount'] = df['volume'] * df['close']
                
            # 确保列的顺序正确
            expected_columns = ['index', 'open', 'high', 'low', 'close', 'volume', 'amount', 'date']
            for col in expected_columns:
                if col not in df.columns:
                    if col == 'index':
                        df['index'] = range(1, len(df) + 1)
                    else:
                        df[col] = 0
                        
            df = df[expected_columns]
            
            self.logger.info(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
            
    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        保存数据到CSV文件（增量更新）
        
        参数:
        df: 数据框
        symbol: 指数代码
        
        返回:
        bool: 是否保存成功
        """
        try:
            # 生成文件名
            filename = f"SHSE.{symbol}_1d.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # 检查现有文件是否存在
            existing_df = None
            if os.path.exists(filepath):
                try:
                    existing_df = pd.read_csv(filepath)
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    self.logger.info(f"读取现有文件: {filename}，共 {len(existing_df)} 条记录")
                except Exception as e:
                    self.logger.warning(f"读取现有文件失败，将创建新文件: {e}")
                    existing_df = None
            
            if existing_df is not None and len(existing_df) > 0:
                # 增量更新：合并现有数据和新数据
                # 确保新数据的日期列是datetime类型
                df['date'] = pd.to_datetime(df['date'])
                
                # 获取现有数据的最新日期
                latest_existing_date = existing_df['date'].max()
                self.logger.info(f"现有数据最新日期: {latest_existing_date.strftime('%Y-%m-%d')}")
                
                # 过滤出新数据（只保留比现有数据更新的数据）
                new_data = df[df['date'] > latest_existing_date]
                
                if len(new_data) > 0:
                    self.logger.info(f"发现 {len(new_data)} 条新数据")
                    
                    # 合并现有数据和新数据
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    
                    # 按日期排序并去重（保留最新的数据）
                    combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                    
                    # 重新生成index列
                    combined_df['index'] = range(1, len(combined_df) + 1)
                    
                    # 确保列的顺序正确
                    expected_columns = ['index', 'open', 'high', 'low', 'close', 'volume', 'amount', 'date']
                    combined_df = combined_df[expected_columns]
                    
                    # 保存合并后的数据
                    combined_df.to_csv(filepath, index=False, encoding='utf-8')
                    
                    self.logger.info(f"增量更新完成: 原有 {len(existing_df)} 条，新增 {len(new_data)} 条，总计 {len(combined_df)} 条")
                    return True
                else:
                    self.logger.info(f"没有新数据需要更新，现有数据已是最新")
                    return True
            else:
                # 如果文件不存在或为空，直接保存新数据
                self.logger.info(f"创建新文件: {filename}")
                df.to_csv(filepath, index=False, encoding='utf-8')
                self.logger.info(f"数据已保存到: {filepath}，共 {len(df)} 条记录")
                return True
            
        except Exception as e:
            self.logger.error(f"保存 {symbol} 数据失败: {e}")
            return False
            
    def fetch_and_save_latest_data(self) -> Dict[str, Any]:
        """
        获取并保存最新数据（增量更新）
        
        返回:
        Dict[str, Any]: 每个指数的保存状态和统计信息
        """
        results = {}
        
        # 获取000852和000905的数据
        symbols = ['000852', '000905']
        
        for symbol in symbols:
            self.logger.info(f"开始处理指数: {symbol}")
            
            # 获取数据
            df = self.fetch_index_data(symbol)
            
            if df is not None and len(df) > 0:
                # 保存数据（增量更新）
                success = self.save_to_csv(df, symbol)
                
                # 获取文件信息用于统计
                filename = f"SHSE.{symbol}_1d.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        final_df = pd.read_csv(filepath)
                        final_df['date'] = pd.to_datetime(final_df['date'])
                        total_records = len(final_df)
                        latest_date = final_df['date'].max().strftime('%Y-%m-%d')
                        earliest_date = final_df['date'].min().strftime('%Y-%m-%d')
                    except Exception as e:
                        self.logger.error(f"读取最终文件失败: {e}")
                        total_records = 0
                        latest_date = "未知"
                        earliest_date = "未知"
                else:
                    total_records = 0
                    latest_date = "未知"
                    earliest_date = "未知"
                
                results[symbol] = {
                    'success': success,
                    'total_records': total_records,
                    'latest_date': latest_date,
                    'earliest_date': earliest_date
                }
                
                if success:
                    self.logger.info(f"{symbol} 数据处理完成，最新数据日期: {latest_date}")
                else:
                    self.logger.error(f"{symbol} 数据保存失败")
            else:
                self.logger.error(f"{symbol} 数据获取失败")
                results[symbol] = {
                    'success': False,
                    'total_records': 0,
                    'latest_date': "未知",
                    'earliest_date': "未知"
                }
                
            # 添加延迟避免请求过于频繁
            time.sleep(1)
            
        return results

def main():
    """主函数"""
    print("开始获取最新数据...")
    
    # 创建数据获取器
    fetcher = DataFetcher()
    
    # 获取并保存数据
    results = fetcher.fetch_and_save_latest_data()
    
    # 输出结果
    print("\n数据获取结果:")
    for symbol, info in results.items():
        status = "成功" if info['success'] else "失败"
        print(f"  {symbol}: {status}, 总记录数: {info['total_records']}, 最新日期: {info['latest_date']}, 最早日期: {info['earliest_date']}")
        
    # 返回JSON格式的结果
    response = {
        "code": 200 if all(info['success'] for info in results.values()) else 500,
        "msg": "数据获取完成" if all(info['success'] for info in results.values()) else "部分数据获取失败",
        "data": {
            "results": results,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    print(f"\n响应结果: {response}")
    return response

if __name__ == "__main__":
    main() 