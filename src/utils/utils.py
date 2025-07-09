#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具模块
包含通用的工具函数和配置管理
"""

import os
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

def setup_logging(level='INFO', log_file='logs/system.log'):
    """
    设置日志配置，既输出到文件，也输出到控制台
    """
    logger = logging.getLogger()
    logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
    # 防止重复添加handler
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # 文件输出
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
    config_path: 配置文件路径
    
    返回:
    dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error("加载配置文件失败: %s", str(e))
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    保存配置文件
    
    参数:
    config: 配置字典
    config_path: 配置文件路径
    
    返回:
    bool: 是否保存成功
    """
    try:
        # 确保配置目录存在
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        logging.error("保存配置文件失败: %s", str(e))
        return False

def get_trading_days(start_date: str, end_date: str) -> list:
    """
    获取交易日列表（简单实现，排除周末）
    
    参数:
    start_date: 开始日期 (YYYY-MM-DD)
    end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
    list: 交易日列表
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current = start
        
        while current <= end:
            # 排除周末（周六=5，周日=6）
            if current.weekday() < 5:
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
            
        return trading_days
    except Exception as e:
        logging.error("获取交易日失败: %s", str(e))
        return []

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    格式化百分比
    
    参数:
    value: 数值
    decimal_places: 小数位数
    
    返回:
    str: 格式化后的百分比字符串
    """
    return f"{value * 100:.{decimal_places}f}%"

def format_currency(value: float, currency: str = '¥') -> str:
    """
    格式化货币
    
    参数:
    value: 数值
    currency: 货币符号
    
    返回:
    str: 格式化后的货币字符串
    """
    return f"{currency}{value:,.2f}"

def calculate_returns(prices: list) -> list:
    """
    计算收益率序列
    
    参数:
    prices: 价格序列
    
    返回:
    list: 收益率序列
    """
    if len(prices) < 2:
        return []
        
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        else:
            returns.append(0.0)
            
    return returns

def calculate_volatility(returns: list, annualize: bool = True) -> float:
    """
    计算波动率
    
    参数:
    returns: 收益率序列
    annualize: 是否年化
    
    返回:
    float: 波动率
    """
    if len(returns) < 2:
        return 0.0
        
    import numpy as np
    
    volatility = np.std(returns)
    
    if annualize:
        # 假设一年有252个交易日
        volatility *= np.sqrt(252)
        
    return volatility

def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率
    
    参数:
    returns: 收益率序列
    risk_free_rate: 无风险利率（年化）
    
    返回:
    float: 夏普比率
    """
    if len(returns) < 2:
        return 0.0
        
    import numpy as np
    
    # 计算年化收益率
    annual_return = np.mean(returns) * 252
    
    # 计算年化波动率
    annual_volatility = calculate_volatility(returns, annualize=True)
    
    if annual_volatility == 0:
        return 0.0
        
    # 计算夏普比率
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    return sharpe_ratio

def calculate_max_drawdown(prices: list) -> Dict[str, Any]:
    """
    计算最大回撤
    
    参数:
    prices: 价格序列
    
    返回:
    dict: 最大回撤信息
    """
    if len(prices) < 2:
        return {
            'max_drawdown': 0.0,
            'start_date': None,
            'end_date': None,
            'recovery_date': None
        }
        
    import numpy as np
    
    prices = np.array(prices)
    
    # 计算累计最高价
    cumulative_max = np.maximum.accumulate(prices)
    
    # 计算回撤
    drawdowns = (prices - cumulative_max) / cumulative_max
    
    # 找到最大回撤
    max_drawdown_idx = np.argmin(drawdowns)
    max_drawdown = drawdowns[max_drawdown_idx]
    
    # 找到最大回撤开始的位置
    start_idx = np.argmax(cumulative_max[:max_drawdown_idx + 1])
    
    # 找到恢复的位置（如果有的话）
    recovery_idx = None
    peak_price = cumulative_max[start_idx]
    
    for i in range(max_drawdown_idx + 1, len(prices)):
        if prices[i] >= peak_price:
            recovery_idx = i
            break
    
    return {
        'max_drawdown': abs(max_drawdown),
        'start_idx': start_idx,
        'end_idx': max_drawdown_idx,
        'recovery_idx': recovery_idx
    }

def validate_date_format(date_str: str) -> bool:
    """
    验证日期格式
    
    参数:
    date_str: 日期字符串
    
    返回:
    bool: 是否为有效的日期格式
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def ensure_directory(directory: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    参数:
    directory: 目录路径
    
    返回:
    bool: 是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        logging.error("创建目录失败 %s: %s", directory, str(e))
        return False

class Timer:
    """简单的计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        
    def stop(self):
        """停止计时"""
        self.end_time = datetime.now()
        
    def elapsed(self) -> float:
        """
        获取经过的时间（秒）
        
        返回:
        float: 经过的时间
        """
        if self.start_time is None:
            return 0.0
            
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()
        
    def elapsed_str(self) -> str:
        """
        获取经过的时间字符串
        
        返回:
        str: 格式化的时间字符串
        """
        elapsed = self.elapsed()
        
        if elapsed < 60:
            return f"{elapsed:.2f}秒"
        elif elapsed < 3600:
            return f"{elapsed/60:.2f}分钟"
        else:
            return f"{elapsed/3600:.2f}小时"

