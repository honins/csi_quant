#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utils包初始化文件
提供统一的导入接口，确保向后兼容性
"""

# 从common模块导入新的功能
from .common import (
    LoggerManager, DataValidator, TimeUtils, PerformanceMonitor,
    MathUtils, FileManager, QuantError, DataError, ConfigError, ModelError,
    safe_execute, error_context, get_project_root, ensure_directory,
    init_project_environment
)

# 从config_loader导入配置相关功能
from .config_loader import load_config, save_config, get_default_config

# 从base_module导入基础模块类
from .base_module import BaseModule

# 从command_processor导入命令处理器
from .command_processor import CommandProcessor

# 兼容性导入：保持与原有utils.py的兼容性
try:
    from .utils import setup_logging, format_percentage, format_currency
    from .utils import calculate_returns, calculate_volatility, calculate_sharpe_ratio
    from .utils import calculate_max_drawdown, validate_date_format, Timer
    from .utils import get_trading_days
except ImportError:
    # 如果原有utils.py不存在，提供默认实现
    def setup_logging(level='INFO', log_file='logs/system.log'):
        """兼容性函数：设置日志"""
        return LoggerManager.setup_logging(level=level, log_file=log_file)
    
    def validate_date_format(date_str: str) -> bool:
        """兼容性函数：验证日期格式"""
        return DataValidator.validate_date_format(date_str)


# 模块导出
__all__ = [
    # 新架构组件
    'LoggerManager', 'DataValidator', 'TimeUtils', 'PerformanceMonitor',
    'MathUtils', 'FileManager', 'QuantError', 'DataError', 'ConfigError', 'ModelError',
    'safe_execute', 'error_context', 'get_project_root', 'ensure_directory',
    'init_project_environment',
    
    # 配置管理
    'load_config', 'save_config', 'get_default_config',
    
    # 基础模块
    'BaseModule',
    
    # 命令处理
    'CommandProcessor',
    
    # 兼容性函数
    'setup_logging', 'format_percentage', 'format_currency',
    'calculate_returns', 'calculate_volatility', 'calculate_sharpe_ratio',
    'calculate_max_drawdown', 'validate_date_format', 'Timer',
    'get_trading_days'
]
