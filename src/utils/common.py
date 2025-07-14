#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
公共工具模块
整合项目中重复出现的通用功能，提高代码复用性和可维护性

功能模块：
- 基础工具函数
- 日志管理系统
- 目录和文件管理
- 数据验证工具
- 时间和日期处理
- 错误处理机制
- 性能监控工具
"""

import os
import sys
import logging
import time
import yaml
import json
import shutil
import traceback
import threading  # 添加线程支持
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
import pandas as pd
import numpy as np


# ================================================================================
# 1. 基础工具函数
# ================================================================================

def get_project_root() -> Path:
    """
    获取项目根目录
    
    返回:
        Path: 项目根目录路径
    """
    # 从当前文件向上查找，直到找到包含特定文件的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'requirements.txt').exists() or (parent / 'setup.py').exists():
            return parent
    # 如果没找到，返回当前文件的祖父目录（适用于 src/utils/common.py 结构）
    return current.parent.parent.parent


def ensure_directory(directory: Union[str, Path], create_parents: bool = True) -> bool:
    """
    确保目录存在，如不存在则创建
    
    参数:
        directory: 目录路径
        create_parents: 是否创建父目录
    
    返回:
        bool: 是否成功创建或已存在
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=create_parents, exist_ok=True)
            logging.debug(f"创建目录: {directory}")
        return True
    except Exception as e:
        logging.error(f"创建目录失败 {directory}: {e}")
        return False


def safe_file_operation(operation: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    """
    安全的文件操作包装器
    
    参数:
        operation: 文件操作函数
        *args: 位置参数
        **kwargs: 关键字参数
    
    返回:
        Tuple[bool, Any]: (是否成功, 结果或错误信息)
    """
    try:
        result = operation(*args, **kwargs)
        return True, result
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        return False, f"文件未找到: {e}"
    except PermissionError as e:
        logging.error(f"权限错误: {e}")
        return False, f"权限错误: {e}"
    except Exception as e:
        logging.error(f"文件操作失败: {e}")
        return False, f"文件操作失败: {e}"


# ================================================================================
# 2. 日志管理系统
# ================================================================================

class LoggerManager:
    """
    统一的日志管理器
    提供标准化的日志配置和管理功能
    """
    
    _loggers = {}
    _configured = False
    _lock = threading.Lock()  # 添加线程锁
    
    @classmethod
    def setup_logging(cls, 
                     level: Union[str, int] = 'INFO',
                     log_file: Optional[str] = None,
                     console_output: bool = True,
                     file_output: bool = True,
                     log_format: Optional[str] = None) -> logging.Logger:
        """
        设置全局日志配置
        
        参数:
            level: 日志级别
            log_file: 日志文件路径
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            log_format: 日志格式
        
        返回:
            logging.Logger: 根日志记录器
        """
        with cls._lock:  # 线程安全保护
            if cls._configured:
                return logging.getLogger()
            
            # 设置日志级别
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)
            
            # 默认日志格式
            if log_format is None:
                log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            # 默认日志文件
            if log_file is None:
                project_root = get_project_root()
                log_file = project_root / 'logs' / 'system.log'
            
            # 获取根日志记录器
            logger = logging.getLogger()
            logger.setLevel(level)
            
            # 只清除我们管理的处理器，保留其他处理器
            handlers_to_remove = []
            for handler in logger.handlers:
                if hasattr(handler, '_logger_manager_created'):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                logger.removeHandler(handler)
            
            formatter = logging.Formatter(log_format)
            
            # 控制台输出
            if console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                console_handler._logger_manager_created = True  # 标记我们创建的处理器
                logger.addHandler(console_handler)
            
            # 文件输出
            if file_output and log_file:
                log_path = Path(log_file)
                ensure_directory(log_path.parent)
                
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                file_handler._logger_manager_created = True  # 标记我们创建的处理器
                logger.addHandler(file_handler)
            
            cls._configured = True
            logging.info("日志系统初始化完成")
            return logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        参数:
            name: 日志记录器名称
        
        返回:
            logging.Logger: 日志记录器
        """
        with cls._lock:  # 线程安全保护
            if name not in cls._loggers:
                if not cls._configured:
                    cls.setup_logging()
                cls._loggers[name] = logging.getLogger(name)
            return cls._loggers[name]


# ================================================================================
# 3. 数据验证工具
# ================================================================================

class DataValidator:
    """数据验证工具类"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: List[str],
                          min_rows: int = 1,
                          allow_null: bool = False) -> Tuple[bool, List[str]]:
        """
        验证DataFrame的完整性
        
        参数:
            df: 要验证的DataFrame
            required_columns: 必需的列名列表
            min_rows: 最小行数
            allow_null: 是否允许空值
        
        返回:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查是否为空或None
        if df is None:
            errors.append("DataFrame为None")
            return False, errors
            
        if df.empty:
            errors.append("DataFrame为空")
            return False, errors
        
        # 检查行数
        if len(df) < min_rows:
            errors.append(f"数据行数不足，需要至少{min_rows}行，实际{len(df)}行")
        
        # 检查必需列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")
        
        # 检查空值（包括NaN、None、空字符串等）
        if not allow_null:
            for col in required_columns:
                if col in df.columns:
                    # 检查多种类型的空值
                    null_mask = df[col].isnull() | (df[col] == '') | (df[col] == 'nan')
                    if null_mask.any():
                        null_count = null_mask.sum()
                        errors.append(f"列 '{col}' 包含 {null_count} 个空值或无效值")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_date_format(date_str: str, date_format: str = '%Y-%m-%d') -> bool:
        """
        验证日期格式
        
        参数:
            date_str: 日期字符串
            date_format: 预期的日期格式
        
        返回:
            bool: 是否为有效日期格式
        """
        if not isinstance(date_str, str):
            return False
            
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            # 额外检查：确保日期在合理范围内
            current_year = datetime.now().year
            if parsed_date.year < 1900 or parsed_date.year > current_year + 50:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str, date_format: str = '%Y-%m-%d') -> Tuple[bool, str]:
        """
        验证日期范围
        
        参数:
            start_date: 开始日期字符串
            end_date: 结束日期字符串
            date_format: 日期格式
        
        返回:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 验证单个日期格式
        if not DataValidator.validate_date_format(start_date, date_format):
            return False, f"开始日期格式无效: {start_date}"
        
        if not DataValidator.validate_date_format(end_date, date_format):
            return False, f"结束日期格式无效: {end_date}"
        
        # 验证日期逻辑关系
        try:
            start = datetime.strptime(start_date, date_format)
            end = datetime.strptime(end_date, date_format)
            
            if start >= end:
                return False, f"开始日期必须早于结束日期: {start_date} >= {end_date}"
            
            # 检查日期范围是否过大（超过50年）
            if (end - start).days > 365 * 50:
                return False, "日期范围过大，超过50年"
                
            return True, ""
            
        except Exception as e:
            return False, f"日期范围验证失败: {e}"
    
    @staticmethod
    def validate_config_section(config: Dict[str, Any], 
                               section: str, 
                               required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        验证配置文件的特定部分
        
        参数:
            config: 配置字典
            section: 配置部分名称
            required_keys: 必需的键列表
        
        返回:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        if section not in config:
            errors.append(f"缺少配置部分: {section}")
            return False, errors
        
        section_config = config[section]
        
        for key in required_keys:
            if key not in section_config:
                errors.append(f"配置部分 '{section}' 缺少必需键: {key}")
        
        return len(errors) == 0, errors


# ================================================================================
# 4. 时间和日期处理
# ================================================================================

class TimeUtils:
    """时间处理工具类"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        格式化时间长度
        
        参数:
            seconds: 秒数
        
        返回:
            str: 格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}分{secs:.0f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分钟"
    
    @staticmethod
    def get_trading_days(start_date: Union[str, datetime], 
                        end_date: Union[str, datetime]) -> List[str]:
        """
        获取交易日列表（排除周末）
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            List[str]: 交易日列表
        """
        # 转换为datetime对象
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current = start_date
        
        while current <= end_date:
            # 排除周末（周六=5，周日=6）
            if current.weekday() < 5:
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return trading_days
    
    @staticmethod
    def get_date_range_info(start_date: Union[str, datetime], 
                           end_date: Union[str, datetime]) -> Dict[str, Any]:
        """
        获取日期范围信息
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            Dict[str, Any]: 日期范围信息
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        delta = end_date - start_date
        trading_days = TimeUtils.get_trading_days(start_date, end_date)
        
        return {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_days': delta.days + 1,
            'trading_days': len(trading_days),
            'weekends': delta.days + 1 - len(trading_days),
            'duration_str': f"{delta.days}天"
        }


# ================================================================================
# 5. 性能监控工具
# ================================================================================

class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self, name: str = "操作"):
        """
        初始化性能监控器
        
        参数:
            name: 操作名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
    
    def start(self) -> 'PerformanceMonitor':
        """开始监控"""
        self.start_time = time.time()
        self.logger.info(f"⏱️  开始执行 '{self.name}'...")
        return self
    
    def stop(self) -> float:
        """停止监控并返回耗时"""
        if self.start_time is None:
            self.logger.warning("性能监控未启动")
            return 0
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"✅ '{self.name}' 执行完成，耗时: {TimeUtils.format_duration(duration)}")
        return duration
    
    def __enter__(self):
        """上下文管理器入口"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        if exc_type is not None:
            self.logger.error(f"'{self.name}' 执行异常: {exc_val}")


# ================================================================================
# 6. 错误处理机制
# ================================================================================

class QuantError(Exception):
    """量化系统基础异常类"""
    pass


class DataError(QuantError):
    """数据相关异常"""
    pass


class ConfigError(QuantError):
    """配置相关异常"""
    pass


class ModelError(QuantError):
    """模型相关异常"""
    pass


def safe_execute(func: Callable, 
                error_message: str = "操作失败",
                default_return: Any = None,
                raise_on_error: bool = False) -> Tuple[bool, Any]:
    """
    安全执行函数，提供统一的错误处理
    
    参数:
        func: 要执行的函数
        error_message: 错误消息前缀
        default_return: 发生错误时的默认返回值
        raise_on_error: 是否在错误时抛出异常
    
    返回:
        Tuple[bool, Any]: (是否成功, 结果或默认值)
    """
    logger = LoggerManager.get_logger('safe_execute')
    
    try:
        result = func()
        return True, result
    except Exception as e:
        error_msg = f"{error_message}: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"错误详情:\n{traceback.format_exc()}")
        
        if raise_on_error:
            raise QuantError(error_msg) from e
        
        return False, default_return


@contextmanager
def error_context(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    错误处理上下文管理器
    
    参数:
        operation_name: 操作名称
        logger: 日志记录器
    """
    if logger is None:
        logger = LoggerManager.get_logger('error_context')
    
    try:
        logger.debug(f"开始执行: {operation_name}")
        yield
        logger.debug(f"完成执行: {operation_name}")
    except Exception as e:
        logger.error(f"执行失败 [{operation_name}]: {str(e)}")
        logger.debug(f"错误详情:\n{traceback.format_exc()}")
        raise


# ================================================================================
# 7. 配置和文件处理工具
# ================================================================================

class FileManager:
    """文件管理工具类"""
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], 
                   backup_suffix: str = None) -> Optional[Path]:
        """
        备份文件
        
        参数:
            file_path: 文件路径
            backup_suffix: 备份后缀
        
        返回:
            Optional[Path]: 备份文件路径，失败返回None
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            if backup_suffix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_suffix = f"backup_{timestamp}"
            
            backup_path = file_path.with_suffix(f".{backup_suffix}{file_path.suffix}")
            shutil.copy2(file_path, backup_path)
            
            logging.info(f"文件备份成功: {file_path} -> {backup_path}")
            return backup_path
            
        except Exception as e:
            logging.error(f"文件备份失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def safe_json_load(file_path: Union[str, Path]) -> Tuple[bool, Any]:
        """
        安全加载JSON文件
        
        参数:
            file_path: JSON文件路径
        
        返回:
            Tuple[bool, Any]: (是否成功, 数据或错误信息)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return True, data
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def safe_json_save(data: Any, file_path: Union[str, Path]) -> bool:
        """
        安全保存JSON文件
        
        参数:
            data: 要保存的数据
            file_path: 保存路径
        
        返回:
            bool: 是否保存成功
        """
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logging.error(f"保存JSON文件失败 {file_path}: {e}")
            return False


# ================================================================================
# 8. 数值计算工具
# ================================================================================

class MathUtils:
    """数学计算工具类"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        安全除法，避免除零错误
        
        参数:
            numerator: 分子
            denominator: 分母
            default: 分母为零时的默认值
        
        返回:
            float: 计算结果
        """
        # 检查输入有效性
        if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
            return default
        
        # 检查无穷大和NaN
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return default
        
        # 避免除零，使用较小的阈值检查
        if abs(denominator) < 1e-10:
            return default
        
        result = numerator / denominator
        
        # 检查结果有效性
        if not np.isfinite(result):
            return default
        
        return result
    
    @staticmethod
    def safe_log(value: float, base: float = np.e, default: float = 0.0) -> float:
        """
        安全对数计算
        
        参数:
            value: 输入值
            base: 对数底数
            default: 无效输入时的默认值
        
        返回:
            float: 对数结果
        """
        try:
            if value <= 0 or base <= 0 or base == 1:
                return default
            
            if base == np.e:
                result = np.log(value)
            else:
                result = np.log(value) / np.log(base)
            
            return result if np.isfinite(result) else default
        except:
            return default
    
    @staticmethod
    def normalize_array(arr: Union[List, np.ndarray], 
                       method: str = 'min-max') -> np.ndarray:
        """
        数组归一化
        
        参数:
            arr: 输入数组
            method: 归一化方法 ('min-max', 'z-score')
        
        返回:
            np.ndarray: 归一化后的数组
        """
        try:
            arr = np.array(arr, dtype=float)
            
            # 检查输入有效性
            if arr.size == 0:
                return arr
            
            # 移除无效值
            valid_mask = np.isfinite(arr)
            if not valid_mask.any():
                return np.zeros_like(arr)
            
            if method == 'min-max':
                min_val, max_val = arr[valid_mask].min(), arr[valid_mask].max()
                if abs(max_val - min_val) < 1e-10:
                    return np.zeros_like(arr)
                
                result = np.zeros_like(arr)
                result[valid_mask] = (arr[valid_mask] - min_val) / (max_val - min_val)
                return result
            
            elif method == 'z-score':
                mean_val, std_val = arr[valid_mask].mean(), arr[valid_mask].std()
                if std_val < 1e-10:
                    return np.zeros_like(arr)
                
                result = np.zeros_like(arr)
                result[valid_mask] = (arr[valid_mask] - mean_val) / std_val
                return result
            
            else:
                raise ValueError(f"未知的归一化方法: {method}")
                
        except Exception as e:
            logging.error(f"数组归一化失败: {e}")
            return np.zeros_like(np.array(arr)) if hasattr(arr, '__len__') else np.array([0.0])
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """
        计算百分比变化
        
        参数:
            old_value: 原值
            new_value: 新值
        
        返回:
            float: 百分比变化（小数形式）
        """
        return MathUtils.safe_divide(new_value - old_value, old_value, 0.0)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        将值限制在指定范围内
        
        参数:
            value: 输入值
            min_val: 最小值
            max_val: 最大值
        
        返回:
            float: 限制后的值
        """
        if not np.isfinite(value):
            return (min_val + max_val) / 2 if np.isfinite(min_val) and np.isfinite(max_val) else 0.0
        
        return max(min_val, min(max_val, value))


# ================================================================================
# 9. 便捷函数
# ================================================================================

def init_project_environment(log_level: str = 'INFO') -> Dict[str, Any]:
    """
    初始化项目环境
    
    参数:
        log_level: 日志级别
    
    返回:
        Dict[str, Any]: 环境信息
    """
    # 设置日志
    logger = LoggerManager.setup_logging(level=log_level)
    
    # 获取项目信息
    project_root = get_project_root()
    
    # 创建必要目录
    for directory in ['logs', 'models', 'results', 'cache']:
        ensure_directory(project_root / directory)
    
    env_info = {
        'project_root': str(project_root),
        'python_version': sys.version,
        'log_level': log_level,
        'initialized_at': datetime.now().isoformat()
    }
    
    logger.info("项目环境初始化完成")
    logger.debug(f"环境信息: {env_info}")
    
    return env_info


def format_dict_for_display(data: Dict[str, Any], 
                           indent: int = 2,
                           max_length: int = 100) -> str:
    """
    格式化字典用于显示
    
    参数:
        data: 要格式化的字典
        indent: 缩进空格数
        max_length: 单行最大长度
    
    返回:
        str: 格式化后的字符串
    """
    def format_value(value, current_indent=0):
        spaces = ' ' * current_indent
        
        if isinstance(value, dict):
            if not value:
                return '{}'
            items = []
            for k, v in value.items():
                formatted_value = format_value(v, current_indent + indent)
                items.append(f"{spaces}  {k}: {formatted_value}")
            return "{\n" + "\n".join(items) + f"\n{spaces}}}"
        
        elif isinstance(value, list):
            if not value:
                return '[]'
            if len(str(value)) <= max_length:
                return str(value)
            items = [f"{spaces}  {format_value(item, current_indent + indent)}" for item in value]
            return "[\n" + "\n".join(items) + f"\n{spaces}]"
        
        else:
            value_str = str(value)
            return value_str if len(value_str) <= max_length else value_str[:max_length] + "..."
    
    return format_value(data)


# ================================================================================
# 模块导出
# ================================================================================

__all__ = [
    # 基础工具
    'get_project_root', 'ensure_directory', 'safe_file_operation',
    
    # 日志管理
    'LoggerManager',
    
    # 数据验证
    'DataValidator',
    
    # 时间处理
    'TimeUtils',
    
    # 性能监控
    'PerformanceMonitor',
    
    # 错误处理
    'QuantError', 'DataError', 'ConfigError', 'ModelError',
    'safe_execute', 'error_context',
    
    # 文件管理
    'FileManager',
    
    # 数值计算
    'MathUtils',
    
    # 便捷函数
    'init_project_environment', 'format_dict_for_display'
] 