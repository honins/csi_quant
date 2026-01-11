#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础模块类
为项目中的所有模块提供统一的基础功能

特性：
- 统一的初始化流程
- 标准化的配置管理
- 统一的日志处理
- 通用的错误处理
- 目录管理自动化
- 性能监控集成
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .common import (
    LoggerManager, PerformanceMonitor,
    ensure_directory, get_project_root, error_context,
    ConfigError
)
# 移除未使用的 ConfigLoader 导入


class BaseModule(ABC):
    """
    基础模块抽象类
    
    所有业务模块都应该继承此类，获得标准化的基础功能
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 module_name: Optional[str] = None,
                 log_level: str = 'INFO',
                 auto_setup_dirs: bool = True):
        """
        初始化基础模块
        
        参数:
            config: 配置字典
            module_name: 模块名称，用于日志和目录管理
            log_level: 日志级别
            auto_setup_dirs: 是否自动创建所需目录
        """
        # 设置模块名称
        self.module_name = module_name or self.__class__.__name__
        
        # 设置项目根目录
        self.project_root = get_project_root()
        
        # 初始化日志
        self.logger = LoggerManager.get_logger(self.module_name)
        
        # 存储配置
        self.config = config
        self._validate_base_config()
        
        # 初始化性能监控
        self.performance_monitor = PerformanceMonitor(f"{self.module_name}模块")
        
        # 自动设置目录
        if auto_setup_dirs:
            self._setup_directories()
        
        # 初始化模块状态
        self._initialized = False
        self._setup_time = datetime.now()
        
        # 调用子类的初始化方法
        with error_context(f"初始化{self.module_name}模块", self.logger):
            self._initialize_module()
            self._initialized = True
            
        self.logger.info(f"{self.module_name}模块初始化完成")
    
    @abstractmethod
    def _initialize_module(self):
        """
        子类必须实现的初始化方法
        在基础初始化完成后调用
        """
        pass
    
    def _validate_base_config(self):
        """验证基础配置"""
        if not isinstance(self.config, dict):
            raise ConfigError("配置必须是字典类型")
        
        # 子类可以重写此方法添加特定验证
        self._validate_module_config()
    
    def _validate_module_config(self):
        """
        子类可重写的配置验证方法
        默认实现为空，子类根据需要实现
        """
        pass
    
    def _setup_directories(self):
        """设置必要的目录"""
        # 基础目录
        base_dirs = self._get_required_directories()
        for directory in base_dirs:
            ensure_directory(self.project_root / directory)
        
        # 模块特定目录
        module_dirs = self._get_module_directories()
        for directory in module_dirs:
            ensure_directory(directory)
    
    def _get_required_directories(self) -> List[str]:
        """
        获取必需的基础目录列表
        子类可以重写以添加特定目录
        """
        return ['logs', 'models', 'results', 'cache']
    
    def _get_module_directories(self) -> List[Path]:
        """
        获取模块特定的目录列表
        子类可以重写以添加特定目录
        """
        return []
    
    def get_config_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取配置的特定部分
        
        参数:
            section: 配置部分名称
            default: 默认值
        
        返回:
            Dict[str, Any]: 配置部分
        """
        if default is None:
            default = {}
        return self.config.get(section, default)
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        更新配置的特定部分
        
        参数:
            section: 配置部分名称
            updates: 更新的内容
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
        self.logger.debug(f"更新配置部分 '{section}': {updates}")
    
    def get_file_path(self, file_type: str, filename: str) -> Path:
        """
        获取标准化的文件路径
        
        参数:
            file_type: 文件类型 ('logs', 'models', 'results', 'cache', 'data')
            filename: 文件名
        
        返回:
            Path: 文件路径
        """
        type_map = {
            'logs': 'logs',
            'models': 'models', 
            'results': 'results',
            'cache': 'cache',
            'data': 'data'
        }
        
        if file_type not in type_map:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        directory = self.project_root / type_map[file_type]
        ensure_directory(directory)
        
        return directory / filename
    
    def safe_operation(self, 
                      operation_name: str,
                      operation_func,
                      *args,
                      **kwargs) -> tuple:
        """
        安全执行操作，包含错误处理和性能监控
        
        参数:
            operation_name: 操作名称
            operation_func: 操作函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        返回:
            tuple: (是否成功, 结果或错误信息)
        """
        with PerformanceMonitor(f"{self.module_name}.{operation_name}"):
            try:
                with error_context(operation_name, self.logger):
                    result = operation_func(*args, **kwargs)
                    return True, result
            except Exception as e:
                self.logger.error(f"操作失败 [{operation_name}]: {str(e)}")
                return False, str(e)
    

    def get_status(self) -> Dict[str, Any]:
        """
        获取模块状态信息
        
        返回:
            Dict[str, Any]: 状态信息
        """
        return {
            'module_name': self.module_name,
            'initialized': self._initialized,
            'setup_time': self._setup_time.isoformat(),
            'project_root': str(self.project_root),
            'config_sections': list(self.config.keys()) if self.config else []
        }
    
    def log_performance(self, operation: str, duration: float):
        """
        记录性能信息
        
        参数:
            operation: 操作名称
            duration: 执行时间（秒）
        """
        self.logger.info(f"性能统计 [{operation}]: {duration:.3f}秒")
        
        # 可以扩展为保存到性能数据库或文件
        performance_data = {
            'module': self.module_name,
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # 这里可以添加性能数据持久化逻辑
        self._save_performance_data(performance_data)
    
    def _save_performance_data(self, performance_data: Dict[str, Any]):
        """
        保存性能数据
        子类可以重写此方法实现特定的性能数据保存逻辑
        """
        # 默认实现：保存到日志
        self.logger.debug(f"性能数据: {performance_data}")
    
    def cleanup(self):
        """
        清理资源
        子类可以重写此方法实现特定的清理逻辑
        """
        self.logger.info(f"{self.module_name}模块开始清理资源")
        
        # 调用子类的清理方法
        self._cleanup_module()
        
        self.logger.info(f"{self.module_name}模块清理完成")
    
    def _cleanup_module(self):
        """
        子类可重写的清理方法
        默认实现为空
        """
        pass
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
        if exc_type is not None:
            self.logger.error(f"{self.module_name}模块执行异常: {exc_val}")


# 已移除重复的 DataModule 基类定义（请使用 src.data.data_module.DataModule）


# 已移除重复的 StrategyModule 基类定义（请使用 src.strategy.strategy_module.StrategyModule）


# 已移除重复的 AIModule 基类定义（请使用 src.ai.ai_optimizer_improved.AIOptimizerImproved 或相关AI模块）


# 已移除 create_module 工厂函数（不再在utils层创建具体模块）


# 模块导出
__all__ = [
    'BaseModule'
]