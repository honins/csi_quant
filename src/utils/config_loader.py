#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器模块
负责加载和合并多个配置文件

功能：
- 多配置文件加载
- 配置合并和覆盖
- 环境变量支持
- 配置验证
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .common import get_project_root, ensure_directory


def load_config(config_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
        config_paths: 配置文件路径列表，如果为None则使用默认路径
    
    返回:
        Dict[str, Any]: 合并后的配置字典
    """
    project_root = get_project_root()
    
    # 默认配置文件路径
    if config_paths is None:
        config_paths = [
            str(project_root / 'config' / 'system.yaml'),
            str(project_root / 'config' / 'strategy.yaml')
        ]
    
    # 添加环境变量指定的配置文件
    env_config = os.getenv('CSI_CONFIG_PATH')
    if env_config:
        config_paths.append(env_config)
    
    merged_config = {}
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                # 深度合并配置
                merged_config = deep_merge_dict(merged_config, config_data)
                logging.debug(f"加载配置文件: {config_path}")
            else:
                logging.debug(f"配置文件不存在，跳过: {config_path}")
                
        except Exception as e:
            logging.warning(f"加载配置文件失败 {config_path}: {e}")
            continue
    
    # 如果没有加载到任何配置，返回默认配置
    if not merged_config:
        logging.warning("未能加载任何配置文件，使用默认配置")
        merged_config = get_default_config()
    
    return merged_config


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并字典
    
    参数:
        base: 基础字典
        override: 覆盖字典
    
    返回:
        Dict[str, Any]: 合并后的字典
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge_dict(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    返回:
        Dict[str, Any]: 默认配置字典
    """
    return {
        'data': {
            'data_file_path': 'data/',
            'index_code': '000905',
            'frequency': '1d'
        },
        'strategy': {
            'rise_threshold': 0.04,
            'max_days': 20,
            'rsi_oversold_threshold': 30,
            'rsi_low_threshold': 40
        },
        'ai': {
            'model_type': 'RandomForest',
            'models_dir': 'models',
            'enable': True
        },
        'logging': {
            'level': 'INFO',
            'file_path': 'logs/system.log'
        },
        'results': {
            'output_dir': 'results'
        },
        'backtest': {
            'start_date': '2020-01-01',
            'end_date': '2024-12-31'
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    保存配置到文件
    
    参数:
        config: 配置字典
        config_path: 保存路径
    
    返回:
        bool: 是否保存成功
    """
    try:
        config_file = Path(config_path)
        ensure_directory(config_file.parent)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logging.info(f"配置已保存到: {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"保存配置失败 {config_path}: {e}")
        return False


class ConfigLoader:
    """
    配置加载器类
    提供面向对象的配置管理接口
    """
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        初始化配置加载器
        
        参数:
            config_paths: 配置文件路径列表
        """
        self.config_paths = config_paths
        self._config = None
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        """
        加载配置
        
        返回:
            Dict[str, Any]: 配置字典
        """
        if not self._loaded:
            self._config = load_config(self.config_paths)
            self._loaded = True
        return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置（如果未加载则先加载）
        
        返回:
            Dict[str, Any]: 配置字典
        """
        if not self._loaded:
            self.load()
        return self._config or {}
    
    def get_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取配置段
        
        参数:
            section: 配置段名称
            default: 默认值
        
        返回:
            Dict[str, Any]: 配置段内容
        """
        config = self.get_config()
        return config.get(section, default or {})
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """
        更新配置段
        
        参数:
            section: 配置段名称
            updates: 更新内容
        """
        config = self.get_config()
        if section not in config:
            config[section] = {}
        config[section].update(updates)
    
    def save(self, config_path: str) -> bool:
        """
        保存配置到文件
        
        参数:
            config_path: 保存路径
        
        返回:
            bool: 是否保存成功
        """
        if self._config:
            return save_config(self._config, config_path)
        return False
    
    def reload(self) -> Dict[str, Any]:
        """
        重新加载配置
        
        返回:
            Dict[str, Any]: 配置字典
        """
        self._loaded = False
        return self.load()
    
    @classmethod
    def load_default(cls) -> 'ConfigLoader':
        """
        使用默认路径创建配置加载器
        
        返回:
            ConfigLoader: 配置加载器实例
        """
        loader = cls()
        loader.load()
        return loader


# 模块导出
__all__ = ['load_config', 'save_config', 'get_default_config', 'deep_merge_dict', 'ConfigLoader'] 