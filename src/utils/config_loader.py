#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器
负责加载和合并主配置文件和策略配置文件
"""

import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        参数:
        config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.logger = logging.getLogger('ConfigLoader')
        
    def load_config(self) -> Dict[str, Any]:
        """
        加载完整配置（主配置 + 策略配置）
        
        返回:
        dict: 合并后的配置字典
        """
        try:
            # 加载主配置文件
            main_config = self._load_yaml_file("config.yaml")
            self.logger.info("主配置文件加载成功")
            
            # 加载策略配置文件
            strategy_config = self._load_yaml_file("strategy_config.yaml")
            self.logger.info("策略配置文件加载成功")
            
            # 合并配置
            merged_config = self._merge_configs(main_config, strategy_config)
            self.logger.info("配置合并完成")
            
            return merged_config
            
        except Exception as e:
            self.logger.error("加载配置失败: %s", str(e))
            raise
    
    def load_main_config(self) -> Dict[str, Any]:
        """
        仅加载主配置文件
        
        返回:
        dict: 主配置字典
        """
        return self._load_yaml_file("config.yaml")
    
    def load_strategy_config(self) -> Dict[str, Any]:
        """
        仅加载策略配置文件
        
        返回:
        dict: 策略配置字典
        """
        return self._load_yaml_file("strategy_config.yaml")
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """
        加载YAML文件
        
        参数:
        filename: 文件名
        
        返回:
        dict: 配置字典
        """
        file_path = os.path.join(self.config_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                return {}
            
            return config
            
        except yaml.YAMLError as e:
            self.logger.error("YAML文件解析失败 %s: %s", filename, str(e))
            raise
        except Exception as e:
            self.logger.error("读取配置文件失败 %s: %s", filename, str(e))
            raise
    
    def _merge_configs(self, main_config: Dict[str, Any], strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并主配置和策略配置
        
        参数:
        main_config: 主配置字典
        strategy_config: 策略配置字典
        
        返回:
        dict: 合并后的配置字典
        """
        merged_config = main_config.copy()
        
        # 将策略配置合并到主配置中
        for key, value in strategy_config.items():
            if key in merged_config:
                # 如果键已存在，递归合并
                if isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key] = self._deep_merge(merged_config[key], value)
                else:
                    # 策略配置优先
                    merged_config[key] = value
            else:
                # 新键直接添加
                merged_config[key] = value
        
        return merged_config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        参数:
        dict1: 字典1
        dict2: 字典2
        
        返回:
        dict: 合并后的字典
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any], filename: str) -> bool:
        """
        保存配置到文件
        
        参数:
        config: 配置字典
        filename: 文件名
        
        返回:
        bool: 是否保存成功
        """
        try:
            file_path = os.path.join(self.config_dir, filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info("配置保存成功: %s", file_path)
            return True
            
        except Exception as e:
            self.logger.error("保存配置失败: %s", str(e))
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的完整性
        
        参数:
        config: 配置字典
        
        返回:
        bool: 配置是否有效
        """
        try:
            required_sections = ['ai', 'data', 'logging', 'strategy']
            
            for section in required_sections:
                if section not in config:
                    self.logger.error("缺少必需的配置节: %s", section)
                    return False
            
            # 验证策略配置
            if 'strategy' in config:
                strategy = config['strategy']
                required_strategy_params = ['rise_threshold', 'max_days', 'ma_periods']
                
                for param in required_strategy_params:
                    if param not in strategy:
                        self.logger.error("缺少必需的策略参数: %s", param)
                        return False
            
            self.logger.info("配置验证通过")
            return True
            
        except Exception as e:
            self.logger.error("配置验证失败: %s", str(e))
            return False
    
    def get_config_path(self, filename: str) -> str:
        """
        获取配置文件完整路径
        
        参数:
        filename: 文件名
        
        返回:
        str: 完整路径
        """
        return os.path.join(self.config_dir, filename)
    
    def list_config_files(self) -> list:
        """
        列出配置文件目录中的所有配置文件
        
        返回:
        list: 配置文件列表
        """
        try:
            config_files = []
            for file in os.listdir(self.config_dir):
                if file.endswith(('.yaml', '.yml')):
                    config_files.append(file)
            return config_files
        except Exception as e:
            self.logger.error("列出配置文件失败: %s", str(e))
            return []


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """
    便捷函数：加载完整配置
    
    参数:
    config_dir: 配置文件目录
    
    返回:
    dict: 合并后的配置字典
    """
    loader = ConfigLoader(config_dir)
    return loader.load_config()


def load_main_config(config_dir: str = "config") -> Dict[str, Any]:
    """
    便捷函数：仅加载主配置
    
    参数:
    config_dir: 配置文件目录
    
    返回:
    dict: 主配置字典
    """
    loader = ConfigLoader(config_dir)
    return loader.load_main_config()


def load_strategy_config(config_dir: str = "config") -> Dict[str, Any]:
    """
    便捷函数：仅加载策略配置
    
    参数:
    config_dir: 配置文件目录
    
    返回:
    dict: 策略配置字典
    """
    loader = ConfigLoader(config_dir)
    return loader.load_strategy_config()


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = load_config()
        print("配置加载成功")
        print(f"配置键: {list(config.keys())}")
        
        if 'strategy' in config:
            print(f"策略配置: {config['strategy']}")
        
    except Exception as e:
        print(f"配置加载失败: {e}") 