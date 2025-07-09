#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多配置文件加载器
支持加载和合并多个YAML配置文件，实现配置文件的模块化管理
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    多配置文件加载器
    
    支持功能：
    1. 加载多个配置文件并合并
    2. 配置文件优先级管理
    3. 环境变量路径支持
    4. 配置验证和错误处理
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化配置加载器
        
        参数:
        base_dir: 配置文件基础目录，默认为项目根目录的config文件夹
        """
        if base_dir is None:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            base_dir = project_root / 'config'
        
        self.base_dir = Path(base_dir)
        self.merged_config = {}
        
    def load_config(self, 
                   config_files: List[str] = None, 
                   custom_path: str = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        参数:
        config_files: 配置文件列表，按优先级排序（后面的会覆盖前面的）
        custom_path: 自定义配置文件路径（通过环境变量CSI_CONFIG_PATH指定）
        
        返回:
        dict: 合并后的配置字典
        """
        if config_files is None:
            config_files = [
                'config_core.yaml',      # 核心系统配置
                'optimization.yaml',     # 优化配置
                'config.yaml'            # 兼容性配置（如果存在）
            ]
        
        # 检查环境变量配置
        env_config_path = custom_path or os.environ.get('CSI_CONFIG_PATH')
        if env_config_path:
            config_files.append(env_config_path)
        
        logger.info("开始加载配置文件...")
        
        merged_config = {}
        loaded_files = []
        
        for config_file in config_files:
            try:
                # 处理绝对路径和相对路径
                if os.path.isabs(config_file):
                    config_path = Path(config_file)
                else:
                    config_path = self.base_dir / config_file
                
                if config_path.exists():
                    config_data = self._load_single_config(config_path)
                    if config_data:
                        merged_config = self._deep_merge(merged_config, config_data)
                        loaded_files.append(str(config_path))
                        logger.info(f"✅ 已加载配置文件: {config_path.name}")
                else:
                    logger.debug(f"⚠️ 配置文件不存在，跳过: {config_path}")
                    
            except Exception as e:
                logger.error(f"❌ 加载配置文件失败 {config_file}: {e}")
                continue
        
        if not loaded_files:
            logger.error("❌ 没有成功加载任何配置文件")
            raise FileNotFoundError("无法加载任何配置文件")
        
        logger.info(f"📁 配置加载完成，共加载 {len(loaded_files)} 个文件")
        for file_path in loaded_files:
            logger.info(f"   - {os.path.basename(file_path)}")
        
        self.merged_config = merged_config
        return merged_config
    
    def _load_single_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """
        加载单个配置文件
        
        参数:
        config_path: 配置文件路径
        
        返回:
        dict: 配置字典，如果加载失败返回None
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            if config_data is None:
                logger.warning(f"配置文件为空: {config_path}")
                return {}
            
            return config_data
            
        except yaml.YAMLError as e:
            logger.error(f"YAML格式错误 {config_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"读取配置文件失败 {config_path}: {e}")
            return None
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        参数:
        base_dict: 基础字典
        update_dict: 更新字典
        
        返回:
        dict: 合并后的字典
        """
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        获取特定的配置部分
        
        参数:
        section: 配置部分名称
        
        返回:
        dict: 指定部分的配置
        """
        return self.merged_config.get(section, {})
    
    def save_config_section(self, section: str, data: Dict[str, Any], target_file: str = None):
        """
        保存配置部分到指定文件
        
        参数:
        section: 配置部分名称
        data: 要保存的数据
        target_file: 目标文件名，默认根据section确定
        """
        if target_file is None:
            if section in ['optimization', 'validation', 'bayesian_optimization', 'genetic_algorithm']:
                target_file = 'optimization.yaml'
            else:
                target_file = 'config_core.yaml'
        
        target_path = self.base_dir / target_file
        
        try:
            # 加载现有配置
            if target_path.exists():
                existing_config = self._load_single_config(target_path)
            else:
                existing_config = {}
            
            # 更新指定部分
            existing_config[section] = data
            
            # 保存回文件
            with open(target_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ 配置部分 '{section}' 已保存到: {target_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存配置失败: {e}")
            raise
    
    def validate_config(self) -> List[str]:
        """
        验证配置完整性
        
        返回:
        list: 验证错误列表，空列表表示验证通过
        """
        errors = []
        required_sections = ['ai', 'data', 'strategy', 'backtest', 'logging']
        
        for section in required_sections:
            if section not in self.merged_config:
                errors.append(f"缺少必需的配置部分: {section}")
        
        # 验证AI配置
        ai_config = self.merged_config.get('ai', {})
        if not ai_config.get('models_dir'):
            errors.append("AI配置缺少models_dir")
        
        # 验证数据配置
        data_config = self.merged_config.get('data', {})
        if not data_config.get('data_file_path'):
            errors.append("数据配置缺少data_file_path")
        
        return errors
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("📋 配置文件摘要")
        print("="*60)
        
        for section, config in self.merged_config.items():
            if isinstance(config, dict):
                print(f"\n📁 {section.upper()}:")
                for key, value in config.items():
                    if isinstance(value, dict):
                        print(f"   📂 {key}: {len(value)} 个子项")
                    elif isinstance(value, list):
                        print(f"   📋 {key}: {len(value)} 个项目")
                    else:
                        print(f"   📄 {key}: {value}")
        
        print("="*60)

# 全局配置加载器实例
_global_loader = None

def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader

def load_config(config_files: List[str] = None, custom_path: str = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置
    
    参数:
    config_files: 配置文件列表
    custom_path: 自定义配置路径
    
    返回:
    dict: 合并后的配置
    """
    loader = get_config_loader()
    return loader.load_config(config_files, custom_path) 