#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件保存模块（保留注释版）
使用ruamel.yaml确保保存配置时保留所有注释、格式和空行
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

class CommentPreservingConfigSaver:
    """
    保留注释的配置文件保存器
    
    特点：
    1. 保留原有的注释内容
    2. 保持原有的格式和缩进
    3. 保留空行和文档结构
    4. 仅更新需要修改的配置值
    """
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置保存器
        
        参数:
        config_dir: 配置文件目录，默认为项目根目录的config文件夹
        """
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_dir = project_root / 'config'
        
        self.config_dir = Path(config_dir)
        
        # 初始化ruamel.yaml
        self.yaml = YAML()
        self.yaml.preserve_quotes = True  # 保留引号
        self.yaml.width = 4096  # 避免长行自动换行
        self.yaml.indent(mapping=2, sequence=4, offset=2)  # 设置缩进
        
    def save_optimized_parameters(self, 
                                optimized_params: Dict[str, Any], 
                                target_file: str = None) -> bool:
        """
        保存优化后的参数，保留所有注释
        
        参数:
        optimized_params: 优化后的参数字典
        target_file: 目标配置文件，默认自动选择
        
        返回:
        bool: 是否保存成功
        """
        try:
            # 自动选择目标文件
            if target_file is None:
                target_file = self._determine_target_file(optimized_params)
            
            target_path = self.config_dir / target_file
            
            if not target_path.exists():
                logger.error(f"目标配置文件不存在: {target_path}")
                return False
            
            # 读取原始配置文件（保留注释）
            with open(target_path, 'r', encoding='utf-8') as f:
                config_data = self.yaml.load(f)
            
            if config_data is None:
                config_data = {}
            
            # 更新配置参数
            self._update_config_recursively(config_data, optimized_params)
            
            # 备份原文件
            backup_path = target_path.with_suffix(f'.backup_{self._get_timestamp()}')
            target_path.rename(backup_path)
            logger.info(f"📁 原配置文件已备份到: {backup_path.name}")
            
            # 保存更新后的配置（保留注释）
            with open(target_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(config_data, f)
            
            logger.info(f"✅ 配置参数已保存到: {target_file}（注释已保留）")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存优化参数失败: {e}")
            return False
    
    def _determine_target_file(self, params: Dict[str, Any]) -> str:
        """
        根据参数类型自动确定目标文件
        
        参数:
        params: 参数字典
        
        返回:
        str: 目标文件名
        """
        # 检查参数类型，决定保存到哪个文件
        if any(key in params for key in ['optimization', 'bayesian_optimization', 'genetic_algorithm']):
            return 'optimization.yaml'
        elif any(key in params for key in ['confidence_smoothing', 'advanced_optimization']):
            return 'config_improved.yaml'
        else:
            return 'config_core.yaml'
    
    def _update_config_recursively(self, base_config: Dict[str, Any], updates: Dict[str, Any]):
        """
        递归更新配置字典
        
        参数:
        base_config: 基础配置字典
        updates: 更新内容
        """
        for key, value in updates.items():
            if key in base_config:
                if isinstance(base_config[key], dict) and isinstance(value, dict):
                    # 递归更新字典
                    self._update_config_recursively(base_config[key], value)
                else:
                    # 直接更新值
                    base_config[key] = value
            else:
                # 新增键值对
                base_config[key] = value
    
    def _get_timestamp(self) -> str:
        """获取时间戳字符串"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def save_strategy_parameters(self, strategy_params: Dict[str, Any]) -> bool:
        """
        保存策略参数（针对strategy配置节）
        
        参数:
        strategy_params: 策略参数字典
        
        返回:
        bool: 是否保存成功
        """
        return self.save_optimized_parameters({'strategy': strategy_params})
    
    def save_ai_parameters(self, ai_params: Dict[str, Any]) -> bool:
        """
        保存AI参数（针对ai配置节）
        
        参数:
        ai_params: AI参数字典
        
        返回:
        bool: 是否保存成功
        """
        return self.save_optimized_parameters({'ai': ai_params})
    
    def batch_save_parameters(self, 
                            parameter_groups: Dict[str, Dict[str, Any]], 
                            target_files: Dict[str, str] = None) -> Dict[str, bool]:
        """
        批量保存多组参数到不同文件
        
        参数:
        parameter_groups: 参数组字典，格式为 {'组名': {'参数': '值'}}
        target_files: 目标文件映射，格式为 {'组名': '文件名'}
        
        返回:
        dict: 保存结果，格式为 {'组名': 是否成功}
        """
        results = {}
        
        for group_name, params in parameter_groups.items():
            try:
                target_file = None
                if target_files and group_name in target_files:
                    target_file = target_files[group_name]
                
                success = self.save_optimized_parameters(params, target_file)
                results[group_name] = success
                
                if success:
                    logger.info(f"✅ 参数组 '{group_name}' 保存成功")
                else:
                    logger.error(f"❌ 参数组 '{group_name}' 保存失败")
                    
            except Exception as e:
                logger.error(f"❌ 保存参数组 '{group_name}' 时发生错误: {e}")
                results[group_name] = False
        
        return results

# 全局配置保存器实例
_global_saver = None

def get_config_saver() -> CommentPreservingConfigSaver:
    """获取全局配置保存器实例"""
    global _global_saver
    if _global_saver is None:
        _global_saver = CommentPreservingConfigSaver()
    return _global_saver

def save_optimized_config(optimized_params: Dict[str, Any], 
                         target_file: str = None) -> bool:
    """
    便捷函数：保存优化后的配置参数（保留注释）
    
    参数:
    optimized_params: 优化后的参数字典
    target_file: 目标文件名
    
    返回:
    bool: 是否保存成功
    """
    saver = get_config_saver()
    return saver.save_optimized_parameters(optimized_params, target_file)

def save_strategy_config(strategy_params: Dict[str, Any]) -> bool:
    """
    便捷函数：保存策略配置参数（保留注释）
    
    参数:
    strategy_params: 策略参数字典
    
    返回:
    bool: 是否保存成功
    """
    saver = get_config_saver()
    return saver.save_strategy_parameters(strategy_params)

def save_ai_config(ai_params: Dict[str, Any]) -> bool:
    """
    便捷函数：保存AI配置参数（保留注释）
    
    参数:
    ai_params: AI参数字典
    
    返回:
    bool: 是否保存成功
    """
    saver = get_config_saver()
    return saver.save_ai_parameters(ai_params)

# =============================================================================
# 使用示例
# =============================================================================
"""
# 在AI优化完成后，使用以下方式保存配置（保留注释）：

from src.utils.config_saver import save_optimized_config, save_strategy_config

# 方法1：保存完整的优化结果
optimized_params = {
    'strategy': {
        'confidence_weights': {
            'final_threshold': 0.55,
            'volume_weight': 0.28
        }
    },
    'ai': {
        'scoring': {
            'success_weight': 0.45
        }
    }
}
save_optimized_config(optimized_params)

# 方法2：仅保存策略参数
strategy_params = {
    'confidence_weights': {
        'final_threshold': 0.55,
        'volume_weight': 0.28
    }
}
save_strategy_config(strategy_params)

""" 