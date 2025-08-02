#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复optimization_ranges工具
添加缺失的优化参数范围
"""

import yaml
import os
from typing import Dict, List, Any
from src.utils.param_config import get_all_optimizable_params

def load_config(config_path: str = 'config/strategy.yaml') -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_config(config: Dict[str, Any], config_path: str = 'config/strategy.yaml'):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def get_missing_optimization_ranges(config: Dict[str, Any]) -> List[str]:
    """获取缺失的优化范围参数"""
    optimizable_params = get_all_optimizable_params()
    existing_ranges = config.get('optimization_ranges', {}).keys()
    
    missing_params = []
    for param in optimizable_params:
        if param not in existing_ranges:
            missing_params.append(param)
    
    return missing_params

def get_default_optimization_ranges(config: Dict[str, Any]) -> Dict[str, Any]:
    """从配置文件获取默认的优化范围参数"""
    return config.get('default_optimization_ranges', {})

def add_missing_optimization_ranges(config: Dict[str, Any]) -> Dict[str, Any]:
    """添加缺失的优化范围参数"""
    missing_params = get_missing_optimization_ranges(config)
    
    if not missing_params:
        print("✅ 所有可优化参数都已配置优化范围")
        return config
    
    print(f"🔧 发现 {len(missing_params)} 个缺失的优化范围参数")
    
    # 从配置文件获取默认的优化范围
    default_ranges = get_default_optimization_ranges(config)
    
    if not default_ranges:
        print("❌ 配置文件中未找到 default_optimization_ranges 配置")
        return config
    
    # 添加缺失的参数
    optimization_ranges = config.get('optimization_ranges', {})
    added_count = 0
    for param in missing_params:
        if param in default_ranges:
            optimization_ranges[param] = default_ranges[param]
            added_count += 1
    
    config['optimization_ranges'] = optimization_ranges
    print(f"✅ 成功添加 {added_count} 个参数")
    return config

def main():
    """主函数"""
    print("🔧 修复optimization_ranges配置")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 添加缺失的参数
    config = add_missing_optimization_ranges(config)
    
    # 保存配置
    missing_params = get_missing_optimization_ranges(config)
    if missing_params:
        save_config(config)
        print("✅ 配置文件已更新")
    
    print("=" * 50)

if __name__ == "__main__":
    main()