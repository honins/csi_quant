#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复optimization_ranges工具
添加缺失的优化参数范围
"""

import yaml
import os
from typing import Dict, List, Any
from src.utils.param_config import get_all_optimizable_params, OPTIMIZABLE_PARAMS

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

def add_missing_optimization_ranges(config: Dict[str, Any]) -> Dict[str, Any]:
    """添加缺失的优化范围参数"""
    missing_params = get_missing_optimization_ranges(config)
    
    if not missing_params:
        print("✅ 所有可优化参数都已配置优化范围")
        return config
    
    print(f"🔧 发现 {len(missing_params)} 个缺失的优化范围参数:")
    for param in missing_params:
        print(f"   - {param}")
    
    # 定义默认的优化范围（14个有效参数）
    default_ranges = {
        # �� 核心决策参数（2个）
        'rsi_oversold_threshold': {
            'max': 35,
            'min': 25,
            'step': 1
        },
        'rsi_low_threshold': {
            'max': 50,
            'min': 40,
            'step': 1
        },
        
        # 🔥 基础权重参数（4个）
        'ma_all_below': {
            'max': 0.4,
            'min': 0.2,
            'step': 0.02
        },
        'dynamic_confidence_adjustment': {
            'max': 0.25,
            'min': 0.05,
            'step': 0.02
        },
        'market_sentiment_weight': {
            'max': 0.25,
            'min': 0.08,
            'step': 0.02
        },
        'trend_strength_weight': {
            'max': 0.25,
            'min': 0.1,
            'step': 0.02
        },
        
        # 🔥 成交量逻辑参数（4个）
        'volume_panic_threshold': {
            'max': 2.0,
            'min': 1.2,
            'step': 0.05
        },
        'volume_panic_bonus': {
            'max': 0.2,
            'min': 0.05,
            'step': 0.02
        },
        'volume_surge_bonus': {
            'max': 0.15,
            'min': 0.02,
            'step': 0.02
        },
        'volume_shrink_penalty': {
            'max': 0.8,
            'min': 0.5,
            'step': 0.05
        },
        
        # 🔥 技术指标参数（4个）
        'bb_near_threshold': {
            'max': 1.05,
            'min': 1.005,
            'step': 0.005
        },
        'recent_decline': {
            'max': 0.3,
            'min': 0.1,
            'step': 0.02
        },
        'macd_negative': {
            'max': 0.15,
            'min': 0.05,
            'step': 0.02
        },
        'price_decline_threshold': {
            'max': -0.01,
            'min': -0.06,
            'step': 0.005
        }
    }
    
    # 添加缺失的参数
    optimization_ranges = config.get('optimization_ranges', {})
    for param in missing_params:
        if param in default_ranges:
            optimization_ranges[param] = default_ranges[param]
            print(f"✅ 添加参数: {param}")
        else:
            print(f"⚠️ 未找到参数 {param} 的默认范围")
    
    config['optimization_ranges'] = optimization_ranges
    return config

def main():
    """主函数"""
    print("🔧 修复optimization_ranges配置")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 检查缺失的参数
    missing_params = get_missing_optimization_ranges(config)
    
    if missing_params:
        print(f"📊 发现 {len(missing_params)} 个缺失的优化范围参数")
        
        # 添加缺失的参数
        config = add_missing_optimization_ranges(config)
        
        # 保存配置
        save_config(config)
        print("✅ 配置文件已更新")
        
        # 验证修复结果
        final_missing = get_missing_optimization_ranges(config)
        if not final_missing:
            print("✅ 所有参数都已正确配置")
        else:
            print(f"❌ 仍有 {len(final_missing)} 个参数缺失: {final_missing}")
    else:
        print("✅ 所有可优化参数都已配置优化范围")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 