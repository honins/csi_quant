#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重置脚本

功能：快速重置策略参数到安全的默认值
使用：python quick_reset.py
"""

import os
import yaml
from datetime import datetime
from pathlib import Path


def quick_reset():
    """快速重置核心参数"""
    
    project_root = Path(__file__).parent
    config_dir = project_root / "config"
    
    # 创建备份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = config_dir / "backups" / f"quick_reset_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 创建备份: {backup_dir}")
    
    # 备份当前配置
    strategy_path = config_dir / "strategy.yaml"
    if strategy_path.exists():
        with open(strategy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_dir / "strategy_backup.yaml", 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 核心默认参数
    core_defaults = {
        "strategy": {
            "rise_threshold": 0.04,
            "max_days": 20,
            "confidence_weights": {
                "rsi_oversold_threshold": 30,
                "rsi_low_threshold": 40,
                "final_threshold": 0.5,
                "dynamic_confidence_adjustment": 0.1,
                "market_sentiment_weight": 0.15,
                "trend_strength_weight": 0.15
            }
        },
        "confidence_weights": {
            "final_threshold": 0.5,
            "rsi_oversold_threshold": 30.0,
            "rsi_low_threshold": 40.0,
            "dynamic_confidence_adjustment": 0.1,
            "market_sentiment_weight": 0.15,
            "trend_strength_weight": 0.15,
            "bb_lower_near": 0.3,
            "bb_near_threshold": 1.02,
            "decline_threshold": -0.05,
            "price_decline_threshold": -0.02,
            "recent_decline": 0.3,
            "ma_all_below": 0.4,
            "rsi_oversold": 0.4,
            "volume_weight": 0.3
        },
        "genetic_algorithm": {
            "enabled": True,
            "population_size": 100,
            "generations": 20,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_ratio": 0.1
        },
        "bayesian_optimization": {
            "enabled": True,
            "n_calls": 100,
            "n_initial_points": 20,
            "acq_func": "EI",
            "kappa": 2.0,
            "xi": 0.01,
            "random_state": 42,
            "n_jobs": 1
        }
    }
    
    # 加载当前配置
    if strategy_path.exists():
        with open(strategy_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f) or {}
    else:
        current_config = {}
    
    # 更新核心参数
    for section, params in core_defaults.items():
        current_config[section] = params
        print(f"✅ 重置 {section} 参数")
    
    # 保存配置
    with open(strategy_path, 'w', encoding='utf-8') as f:
        yaml.dump(current_config, f, default_flow_style=False, 
                 allow_unicode=True, sort_keys=False, indent=2)
    
    # 删除优化参数文件
    optimized_params_path = config_dir / "optimized_params.yaml"
    if optimized_params_path.exists():
        optimized_params_path.unlink()
        print("✅ 删除 optimized_params.yaml")
    
    print("\n🎉 快速重置完成!")
    print(f"📁 备份位置: {backup_dir}")
    print("\n重置的参数:")
    print("  - 策略核心参数 (rise_threshold, max_days 等)")
    print("  - 置信度权重 (final_threshold, rsi_oversold_threshold 等)")
    print("  - 优化算法参数 (遗传算法、贝叶斯优化)")
    print("\n💡 建议运行测试确认配置正确")


if __name__ == "__main__":
    print("🔄 快速重置策略参数...")
    print("⚠️  此操作将重置核心参数到默认值")
    
    confirm = input("是否继续? (y/N): ").strip().lower()
    if confirm in ['y', 'yes', '是']:
        quick_reset()
    else:
        print("❌ 操作已取消")