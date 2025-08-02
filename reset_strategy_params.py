#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略参数重置脚本

功能：
1. 重置策略参数到默认值
2. 备份当前配置
3. 支持选择性重置
4. 提供安全确认机制

使用方法：
    python reset_strategy_params.py [选项]
    
选项：
    --all           重置所有参数
    --strategy      仅重置策略参数
    --confidence    仅重置置信度权重
    --optimization  仅重置优化参数
    --backup        仅创建备份，不重置
    --force         跳过确认提示
    --help          显示帮助信息
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class StrategyParamsResetter:
    """策略参数重置器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.config_dir / "backups"
        
        # 确保备份目录存在
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认参数配置
        self.default_params = self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数配置"""
        return {
            "strategy.yaml": {
                "advanced_optimization": {
                    "enabled": True,
                    "high_precision_mode": True,
                    "use_hierarchical": True,
                    "use_scipy": True,
                    "use_time_series_cv": True
                },
                "ai_scoring": {
                    "rise_benchmark": 0.1,
                    "rise_weight": 0.25,
                    "risk_benchmark": 0.2,
                    "risk_weight": 0.1,
                    "speed_weight": 0.15,
                    "success_weight": 0.5
                },
                "bayesian_optimization": {
                    "acq_func": "EI",
                    "enabled": True,
                    "kappa": 2.0,
                    "n_calls": 120,
                    "n_initial_points": 25,
                    "n_jobs": 1,
                    "random_state": 42,
                    "xi": 0.008
                },
                "confidence_std_threshold": 0.05,
                "confidence_weights": {
                    "bb_lower_near": 0.3,
                    "bb_near_threshold": 1.02,
                    "consolidation_breakout": 0.2,
                    "decline_threshold": -0.05,
                    "dynamic_confidence_adjustment": 0.1,
                    "final_threshold": 0.5,
                    "ma_all_below": 0.4,
                    "ma_partial_below": 0.2,
                    "macd_negative": 0.15,
                    "macd_rsi_combo": 0.15,
                    "market_sentiment_weight": 0.15,
                    "moderate_rsi_bonus": 0.3,
                    "momentum_reversal_bonus": 0.25,
                    "price_decline_threshold": -0.02,
                    "price_momentum_weight": 0.2,
                    "recent_decline": 0.3,
                    "resistance_break_bonus": 0.3,
                    "rsi_low": 0.2,
                    "rsi_low_threshold": 40.0,
                    "rsi_overbought_correction": 0.3,
                    "rsi_oversold": 0.4,
                    "rsi_oversold_threshold": 30.0,
                    "rsi_pullback_threshold": 3,
                    "rsi_uptrend_max": 80,
                    "rsi_uptrend_min": 30,
                    "rsi_uptrend_pullback": 0.4,
                    "support_level_bonus": 0.25,
                    "trend_strength_weight": 0.15,
                    "uptrend_consolidation_bonus": 0.25,
                    "uptrend_ma_support": 0.3,
                    "uptrend_pullback_bonus": 0.3,
                    "uptrend_support_volume": 0.15,
                    "uptrend_volume_pullback": 0.2,
                    "volume_panic_bonus": 0.15,
                    "volume_panic_threshold": 1.5,
                    "volume_price_divergence": 0.2,
                    "volume_shrink_penalty": 0.7,
                    "volume_shrink_threshold": 0.8,
                    "volume_surge_bonus": 0.1,
                    "volume_surge_threshold": 1.2,
                    "volume_weight": 0.3
                },
                "early_stopping": {
                    "enabled": True,
                    "min_delta": 0.001,
                    "patience": 50
                },
                "execution": {
                    "data_decay_rate": 0.3,
                    "optimization_interval": 30,
                    "parallel_jobs": 1,
                    "save_intermediate": False,
                    "save_results": True,
                    "train_test_split_ratio": 0.8,
                    "use_multiprocessing": False
                },
                "genetic_algorithm": {
                    "crossover_rate": 0.8,
                    "elite_ratio": 0.1,
                    "enabled": True,
                    "generations": 20,
                    "mutation_rate": 0.1,
                    "population_size": 100
                },
                "optimization": {
                    "enable_history": True,
                    "enable_incremental": True,
                    "global_iterations": 500,
                    "incremental_contraction_factor": 0.5,
                    "incremental_iterations": 1000,
                    "max_history_records": 100
                },
                "overfitting_threshold": 0.9,
                "strategy_scoring": {
                    "days_benchmark": 10.0,
                    "days_weight": 0.2,
                    "rise_benchmark": 0.1,
                    "rise_weight": 0.3,
                    "success_weight": 0.5
                },
                "validation": {
                    "test_ratio": 0.15,
                    "train_ratio": 0.7,
                    "validation_ratio": 0.15,
                    "walk_forward": {
                        "enabled": True,
                        "step_size": 63,
                        "window_size": 252
                    }
                },
                "zero_confidence_threshold": 0.5,
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
                }
            },
            "system.yaml": {
                "ai": {
                    "enable": True,
                    "model_type": "machine_learning",
                    "model_save_path": "models/ai_model.pkl",
                    "models_dir": "models",
                    "retrain_interval_days": 30,
                    "enable_model_reuse": True,
                    "training_data": {
                        "full_train_years": 5,
                        "optimize_years": 5,
                        "incremental_years": 1
                    }
                },
                "data": {
                    "data_file_path": "data/SHSE.000905_1d.csv",
                    "data_source": "akshare",
                    "index_code": "SHSE.000905",
                    "frequency": "1d",
                    "history_days": 1000,
                    "cache_enabled": True,
                    "cache_dir": "cache"
                },
                "strategy": {
                    "rise_threshold": 0.04,
                    "max_days": 20,
                    "results_dir": "results",
                    "bb_period": 20,
                    "bb_std": 2,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_signal": 9,
                    "macd_slow": 26,
                    "ma_periods": [5, 10, 20, 60]
                },
                "backtest": {
                    "start_date": "2022-01-01",
                    "end_date": "2025-06-21",
                    "default_start_date": "2023-01-01",
                    "default_end_date": "2025-06-21",
                    "rolling_window": 252,
                    "rolling_step": 63,
                    "generate_charts": True,
                    "charts_dir": "charts"
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/system.log",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "max_bytes": 10485760,
                    "backup_count": 5
                },
                "notification": {
                    "methods": ["console", "email"],
                    "logs_dir": "logs",
                    "email": {
                        "sender": "quant_system@example.com",
                        "recipients": ["your_email@example.com"],
                        "smtp_server": "smtp.example.com",
                        "smtp_port": 587,
                        "username": "your_username",
                        "password": "your_password"
                    }
                },
                "results": {
                    "save_path": "results",
                    "save_detailed": True,
                    "images_path": "results/images",
                    "tables_path": "results/tables"
                },
                "risk": {
                    "min_confidence": 0.6,
                    "max_daily_signals": 3,
                    "cooldown_days": 5
                },
                "system": {
                    "mode": "backtest",
                    "log_level": "INFO",
                    "log_file": "logs/system.log"
                }
            }
        }
    
    def create_backup(self) -> str:
        """创建当前配置的备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = self.backup_dir / f"reset_backup_{timestamp}"
        backup_folder.mkdir(exist_ok=True)
        
        # 备份配置文件
        config_files = ["strategy.yaml", "system.yaml", "optimized_params.yaml"]
        backed_up_files = []
        
        for config_file in config_files:
            source_path = self.config_dir / config_file
            if source_path.exists():
                backup_path = backup_folder / f"{config_file.replace('.yaml', '_backup.yaml')}"
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                backed_up_files.append(config_file)
        
        print(f"✅ 备份完成: {backup_folder}")
        print(f"📁 已备份文件: {', '.join(backed_up_files)}")
        return str(backup_folder)
    
    def reset_config_file(self, config_file: str, sections: list = None):
        """重置指定配置文件"""
        file_path = self.config_dir / config_file
        
        if not file_path.exists():
            print(f"⚠️  配置文件不存在: {config_file}")
            return
        
        # 加载当前配置
        with open(file_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f) or {}
        
        # 获取默认配置
        default_config = self.default_params.get(config_file, {})
        
        if sections:
            # 只重置指定部分
            for section in sections:
                if section in default_config:
                    current_config[section] = default_config[section]
                    print(f"✅ 重置 {config_file} 中的 {section} 部分")
        else:
            # 重置整个文件
            current_config = default_config
            print(f"✅ 完全重置 {config_file}")
        
        # 保存配置
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False, indent=2)
    
    def reset_strategy_params(self):
        """重置策略参数"""
        self.reset_config_file("strategy.yaml", ["strategy"])
    
    def reset_confidence_weights(self):
        """重置置信度权重"""
        self.reset_config_file("strategy.yaml", ["confidence_weights"])
    
    def reset_optimization_params(self):
        """重置优化参数"""
        sections = ["bayesian_optimization", "genetic_algorithm", "optimization", 
                   "advanced_optimization", "early_stopping"]
        self.reset_config_file("strategy.yaml", sections)
    
    def reset_all_params(self):
        """重置所有参数"""
        self.reset_config_file("strategy.yaml")
        self.reset_config_file("system.yaml")
        
        # 删除优化参数文件（如果存在）
        optimized_params_path = self.config_dir / "optimized_params.yaml"
        if optimized_params_path.exists():
            optimized_params_path.unlink()
            print("✅ 删除 optimized_params.yaml")
    
    def show_current_config(self, config_file: str):
        """显示当前配置"""
        file_path = self.config_dir / config_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"\n📄 当前 {config_file} 配置:")
            print("=" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="策略参数重置脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python reset_strategy_params.py --all              # 重置所有参数
  python reset_strategy_params.py --strategy         # 仅重置策略参数
  python reset_strategy_params.py --confidence       # 仅重置置信度权重
  python reset_strategy_params.py --optimization     # 仅重置优化参数
  python reset_strategy_params.py --backup           # 仅创建备份
        """
    )
    
    parser.add_argument("--all", action="store_true", help="重置所有参数")
    parser.add_argument("--strategy", action="store_true", help="仅重置策略参数")
    parser.add_argument("--confidence", action="store_true", help="仅重置置信度权重")
    parser.add_argument("--optimization", action="store_true", help="仅重置优化参数")
    parser.add_argument("--backup", action="store_true", help="仅创建备份，不重置")
    parser.add_argument("--force", action="store_true", help="跳过确认提示")
    parser.add_argument("--show", choices=["strategy", "system"], help="显示当前配置")
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，显示帮助
    if not any([args.all, args.strategy, args.confidence, args.optimization, 
                args.backup, args.show]):
        parser.print_help()
        return
    
    resetter = StrategyParamsResetter()
    
    # 显示配置
    if args.show:
        resetter.show_current_config(f"{args.show}.yaml")
        return
    
    # 仅备份
    if args.backup:
        resetter.create_backup()
        return
    
    # 确认操作
    if not args.force:
        print("⚠️  警告: 此操作将重置策略参数到默认值!")
        print("📁 建议先备份当前配置")
        
        if args.all:
            print("🔄 将重置: 所有参数")
        elif args.strategy:
            print("🔄 将重置: 策略参数")
        elif args.confidence:
            print("🔄 将重置: 置信度权重")
        elif args.optimization:
            print("🔄 将重置: 优化参数")
        
        confirm = input("\n是否继续? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes', '是']:
            print("❌ 操作已取消")
            return
    
    # 创建备份
    print("\n📦 创建备份...")
    backup_path = resetter.create_backup()
    
    # 执行重置
    print("\n🔄 开始重置参数...")
    
    try:
        if args.all:
            resetter.reset_all_params()
        elif args.strategy:
            resetter.reset_strategy_params()
        elif args.confidence:
            resetter.reset_confidence_weights()
        elif args.optimization:
            resetter.reset_optimization_params()
        
        print("\n✅ 参数重置完成!")
        print(f"📁 备份位置: {backup_path}")
        print("\n💡 提示:")
        print("  - 重置后建议运行一次测试确认配置正确")
        print("  - 如需恢复，可从备份文件夹复制配置")
        
    except Exception as e:
        print(f"\n❌ 重置失败: {e}")
        print(f"📁 可从备份恢复: {backup_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()