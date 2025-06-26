#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参数重置脚本
用于将策略参数重置到初始默认值，避免数据泄露风险
"""

import os
import sys
import yaml
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def reset_parameters_to_default():
    """
    将策略参数重置到初始默认值
    这是为了避免之前没有严格数据分割时产生的数据泄露
    """
    
    # 默认参数值 (基于parameter_importance_analysis.md)
    default_params = {
        # 策略核心参数（保持不变）
        'rise_threshold': 0.04,
        'max_days': 20,
        
        # 置信度权重参数
        'final_threshold': 0.5,              # 最终置信度阈值
        'rsi_oversold_threshold': 30,        # RSI超卖阈值  
        'rsi_low_threshold': 40,             # RSI低值阈值
        
        # AI优化参数
        'dynamic_confidence_adjustment': 0.05,   # 动态置信度调整系数
        'market_sentiment_weight': 0.16,         # 市场情绪权重
        'trend_strength_weight': 0.16,           # 趋势强度权重
        'volume_weight': 0.25,                   # 成交量权重
        'price_momentum_weight': 0.20,           # 价格动量权重
    }
    
    # 配置文件路径
    config_path = project_root / 'config' / 'config.yaml'
    
    print("🔄 参数重置脚本")
    print("=" * 60)
    print("⚠️  数据泄露风险检测：")
    print("   之前的参数优化没有使用严格的数据分割")
    print("   当前参数可能已经'看过'了测试集数据")
    print("   为确保公正的性能评估，需要重置参数到初始值")
    print()
    
    try:
        # 读取当前配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("📊 当前参数 vs 默认参数对比：")
        print("-" * 60)
        
        # 显示参数对比
        strategy_config = config.get('strategy', {})
        confidence_weights = strategy_config.get('confidence_weights', {})
        
        changes_made = []
        
        # 检查并更新参数
        params_to_check = [
            ('final_threshold', confidence_weights),
            ('rsi_oversold_threshold', confidence_weights), 
            ('rsi_low_threshold', confidence_weights),
            ('dynamic_confidence_adjustment', confidence_weights),
            ('market_sentiment_weight', confidence_weights),
            ('trend_strength_weight', confidence_weights),
            ('volume_weight', confidence_weights),
            ('price_momentum_weight', confidence_weights),
        ]
        
        for param_name, config_section in params_to_check:
            current_value = config_section.get(param_name, 'N/A')
            default_value = default_params[param_name]
            
            if current_value != default_value:
                print(f"📝 {param_name}:")
                print(f"   当前值: {current_value}")
                print(f"   默认值: {default_value} ← 将重置")
                changes_made.append(param_name)
                
                # 更新配置
                config_section[param_name] = default_value
            else:
                print(f"✅ {param_name}: {current_value} (已是默认值)")
        
        if changes_made:
            print()
            print(f"🔧 需要重置 {len(changes_made)} 个参数:")
            for param in changes_made:
                print(f"   - {param}")
            
            # 确认重置
            response = input("\\n是否确认重置参数到默认值? (y/N): ").strip().lower()
            
            if response in ['y', 'yes', '是']:
                # 保存配置
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                
                print()
                print("✅ 参数重置完成！")
                print("📋 重置的参数:")
                for param in changes_made:
                    print(f"   ✓ {param} = {default_params[param]}")
                
                print()
                print("🎯 下一步操作建议:")
                print("   1. 运行 'python run.py ai' 进行重新优化")
                print("   2. 新的优化将使用严格的65/20/15数据分割")
                print("   3. 测试集将完全隔离，确保结果可靠性")
                
            else:
                print("❌ 参数重置已取消")
                
        else:
            print()
            print("✅ 所有参数都已是默认值，无需重置")
            
    except Exception as e:
        print(f"❌ 参数重置失败: {str(e)}")
        return False
    
    return True

def backup_current_config():
    """备份当前配置文件"""
    from datetime import datetime
    
    config_path = project_root / 'config' / 'config.yaml'
    backup_dir = project_root / 'config' / 'backups'
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'config_backup_{timestamp}.yaml'
    
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"📋 配置文件已备份到: {backup_path}")
        return True
    except Exception as e:
        print(f"⚠️ 配置备份失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔒 严格数据分割 - 参数重置工具")
    print("=" * 60)
    
    # 备份当前配置
    backup_current_config()
    print()
    
    # 重置参数
    reset_parameters_to_default() 