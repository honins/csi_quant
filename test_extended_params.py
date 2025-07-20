#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的优化参数
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved

def test_extended_params():
    """测试新增的优化参数"""
    print("🧪 测试新增的优化参数")
    print("=" * 50)
    
    try:
        # 加载配置
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # 初始化策略模块
        strategy_module = StrategyModule(config)
        
        # 获取当前参数
        current_params = strategy_module.get_params()
        
        print("📋 当前策略参数（包含新增参数）:")
        print("-" * 30)
        
        # 基础参数
        print("🔧 基础参数:")
        print(f"   rise_threshold: {current_params.get('rise_threshold', 'N/A')}")
        print(f"   max_days: {current_params.get('max_days', 'N/A')}")
        print()
        
        # 原有AI优化参数
        print("🤖 原有AI优化参数:")
        print(f"   final_threshold: {current_params.get('final_threshold', 'N/A')}")
        print(f"   rsi_low_threshold: {current_params.get('rsi_low_threshold', 'N/A')}")
        print(f"   rsi_oversold_threshold: {current_params.get('rsi_oversold_threshold', 'N/A')}")
        print(f"   dynamic_confidence_adjustment: {current_params.get('dynamic_confidence_adjustment', 'N/A')}")
        print(f"   market_sentiment_weight: {current_params.get('market_sentiment_weight', 'N/A')}")
        print(f"   trend_strength_weight: {current_params.get('trend_strength_weight', 'N/A')}")
        print()
        
        # 新增高重要度参数
        print("🚨 新增高重要度参数:")
        print(f"   volume_panic_threshold: {current_params.get('volume_panic_threshold', 'N/A')}")
        print(f"   volume_surge_threshold: {current_params.get('volume_surge_threshold', 'N/A')}")
        print(f"   volume_shrink_threshold: {current_params.get('volume_shrink_threshold', 'N/A')}")
        print(f"   bb_near_threshold: {current_params.get('bb_near_threshold', 'N/A')}")
        print(f"   rsi_uptrend_min: {current_params.get('rsi_uptrend_min', 'N/A')}")
        print(f"   rsi_uptrend_max: {current_params.get('rsi_uptrend_max', 'N/A')}")
        print()
        
        # 新增中重要度参数
        print("📊 新增中重要度参数:")
        print(f"   volume_panic_bonus: {current_params.get('volume_panic_bonus', 'N/A')}")
        print(f"   volume_surge_bonus: {current_params.get('volume_surge_bonus', 'N/A')}")
        print(f"   volume_shrink_penalty: {current_params.get('volume_shrink_penalty', 'N/A')}")
        print(f"   bb_lower_near: {current_params.get('bb_lower_near', 'N/A')}")
        print(f"   price_decline_threshold: {current_params.get('price_decline_threshold', 'N/A')}")
        print(f"   decline_threshold: {current_params.get('decline_threshold', 'N/A')}")
        print()
        
        # 检查AI优化器的参数范围
        ai_optimizer = AIOptimizerImproved(config)
        param_ranges = ai_optimizer._get_enhanced_parameter_ranges({})
        
        print("🎯 AI优化器参数范围:")
        print("-" * 30)
        print(f"优化参数总数: {len(param_ranges)}")
        
        # 分类显示参数
        original_params = ['final_threshold', 'rsi_low_threshold', 'rsi_oversold_threshold', 
                          'dynamic_confidence_adjustment', 'market_sentiment_weight', 
                          'price_momentum_weight', 'trend_strength_weight', 'volume_weight']
        
        high_importance_params = ['volume_panic_threshold', 'volume_surge_threshold', 
                                'volume_shrink_threshold', 'bb_near_threshold', 
                                'rsi_uptrend_min', 'rsi_uptrend_max']
        
        medium_importance_params = ['volume_panic_bonus', 'volume_surge_bonus', 
                                  'volume_shrink_penalty', 'bb_lower_near', 
                                  'price_decline_threshold', 'decline_threshold']
        
        print("🔧 原有参数 (8个):")
        for param in original_params:
            if param in param_ranges:
                range_config = param_ranges[param]
                print(f"   {param}: {range_config['min']} - {range_config['max']} ({range_config['type']})")
        
        print("\n🚨 新增高重要度参数 (6个):")
        for param in high_importance_params:
            if param in param_ranges:
                range_config = param_ranges[param]
                print(f"   {param}: {range_config['min']} - {range_config['max']} ({range_config['type']})")
        
        print("\n📊 新增中重要度参数 (6个):")
        for param in medium_importance_params:
            if param in param_ranges:
                range_config = param_ranges[param]
                print(f"   {param}: {range_config['min']} - {range_config['max']} ({range_config['type']})")
        
        # 验证参数数量
        expected_total = len(original_params) + len(high_importance_params) + len(medium_importance_params)
        actual_total = len(param_ranges)
        
        print(f"\n📊 参数数量验证:")
        print(f"   预期总数: {expected_total}")
        print(f"   实际总数: {actual_total}")
        print(f"   验证结果: {'✅ 通过' if actual_total == expected_total else '❌ 失败'}")
        
        return actual_total == expected_total
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genetic_algorithm_config():
    """测试遗传算法配置"""
    print("\n🧬 测试遗传算法配置:")
    print("-" * 30)
    
    try:
        with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ga_config = config.get('genetic_algorithm', {})
        
        print(f"种群大小: {ga_config.get('population_size', 'N/A')}")
        print(f"进化代数: {ga_config.get('generations', 'N/A')}")
        print(f"交叉率: {ga_config.get('crossover_rate', 'N/A')}")
        print(f"变异率: {ga_config.get('mutation_rate', 'N/A')}")
        print(f"精英保留比例: {ga_config.get('elite_ratio', 'N/A')}")
        
        # 验证配置是否已更新
        population_size = ga_config.get('population_size', 0)
        generations = ga_config.get('generations', 0)
        
        if population_size >= 120 and generations >= 30:
            print("✅ 遗传算法配置已正确更新")
            return True
        else:
            print("❌ 遗传算法配置需要更新")
            return False
            
    except Exception as e:
        print(f"❌ 配置测试异常: {e}")
        return False

def main():
    """主测试函数"""
    try:
        # 测试新增参数
        success1 = test_extended_params()
        
        # 测试遗传算法配置
        success2 = test_genetic_algorithm_config()
        
        # 总结
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"新增参数测试: {'✅ 成功' if success1 else '❌ 失败'}")
        print(f"遗传算法配置测试: {'✅ 成功' if success2 else '❌ 失败'}")
        
        if success1 and success2:
            print("\n🎉 参数扩展成功！")
            print("💡 现在AI优化器将优化17个参数（8个原有 + 9个新增）")
            print("🔧 遗传算法配置已调整为：种群120，代数30")
        else:
            print("\n💡 参数扩展需要进一步调整")
        
    except Exception as e:
        print(f"❌ 主函数异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 