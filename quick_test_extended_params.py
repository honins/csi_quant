#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试参数扩展是否成功
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """快速测试参数扩展"""
    print("⚡ 快速测试参数扩展")
    print("=" * 40)
    
    try:
        # 1. 检查配置文件
        print("📋 1. 检查配置文件...")
        with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        optimization_ranges = config.get('optimization_ranges', {})
        genetic_config = config.get('genetic_algorithm', {})
        
        # 检查新增参数
        new_params = [
            'volume_panic_threshold', 'volume_surge_threshold', 'volume_shrink_threshold',
            'bb_near_threshold', 'rsi_uptrend_min', 'rsi_uptrend_max',
            'volume_panic_bonus', 'volume_surge_bonus', 'volume_shrink_penalty',
            'bb_lower_near', 'price_decline_threshold', 'decline_threshold'
        ]
        
        found_params = []
        for param in new_params:
            if param in optimization_ranges:
                found_params.append(param)
                range_config = optimization_ranges[param]
                print(f"   ✅ {param}: {range_config.get('min', 'N/A')} - {range_config.get('max', 'N/A')}")
            else:
                print(f"   ❌ {param}: 未找到")
        
        print(f"\n   新增参数检查: {len(found_params)}/{len(new_params)} 个参数已添加")
        
        # 检查遗传算法配置
        print(f"\n🧬 2. 检查遗传算法配置...")
        population_size = genetic_config.get('population_size', 0)
        generations = genetic_config.get('generations', 0)
        
        print(f"   种群大小: {population_size}")
        print(f"   进化代数: {generations}")
        
        ga_ok = population_size >= 120 and generations >= 6
        print(f"   遗传算法配置: {'✅ 正确' if ga_ok else '❌ 需要调整'}")
        
        # 2. 检查策略模块
        print(f"\n🔧 3. 检查策略模块...")
        try:
            from src.strategy.strategy_module import StrategyModule
            from src.utils.config_loader import ConfigLoader
            
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            strategy_module = StrategyModule(config)
            
            current_params = strategy_module.get_params()
            
            # 检查新增参数是否在get_params中
            params_found = []
            for param in new_params:
                if param in current_params:
                    params_found.append(param)
                    print(f"   ✅ {param}: {current_params[param]}")
                else:
                    print(f"   ❌ {param}: 未在get_params中找到")
            
            print(f"\n   策略模块参数检查: {len(params_found)}/{len(new_params)} 个参数已支持")
            
        except Exception as e:
            print(f"   ❌ 策略模块检查失败: {e}")
            return False
        
        # 3. 检查AI优化器
        print(f"\n🤖 4. 检查AI优化器...")
        try:
            from src.ai.ai_optimizer_improved import AIOptimizerImproved
            
            ai_optimizer = AIOptimizerImproved(config)
            param_ranges = ai_optimizer._get_enhanced_parameter_ranges({})
            
            # 检查参数范围
            ranges_found = []
            for param in new_params:
                if param in param_ranges:
                    ranges_found.append(param)
                    range_config = param_ranges[param]
                    print(f"   ✅ {param}: {range_config['min']} - {range_config['max']}")
                else:
                    print(f"   ❌ {param}: 未在优化器参数范围中找到")
            
            print(f"\n   AI优化器参数范围检查: {len(ranges_found)}/{len(new_params)} 个参数已支持")
            
            total_params = len(param_ranges)
            print(f"   总优化参数数量: {total_params}")
            
        except Exception as e:
            print(f"   ❌ AI优化器检查失败: {e}")
            return False
        
        # 总结
        print(f"\n📊 测试总结:")
        print("=" * 40)
        
        config_ok = len(found_params) == len(new_params)
        strategy_ok = len(params_found) == len(new_params)
        optimizer_ok = len(ranges_found) == len(new_params)
        
        print(f"配置文件检查: {'✅ 通过' if config_ok else '❌ 失败'}")
        print(f"策略模块检查: {'✅ 通过' if strategy_ok else '❌ 失败'}")
        print(f"AI优化器检查: {'✅ 通过' if optimizer_ok else '❌ 失败'}")
        print(f"遗传算法配置: {'✅ 通过' if ga_ok else '❌ 失败'}")
        
        overall_success = config_ok and strategy_ok and optimizer_ok and ga_ok
        
        if overall_success:
            print(f"\n🎉 参数扩展测试成功！")
            print(f"💡 现在AI优化器将优化 {total_params} 个参数")
            print(f"🔧 遗传算法配置: 种群{population_size}, 代数{generations}")
        else:
            print(f"\n💡 参数扩展需要进一步调整")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ 快速测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1) 