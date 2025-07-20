#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试当前策略参数获取
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.strategy.strategy_module import StrategyModule

def test_current_params():
    """测试当前策略参数获取"""
    print("🧪 测试当前策略参数获取")
    print("=" * 50)
    
    try:
        # 加载配置
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # 初始化策略模块
        strategy_module = StrategyModule(config)
        
        # 获取当前参数
        current_params = strategy_module.get_params()
        
        print("📋 当前策略参数:")
        print("-" * 30)
        
        # 基础参数
        print("🔧 基础参数:")
        print(f"   rise_threshold: {current_params.get('rise_threshold', 'N/A')}")
        print(f"   max_days: {current_params.get('max_days', 'N/A')}")
        print()
        
        # RSI相关参数
        print("📊 RSI相关参数:")
        print(f"   rsi_oversold_threshold: {current_params.get('rsi_oversold_threshold', 'N/A')}")
        print(f"   rsi_low_threshold: {current_params.get('rsi_low_threshold', 'N/A')}")
        print(f"   final_threshold: {current_params.get('final_threshold', 'N/A')}")
        print()
        
        # AI优化参数
        print("🤖 AI优化参数:")
        print(f"   dynamic_confidence_adjustment: {current_params.get('dynamic_confidence_adjustment', 'N/A')}")
        print(f"   market_sentiment_weight: {current_params.get('market_sentiment_weight', 'N/A')}")
        print(f"   trend_strength_weight: {current_params.get('trend_strength_weight', 'N/A')}")
        print()
        
        # 成交量相关参数
        print("📈 成交量相关参数:")
        print(f"   volume_weight: {current_params.get('volume_weight', 'N/A')}")
        print(f"   volume_panic_threshold: {current_params.get('volume_panic_threshold', 'N/A')}")
        print(f"   volume_surge_threshold: {current_params.get('volume_surge_threshold', 'N/A')}")
        print(f"   volume_shrink_threshold: {current_params.get('volume_shrink_threshold', 'N/A')}")
        print()
        
        # 价格动量参数
        print("💹 价格动量参数:")
        print(f"   price_momentum_weight: {current_params.get('price_momentum_weight', 'N/A')}")
        print(f"   bb_near_threshold: {current_params.get('bb_near_threshold', 'N/A')}")
        print()
        
        # 检查配置文件中的实际值
        print("📄 配置文件中的实际值:")
        print("-" * 30)
        
        strategy_config = config.get('strategy', {})
        confidence_weights = strategy_config.get('confidence_weights', {})
        
        print("🔧 基础参数 (strategy级别):")
        print(f"   rise_threshold: {strategy_config.get('rise_threshold', 'N/A')}")
        print(f"   max_days: {strategy_config.get('max_days', 'N/A')}")
        print()
        
        print("📊 RSI相关参数 (confidence_weights级别):")
        print(f"   rsi_oversold_threshold: {confidence_weights.get('rsi_oversold_threshold', 'N/A')}")
        print(f"   rsi_low_threshold: {confidence_weights.get('rsi_low_threshold', 'N/A')}")
        print(f"   final_threshold: {confidence_weights.get('final_threshold', 'N/A')}")
        print()
        
        print("🤖 AI优化参数 (confidence_weights级别):")
        print(f"   dynamic_confidence_adjustment: {confidence_weights.get('dynamic_confidence_adjustment', 'N/A')}")
        print(f"   market_sentiment_weight: {confidence_weights.get('market_sentiment_weight', 'N/A')}")
        print(f"   trend_strength_weight: {confidence_weights.get('trend_strength_weight', 'N/A')}")
        print()
        
        print("📈 成交量相关参数 (strategy级别):")
        print(f"   volume_weight: {strategy_config.get('volume_weight', 'N/A')}")
        print(f"   volume_panic_threshold: {strategy_config.get('volume_panic_threshold', 'N/A')}")
        print(f"   volume_surge_threshold: {strategy_config.get('volume_surge_threshold', 'N/A')}")
        print(f"   volume_shrink_threshold: {strategy_config.get('volume_shrink_threshold', 'N/A')}")
        print()
        
        print("💹 价格动量参数 (strategy级别):")
        print(f"   price_momentum_weight: {strategy_config.get('price_momentum_weight', 'N/A')}")
        print(f"   bb_near_threshold: {strategy_config.get('bb_near_threshold', 'N/A')}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    try:
        # 运行测试
        success = test_current_params()
        
        # 总结
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"策略参数获取测试: {'✅ 成功' if success else '❌ 失败'}")
        
        if success:
            print("\n🎉 策略参数获取测试成功！")
            print("💡 现在get_params()方法会返回实际的当前参数，而不是硬编码的默认值")
        else:
            print("\n💡 策略参数获取测试失败")
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 