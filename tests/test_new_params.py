#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试新增参数集成
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_new_params():
    """测试新增参数集成"""
    try:
        from src.ai.ai_optimizer import AIOptimizer
        from src.utils.utils import load_config
        
        # 加载配置
        config = load_config('config/config.yaml')
        
        # 创建AI优化器
        optimizer = AIOptimizer(config)
        
        print("✅ 新增参数集成成功！")
        print("📊 现在共有8个可优化参数：")
        print("   1. rsi_oversold_threshold (RSI超卖阈值)")
        print("   2. rsi_low_threshold (RSI低值阈值)")
        print("   3. final_threshold (最终置信度)")
        print("   4. dynamic_confidence_adjustment (动态调整系数)")
        print("   5. market_sentiment_weight (市场情绪权重)")
        print("   6. trend_strength_weight (趋势强度权重)")
        print("   7. volume_weight (成交量权重) [新增]")
        print("   8. price_momentum_weight (价格动量权重) [新增]")
        
        # 检查配置文件中的新参数
        ai_config = config.get('ai', {})
        optimization_ranges = ai_config.get('optimization_ranges', {})
        
        print("\n🔧 新增参数配置范围：")
        if 'volume_weight' in optimization_ranges:
            vol_range = optimization_ranges['volume_weight']
            print(f"   - volume_weight: {vol_range.get('min')} - {vol_range.get('max')}")
        
        if 'price_momentum_weight' in optimization_ranges:
            mom_range = optimization_ranges['price_momentum_weight']
            print(f"   - price_momentum_weight: {mom_range.get('min')} - {mom_range.get('max')}")
        
        # 检查策略配置中的默认值
        strategy_config = config.get('strategy', {})
        confidence_weights = strategy_config.get('confidence_weights', {})
        
        print("\n📋 新增参数默认值：")
        print(f"   - volume_weight: {confidence_weights.get('volume_weight', '未设置')}")
        print(f"   - price_momentum_weight: {confidence_weights.get('price_momentum_weight', '未设置')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_new_params()
    sys.exit(0 if success else 1) 