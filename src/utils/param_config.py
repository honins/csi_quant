#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参数配置文件
定义所有参数的分类和保存位置，避免硬编码
"""

# ============================================================================
# 参数分类定义
# ============================================================================

# 🔧 固定参数（不参与优化）
FIXED_PARAMS = [
    'rise_threshold',      # 涨幅阈值
    'max_days',           # 最大天数
    'final_threshold'     # 最终置信度阈值 - 从优化中分离，应单独优化
]

# 🎯 最终选择的14个有效优化参数（已移除final_threshold）
# 根据用户确定的优化方案，只选择真正有效的参数

# 🔥 核心决策参数（2个）- 每次预测都使用
CORE_DECISION_PARAMS = [
    'rsi_oversold_threshold',            # RSI超卖阈值 - 有效性：★★★★★
    'rsi_low_threshold',                 # RSI低阈值 - 有效性：★★★★☆
    # 注意：final_threshold 已移至固定参数，不再参与优化
]

# 🔥 基础权重参数（4个）- 高频使用，重要逻辑
BASIC_WEIGHT_PARAMS = [
    'ma_all_below',                      # 价格跌破所有均线权重 - 有效性：★★★★★
    'dynamic_confidence_adjustment',      # 动态置信度调整 - 有效性：★★★★☆
    'market_sentiment_weight',           # 市场情绪权重 - 有效性：★★★★☆
    'trend_strength_weight'              # 趋势强度权重 - 有效性：★★★★☆
]

# 🔥 成交量逻辑参数（4个）- 代码中大量使用的核心逻辑
VOLUME_LOGIC_PARAMS = [
    'volume_panic_threshold',            # 成交量恐慌阈值 - 有效性：★★★★☆
    'volume_panic_bonus',                # 恐慌性抛售奖励 - 有效性：★★★★☆
    'volume_surge_bonus',                # 温和放量奖励 - 有效性：★★★☆☆
    'volume_shrink_penalty'              # 成交量萎缩惩罚 - 有效性：★★★☆☆
]

# 🔥 技术指标参数（4个）- 基础但重要的技术指标
TECHNICAL_INDICATOR_PARAMS = [
    'bb_near_threshold',                 # 布林带接近阈值 - 有效性：★★★☆☆
    'recent_decline',                    # 近期下跌权重 - 有效性：★★★☆☆
    'macd_negative',                     # MACD负值权重 - 有效性：★★★☆☆
    'price_decline_threshold'            # 价格下跌阈值 - 有效性：★★★☆☆
]

# 🎯 所有可优化参数（15个）
OPTIMIZABLE_PARAMS = (
    CORE_DECISION_PARAMS + 
    BASIC_WEIGHT_PARAMS + 
    VOLUME_LOGIC_PARAMS + 
    TECHNICAL_INDICATOR_PARAMS
)

# 📊 其他参数（不参与优化）
OTHER_PARAMS = [
    # 其他confidence_weights参数
    'bb_lower_near', 'decline_threshold', 'rsi_uptrend_min', 'rsi_uptrend_max',
    'rsi_pullback_threshold', 'rsi_uptrend_pullback', 'rsi_overbought_correction',
    # 其他strategy级别参数
    'volume_weight', 'price_momentum_weight', 'volume_surge_threshold', 'volume_shrink_threshold'
]

# 🎯 confidence_weights参数（参与优化，保存在confidence_weights部分）
CONFIDENCE_WEIGHT_PARAMS = [
    # 核心决策参数
    'final_threshold',                    # 最终置信度阈值
    'rsi_oversold_threshold',            # RSI超卖阈值
    'rsi_low_threshold',                 # RSI偏低阈值
    
    # 动态调整参数
    'dynamic_confidence_adjustment',      # 动态置信度调整权重
    'market_sentiment_weight',           # 市场情绪权重
    'trend_strength_weight',             # 趋势强度权重
    
    # 成交量相关参数
    'volume_panic_bonus',                # 成交量恐慌奖励权重
    'volume_surge_bonus',                # 成交量激增奖励权重
    'volume_shrink_penalty',             # 成交量萎缩惩罚权重
    
    # 技术指标参数
    'bb_lower_near',                     # 布林带下轨接近权重
    'price_decline_threshold',           # 价格下跌阈值
    'decline_threshold',                 # 下跌阈值
    
    # RSI相关参数
    'rsi_uptrend_min',                   # RSI上升趋势最小值
    'rsi_uptrend_max',                   # RSI上升趋势最大值
    'rsi_pullback_threshold',            # RSI回调阈值
    'rsi_uptrend_pullback',              # RSI上升趋势回调权重
    'rsi_overbought_correction'          # RSI超买修正权重
]

# 📊 strategy级别参数（参与优化，保存在strategy部分）
STRATEGY_LEVEL_PARAMS = [
    'volume_weight',                      # 成交量权重
    'price_momentum_weight',             # 价格动量权重
    'bb_near_threshold',                 # 布林带接近阈值
    'volume_panic_threshold',            # 成交量恐慌阈值
    'volume_surge_threshold',            # 成交量激增阈值
    'volume_shrink_threshold'            # 成交量萎缩阈值
]

# ============================================================================
# 参数验证函数
# ============================================================================

def is_fixed_param(param_name: str) -> bool:
    """检查是否为固定参数"""
    return param_name in FIXED_PARAMS

def is_optimizable_param(param_name: str) -> bool:
    """检查是否为可优化参数（14个有效参数，已移除final_threshold）"""
    return param_name in OPTIMIZABLE_PARAMS

def is_confidence_weight_param(param_name: str) -> bool:
    """检查是否为confidence_weights参数"""
    return param_name in CONFIDENCE_WEIGHT_PARAMS

def is_strategy_level_param(param_name: str) -> bool:
    """检查是否为strategy级别参数"""
    return param_name in STRATEGY_LEVEL_PARAMS

def get_param_category(param_name: str) -> str:
    """获取参数分类"""
    if is_fixed_param(param_name):
        return 'fixed'
    elif is_optimizable_param(param_name):
        return 'optimizable'
    elif is_confidence_weight_param(param_name):
        return 'confidence_weights'
    elif is_strategy_level_param(param_name):
        return 'strategy_level'
    else:
        return 'other'

def get_all_optimizable_params() -> list:
    """获取所有可优化参数（14个有效参数）"""
    return list(OPTIMIZABLE_PARAMS)

def get_all_params() -> dict:
    """获取所有参数分类"""
    return {
        'fixed': FIXED_PARAMS,
        'optimizable': list(OPTIMIZABLE_PARAMS),
        'confidence_weights': CONFIDENCE_WEIGHT_PARAMS,
        'strategy_level': STRATEGY_LEVEL_PARAMS,
        'other': OTHER_PARAMS
    }

def get_param_effectiveness(param_name: str) -> str:
    """获取参数有效性评级"""
    effectiveness_map = {
        # 🔥 核心决策参数（2个）
        'rsi_oversold_threshold': '★★★★★',
        'rsi_low_threshold': '★★★★☆',
        
        # 🔥 基础权重参数（4个）
        'ma_all_below': '★★★★★',
        'dynamic_confidence_adjustment': '★★★★☆',
        'market_sentiment_weight': '★★★★☆',
        'trend_strength_weight': '★★★★☆',
        
        # 🔥 成交量逻辑参数（4个）
        'volume_panic_threshold': '★★★★☆',
        'volume_panic_bonus': '★★★★☆',
        'volume_surge_bonus': '★★★☆☆',
        'volume_shrink_penalty': '★★★☆☆',
        
        # 🔥 技术指标参数（4个）
        'bb_near_threshold': '★★★☆☆',
        'recent_decline': '★★★☆☆',
        'macd_negative': '★★★☆☆',
        'price_decline_threshold': '★★★☆☆',
        
        # 🔒 固定参数（3个）
        'rise_threshold': '★★★★★',
        'max_days': '★★★★★',
        'final_threshold': '★★★★★'
    }
    
    return effectiveness_map.get(param_name, '★★☆☆☆')

# ============================================================================
# 参数统计信息
# ============================================================================

def get_param_statistics() -> dict:
    """获取参数统计信息"""
    return {
        'total_fixed': len(FIXED_PARAMS),
        'total_optimizable': len(OPTIMIZABLE_PARAMS),
        'total_confidence_weights': len(CONFIDENCE_WEIGHT_PARAMS),
        'total_strategy_level': len(STRATEGY_LEVEL_PARAMS),
        'total_other': len(OTHER_PARAMS),
        'total_all': len(FIXED_PARAMS) + len(CONFIDENCE_WEIGHT_PARAMS) + len(STRATEGY_LEVEL_PARAMS) + len(OTHER_PARAMS)
    }

def print_param_summary():
    """打印参数摘要"""
    stats = get_param_statistics()
    print("=" * 80)
    print("📊 参数分类摘要（基于14个有效参数方案）")
    print("=" * 80)
    print(f"🔒 固定参数: {len(FIXED_PARAMS)} 个")
    print(f"   {', '.join(FIXED_PARAMS)}")
    print()
    print(f"🎯 可优化参数: {stats['total_optimizable']} 个（14个有效参数）")
    print("   🔥 核心决策参数（3个）:")
    for param in CORE_DECISION_PARAMS:
        effectiveness = get_param_effectiveness(param)
        print(f"      {param} - 有效性: {effectiveness}")
    print("   🔥 基础权重参数（4个）:")
    for param in BASIC_WEIGHT_PARAMS:
        effectiveness = get_param_effectiveness(param)
        print(f"      {param} - 有效性: {effectiveness}")
    print("   🔥 成交量逻辑参数（4个）:")
    for param in VOLUME_LOGIC_PARAMS:
        effectiveness = get_param_effectiveness(param)
        print(f"      {param} - 有效性: {effectiveness}")
    print("   🔥 技术指标参数（4个）:")
    for param in TECHNICAL_INDICATOR_PARAMS:
        effectiveness = get_param_effectiveness(param)
        print(f"      {param} - 有效性: {effectiveness}")
    print()
    print(f"📊 其他参数: {stats['total_other']} 个（不参与优化）")
    print(f"📈 所有参数总数: {stats['total_all']} 个")
    print("=" * 80)

if __name__ == "__main__":
    print_param_summary()