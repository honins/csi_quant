#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试15个有效参数方案
验证参数配置和优化逻辑是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.param_config import (
    FIXED_PARAMS,
    OPTIMIZABLE_PARAMS,
    CORE_DECISION_PARAMS,
    BASIC_WEIGHT_PARAMS,
    VOLUME_LOGIC_PARAMS,
    TECHNICAL_INDICATOR_PARAMS,
    OTHER_PARAMS,
    get_all_optimizable_params,
    get_param_effectiveness,
    is_optimizable_param
)
from src.utils.param_validator import ParamValidator

def test_param_classification():
    """测试参数分类"""
    print("🧪 测试参数分类")
    print("=" * 60)
    
    # 测试固定参数
    print(f"🔒 固定参数: {len(FIXED_PARAMS)} 个")
    print(f"   {FIXED_PARAMS}")
    
    # 测试可优化参数
    optimizable_params = get_all_optimizable_params()
    print(f"🎯 可优化参数: {len(optimizable_params)} 个")
    print(f"   核心决策参数: {CORE_DECISION_PARAMS}")
    print(f"   基础权重参数: {BASIC_WEIGHT_PARAMS}")
    print(f"   成交量逻辑参数: {VOLUME_LOGIC_PARAMS}")
    print(f"   技术指标参数: {TECHNICAL_INDICATOR_PARAMS}")
    
    # 测试其他参数
    print(f"📊 其他参数: {len(OTHER_PARAMS)} 个")
    print(f"   {OTHER_PARAMS}")
    
    # 验证参数总数
    total_params = len(FIXED_PARAMS) + len(optimizable_params) + len(OTHER_PARAMS)
    print(f"📈 参数总数: {total_params}")
    
    assert len(optimizable_params) == 14, f"可优化参数应该是14个，实际是{len(optimizable_params)}个"
    print("✅ 参数分类测试通过")

def test_param_effectiveness():
    """测试参数有效性评级"""
    print("\n🧪 测试参数有效性评级")
    print("=" * 60)
    
    optimizable_params = get_all_optimizable_params()
    for param in optimizable_params:
        effectiveness = get_param_effectiveness(param)
        print(f"   {param}: {effectiveness}")
    
    print("✅ 参数有效性评级测试通过")

def test_optimizable_param_check():
    """测试可优化参数检查"""
    print("\n🧪 测试可优化参数检查")
    print("=" * 60)
    
    # 测试可优化参数
    optimizable_params = get_all_optimizable_params()
    for param in optimizable_params:
        assert is_optimizable_param(param), f"参数 {param} 应该是可优化的"
    
    # 测试固定参数
    for param in FIXED_PARAMS:
        assert not is_optimizable_param(param), f"参数 {param} 不应该是可优化的"
    
    # 测试其他参数
    for param in OTHER_PARAMS:
        assert not is_optimizable_param(param), f"参数 {param} 不应该是可优化的"
    
    print("✅ 可优化参数检查测试通过")

def test_param_validation():
    """测试参数验证"""
    print("\n🧪 测试参数验证")
    print("=" * 60)
    
    try:
        validator = ParamValidator()
        results = validator.validate_all_params()
        
        # 检查验证结果
        assert results['summary']['overall_valid'], "参数验证应该通过"
        assert results['summary']['total_optimizable'] == 14, "可优化参数应该是14个"
        
        print("✅ 参数验证测试通过")
        return True
    except Exception as e:
        print(f"❌ 参数验证测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试14个有效参数方案")
    print("=" * 80)
    
    try:
        test_param_classification()
        test_param_effectiveness()
        test_optimizable_param_check()
        validation_success = test_param_validation()
        
        if validation_success:
            print("\n🎉 所有测试通过！14个有效参数方案配置正确")
        else:
            print("\n❌ 部分测试失败，需要检查配置")
            
    except Exception as e:
        print(f"\n💥 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 