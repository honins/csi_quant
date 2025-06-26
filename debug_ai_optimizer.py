#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调试AIOptimizer问题的脚本
"""

import sys
import os
sys.path.insert(0, 'src')

def check_ai_optimizer():
    """检查AIOptimizer类的问题"""
    print("🔍 正在调试AIOptimizer问题...")
    
    try:
        # 尝试导入
        from ai.ai_optimizer import AIOptimizer
        print("✅ 成功导入AIOptimizer")
        
        # 检查类的方法
        methods = [m for m in dir(AIOptimizer) if not m.startswith('_')]
        print(f"📋 AIOptimizer的所有公共方法 ({len(methods)}个):")
        for method in sorted(methods):
            print(f"   - {method}")
        
        # 检查具体方法
        methods_to_check = ['train_model', 'validate_model', 'predict_low_point']
        print("\n🔍 检查特定方法:")
        for method in methods_to_check:
            if hasattr(AIOptimizer, method):
                print(f"   ✅ {method}: 存在")
            else:
                print(f"   ❌ {method}: 不存在")
        
        # 尝试创建实例
        from utils.utils import load_config
        config = load_config('config/config.yaml')
        ai_optimizer = AIOptimizer(config)
        print("✅ 成功创建AIOptimizer实例")
        
        # 检查实例方法
        print("\n🔍 检查实例方法:")
        for method in methods_to_check:
            if hasattr(ai_optimizer, method):
                print(f"   ✅ 实例.{method}: 存在")
            else:
                print(f"   ❌ 实例.{method}: 不存在")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_ai_optimizer() 