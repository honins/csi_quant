#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试配置分离功能
验证主配置文件和策略配置文件的分离开是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_loader import ConfigLoader, load_config, load_strategy_config
from src.utils.utils import load_main_config

def test_config_separation():
    """测试配置分离功能"""
    print("=" * 60)
    print("🧪 测试配置分离功能")
    print("=" * 60)
    
    try:
        # 1. 测试配置加载器
        print("\n1️⃣ 测试配置加载器...")
        loader = ConfigLoader("config")
        
        # 列出配置文件
        config_files = loader.list_config_files()
        print(f"   配置文件列表: {config_files}")
        
        # 2. 测试加载主配置
        print("\n2️⃣ 测试加载主配置...")
        main_config = loader.load_main_config()
        print(f"   主配置键: {list(main_config.keys())}")
        
        # 3. 测试加载策略配置
        print("\n3️⃣ 测试加载策略配置...")
        strategy_config = loader.load_strategy_config()
        print(f"   策略配置键: {list(strategy_config.keys())}")
        
        # 4. 测试合并配置
        print("\n4️⃣ 测试合并配置...")
        merged_config = loader.load_config()
        print(f"   合并配置键: {list(merged_config.keys())}")
        
        # 5. 验证策略配置是否正确合并
        print("\n5️⃣ 验证策略配置合并...")
        if 'strategy' in merged_config:
            strategy = merged_config['strategy']
            print(f"   ✅ 策略配置已合并")
            print(f"   - 涨幅阈值: {strategy.get('rise_threshold')}")
            print(f"   - 最大天数: {strategy.get('max_days')}")
            print(f"   - 移动平均线周期: {strategy.get('ma_periods')}")
        else:
            print("   ❌ 策略配置未找到")
        
        # 6. 验证置信度权重配置
        print("\n6️⃣ 验证置信度权重配置...")
        if 'confidence_weights' in merged_config:
            weights = merged_config['confidence_weights']
            print(f"   ✅ 置信度权重配置已合并")
            print(f"   - RSI超卖权重: {weights.get('rsi_oversold')}")
            print(f"   - RSI偏低权重: {weights.get('rsi_low')}")
            print(f"   - 最终阈值: {weights.get('final_threshold')}")
        else:
            print("   ❌ 置信度权重配置未找到")
        
        # 7. 验证优化配置
        print("\n7️⃣ 验证优化配置...")
        if 'optimization' in merged_config:
            optimization = merged_config['optimization']
            print(f"   ✅ 优化配置已合并")
            print(f"   - 遗传算法种群大小: {optimization.get('genetic_algorithm', {}).get('population_size')}")
            print(f"   - 参数搜索范围: {list(optimization.get('param_ranges', {}).keys())}")
        else:
            print("   ❌ 优化配置未找到")
        
        # 8. 验证主配置是否保留
        print("\n8️⃣ 验证主配置保留...")
        if 'ai' in merged_config:
            ai_config = merged_config['ai']
            print(f"   ✅ AI配置已保留")
            print(f"   - 模型类型: {ai_config.get('model_type')}")
            print(f"   - 优化间隔: {ai_config.get('optimization_interval')}")
        else:
            print("   ❌ AI配置未找到")
        
        if 'data' in merged_config:
            data_config = merged_config['data']
            print(f"   ✅ 数据配置已保留")
            print(f"   - 数据源: {data_config.get('data_source')}")
            print(f"   - 指数代码: {data_config.get('index_code')}")
        else:
            print("   ❌ 数据配置未找到")
        
        # 9. 测试配置验证
        print("\n9️⃣ 测试配置验证...")
        is_valid = loader.validate_config(merged_config)
        if is_valid:
            print("   ✅ 配置验证通过")
        else:
            print("   ❌ 配置验证失败")
        
        # 10. 测试便捷函数
        print("\n🔟 测试便捷函数...")
        config_from_utils = load_config()
        strategy_from_utils = load_strategy_config()
        main_from_utils = load_main_config()
        
        print(f"   ✅ 便捷函数测试通过")
        print(f"   - 合并配置键数: {len(config_from_utils)}")
        print(f"   - 策略配置键数: {len(strategy_from_utils)}")
        print(f"   - 主配置键数: {len(main_from_utils)}")
        
        print("\n" + "=" * 60)
        print("🎉 配置分离测试完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config_structure():
    """测试配置结构"""
    print("\n" + "=" * 60)
    print("📋 配置结构分析")
    print("=" * 60)
    
    try:
        config = load_config()
        
        print("\n📊 配置结构:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"     {sub_key}: {list(sub_value.keys())}")
                    else:
                        print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        print("\n✅ 配置结构分析完成")
        
    except Exception as e:
        print(f"❌ 配置结构分析失败: {str(e)}")

if __name__ == "__main__":
    # 运行测试
    success = test_config_separation()
    
    if success:
        test_config_structure()
    else:
        print("❌ 配置分离测试失败，跳过结构分析") 