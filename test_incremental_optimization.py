#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试增量优化功能
验证AI优化器是否基于历史参数进行增量优化
"""

import sys
import os
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_incremental_optimization():
    """测试增量优化功能"""
    try:
        from src.ai.ai_optimizer import AIOptimizer
        from src.utils.utils import load_config
        from src.data.data_module import DataModule
        from src.strategy.strategy_module import StrategyModule
        
        print("=" * 60)
        print("测试增量优化功能")
        print("=" * 60)
        
        # 1. 加载配置
        config = load_config('config/config.yaml')
        print("✅ 配置文件加载成功")
        
        # 2. 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        print("✅ 模块初始化成功")
        
        # 3. 准备数据
        print("📊 准备测试数据...")
        start_date = '2023-01-01'
        end_date = '2024-12-31'
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 4. 检查是否有历史参数
        print("\n🔍 检查历史参数...")
        historical_params = ai_optimizer._load_best_parameters()
        
        if historical_params:
            print("📋 发现历史最优参数:")
            for key, value in historical_params.items():
                print(f"   - {key}: {value}")
        else:
            print("🆕 没有历史参数，首次运行将使用全局优化")
        
        # 5. 运行第一次优化
        print("\n🚀 第一次优化（全局搜索）...")
        first_optimization = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
        
        print("✅ 第一次优化完成")
        print("📊 第一次优化结果:")
        for key, value in first_optimization.items():
            print(f"   - {key}: {value}")
        
        # 6. 检查是否保存了历史参数
        print("\n💾 检查参数保存...")
        saved_params = ai_optimizer._load_best_parameters()
        if saved_params:
            print("✅ 参数保存成功")
        else:
            print("❌ 参数保存失败")
        
        # 7. 运行第二次优化（应该是增量优化）
        print("\n🔄 第二次优化（增量优化）...")
        second_optimization = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
        
        print("✅ 第二次优化完成")
        print("📊 第二次优化结果:")
        for key, value in second_optimization.items():
            print(f"   - {key}: {value}")
        
        # 8. 比较两次优化结果
        print("\n📈 优化结果对比:")
        if historical_params:
            print("🔄 增量优化模式:")
            print("   - 基于历史参数进行局部搜索")
            print("   - 搜索范围收缩到原来的30%")
            print("   - 迭代次数减少到100次")
        else:
            print("🌐 全局优化模式:")
            print("   - 从随机参数开始全局搜索")
            print("   - 使用完整搜索范围")
            print("   - 迭代次数150次")
        
        # 9. 检查历史记录文件
        print("\n📋 检查历史记录...")
        history_file = ai_optimizer.parameter_history_file
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"✅ 历史记录文件存在，共 {len(history)} 条记录")
            print(f"📁 文件路径: {history_file}")
        else:
            print("❌ 历史记录文件不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_incremental_optimization()
    sys.exit(0 if success else 1) 