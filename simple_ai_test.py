#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的AI优化测试
"""

import os
import sys
import yaml
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def main():
    """主测试函数"""
    print("🧪 简单AI优化测试")
    print("=" * 50)
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        print("✅ 配置加载成功")
        
        # 2. 初始化模块
        print("🔧 初始化模块...")
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        print("✅ 模块初始化成功")
        
        # 3. 获取数据
        print("📊 获取数据...")
        data_config = config.get('data', {})
        time_range = data_config.get('time_range', {})
        start_date = time_range.get('start_date', '2019-01-01')
        end_date = time_range.get('end_date', '2025-07-15')
        
        data = data_module.get_history_data(start_date, end_date)
        
        if data is None or data.empty:
            print("❌ 无法获取数据")
            return
        
        print(f"✅ 获取数据成功: {len(data)} 条记录")
        
        # 4. 记录优化前的配置
        print("📋 记录优化前配置...")
        before_strategy = None
        if os.path.exists("config/strategy.yaml"):
            with open("config/strategy.yaml", 'r', encoding='utf-8') as f:
                before_strategy = yaml.safe_load(f)
        print(f"✅ 策略配置: {'已加载' if before_strategy else '不存在'}")
        
        # 5. 运行AI优化
        print("🚀 开始AI优化...")
        start_time = datetime.now()
        
        result = ai_optimizer.run_complete_optimization(data, strategy_module)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  优化耗时: {duration:.1f}秒")
        
        if result.get('success', False):
            print("✅ AI优化成功完成")
            
            # 6. 检查策略参数是否更新
            print("📋 检查策略参数更新...")
            after_strategy = None
            if os.path.exists("config/strategy.yaml"):
                with open("config/strategy.yaml", 'r', encoding='utf-8') as f:
                    after_strategy = yaml.safe_load(f)
            
            if before_strategy != after_strategy:
                print("✅ 策略参数已更新！")
                print("🔍 变化详情:")
                if before_strategy and after_strategy:
                    for key in set(before_strategy.keys()) | set(after_strategy.keys()):
                        if key not in before_strategy:
                            print(f"  ➕ 新增: {key} = {after_strategy[key]}")
                        elif key not in after_strategy:
                            print(f"  ➖ 删除: {key} = {before_strategy[key]}")
                        elif before_strategy[key] != after_strategy[key]:
                            print(f"  🔄 修改: {key} = {before_strategy[key]} → {after_strategy[key]}")
            else:
                print("❌ 策略参数未更新")
            
            # 7. 显示优化结果
            print("\n📊 优化结果:")
            print(f"  策略优化: {'成功' if result.get('strategy_optimization', {}).get('success', False) else '失败'}")
            print(f"  模型训练: {'成功' if result.get('model_training', {}).get('success', False) else '失败'}")
            print(f"  最终评估: {'成功' if result.get('final_evaluation', {}).get('success', False) else '失败'}")
            
        else:
            print("❌ AI优化失败")
            print(f"错误: {result.get('errors', [])}")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 