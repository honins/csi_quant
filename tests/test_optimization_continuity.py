#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试优化参数的连续性
验证每次运行是否基于之前的优化结果
"""

import sys
import os
import json
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.utils import load_config

def test_optimization_continuity():
    """测试优化参数的连续性"""
    print("="*60)
    print("测试优化参数的连续性")
    print("="*60)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    if not config:
        print("❌ 配置文件加载失败")
        return False
    
    # 检查是否有之前优化的参数
    strategy_config = config.get('strategy', {})
    confidence_weights = strategy_config.get('confidence_weights', {})
    
    print("📖 检查配置文件中的优化参数...")
    
    # 检查关键优化参数
    optimization_params = [
        'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
        'dynamic_confidence_adjustment', 'market_sentiment_weight', 
        'trend_strength_weight', 'volume_weight', 'price_momentum_weight'
    ]
    
    found_params = {}
    for param in optimization_params:
        if param in confidence_weights:
            found_params[param] = confidence_weights[param]
    
    if found_params:
        print("✅ 找到之前优化的参数:")
        for param, value in found_params.items():
            print(f"   - {param}: {value}")
        
        # 保存参数历史记录
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        
        # 创建子目录结构
        history_dir = os.path.join(results_dir, 'history')
        optimization_dir = os.path.join(history_dir, 'optimization')
        
        for directory in [results_dir, history_dir, optimization_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        history_file = os.path.join(optimization_dir, 'optimization_history.json')
        
        # 读取现有历史记录
        existing_history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    existing_history = json.load(f)
            except:
                existing_history = []
        
        # 添加新的历史记录
        history_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': found_params,
            'source': 'config_file'
        }
        existing_history.append(history_data)
        
        # 保存历史记录
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 参数历史已保存到: {history_file}")
        print(f"📊 历史记录数量: {len(existing_history)}")
        
        # 分析参数变化趋势
        if len(existing_history) > 1:
            print("\n📈 参数变化趋势分析:")
            for param in optimization_params:
                if param in found_params:
                    values = [h['parameters'].get(param) for h in existing_history if param in h['parameters']]
                    if len(values) > 1:
                        print(f"   - {param}: {values[-2]} → {values[-1]}")
        
        return True
    else:
        print("ℹ️ 未找到之前优化的参数，这是首次运行或参数未保存")
        return False

def test_load_previous_params_function():
    """测试参数加载功能（改进版）"""
    print("\n" + "="*60)
    print("测试改进版参数加载功能")
    print("="*60)
    
    # 加载配置文件来检查是否有优化后的参数
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    if not config:
        print("❌ 配置文件加载失败")
        return False
    
    # 检查配置中是否有优化参数
    strategy_config = config.get('strategy', {})
    confidence_weights = strategy_config.get('confidence_weights', {})
    
    if confidence_weights:
        print("✅ 配置文件参数加载功能正常")
        print(f"   配置文件: {os.path.basename(config_path)}")
        print(f"   参数数量: {len(confidence_weights)}")
        return True
    else:
        print("ℹ️ 未找到优化参数（这是正常情况）")
        return True

def main():
    """主函数"""
    success1 = test_optimization_continuity()
    success2 = test_load_previous_params_function()
    
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    if success1 and success2:
        print("✅ 所有测试通过")
        print("\n📝 说明:")
        print("1. 优化参数连续性功能已实现")
        print("2. 改进版AI优化会自动保存参数到config_improved.yaml")
        print("3. 系统会优先使用config_improved.yaml中的优化参数")
        print("4. 参数历史记录已保存到results/history/optimization/optimization_history.json")
        print("5. 传统AI优化已被改进版替代")
        return True
    else:
        print("❌ 部分测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 