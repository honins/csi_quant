#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查pkl文件中的实际参数
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_pkl_parameters():
    """检查pkl文件中的实际参数"""
    print("🔍 检查pkl文件中的实际参数")
    print("=" * 50)
    
    try:
        # 读取最新模型路径
        latest_model_path = "models/latest_improved_model.txt"
        if os.path.exists(latest_model_path):
            with open(latest_model_path, 'r') as f:
                model_path = f.read().strip()
        else:
            # 如果没有latest文件，使用最新的pkl文件
            models_dir = "models"
            pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if pkl_files:
                pkl_files.sort(reverse=True)  # 按文件名排序，最新的在前面
                model_path = os.path.join(models_dir, pkl_files[0])
            else:
                print("❌ 没有找到pkl文件")
                return False
        
        print(f"📁 检查模型文件: {model_path}")
        
        # 加载pkl文件
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("\n📋 pkl文件内容结构:")
        print("-" * 30)
        for key, value in model_data.items():
            if key == 'model':
                print(f"   {key}: {type(value).__name__}")
                if hasattr(value, 'named_steps'):
                    print(f"       Pipeline步骤: {list(value.named_steps.keys())}")
                    if 'classifier' in value.named_steps:
                        classifier = value.named_steps['classifier']
                        print(f"       分类器类型: {type(classifier).__name__}")
                        if hasattr(classifier, 'n_estimators'):
                            print(f"       决策树数量: {classifier.n_estimators}")
                        if hasattr(classifier, 'max_depth'):
                            print(f"       最大深度: {classifier.max_depth}")
            elif key == 'feature_names':
                print(f"   {key}: {len(value)} 个特征")
                print(f"      特征列表: {value}")
            elif key == 'incremental_count':
                print(f"   {key}: {value}")
            elif key == 'scaler':
                print(f"   {key}: {type(value).__name__}")
            else:
                print(f"   {key}: {type(value).__name__}")
        
        print("\n🎯 特征分析:")
        print("-" * 30)
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
            print(f"特征总数: {len(feature_names)}")
            
            # 分类特征
            trend_features = [f for f in feature_names if 'trend' in f]
            volume_features = [f for f in feature_names if 'volume' in f]
            ma_features = [f for f in feature_names if 'ma' in f and f != 'macd']
            price_features = [f for f in feature_names if 'price' in f]
            technical_features = [f for f in feature_names if f in ['rsi', 'macd', 'signal', 'hist', 'bb_upper', 'bb_lower']]
            distance_features = [f for f in feature_names if 'dist_' in f]
            
            print(f"趋势特征 ({len(trend_features)}): {trend_features}")
            print(f"成交量特征 ({len(volume_features)}): {volume_features}")
            print(f"均线特征 ({len(ma_features)}): {ma_features}")
            print(f"价格特征 ({len(price_features)}): {price_features}")
            print(f"技术指标 ({len(technical_features)}): {technical_features}")
            print(f"距离特征 ({len(distance_features)}): {distance_features}")
        
        print("\n🤖 模型信息:")
        print("-" * 30)
        if 'model' in model_data:
            model = model_data['model']
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                print(f"模型类型: {type(classifier).__name__}")
                if hasattr(classifier, 'n_estimators'):
                    print(f"决策树数量: {classifier.n_estimators}")
                if hasattr(classifier, 'max_depth'):
                    print(f"最大深度: {classifier.max_depth}")
                if hasattr(classifier, 'min_samples_split'):
                    print(f"最小分割样本数: {classifier.min_samples_split}")
                if hasattr(classifier, 'min_samples_leaf'):
                    print(f"最小叶子节点样本数: {classifier.min_samples_leaf}")
                if hasattr(classifier, 'class_weight'):
                    print(f"类别权重: {classifier.class_weight}")
        
        print("\n📊 文件大小信息:")
        print("-" * 30)
        file_size = os.path.getsize(model_path)
        print(f"文件大小: {file_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_optimization_params():
    """对比pkl文件参数与优化参数"""
    print("\n🔄 对比pkl文件参数与优化参数")
    print("=" * 50)
    
    try:
        # 从配置文件读取优化参数范围
        import yaml
        with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        strategy_ranges = config.get('strategy_ranges', {})
        optimization_ranges = config.get('optimization_ranges', {})
        
        print("📋 配置文件中的优化参数:")
        print("-" * 30)
        
        print("🔧 strategy_ranges (基础参数):")
        for param_name, param_config in strategy_ranges.items():
            if param_name not in ['rise_threshold', 'max_days']:
                print(f"   {param_name}: {param_config.get('min', 'N/A')} - {param_config.get('max', 'N/A')}")
        
        print("\n🤖 optimization_ranges (AI优化参数):")
        for param_name, param_config in optimization_ranges.items():
            print(f"   {param_name}: {param_config.get('min', 'N/A')} - {param_config.get('max', 'N/A')}")
        
        print(f"\n📊 总结:")
        print(f"   优化参数总数: {len(strategy_ranges) + len(optimization_ranges) - 2} (减去固定参数)")
        print(f"   pkl文件特征数: 根据实际加载的模型确定")
        print(f"   参数类型: 策略参数(影响标签生成) vs 模型特征(用于AI预测)")
        
        return True
        
    except Exception as e:
        print(f"❌ 对比异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        # 检查pkl文件参数
        success1 = check_pkl_parameters()
        
        # 对比优化参数
        success2 = compare_with_optimization_params()
        
        # 总结
        print("\n📊 总结:")
        print("=" * 30)
        print(f"pkl文件检查: {'✅ 成功' if success1 else '❌ 失败'}")
        print(f"参数对比: {'✅ 成功' if success2 else '❌ 失败'}")
        
        if success1 and success2:
            print("\n💡 说明:")
            print("1. pkl文件包含AI模型和特征信息，用于预测")
            print("2. 优化参数影响标签生成，用于训练")
            print("3. 两者是不同的概念：模型特征 vs 策略参数")
        
    except Exception as e:
        print(f"❌ 主函数异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 