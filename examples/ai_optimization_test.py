#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化测试示例
演示如何使用AI优化功能
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.utils import setup_logging, load_config, Timer
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from ai.ai_optimizer import AIOptimizer

def main():
    """主函数"""
    print("="*60)
    print("中证1000指数相对低点识别系统 - AI优化测试")
    print("="*60)
    
    # 设置日志
    setup_logging('INFO')
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    if not config:
        print("❌ 配置文件加载失败，使用默认配置")
        config = {
            'data': {
                'index_code': 'SHSE.000852',
                'frequency': '1d'
            },
            'strategy': {
                'rise_threshold': 0.05,
                'max_days': 20
            },
            'ai': {
                'model_type': 'machine_learning',
                'optimization_interval': 30
            }
        }
    
    try:
        # 1. 准备数据
        print("\n📊 准备数据...")
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        
        # 获取历史数据
        start_date = '2023-01-01'
        end_date = '2024-12-31'
        print(f"获取历史数据: {start_date} 到 {end_date}")
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 2. 基础策略测试
        print("\n🎯 基础策略测试...")
        timer = Timer()
        timer.start()
        
        backtest_results = strategy_module.backtest(processed_data)
        baseline_evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        timer.stop()
        print(f"✅ 基础策略测试完成 (耗时: {timer.elapsed_str()})")
        print(f"   - 识别点数: {baseline_evaluation['total_points']}")
        print(f"   - 成功率: {baseline_evaluation['success_rate']:.2%}")
        print(f"   - 平均涨幅: {baseline_evaluation['avg_rise']:.2%}")
        print(f"   - 综合得分: {baseline_evaluation['score']:.4f}")
        
        # 3. 参数优化测试
        print("\n🔧 参数优化测试...")
        timer.start()
        
        optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
        
        timer.stop()
        print(f"✅ 参数优化完成 (耗时: {timer.elapsed_str()})")
        print(f"   - 优化后参数: {optimized_params}")
        
        # 使用优化后的参数测试
        strategy_module.update_params(optimized_params)
        optimized_backtest = strategy_module.backtest(processed_data)
        optimized_evaluation = strategy_module.evaluate_strategy(optimized_backtest)
        
        print(f"   - 优化后成功率: {optimized_evaluation['success_rate']:.2%}")
        print(f"   - 优化后平均涨幅: {optimized_evaluation['avg_rise']:.2%}")
        print(f"   - 优化后综合得分: {optimized_evaluation['score']:.4f}")
        
        # 4. AI模型训练测试
        print("\n🤖 AI模型训练测试...")
        timer.start()
        
        training_result = ai_optimizer.train_prediction_model(processed_data, strategy_module)
        
        timer.stop()
        print(f"✅ AI模型训练完成 (耗时: {timer.elapsed_str()})")
        
        if training_result['success']:
            print(f"   - 准确率: {training_result['accuracy']:.4f}")
            print(f"   - 精确率: {training_result['precision']:.4f}")
            print(f"   - 召回率: {training_result['recall']:.4f}")
            print(f"   - F1得分: {training_result['f1_score']:.4f}")
            print(f"   - 特征数量: {training_result['feature_count']}")
            print(f"   - 训练样本: {training_result['train_samples']}")
            print(f"   - 测试样本: {training_result['test_samples']}")
        else:
            print(f"   - 训练失败: {training_result.get('error', '未知错误')}")
        
        # 5. AI预测测试
        if training_result['success']:
            print("\n🔮 AI预测测试...")
            
            # 使用最新数据进行预测
            prediction_result = ai_optimizer.predict_low_point(processed_data)
            
            print(f"✅ AI预测完成")
            print(f"   - 预测结果: {'相对低点' if prediction_result['is_low_point'] else '非相对低点'}")
            print(f"   - 置信度: {prediction_result['confidence']:.4f}")
            
            # 获取特征重要性
            feature_importance = ai_optimizer.get_feature_importance()
            if feature_importance:
                print(f"   - 前5个重要特征:")
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                    print(f"     {i+1}. {feature}: {importance:.4f}")
        
        # 6. 遗传算法优化测试
        print("\n🧬 遗传算法优化测试...")
        timer.start()
        
        def evaluate_individual(params):
            """评估个体的适应度"""
            strategy_module.update_params(params)
            backtest_results = strategy_module.backtest(processed_data)
            evaluation = strategy_module.evaluate_strategy(backtest_results)
            return evaluation['score']
        
        genetic_params = ai_optimizer.run_genetic_algorithm(
            evaluate_individual, 
            population_size=10,  # 减少种群大小以加快测试
            generations=5        # 减少迭代次数以加快测试
        )
        
        timer.stop()
        print(f"✅ 遗传算法优化完成 (耗时: {timer.elapsed_str()})")
        print(f"   - 遗传算法最优参数: {genetic_params}")
        
        # 使用遗传算法优化后的参数测试
        strategy_module.update_params(genetic_params)
        genetic_backtest = strategy_module.backtest(processed_data)
        genetic_evaluation = strategy_module.evaluate_strategy(genetic_backtest)
        
        print(f"   - 遗传算法优化后得分: {genetic_evaluation['score']:.4f}")
        
        # 7. 结果对比
        print("\n📊 优化结果对比:")
        print(f"   基础策略得分:     {baseline_evaluation['score']:.4f}")
        print(f"   参数优化得分:     {optimized_evaluation['score']:.4f}")
        print(f"   遗传算法得分:     {genetic_evaluation['score']:.4f}")
        
        # 计算改进幅度
        param_improvement = (optimized_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        genetic_improvement = (genetic_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        
        print(f"   参数优化改进:     {param_improvement:+.2f}%")
        print(f"   遗传算法改进:     {genetic_improvement:+.2f}%")
        
        print("\n🎉 AI优化测试完成！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

