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

def save_optimized_params_to_config(config, optimized_params):
    """
    将优化后的参数保存到配置文件
    
    参数:
    config: 配置字典
    optimized_params: 优化后的参数字典
    """
    import yaml
    import os
    
    try:
        # 更新配置字典
        if 'strategy' not in config:
            config['strategy'] = {}
        
        # 更新策略参数
        for key, value in optimized_params.items():
            # 将numpy类型转换为Python原生类型
            if hasattr(value, 'item'):
                config['strategy'][key] = value.item()
            else:
                config['strategy'][key] = value
        
        # 保存到配置文件
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
        print(f"✅ 配置已更新: {config_path}")
        
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")

def run_ai_optimization(config):
    """
    运行AI优化的主要函数，供run.py调用
    
    参数:
    config: 配置字典
    
    返回:
    bool: 是否成功
    """
    print("🤖 启动AI优化...")
    
    try:
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        
        # 获取数据
        print("📊 准备数据...")
        start_date = '2022-01-01'
        end_date = '2025-06-19'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 检查高级优化选项
        advanced_config = config.get('ai', {}).get('advanced_optimization', {})
        use_hierarchical = advanced_config.get('use_hierarchical', True)
        
        if use_hierarchical:
            print("🏗️ 使用分层优化策略...")
            result = ai_optimizer.hierarchical_optimization(processed_data)
            
            print(f"✅ 分层优化完成")
            print(f"   - 最终参数: {result['strategy_params']}")
            print(f"   - 交叉验证得分: {result['cv_score']:.4f}")
            print(f"   - 最终得分: {result['final_score']:.4f}")
            print(f"   - 优化方法: {result['optimization_method']}")
            
            # 使用优化后的参数更新策略
            strategy_module.update_params(result['strategy_params'])
            
        else:
            # 传统优化方法
            print("🔧 使用传统参数优化...")
            optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
            strategy_module.update_params(optimized_params)
            print(f"✅ 参数优化完成: {optimized_params}")
        
        # 训练AI模型
        print("🤖 训练AI模型...")
        training_result = ai_optimizer.train_model(processed_data, strategy_module)
        
        if training_result['success']:
            print(f"✅ AI模型训练成功")
            print(f"   - 训练样本数: {training_result.get('train_samples')}")
            print(f"   - 特征数: {training_result.get('feature_count')}")
            
            # 验证模型
            validation_result = ai_optimizer.validate_model(processed_data, strategy_module)
            if validation_result['success']:
                print(f"   - 验证集准确率: {validation_result.get('accuracy'):.4f}")
                print(f"   - 精确率: {validation_result.get('precision'):.4f}")
                print(f"   - 召回率: {validation_result.get('recall'):.4f}")
                print(f"   - F1: {validation_result.get('f1_score'):.4f}")
        else:
            print(f"❌ AI模型训练失败: {training_result.get('error')}")
        
        # 运行回测
        print("📊 运行回测...")
        backtest_results = strategy_module.backtest(processed_data)
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        
        print(f"✅ 回测完成")
        print(f"   - 识别点数: {evaluation['total_points']}")
        print(f"   - 成功率: {evaluation['success_rate']:.2%}")
        print(f"   - 平均涨幅: {evaluation['avg_rise']:.2%}")
        print(f"   - 综合得分: {evaluation['score']:.4f}")
        
        # 保存优化后的参数到配置文件
        if use_hierarchical:
            optimized_params = result['strategy_params']
        else:
            optimized_params = optimized_params
            
        print("💾 保存优化后的参数到配置文件...")
        # 只保存rise_threshold，max_days保持原值
        params_to_save = {
            'rise_threshold': optimized_params['rise_threshold']
            # max_days 不保存，保持配置文件中的原值
        }
        save_optimized_params_to_config(config, params_to_save)
        print(f"✅ 参数已保存: {params_to_save}")
        print(f"📝 注意: max_days 保持配置文件中的原值: {config.get('strategy', {}).get('max_days', 20)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ AI优化过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
        start_date = '2022-01-01'
        end_date = '2025-06-19'
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
        
        # 使用改进的优化方法
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
        
        # 4. 分层优化测试
        print("\n🏗️ 分层优化测试...")
        timer.start()
        
        hierarchical_result = ai_optimizer.hierarchical_optimization(processed_data)
        
        timer.stop()
        print(f"✅ 分层优化完成 (耗时: {timer.elapsed_str()})")
        print(f"   - 分层优化参数: {hierarchical_result['strategy_params']}")
        print(f"   - 交叉验证得分: {hierarchical_result['cv_score']:.4f}")
        print(f"   - 最终得分: {hierarchical_result['final_score']:.4f}")
        print(f"   - 优化方法: {hierarchical_result['optimization_method']}")
        
        # 使用分层优化后的参数测试
        strategy_module.update_params(hierarchical_result['strategy_params'])
        hierarchical_backtest = strategy_module.backtest(processed_data)
        hierarchical_evaluation = strategy_module.evaluate_strategy(hierarchical_backtest)
        
        print(f"   - 分层优化后成功率: {hierarchical_evaluation['success_rate']:.2%}")
        print(f"   - 分层优化后平均涨幅: {hierarchical_evaluation['avg_rise']:.2%}")
        print(f"   - 分层优化后综合得分: {hierarchical_evaluation['score']:.4f}")
        
        # 5. AI模型训练测试
        print("\n🤖 AI模型训练测试...")
        timer.start()
        
        training_result = ai_optimizer.train_model(processed_data, strategy_module)
        validation_result = ai_optimizer.validate_model(processed_data, strategy_module)
        print('训练结果:', training_result)
        print('验证结果:', validation_result)
        if training_result.get('success'):
            print(f"   - 训练样本数: {training_result.get('train_samples')}")
            print(f"   - 特征数: {training_result.get('feature_count')}")
        if validation_result.get('success'):
            print(f"   - 验证集准确率: {validation_result.get('accuracy'):.4f}")
            print(f"   - 精确率: {validation_result.get('precision'):.4f}")
            print(f"   - 召回率: {validation_result.get('recall'):.4f}")
            print(f"   - F1: {validation_result.get('f1_score'):.4f}")
            print(f"   - 验证样本数: {validation_result.get('test_samples')}")
            print(f"   - 验证集正样本数: {validation_result.get('positive_samples_test')}")
        
        timer.stop()
        print(f"✅ AI模型训练完成 (耗时: {timer.elapsed_str()})")
        
        # 6. AI预测测试
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
        
        # 7. 遗传算法优化测试
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
        
        # 8. 结果对比
        print("\n📊 优化结果对比:")
        print(f"   基础策略得分:     {baseline_evaluation['score']:.4f}")
        print(f"   参数优化得分:     {optimized_evaluation['score']:.4f}")
        print(f"   分层优化得分:     {hierarchical_evaluation['score']:.4f}")
        print(f"   遗传算法得分:     {genetic_evaluation['score']:.4f}")
        
        # 计算改进幅度
        param_improvement = (optimized_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        hierarchical_improvement = (hierarchical_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        genetic_improvement = (genetic_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        
        print(f"   参数优化改进:     {param_improvement:+.2f}%")
        print(f"   分层优化改进:     {hierarchical_improvement:+.2f}%")
        print(f"   遗传算法改进:     {genetic_improvement:+.2f}%")
        
        # 找出最佳方法
        methods = [
            ("基础策略", baseline_evaluation['score']),
            ("参数优化", optimized_evaluation['score']),
            ("分层优化", hierarchical_evaluation['score']),
            ("遗传算法", genetic_evaluation['score'])
        ]
        
        best_method = max(methods, key=lambda x: x[1])
        print(f"\n🏆 最佳优化方法: {best_method[0]} (得分: {best_method[1]:.4f})")
        
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

