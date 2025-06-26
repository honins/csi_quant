#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化测试示例
演示如何使用AI优化功能
"""

import sys
import os
import re

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.utils import setup_logging, load_config, Timer
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from ai.ai_optimizer import AIOptimizer

def load_previous_optimized_params(config):
    """
    从配置文件中读取之前优化的参数
    
    Args:
        config: 当前配置字典
    
    Returns:
        dict: 之前优化的参数字典，如果没有则返回None
    """
    try:
        # 从当前配置中提取之前保存的参数
        strategy_config = config.get('strategy', {})
        confidence_weights = strategy_config.get('confidence_weights', {})
        
        # 检查是否有之前保存的优化参数
        optimized_params = {}
        
        # 检查非核心参数
        param_keys = [
            'rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold',
            'dynamic_confidence_adjustment', 'market_sentiment_weight', 
            'trend_strength_weight', 'volume_weight', 'price_momentum_weight'
        ]
        
        has_optimized_params = False
        
        for key in param_keys:
            if key in confidence_weights:
                optimized_params[key] = confidence_weights[key]
                has_optimized_params = True
            elif key in strategy_config:
                optimized_params[key] = strategy_config[key]
                has_optimized_params = True
        
        if has_optimized_params:
            print(f"📖 从配置文件中读取到之前优化的参数:")
            for key, value in optimized_params.items():
                print(f"   - {key}: {value}")
            return optimized_params
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ 读取之前优化参数时出错: {e}")
        return None

def save_optimized_params_to_config(config, optimized_params):
    """
    保存优化后的参数到配置文件，保留原始注释
    
    Args:
        config: 当前配置字典
        optimized_params: 优化后的参数字典
    """
    try:
        # 读取原始配置文件以保留注释
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        # 读取原始文件内容
        with open(config_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 更新配置字典
        for key, value in optimized_params.items():
            if hasattr(value, 'item'):
                value = value.item()
            
            if key in ['rise_threshold', 'max_days']:
                config['strategy'][key] = value
                print(f"✅ 更新参数: {key} = {value}")
            elif key in ['rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold']:
                config['strategy']['confidence_weights'][key] = value
                print(f"✅ 更新参数: {key} = {value}")
            elif key in ['dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight']:
                config['strategy']['confidence_weights'][key] = value
                print(f"✅ 更新AI优化参数: {key} = {value}")
        
        # 使用ruamel.yaml保留注释和格式
        try:
            from ruamel.yaml import YAML
            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.indent(mapping=2, sequence=4, offset=2)
            
            # 读取原始文件
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.load(f)
            
            # 更新参数
            for key, value in optimized_params.items():
                if hasattr(value, 'item'):
                    value = value.item()
                
                if key in ['rise_threshold', 'max_days']:
                    yaml_data['strategy'][key] = value
                elif key in ['rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold']:
                    yaml_data['strategy']['confidence_weights'][key] = value
                elif key in ['dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight']:
                    yaml_data['strategy']['confidence_weights'][key] = value
            
            # 保存并保留注释
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f)
                
        except ImportError:
            # 如果没有ruamel.yaml，使用传统方法但尝试保留注释
            print("⚠️ 未安装ruamel.yaml，使用传统方法保存（可能丢失部分注释）")
            
            # 尝试手动替换参数值，保留注释
            updated_content = original_content
            
            for key, value in optimized_params.items():
                if hasattr(value, 'item'):
                    value = value.item()
                
                # 查找并替换参数值
                if key in ['rise_threshold', 'max_days']:
                    pattern = rf'(\s*{key}:\s*)[0-9.]+'
                    replacement = rf'\g<1>{value}'
                    updated_content = re.sub(pattern, replacement, updated_content)
                elif key in ['rsi_oversold_threshold', 'rsi_low_threshold', 'final_threshold']:
                    pattern = rf'(\s*{key}:\s*)[0-9.]+'
                    replacement = rf'\g<1>{value}'
                    updated_content = re.sub(pattern, replacement, updated_content)
                elif key in ['dynamic_confidence_adjustment', 'market_sentiment_weight', 'trend_strength_weight']:
                    pattern = rf'(\s*{key}:\s*)[0-9.]+'
                    replacement = rf'\g<1>{value}'
                    updated_content = re.sub(pattern, replacement, updated_content)
            
            # 保存更新后的内容
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
        
        print(f"✅ 配置已更新: {config_path}")
        print(f"📝 更新的参数:")
        for key, value in optimized_params.items():
            if hasattr(value, 'item'):
                value = value.item()
            if key not in ['rise_threshold', 'max_days']:
                print(f"   - {key}: {value}")
        print(f"🔒 固定参数:")
        print(f"   - rise_threshold: {config.get('strategy', {}).get('rise_threshold', 0.04)} (未修改)")
        print(f"   - max_days: {config.get('strategy', {}).get('max_days', 20)} (未修改)")
        
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")
        import traceback
        traceback.print_exc()

def run_ai_optimization(config):
    """
    运行AI优化的主要函数，供run.py调用
    
    参数:
    config: 配置字典
    
    返回:
    bool: 是否成功
    """
    print("🤖 启动AI优化...")
    
    # 设置日志，确保进度日志能正确显示
    setup_logging('INFO')
    
    try:
        print("📋 初始化模块...")
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        print("✅ 模块初始化完成")
        
        # 读取之前优化的参数作为初始值
        print("📖 读取之前优化的参数...")
        previous_params = load_previous_optimized_params(config)
        if previous_params:
            print(f"✅ 找到之前优化的参数: {previous_params}")
            # 更新策略模块的参数
            strategy_module.update_params(previous_params)
            print("✅ 已加载之前优化的参数作为初始值")
        else:
            print("ℹ️ 未找到之前优化的参数，使用默认参数")
        
        # 获取数据
        print("📊 准备数据...")
        start_date = '2020-01-01'
        end_date = '2025-06-19'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 检查高级优化选项
        advanced_config = config.get('ai', {}).get('advanced_optimization', {})
        bayesian_config = config.get('ai', {}).get('bayesian_optimization', {})
        use_hierarchical = advanced_config.get('use_hierarchical', True)
        use_bayesian = bayesian_config.get('enabled', True)
        
        if use_hierarchical:
            print("🏗️ 使用严格数据分割策略优化...")
            # 如果有之前的参数，调整搜索范围
            if previous_params:
                print("🔍 基于之前参数调整搜索范围...")
                # 可以在这里调整优化范围，使其围绕之前的参数进行微调
                pass
            
            # 使用严格数据分割进行优化
            print("📊 进行严格数据分割...")
            data_splits = ai_optimizer.strict_data_split(processed_data, preserve_test_set=True)
            train_data = data_splits['train']
            validation_data = data_splits['validation']
            test_data = data_splits['test']
            
            print(f"   - 训练集: {len(train_data)} 条")
            print(f"   - 验证集: {len(validation_data)} 条")
            print(f"   - 测试集: {len(test_data)} 条")
            
            # 在训练集上优化参数
            print("🔧 在训练集上优化参数...")
            timer_opt = Timer()
            timer_opt.start()
            optimized_params = ai_optimizer.optimize_strategy_parameters_on_train_only(
                strategy_module, train_data
            )
            timer_opt.stop()
            
            # 验证优化结果
            strategy_module.update_params(optimized_params)
            
            # 在验证集上评估
            print("📊 在验证集上评估...")
            val_backtest = strategy_module.backtest(validation_data)
            val_evaluation = strategy_module.evaluate_strategy(val_backtest)
            cv_score = val_evaluation['score']
            
            # 在测试集上最终评估
            print("🎯 在测试集上最终评估...")
            test_result = ai_optimizer.evaluate_on_test_set_only(strategy_module, test_data)
            test_score = test_result.get('test_score', 0.0) if test_result['success'] else 0.0
            
            # 过拟合检测
            overfitting_passed = test_score >= cv_score * 0.8
            difference_ratio = (cv_score - test_score) / cv_score if cv_score > 0 else 0
            
            print(f"✅ 严格数据分割优化完成")
            print(f"   - 最终参数: {optimized_params}")
            print(f"   - 验证集得分: {cv_score:.4f}")
            print(f"   - 测试集得分: {test_score:.4f}")
            print(f"   - 最佳得分: {cv_score:.4f}")
            print(f"   - 总耗时: {timer_opt.elapsed_str()}")
            print(f"   - 过拟合检测: {'通过' if overfitting_passed else '警告'}")
            
            # 检查过拟合风险
            if not overfitting_passed:
                print(f"   ⚠️ 检测到可能的过拟合风险:")
                print(f"      - 验证集得分: {cv_score:.4f}")
                print(f"      - 测试集得分: {test_score:.4f}")
                print(f"      - 差异比例: {difference_ratio:.1%}")
            
            # 使用优化后的参数更新策略
            strategy_module.update_params(optimized_params)
            
        else:
            # 智能参数优化（包含贝叶斯优化）
            print("🎯 使用智能参数优化...")
            
            if use_bayesian:
                print("🔍 启用贝叶斯优化")
                timer_bayes = Timer()
                timer_bayes.start()
                
                bayesian_result = ai_optimizer.bayesian_optimize_parameters(strategy_module, processed_data)
                
                timer_bayes.stop()
                
                if bayesian_result['success']:
                    optimized_params = bayesian_result['best_params']
                    print(f"✅ 贝叶斯优化成功 (耗时: {timer_bayes.elapsed_str()})")
                    print(f"   - 最优得分: {bayesian_result['best_score']:.4f}")
                    print(f"   - 评估次数: {bayesian_result['n_evaluations']}")
                    print(f"   - 改进率: {bayesian_result['improvement_rate']:.2%}")
                    print(f"   - 优化参数: {optimized_params}")
                else:
                    print(f"❌ 贝叶斯优化失败: {bayesian_result.get('error')}")
                    print("🔧 回退到传统优化方法...")
                    optimized_params = ai_optimizer._traditional_parameter_optimization(strategy_module, processed_data)
                    print(f"✅ 传统优化完成: {optimized_params}")
            else:
                print("🔧 使用传统参数优化...")
                optimized_params = ai_optimizer._traditional_parameter_optimization(strategy_module, processed_data)
                print(f"✅ 传统优化完成: {optimized_params}")
            
            strategy_module.update_params(optimized_params)
        
        # 严格数据分割后的模型训练和验证
        print("📊 使用严格数据分割进行模型验证...")
        data_splits = ai_optimizer.strict_data_split(processed_data, preserve_test_set=True)
        train_data = data_splits['train']
        validation_data = data_splits['validation']
        test_data = data_splits['test']
        
        print(f"   数据分割:")
        print(f"   - 训练集: {len(train_data)} 条")
        print(f"   - 验证集: {len(validation_data)} 条")
        print(f"   - 测试集: {len(test_data)} 条")
        
        # 验证优化效果
        print("📊 验证优化效果...")
        
        # 在验证集上评估优化后的策略
        val_backtest = strategy_module.backtest(validation_data)
        val_evaluation = strategy_module.evaluate_strategy(val_backtest)
        print(f"   - 验证集得分: {val_evaluation['score']:.4f}")
        print(f"   - 验证集成功率: {val_evaluation['success_rate']:.2%}")
        print(f"   - 验证集识别点数: {val_evaluation['total_points']}")
        
        # 在测试集上进行最终评估
        print("🎯 在测试集上进行最终评估...")
        test_result = ai_optimizer.evaluate_on_test_set_only(strategy_module, test_data)
        
        if test_result['success']:
            print(f"✅ 测试集评估完成")
            print(f"   - 测试集得分: {test_result['test_score']:.4f}")
            print(f"   - 成功率: {test_result['success_rate']:.2%}")
            print(f"   - 识别点数: {test_result['total_points']}")
            print(f"   - 平均涨幅: {test_result['avg_rise']:.2%}")
        else:
            print(f"❌ 测试集评估失败: {test_result.get('error')}")
        
        # 保存优化后的参数到配置文件
            
        print("💾 保存优化后的参数到配置文件...")
        # 只保存非核心参数，核心参数保持固定
        params_to_save = {
            'rsi_oversold_threshold': optimized_params.get('rsi_oversold_threshold', 30),
            'rsi_low_threshold': optimized_params.get('rsi_low_threshold', 40),
            'final_threshold': optimized_params.get('final_threshold', 0.5),
            # 新增AI优化参数
            'dynamic_confidence_adjustment': optimized_params.get('dynamic_confidence_adjustment', 0.1),
            'market_sentiment_weight': optimized_params.get('market_sentiment_weight', 0.15),
            'trend_strength_weight': optimized_params.get('trend_strength_weight', 0.12),
            # 新增2个高重要度参数
            'volume_weight': optimized_params.get('volume_weight', 0.25),
            'price_momentum_weight': optimized_params.get('price_momentum_weight', 0.20)
        }
        save_optimized_params_to_config(config, params_to_save)
        print(f"✅ 非核心参数已保存: {params_to_save}")
        print(f"🔒 核心参数保持固定: rise_threshold={config.get('strategy', {}).get('rise_threshold', 0.04)}, max_days={config.get('strategy', {}).get('max_days', 20)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ AI优化过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print("中证500指数相对低点识别系统 - AI优化测试")
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
                'index_code': 'SHSE.000905',
                'frequency': '1d'
            },
            'strategy': {
                'rise_threshold': 0.04,
                'max_days': 20
            },
            'optimization': {
                'param_ranges': {
                    'rise_threshold': {
                        'min': 0.03,
                        'max': 0.08,
                        'step': 0.005
                    },
                    'max_days': {
                        'min': 10,
                        'max': 30,
                        'step': 1
                    }
                },
                'genetic_algorithm': {
                    'population_size': 20,
                    'generations': 10,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.1
                }
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
        
        # 读取之前优化的参数
        print("\n📖 读取之前优化的参数...")
        previous_params = load_previous_optimized_params(config)
        if previous_params:
            print(f"✅ 找到之前优化的参数，将作为优化起点")
            # 更新策略模块的参数
            strategy_module.update_params(previous_params)
        else:
            print("ℹ️ 未找到之前优化的参数，使用默认参数")
        
        # 获取历史数据
        start_date = '2022-01-01'
        end_date = '2025-06-19'
        print(f"获取历史数据: {start_date} 到 {end_date}")
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 2. 基础策略测试（使用当前参数，可能是之前优化的）
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
        
        # 3. 参数优化测试（基于当前参数进行进一步优化）
        print("\n🔧 参数优化测试...")
        timer.start()
        
        # 使用改进的优化方法，基于当前参数进行优化
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
        
        # 4. 严格数据分割优化测试（基于当前参数进行分层优化）
        print("\n🏗️ 严格数据分割优化测试...")
        timer.start()
        
        # 进行严格数据分割
        data_splits = ai_optimizer.strict_data_split(processed_data, preserve_test_set=True)
        train_data = data_splits['train']
        validation_data = data_splits['validation']
        test_data = data_splits['test']
        
        # 在训练集上优化参数
        hierarchical_params = ai_optimizer.optimize_strategy_parameters_on_train_only(
            strategy_module, train_data
        )
        
        # 在验证集上评估
        strategy_module.update_params(hierarchical_params)
        val_backtest = strategy_module.backtest(validation_data)
        val_evaluation = strategy_module.evaluate_strategy(val_backtest)
        cv_score = val_evaluation['score']
        
        # 在测试集上评估
        test_result = ai_optimizer.evaluate_on_test_set_only(strategy_module, test_data)
        test_score = test_result.get('test_score', 0.0) if test_result['success'] else 0.0
        
        timer.stop()
        print(f"✅ 严格数据分割优化完成 (耗时: {timer.elapsed_str()})")
        print(f"   - 分层优化参数: {hierarchical_params}")
        print(f"   - 交叉验证得分: {cv_score:.4f}")
        print(f"   - 测试集得分: {test_score:.4f}")
        print(f"   - 最佳得分: {cv_score:.4f}")
        print(f"   - 总耗时: {timer.elapsed_str()}")
        
        # 使用分层优化后的参数测试
        strategy_module.update_params(hierarchical_params)
        hierarchical_backtest = strategy_module.backtest(processed_data)
        hierarchical_evaluation = strategy_module.evaluate_strategy(hierarchical_backtest)
        
        # 创建兼容的结果字典
        hierarchical_result = {
            'params': hierarchical_params,
            'cv_score': cv_score,
            'test_score': test_score,
            'best_score': cv_score
        }
        
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
        
        # 7. 贝叶斯优化测试
        print("\n🔍 贝叶斯优化测试...")
        timer.start()
        
        bayesian_result = ai_optimizer.bayesian_optimize_parameters(strategy_module, processed_data)
        
        timer.stop()
        
        if bayesian_result['success']:
            print(f"✅ 贝叶斯优化完成 (耗时: {timer.elapsed_str()})")
            print(f"   - 最优参数: {bayesian_result['best_params']}")
            print(f"   - 最优得分: {bayesian_result['best_score']:.4f}")
            print(f"   - 评估次数: {bayesian_result['n_evaluations']}")
            print(f"   - 改进率: {bayesian_result['improvement_rate']:.2%}")
            
            # 使用贝叶斯优化后的参数测试
            strategy_module.update_params(bayesian_result['best_params'])
            bayesian_backtest = strategy_module.backtest(processed_data)
            bayesian_evaluation = strategy_module.evaluate_strategy(bayesian_backtest)
            
            print(f"   - 贝叶斯优化后得分: {bayesian_evaluation['score']:.4f}")
            print(f"   - 成功率: {bayesian_evaluation['success_rate']:.2%}")
            print(f"   - 平均涨幅: {bayesian_evaluation['avg_rise']:.2%}")
        else:
            print(f"❌ 贝叶斯优化失败: {bayesian_result.get('error')}")
            bayesian_evaluation = {'score': 0.0}  # 设置默认值以避免后续错误
        
        # 8. 遗传算法优化测试
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
        
        # 9. 结果对比
        print("\n📊 优化结果对比:")
        print(f"   基础策略得分:     {baseline_evaluation['score']:.4f}")
        print(f"   参数优化得分:     {optimized_evaluation['score']:.4f}")
        print(f"   分层优化得分:     {hierarchical_evaluation['score']:.4f}")
        print(f"   贝叶斯优化得分:   {bayesian_evaluation['score']:.4f}")
        print(f"   遗传算法得分:     {genetic_evaluation['score']:.4f}")
        
        # 计算改进幅度
        param_improvement = (optimized_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        hierarchical_improvement = (hierarchical_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        bayesian_improvement = (bayesian_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        genetic_improvement = (genetic_evaluation['score'] - baseline_evaluation['score']) / baseline_evaluation['score'] * 100
        
        print(f"   参数优化改进:     {param_improvement:+.2f}%")
        print(f"   分层优化改进:     {hierarchical_improvement:+.2f}%")
        print(f"   贝叶斯优化改进:   {bayesian_improvement:+.2f}%")
        print(f"   遗传算法改进:     {genetic_improvement:+.2f}%")
        
        # 找出最佳方法
        methods = [
            ("基础策略", baseline_evaluation['score']),
            ("参数优化", optimized_evaluation['score']),
            ("分层优化", hierarchical_evaluation['score']),
            ("贝叶斯优化", bayesian_evaluation['score']),
            ("遗传算法", genetic_evaluation['score'])
        ]
        
        best_method = max(methods, key=lambda x: x[1])
        print(f"\n🏆 最佳优化方法: {best_method[0]} (得分: {best_method[1]:.4f})")
        
        # 保存最佳参数到配置文件
        print("\n💾 保存最佳参数到配置文件...")
        if best_method[0] == "分层优化":
            best_params = hierarchical_result['params']
        elif best_method[0] == "参数优化":
            best_params = optimized_params
        elif best_method[0] == "贝叶斯优化":
            best_params = bayesian_result.get('best_params', {}) if bayesian_result['success'] else {}
        elif best_method[0] == "遗传算法":
            best_params = genetic_params
        else:
            best_params = previous_params or {}
        
        # 只保存非核心参数
        params_to_save = {
            'rsi_oversold_threshold': best_params.get('rsi_oversold_threshold', 30),
            'rsi_low_threshold': best_params.get('rsi_low_threshold', 40),
            'final_threshold': best_params.get('final_threshold', 0.5),
            'dynamic_confidence_adjustment': best_params.get('dynamic_confidence_adjustment', 0.1),
            'market_sentiment_weight': best_params.get('market_sentiment_weight', 0.15),
            'trend_strength_weight': best_params.get('trend_strength_weight', 0.12),
            'volume_weight': best_params.get('volume_weight', 0.25),
            'price_momentum_weight': best_params.get('price_momentum_weight', 0.20)
        }
        save_optimized_params_to_config(config, params_to_save)
        print(f"✅ 最佳参数已保存: {params_to_save}")
        
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

