#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速运行脚本
提供简单的命令行界面来运行系统的各种功能
"""

import sys
import os
import argparse
import re
import time
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class PerformanceTimer:
    """性能监控类"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.command = None
    
    def start(self, command_name):
        """开始计时"""
        self.command = command_name
        self.start_time = time.time()
        print(f"⏱️  开始执行 '{command_name}' 命令...")
    
    def stop(self):
        """停止计时"""
        if self.start_time:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print(f"\n⏱️  '{self.command}' 命令执行完成")
            print(f"📊 执行时间: {duration:.2f}秒 ({self.format_duration(duration)})")
            return duration
        return 0
    
    @staticmethod
    def format_duration(seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}分{secs:.0f}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}小时{int(minutes)}分钟"

def get_config_files():
    """获取配置文件列表，支持环境变量配置"""
    # 默认配置文件列表（按优先级排序）
    default_config_files = [
        'config_core.yaml',      # 核心系统配置
        'optimization.yaml',     # 优化配置
        'config.yaml'            # 兼容性配置（如果存在）
    ]
    
    # 检查环境变量配置
    env_config_path = os.environ.get('CSI_CONFIG_PATH')
    if env_config_path:
        if os.path.isabs(env_config_path):
            # 绝对路径，直接添加
            default_config_files.append(env_config_path)
        else:
            # 相对路径，添加到列表
            default_config_files.append(env_config_path)
        print(f"🔧 使用环境变量指定的额外配置文件: {env_config_path}")
    
    return default_config_files

def validate_date_format(date_str):
    """验证日期格式"""
    if not date_str:
        return False
    
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def load_config_safely(custom_config_files=None):
    """安全加载配置文件"""
    try:
        from src.utils.config_loader import load_config
        
        # 如果指定了自定义配置文件，直接使用
        if custom_config_files:
            if isinstance(custom_config_files, str):
                custom_config_files = [custom_config_files]
            config_files = custom_config_files
        else:
            config_files = get_config_files()
        
        print(f"📁 使用多配置文件加载: {', '.join([os.path.basename(f) for f in config_files[:2]])}...")
        return load_config(config_files=config_files)
    except ImportError as e:
        print(f"❌ 无法导入配置加载模块: {e}")
        print("💡 请确保已激活虚拟环境并安装了所有依赖包")
        return None
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        print("💡 提示: 可以通过环境变量 CSI_CONFIG_PATH 指定额外配置文件")
        return None

def check_virtual_environment():
    """检查是否在虚拟环境中运行"""
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if not in_venv:
        print("⚠️  警告: 您似乎没有在虚拟环境中运行")
        print("💡 建议: 激活虚拟环境以避免依赖冲突")
        print("   Windows: venv\\Scripts\\activate")
        print("   Linux/Mac: source venv/bin/activate")
        return False
    else:
        venv_path = os.environ.get('VIRTUAL_ENV', '当前虚拟环境')
        print(f"✅ 运行在虚拟环境: {os.path.basename(venv_path)}")
        return True

def run_data_fetch():
    """运行数据获取"""
    print("运行数据获取...")
    try:
        from src.data.data_fetch import main
        return main()
    except Exception as e:
        print(f"❌ 数据获取执行失败: {e}")
        return False

def run_basic_test():
    """运行基础测试"""
    print("运行基础测试...")
    try:
        from examples.basic_test import main
        return main()
    except ImportError as e:
        print(f"❌ 无法导入基础测试模块: {e}")
        print("💡 请检查是否已安装所有依赖包: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 基础测试执行失败: {e}")
        return False

def run_ai_test():
    """运行AI优化测试"""
    print("运行改进版AI优化测试...")
    try:
        config = load_config_safely()
        if not config:
            print("❌ 配置文件加载失败")
            return False
        
        # 使用改进版AI优化替代传统优化
        return run_ai_optimization_improved(config)
    except Exception as e:
        print(f"❌ AI测试执行失败: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    try:
        import unittest
        
        # 发现并运行所有测试
        loader = unittest.TestLoader()
        start_dir = os.path.join(os.path.dirname(__file__), 'tests')
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"❌ 单元测试执行失败: {e}")
        return False

def run_rolling_backtest(start_date, end_date):
    from examples.run_rolling_backtest import run_rolling_backtest as rolling_func
    import os
    
    print("="*80)
    print("📈 AI模型滚动回测系统")
    print("="*80)
    print(f"📋 回测配置:")
    print(f"   📅 开始日期: {start_date}")
    print(f"   📅 结束日期: {end_date}")
    print(f"   🤖 使用AI预测模型进行滚动回测")
    print(f"   💾 回测图表将自动保存到 results/charts/rolling_backtest/ 目录")
    print("="*80)
    
    # 智能训练策略配置 - 默认使用保守训练模式
    print("\n🤖 训练策略配置:")
    print("   1. 智能训练 - 30天重训练一次，大幅提升效率")
    print("   2. 保守训练 (当前) - 10天重训练一次，保证准确性")
    print("   3. 传统模式 - 每日重训练，最高准确性但效率低")
    
    # 默认使用保守训练模式 (选项2)
    choice = "2"
    print("✅ 自动选择: 保守训练模式 (10天间隔)")
    reuse_model = True
    retrain_interval = 10
    
    print(f"\n🚀 开始回测...")
    success = rolling_func(start_date, end_date, reuse_model=reuse_model, retrain_interval_days=retrain_interval)
    
    if success:
        print("\n" + "="*80)
        print("📁 回测结果文件已保存")
        print("="*80)
        print("🔍 查看保存的文件:")
        
        # 检查results目录中最新生成的文件
        results_dir = "results"
        if os.path.exists(results_dir):
            import glob
            from datetime import datetime as dt
            
            # 定义回测图表目录
            charts_dir = os.path.join(results_dir, 'charts')
            backtest_dir = os.path.join(charts_dir, 'rolling_backtest')
            
            if os.path.exists(backtest_dir):
                # 查找最新生成的回测图表文件
                pattern_results = os.path.join(backtest_dir, 'rolling_backtest_results_*.png')
                pattern_details = os.path.join(backtest_dir, 'prediction_details_*.png')
                
                results_files = glob.glob(pattern_results)
                details_files = glob.glob(pattern_details)
                
                if results_files:
                    latest_results = max(results_files, key=os.path.getctime)
                    print(f"   📊 回测结果图表: {os.path.relpath(latest_results)}")
                    print(f"      位置: charts/rolling_backtest/ 目录")
                
                if details_files:
                    latest_details = max(details_files, key=os.path.getctime)
                    print(f"   📋 预测详情表格: {os.path.relpath(latest_details)}")
                    print(f"      位置: charts/rolling_backtest/ 目录")
                
        
        print("="*80)
    
    return success

def run_single_day_test(predict_date):
    from examples.predict_single_day import predict_single_day
    import os
    
    print("="*80)
    print("🔮 单日相对低点预测系统")
    print("="*80)
    
    # 检查最新模型信息
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    latest_model_path = os.path.join(models_dir, 'latest_model.txt')
    
    if os.path.exists(latest_model_path):
        with open(latest_model_path, 'r') as f:
            model_files = f.read().strip().split('\n')
            if len(model_files) >= 1:
                model_file = os.path.basename(model_files[0])
                if 'model_' in model_file:
                    timestamp_str = model_file.replace('model_', '').replace('.pkl', '')
                    try:
                        from datetime import datetime
                        model_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        print(f"📋 系统配置:")
                        print(f"   🤖 AI模型版本: {model_file}")
                        print(f"   🕐 模型训练时间: {model_time.strftime('%Y年%m月%d日 %H:%M:%S')}")
                        
                        # 计算模型年龄
                        model_age = datetime.now() - model_time
                        if model_age.days == 0:
                            age_str = f"{model_age.seconds // 3600}小时{(model_age.seconds % 3600) // 60}分钟"
                            status = "🟢 非常新鲜"
                        elif model_age.days < 7:
                            age_str = f"{model_age.days}天"
                            status = "🟢 较新"
                        elif model_age.days < 30:
                            age_str = f"{model_age.days}天"
                            status = "🟡 需考虑更新"
                        else:
                            age_str = f"{model_age.days}天"
                            status = "🔴 建议重新训练"
                        
                        print(f"   📅 模型年龄: {age_str} ({status})")
                    except Exception as e:
                        print(f"   🤖 AI模型: {model_file}")
                        print(f"   ⚠️  无法解析模型时间: {e}")
    
    print(f"   🎯 预测日期: {predict_date}")
    print(f"   ✅ 使用已训练模型进行预测")
    print(f"   💾 预测结果将自动保存到 results/ 子目录")
    print("="*80)
    
    # 默认使用已训练模型，如果需要重新训练可以添加参数
    success = predict_single_day(predict_date, use_trained_model=True)
    
    if success:
        print("\n" + "="*80)
        print("📁 结果文件已保存")
        print("="*80)
        print("🔍 查看保存的文件:")
        
        # 检查results目录中最新生成的文件
        results_dir = "results"
        if os.path.exists(results_dir):
            import glob
            from datetime import datetime as dt
            
            # 定义子目录
            single_predictions_dir = os.path.join(results_dir, 'single_predictions')
            reports_dir = os.path.join(results_dir, 'reports')
            history_dir = os.path.join(results_dir, 'history')
            
            # 查找今天生成的预测文件
            pattern_json = os.path.join(single_predictions_dir, f'prediction_{predict_date}_*.json')
            pattern_md = os.path.join(reports_dir, f'report_{predict_date}_*.md')
            
            json_files = glob.glob(pattern_json)
            md_files = glob.glob(pattern_md)
            
            if json_files:
                latest_json = max(json_files, key=os.path.getctime)
                print(f"   📄 JSON数据文件: {os.path.relpath(latest_json)}")
                print(f"      位置: single_predictions/ 目录")
            
            if md_files:
                latest_md = max(md_files, key=os.path.getctime)
                print(f"   📋 Markdown报告文件: {os.path.relpath(latest_md)}")
                print(f"      位置: reports/ 目录")
            
            # 检查预测历史文件
            history_file = os.path.join(history_dir, 'prediction_history.json')
            if os.path.exists(history_file):
                print(f"   📊 预测历史记录: {os.path.relpath(history_file)}")
                print(f"      位置: history/ 目录")
            

        
        print("="*80)
    
    return success

def run_strategy_test(iterations):
    try:
        from examples.llm_strategy_optimizer import LLMStrategyOptimizer
        
        config = load_config_safely()
        if not config:
            return False
            
        optimizer = LLMStrategyOptimizer(config)
        return optimizer.optimize_strategy(num_iterations=iterations)
    except ImportError as e:
        print(f"❌ 无法导入策略优化模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 策略优化执行失败: {e}")
        return False

def run_incremental_training(mode='incremental'):
    """运行增量训练"""
    print("="*60)
    print("🤖 AI模型训练系统")
    print("="*60)
    
    try:
        from src.ai.ai_optimizer_improved import AIOptimizerImproved
        
        config = load_config_safely()
        if not config:
            return False
            
        # 使用多配置文件加载
        
        print(f"📋 训练配置:")
        print(f"   🎯 训练模式: {mode}")
        print("="*60)
        
        # 创建AI优化器
        ai_optimizer = AIOptimizerImproved(config)
        
        if mode == 'incremental':
            print("🔄 开始增量训练...")
            from datetime import datetime, timedelta
            from src.data.data_module import DataModule
            from src.strategy.strategy_module import StrategyModule
            
            # 准备训练数据
            data_module = DataModule(config)
            strategy_module = StrategyModule(config)
            
            # 获取最近的数据
            end_date = datetime.now()
            incremental_years = config.get('ai', {}).get('training_data', {}).get('incremental_years', 1)
            start_date = end_date - timedelta(days=365*incremental_years)
            
            training_data = data_module.get_history_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not training_data.empty:
                training_data = data_module.preprocess_data(training_data)
                train_result = ai_optimizer.incremental_train(training_data, strategy_module)
                
                print(f"\n📊 训练结果:")
                print(f"   ✅ 训练状态: {'成功' if train_result['success'] else '失败'}")
                print(f"   🔄 训练方式: {train_result.get('method', 'unknown')}")
                print(f"   📊 更新次数: {train_result.get('update_count', 0)}")
                print(f"   🔢 新增样本: {train_result.get('new_samples', 0)}")
                
                success = train_result['success']
            else:
                print("❌ 无法获取训练数据")
                success = False
                
        elif mode == 'full':
            print("🔄 开始完全重训练...")
            from datetime import datetime, timedelta
            from src.data.data_module import DataModule
            from src.strategy.strategy_module import StrategyModule
            
            # 准备训练数据
            data_module = DataModule(config)
            strategy_module = StrategyModule(config)
            
            # 获取更长时间的数据用于完全重训练
            end_date = datetime.now()
            training_years = config.get('ai', {}).get('training_data', {}).get('full_train_years', 8)
            start_date = end_date - timedelta(days=365*training_years)  # 可配置年数，默认8年
            
            training_data = data_module.get_history_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not training_data.empty:
                training_data = data_module.preprocess_data(training_data)
                train_result = ai_optimizer.full_train(training_data, strategy_module)
                
                print(f"\n📊 训练结果:")
                print(f"   ✅ 训练状态: {'成功' if train_result['success'] else '失败'}")
                print(f"   📊 训练样本数: {train_result.get('train_samples', 0)}")
                print(f"   📈 特征数量: {train_result.get('feature_count', 0)}")
                print(f"   🔄 训练方式: {train_result.get('method', 'unknown')}")
                
                success = train_result['success']
            else:
                print("❌ 无法获取训练数据")
                success = False
        else:  # demo
            print("🔮 开始演示预测...")
            from datetime import datetime, timedelta
            
            # 获取最近的数据进行预测演示
            predict_date = datetime.now().strftime('%Y-%m-%d')
            
            # 准备演示数据（最近60天）
            from src.data.data_module import DataModule
            data_module = DataModule(config)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            demo_data = data_module.get_history_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not demo_data.empty:
                demo_data = data_module.preprocess_data(demo_data)
                prediction_result = ai_optimizer.predict_low_point(demo_data, predict_date)
                
                print(f"\n📊 演示预测结果:")
                print(f"   📅 预测日期: {predict_date}")
                print(f"   🎯 预测结果: {'📈 相对低点' if prediction_result.get('is_low_point', False) else '📉 非相对低点'}")
                print(f"   🔢 原始置信度: {prediction_result.get('confidence', 0):.4f}")
                print(f"   ✨ 最终置信度: {prediction_result.get('final_confidence', 0):.4f}")
                print(f"   🤖 模型类型: {prediction_result.get('model_type', 'unknown')}")
                
                success = True
            else:
                print("❌ 无法获取演示数据")
                success = False
        
        if success:
            print("\n✅ 训练完成！")
            return True
        else:
            print("\n❌ 训练失败！")
            return False
            
    except ImportError as e:
        print(f"\n❌ 无法导入AI训练模块: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def print_complete_optimization_results(optimization_result: dict, config: dict):
    """
    打印完整的优化结果和所有参数
    
    参数:
    optimization_result: 优化结果字典
    config: 配置字典
    """
    print("\n" + "="*80)
    print("📊 完整优化参数报告")
    print("="*80)
    
    # 1. 策略优化参数
    strategy_opt = optimization_result.get('strategy_optimization', {})
    if strategy_opt.get('success'):
        print("\n🔧 策略参数优化结果:")
        print("-"*60)
        best_params = strategy_opt.get('best_params', {})
        print(f"   ✅ 优化方法: {strategy_opt.get('optimization_method', 'unknown')}")
        
        # 显示三层数据分割信息
        data_split = strategy_opt.get('data_split', {})
        if data_split:
            print(f"\n   📊 严格三层数据分割:")
            print(f"      • 训练集: {data_split.get('train_samples', 0):,}条 ({data_split.get('train_ratio', 0):.1%}) - 仅用于参数优化")
            print(f"      • 验证集: {data_split.get('validation_samples', 0):,}条 ({data_split.get('validation_ratio', 0):.1%}) - 用于模型验证")
            print(f"      • 测试集: {data_split.get('test_samples', 0):,}条 ({data_split.get('test_ratio', 0):.1%}) - 完全锁定最终评估")
        
        # 显示三层验证结果
        print(f"\n   📈 三层验证结果:")
        print(f"      • 训练集得分: {strategy_opt.get('best_score', 0):.4f}")
        print(f"      • 验证集得分: {strategy_opt.get('validation_score', 0):.4f} | 成功率: {strategy_opt.get('validation_success_rate', 0):.2%} | 识别点数: {strategy_opt.get('validation_total_points', 0)} | 平均涨幅: {strategy_opt.get('validation_avg_rise', 0):.2%}")
        if 'test_score' in strategy_opt:
            print(f"      • 测试集得分: {strategy_opt.get('test_score', 0):.4f} | 成功率: {strategy_opt.get('test_success_rate', 0):.2%} | 识别点数: {strategy_opt.get('test_total_points', 0)} | 平均涨幅: {strategy_opt.get('test_avg_rise', 0):.2%}")
            print(f"      • 🛡️ 过拟合检测: {'✅ 通过' if strategy_opt.get('overfitting_passed', False) else '⚠️ 警告'}")
            print(f"      • 🎯 泛化能力: {'✅ 良好' if strategy_opt.get('generalization_passed', False) else '⚠️ 一般'} (比率: {strategy_opt.get('generalization_ratio', 0):.3f})")
        else:
            print(f"      • 🛡️ 过拟合检测: {'✅ 通过' if strategy_opt.get('overfitting_passed', False) else '⚠️ 警告'}")
        
        print(f"\n   🎯 优化后的策略参数:")
        for param_name, param_value in best_params.items():
            if isinstance(param_value, float):
                print(f"      • {param_name}: {param_value:.4f}")
            else:
                print(f"      • {param_name}: {param_value}")
    
    # 2. 模型训练参数
    model_training = optimization_result.get('model_training', {})
    if model_training.get('success'):
        print(f"\n🤖 模型训练参数:")
        print("-"*60)
        print(f"   ✅ 训练状态: 成功")
        print(f"   📊 训练方式: {model_training.get('method', 'unknown')}")
        print(f"   🔢 训练样本数: {model_training.get('train_samples', 0):,}")
        print(f"   📈 特征数量: {model_training.get('feature_count', 0)}")
        
        # 添加模型配置参数
        print(f"\n   🌲 RandomForest模型配置:")
        print(f"      • n_estimators: 150 (决策树数量)")
        print(f"      • max_depth: 12 (最大深度)")
        print(f"      • min_samples_split: 8 (最小分割样本数)")
        print(f"      • min_samples_leaf: 3 (最小叶子节点样本数)")
        print(f"      • class_weight: balanced (类别权重)")
        print(f"      • n_jobs: -1 (并行训练)")
        print(f"      • random_state: 42 (随机种子)")
    
    # 3. AI配置参数
    ai_config = config.get('ai', {})
    print(f"\n⚙️ AI系统配置:")
    print("-"*60)
    print(f"   🔄 模型重训练间隔: {ai_config.get('retrain_interval_days', 30)} 天")
    print(f"   💾 模型复用: {'✅ 启用' if ai_config.get('enable_model_reuse', True) else '❌ 禁用'}")
    print(f"   📊 训练测试分割比例: {ai_config.get('train_test_split_ratio', 0.8):.1%} : {(1-ai_config.get('train_test_split_ratio', 0.8)):.1%}")
    print(f"   📉 数据衰减率: {ai_config.get('data_decay_rate', 0.4):.2f}")
    
    training_data_config = ai_config.get('training_data', {})
    print(f"\n   📅 训练数据配置:")
    print(f"      • 完全训练年数: {training_data_config.get('full_train_years', 6)} 年")
    print(f"      • 优化模式年数: {training_data_config.get('optimize_years', 6)} 年")
    print(f"      • 增量训练年数: {training_data_config.get('incremental_years', 1)} 年")
    
    # 显示三层数据验证配置
    validation_config = ai_config.get('validation', {})
    if validation_config:
        print(f"\n   🎯 严格三层验证配置:")
        print(f"      • 训练集比例: {validation_config.get('train_ratio', 0.65):.1%} (参数优化)")
        print(f"      • 验证集比例: {validation_config.get('validation_ratio', 0.2):.1%} (模型验证)")
        print(f"      • 测试集比例: {validation_config.get('test_ratio', 0.15):.1%} (最终评估)")
    
    # 4. 策略配置参数
    strategy_config = config.get('strategy', {})
    print(f"\n📈 策略系统配置:")
    print("-"*60)
    print(f"   💰 基础涨幅阈值: {strategy_config.get('rise_threshold', 0.04):.2%}")
    print(f"   ⏱️ 最大持有天数: {strategy_config.get('max_days', 20)} 天")
    
    # 技术指标参数
    print(f"\n   📊 技术指标参数:")
    print(f"      • 布林带周期: {strategy_config.get('bb_period', 20)} 天")
    print(f"      • 布林带标准差: {strategy_config.get('bb_std', 2)}")
    print(f"      • RSI周期: {strategy_config.get('rsi_period', 14)} 天")
    print(f"      • MACD快线: {strategy_config.get('macd_fast', 12)} 天")
    print(f"      • MACD慢线: {strategy_config.get('macd_slow', 26)} 天")
    print(f"      • MACD信号线: {strategy_config.get('macd_signal', 9)} 天")
    
    ma_periods = strategy_config.get('ma_periods', [5, 10, 20, 60])
    print(f"      • 移动平均线周期: {', '.join(map(str, ma_periods))} 天")
    
    # 5. 置信度权重参数
    confidence_weights = strategy_config.get('confidence_weights', {})
    if confidence_weights:
        print(f"\n   🎯 置信度权重参数:")
        for weight_name, weight_value in confidence_weights.items():
            if isinstance(weight_value, float):
                if 'threshold' in weight_name:
                    print(f"      • {weight_name}: {weight_value:.3f}")
                else:
                    print(f"      • {weight_name}: {weight_value:.3f}")
            else:
                print(f"      • {weight_name}: {weight_value}")
    
    # 6. 最终性能评估
    evaluation = optimization_result.get('final_evaluation', {})
    if evaluation.get('success'):
        print(f"\n📈 系统性能评估:")
        print("-"*60)
        print(f"   🎯 策略总得分: {evaluation.get('strategy_score', 0):.4f}")
        print(f"   📊 策略成功率: {evaluation.get('strategy_success_rate', 0):.2%}")
        print(f"   🔍 识别低点数: {evaluation.get('identified_points', 0)} 个")
        print(f"   📈 平均涨幅: {evaluation.get('avg_rise', 0):.2%}")
        print(f"   🤖 AI置信度: {evaluation.get('ai_confidence', 0):.4f}")
        print(f"   🎲 AI预测结果: {'📈 相对低点' if evaluation.get('ai_prediction', False) else '📉 非相对低点'}")
    
    # 7. 优化算法配置
    optimization_config = config.get('optimization', {})
    if optimization_config:
        print(f"\n🔬 优化算法配置:")
        print("-"*60)
        print(f"   🔄 全局迭代次数: {optimization_config.get('global_iterations', 500)}")
        print(f"   📈 增量迭代次数: {optimization_config.get('incremental_iterations', 1000)}")
        print(f"   🔧 启用增量优化: {'✅' if optimization_config.get('enable_incremental', True) else '❌'}")
        print(f"   📚 启用历史记录: {'✅' if optimization_config.get('enable_history', True) else '❌'}")
        print(f"   💾 最大历史记录: {optimization_config.get('max_history_records', 100)} 条")
    
    # 8. 数据配置
    data_config = config.get('data', {})
    print(f"\n📊 数据源配置:")
    print("-"*60)
    print(f"   📁 数据文件: {data_config.get('data_file_path', 'unknown')}")
    print(f"   🌐 数据源: {data_config.get('data_source', 'unknown')}")
    print(f"   📈 指数代码: {data_config.get('index_code', 'unknown')}")
    print(f"   ⏰ 数据频率: {data_config.get('frequency', 'unknown')}")
    print(f"   📅 历史数据天数: {data_config.get('history_days', 1000)} 天")
    print(f"   💾 缓存启用: {'✅' if data_config.get('cache_enabled', True) else '❌'}")
    
    print("\n" + "="*80)
    print("🎉 优化参数报告完成！")
    print("💡 提示: 所有参数已保存到配置文件中，可随时查看和调整")
    print("🔬 新特性: 现在使用严格三层数据分割，确保模型泛化能力评估的可靠性")
    print("="*80)

def run_ai_optimization_improved(config):
    """
    运行改进版AI完整优化（包含参数优化 + 模型训练）
    
    参数:
    config: 配置字典
    
    返回:
    bool: 是否成功
    """
    import time
    optimization_start_time = time.time()
    
    print("🚀 启动改进版AI完整优化...")
    print("=" * 80)
    
    try:
        from src.ai.ai_optimizer_improved import AIOptimizerImproved
        from src.data.data_module import DataModule
        from src.strategy.strategy_module import StrategyModule
        from datetime import datetime, timedelta
        
        # 步骤1: 模块初始化
        print("📋 步骤1: 初始化改进版模块...")
        init_start_time = time.time()
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        init_time = time.time() - init_start_time
        print(f"✅ 改进版模块初始化完成 (耗时: {init_time:.2f}s)")
        print("-" * 60)
        
        # 步骤2: 数据准备
        print("📊 步骤2: 准备训练数据...")
        data_start_time = time.time()
        
        training_years = config.get('ai', {}).get('training_data', {}).get('optimize_years', 8)
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=365*training_years)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        
        print(f"   📅 数据时间范围: {start_date} ~ {end_date} ({training_years}年)")
        print("   🔄 获取历史数据...")
        
        raw_data = data_module.get_history_data(start_date, end_date)
        print(f"   📥 获取到 {len(raw_data)} 条原始数据")
        
        print("   ⚙️ 数据预处理中...")
        processed_data = data_module.preprocess_data(raw_data)
        
        data_time = time.time() - data_start_time
        print(f"✅ 数据准备完成 (耗时: {data_time:.2f}s)")
        print(f"   📊 处理后数据: {len(processed_data)} 条记录")
        print("-" * 60)
        
        # 步骤3: 运行完整优化流程
        print("🔧 步骤3: 运行完整优化流程...")
        optimization_result = ai_optimizer.run_complete_optimization(processed_data, strategy_module)
        
        # 计算总耗时
        total_time = time.time() - optimization_start_time
        
        # 输出简要结果
        print("\n" + "="*80)
        print("📊 改进版AI优化结果汇总")
        print("="*80)
        print(f"⏱️ 总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
        
        if optimization_result['success']:
            print("✅ 完整优化成功！")
            
            # 策略优化结果
            strategy_opt = optimization_result.get('strategy_optimization', {})
            if strategy_opt.get('success'):
                print(f"\n🔧 策略参数优化:")
                optimization_method = strategy_opt.get('optimization_method', 'unknown')
                optimization_time = strategy_opt.get('optimization_time', 0)
                print(f"   🔬 优化方法: {optimization_method}")
                print(f"   ⏱️ 优化耗时: {optimization_time:.2f}s ({optimization_time/60:.1f}分钟)")
                print(f"   ✅ 最佳参数: {strategy_opt.get('best_params', {})}")
                print(f"   📊 训练集得分: {strategy_opt.get('best_score', 0):.4f}")
                print(f"   📈 验证集得分: {strategy_opt.get('validation_score', 0):.4f} | 成功率: {strategy_opt.get('validation_success_rate', 0):.2%}")
                if 'test_score' in strategy_opt:
                    print(f"   🔒 测试集得分: {strategy_opt.get('test_score', 0):.4f} | 成功率: {strategy_opt.get('test_success_rate', 0):.2%}")
                    print(f"   🎯 泛化能力: {'✅ 良好' if strategy_opt.get('generalization_passed', False) else '⚠️ 一般'}")
                print(f"   🛡️ 过拟合检测: {'通过' if strategy_opt.get('overfitting_passed', False) else '警告'}")
                
                # 如果使用了遗传算法，显示特殊标识
                if strategy_opt.get('genetic_algorithm_used', False):
                    print("   🧬 使用了高精度遗传算法优化")
            
            # 生成优化报告
            try:
                print(f"\n📊 正在生成优化报告...")
                from src.ai.optimization_reporter import create_optimization_report
                
                # 收集模型信息
                model_training = optimization_result.get('model_training', {})
                model_info = {
                    'model_type': 'RandomForest',
                    'feature_count': model_training.get('feature_count', 'N/A'),
                    'train_samples': model_training.get('train_samples', 'N/A'),
                    'positive_ratio': model_training.get('positive_ratio', 0),
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 15,
                    'min_samples_leaf': 8
                }
                
                # 准备报告数据
                report_data = {
                    'success': True,
                    'method': strategy_opt.get('optimization_method', 'unknown'),
                    'total_time': total_time,
                    'iterations': strategy_opt.get('iterations', 'N/A'),
                    'best_score': strategy_opt.get('best_score', 0),
                    'accuracy': strategy_opt.get('validation_success_rate', 0),
                    'success_rate': strategy_opt.get('validation_success_rate', 0),
                    'avg_rise': optimization_result.get('final_evaluation', {}).get('avg_rise', 0),
                    'best_params': strategy_opt.get('best_params', {}),
                    'training_time_breakdown': model_training.get('training_time_breakdown', {}),
                    'overfitting_detection': strategy_opt
                }
                
                # 生成报告
                report_path = create_optimization_report(
                    optimization_result=report_data,
                    config=config,
                    model_info=model_info,
                    overfitting_detection=strategy_opt
                )
                
                print(f"✅ 优化报告已生成: {report_path}")
                print(f"💡 报告包含: 详细结果、参数配置、过拟合检测、性能图表")
                
            except Exception as e:
                print(f"⚠️ 报告生成失败: {e}")
                import traceback
                print("详细错误信息:")
                print(traceback.format_exc())
            
            # 模型训练结果
            model_training = optimization_result.get('model_training', {})
            if model_training.get('success'):
                print(f"\n🤖 改进版模型训练:")
                training_time = model_training.get('training_time', 0)
                print(f"   ⏱️ 训练耗时: {training_time:.2f}s ({training_time/60:.1f}分钟)")
                print(f"   ✅ 训练状态: 成功")
                print(f"   📊 训练方式: {model_training.get('method', 'unknown')}")
                print(f"   🔢 训练样本数: {model_training.get('train_samples', 0):,}")
                print(f"   📈 特征数量: {model_training.get('feature_count', 0)}")
                print(f"   📊 正样本比例: {model_training.get('positive_ratio', 0):.2%}")
                print(f"   💾 模型保存: {'成功' if model_training.get('save_success', False) else '失败'}")
            
            # 最终评估
            evaluation = optimization_result.get('final_evaluation', {})
            if evaluation.get('success'):
                print(f"\n📈 系统性能评估:")
                print(f"   🎯 策略得分: {evaluation.get('strategy_score', 0):.4f}")
                print(f"   📊 成功率: {evaluation.get('strategy_success_rate', 0):.2%}")
                print(f"   🔍 识别点数: {evaluation.get('identified_points', 0)}")
                print(f"   📈 平均涨幅: {evaluation.get('avg_rise', 0):.2%}")
                print(f"   🤖 AI置信度: {evaluation.get('ai_confidence', 0):.4f}")
            
            # 🔥 调用详细参数打印功能
            print_complete_optimization_results(optimization_result, config)
            
        else:
            print("❌ 优化失败！")
            errors = optimization_result.get('errors', [])
            if errors:
                print("错误信息:")
                for error in errors:
                    print(f"   - {error}")
        
        print("="*80)
        print("🎉 改进版AI优化流程完成！")
        
        # 性能分析
        if optimization_result['success']:
            strategy_opt = optimization_result.get('strategy_optimization', {})
            model_training = optimization_result.get('model_training', {})
            
            init_pct = (init_time / total_time) * 100
            data_pct = (data_time / total_time) * 100
            strategy_pct = (strategy_opt.get('optimization_time', 0) / total_time) * 100
            model_pct = (model_training.get('training_time', 0) / total_time) * 100
            
            print(f"📊 时间分析:")
            print(f"   初始化: {init_time:.2f}s ({init_pct:.1f}%)")
            print(f"   数据准备: {data_time:.2f}s ({data_pct:.1f}%)")
            print(f"   参数优化: {strategy_opt.get('optimization_time', 0):.2f}s ({strategy_pct:.1f}%)")
            print(f"   模型训练: {model_training.get('training_time', 0):.2f}s ({model_pct:.1f}%)")
        
        print("💡 提示: 所有改进功能已启用（改进特征工程、增量学习等）")
        print("=" * 80)
        
        return optimization_result['success']
        
    except ImportError as e:
        print(f"\n❌ 无法导入AI优化模块: {str(e)}")
        return False
    except Exception as e:
        total_time = time.time() - optimization_start_time
        print(f"\n❌ 改进版AI优化过程中发生错误 (已运行 {total_time:.2f}s): {str(e)}")
        import traceback
        traceback.print_exc()
        return False



def main():
    """主函数"""
    print("="*60)
    print("中证500指数相对低点识别系统")
    print("="*60)
    
    # 检查虚拟环境
    check_virtual_environment()
    print()
    
    # 显示环境变量配置提示
    if 'CSI_CONFIG_PATH' in os.environ:
        print(f"🔧 检测到配置文件环境变量: {os.environ['CSI_CONFIG_PATH']}")
    else:
        print("💡 提示: 可通过设置环境变量 CSI_CONFIG_PATH 自定义配置文件路径")
    print()

    parser = argparse.ArgumentParser(
        description='中证500指数相对低点识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run.py b                    # 基础测试
  python run.py fetch                # 数据获取
  python run.py r 2023-01-01 2023-12-31  # 回测
  python run.py s 2023-12-01         # 单日预测
  python run.py ai -m optimize       # AI优化训练（自动生成报告）
  

  
  python run.py report               # 生成最近7天的汇总报告
  python run.py report 14            # 生成最近14天的汇总报告

环境变量配置:
  CSI_CONFIG_PATH=path/to/config.yaml python run.py ai  # 使用自定义配置文件


        """
    )
    
    parser.add_argument('command', choices=['b', 'a', 't', 'all', 'r', 's', 'opt', 'ai', 'fetch', 'report'], 
                       help='命令: b=基础测试, a=AI测试, t=单元测试, r=回测, s=单日预测, opt=策略优化, ai=AI优化/训练, fetch=数据获取, report=生成汇总报告, all=全部')
    parser.add_argument('-v', action='store_true', help='详细输出')
    parser.add_argument('start', nargs='?', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('end', nargs='?', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('-i', '--iter', type=int, default=10, help='迭代次数 (默认: 10)')
    parser.add_argument('-m', '--mode', type=str, help='模式: optimize/incremental/full/demo (AI)')
    parser.add_argument('--no-timer', action='store_true', help='禁用性能计时器')
    
    args = parser.parse_args()

    # 初始化性能计时器
    timer = PerformanceTimer()
    if not args.no_timer:
        timer.start(args.command)

    success = True
    
    # 参数验证
    if args.command == 'r':
        if not args.start or not args.end:
            print('❌ 回测需要指定开始和结束日期: python run.py r 2023-01-01 2023-12-31')
            return 1
        if not validate_date_format(args.start) or not validate_date_format(args.end):
            print('❌ 日期格式错误，请使用 YYYY-MM-DD 格式')
            return 1
    elif args.command == 's':
        if not args.start:
            print('❌ 单日预测需要指定日期: python run.py s 2023-12-01')
            return 1
        if not validate_date_format(args.start):
            print('❌ 日期格式错误，请使用 YYYY-MM-DD 格式')
            return 1
    elif args.command == 'ai':
        mode = args.mode if args.mode else 'optimize'
        if mode not in ['optimize', 'incremental', 'full', 'demo']:
            print('❌ AI模式必须是: optimize, incremental, full, 或 demo')
            print('   例如: python run.py ai -m incremental')
            return 1

    
    # 执行命令
    if args.command == 'fetch':
        fetch_result = run_data_fetch()
        success = (fetch_result.get('code') == 200 if isinstance(fetch_result, dict) else bool(fetch_result))
    elif args.command == 'b':
        success = run_basic_test()
    elif args.command == 'a':
        success = run_ai_test()
    elif args.command == 't':
        success = run_unit_tests()
    elif args.command == 'r':
        success = run_rolling_backtest(args.start, args.end)
    elif args.command == 's':
        success = run_single_day_test(args.start)
    elif args.command == 'opt':
        success = run_strategy_test(args.iter)
    elif args.command == 'ai':
        # AI模式，默认为完整优化
        mode = args.mode if args.mode else 'optimize'
        
        if mode == 'optimize':
            print("🤖 启动改进版AI完整优化...")
            config = load_config_safely()
            if config:
                success = run_ai_optimization_improved(config)
            else:
                success = False
        else:
            print(f"🤖 启动AI训练模式: {mode}...")
            success = run_incremental_training(mode)

    elif args.command == 'report':
        # 生成汇总报告
        print("📊 生成AI优化汇总报告...")
        try:
            from src.ai.optimization_reporter import OptimizationReporter
            config = load_config_safely()
            if config:
                reporter = OptimizationReporter(config)
                days_back = int(args.start) if args.start and args.start.isdigit() else 7
                summary_path = reporter.create_summary_report(days_back)
                if summary_path:
                    print(f"✅ 汇总报告已生成: {summary_path}")
                    success = True
                else:
                    print("❌ 没有找到报告数据")
                    success = False
            else:
                success = False
        except Exception as e:
            print(f"❌ 汇总报告生成失败: {e}")
            success = False
    elif args.command == 'all':
        print("\n1. 运行数据获取...")
        fetch_result = run_data_fetch()
        success &= (fetch_result.get('code') == 200 if isinstance(fetch_result, dict) else bool(fetch_result))
        
        print("\n2. 运行基础测试...")
        success &= run_basic_test()
        
        print("\n3. 运行AI优化测试...")
        success &= run_ai_test()
        
        print("\n4. 运行单元测试...")
        success &= run_unit_tests()

        if args.start and args.end:
            print("\n5. 运行回测...")
            success &= run_rolling_backtest(args.start, args.end)

        if args.start:
            print("\n6. 运行单日预测...")
            success &= run_single_day_test(args.start)

        print("\n7. 运行策略优化...")
        success &= run_strategy_test(args.iter)

    # 停止性能计时器
    if not args.no_timer:
        execution_time = timer.stop()
    
    print("\n" + "="*60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败！")
    print("="*60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

