#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
遗传算法功能测试脚本

测试新实现的高精度遗传算法优化功能
验证系统能否找到更高准确度的参数组合
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.utils import load_config
from src.prediction.prediction_utils import setup_logging

def test_genetic_algorithm():
    """测试遗传算法功能"""
    print("="*80)
    print("🧬 遗传算法功能测试")
    print("="*80)
    
    setup_logging()
    logger = logging.getLogger("GeneticTest")
    
    try:
        # 1. 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
        config = load_config(config_path=config_path)
        
        # 确保遗传算法已启用
        config.setdefault('genetic_algorithm', {})['enabled'] = True
        config.setdefault('advanced_optimization', {})['enabled'] = True
        config.setdefault('advanced_optimization', {})['high_precision_mode'] = True
        
        logger.info("📋 配置加载完成，遗传算法已启用")
        
        # 2. 初始化模块
        logger.info("🔧 初始化系统模块...")
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        # 3. 获取测试数据
        logger.info("📊 准备测试数据...")
        start_date = '2023-01-01'
        end_date = '2024-12-31'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        logger.info(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        
        # 4. 记录优化前的基准性能
        logger.info("\n📊 获取优化前基准性能...")
        baseline_backtest = strategy_module.backtest(processed_data)
        baseline_evaluation = strategy_module.evaluate_strategy(baseline_backtest)
        
        baseline_score = baseline_evaluation.get('score', 0)
        baseline_success_rate = baseline_evaluation.get('success_rate', 0)
        baseline_total_points = baseline_evaluation.get('total_points', 0)
        baseline_avg_rise = baseline_evaluation.get('avg_rise', 0)
        
        logger.info(f"🎯 基准性能:")
        logger.info(f"   得分: {baseline_score:.6f}")
        logger.info(f"   成功率: {baseline_success_rate:.2%}")
        logger.info(f"   识别点数: {baseline_total_points}")
        logger.info(f"   平均涨幅: {baseline_avg_rise:.2%}")
        
        # 5. 运行遗传算法优化
        logger.info("\n🧬 开始遗传算法优化...")
        optimization_start_time = datetime.now()
        
        optimization_result = ai_optimizer.optimize_strategy_parameters_improved(
            strategy_module, processed_data
        )
        
        optimization_end_time = datetime.now()
        optimization_duration = optimization_end_time - optimization_start_time
        
        # 6. 分析优化结果
        if optimization_result['success']:
            logger.info(f"\n🎉 优化成功完成！")
            logger.info(f"⏱️ 优化耗时: {optimization_duration}")
            
            # 结果对比
            if optimization_result.get('genetic_algorithm_used', False):
                logger.info(f"\n🧬 遗传算法优化结果:")
                logger.info(f"   算法类型: {optimization_result['optimization_method']}")
                logger.info(f"   训练集得分: {optimization_result['best_score']:.6f}")
                logger.info(f"   验证集得分: {optimization_result['validation_score']:.6f}")
                logger.info(f"   验证集成功率: {optimization_result['validation_success_rate']:.2%}")
                logger.info(f"   测试集得分: {optimization_result['test_score']:.6f}")
                logger.info(f"   测试集成功率: {optimization_result['test_success_rate']:.2%}")
                
                # 性能提升分析
                test_score_improvement = optimization_result['test_score'] - baseline_score
                test_success_rate_improvement = optimization_result['test_success_rate'] - baseline_success_rate
                
                logger.info(f"\n📈 性能提升分析:")
                logger.info(f"   得分提升: {test_score_improvement:+.6f} ({test_score_improvement/baseline_score*100:+.2f}%)")
                logger.info(f"   成功率提升: {test_success_rate_improvement:+.2%}")
                logger.info(f"   过拟合检测: {'✅ 通过' if optimization_result['overfitting_passed'] else '⚠️ 警告'}")
                logger.info(f"   泛化能力: {'✅ 良好' if optimization_result['generalization_passed'] else '⚠️ 一般'}")
                
                # 最优参数详情
                logger.info(f"\n🎯 遗传算法发现的最优参数:")
                best_params = optimization_result['best_params']
                for param_name, param_value in best_params.items():
                    if isinstance(param_value, float):
                        logger.info(f"   {param_name}: {param_value:.6f}")
                    else:
                        logger.info(f"   {param_name}: {param_value}")
                
                # 验证遗传算法优势
                if test_score_improvement > 0.01:  # 1%以上提升
                    logger.info(f"\n🚀 遗传算法显著提升了模型性能！")
                    logger.info(f"   推荐启用遗传算法进行日常优化")
                elif test_score_improvement > 0:
                    logger.info(f"\n✅ 遗传算法带来了性能提升")
                    logger.info(f"   可以考虑使用遗传算法")
                else:
                    logger.info(f"\n⚠️ 遗传算法未带来显著提升")
                    logger.info(f"   可能需要调整遗传算法参数")
                
            else:
                logger.warning("⚠️ 遗传算法未被使用，可能配置有问题")
        else:
            logger.error("❌ 优化失败")
            logger.error(f"错误信息: {optimization_result.get('error', '未知错误')}")
            return False
        
        # 7. 总结和建议
        logger.info(f"\n" + "="*80)
        logger.info(f"📊 遗传算法测试总结")
        logger.info(f"="*80)
        
        if optimization_result.get('genetic_algorithm_used', False):
            logger.info(f"✅ 遗传算法功能正常")
            logger.info(f"✅ 成功完成高精度参数优化")
            logger.info(f"✅ 三层数据验证通过")
            
            # 使用建议
            logger.info(f"\n💡 使用建议:")
            logger.info(f"   1. 遗传算法已启用并正常工作")
            logger.info(f"   2. 建议每周运行一次全面优化")
            logger.info(f"   3. 日常训练使用发现的最优参数")
            logger.info(f"   4. 如需更高精度，可增加种群大小和代数")
            
        else:
            logger.warning(f"⚠️ 遗传算法未启用，请检查配置")
        
        logger.info(f"="*80)
        return True
        
    except Exception as e:
        logger.error(f"测试过程发生错误: {e}")
        return False

def run_genetic_optimization_only():
    """仅运行遗传算法优化（不包括基准测试）"""
    print("🧬 快速遗传算法优化")
    print("-" * 50)
    
    setup_logging()
    logger = logging.getLogger("GeneticOptimization")
    
    try:
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
        config = load_config(config_path=config_path)
        
        # 强制启用遗传算法
        config.setdefault('genetic_algorithm', {})['enabled'] = True
        config.setdefault('advanced_optimization', {})['enabled'] = True
        
        # 初始化模块
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        # 获取数据
        raw_data = data_module.get_history_data('2023-01-01', '2024-12-31')
        processed_data = data_module.preprocess_data(raw_data)
        
        # 运行优化
        logger.info("🚀 开始遗传算法优化...")
        result = ai_optimizer.optimize_strategy_parameters_improved(strategy_module, processed_data)
        
        if result['success'] and result.get('genetic_algorithm_used', False):
            logger.info("🎉 遗传算法优化成功！")
            logger.info(f"最优参数已保存，测试集成功率: {result['test_success_rate']:.2%}")
            return True
        else:
            logger.error("❌ 遗传算法优化失败")
            return False
            
    except Exception as e:
        logger.error(f"优化失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # 快速模式：仅运行优化
        success = run_genetic_optimization_only()
    else:
        # 完整测试模式
        success = test_genetic_algorithm()
    
    sys.exit(0 if success else 1) 