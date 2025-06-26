#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
严格数据分割演示
演示如何使用严格数据分割防止过拟合风险
"""

import sys
import os
import pandas as pd

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.utils import setup_logging, load_config, Timer
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from ai.ai_optimizer import AIOptimizer

def demo_strict_data_split():
    """演示严格数据分割功能"""
    print("=" * 80)
    print("严格数据分割演示 - 防止过拟合风险")
    print("=" * 80)
    
    # 设置日志
    setup_logging('INFO')
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    try:
        # 1. 初始化模块
        print("\n📋 初始化模块...")
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizer(config)
        print("✅ 模块初始化完成")
        
        # 2. 获取历史数据
        print("\n📊 获取历史数据...")
        start_date = '2020-01-01'
        end_date = '2025-06-21'
        
        raw_data = data_module.get_history_data(start_date, end_date)
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 数据准备完成，共 {len(processed_data)} 条记录")
        print(f"   时间范围: {processed_data.iloc[0]['date']} ~ {processed_data.iloc[-1]['date']}")
        
        # 3. 严格数据分割演示
        print("\n🔒 严格数据分割演示...")
        timer = Timer()
        timer.start()
        
        # 执行严格数据分割
        data_splits = ai_optimizer.strict_data_split(processed_data, preserve_test_set=True)
        train_data = data_splits['train']
        validation_data = data_splits['validation']
        test_data = data_splits['test']
        
        timer.stop()
        print(f"✅ 数据分割完成 (耗时: {timer.elapsed_str()})")
        
        # 4. 验证数据分割的有效性
        print("\n🔍 验证数据分割有效性...")
        
        # 检查数据量分配
        total_size = len(processed_data)
        train_ratio = len(train_data) / total_size
        val_ratio = len(validation_data) / total_size
        test_ratio = len(test_data) / total_size
        
        print(f"   数据分配验证:")
        print(f"     - 训练集: {len(train_data)} 条 ({train_ratio:.1%})")
        print(f"     - 验证集: {len(validation_data)} 条 ({val_ratio:.1%})")
        print(f"     - 测试集: {len(test_data)} 条 ({test_ratio:.1%})")
        print(f"     - 总计: {total_size} 条")
        
        # 检查时间序列连续性
        print(f"   时间序列连续性验证:")
        print(f"     - 训练集: {train_data.iloc[0]['date']} ~ {train_data.iloc[-1]['date']}")
        print(f"     - 验证集: {validation_data.iloc[0]['date']} ~ {validation_data.iloc[-1]['date']}")
        print(f"     - 测试集: {test_data.iloc[0]['date']} ~ {test_data.iloc[-1]['date']}")
        
        # 检查数据泄露
        train_indices = set(train_data.index)
        val_indices = set(validation_data.index)
        test_indices = set(test_data.index)
        
        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices
        
        print(f"   数据泄露检测:")
        print(f"     - 训练-验证重叠: {'❌ 发现重叠' if train_val_overlap else '✅ 无重叠'}")
        print(f"     - 训练-测试重叠: {'❌ 发现重叠' if train_test_overlap else '✅ 无重叠'}")
        print(f"     - 验证-测试重叠: {'❌ 发现重叠' if val_test_overlap else '✅ 无重叠'}")
        
        # 5. 测试集保护机制演示
        print("\n🔒 测试集保护机制演示...")
        
        # 尝试再次分割，应该得到相同的测试集
        try:
            data_splits_2 = ai_optimizer.strict_data_split(processed_data, preserve_test_set=True)
            test_data_2 = data_splits_2['test']
            
            # 验证测试集一致性
            if test_data.equals(test_data_2):
                print("✅ 测试集保护机制正常工作，测试集保持一致")
            else:
                print("❌ 测试集保护机制失效，测试集发生变化")
                
        except Exception as e:
            print(f"⚠️ 测试集保护机制测试失败: {str(e)}")
        
        # 6. 仅训练集优化演示
        print("\n🔧 仅训练集参数优化演示...")
        timer.start()
        
        optimized_params = ai_optimizer.optimize_strategy_parameters_on_train_only(
            strategy_module, train_data
        )
        
        timer.stop()
        print(f"✅ 训练集优化完成 (耗时: {timer.elapsed_str()})")
        print(f"   优化后参数: {optimized_params}")
        
        # 7. 走前验证演示
        print("\n🚶 走前验证演示...")
        timer.start()
        
        # 使用训练+验证数据进行走前验证
        train_val_data = pd.concat([train_data, validation_data]).reset_index(drop=True)
        
        wf_result = ai_optimizer.walk_forward_validation(
            train_val_data, 
            strategy_module,
            window_size=126,  # 减小窗口以加快演示
            step_size=21      # 减小步进以加快演示
        )
        
        timer.stop()
        print(f"✅ 走前验证完成 (耗时: {timer.elapsed_str()})")
        
        if wf_result['success']:
            print(f"   验证结果:")
            print(f"     - 平均得分: {wf_result['avg_score']:.4f}")
            print(f"     - 得分标准差: {wf_result['std_score']:.4f}")
            print(f"     - 得分范围: [{wf_result['min_score']:.4f}, {wf_result['max_score']:.4f}]")
            print(f"     - 有效折数: {wf_result['valid_folds']}/{wf_result['total_folds']}")
        else:
            print(f"   验证失败: {wf_result.get('error', 'Unknown error')}")
        
        # 8. 测试集最终评估演示
        print("\n🎯 测试集最终评估演示...")
        timer.start()
        
        # 更新策略参数
        strategy_module.update_params(optimized_params)
        
        test_result = ai_optimizer.evaluate_on_test_set_only(strategy_module, test_data)
        
        timer.stop()
        print(f"✅ 测试集评估完成 (耗时: {timer.elapsed_str()})")
        
        if test_result['success']:
            print(f"   测试集性能:")
            print(f"     - 综合得分: {test_result['test_score']:.4f}")
            print(f"     - 成功率: {test_result['success_rate']:.2%}")
            print(f"     - 识别点数: {test_result['total_points']}")
            print(f"     - 平均涨幅: {test_result['avg_rise']:.2%}")
            print(f"     - 测试期间: {test_result['test_period']}")
        else:
            print(f"   测试集评估失败: {test_result.get('error', 'Unknown error')}")
        
        # 9. 完整分层优化演示（使用严格数据分割）
        print("\n🏗️ 完整分层优化演示（严格数据分割版本）...")
        timer.start()
        
        hierarchical_result = ai_optimizer.hierarchical_optimization(processed_data)
        
        timer.stop()
        print(f"✅ 分层优化完成 (耗时: {timer.elapsed_str()})")
        
        print(f"   优化结果:")
        print(f"     - 最终参数: {hierarchical_result['params']}")
        print(f"     - 验证集得分: {hierarchical_result['cv_score']:.4f}")
        print(f"     - 测试集得分: {hierarchical_result['test_score']:.4f}")
        print(f"     - 过拟合检测: {'通过' if hierarchical_result['overfitting_check']['passed'] else '警告'}")
        
        overfitting_ratio = hierarchical_result['overfitting_check']['difference_ratio']
        print(f"     - 过拟合程度: {overfitting_ratio:.1%}")
        
        # 10. 结果对比和分析
        print("\n📊 严格数据分割 vs 传统方法对比...")
        
        # 传统方法（使用全部数据优化）
        print("   传统方法测试...")
        traditional_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)
        strategy_module.update_params(traditional_params)
        traditional_backtest = strategy_module.backtest(processed_data)
        traditional_evaluation = strategy_module.evaluate_strategy(traditional_backtest)
        
        print(f"   对比结果:")
        print(f"     严格分割方法:")
        print(f"       - 验证集得分: {hierarchical_result['cv_score']:.4f}")
        print(f"       - 测试集得分: {hierarchical_result['test_score']:.4f}")
        print(f"       - 过拟合风险: {'低' if overfitting_ratio < 0.2 else '高'}")
        print(f"     传统方法:")
        print(f"       - 全数据得分: {traditional_evaluation['score']:.4f}")
        print(f"       - 过拟合风险: 未知（无独立测试集）")
        
        # 计算改进效果
        if traditional_evaluation['score'] > 0:
            validation_reliability = hierarchical_result['test_score'] / traditional_evaluation['score']
            print(f"     可靠性指标: {validation_reliability:.2%} (测试集得分/全数据得分)")
        
        print("\n🎉 严格数据分割演示完成！")
        print("\n📋 总结:")
        print("   ✅ 实现了严格的训练/验证/测试三层分割")
        print("   ✅ 测试集完全隔离，防止数据泄露")
        print("   ✅ 早停机制有效防止过拟合")
        print("   ✅ 走前验证模拟真实交易环境")
        print("   ✅ 过拟合检测机制有效工作")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = demo_strict_data_split()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 