#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进版AI优化器训练脚本
提供完整训练、增量训练和模型验证功能
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.utils import load_config, setup_logging


def full_train(config, start_date: str = None, end_date: str = None):
    """
    完整训练模式
    
    参数:
    config: 配置对象
    start_date: 训练开始日期 (YYYY-MM-DD)
    end_date: 训练结束日期 (YYYY-MM-DD)
    """
    print("🔥 启动完整训练模式")
    print("="*60)
    
    # 设置默认日期
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=800)  # 默认使用800天历史数据
        start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"📅 训练数据范围: {start_date} 至 {end_date}")
    
    # 初始化模块
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    ai_improved = AIOptimizerImproved(config)
    
    # 获取训练数据
    print("📊 正在获取训练数据...")
    training_data = data_module.get_history_data(start_date, end_date)
    
    if training_data.empty:
        print("❌ 无法获取训练数据，请检查日期范围和数据文件")
        return False
    
    # 预处理数据
    training_data = data_module.preprocess_data(training_data)
    print(f"✅ 获取到 {len(training_data)} 条训练样本")
    
    # 开始训练
    print("\n🤖 开始AI模型训练...")
    print("⏳ 预计耗时: 2-5分钟（取决于数据量和硬件配置）")
    
    train_result = ai_improved.full_train(training_data, strategy_module)
    
    if train_result['success']:
        print("✅ 模型训练成功！")
        print(f"📊 训练样本数: {train_result.get('train_samples', 'N/A')}")
        print(f"🔧 特征数量: {train_result.get('feature_count', 'N/A')}")
        print(f"💾 模型已保存到 models/ 目录")
        
        # 验证训练效果
        print("\n🧪 开始训练效果验证...")
        validate_model(ai_improved, training_data)
        
        return True
    else:
        print(f"❌ 模型训练失败: {train_result.get('error', '未知错误')}")
        return False


def incremental_train(config, new_data_date: str = None):
    """
    增量训练模式
    
    参数:
    config: 配置对象  
    new_data_date: 新数据日期 (YYYY-MM-DD)
    """
    print("📈 启动增量训练模式")
    print("="*60)
    
    if not new_data_date:
        new_data_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📅 新数据日期: {new_data_date}")
    
    # 初始化模块
    data_module = DataModule(config)
    strategy_module = StrategyModule(config)
    ai_improved = AIOptimizerImproved(config)
    
    # 检查是否存在已训练模型
    if not ai_improved._load_model():
        print("⚠️  未找到已训练的模型，将执行完整训练")
        end_dt = datetime.strptime(new_data_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=800)
        return full_train(config, start_dt.strftime('%Y-%m-%d'), new_data_date)
    
    print("✅ 加载已训练模型成功")
    
    # 获取增量数据（最近30天，包含新数据）
    end_dt = datetime.strptime(new_data_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=30)
    
    print("📊 正在获取增量数据...")
    incremental_data = data_module.get_history_data(
        start_dt.strftime('%Y-%m-%d'), 
        new_data_date
    )
    
    if incremental_data.empty:
        print("❌ 无法获取增量数据")
        return False
    
    # 预处理数据
    incremental_data = data_module.preprocess_data(incremental_data)
    print(f"✅ 获取到 {len(incremental_data)} 条增量样本")
    
    # 开始增量训练
    print("\n🔄 开始增量训练...")
    train_result = ai_improved.incremental_train(incremental_data, strategy_module)
    
    if train_result['success']:
        method = train_result.get('method', 'unknown')
        if method == 'incremental':
            print("✅ 增量训练成功！")
            print(f"🔧 更新次数: {train_result.get('update_count', 'N/A')}")
            print(f"📊 新增样本: {train_result.get('new_samples', 'N/A')}")
        elif method == 'full_retrain':
            print("✅ 自动触发完整重训练成功！")
            print(f"📊 训练样本数: {train_result.get('train_samples', 'N/A')}")
        
        print(f"💾 模型已更新保存")
        return True
    else:
        print(f"❌ 增量训练失败: {train_result.get('error', '未知错误')}")
        return False


def validate_model(ai_improved, test_data):
    """
    验证模型效果
    
    参数:
    ai_improved: AI优化器实例
    test_data: 测试数据
    """
    print("\n🧪 模型效果验证")
    print("-" * 40)
    
    # 选择最近几天进行测试
    if len(test_data) < 5:
        print("⚠️  数据量不足，跳过验证")
        return
    
    test_samples = test_data.tail(5)  # 最近5天
    results = []
    
    for idx, (date_idx, row) in enumerate(test_samples.iterrows()):
        # 使用到当前日期为止的数据进行预测
        current_data = test_data.iloc[:date_idx+1]
        
        if len(current_data) < 2:
            continue
            
        pred_result = ai_improved.predict_low_point(
            current_data.tail(1), 
            row['date'] if 'date' in row else str(date_idx)
        )
        
        results.append({
            'date': row.get('date', f'Day-{date_idx}'),
            'confidence': pred_result.get('confidence', 0),
            'smoothed_confidence': pred_result.get('smoothed_confidence', 0),
            'prediction': pred_result.get('is_low_point', False)
        })
    
    # 输出验证结果
    print("日期\t\t原始置信度\t平滑置信度\t预测结果")
    print("-" * 50)
    
    for result in results:
        pred_str = "低点" if result['prediction'] else "非低点"
        print(f"{result['date']}\t{result['confidence']:.4f}\t\t"
              f"{result['smoothed_confidence']:.4f}\t\t{pred_str}")
    
    # 分析置信度稳定性
    if len(results) >= 2:
        changes = []
        for i in range(1, len(results)):
            change = abs(results[i]['smoothed_confidence'] - results[i-1]['smoothed_confidence'])
            changes.append(change)
        
        avg_change = sum(changes) / len(changes) if changes else 0
        max_change = max(changes) if changes else 0
        
        print(f"\n📊 置信度稳定性:")
        print(f"   平均变化: {avg_change:.4f}")
        print(f"   最大变化: {max_change:.4f}")
        
        if max_change < 0.25:
            print("   ✅ 置信度变化稳定")
        elif max_change < 0.35:
            print("   ⚠️  置信度变化适中")
        else:
            print("   ❌ 置信度变化较大，建议检查配置")


def predict_demo(config, target_date: str = None):
    """
    预测演示
    
    参数:
    config: 配置对象
    target_date: 目标预测日期
    """
    print("🔮 预测演示模式")
    print("="*60)
    
    if not target_date:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"🎯 预测日期: {target_date}")
    
    # 初始化模块
    data_module = DataModule(config)
    ai_improved = AIOptimizerImproved(config)
    
    # 加载模型
    if not ai_improved._load_model():
        print("❌ 未找到已训练模型，请先运行训练")
        print("💡 运行命令: python examples/train_improved_ai.py --mode full")
        return False
    
    print("✅ 模型加载成功")
    
    # 获取预测数据
    end_dt = datetime.strptime(target_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=100)  # 使用最近100天数据
    
    prediction_data = data_module.get_history_data(
        start_dt.strftime('%Y-%m-%d'),
        target_date
    )
    
    if prediction_data.empty:
        print(f"❌ 无法获取 {target_date} 的数据")
        return False
    
    # 预处理数据
    prediction_data = data_module.preprocess_data(prediction_data)
    
    # 进行预测
    print("\n🔮 开始预测...")
    pred_result = ai_improved.predict_low_point(prediction_data, target_date)
    
    # 输出结果
    print("\n" + "="*50)
    print("📈 预测结果")
    print("="*50)
    print(f"📅 预测日期: {target_date}")
    print(f"🎯 预测结果: {'✅ 相对低点' if pred_result['is_low_point'] else '❌ 非相对低点'}")
    print(f"📊 原始置信度: {pred_result['confidence']:.4f}")
    print(f"📊 平滑置信度: {pred_result['smoothed_confidence']:.4f}")
    
    # 置信度级别评估
    confidence = pred_result['smoothed_confidence']
    if confidence >= 0.8:
        level = "🔥 极高置信度"
    elif confidence >= 0.6:
        level = "🟢 高置信度"
    elif confidence >= 0.4:
        level = "🟡 中等置信度"
    else:
        level = "🔴 低置信度"
    
    print(f"📈 置信度级别: {level}")
    print("="*50)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进版AI优化器训练脚本')
    parser.add_argument('--mode', choices=['full', 'incremental', 'demo'], 
                       default='full', help='训练模式')
    parser.add_argument('--start', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--date', help='目标日期 (YYYY-MM-DD)')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    # 加载配置
    if args.config:
        config_path = args.config
    else:
        # 优先使用改进版配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    
    try:
        config = load_config(config_path=config_path)
        print(f"✅ 加载配置文件: {config_path}")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return 1
    
    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 运行在虚拟环境中")
    else:
        print("⚠️  建议在虚拟环境中运行")
    
    # 执行相应模式
    success = False
    
    try:
        if args.mode == 'full':
            success = full_train(config, args.start, args.end)
        elif args.mode == 'incremental':
            success = incremental_train(config, args.date)
        elif args.mode == 'demo':
            success = predict_demo(config, args.date)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    if success:
        print("\n🎉 执行成功完成！")
        return 0
    else:
        print("\n💥 执行失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 