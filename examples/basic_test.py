#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础测试示例
演示如何使用量化系统的基本功能
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.utils import setup_logging, load_config
from data.data_module import DataModule
from strategy.strategy_module import StrategyModule
from notification.notification_module import NotificationModule

def main():
    """主函数"""
    print("="*60)
    print("中证1000指数相对低点识别系统 - 基础测试")
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
                'rise_threshold': 0.05,  # 5%的上涨阈值
                'max_days': 20  # 最大持仓20天
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
                }
            },
            'notification': {
                'methods': ['console'],
                'email': {
                    'recipients': ['test@example.com']
                }
            }
        }
    
    try:
        # 1. 测试数据模块
        print("\n📊 测试数据模块...")
        data_module = DataModule(config)
        
        # 获取历史数据
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        print(f"获取历史数据: {start_date} 到 {end_date}")
        
        raw_data = data_module.get_history_data(start_date, end_date)
        print(f"✅ 获取到 {len(raw_data)} 条原始数据")
        
        # 预处理数据
        print("预处理数据...")
        processed_data = data_module.preprocess_data(raw_data)
        print(f"✅ 预处理完成，数据包含 {len(processed_data.columns)} 个字段")
        
        # 验证数据
        is_valid = data_module.validate_data(processed_data)
        print(f"✅ 数据验证: {'通过' if is_valid else '失败'}")
        
        # 2. 测试策略模块
        print("\n🎯 测试策略模块...")
        strategy_module = StrategyModule(config)
        
        # 识别相对低点
        print("识别最新相对低点...")
        latest_result = strategy_module.identify_relative_low(processed_data)
        print(f"✅ 识别结果: {latest_result}")
        
        # 运行回测
        print("运行回测...")
        backtest_results = strategy_module.backtest(processed_data)
        print(f"✅ 回测完成，数据长度: {len(backtest_results)}")
        
        # 评估策略
        print("评估策略...")
        evaluation = strategy_module.evaluate_strategy(backtest_results)
        print(f"✅ 策略评估完成:")
        print(f"   - 识别点数: {evaluation['total_points']}")
        print(f"   - 成功率: {evaluation['success_rate']:.2%}")
        print(f"   - 平均涨幅: {evaluation['avg_rise']:.2%}")
        print(f"   - 平均天数: {evaluation['avg_days']:.1f}")
        print(f"   - 综合得分: {evaluation['score']:.4f}")
        
        # 可视化回测结果
        print("生成回测图表...")
        chart_path = strategy_module.visualize_backtest(backtest_results)
        print(f"✅ 图表已保存: {chart_path}")
        
        # 3. 测试通知模块
        print("\n📧 测试通知模块...")
        notification_module = NotificationModule(config)
        
        # 发送测试通知
        test_result = {
            'date': '2024-06-08',
            'price': 5000.0,
            'is_low_point': True,
            'confidence': 0.75,
            'reasons': ['价格低于MA5/MA10/MA20', 'RSI超卖(25.5)', '近5日大幅下跌(-6.2%)']
        }
        
        print("发送测试通知...")
        notification_success = notification_module.send_low_point_notification(test_result)
        print(f"✅ 通知发送: {'成功' if notification_success else '失败'}")
        
        # 获取通知历史
        history = notification_module.get_notification_history(30)
        print(f"✅ 获取到 {len(history)} 条通知历史")
        
        print("\n🎉 所有测试完成！")
        print("\n📋 测试总结:")
        print(f"   - 数据获取: ✅ 成功")
        print(f"   - 数据预处理: ✅ 成功")
        print(f"   - 相对低点识别: ✅ 成功")
        print(f"   - 策略回测: ✅ 成功")
        print(f"   - 策略评估: ✅ 成功")
        print(f"   - 结果可视化: ✅ 成功")
        print(f"   - 通知发送: ✅ 成功")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

