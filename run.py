#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速运行脚本
提供简单的命令行界面来运行系统的各种功能
"""

import sys
import os
import argparse


# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_basic_test():
    """运行基础测试"""
    print("运行基础测试...")
    from examples.basic_test import main
    return main()

def run_ai_test():
    """运行AI优化测试"""
    print("运行AI优化测试...")
    from examples.ai_optimization_test import main
    return main()

def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    import unittest
    
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_rolling_backtest(start_date, end_date):
    from examples.run_rolling_backtest import run_rolling_backtest as rolling_func
    return rolling_func(start_date, end_date)

def run_single_day_test(predict_date):
    from examples.predict_single_day import predict_single_day
    return predict_single_day(predict_date)

def run_strategy_test(iterations):
    from examples.llm_strategy_optimizer import LLMStrategyOptimizer
    from src.utils.utils import load_config
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)
    optimizer = LLMStrategyOptimizer(config)
    return optimizer.optimize_strategy(num_iterations=iterations)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中证1000指数相对低点识别系统')
    parser.add_argument('command', choices=['b', 'a', 't', 'all', 'r', 's', 'opt'], 
                       help='命令: b=基础测试, a=AI测试, t=单元测试, r=回测, s=单日预测, opt=策略优化, all=全部')
    parser.add_argument('-v', action='store_true', help='详细输出')
    parser.add_argument('start', nargs='?', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('end', nargs='?', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('-i', '--iter', type=int, default=10, help='迭代次数 (默认: 10)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("中证1000指数相对低点识别系统")
    print("="*60)
    
    success = True
    
    if args.command == 'b':
        success = run_basic_test()
    elif args.command == 'a':
        success = run_ai_test()
    elif args.command == 't':
        success = run_unit_tests()
    elif args.command == 'r':
        if not args.start or not args.end:
            print('回测需要指定开始和结束日期: python run.py r 2023-01-01 2023-12-31')
            return 1
        success = run_rolling_backtest(args.start, args.end)
    elif args.command == 's':
        if not args.start:
            print('单日预测需要指定日期: python run.py s 2023-12-01')
            return 1
        success = run_single_day_test(args.start)
    elif args.command == 'opt':
        success = run_strategy_test(args.iter)
    elif args.command == 'all':
        print("\n1. 运行基础测试...")
        success &= run_basic_test()
        
        print("\n2. 运行AI优化测试...")
        success &= run_ai_test()
        
        print("\n3. 运行单元测试...")
        success &= run_unit_tests()

        if args.start and args.end:
            print("\n4. 运行回测...")
            success &= run_rolling_backtest(args.start, args.end)

        if args.start:
            print("\n5. 运行单日预测...")
            success &= run_single_day_test(args.start)

        print("\n6. 运行策略优化...")
        success &= run_strategy_test(args.iter)

    print("\n" + "="*60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败！")
    print("="*60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

