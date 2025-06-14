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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中证1000指数相对低点识别系统')
    parser.add_argument('command', choices=['basic', 'ai', 'test', 'all'], 
                       help='要运行的命令')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出')
    
    args = parser.parse_args()
    
    print("="*60)
    print("中证1000指数相对低点识别系统")
    print("="*60)
    
    success = True
    
    if args.command == 'basic':
        success = run_basic_test()
    elif args.command == 'ai':
        success = run_ai_test()
    elif args.command == 'test':
        success = run_unit_tests()
    elif args.command == 'all':
        print("\n1. 运行基础测试...")
        success &= run_basic_test()
        
        print("\n2. 运行AI优化测试...")
        success &= run_ai_test()
        
        print("\n3. 运行单元测试...")
        success &= run_unit_tests()
    
    print("\n" + "="*60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败！")
    print("="*60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

