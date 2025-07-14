#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代码修复验证测试
验证所有在code review中发现和修复的问题

测试内容：
- 线程安全性
- 参数验证
- 错误处理
- 内存管理
- 数值稳定性
"""

import sys
import threading
import time
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# 添加src目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from src.utils.common import (
    LoggerManager, DataValidator, MathUtils, 
    PerformanceMonitor, safe_execute
)
from src.utils.command_processor import CommandProcessor
from src.utils.config_loader import load_config, deep_merge_dict


class TestCodeReviewFixes(unittest.TestCase):
    """代码修复验证测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'close': [100.0, 101.0, 99.0],
            'volume': [1000, 1100, 900]
        })
    
    def test_logger_thread_safety(self):
        """测试日志管理器的线程安全性"""
        results = []
        
        def worker():
            """工作线程函数"""
            logger = LoggerManager.get_logger(f"test_worker_{threading.current_thread().ident}")
            results.append(logger is not None)
        
        # 创建多个线程同时获取日志记录器
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有线程都成功获取了日志记录器
        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))
    
    def test_data_validator_improvements(self):
        """测试数据验证器的改进"""
        # 测试空DataFrame检查
        empty_df = pd.DataFrame()
        valid, errors = DataValidator.validate_dataframe(empty_df, ['test'])
        self.assertFalse(valid)
        self.assertIn("DataFrame为空", errors)
        
        # 测试None DataFrame检查
        valid, errors = DataValidator.validate_dataframe(None, ['test'])
        self.assertFalse(valid)
        self.assertIn("DataFrame为None", errors)
        
        # 测试多种类型的空值检查
        df_with_nulls = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': ['a', '', 'c'],
            'col3': [1.0, 2.0, 3.0]
        })
        
        valid, errors = DataValidator.validate_dataframe(
            df_with_nulls, ['col1', 'col2'], allow_null=False
        )
        self.assertFalse(valid)
        self.assertTrue(len(errors) >= 2)  # col1和col2都有问题
    
    def test_date_validation_improvements(self):
        """测试日期验证的改进"""
        # 测试有效日期
        self.assertTrue(DataValidator.validate_date_format('2023-01-01'))
        
        # 测试无效日期格式
        self.assertFalse(DataValidator.validate_date_format('2023/01/01'))
        self.assertFalse(DataValidator.validate_date_format('invalid'))
        self.assertFalse(DataValidator.validate_date_format(None))
        self.assertFalse(DataValidator.validate_date_format(123))
        
        # 测试日期范围验证
        valid, error = DataValidator.validate_date_range('2023-01-01', '2023-12-31')
        self.assertTrue(valid)
        self.assertEqual(error, "")
        
        # 测试无效日期范围
        valid, error = DataValidator.validate_date_range('2023-12-31', '2023-01-01')
        self.assertFalse(valid)
        self.assertIn("开始日期必须早于结束日期", error)
    
    def test_math_utils_safety(self):
        """测试数学工具的安全性"""
        # 测试安全除法
        self.assertEqual(MathUtils.safe_divide(10, 2), 5.0)
        self.assertEqual(MathUtils.safe_divide(10, 0), 0.0)  # 默认值
        self.assertEqual(MathUtils.safe_divide(10, 0, -1), -1)  # 自定义默认值
        
        # 测试无穷大和NaN处理
        self.assertEqual(MathUtils.safe_divide(np.inf, 1), 0.0)
        self.assertEqual(MathUtils.safe_divide(1, np.inf), 0.0)
        self.assertEqual(MathUtils.safe_divide(np.nan, 1), 0.0)
        
        # 测试类型检查
        self.assertEqual(MathUtils.safe_divide("10", 2), 0.0)
        self.assertEqual(MathUtils.safe_divide(10, "2"), 0.0)
    
    def test_array_normalization_safety(self):
        """测试数组归一化的安全性"""
        # 测试正常数组
        arr = [1, 2, 3, 4, 5]
        normalized = MathUtils.normalize_array(arr, 'min-max')
        self.assertEqual(len(normalized), 5)
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[-1], 1.0)
        
        # 测试包含无效值的数组
        arr_with_nan = [1, 2, np.nan, 4, np.inf]
        normalized = MathUtils.normalize_array(arr_with_nan, 'min-max')
        self.assertEqual(len(normalized), 5)
        self.assertTrue(np.isfinite(normalized[:2]).all())  # 前两个值应该是有限的
        
        # 测试空数组
        empty_arr = []
        normalized = MathUtils.normalize_array(empty_arr)
        self.assertEqual(len(normalized), 0)
        
        # 测试所有值相同的数组
        same_values = [5, 5, 5, 5]
        normalized = MathUtils.normalize_array(same_values)
        self.assertTrue(np.allclose(normalized, 0))
    
    def test_command_processor_error_handling(self):
        """测试命令处理器的错误处理"""
        # 测试配置加载失败的情况
        processor = CommandProcessor()
        self.assertIsNotNone(processor.config)  # 应该有默认配置
        
        # 测试命令别名冲突检查
        processor = CommandProcessor({})
        
        # 先注册一个命令
        processor.register_command('test', lambda x: "test", aliases=['t'])
        
        # 尝试注册冲突的别名应该抛出异常
        with self.assertRaises(ValueError):
            processor.register_command('test2', lambda x: "test2", aliases=['t'])
    
    def test_config_loading_robustness(self):
        """测试配置加载的健壮性"""
        # 测试不存在的配置文件
        config = load_config(['nonexistent.yaml'])
        self.assertIsInstance(config, dict)
        self.assertIn('data', config)  # 应该有默认配置
        
        # 测试配置合并
        base_config = {'a': {'x': 1}, 'b': 2}
        override_config = {'a': {'y': 2}, 'c': 3}
        
        merged = deep_merge_dict(base_config, override_config)
        
        self.assertEqual(merged['a']['x'], 1)  # 保留原有值
        self.assertEqual(merged['a']['y'], 2)  # 添加新值
        self.assertEqual(merged['b'], 2)      # 保留原有值
        self.assertEqual(merged['c'], 3)      # 添加新值
    
    def test_performance_monitor_context(self):
        """测试性能监控器的上下文管理"""
        with PerformanceMonitor("测试操作") as monitor:
            time.sleep(0.1)  # 模拟操作
            self.assertIsNotNone(monitor.start_time)
        
        # 验证监控器正确结束
        self.assertIsNotNone(monitor.end_time)
        self.assertGreater(monitor.end_time, monitor.start_time)
    
    def test_safe_execute_function(self):
        """测试安全执行函数"""
        # 测试成功执行
        def success_func():
            return "success"
        
        success, result = safe_execute(success_func)
        self.assertTrue(success)
        self.assertEqual(result, "success")
        
        # 测试异常处理
        def error_func():
            raise ValueError("test error")
        
        success, result = safe_execute(error_func, default_return="default")
        self.assertFalse(success)
        self.assertEqual(result, "default")
        
        # 测试异常抛出模式
        with self.assertRaises(Exception):
            safe_execute(error_func, raise_on_error=True)
    
    def test_clamp_function(self):
        """测试值限制函数"""
        # 测试正常值
        self.assertEqual(MathUtils.clamp(5, 0, 10), 5)
        self.assertEqual(MathUtils.clamp(-5, 0, 10), 0)
        self.assertEqual(MathUtils.clamp(15, 0, 10), 10)
        
        # 测试无效值
        result = MathUtils.clamp(np.nan, 0, 10)
        self.assertEqual(result, 5.0)  # 应该返回中间值
        
        result = MathUtils.clamp(np.inf, 0, 10)
        self.assertEqual(result, 10)   # 应该被限制到最大值


def run_comprehensive_test():
    """运行综合测试"""
    print("🔍 开始代码修复验证测试...")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCodeReviewFixes)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 运行测试
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 所有代码修复验证测试通过！")
        print(f"📊 共运行 {result.testsRun} 个测试")
    else:
        print("❌ 部分测试失败")
        print(f"📊 共运行 {result.testsRun} 个测试")
        print(f"❌ 失败: {len(result.failures)}")
        print(f"💥 错误: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 