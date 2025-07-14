#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新架构使用演示
展示如何使用重构后的新架构开发模块和命令

本示例包含：
1. 使用公共工具模块
2. 继承基础模块类
3. 创建自定义命令
4. 集成到命令处理器
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# 导入重构后的模块
from src.utils.common import (
    LoggerManager, PerformanceMonitor, DataValidator,
    TimeUtils, safe_execute, error_context, init_project_environment
)
from src.utils.base_module import AIModule
from src.utils.command_processor import CommandProcessor


# ================================================================================
# 示例1: 使用公共工具模块
# ================================================================================

def demo_common_utilities():
    """演示公共工具模块的使用"""
    print("\n" + "="*60)
    print("📋 示例1: 公共工具模块使用演示")
    print("="*60)
    
    # 1. 日志管理
    logger = LoggerManager.get_logger('DemoModule')
    logger.info("开始演示公共工具模块")
    
    # 2. 性能监控
    with PerformanceMonitor("数据处理演示"):
        # 模拟一些数据处理
        data = np.random.randn(1000, 5)
        df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
        
        # 数据验证
        valid, errors = DataValidator.validate_dataframe(
            df, 
            required_columns=['A', 'B', 'C'],
            min_rows=100
        )
        
        if valid:
            logger.info("数据验证通过")
        else:
            logger.error(f"数据验证失败: {errors}")
    
    # 3. 时间工具
    date_info = TimeUtils.get_date_range_info('2023-01-01', '2023-12-31')
    print(f"📅 日期范围信息: {date_info}")
    
    # 4. 安全执行
    def risky_operation():
        # 模拟可能失败的操作
        if np.random.random() < 0.3:
            raise ValueError("随机错误")
        return "操作成功"
    
    success, result = safe_execute(
        risky_operation,
        error_message="演示操作失败",
        default_return="默认结果"
    )
    
    print(f"🔧 安全执行结果: {'成功' if success else '失败'} - {result}")


# ================================================================================
# 示例2: 继承基础模块类
# ================================================================================

class DemoAIModule(AIModule):
    """演示AI模块，继承基础模块类"""
    
    def _initialize_module(self):
        """模块特定的初始化"""
        self.logger.info("DemoAI模块初始化")
        
        # 模拟加载模型
        self.model_loaded = True
        self.processing_count = 0
    
    def _validate_module_config(self):
        """验证模块配置"""
        # 这里可以添加特定的配置验证
        pass
    
    def _get_module_directories(self):
        """获取模块特定目录"""
        base_dirs = super()._get_module_directories()
        return base_dirs + [
            self.project_root / 'demo_results'
        ]
    
    def process_data(self, data):
        """演示数据处理方法"""
        # 使用基础模块提供的安全操作
        return self.safe_operation(
            "数据处理",
            self._do_process_data,
            data
        )
    
    def _do_process_data(self, data):
        """实际的数据处理逻辑"""
        if not self.model_loaded:
            raise RuntimeError("模型未加载")
        
        # 模拟数据处理
        self.processing_count += 1
        result = {
            'processed_rows': len(data),
            'processing_count': self.processing_count,
            'mean_values': data.mean().to_dict() if hasattr(data, 'mean') else None
        }
        
        self.logger.info(f"处理了 {len(data)} 行数据")
        return result
    
    def get_module_status(self):
        """获取模块状态"""
        base_status = self.get_status()
        base_status.update({
            'model_loaded': self.model_loaded,
            'processing_count': self.processing_count
        })
        return base_status


def demo_base_module():
    """演示基础模块类的使用"""
    print("\n" + "="*60)
    print("📋 示例2: 基础模块类使用演示")
    print("="*60)
    
    # 模拟配置
    config = {
        'ai': {
            'model_type': 'demo',
            'models_dir': 'models'
        },
        'data': {
            'data_file_path': 'demo.csv'
        }
    }
    
    try:
        # 创建模块实例
        demo_module = DemoAIModule(config)
        
        # 生成演示数据
        demo_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        # 处理数据
        success, result = demo_module.process_data(demo_data)
        
        if success:
            print(f"✅ 数据处理成功: {result}")
        else:
            print(f"❌ 数据处理失败: {result}")
        
        # 获取模块状态
        status = demo_module.get_module_status()
        print(f"📊 模块状态: {status}")
        
        # 清理资源
        demo_module.cleanup()
        
    except Exception as e:
        print(f"❌ 模块创建失败: {e}")


# ================================================================================
# 示例3: 创建自定义命令
# ================================================================================

class DemoCommands:
    """演示命令集合"""
    
    def __init__(self, processor: CommandProcessor):
        self.processor = processor
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        
        # 注册演示命令
        self._register_demo_commands()
    
    def _register_demo_commands(self):
        """注册演示命令"""
        
        # 数据生成命令
        self.processor.register_command(
            name='generate',
            aliases=['gen'],
            description='生成演示数据',
            handler=self.generate_demo_data,
            require_config=False
        )
        
        # 数据分析命令
        self.processor.register_command(
            name='analyze',
            aliases=['ana'],
            description='分析演示数据',
            handler=self.analyze_demo_data,
            require_config=False
        )
        
        # 模块测试命令
        self.processor.register_command(
            name='test-module',
            aliases=['tm'],
            description='测试演示模块',
            handler=self.test_demo_module,
            require_config=False
        )
    
    def generate_demo_data(self, args):
        """生成演示数据命令"""
        self.logger.info("生成演示数据")
        
        try:
            # 生成随机数据
            n_samples = 1000
            data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'value1': np.random.randn(n_samples).cumsum(),
                'value2': np.random.randn(n_samples),
                'category': np.random.choice(['A', 'B', 'C'], n_samples)
            })
            
            # 保存数据
            output_file = project_root / 'demo_data.csv'
            data.to_csv(output_file, index=False)
            
            return f"✅ 已生成 {n_samples} 行演示数据，保存到: {output_file}"
            
        except Exception as e:
            return f"❌ 生成数据失败: {e}"
    
    def analyze_demo_data(self, args):
        """分析演示数据命令"""
        self.logger.info("分析演示数据")
        
        try:
            # 加载数据
            data_file = project_root / 'demo_data.csv'
            if not data_file.exists():
                return "❌ 演示数据文件不存在，请先运行 'generate' 命令"
            
            data = pd.read_csv(data_file)
            
            # 数据分析
            analysis = {
                'total_rows': len(data),
                'date_range': f"{data['date'].min()} ~ {data['date'].max()}",
                'value1_stats': {
                    'mean': data['value1'].mean(),
                    'std': data['value1'].std(),
                    'min': data['value1'].min(),
                    'max': data['value1'].max()
                },
                'category_counts': data['category'].value_counts().to_dict()
            }
            
            # 格式化输出
            result = "📊 数据分析结果:\n"
            result += f"  总行数: {analysis['total_rows']}\n"
            result += f"  日期范围: {analysis['date_range']}\n"
            result += f"  Value1统计: 均值={analysis['value1_stats']['mean']:.2f}, "
            result += f"标准差={analysis['value1_stats']['std']:.2f}\n"
            result += f"  类别分布: {analysis['category_counts']}"
            
            return result
            
        except Exception as e:
            return f"❌ 分析数据失败: {e}"
    
    def test_demo_module(self, args):
        """测试演示模块命令"""
        self.logger.info("测试演示模块")
        
        try:
            # 创建演示模块
            config = {
                'ai': {'model_type': 'demo', 'models_dir': 'models'},
                'data': {'data_file_path': 'demo.csv'}
            }
            
            demo_module = DemoAIModule(config)
            
            # 生成测试数据
            test_data = pd.DataFrame({
                'x': np.random.randn(50),
                'y': np.random.randn(50)
            })
            
            # 测试处理
            success, result = demo_module.process_data(test_data)
            
            # 清理
            demo_module.cleanup()
            
            if success:
                return f"✅ 模块测试通过: {result}"
            else:
                return f"❌ 模块测试失败: {result}"
                
        except Exception as e:
            return f"❌ 模块测试异常: {e}"


def demo_command_processor():
    """演示命令处理器的使用"""
    print("\n" + "="*60)
    print("📋 示例3: 命令处理器使用演示")
    print("="*60)
    
    # 创建命令处理器
    processor = CommandProcessor()
    
    # 注册演示命令
    demo_commands = DemoCommands(processor)
    
    print("已注册演示命令，可以使用以下命令:")
    print("  generate (gen) - 生成演示数据")
    print("  analyze (ana) - 分析演示数据") 
    print("  test-module (tm) - 测试演示模块")
    print("\n可以通过以下方式测试:")
    print("  python examples/new_architecture_demo.py generate")
    print("  python examples/new_architecture_demo.py analyze")
    
    return processor


# ================================================================================
# 主函数
# ================================================================================

def main():
    """主演示函数"""
    print("🚀 新架构使用演示")
    print("本演示将展示如何使用重构后的新架构")
    
    # 初始化环境
    init_project_environment()
    
    # 运行各个演示
    demo_common_utilities()
    demo_base_module()
    processor = demo_command_processor()
    
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("\n💡 使用建议:")
    print("1. 参考本示例创建自己的模块")
    print("2. 使用公共工具模块减少重复代码")
    print("3. 继承基础模块类获得标准功能")
    print("4. 通过命令处理器扩展功能")
    
    # 如果有命令行参数，尝试执行命令
    if len(sys.argv) > 1:
        print(f"\n🔧 执行命令: {' '.join(sys.argv[1:])}")
        exit_code = processor.run(sys.argv[1:])
        return exit_code
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 