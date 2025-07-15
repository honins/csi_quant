#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版快速运行脚本
提供简洁的命令行界面来运行系统的各种功能

使用新的模块化架构：
- 命令处理器负责参数解析和路由
- 公共工具模块提供基础功能
- 模块化的功能实现
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.utils.command_processor import CommandProcessor
from src.utils.common import (
    LoggerManager, PerformanceMonitor, init_project_environment,
    error_context, safe_execute
)


class QuantSystemCommands:
    """量化系统命令集合"""
    
    def __init__(self, processor: CommandProcessor):
        """
        初始化命令集合
        
        参数:
            processor: 命令处理器实例
        """
        self.processor = processor
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        
        # 注册所有命令
        self._register_commands()
    
    def _register_commands(self):
        """注册所有业务命令"""
        
        # 基础测试命令
        self.processor.register_command(
            name='basic',
            aliases=['b'],
            description='运行基础策略测试',
            handler=self.run_basic_test,
            require_config=True
        )
        
        # AI测试命令
        self.processor.register_command(
            name='ai',
            aliases=['a'],
            description='运行AI优化和训练',
            handler=self.run_ai_optimization,
            require_config=True,
            args_spec=[
                {'name': 'mode', 'type': str, 'required': False}
            ]
        )
        
        # 单日预测命令
        self.processor.register_command(
            name='predict',
            aliases=['s'],
            description='单日预测功能',
            handler=self.run_single_prediction,
            require_config=True,
            args_spec=[
                {'name': 'params', 'type': list, 'required': True}
            ]
        )
        
        # 滚动回测命令
        self.processor.register_command(
            name='backtest',
            aliases=['r'],
            description='运行滚动回测',
            handler=self.run_rolling_backtest,
            require_config=True,
            args_spec=[
                {'name': 'params', 'type': list, 'required': True}
            ]
        )
        
        # 数据获取命令
        self.processor.register_command(
            name='fetch',
            aliases=['f'],
            description='获取最新数据',
            handler=self.run_data_fetch,
            require_config=True
        )
        
        # 单元测试命令
        self.processor.register_command(
            name='test',
            aliases=['t'],
            description='运行单元测试',
            handler=self.run_unit_tests,
            require_config=False
        )
        
        # 全套测试命令
        self.processor.register_command(
            name='all',
            description='运行全套测试和回测',
            handler=self.run_all_tests,
            require_config=True,
            args_spec=[
                {'name': 'params', 'type': list, 'required': False}
            ]
        )
    
    def run_basic_test(self, args, config):
        """运行基础策略测试"""
        try:
            from examples.basic_test import main as run_basic
            
            self.logger.info("开始运行基础策略测试")
            result = run_basic()
            
            if result:
                return "✅ 基础策略测试完成"
            else:
                return "❌ 基础策略测试失败"
                
        except ImportError as e:
            return f"❌ 无法导入基础测试模块: {e}"
        except Exception as e:
            self.logger.error(f"基础测试执行异常: {e}")
            return f"❌ 基础测试执行异常: {e}"
    
    def run_ai_optimization(self, args, config):
        """运行AI优化"""
        try:
            # 根据模式选择不同的执行方式
            mode = getattr(args, 'mode', 'optimize')
            
            if mode in ['incremental', 'full', 'demo']:
                return self._run_ai_training(mode, config)
            else:
                return self._run_ai_optimization(config)
                
        except ImportError as e:
            self.logger.error(f"AI模块导入失败: {e}")
            return f"❌ AI模块不可用，请检查依赖包安装: {e}"
        except Exception as e:
            self.logger.error(f"AI优化执行异常: {e}")
            return f"❌ AI优化执行异常: {e}"
    
    def _run_ai_training(self, mode, config):
        """运行AI训练"""
        self.logger.info(f"开始AI训练，模式: {mode}")
        
        try:
            # 尝试导入并调用真实的AI训练模块
            from src.ai.ai_optimizer_improved import AIOptimizerImproved
            from src.strategy.strategy_module import StrategyModule
            from src.data.data_module import DataModule
            
            # 初始化数据和策略模块
            data_module = DataModule(config)
            
            # 获取数据配置
            data_config = config.get('data', {})
            time_range = data_config.get('time_range', {})
            start_date = time_range.get('start_date', '2019-01-01')
            end_date = time_range.get('end_date', '2025-07-15')
            
            data = data_module.get_history_data(start_date, end_date)
            strategy_module = StrategyModule(config)
            
            if data is None or data.empty:
                return "❌ 无法获取数据，请检查数据配置"
            
            ai_optimizer = AIOptimizerImproved(config)
            
            if mode == 'incremental':
                # 增量训练逻辑
                result = ai_optimizer.incremental_train(data, strategy_module)
                return f"✅ AI增量训练完成: {result.get('summary', '成功')}"
                
            elif mode == 'full':
                # 完全重训练逻辑
                result = ai_optimizer.full_train(data, strategy_module)
                return f"✅ AI完全重训练完成: {result.get('summary', '成功')}"
                
            elif mode == 'demo':
                # 演示预测逻辑
                from examples.predict_single_day import main as demo_predict
                result = demo_predict()
                return f"✅ AI演示预测完成: {result}"
                
            else:
                return f"❌ 未知的AI训练模式: {mode}"
                
        except ImportError as e:
            self.logger.warning(f"AI训练模块不可用: {e}")
            # 降级到基础功能
            training_modes = {
                'incremental': '增量训练',
                'full': '完全重训练',
                'demo': '演示预测'
            }
            return f"⚠️ AI模块不可用，模拟执行 {training_modes.get(mode, mode)}"
        except Exception as e:
            self.logger.error(f"AI训练执行失败: {e}")
            return f"❌ AI训练失败: {e}"
    
    def _run_ai_optimization(self, config):
        """运行AI优化"""
        self.logger.info("开始AI参数优化")
        
        try:
            # 导入所需模块
            from src.ai.ai_optimizer_improved import AIOptimizerImproved
            from src.strategy.strategy_module import StrategyModule
            from src.data.data_module import DataModule
            
            # 初始化数据模块获取数据
            data_module = DataModule(config)
            
            # 获取数据配置
            data_config = config.get('data', {})
            time_range = data_config.get('time_range', {})
            start_date = time_range.get('start_date', '2019-01-01')
            end_date = time_range.get('end_date', '2025-07-15')
            
            data = data_module.get_history_data(start_date, end_date)
            
            if data is None or data.empty:
                return "❌ 无法获取数据，请检查数据配置"
            
            # 🔧 关键修复：对数据进行预处理，计算技术指标
            self.logger.info("对数据进行预处理，计算技术指标...")
            data = data_module.preprocess_data(data)
            self.logger.info(f"预处理完成，数据列: {list(data.columns)}")
            
            # 初始化策略模块
            strategy_module = StrategyModule(config)
            
            # 运行完整的AI优化（包含策略优化 + 模型训练）
            ai_optimizer = AIOptimizerImproved(config)
            optimization_result = ai_optimizer.run_complete_optimization(
                data, strategy_module
            )
            
            if optimization_result.get('success'):
                best_score = optimization_result.get('best_score', 0)
                return f"✅ AI参数优化完成，最佳得分: {best_score:.4f}"
            else:
                error_msg = optimization_result.get('error', '未知错误')
                return f"❌ AI参数优化失败: {error_msg}"
                
        except ImportError as e:
            self.logger.warning(f"AI优化模块不可用: {e}")
            return "⚠️ AI优化模块不可用，请检查模块安装"
        except Exception as e:
            self.logger.error(f"AI优化失败: {e}")
            return f"❌ AI优化失败: {e}"
    
    def run_single_prediction(self, args, config):
        """运行单日预测"""
        if not args.params:
            return "❌ 请提供预测日期，格式: YYYY-MM-DD"
        
        predict_date = args.params[0]
        
        # 验证日期格式
        from src.utils.common import DataValidator
        if not DataValidator.validate_date_format(predict_date):
            return f"❌ 无效的日期格式: {predict_date}"
        
        self.logger.info(f"开始单日预测: {predict_date}")
        
        try:
            # 尝试调用真实的预测模块 - 只使用已训练模型，不重新训练
            from examples.predict_single_day import predict_single_day
            
            # 🔧 修改：强制只使用已训练模型，不允许重新训练
            result = predict_single_day(predict_date, use_trained_model=True)
            
            if result:
                return f"✅ {predict_date} 预测完成（仅使用已训练模型）"
            else:
                return f"❌ {predict_date} 预测失败（请先运行 'python run.py ai' 训练模型）"
            
        except ImportError as e:
            self.logger.warning(f"预测模块不可用: {e}")
            return f"⚠️ 预测模块不可用: {e}"
        except Exception as e:
            self.logger.error(f"单日预测异常: {e}")
            return f"❌ 单日预测失败: {e}"
    
    def run_rolling_backtest(self, args, config):
        """运行滚动回测"""
        if len(args.params) < 2:
            return "❌ 请提供开始和结束日期，格式: YYYY-MM-DD YYYY-MM-DD"
        
        start_date = args.params[0]
        end_date = args.params[1]
        
        # 验证日期格式和范围
        from src.utils.common import DataValidator
        valid, error_msg = DataValidator.validate_date_range(start_date, end_date)
        if not valid:
            return f"❌ {error_msg}"
        
        self.logger.info(f"开始滚动回测: {start_date} 到 {end_date}")
        
        try:
            # 尝试调用真实的回测模块
            from examples.run_rolling_backtest import run_rolling_backtest
            
            result = run_rolling_backtest(start_date, end_date)
            
            if result.get('success'):
                success_rate = result.get('success_rate', 0)
                total_signals = result.get('total_signals', 0)
                return f"✅ 滚动回测完成 ({start_date} ~ {end_date}): 成功率 {success_rate:.1%}, 信号数 {total_signals}"
            else:
                error_msg = result.get('error', '回测失败')
                return f"❌ 滚动回测失败: {error_msg}"
            
        except ImportError as e:
            self.logger.warning(f"回测模块不可用: {e}")
            return f"⚠️ 回测模块不可用: {e}"
        except Exception as e:
            self.logger.error(f"滚动回测异常: {e}")
            return f"❌ 滚动回测失败: {e}"
    
    def run_data_fetch(self, args, config):
        """运行数据获取"""
        self.logger.info("开始获取最新数据")
        
        try:
            from src.data.fetch_latest_data import DataFetcher
            
            fetcher = DataFetcher()
            results = fetcher.fetch_and_save_latest_data()
            
            # 检查结果
            if results and all(info.get('success', False) for info in results.values()):
                return "✅ 数据获取完成"
            else:
                return "❌ 数据获取失败"
                
        except ImportError as e:
            return f"❌ 无法导入数据获取模块: {e}"
        except Exception as e:
            self.logger.error(f"数据获取异常: {e}")
            return f"❌ 数据获取异常: {e}"
    
    def run_unit_tests(self, args):
        """运行单元测试"""
        self.logger.info("开始运行单元测试")
        
        try:
            import subprocess
            
            # 运行pytest测试
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v'],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                return f"✅ 单元测试通过\n{result.stdout}"
            else:
                return f"❌ 单元测试失败\n{result.stderr}"
                
        except Exception as e:
            self.logger.error(f"单元测试异常: {e}")
            return f"❌ 单元测试异常: {e}"
    
    def run_all_tests(self, args, config):
        """运行全套测试"""
        results = []
        
        self.logger.info("开始运行全套测试")
        
        # 1. 基础测试
        with PerformanceMonitor("基础测试"):
            result = self.run_basic_test(args, config)
            results.append(f"📊 基础测试: {result}")
        
        # 2. 数据获取
        with PerformanceMonitor("数据获取"):
            result = self.run_data_fetch(args, config)
            results.append(f"📥 数据获取: {result}")
        
        # 3. AI优化（如果有参数指定）
        if hasattr(args, 'mode') and args.mode:
            with PerformanceMonitor("AI优化"):
                result = self.run_ai_optimization(args, config)
                results.append(f"🤖 AI优化: {result}")
        
        # 4. 回测（如果提供了日期参数）
        if args.params and len(args.params) >= 2:
            with PerformanceMonitor("滚动回测"):
                result = self.run_rolling_backtest(args, config)
                results.append(f"📈 回测: {result}")
        
        # 汇总结果
        summary = "\n".join(results)
        return f"🎯 全套测试完成:\n\n{summary}"


def check_virtual_environment():
    """检查虚拟环境"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 检测到虚拟环境")
        return True
    else:
        print("⚠️ 未检测到虚拟环境")
        print("💡 建议使用虚拟环境运行项目:")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # Linux/Mac")
        return False


def main():
    """主函数"""
    try:
        # 检查虚拟环境
        check_virtual_environment()
        
        # 初始化项目环境
        init_project_environment()
        
        # 创建命令处理器
        processor = CommandProcessor()
        
        # 注册量化系统命令
        commands = QuantSystemCommands(processor)
        
        # 运行命令处理器
        exit_code = processor.run()
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 