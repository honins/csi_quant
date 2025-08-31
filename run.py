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
from copy import deepcopy
from datetime import date, timedelta

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
            aliases=['p'],
            description='单日预测功能',
            handler=self.run_single_prediction,
            require_config=True,
            args_spec=[
                {'name': 'date', 'type': str, 'required': False}
            ]
        )
        
        # 滚动回测命令
        self.processor.register_command(
            name='backtest',
            aliases=['bt'],
            description='运行滚动回测',
            handler=self.run_rolling_backtest,
            require_config=True,
            args_spec=[
                {'name': 'start_date', 'type': str, 'required': False},
                {'name': 'end_date', 'type': str, 'required': False}
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
            print("🎯 AI优化模式")
            print("📋 功能说明: 完整AI优化 - 策略参数优化 + 模型训练")
            print()
            
            # 处理快速验证模式覆盖
            use_quick = getattr(args, 'quick', False)
            local_config = deepcopy(config)
            if use_quick:
                print("⚡ 已启用快速验证模式：缩小数据范围并减少优化迭代")
                # 缩小数据时间范围到最近约180天
                end_d = date.today()
                start_d = end_d - timedelta(days=180)
                local_config.setdefault('data', {})
                local_config['data'].setdefault('time_range', {})
                local_config['data']['time_range']['start_date'] = start_d.strftime('%Y-%m-%d')
                local_config['data']['time_range']['end_date'] = end_d.strftime('%Y-%m-%d')
                # 调小贝叶斯优化迭代次数
                local_config.setdefault('bayesian_optimization', {})
                local_config['bayesian_optimization']['n_calls'] = max(10, int(local_config['bayesian_optimization'].get('n_calls', 120) * 0.2))
                calc_init = int(local_config['bayesian_optimization'].get('n_initial_points', 25) * 0.2)
                local_config['bayesian_optimization']['n_initial_points'] = max(5, min(10, calc_init))
                # 明确启用
                local_config['bayesian_optimization']['enabled'] = True
                print(f"   📅 快速数据范围: {local_config['data']['time_range']['start_date']} ~ {local_config['data']['time_range']['end_date']}")
                print(f"   🔬 快速优化配置: n_calls={local_config['bayesian_optimization']['n_calls']}, n_initial_points={local_config['bayesian_optimization']['n_initial_points']}")
            
            return self._run_ai_optimization(local_config)
                
        except ImportError as e:
            self.logger.error(f"AI模块导入失败: {e}")
            return f"❌ AI模块不可用，请检查依赖包安装: {e}"
        except Exception as e:
            self.logger.error(f"AI优化执行异常: {e}")
            return f"❌ AI优化执行异常: {e}"
    
    def _run_ai_optimization(self, config):
        """运行AI优化"""
        from datetime import datetime
        start_time = datetime.now()
        
        print("🤖 开始AI参数优化")
        print("=" * 60)
        self.logger.info("🤖 开始AI参数优化")
        
        try:
            # 导入所需模块
            print("📦 导入AI优化模块...")
            from src.ai.ai_optimizer_improved import AIOptimizerImproved
            from src.strategy.strategy_module import StrategyModule
            from src.data.data_module import DataModule
            print("✅ 模块导入成功")
            
            # 初始化数据模块获取数据
            print("\n📊 获取历史数据...")
            data_module = DataModule(config)
            
            # 获取数据配置
            data_config = config.get('data', {})
            time_range = data_config.get('time_range', {})
            start_date = time_range.get('start_date', '2019-01-01')
            end_date = time_range.get('end_date', '2025-07-15')
            
            print(f"📅 数据时间范围: {start_date} ~ {end_date}")
            data = data_module.get_history_data(start_date, end_date)
            
            if data is None or data.empty:
                return "❌ 无法获取数据，请检查数据配置"
            
            print(f"✅ 数据获取成功: {len(data)} 条记录")
            
            # 🔧 关键修复：对数据进行预处理，计算技术指标
            print("\n🔧 数据预处理...")
            self.logger.info("对数据进行预处理，计算技术指标...")
            data = data_module.preprocess_data(data)
            print(f"✅ 预处理完成，数据列: {list(data.columns)}")
            self.logger.info(f"预处理完成，数据列: {list(data.columns)}")
            
            # 初始化策略模块
            print("\n⚙️ 初始化策略模块...")
            strategy_module = StrategyModule(config)
            print("✅ 策略模块初始化完成")
            
            # 显示当前策略参数
            current_params = strategy_module.get_params()
            print(f"📋 当前策略参数:")
            for key, value in current_params.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # 运行完整的AI优化（包含策略优化 + 模型训练）
            print("\n🚀 开始完整AI优化流程...")
            print("💡 包含: 策略参数优化 + 模型训练 + 参数保存")
            ai_optimizer = AIOptimizerImproved(config)
            
            optimization_result = ai_optimizer.run_complete_optimization(
                data, strategy_module
            )
            
            # 计算总耗时
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("📊 AI优化结果总结")
            print("=" * 60)
            
            if optimization_result.get('success'):
                # 获取详细结果
                strategy_result = optimization_result.get('strategy_optimization', {})
                model_result = optimization_result.get('model_training', {})
                evaluation_result = optimization_result.get('final_evaluation', {})
                
                print(f"⏱️  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
                
                # 策略优化结果
                if strategy_result.get('success'):
                    best_score = strategy_result.get('best_score', 0)
                    test_success_rate = strategy_result.get('test_success_rate', 0)
                    optimization_method = strategy_result.get('optimization_method', 'unknown')
                    
                    print(f"🎯 策略参数优化:")
                    print(f"   ✅ 优化方法: {optimization_method}")
                    print(f"   📈 最优得分: {best_score:.6f}")
                    print(f"   📊 测试集成功率: {test_success_rate:.2%}")
                    
                    # 显示优化后的参数
                    best_params = strategy_result.get('best_params', {})
                    if best_params:
                        print(f"   🔧 优化后参数:")
                        for key, value in best_params.items():
                            if isinstance(value, float):
                                print(f"      {key}: {value:.4f}")
                            else:
                                print(f"      {key}: {value}")
                else:
                    print(f"⚠️ 策略参数优化: 失败")
                
                # 模型训练结果
                if model_result.get('success'):
                    train_samples = model_result.get('train_samples', 0)
                    feature_count = model_result.get('feature_count', 0)
                    positive_ratio = model_result.get('positive_ratio', 0)
                    save_success = model_result.get('save_success', False)
                    
                    print(f"🤖 模型训练:")
                    print(f"   ✅ 训练样本: {train_samples:,} 条")
                    print(f"   📈 特征数量: {feature_count} 个")
                    print(f"   📊 正样本比例: {positive_ratio:.2%}")
                    print(f"   💾 模型保存: {'成功' if save_success else '失败'}")
                else:
                    print(f"❌ 模型训练: 失败")
                
                # 最终评估结果
                if evaluation_result.get('success'):
                    strategy_score = evaluation_result.get('strategy_score', 0)
                    strategy_success_rate = evaluation_result.get('strategy_success_rate', 0)
                    identified_points = evaluation_result.get('identified_points', 0)
                    ai_confidence = evaluation_result.get('ai_confidence', 0)
                    
                    print(f"📊 最终评估:")
                    print(f"   🎯 策略得分: {strategy_score:.4f}")
                    print(f"   📈 成功率: {strategy_success_rate:.2%}")
                    print(f"   🔍 交易数: {identified_points}")
                    print(f"   🤖 AI置信度: {ai_confidence:.4f}")
                else:
                    print(f"⚠️ 最终评估: 部分失败")
                
                print("\n🎉 AI优化完成！")
                print("💡 优化后的策略参数已保存到 config/strategy.yaml")
                print("💡 新训练的模型已保存到 models/ 目录")
                
                return f"✅ AI参数优化完成，总耗时: {total_time:.1f}秒"
            else:
                error_msg = optimization_result.get('error', '未知错误')
                errors = optimization_result.get('errors', [])
                
                print(f"❌ AI优化失败 (耗时: {total_time:.1f}秒)")
                if errors:
                    print(f"📋 错误详情:")
                    for error in errors:
                        print(f"   • {error}")
                
                return f"❌ AI参数优化失败: {error_msg}"
                
        except ImportError as e:
            self.logger.warning(f"AI优化模块不可用: {e}")
            return "⚠️ AI优化模块不可用，请检查模块安装"
        except Exception as e:
            self.logger.error(f"AI优化失败: {e}")
            return f"❌ AI优化失败: {e}"
    
    def run_single_prediction(self, args, config):
        """运行单日预测"""
        # 获取日期参数
        if hasattr(args, 'date') and args.date:
            predict_date = args.date
        else:
            # 使用最新交易日
            from src.utils.common import TimeUtils
            predict_date = TimeUtils.get_latest_trading_day().strftime('%Y-%m-%d')
        
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
        # 获取日期参数
        if hasattr(args, 'start_date') and args.start_date:
            start_date = args.start_date
        else:
            return "❌ 请提供开始日期，使用 -s 或 --start-date 参数"
            
        if hasattr(args, 'end_date') and args.end_date:
            end_date = args.end_date
        else:
            return "❌ 请提供结束日期，使用 -e 或 --end-date 参数"
        
        # 验证日期格式和范围
        from src.utils.common import DataValidator
        valid, error_msg = DataValidator.validate_date_range(start_date, end_date)
        if not valid:
            return f"❌ {error_msg}"
        
        self.logger.info(f"开始滚动回测: {start_date} 到 {end_date}")
        
        try:
            # 使用修改后的回测模块，生成Markdown报告
            from examples.run_rolling_backtest import run_rolling_backtest
            
            result = run_rolling_backtest(start_date, end_date, generate_report=True)
            
            if result.get('success'):
                metrics = result.get('metrics', {})
                success_rate = metrics.get('success_rate', 0)
                total_predictions = metrics.get('total_predictions', 0)
                f1_score = metrics.get('f1', 0)
                recall = metrics.get('recall', 0)
                precision = metrics.get('precision', 0)
                report_path = result.get('report_path')
                
                # 输出关键指标
                self.logger.info(f"回测完成: 成功率={success_rate:.2%}, 预测数={total_predictions}, F1={f1_score:.3f}, Recall={recall:.3f}, Precision={precision:.3f}")
                
                # 输出报告路径
                if report_path:
                    relative_path = os.path.relpath(report_path, project_root)
                    self.logger.info(f"📄 回测报告已生成: {relative_path}")
                    return f"✅ 滚动回测完成 ({start_date} ~ {end_date}): 成功率 {success_rate:.1%}, 预测数 {total_predictions}, F1 {f1_score:.3f}\n📄 报告: {relative_path}"
                else:
                    return f"✅ 滚动回测完成 ({start_date} ~ {end_date}): 成功率 {success_rate:.1%}, 预测数 {total_predictions}, F1 {f1_score:.3f}"
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
        
        # 创建并注册量化系统命令
        quant_commands = QuantSystemCommands(processor)
        
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