#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
命令处理器模块
负责解析、验证和执行用户命令

功能：
- 命令行参数解析
- 参数验证
- 命令路由和执行
- 统一的错误处理
- 性能监控集成
"""

import sys
import argparse
import re
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Tuple
from pathlib import Path

from .common import (
    LoggerManager, PerformanceMonitor, DataValidator,
    TimeUtils, QuantError, safe_execute, error_context,
    init_project_environment
)
from .config_loader import load_config


class CommandProcessor:
    """
    命令处理器类
    
    负责处理所有用户命令，提供统一的接口和错误处理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化命令处理器
        
        参数:
            config: 配置字典，如果为None则自动加载
        """
        # 初始化环境
        init_project_environment()
        
        # 获取日志记录器
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        
        # 加载配置
        if config is None:
            config, config_error = self._load_default_config()
            if config_error:
                self.logger.error(f"配置加载失败: {config_error}")
                print(f"⚠️ 配置加载失败: {config_error}")
                print("💡 某些功能可能无法正常使用")
        
        self.config = config
        
        # 命令注册表
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._command_aliases: Dict[str, str] = {}  # 别名到主命令名的映射
        
        # 注册内置命令
        self._register_builtin_commands()
        
        self.logger.info("命令处理器初始化完成")
    
    def _load_default_config(self) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        加载默认配置
        
        返回:
            Tuple[Dict[str, Any], Optional[str]]: (配置字典, 错误信息)
        """
        try:
            config = load_config()
            return config, None
        except Exception as e:
            error_msg = f"加载配置失败: {e}"
            raise QuantError(error_msg)
    
    def _register_builtin_commands(self):
        """注册内置命令"""
        # 基础命令
        self.register_command(
            name='help',
            aliases=['h'],
            description='显示帮助信息',
            handler=self._show_help,
            args_spec=[],
            require_config=False
        )
        
        # 配置命令
        self.register_command(
            name='config',
            aliases=['cfg'],
            description='显示配置信息',
            handler=self._show_config,
            args_spec=[],
            require_config=False
        )
        
        # 状态命令
        self.register_command(
            name='status',
            aliases=['stat'],
            description='显示系统状态',
            handler=self._show_status,
            args_spec=[],
            require_config=False
        )
    
    def register_command(self,
                        name: str,
                        handler: Callable,
                        description: str = "",
                        aliases: Optional[List[str]] = None,
                        args_spec: Optional[List[Dict]] = None,
                        require_config: bool = True):
        """
        注册命令
        
        参数:
            name: 命令名称
            handler: 命令处理函数
            description: 命令描述
            aliases: 命令别名列表
            args_spec: 参数规格列表
            require_config: 是否需要配置文件
        """
        if aliases is None:
            aliases = []
        if args_spec is None:
            args_spec = []
        
        # 检查主命令名是否已存在
        if name in self._commands:
            self.logger.warning(f"命令 '{name}' 已存在，将被覆盖")
        
        # 检查别名冲突
        conflicting_aliases = []
        for alias in aliases:
            if alias in self._commands or alias in self._command_aliases:
                conflicting_aliases.append(alias)
        
        if conflicting_aliases:
            self.logger.error(f"别名冲突: {conflicting_aliases}")
            raise ValueError(f"别名冲突: {conflicting_aliases}")
        
        command_info = {
            'name': name,
            'handler': handler,
            'description': description,
            'aliases': aliases,
            'args_spec': args_spec,
            'require_config': require_config
        }
        
        # 注册主命令名
        self._commands[name] = command_info
        
        # 注册别名
        for alias in aliases:
            self._command_aliases[alias] = name
            self._commands[alias] = command_info
        
        self.logger.debug(f"注册命令: {name} (别名: {aliases})")
    
    def parse_arguments(self, args: List[str] = None) -> argparse.Namespace:
        """
        解析命令行参数
        
        参数:
            args: 参数列表，如果为None则使用sys.argv
        
        返回:
            argparse.Namespace: 解析后的参数
        """
        parser = argparse.ArgumentParser(
            description='中证500指数量化交易系统',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 基本参数
        parser.add_argument(
            'command',
            nargs='?',
            default='help',
            help='要执行的命令'
        )
        
        parser.add_argument(
            'params',
            nargs='*',
            help='命令参数'
        )
        
        # 全局选项
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='详细输出'
        )
        
        parser.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='静默模式'
        )
        
        parser.add_argument(
            '--no-timer',
            action='store_true',
            help='禁用性能计时器'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='指定配置文件路径'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='日志级别'
        )
        
        # 特定命令参数
        parser.add_argument(
            '-m', '--mode',
            type=str,
            help='运行模式'
        )
        
        parser.add_argument(
            '-i', '--iter',
            type=int,
            help='迭代次数'
        )
        
        return parser.parse_args(args)
    
    def validate_arguments(self, command: str, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """
        验证命令参数
        
        参数:
            command: 命令名称
            args: 解析后的参数
        
        返回:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 解析别名到主命令名
        actual_command = self._command_aliases.get(command, command)
        
        # 检查命令是否存在
        if actual_command not in self._commands:
            errors.append(f"未知命令: {command}")
            return False, errors
        
        command_info = self._commands[actual_command]
        
        # 验证参数规格
        for arg_spec in command_info['args_spec']:
            arg_name = arg_spec['name']
            required = arg_spec.get('required', False)
            arg_type = arg_spec.get('type', str)
            
            # 检查必需参数
            if required and not hasattr(args, arg_name):
                errors.append(f"缺少必需参数: {arg_name}")
                continue
            
            # 类型检查
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None and not isinstance(value, arg_type):
                    errors.append(f"参数 {arg_name} 类型错误，期望 {arg_type.__name__}")
        
        # 改进的日期格式验证
        for param in args.params:
            if re.match(r'\d{4}-\d{2}-\d{2}', param):
                if not DataValidator.validate_date_format(param):
                    errors.append(f"无效的日期格式: {param}")
        
        # 验证日期范围（如果有两个日期参数）
        date_params = [p for p in args.params if re.match(r'\d{4}-\d{2}-\d{2}', p)]
        if len(date_params) >= 2:
            valid, error_msg = DataValidator.validate_date_range(date_params[0], date_params[1])
            if not valid:
                errors.append(error_msg)
        
        return len(errors) == 0, errors
    
    def execute_command(self, command: str, args: argparse.Namespace) -> Tuple[bool, Any]:
        """
        执行命令
        
        参数:
            command: 命令名称
            args: 命令参数
        
        返回:
            Tuple[bool, Any]: (是否成功, 结果或错误信息)
        """
        # 解析别名到主命令名
        actual_command = self._command_aliases.get(command, command)
        
        if actual_command not in self._commands:
            return False, f"未知命令: {command}"
        
        command_info = self._commands[actual_command]
        handler = command_info['handler']
        
        # 检查是否需要配置
        if command_info['require_config'] and not self.config:
            return False, "命令需要配置文件，但配置加载失败"
        
        # 执行命令
        with PerformanceMonitor(f"命令执行: {command}") as monitor:
            try:
                with error_context(f"执行命令 {command}", self.logger):
                    # 调用处理函数
                    if command_info['require_config']:
                        result = handler(args, self.config)
                    else:
                        result = handler(args)
                    
                    return True, result
                    
            except Exception as e:
                self.logger.error(f"命令执行失败 [{command}]: {str(e)}")
                return False, str(e)
    
    def run(self, args: List[str] = None) -> int:
        """
        运行命令处理器
        
        参数:
            args: 命令行参数列表
        
        返回:
            int: 退出码 (0=成功, 1=失败)
        """
        try:
            # 解析参数
            parsed_args = self.parse_arguments(args)
            
            # 调整日志级别
            if parsed_args.verbose:
                LoggerManager.setup_logging(level='DEBUG')
            elif parsed_args.quiet:
                LoggerManager.setup_logging(level='ERROR')
            else:
                LoggerManager.setup_logging(level=parsed_args.log_level)
            
            # 加载自定义配置
            if parsed_args.config:
                try:
                    self.config = load_config([parsed_args.config])
                except Exception as e:
                    self.logger.error(f"加载自定义配置失败: {e}")
                    return 1
            
            command = parsed_args.command
            
            # 验证参数
            valid, errors = self.validate_arguments(command, parsed_args)
            if not valid:
                for error in errors:
                    print(f"❌ {error}")
                print(f"\n💡 使用 'python run.py help' 查看帮助信息")
                return 1
            
            # 执行命令
            success, result = self.execute_command(command, parsed_args)
            
            if success:
                if result is not None:
                    print(result)
                return 0
            else:
                print(f"❌ 命令执行失败: {result}")
                return 1
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断操作")
            return 1
        except Exception as e:
            self.logger.error(f"命令处理器运行异常: {e}")
            print(f"❌ 系统错误: {e}")
            return 1
    
    # ================================================================================
    # 内置命令处理函数
    # ================================================================================
    
    def _show_help(self, args: argparse.Namespace) -> str:
        """显示帮助信息"""
        help_text = """
🚀 中证500指数量化交易系统 - 命令帮助

📋 可用命令：
"""
        
        # 按命令名排序，去重
        unique_commands = {}
        for name, info in self._commands.items():
            if name == info['name']:  # 只显示主命令名，不显示别名
                unique_commands[name] = info
        
        for name in sorted(unique_commands.keys()):
            info = unique_commands[name]
            aliases_str = f" ({', '.join(info['aliases'])})" if info['aliases'] else ""
            help_text += f"  {name:<12}{aliases_str:<15} - {info['description']}\n"
        
        help_text += """
🔧 全局选项：
  -v, --verbose         详细输出
  -q, --quiet          静默模式
  --no-timer           禁用性能计时器
  --config FILE        指定配置文件
  --log-level LEVEL    设置日志级别 (DEBUG/INFO/WARNING/ERROR)

📝 示例：
  python run.py help                    # 显示帮助
  python run.py config                  # 显示配置
  python run.py status                  # 显示系统状态
  python run.py basic --verbose         # 详细模式运行基础测试

💡 提示：
  - 使用虚拟环境运行项目
  - 确保配置文件正确设置
  - 查看 USER_GUIDE.md 获取详细使用说明
"""
        return help_text
    
    def _show_config(self, args: argparse.Namespace) -> str:
        """显示配置信息"""
        if not self.config:
            return "❌ 配置未加载"
        
        config_text = "📋 当前配置信息：\n\n"
        
        # 显示主要配置部分
        sections = ['data', 'strategy', 'ai', 'backtest', 'logging']
        
        for section in sections:
            if section in self.config:
                config_text += f"📁 {section.upper()}:\n"
                section_config = self.config[section]
                
                if isinstance(section_config, dict):
                    for key, value in section_config.items():
                        if isinstance(value, dict):
                            config_text += f"   📂 {key}: {len(value)} 个子项\n"
                        elif isinstance(value, list):
                            config_text += f"   📋 {key}: {len(value)} 个项目\n"
                        else:
                            # 截断过长的值
                            value_str = str(value)
                            if len(value_str) > 50:
                                value_str = value_str[:50] + "..."
                            config_text += f"   📄 {key}: {value_str}\n"
                else:
                    config_text += f"   值: {section_config}\n"
                
                config_text += "\n"
        
        # 显示统计信息
        config_text += f"📊 配置统计：\n"
        config_text += f"   总配置部分: {len(self.config)}\n"
        config_text += f"   已加载部分: {', '.join(self.config.keys())}\n"
        
        return config_text
    
    def _show_status(self, args: argparse.Namespace) -> str:
        """显示系统状态"""
        status_text = "📊 系统状态信息：\n\n"
        
        # Python 环境信息
        status_text += f"🐍 Python版本: {sys.version.split()[0]}\n"
        status_text += f"📁 工作目录: {Path.cwd()}\n"
        
        # 配置状态
        if self.config:
            status_text += f"⚙️ 配置状态: ✅ 已加载 ({len(self.config)} 个部分)\n"
        else:
            status_text += f"⚙️ 配置状态: ❌ 未加载\n"
        
        # 命令状态
        unique_commands = len(set(info['name'] for info in self._commands.values()))
        status_text += f"🎯 可用命令: {unique_commands} 个\n"
        
        # 系统时间
        status_text += f"⏰ 系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return status_text


# 便捷函数
def create_command_processor(config: Optional[Dict[str, Any]] = None) -> CommandProcessor:
    """
    创建命令处理器实例
    
    参数:
        config: 配置字典
    
    返回:
        CommandProcessor: 命令处理器实例
    """
    return CommandProcessor(config)


def run_command(command: str, *params, **options) -> Tuple[bool, Any]:
    """
    便捷函数：执行单个命令
    
    参数:
        command: 命令名称
        *params: 命令参数
        **options: 命令选项
    
    返回:
        Tuple[bool, Any]: (是否成功, 结果)
    """
    processor = create_command_processor()
    
    # 构造参数列表
    args = [command] + list(params)
    
    # 添加选项
    for key, value in options.items():
        if isinstance(value, bool) and value:
            args.append(f"--{key.replace('_', '-')}")
        else:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # 解析并执行
    parsed_args = processor.parse_arguments(args)
    return processor.execute_command(command, parsed_args)


# 模块导出
__all__ = [
    'CommandProcessor',
    'create_command_processor',
    'run_command'
] 