#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版日常交易流程自动化机器人
支持无人值守、常驻运行、自动数据更新、性能监控、数据备份等功能
"""

import sys
import os
import json
import argparse
import schedule
import time
import logging
import threading
import subprocess
import psutil
import signal
import shutil
import git
import pandas as pd
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import tempfile
import uuid

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.utils import load_config, setup_logging
from src.notification.notification_module import NotificationModule

@dataclass
class SystemMetrics:
    """系统性能指标"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    process_count: int
    timestamp: datetime

class EnhancedDailyTradingBot:
    """增强版日常交易流程自动化机器人"""
    
    def __init__(self, config_path: Optional[str] = None, daemon_mode: bool = False):
        """初始化增强版交易机器人"""
        
        # 守护进程模式标志
        self.daemon_mode = daemon_mode
        self.running = True
        self.restart_flag = False
        
        # 线程锁
        self._metrics_lock = threading.Lock()
        self._state_lock = threading.Lock()
        
        # 平台检测
        self.is_windows = platform.system() == 'Windows'
        
        # 加载配置
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_core.yaml')
        
        try:
            self.config = load_config(config_path=config_path)
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            # 使用默认配置文件
            fallback_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
            if os.path.exists(fallback_config):
                self.config = load_config(config_path=fallback_config)
            else:
                raise RuntimeError(f"无法加载配置文件: {config_path}")
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化模块
        try:
            self.data_module = DataModule(self.config)
            self.strategy_module = StrategyModule(self.config)
            self.ai_improved = AIOptimizerImproved(self.config)
            self.notification_manager = NotificationModule(self.config)
        except Exception as e:
            self.logger.error(f"模块初始化失败: {e}")
            raise
        
        # 设置工作目录
        self.work_dir = Path(os.path.dirname(__file__)).parent
        self.results_dir = self.work_dir / 'results' / 'daily_trading'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份目录
        self.backup_dir = self.work_dir / 'results' / 'backup'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态和历史文件
        self.state_file = self.results_dir / 'bot_state.json'
        self.history_file = self.results_dir / 'trading_history.json'
        self.metrics_file = self.results_dir / 'performance_metrics.json'
        self.pid_file = self.results_dir / 'bot.pid'
        
        # 加载状态
        self.state = self.load_state()
        
        # 性能监控
        self.metrics_history = []
        self.last_health_check = datetime.now()
        
        # Git仓库（用于数据提交）
        try:
            self.git_repo = git.Repo(self.work_dir)
            self.logger.info("Git仓库初始化成功")
        except git.exc.GitError as e:
            self.git_repo = None
            self.logger.warning(f"Git仓库初始化失败: {e}，数据提交功能将被禁用")
        except Exception as e:
            self.git_repo = None
            self.logger.warning(f"Git仓库初始化异常: {e}，数据提交功能将被禁用")
        
        # 设置信号处理器（守护进程）
        if self.daemon_mode:
            self._setup_signal_handlers()
        
        self.logger.info("增强版日常交易机器人初始化完成")
    
    def _setup_signal_handlers(self):
        """设置信号处理器（跨平台兼容）"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # SIGUSR1仅在非Windows系统中可用
            if not self.is_windows:
                signal.signal(signal.SIGUSR1, self._restart_handler)
                self.logger.info("信号处理器设置完成（包含SIGUSR1）")
            else:
                self.logger.info("信号处理器设置完成（Windows模式，跳过SIGUSR1）")
                
        except Exception as e:
            self.logger.warning(f"信号处理器设置失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"接收到信号 {signum}，准备退出...")
        self.running = False
    
    def _restart_handler(self, signum, frame):
        """重启信号处理器（仅限非Windows系统）"""
        self.logger.info("接收到重启信号...")
        self.restart_flag = True
        self.running = False
    
    def setup_logging(self):
        """设置增强版日志"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 主日志文件
        main_log = os.path.join(log_dir, 'enhanced_trading_bot.log')
        
        # 创建日志格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # 主日志处理器
        main_handler = logging.FileHandler(main_log, encoding='utf-8')
        main_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(main_handler)
        
        # 守护进程模式下不输出到控制台
        if not self.daemon_mode:
            root_logger.addHandler(console_handler)
    
    def write_pid_file(self):
        """写入PID文件"""
        try:
            with open(self.pid_file, 'w', encoding='utf-8') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            self.logger.error(f"写入PID文件失败: {e}")
    
    def remove_pid_file(self):
        """删除PID文件"""
        try:
            if self.pid_file.exists():
                os.remove(self.pid_file)
        except Exception as e:
            self.logger.error(f"删除PID文件失败: {e}")
    
    def load_state(self) -> Dict[str, Any]:
        """加载机器人状态（向后兼容）"""
        # 默认状态
        default_state = {
            'last_training_date': None,
            'last_prediction_date': None,
            'last_data_fetch': None,
            'last_backup': None,
            'consecutive_errors': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'training_count': 0,
            'data_fetch_count': 0,
            'backup_count': 0,
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'uptime_start': datetime.now().isoformat()
        }
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
                
                # 合并现有状态和默认状态（现有状态优先）
                merged_state = default_state.copy()
                merged_state.update(existing_state)
                
                # 确保新字段存在
                if 'uptime_start' not in merged_state:
                    merged_state['uptime_start'] = datetime.now().isoformat()
                if 'data_fetch_count' not in merged_state:
                    merged_state['data_fetch_count'] = 0
                if 'backup_count' not in merged_state:
                    merged_state['backup_count'] = 0
                if 'last_data_fetch' not in merged_state:
                    merged_state['last_data_fetch'] = None
                if 'last_backup' not in merged_state:
                    merged_state['last_backup'] = None
                
                return merged_state
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"加载状态文件失败: {e}")
        
        return default_state
    
    def save_state(self):
        """保存机器人状态（线程安全）"""
        with self._state_lock:
            try:
                # 使用临时文件确保原子写入
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
                
                # 原子替换
                if self.is_windows:
                    # Windows需要先删除目标文件
                    if self.state_file.exists():
                        os.remove(self.state_file)
                temp_file.replace(self.state_file)
                
            except Exception as e:
                self.logger.error(f"保存状态文件失败: {e}")
                # 清理临时文件
                if temp_file.exists():
                    os.remove(temp_file)
    
    def collect_performance_metrics(self) -> SystemMetrics:
        """收集系统性能指标（线程安全，跨平台兼容）"""
        try:
            # 获取磁盘使用率（跨平台兼容）
            if self.is_windows:
                # Windows使用C盘
                disk_path = 'C:\\'
            else:
                # Unix/Linux使用根目录
                disk_path = '/'
            
            try:
                disk_usage = psutil.disk_usage(disk_path).percent
            except Exception as e:
                self.logger.warning(f"无法获取磁盘使用率: {e}")
                disk_usage = 0.0
            
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage=disk_usage,
                process_count=len(psutil.pids()),
                timestamp=datetime.now()
            )
            
            # 线程安全地更新历史记录
            with self._metrics_lock:
                self.metrics_history.append(metrics)
                
                # 只保留最近24小时的记录
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
            
            # 保存到文件
            self.save_metrics_to_file()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            return SystemMetrics(0, 0, 0, 0, datetime.now())
    
    def save_metrics_to_file(self):
        """保存性能指标到文件（线程安全）"""
        try:
            with self._metrics_lock:
                metrics_data = []
                for metric in self.metrics_history[-100:]:  # 只保存最近100条
                    metrics_data.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'cpu_percent': metric.cpu_percent,
                        'memory_percent': metric.memory_percent,
                        'disk_usage': metric.disk_usage,
                        'process_count': metric.process_count
                    })
            
            # 使用临时文件确保原子写入
            temp_file = self.metrics_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # 原子替换
            if self.is_windows:
                if self.metrics_file.exists():
                    os.remove(self.metrics_file)
            temp_file.replace(self.metrics_file)
                
        except Exception as e:
            self.logger.error(f"保存性能指标失败: {e}")
            # 清理临时文件
            if temp_file.exists():
                os.remove(temp_file)
    
    def auto_fetch_and_commit_data(self) -> Dict[str, Any]:
        """自动拉取最新数据并提交"""
        try:
            self.logger.info("🔄 开始自动数据拉取和提交流程...")
            
            result = {
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'fetch_success': False,
                'commit_success': False,
                'data_updated': False,
                'error': None
            }
            
            # 步骤1: 拉取最新数据
            self.logger.info("📥 拉取最新数据...")
            try:
                from src.data.data_fetch import main as fetch_main
                fetch_result = fetch_main()
                
                if isinstance(fetch_result, dict):
                    result['fetch_success'] = fetch_result.get('code') == 200
                    result['data_updated'] = fetch_result.get('updated', False)
                else:
                    result['fetch_success'] = bool(fetch_result)
                    result['data_updated'] = True
                
                if result['fetch_success']:
                    self.logger.info("✅ 数据拉取成功")
                    with self._state_lock:
                        self.state['data_fetch_count'] += 1
                        self.state['last_data_fetch'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    self.logger.error("❌ 数据拉取失败")
                    return result
                    
            except Exception as e:
                self.logger.error(f"数据拉取异常: {e}")
                result['error'] = f"数据拉取失败: {str(e)}"
                return result
            
            # 步骤2: 数据质量检查
            self.logger.info("🔍 数据质量检查...")
            quality_check = self.check_data_quality()
            
            if not quality_check['success']:
                self.logger.warning(f"数据质量检查未通过: {quality_check['error']}")
                # 质量检查失败不影响整体流程，但会记录警告
            
            # 步骤3: Git提交（如果有更新的数据）
            if result['data_updated'] and self.git_repo:
                self.logger.info("📤 提交数据到Git仓库...")
                try:
                    # 添加数据文件
                    data_dir = self.work_dir / 'data'
                    if data_dir.exists():
                        self.git_repo.index.add([str(data_dir)])
                    
                    # 检查是否有变更
                    if self.git_repo.is_dirty():
                        commit_message = f"自动数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        self.git_repo.index.commit(commit_message)
                        self.logger.info("✅ 数据已提交到Git仓库")
                        result['commit_success'] = True
                    else:
                        self.logger.info("ℹ️ 没有数据变更，跳过Git提交")
                        result['commit_success'] = True
                        
                except git.exc.GitError as e:
                    self.logger.error(f"Git提交失败: {e}")
                    result['error'] = f"Git提交失败: {str(e)}"
                except Exception as e:
                    self.logger.error(f"Git提交异常: {e}")
                    result['error'] = f"Git提交异常: {str(e)}"
            else:
                result['commit_success'] = True  # 没有Git或没有更新时认为成功
            
            result['success'] = result['fetch_success'] and result['commit_success']
            
            if result['success']:
                self.logger.info("✅ 自动数据拉取和提交流程完成")
            else:
                self.logger.error("❌ 自动数据拉取和提交流程失败")
            
            return result
            
        except Exception as e:
            self.logger.error(f"自动数据拉取和提交异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_data_quality(self) -> Dict[str, Any]:
        """数据质量检查"""
        try:
            # 获取最新数据
            latest_data = self.data_module.get_latest_data()
            
            if latest_data is None or latest_data.empty:
                return {
                    'success': False,
                    'error': '无最新数据'
                }
            
            # 检查数据完整性
            required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_fields = []
            
            for field in required_fields:
                if field not in latest_data or pd.isna(latest_data[field]):
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'缺失字段: {", ".join(missing_fields)}'
                }
            
            # 检查数据合理性
            if latest_data['high'] < latest_data['low']:
                return {
                    'success': False,
                    'error': '最高价小于最低价'
                }
            
            if latest_data['close'] <= 0 or latest_data['volume'] < 0:
                return {
                    'success': False,
                    'error': '价格或成交量数据异常'
                }
            
            return {
                'success': True,
                'latest_date': latest_data['date'],
                'data_points': len(latest_data) if hasattr(latest_data, '__len__') else 1
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """系统健康检查"""
        try:
            current_metrics = self.collect_performance_metrics()
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_healthy': True,
                'warnings': [],
                'errors': [],
                'metrics': {
                    'cpu_percent': current_metrics.cpu_percent,
                    'memory_percent': current_metrics.memory_percent,
                    'disk_usage': current_metrics.disk_usage,
                    'process_count': current_metrics.process_count
                }
            }
            
            # CPU检查
            if current_metrics.cpu_percent > 80:
                health_status['errors'].append(f"CPU使用率过高: {current_metrics.cpu_percent:.1f}%")
                health_status['overall_healthy'] = False
            elif current_metrics.cpu_percent > 60:
                health_status['warnings'].append(f"CPU使用率较高: {current_metrics.cpu_percent:.1f}%")
            
            # 内存检查
            if current_metrics.memory_percent > 85:
                health_status['errors'].append(f"内存使用率过高: {current_metrics.memory_percent:.1f}%")
                health_status['overall_healthy'] = False
            elif current_metrics.memory_percent > 70:
                health_status['warnings'].append(f"内存使用率较高: {current_metrics.memory_percent:.1f}%")
            
            # 磁盘检查
            if current_metrics.disk_usage > 90:
                health_status['errors'].append(f"磁盘使用率过高: {current_metrics.disk_usage:.1f}%")
                health_status['overall_healthy'] = False
            elif current_metrics.disk_usage > 80:
                health_status['warnings'].append(f"磁盘使用率较高: {current_metrics.disk_usage:.1f}%")
            
            # 连续错误检查
            if self.state['consecutive_errors'] >= 5:
                health_status['errors'].append(f"连续错误次数过多: {self.state['consecutive_errors']}")
                health_status['overall_healthy'] = False
            elif self.state['consecutive_errors'] >= 3:
                health_status['warnings'].append(f"连续错误次数较多: {self.state['consecutive_errors']}")
            
            # 数据新鲜度检查
            if self.state['last_data_fetch']:
                last_fetch = datetime.fromisoformat(self.state['last_data_fetch'].replace(' ', 'T'))
                hours_since_fetch = (datetime.now() - last_fetch).total_seconds() / 3600
                
                if hours_since_fetch > 48:
                    health_status['errors'].append(f"数据更新滞后: {hours_since_fetch:.1f}小时")
                    health_status['overall_healthy'] = False
                elif hours_since_fetch > 24:
                    health_status['warnings'].append(f"数据更新较晚: {hours_since_fetch:.1f}小时")
            
            # 发送告警（如果有严重问题）
            if health_status['errors']:
                self.send_health_alert(health_status)
            
            self.last_health_check = datetime.now()
            return health_status
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_healthy': False,
                'errors': [f"健康检查异常: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    def send_health_alert(self, health_status: Dict[str, Any]):
        """发送健康告警"""
        try:
            alert_message = f"""
⚠️ 系统健康告警

时间: {health_status['timestamp']}
状态: {'健康' if health_status['overall_healthy'] else '异常'}

错误信息:
{chr(10).join(['❌ ' + error for error in health_status['errors']])}

警告信息:
{chr(10).join(['⚠️ ' + warning for warning in health_status['warnings']])}

系统指标:
- CPU: {health_status['metrics'].get('cpu_percent', 0):.1f}%
- 内存: {health_status['metrics'].get('memory_percent', 0):.1f}%
- 磁盘: {health_status['metrics'].get('disk_usage', 0):.1f}%
"""
            
            # 使用通知模块发送告警
            self.notification_manager._send_console_notification({
                'subject': '🚨 系统健康告警',
                'body': alert_message
            })
            
            self.logger.warning("已发送系统健康告警")
            
        except Exception as e:
            self.logger.error(f"发送健康告警失败: {e}")
    
    def auto_backup_data(self) -> Dict[str, Any]:
        """自动数据备份"""
        try:
            self.logger.info("💾 开始自动数据备份...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_subdir = self.backup_dir / f'backup_{timestamp}'
            backup_subdir.mkdir(exist_ok=True)
            
            result = {
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'backup_path': str(backup_subdir),
                'backed_up_items': [],
                'total_size': 0,
                'error': None
            }
            
            # 备份重要目录和文件
            backup_targets = [
                ('data', self.work_dir / 'data'),
                ('models', self.work_dir / 'models'),
                ('results', self.work_dir / 'results'),
                ('config', self.work_dir / 'config'),
                ('logs', self.work_dir / 'logs')
            ]
            
            for name, source_path in backup_targets:
                if source_path.exists():
                    target_path = backup_subdir / name
                    
                    try:
                        if source_path.is_dir():
                            shutil.copytree(source_path, target_path, ignore_dangling_symlinks=True)
                        else:
                            shutil.copy2(source_path, target_path)
                        
                        # 计算大小
                        if target_path.is_dir():
                            size = sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file())
                        else:
                            size = target_path.stat().st_size
                        
                        result['backed_up_items'].append({
                            'name': name,
                            'size': size,
                            'path': str(target_path)
                        })
                        result['total_size'] += size
                        
                        self.logger.info(f"✅ 已备份: {name} ({size/1024/1024:.1f}MB)")
                        
                    except Exception as e:
                        self.logger.error(f"备份 {name} 失败: {e}")
                        continue
            
            # 创建备份清单
            manifest = {
                'backup_time': timestamp,
                'total_items': len(result['backed_up_items']),
                'total_size': result['total_size'],
                'items': result['backed_up_items'],
                'created_by': 'EnhancedDailyTradingBot'
            }
            
            manifest_file = backup_subdir / 'backup_manifest.json'
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # 清理旧备份（保留最近10个）
            self.cleanup_old_backups(keep_count=10)
            
            result['success'] = True
            self.state['backup_count'] += 1
            self.state['last_backup'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info(f"✅ 数据备份完成: {result['total_size']/1024/1024:.1f}MB，共{len(result['backed_up_items'])}项")
            
            return result
            
        except Exception as e:
            self.logger.error(f"自动数据备份失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """清理旧备份"""
        try:
            backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir() and d.name.startswith('backup_')]
            backup_dirs.sort(key=lambda x: x.name, reverse=True)
            
            # 删除超出保留数量的备份
            for old_backup in backup_dirs[keep_count:]:
                shutil.rmtree(old_backup)
                self.logger.info(f"已删除旧备份: {old_backup.name}")
                
        except Exception as e:
            self.logger.error(f"清理旧备份失败: {e}")
    
    def restore_from_backup(self, backup_timestamp: str) -> Dict[str, Any]:
        """从备份恢复数据"""
        try:
            backup_path = self.backup_dir / f'backup_{backup_timestamp}'
            
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'备份不存在: {backup_timestamp}'
                }
            
            # 读取备份清单
            manifest_file = backup_path / 'backup_manifest.json'
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                self.logger.info(f"恢复备份: {manifest['backup_time']}，共{manifest['total_items']}项")
            
            # 恢复主要目录
            restore_targets = ['data', 'models', 'config']
            restored_items = []
            
            for target in restore_targets:
                source = backup_path / target
                dest = self.work_dir / target
                
                if source.exists():
                    # 备份当前版本
                    if dest.exists():
                        backup_current = dest.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        if dest.is_dir():
                            shutil.copytree(dest, backup_current)
                        else:
                            shutil.copy2(dest, backup_current)
                    
                    # 恢复数据
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    
                    if source.is_dir():
                        shutil.copytree(source, dest)
                    else:
                        shutil.copy2(source, dest)
                    
                    restored_items.append(target)
                    self.logger.info(f"✅ 已恢复: {target}")
            
            return {
                'success': True,
                'restored_items': restored_items,
                'backup_timestamp': backup_timestamp
            }
            
        except Exception as e:
            self.logger.error(f"数据恢复失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def daily_workflow(self) -> Dict[str, Any]:
        """执行增强版日常工作流程"""
        today = datetime.now().strftime('%Y-%m-%d')
        workflow_result = {
            'date': today,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'steps': {},
            'errors': []
        }
        
        try:
            self.logger.info(f"🚀 开始执行增强版日常交易流程: {today}")
            
            # 步骤1: 系统健康检查
            self.logger.info("🏥 步骤1: 系统健康检查")
            health_result = self.check_system_health()
            workflow_result['steps']['health_check'] = health_result
            
            if not health_result['overall_healthy']:
                self.logger.warning("系统健康状态异常，但继续执行")
            
            # 步骤2: 自动数据拉取和提交
            self.logger.info("📡 步骤2: 自动数据拉取和提交")
            data_fetch_result = self.auto_fetch_and_commit_data()
            workflow_result['steps']['data_fetch'] = data_fetch_result
            
            if not data_fetch_result['success']:
                workflow_result['errors'].append("数据拉取和提交失败")
                # 数据拉取失败不中断流程，使用现有数据继续
            
            # 步骤3: 数据检查和更新
            self.logger.info("📊 步骤3: 检查数据状态")
            data_result = self.check_and_update_data()
            workflow_result['steps']['data_check'] = data_result
            
            if not data_result['success']:
                workflow_result['errors'].append("数据检查失败")
                return workflow_result
            
            # 步骤4: 增量训练
            self.logger.info("🤖 步骤4: 执行增量训练")
            training_result = self.execute_incremental_training(today)
            workflow_result['steps']['training'] = training_result
            
            if training_result['success']:
                self.state['training_count'] += 1
                self.state['last_training_date'] = today
            
            # 步骤5: 预测执行
            self.logger.info("🔮 步骤5: 执行预测")
            prediction_result = self.execute_prediction(today)
            workflow_result['steps']['prediction'] = prediction_result
            
            if prediction_result['success']:
                self.state['total_predictions'] += 1
                self.state['last_prediction_date'] = today
            
            # 步骤6: 交易信号生成
            self.logger.info("📈 步骤6: 生成交易信号")
            signal_result = self.generate_trading_signal(prediction_result)
            workflow_result['steps']['signal'] = signal_result
            
            # 步骤7: 结果记录
            self.logger.info("📝 步骤7: 记录结果")
            record_result = self.record_results(workflow_result)
            workflow_result['steps']['record'] = record_result
            
            # 步骤8: 发送通知
            self.logger.info("📨 步骤8: 发送通知")
            notification_result = self.send_notifications(workflow_result)
            workflow_result['steps']['notification'] = notification_result
            
            # 步骤9: 自动备份（每周执行一次）
            if datetime.now().weekday() == 6:  # 周日执行备份
                self.logger.info("💾 步骤9: 自动数据备份")
                backup_result = self.auto_backup_data()
                workflow_result['steps']['backup'] = backup_result
            
            # 检查整体成功状态
            workflow_result['success'] = all([
                data_result['success'],
                training_result['success'] or training_result.get('skipped', False),
                prediction_result['success'],
                signal_result['success']
            ])
            
            if workflow_result['success']:
                self.state['consecutive_errors'] = 0
                self.state['successful_predictions'] += 1
                self.logger.info("✅ 增强版日常交易流程执行成功")
            else:
                self.state['consecutive_errors'] += 1
                self.logger.error("❌ 增强版日常交易流程执行失败")
            
        except Exception as e:
            self.logger.error(f"日常工作流程异常: {e}")
            workflow_result['errors'].append(f"工作流程异常: {str(e)}")
            self.state['consecutive_errors'] += 1
        
        finally:
            self.save_state()
        
        return workflow_result
    
    def check_and_update_data(self) -> Dict[str, Any]:
        """检查和更新数据"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 检查最新数据日期
            latest_data = self.data_module.get_latest_data()
            
            if latest_data is None or latest_data.empty:
                return {
                    'success': False,
                    'error': '无法获取最新数据',
                    'latest_date': None
                }
            
            # latest_data 是 pd.Series
            if latest_data is not None:
                latest_date = latest_data['date']
                if isinstance(latest_date, str):
                    latest_date = datetime.strptime(latest_date, '%Y-%m-%d').strftime('%Y-%m-%d')
                else:
                    latest_date = latest_date.strftime('%Y-%m-%d')
            else:
                latest_date = None
            
            # 检查数据是否足够新
            if latest_date:
                days_behind = (datetime.now() - datetime.strptime(latest_date, '%Y-%m-%d')).days
                data_fresh = days_behind <= 1
            else:
                days_behind = 999
                data_fresh = False
            
            return {
                'success': True,
                'latest_date': latest_date,
                'days_behind': days_behind,
                'data_fresh': data_fresh,
                'records_count': 1 if latest_data is not None else 0
            }
            
        except Exception as e:
            self.logger.error(f"数据检查失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_incremental_training(self, target_date: str) -> Dict[str, Any]:
        """执行增量训练"""
        try:
            # 检查是否需要训练
            if self.state['last_training_date'] == target_date:
                self.logger.info("今日已完成训练，跳过")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': '今日已训练'
                }
            
            # 获取训练数据
            end_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=30)  # 获取最近30天数据
            
            training_data = self.data_module.get_history_data(
                start_dt.strftime('%Y-%m-%d'),
                target_date
            )
            
            if training_data.empty:
                return {
                    'success': False,
                    'error': '无法获取训练数据'
                }
            
            # 预处理数据
            training_data = self.data_module.preprocess_data(training_data)
            
            # 执行增量训练
            train_result = self.ai_improved.incremental_train(training_data, self.strategy_module)
            
            return {
                'success': train_result['success'],
                'method': train_result.get('method', 'unknown'),
                'update_count': train_result.get('update_count', 0),
                'new_samples': train_result.get('new_samples', 0),
                'error': train_result.get('error', None)
            }
            
        except Exception as e:
            self.logger.error(f"增量训练失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_prediction(self, target_date: str) -> Dict[str, Any]:
        """执行预测"""
        try:
            # 获取预测数据
            end_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=100)  # 使用最近100天数据
            
            prediction_data = self.data_module.get_history_data(
                start_dt.strftime('%Y-%m-%d'),
                target_date
            )
            
            if prediction_data.empty:
                return {
                    'success': False,
                    'error': '无法获取预测数据'
                }
            
            # 预处理数据
            prediction_data = self.data_module.preprocess_data(prediction_data)
            
            # 执行预测
            pred_result = self.ai_improved.predict_low_point(prediction_data, target_date)
            
            return {
                'success': True,
                'is_low_point': pred_result.get('is_low_point', False),
                'confidence': pred_result.get('confidence', 0.0),
                'final_confidence': pred_result.get('final_confidence', 0.0),
                'model_type': pred_result.get('model_type', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"预测执行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_trading_signal(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        try:
            if not prediction_result['success']:
                return {
                    'success': False,
                    'error': '预测失败，无法生成信号'
                }
            
            is_low_point = prediction_result['is_low_point']
            confidence = prediction_result['final_confidence']
            
            # 信号生成逻辑
            signal = {
                'action': 'HOLD',  # 默认持有
                'strength': 0,     # 信号强度 (0-5)
                'reason': '',
                'risk_level': 'MEDIUM'
            }
            
            if is_low_point:
                if confidence >= 0.8:
                    signal.update({
                        'action': 'BUY_STRONG',
                        'strength': 5,
                        'reason': f'强烈买入信号 (置信度: {confidence:.3f})',
                        'risk_level': 'LOW'
                    })
                elif confidence >= 0.6:
                    signal.update({
                        'action': 'BUY',
                        'strength': 4,
                        'reason': f'买入信号 (置信度: {confidence:.3f})',
                        'risk_level': 'LOW'
                    })
                elif confidence >= 0.4:
                    signal.update({
                        'action': 'BUY_WEAK',
                        'strength': 2,
                        'reason': f'弱买入信号 (置信度: {confidence:.3f})',
                        'risk_level': 'MEDIUM'
                    })
                else:
                    signal.update({
                        'action': 'WATCH',
                        'strength': 1,
                        'reason': f'观望 (置信度较低: {confidence:.3f})',
                        'risk_level': 'HIGH'
                    })
            else:
                if confidence < 0.2:
                    signal.update({
                        'action': 'HOLD',
                        'strength': 1,
                        'reason': f'继续持有 (非低点，置信度: {confidence:.3f})',
                        'risk_level': 'MEDIUM'
                    })
                else:
                    signal.update({
                        'action': 'WAIT',
                        'strength': 0,
                        'reason': f'等待机会 (非低点，置信度: {confidence:.3f})',
                        'risk_level': 'MEDIUM'
                    })
            
            return {
                'success': True,
                'signal': signal,
                'confidence': confidence,
                'is_low_point': is_low_point
            }
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def record_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """记录结果"""
        try:
            # 加载历史记录
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加新记录
            history.append(workflow_result)
            
            # 只保留最近100天的记录
            history = history[-100:]
            
            # 保存记录
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            # 生成日报告
            report_result = self.generate_daily_report(workflow_result)
            
            return {
                'success': True,
                'history_records': len(history),
                'report_generated': report_result['success']
            }
            
        except Exception as e:
            self.logger.error(f"结果记录失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_daily_report(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成日报告"""
        try:
            date = workflow_result['date']
            report_file = self.results_dir / f'daily_report_{date}.md'
            
            # 提取关键信息
            prediction = workflow_result['steps'].get('prediction', {})
            signal = workflow_result['steps'].get('signal', {})
            training = workflow_result['steps'].get('training', {})
            
            # 生成报告内容
            report_content = f"""# 日常交易报告 - {date}

## 📊 执行摘要
- **执行时间**: {workflow_result['timestamp']}
- **整体状态**: {'✅ 成功' if workflow_result['success'] else '❌ 失败'}
- **连续成功**: {self.state['total_predictions'] - self.state['consecutive_errors']} 天

## 🤖 AI预测结果
- **预测结果**: {'📈 相对低点' if prediction.get('is_low_point', False) else '📉 非相对低点'}
- **原始置信度**: {prediction.get('confidence', 0):.4f}
                - **最终置信度**: {prediction.get('final_confidence', 0):.4f}
- **模型类型**: {prediction.get('model_type', 'N/A')}

## 📈 交易信号
- **建议操作**: {signal.get('signal', {}).get('action', 'N/A')}
- **信号强度**: {signal.get('signal', {}).get('strength', 0)}/5
- **风险等级**: {signal.get('signal', {}).get('risk_level', 'N/A')}
- **操作理由**: {signal.get('signal', {}).get('reason', 'N/A')}

## 🔄 模型训练
- **训练状态**: {'✅ 成功' if training.get('success', False) else ('⏭️ 跳过' if training.get('skipped', False) else '❌ 失败')}
- **训练方式**: {training.get('method', 'N/A')}
- **更新次数**: {training.get('update_count', 'N/A')}
- **新增样本**: {training.get('new_samples', 'N/A')}

## 📊 统计信息
- **总预测次数**: {self.state['total_predictions']}
- **成功预测次数**: {self.state['successful_predictions']}
- **训练执行次数**: {self.state['training_count']}
- **连续错误次数**: {self.state['consecutive_errors']}
- **运行天数**: {(datetime.now() - datetime.strptime(self.state['start_date'], '%Y-%m-%d')).days + 1}

## ⚠️ 注意事项
"""
            
            # 添加错误信息
            if workflow_result['errors']:
                report_content += "\n### 错误信息:\n"
                for error in workflow_result['errors']:
                    report_content += f"- ❌ {error}\n"
            
            # 添加建议
            report_content += "\n### 💡 建议:\n"
            confidence = prediction.get('final_confidence', 0)
            if confidence >= 0.6:
                report_content += "- 置信度较高，可考虑适当加仓\n"
            elif confidence >= 0.4:
                report_content += "- 置信度中等，建议谨慎操作\n"
            else:
                report_content += "- 置信度较低，建议观望等待\n"
            
            if self.state['consecutive_errors'] >= 3:
                report_content += "- ⚠️ 连续错误较多，建议检查系统状态\n"
            
            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return {
                'success': True,
                'report_file': str(report_file)
            }
            
        except Exception as e:
            self.logger.error(f"生成日报告失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_notifications(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """发送通知"""
        try:
            # 构建通知消息
            date = workflow_result['date']
            prediction = workflow_result['steps'].get('prediction', {})
            signal = workflow_result['steps'].get('signal', {})
            
            # 基础信息
            status_emoji = "✅" if workflow_result['success'] else "❌"
            prediction_emoji = "📈" if prediction.get('is_low_point', False) else "📉"
            
            subject = f"{status_emoji} 交易信号 {date}"
            
            message = f"""
{status_emoji} 日常交易流程执行完成

📅 日期: {date}
{prediction_emoji} 预测: {'相对低点' if prediction.get('is_low_point', False) else '非相对低点'}
🎯 置信度: {prediction.get('final_confidence', 0):.3f}
📈 建议: {signal.get('signal', {}).get('action', 'N/A')}
⭐ 强度: {signal.get('signal', {}).get('strength', 0)}/5

📊 统计: 总计{self.state['total_predictions']}次预测，成功{self.state['successful_predictions']}次
"""
            
            # 发送通知
            try:
                # 构建通知结果格式，符合NotificationModule的API
                notification_result = {
                    'is_low_point': prediction.get('is_low_point', False),
                    'date': date,
                    'confidence': prediction.get('final_confidence', 0),
                    'price': 0,  # 这里可以从数据中获取价格
                    'reasons': [signal.get('signal', {}).get('reason', '未知原因')]
                }
                
                # 始终发送通知（包括非低点的日常总结）
                if prediction.get('is_low_point', False):
                    # 发送低点通知
                    success = self.notification_manager.send_low_point_notification(notification_result)
                else:
                    # 发送控制台通知作为日常总结
                    self.notification_manager._send_console_notification({
                        'subject': subject,
                        'body': message
                    })
                    success = True
                
                return {'success': success}
            except Exception as e:
                self.logger.warning(f"通知发送失败: {e}")
                return {'success': False, 'error': str(e)}
            
        except Exception as e:
            self.logger.error(f"通知处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            'bot_state': self.state.copy(),
            'last_run': datetime.now().isoformat(),
            'config_loaded': bool(self.config),
            'modules_status': {
                'data_module': bool(self.data_module),
                'ai_module': bool(self.ai_improved),
                'strategy_module': bool(self.strategy_module)
            }
        }
    
    def run_daemon_mode(self):
        """运行守护进程模式"""
        self.logger.info("🛡️ 启动守护进程模式...")
        
        # 写入PID文件
        self.write_pid_file()
        
        try:
            # 设置定时任务
            self.setup_scheduled_tasks()
            
            self.logger.info("⏰ 定时任务已设置，进入守护进程循环...")
            
            while self.running:
                try:
                    # 执行定时任务
                    schedule.run_pending()
                    
                    # 每30分钟进行一次健康检查
                    if (datetime.now() - self.last_health_check).total_seconds() > 1800:
                        threading.Thread(target=self.check_system_health, daemon=True).start()
                    
                    # 短暂休眠
                    time.sleep(60)  # 每分钟检查一次
                    
                except Exception as e:
                    self.logger.error(f"守护进程循环异常: {e}")
                    time.sleep(300)  # 异常时等待5分钟后继续
            
            self.logger.info("守护进程正常退出")
            
        except KeyboardInterrupt:
            self.logger.info("接收到中断信号，正在停止守护进程...")
        except Exception as e:
            self.logger.error(f"守护进程异常: {e}")
        finally:
            self.cleanup_daemon()
            
        # 检查是否需要重启
        if self.restart_flag:
            self.logger.info("重启守护进程...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    
    def setup_scheduled_tasks(self):
        """设置定时任务"""
        # 清除现有任务
        schedule.clear()
        
        # 每天15:05自动拉取数据并提交
        schedule.every().day.at("15:05").do(self._scheduled_data_fetch)
        
        # 每天09:30执行日常交易流程
        schedule.every().day.at("09:30").do(self._scheduled_daily_workflow)
        
        # 每天01:00执行系统健康检查
        schedule.every().day.at("01:00").do(self._scheduled_health_check)
        
        # 每周日02:00执行数据备份
        schedule.every().sunday.at("02:00").do(self._scheduled_backup)
        
        # 每小时收集性能指标
        schedule.every().hour.do(self._scheduled_metrics_collection)
        
        self.logger.info("✅ 定时任务设置完成:")
        self.logger.info("   - 15:05 自动数据拉取和提交")
        self.logger.info("   - 09:30 日常交易流程")
        self.logger.info("   - 01:00 系统健康检查")
        self.logger.info("   - 周日02:00 数据备份")
        self.logger.info("   - 每小时性能指标收集")
    
    def _scheduled_data_fetch(self):
        """定时数据拉取任务"""
        try:
            self.logger.info("⏰ 执行定时数据拉取任务...")
            result = self.auto_fetch_and_commit_data()
            if result['success']:
                self.logger.info("✅ 定时数据拉取任务完成")
            else:
                self.logger.error(f"❌ 定时数据拉取任务失败: {result.get('error', '未知错误')}")
        except Exception as e:
            self.logger.error(f"定时数据拉取任务异常: {e}")
    
    def _scheduled_daily_workflow(self):
        """定时日常交易流程任务"""
        try:
            self.logger.info("⏰ 执行定时日常交易流程...")
            result = self.daily_workflow()
            if result['success']:
                self.logger.info("✅ 定时日常交易流程完成")
            else:
                self.logger.error("❌ 定时日常交易流程失败")
        except Exception as e:
            self.logger.error(f"定时日常交易流程异常: {e}")
    
    def _scheduled_health_check(self):
        """定时健康检查任务"""
        try:
            self.logger.info("⏰ 执行定时健康检查...")
            result = self.check_system_health()
            if result['overall_healthy']:
                self.logger.info("✅ 系统健康状态良好")
            else:
                self.logger.warning("⚠️ 系统健康状态异常")
        except Exception as e:
            self.logger.error(f"定时健康检查异常: {e}")
    
    def _scheduled_backup(self):
        """定时备份任务"""
        try:
            self.logger.info("⏰ 执行定时备份任务...")
            result = self.auto_backup_data()
            if result['success']:
                self.logger.info("✅ 定时备份任务完成")
            else:
                self.logger.error(f"❌ 定时备份任务失败: {result.get('error', '未知错误')}")
        except Exception as e:
            self.logger.error(f"定时备份任务异常: {e}")
    
    def _scheduled_metrics_collection(self):
        """定时性能指标收集任务"""
        try:
            metrics = self.collect_performance_metrics()
            self.logger.debug(f"性能指标收集完成: CPU {metrics.cpu_percent:.1f}%, 内存 {metrics.memory_percent:.1f}%")
        except Exception as e:
            self.logger.error(f"性能指标收集异常: {e}")
    
    def cleanup_daemon(self):
        """清理守护进程资源"""
        try:
            self.remove_pid_file()
            self.logger.info("守护进程资源清理完成")
        except Exception as e:
            self.logger.error(f"守护进程资源清理失败: {e}")
    
    def run_scheduled(self):
        """运行定时任务（兼容原有接口）"""
        if self.daemon_mode:
            self.run_daemon_mode()
        else:
            self.logger.info("启动简单定时任务调度器")
            
            # 设置基本定时任务
            schedule.every().day.at("09:30").do(self.daily_workflow)
            schedule.every().day.at("15:05").do(self.auto_fetch_and_commit_data)
            
            self.logger.info("定时任务已设置: 每天09:30执行日常交易流程，15:05自动拉取数据")
            
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(60)  # 每分钟检查一次
            except KeyboardInterrupt:
                self.logger.info("接收到停止信号，退出定时任务")
            except Exception as e:
                self.logger.error(f"定时任务异常: {e}")


def main():
    """增强版主函数"""
    parser = argparse.ArgumentParser(description='增强版日常交易流程自动化机器人')
    parser.add_argument('--mode', choices=['run', 'schedule', 'daemon', 'status', 'backup', 'restore', 'health'], 
                       default='run', help='运行模式')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--date', help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--daemon', action='store_true', help='守护进程模式')
    parser.add_argument('--backup-timestamp', help='备份时间戳 (用于恢复)')
    
    args = parser.parse_args()
    
    try:
        # 初始化增强版机器人
        bot = EnhancedDailyTradingBot(args.config, daemon_mode=args.daemon or args.mode == 'daemon')
        
        if args.mode == 'run':
            # 单次执行
            print("🚀 执行单次增强版日常交易流程...")
            result = bot.daily_workflow()
            
            print("\n" + "="*60)
            print("📋 执行结果:")
            print(f"状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
            print(f"日期: {result['date']}")
            print(f"执行时间: {result['timestamp']}")
            
            # 健康检查结果
            if result['steps'].get('health_check'):
                health = result['steps']['health_check']
                print(f"系统健康: {'✅ 正常' if health['overall_healthy'] else '⚠️ 异常'}")
                if health['warnings']:
                    print(f"警告: {len(health['warnings'])}项")
                if health['errors']:
                    print(f"错误: {len(health['errors'])}项")
            
            # 数据拉取结果
            if result['steps'].get('data_fetch'):
                fetch = result['steps']['data_fetch']
                print(f"数据拉取: {'✅ 成功' if fetch['success'] else '❌ 失败'}")
                if fetch.get('data_updated'):
                    print("   📡 数据已更新")
                if fetch.get('commit_success'):
                    print("   📤 已提交到Git")
            
            # 预测结果
            if result['steps'].get('prediction', {}).get('success'):
                pred = result['steps']['prediction']
                print(f"预测: {'📈 相对低点' if pred['is_low_point'] else '📉 非相对低点'}")
                print(f"置信度: {pred['final_confidence']:.3f}")
            
            # 交易信号
            if result['steps'].get('signal', {}).get('success'):
                signal = result['steps']['signal']['signal']
                print(f"建议: {signal['action']} (强度: {signal['strength']}/5)")
                print(f"风险: {signal['risk_level']}")
            
            # 备份结果
            if result['steps'].get('backup'):
                backup = result['steps']['backup']
                if backup['success']:
                    print(f"备份: ✅ 完成 ({backup['total_size']/1024/1024:.1f}MB)")
            
            # 错误信息
            if result['errors']:
                print("错误:")
                for error in result['errors']:
                    print(f"  ❌ {error}")
            
            print("="*60)
            
        elif args.mode == 'schedule':
            # 简单定时执行
            print("⏰ 启动简单定时任务模式...")
            bot.run_scheduled()
            
        elif args.mode == 'daemon':
            # 守护进程模式
            print("🛡️ 启动守护进程模式...")
            bot.run_daemon_mode()
            
        elif args.mode == 'status':
            # 状态查询
            status = bot.get_status_report()
            print("\n" + "="*60)
            print("📊 增强版机器人状态报告")
            print("="*60)
            
            # 基本状态
            print(f"总预测次数: {status['bot_state']['total_predictions']}")
            print(f"成功次数: {status['bot_state']['successful_predictions']}")
            print(f"训练次数: {status['bot_state']['training_count']}")
            print(f"数据拉取次数: {status['bot_state']['data_fetch_count']}")
            print(f"备份次数: {status['bot_state']['backup_count']}")
            
            # 最后执行时间
            print(f"\n最后执行时间:")
            print(f"  训练: {status['bot_state']['last_training_date'] or '无'}")
            print(f"  预测: {status['bot_state']['last_prediction_date'] or '无'}")
            print(f"  数据拉取: {status['bot_state']['last_data_fetch'] or '无'}")
            print(f"  备份: {status['bot_state']['last_backup'] or '无'}")
            
            # 系统状态
            print(f"\n系统状态:")
            print(f"  连续错误: {status['bot_state']['consecutive_errors']}")
            print(f"  运行时长: {status['bot_state']['uptime_start']}")
            
            # 模块状态
            modules = status['modules_status']
            print(f"\n模块状态:")
            print(f"  数据模块: {'✅' if modules['data_module'] else '❌'}")
            print(f"  AI模块: {'✅' if modules['ai_module'] else '❌'}")
            print(f"  策略模块: {'✅' if modules['strategy_module'] else '❌'}")
            
            # 健康检查
            health = bot.check_system_health()
            print(f"\n系统健康检查:")
            print(f"  整体状态: {'✅ 健康' if health['overall_healthy'] else '⚠️ 异常'}")
            print(f"  CPU使用率: {health['metrics']['cpu_percent']:.1f}%")
            print(f"  内存使用率: {health['metrics']['memory_percent']:.1f}%")
            print(f"  磁盘使用率: {health['metrics']['disk_usage']:.1f}%")
            
            if health['warnings']:
                print(f"\n⚠️ 警告 ({len(health['warnings'])}项):")
                for warning in health['warnings']:
                    print(f"  - {warning}")
            
            if health['errors']:
                print(f"\n❌ 错误 ({len(health['errors'])}项):")
                for error in health['errors']:
                    print(f"  - {error}")
            
            if not health['warnings'] and not health['errors']:
                print("\n✅ 未发现问题")
        
        elif args.mode == 'backup':
            # 手动备份
            print("💾 执行手动数据备份...")
            result = bot.auto_backup_data()
            
            if result['success']:
                print(f"✅ 备份完成!")
                print(f"备份路径: {result['backup_path']}")
                print(f"备份大小: {result['total_size']/1024/1024:.1f}MB")
                print(f"备份项目: {len(result['backed_up_items'])}个")
                
                for item in result['backed_up_items']:
                    print(f"  - {item['name']}: {item['size']/1024/1024:.1f}MB")
            else:
                print(f"❌ 备份失败: {result.get('error', '未知错误')}")
                
        elif args.mode == 'restore':
            # 数据恢复
            if not args.backup_timestamp:
                print("❌ 恢复模式需要指定备份时间戳: --backup-timestamp YYYYMMDD_HHMMSS")
                return 1
            
            print(f"🔄 从备份恢复数据: {args.backup_timestamp}")
            result = bot.restore_from_backup(args.backup_timestamp)
            
            if result['success']:
                print(f"✅ 恢复完成!")
                print(f"恢复项目: {', '.join(result['restored_items'])}")
            else:
                print(f"❌ 恢复失败: {result.get('error', '未知错误')}")
                
        elif args.mode == 'health':
            # 健康检查
            print("🏥 执行系统健康检查...")
            health = bot.check_system_health()
            
            print(f"\n系统健康状态: {'✅ 健康' if health['overall_healthy'] else '⚠️ 异常'}")
            print(f"检查时间: {health['timestamp']}")
            
            print(f"\n📊 系统指标:")
            print(f"  CPU使用率: {health['metrics']['cpu_percent']:.1f}%")
            print(f"  内存使用率: {health['metrics']['memory_percent']:.1f}%")
            print(f"  磁盘使用率: {health['metrics']['disk_usage']:.1f}%")
            print(f"  进程数量: {health['metrics']['process_count']}")
            
            if health['warnings']:
                print(f"\n⚠️ 警告 ({len(health['warnings'])}项):")
                for warning in health['warnings']:
                    print(f"  - {warning}")
            
            if health['errors']:
                print(f"\n❌ 错误 ({len(health['errors'])}项):")
                for error in health['errors']:
                    print(f"  - {error}")
            
            if not health['warnings'] and not health['errors']:
                print("\n✅ 未发现问题")
        
        return 0
        
    except Exception as e:
        print(f"❌ 增强版机器人执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 