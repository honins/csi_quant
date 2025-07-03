#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日常交易流程自动化机器人
自动执行增量训练、预测、信号生成和结果记录
"""

import sys
import os
import json
import argparse
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.utils import load_config, setup_logging
from src.notification.notification_module import NotificationModule


class DailyTradingBot:
    """日常交易流程自动化机器人"""
    
    def __init__(self, config_path: str = None):
        """初始化交易机器人"""
        
        # 加载配置
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_improved.yaml')
        
        self.config = load_config(config_path=config_path)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化模块
        self.data_module = DataModule(self.config)
        self.strategy_module = StrategyModule(self.config)
        self.ai_improved = AIOptimizerImproved(self.config)
        self.notification_manager = NotificationModule(self.config)
        
        # 设置工作目录
        self.work_dir = Path(os.path.dirname(__file__)).parent
        self.results_dir = self.work_dir / 'results' / 'daily_trading'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 交易状态文件
        self.state_file = self.results_dir / 'bot_state.json'
        self.history_file = self.results_dir / 'trading_history.json'
        
        # 加载状态
        self.state = self.load_state()
        
        self.logger.info("日常交易机器人初始化完成")
    
    def setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'daily_trading_bot.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_state(self) -> Dict[str, Any]:
        """加载机器人状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"加载状态文件失败: {e}")
        
        return {
            'last_training_date': None,
            'last_prediction_date': None,
            'consecutive_errors': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'training_count': 0,
            'start_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def save_state(self):
        """保存机器人状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存状态文件失败: {e}")
    
    def daily_workflow(self) -> Dict[str, Any]:
        """执行日常工作流程"""
        today = datetime.now().strftime('%Y-%m-%d')
        workflow_result = {
            'date': today,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'steps': {},
            'errors': []
        }
        
        try:
            self.logger.info(f"🚀 开始执行日常交易流程: {today}")
            
            # 步骤1: 数据检查和更新
            self.logger.info("📊 步骤1: 检查数据更新")
            data_result = self.check_and_update_data()
            workflow_result['steps']['data_check'] = data_result
            
            if not data_result['success']:
                workflow_result['errors'].append("数据检查失败")
                return workflow_result
            
            # 步骤2: 增量训练
            self.logger.info("🤖 步骤2: 执行增量训练")
            training_result = self.execute_incremental_training(today)
            workflow_result['steps']['training'] = training_result
            
            if training_result['success']:
                self.state['training_count'] += 1
                self.state['last_training_date'] = today
            
            # 步骤3: 预测执行
            self.logger.info("🔮 步骤3: 执行预测")
            prediction_result = self.execute_prediction(today)
            workflow_result['steps']['prediction'] = prediction_result
            
            if prediction_result['success']:
                self.state['total_predictions'] += 1
                self.state['last_prediction_date'] = today
            
            # 步骤4: 交易信号生成
            self.logger.info("📈 步骤4: 生成交易信号")
            signal_result = self.generate_trading_signal(prediction_result)
            workflow_result['steps']['signal'] = signal_result
            
            # 步骤5: 结果记录
            self.logger.info("📝 步骤5: 记录结果")
            record_result = self.record_results(workflow_result)
            workflow_result['steps']['record'] = record_result
            
            # 步骤6: 发送通知
            self.logger.info("📨 步骤6: 发送通知")
            notification_result = self.send_notifications(workflow_result)
            workflow_result['steps']['notification'] = notification_result
            
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
                self.logger.info("✅ 日常交易流程执行成功")
            else:
                self.state['consecutive_errors'] += 1
                self.logger.error("❌ 日常交易流程执行失败")
            
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
    
    def run_scheduled(self):
        """运行定时任务"""
        self.logger.info("启动定时任务调度器")
        
        # 设置定时任务
        schedule.every().day.at("09:30").do(self.daily_workflow)  # 每天9:30执行
        
        self.logger.info("定时任务已设置: 每天09:30执行日常交易流程")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            self.logger.info("接收到停止信号，退出定时任务")
        except Exception as e:
            self.logger.error(f"定时任务异常: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='日常交易流程自动化机器人')
    parser.add_argument('--mode', choices=['run', 'schedule', 'status'], 
                       default='run', help='运行模式')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--date', help='指定日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # 初始化机器人
        bot = DailyTradingBot(args.config)
        
        if args.mode == 'run':
            # 单次执行
            print("🚀 执行单次日常交易流程...")
            result = bot.daily_workflow()
            
            print("\n" + "="*50)
            print("📋 执行结果:")
            print(f"状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
            print(f"日期: {result['date']}")
            
            if result['steps'].get('prediction', {}).get('success'):
                pred = result['steps']['prediction']
                print(f"预测: {'📈 相对低点' if pred['is_low_point'] else '📉 非相对低点'}")
                print(f"置信度: {pred['final_confidence']:.3f}")
            
            if result['steps'].get('signal', {}).get('success'):
                signal = result['steps']['signal']['signal']
                print(f"建议: {signal['action']} (强度: {signal['strength']}/5)")
            
            if result['errors']:
                print("错误:")
                for error in result['errors']:
                    print(f"  ❌ {error}")
            
            print("="*50)
            
        elif args.mode == 'schedule':
            # 定时执行
            print("⏰ 启动定时任务模式...")
            bot.run_scheduled()
            
        elif args.mode == 'status':
            # 状态查询
            status = bot.get_status_report()
            print("\n" + "="*50)
            print("📊 机器人状态报告")
            print("="*50)
            print(f"总预测次数: {status['bot_state']['total_predictions']}")
            print(f"成功次数: {status['bot_state']['successful_predictions']}")
            print(f"训练次数: {status['bot_state']['training_count']}")
            print(f"最后训练: {status['bot_state']['last_training_date'] or '无'}")
            print(f"最后预测: {status['bot_state']['last_prediction_date'] or '无'}")
            print(f"连续错误: {status['bot_state']['consecutive_errors']}")
            print("="*50)
        
        return 0
        
    except Exception as e:
        print(f"❌ 机器人执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 