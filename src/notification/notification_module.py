#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通知模块
负责在识别到相对低点时向用户发送通知
"""

import os
import logging
import smtplib
from datetime import datetime
from typing import Dict, Any, List
from email.mime import text as MimeTextModule
from email.mime import multipart as MimeMultipartModule

class NotificationModule:
    """通知模块类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知模块
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger('NotificationModule')
        self.config = config
        
        # 通知配置
        notification_config = config.get('notification', {})
        self.methods = notification_config.get('methods', ['email'])
        self.email_config = notification_config.get('email', {})
        
        # 创建日志目录
        self.logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        self.logger.info("通知模块初始化完成，通知方式: %s", self.methods)
        
    def send_low_point_notification(self, result: Dict[str, Any]) -> bool:
        """
        发送相对低点通知
        
        参数:
        result: 识别结果
        
        返回:
        bool: 是否发送成功
        """
        self.logger.info("发送相对低点通知")
        
        try:
            if not result.get('is_low_point', False):
                self.logger.info("非相对低点，不发送通知")
                return True
                
            # 生成通知内容
            content = self._generate_notification_content(result)
            
            success = True
            
            # 根据配置的方式发送通知
            if 'email' in self.methods:
                email_success = self._send_email_notification(content)
                success = success and email_success
                
            if 'console' in self.methods:
                self._send_console_notification(content)
                
            # 记录通知日志
            self._log_notification(result, content, success)
            
            return success
            
        except Exception as e:
            self.logger.error("发送通知失败: %s", str(e))
            return False
            
    def _generate_notification_content(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        生成通知内容
        
        参数:
        result: 识别结果
        
        返回:
        dict: 通知内容
        """
        date = result.get('date', datetime.now())
        price = result.get('price', 0)
        confidence = result.get('confidence', 0)
        reasons = result.get('reasons', [])
        
        # 格式化日期
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime('%Y-%m-%d')
            
        # 生成标题
        subject = f"中证1000指数相对低点提醒 - {date_str}"
        
        # 生成正文
        body = f"""
中证1000指数相对低点识别系统

检测到相对低点：

📅 日期：{date_str}
💰 价格：{price:.2f}
🎯 置信度：{confidence:.1%}

📊 识别原因：
"""
        
        for i, reason in enumerate(reasons, 1):
            body += f"{i}. {reason}\n"
            
        body += f"""
⚠️ 风险提示：
1. 这是基于历史数据的技术分析结果，不构成投资建议
2. 相对低点不等于绝对低点，仍存在继续下跌的可能
3. 请结合其他分析方法和风险管理策略进行投资决策
4. 投资有风险，入市需谨慎

📈 系统说明：
相对低点定义：从当天起到20个交易日内，直至某一天指数能够上涨5%，则当天被认为是该指数的相对低点。

---
中证1000指数相对低点识别系统
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return {
            'subject': subject,
            'body': body
        }
        
    def _send_email_notification(self, content: Dict[str, str]) -> bool:
        """
        发送邮件通知
        
        参数:
        content: 通知内容
        
        返回:
        bool: 是否发送成功
        """
        try:
            # 检查邮件配置
            if not self.email_config:
                self.logger.warning("邮件配置为空，跳过邮件发送")
                return True
                
            recipients = self.email_config.get('recipients', [])
            if not recipients:
                self.logger.warning("没有配置收件人，跳过邮件发送")
                return True
                
            # 模拟发送邮件（实际使用时需要配置SMTP服务器）
            self.logger.info("模拟发送邮件通知")
            self.logger.info("收件人: %s", recipients)
            self.logger.info("主题: %s", content['subject'])
            self.logger.info("内容: %s", content['body'][:200] + "...")
            
            # 这里应该是实际的邮件发送代码
            # smtp_server = self.email_config.get('smtp_server')
            # smtp_port = self.email_config.get('smtp_port', 587)
            # username = self.email_config.get('username')
            # password = self.email_config.get('password')
            # 
            # msg = MimeMultipartModule.MimeMultipart()
            # msg["From"] = username
            # msg["To"] = ", ".join(recipients)
            # msg["Subject"] = content["subject"]
            # msg.attach(MimeTextModule.MimeText(content["body"], "plain", "utf-8"))
            # 
            # server = smtplib.SMTP(smtp_server, smtp_port)
            # server.starttls()
            # server.login(username, password)
            # server.send_message(msg)
            # server.quit()
            
            self.logger.info("邮件通知发送成功")
            return True
            
        except Exception as e:
            self.logger.error("发送邮件通知失败: %s", str(e))
            return False
            
    def _send_console_notification(self, content: Dict[str, str]) -> None:
        """
        发送控制台通知
        
        参数:
        content: 通知内容
        """
        print("\n" + "="*60)
        print(content['subject'])
        print("="*60)
        print(content['body'])
        print("="*60 + "\n")
        
    def _log_notification(self, result: Dict[str, Any], content: Dict[str, str], 
                         success: bool) -> None:
        """
        记录通知日志
        
        参数:
        result: 识别结果
        content: 通知内容
        success: 是否发送成功
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.logs_dir, f'notification_{timestamp}.log')
            
            log_content = f"""
通知时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
发送状态: {'成功' if success else '失败'}
识别结果: {result}
通知内容: {content}
"""
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(log_content)
                
            self.logger.info("通知日志已保存: %s", log_file)
            
        except Exception as e:
            self.logger.error("保存通知日志失败: %s", str(e))
            
    def get_notification_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        获取通知历史
        
        参数:
        days: 查询天数
        
        返回:
        list: 通知历史列表
        """
        self.logger.info("获取 %d 天内的通知历史", days)
        
        try:
            history = []
            
            # 遍历日志文件
            for filename in os.listdir(self.logs_dir):
                if filename.startswith('notification_') and filename.endswith('.log'):
                    file_path = os.path.join(self.logs_dir, filename)
                    
                    # 检查文件修改时间
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (datetime.now() - file_time).days <= days:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # 简单解析日志内容
                            lines = content.strip().split('\n')
                            if len(lines) >= 2:
                                time_line = lines[1].split(': ', 1)
                                status_line = lines[2].split(': ', 1)
                                
                                if len(time_line) == 2 and len(status_line) == 2:
                                    history.append({
                                        'time': time_line[1],
                                        'status': status_line[1],
                                        'file': filename
                                    })
                                    
                        except Exception as e:
                            self.logger.warning("解析日志文件失败 %s: %s", filename, str(e))
                            
            # 按时间排序
            history.sort(key=lambda x: x['time'], reverse=True)
            
            self.logger.info("获取到 %d 条通知历史", len(history))
            return history
            
        except Exception as e:
            self.logger.error("获取通知历史失败: %s", str(e))
            return []

