#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略执行模块
实现相对低点识别算法和回测功能
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

class StrategyModule:
    """策略执行模块类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略模块
        
        参数:
        config: 配置字典
        """
        self.logger = logging.getLogger('StrategyModule')
        self.config = config
        
        # 策略参数
        strategy_config = config.get('strategy', {})
        self.rise_threshold = strategy_config.get('rise_threshold', 0.05)
        self.max_days = strategy_config.get('max_days', 20)
        
        # 创建结果目录
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.logger.info("策略模块初始化完成，参数: rise_threshold=%.4f, max_days=%d", 
                        self.rise_threshold, self.max_days)
        
    def identify_relative_low(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        识别相对低点
        
        参数:
        data: 市场数据
        
        返回:
        dict: 识别结果
        """
        self.logger.info("识别相对低点")
        
        try:
            if len(data) == 0:
                return {
                    'date': None,
                    'price': None,
                    'is_low_point': False,
                    'confidence': 0.0,
                    'reason': '数据为空'
                }
            
            # 获取最新日期的数据
            latest_data = data.iloc[-1]
            latest_date = latest_data['date']
            latest_price = latest_data['close']
            
            # 获取技术指标
            ma5 = latest_data.get('ma5', None)
            ma10 = latest_data.get('ma10', None)
            ma20 = latest_data.get('ma20', None)
            rsi = latest_data.get('rsi', None)
            macd = latest_data.get('macd', None)
            bb_lower = latest_data.get('bb_lower', None)
            
            # 判断是否可能是相对低点
            is_low_point = False
            confidence = 0.0
            reasons = []
            
            # 条件1: 价格低于多条移动平均线
            if ma5 is not None and ma10 is not None and ma20 is not None:
                if latest_price < ma5 and latest_price < ma10 and latest_price < ma20:
                    is_low_point = True
                    confidence += 0.3
                    reasons.append("价格低于MA5/MA10/MA20")
                elif latest_price < ma10 and latest_price < ma20:
                    confidence += 0.2
                    reasons.append("价格低于MA10/MA20")
                    
            # 条件2: RSI超卖
            if rsi is not None:
                if rsi < 30:
                    is_low_point = True
                    confidence += 0.3
                    reasons.append(f"RSI超卖({rsi:.2f})")
                elif rsi < 40:
                    confidence += 0.2
                    reasons.append(f"RSI偏低({rsi:.2f})")
                    
            # 条件3: MACD负值
            if macd is not None and macd < 0:
                confidence += 0.1
                reasons.append("MACD负值")
                
            # 条件4: 价格接近布林带下轨
            if bb_lower is not None and latest_price <= bb_lower * 1.02:
                is_low_point = True
                confidence += 0.2
                reasons.append("价格接近布林带下轨")
                
            # 条件5: 近期大幅下跌
            if len(data) >= 5:
                price_5d_ago = data.iloc[-6]['close'] if len(data) >= 6 else data.iloc[0]['close']
                decline_5d = (latest_price - price_5d_ago) / price_5d_ago
                if decline_5d < -0.05:  # 5天内下跌超过5%
                    confidence += 0.2
                    reasons.append(f"近5日大幅下跌({decline_5d:.2%})")
                    
            # 最终判断
            if confidence >= 0.5:
                is_low_point = True
                
            # 限制置信度在0-1之间
            confidence = min(confidence, 1.0)
            
            # 构建结果
            result = {
                'date': latest_date,
                'price': latest_price,
                'is_low_point': is_low_point,
                'confidence': confidence,
                'reasons': reasons,
                'technical_indicators': {
                    'ma5': ma5,
                    'ma10': ma10,
                    'ma20': ma20,
                    'rsi': rsi,
                    'macd': macd,
                    'bb_lower': bb_lower
                }
            }
            
            self.logger.info("相对低点识别结果: %s, 置信度: %.2f", 
                           "是" if is_low_point else "否", confidence)
            return result
            
        except Exception as e:
            self.logger.error("识别相对低点失败: %s", str(e))
            return {
                'date': None,
                'price': None,
                'is_low_point': False,
                'confidence': 0.0,
                'error': str(e)
            }
            
    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        回测策略
        
        参数:
        data: 历史数据
        
        返回:
        pandas.DataFrame: 回测结果
        """
        self.logger.info("开始回测，数据长度: %d", len(data))
        
        try:
            # 复制数据避免修改原数据
            backtest_data = data.copy()
            
            # 添加回测结果列
            backtest_data['is_low_point'] = False
            backtest_data['future_max_rise'] = 0.0
            backtest_data['days_to_rise'] = 0
            backtest_data['max_rise_date'] = None
            
            # 遍历每个交易日（除了最后max_days天）
            for i in range(len(backtest_data) - self.max_days):
                current_price = backtest_data.iloc[i]['close']
                current_date = backtest_data.iloc[i]['date']
                
                # 计算未来max_days天内的最大涨幅和天数
                max_rise = 0.0
                days_to_rise = 0
                max_rise_date = None
                
                for j in range(1, min(self.max_days + 1, len(backtest_data) - i)):
                    future_price = backtest_data.iloc[i + j]['close']
                    future_date = backtest_data.iloc[i + j]['date']
                    rise_rate = (future_price - current_price) / current_price
                    
                    if rise_rate > max_rise:
                        max_rise = rise_rate
                        max_rise_date = future_date
                        
                    if rise_rate >= self.rise_threshold and days_to_rise == 0:
                        days_to_rise = j
                        
                # 更新数据
                backtest_data.loc[i, 'future_max_rise'] = max_rise
                backtest_data.loc[i, 'days_to_rise'] = days_to_rise
                backtest_data.loc[i, 'max_rise_date'] = max_rise_date
                
                # 判断是否为相对低点
                if days_to_rise > 0:
                    backtest_data.loc[i, 'is_low_point'] = True
                    
            self.logger.info("回测完成")
            return backtest_data
            
        except Exception as e:
            self.logger.error("回测失败: %s", str(e))
            raise
            
    def evaluate_strategy(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        评估策略
        
        参数:
        backtest_results: 回测结果
        
        返回:
        dict: 评估结果
        """
        self.logger.info("评估策略")
        
        try:
            # 获取相对低点
            low_points = backtest_results[backtest_results['is_low_point']]
            total_points = len(low_points)
            
            if total_points == 0:
                return {
                    'total_points': 0,
                    'success_rate': 0.0,
                    'avg_rise': 0.0,
                    'avg_days': 0.0,
                    'max_rise': 0.0,
                    'min_rise': 0.0,
                    'score': 0.0
                }
            
            # 计算统计数据
            avg_rise = low_points['future_max_rise'].mean()
            avg_days = low_points['days_to_rise'].mean()
            max_rise = low_points['future_max_rise'].max()
            min_rise = low_points['future_max_rise'].min()
            
            # 计算成功率（未来实际涨幅超过阈值的比例）
            successful_points = low_points[low_points['future_max_rise'] >= self.rise_threshold]
            success_rate = len(successful_points) / total_points
            
            # 计算综合得分
            score = self._calculate_score(success_rate, avg_rise, avg_days)
            
            # 构建评估结果
            evaluation = {
                'total_points': total_points,
                'success_rate': success_rate,
                'avg_rise': avg_rise,
                'avg_days': avg_days,
                'max_rise': max_rise,
                'min_rise': min_rise,
                'score': score,
                'rise_threshold': self.rise_threshold,
                'max_days': self.max_days
            }
            
            self.logger.info("策略评估完成: 识别点数=%d, 成功率=%.2f%%, 平均涨幅=%.2f%%, 得分=%.4f", 
                           total_points, success_rate * 100, avg_rise * 100, score)
            
            return evaluation
            
        except Exception as e:
            self.logger.error("评估策略失败: %s", str(e))
            raise
            
    def _calculate_score(self, success_rate: float, avg_rise: float, avg_days: float) -> float:
        """
        计算策略得分
        
        参数:
        success_rate: 成功率
        avg_rise: 平均涨幅
        avg_days: 平均天数
        
        返回:
        float: 策略得分
        """
        # 成功率权重：50%
        success_score = success_rate * 0.5
        
        # 平均涨幅权重：30%（相对于10%的基准）
        rise_score = min(avg_rise / 0.1, 1.0) * 0.3
        
        # 平均天数权重：20%（天数越少越好，以10天为基准）
        if avg_days > 0:
            days_score = min(10.0 / avg_days, 1.0) * 0.2
        else:
            days_score = 0.0
            
        total_score = success_score + rise_score + days_score
        return total_score
        
    def visualize_backtest(self, backtest_results: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        可视化回测结果
        
        参数:
        backtest_results: 回测结果
        save_path: 保存路径，如果为None则自动生成
        
        返回:
        str: 图表文件路径
        """
        self.logger.info("可视化回测结果")
        
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('回测结果分析', fontsize=16, fontweight='bold')
            
            # 1. 价格曲线和相对低点
            ax1 = axes[0, 0]
            ax1.plot(backtest_results['date'], backtest_results['close'], 
                    label='收盘价', linewidth=1, alpha=0.8)
            
            # 标记相对低点
            low_points = backtest_results[backtest_results['is_low_point']]
            if len(low_points) > 0:
                ax1.scatter(low_points['date'], low_points['close'], 
                          color='red', marker='^', s=50, label='相对低点', zorder=5)
            
            ax1.set_title('价格走势与相对低点')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 涨幅分布
            ax2 = axes[0, 1]
            if len(low_points) > 0:
                rises = low_points['future_max_rise'] * 100
                ax2.hist(rises, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(x=self.rise_threshold * 100, color='red', linestyle='--', 
                          label=f'目标涨幅: {self.rise_threshold:.1%}')
                ax2.set_title('相对低点后的最大涨幅分布')
                ax2.set_xlabel('涨幅 (%)')
                ax2.set_ylabel('频次')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '无相对低点数据', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('相对低点后的最大涨幅分布')
            
            # 3. 达到目标涨幅的天数分布
            ax3 = axes[1, 0]
            if len(low_points) > 0:
                successful_points = low_points[low_points['days_to_rise'] > 0]
                if len(successful_points) > 0:
                    days = successful_points['days_to_rise']
                    ax3.hist(days, bins=range(1, self.max_days + 2), alpha=0.7, 
                           color='lightgreen', edgecolor='black')
                    ax3.set_title('达到目标涨幅所需天数分布')
                    ax3.set_xlabel('天数')
                    ax3.set_ylabel('频次')
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, '无成功案例', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=14)
                    ax3.set_title('达到目标涨幅所需天数分布')
            else:
                ax3.text(0.5, 0.5, '无相对低点数据', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('达到目标涨幅所需天数分布')
            
            # 4. 策略评估指标
            ax4 = axes[1, 1]
            evaluation = self.evaluate_strategy(backtest_results)
            
            metrics = ['成功率', '平均涨幅', '平均天数', '综合得分']
            values = [
                evaluation['success_rate'],
                evaluation['avg_rise'],
                evaluation['avg_days'] / self.max_days,  # 标准化
                evaluation['score']
            ]
            
            bars = ax4.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            ax4.set_title('策略评估指标')
            ax4.set_ylabel('数值')
            ax4.set_ylim(0, 1)
            
            # 在柱状图上添加数值标签
            for bar, value, metric in zip(bars, values, metrics):
                height = bar.get_height()
                if metric == '平均天数':
                    label = f'{evaluation["avg_days"]:.1f}天'
                elif metric == '平均涨幅':
                    label = f'{value:.1%}'
                elif metric == '成功率':
                    label = f'{value:.1%}'
                else:
                    label = f'{value:.3f}'
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        label, ha='center', va='bottom')
            
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.results_dir, f'backtest_analysis_{timestamp}.png')
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("回测结果可视化完成，保存到: %s", save_path)
            return save_path
            
        except Exception as e:
            self.logger.error("可视化回测结果失败: %s", str(e))
            raise
            
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        更新策略参数
        
        参数:
        params: 新参数
        """
        self.logger.info("更新策略参数: %s", params)
        
        if 'rise_threshold' in params:
            self.rise_threshold = params['rise_threshold']
            
        if 'max_days' in params:
            self.max_days = params['max_days']
            
        self.logger.info("策略参数已更新: rise_threshold=%.4f, max_days=%d", 
                        self.rise_threshold, self.max_days)
                        
    def get_params(self) -> Dict[str, Any]:
        """
        获取当前策略参数
        
        返回:
        dict: 当前参数
        """
        return {
            'rise_threshold': self.rise_threshold,
            'max_days': self.max_days
        }

