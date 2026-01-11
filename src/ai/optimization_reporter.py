#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI优化报告生成器
每次优化完成后自动生成详细的报告文件
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

class OptimizationReporter:
    """优化报告生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "results" / "optimization_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, optimization_result: Dict[str, Any], 
                       model_info: Dict[str, Any] = None,
                       overfitting_detection: Dict[str, Any] = None) -> str:
        """
        生成优化报告
        
        参数:
        optimization_result: 优化结果
        model_info: 模型信息
        overfitting_detection: 过拟合检测结果
        
        返回:
        str: 报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"optimization_report_{timestamp}.md"
        report_path = self.reports_dir / report_filename
        
        # 生成报告内容
        report_content = self._generate_report_content(
            optimization_result, model_info, overfitting_detection, timestamp
        )
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 生成JSON格式的数据文件
        self._save_json_data(optimization_result, model_info, overfitting_detection, timestamp)
        
        # 生成图表
        self._generate_charts(optimization_result, timestamp)
        
        self.logger.info(f"✅ 优化报告已生成: {report_path}")
        return str(report_path)
    
    def _generate_report_content(self, optimization_result: Dict[str, Any], 
                               model_info: Dict[str, Any],
                               overfitting_detection: Dict[str, Any],
                               timestamp: str) -> str:
        """生成报告内容"""
        
        report = f"""# AI优化报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**报告编号**: {timestamp}

---

## 📋 执行摘要

### 优化状态
- **优化方法**: {optimization_result.get('method', 'N/A')}
- **执行状态**: {'✅ 成功' if optimization_result.get('success', False) else '❌ 失败'}
- **总耗时**: {optimization_result.get('total_time', 0):.2f} 秒
- **优化轮次**: {optimization_result.get('iterations', 'N/A')}

### 关键指标
- **最终得分**: {optimization_result.get('best_score', 0):.4f}
- **准确率**: {optimization_result.get('accuracy', 0) * 100:.2f}%
- **成功率**: {optimization_result.get('success_rate', 0) * 100:.2f}%
- **平均收益**: {optimization_result.get('avg_return', optimization_result.get('avg_rise', 0)) * 100:.2f}%
- **总利润**: {optimization_result.get('total_profit', 0):.4f}

---

## 🎯 优化结果详情

### 最优参数
"""
        
        # 添加最优参数
        best_params = optimization_result.get('best_params', {})
        if best_params:
            report += "```yaml\n"
            for key, value in best_params.items():
                if isinstance(value, float):
                    report += f"{key}: {value:.4f}\n"
                else:
                    report += f"{key}: {value}\n"
            report += "```\n\n"
        
        # 添加模型信息
        if model_info:
            report += f"""### 模型配置
- **模型类型**: {model_info.get('model_type', 'RandomForest')}
- **特征数量**: {model_info.get('feature_count', 'N/A')}
- **训练样本**: {model_info.get('train_samples', 'N/A')}
- **正样本比例**: {model_info.get('positive_ratio', 0) * 100:.2f}%

### 模型参数
- **决策树数量**: {model_info.get('n_estimators', 'N/A')}
- **最大深度**: {model_info.get('max_depth', 'N/A')}
- **最小分割样本**: {model_info.get('min_samples_split', 'N/A')}
- **最小叶子样本**: {model_info.get('min_samples_leaf', 'N/A')}

"""
        
        # 添加过拟合检测结果
        if overfitting_detection:
            report += f"""---

## 🔍 过拟合检测

### 检测结果
- **过拟合状态**: {'🚨 检测到过拟合' if overfitting_detection.get('overfitting_detected', False) else '✅ 未检测到过拟合'}

### 关键指标
"""
            metrics = overfitting_detection.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, float):
                    report += f"- **{key}**: {value:.4f}\n"
                else:
                    report += f"- **{key}**: {value}\n"
            
            # 添加警告和建议
            warnings = overfitting_detection.get('warnings', [])
            if warnings:
                report += "\n### ⚠️ 警告\n"
                for warning in warnings:
                    report += f"- {warning}\n"
            
            recommendations = overfitting_detection.get('recommendations', [])
            if recommendations:
                report += "\n### 💡 建议\n"
                for rec in recommendations:
                    report += f"- {rec}\n"
        
        # 添加配置信息
        report += f"""

---

## ⚙️ 配置信息

### 数据分割
- **训练集比例**: {self.config.get('ai', {}).get('validation', {}).get('train_ratio', 0.6) * 100:.0f}%
- **验证集比例**: {self.config.get('ai', {}).get('validation', {}).get('validation_ratio', 0.25) * 100:.0f}%
- **测试集比例**: {self.config.get('ai', {}).get('validation', {}).get('test_ratio', 0.15) * 100:.0f}%

### 早停配置
- **耐心值**: {self.config.get('ai', {}).get('early_stopping', {}).get('patience', 20)}
- **最小改善**: {self.config.get('ai', {}).get('early_stopping', {}).get('min_delta', 0.005)}

### 策略参数
- **涨幅阈值**: {self.config.get('strategy', {}).get('rise_threshold', 0.04) * 100:.1f}%
- **最大天数**: {self.config.get('strategy_params', {}).get('max_days', 20)}

---

## 📊 性能分析

### 训练效率
"""
        
        # 添加训练时间分析
        if 'training_time_breakdown' in optimization_result:
            breakdown = optimization_result['training_time_breakdown']
            total_time = sum(breakdown.values())
            for phase, time_spent in breakdown.items():
                percentage = (time_spent / total_time) * 100 if total_time > 0 else 0
                report += f"- **{phase}**: {time_spent:.2f}s ({percentage:.1f}%)\n"
        
        # 添加历史对比
        report += """

### 历史对比
*注: 与之前的优化结果对比，需要积累更多历史数据*

---

## 📝 建议下一步

"""
        
        # 根据结果生成建议
        if optimization_result.get('success', False):
            if overfitting_detection and overfitting_detection.get('overfitting_detected', False):
                report += """
1. **降低模型复杂度**: 减少决策树数量或最大深度
2. **增加正则化**: 提高min_samples_split和min_samples_leaf
3. **数据增强**: 收集更多训练数据
4. **特征选择**: 移除冗余或噪声特征
"""
            else:
                report += """
1. **运行滚动回测**: `python run.py r 2025-06-27 2025-07-07`
2. **单日预测测试**: `python run.py p 2025-07-08`
3. **实盘验证**: 在真实环境中测试模型表现
4. **定期重训练**: 根据新数据定期更新模型
"""
        else:
            report += """
1. **检查数据质量**: 确认训练数据的完整性和准确性
2. **调整参数范围**: 扩大或缩小参数搜索空间
3. **增加训练数据**: 收集更多历史数据
4. **检查特征工程**: 验证特征提取的正确性
"""
        
        report += f"""

---

## 📁 文件信息

- **报告文件**: `{self.reports_dir.name}/optimization_report_{timestamp}.md`
- **数据文件**: `{self.reports_dir.name}/optimization_data_{timestamp}.json`
- **图表文件**: `{self.reports_dir.name}/charts/optimization_charts_{timestamp}.png`

---

*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def _save_json_data(self, optimization_result: Dict[str, Any], 
                       model_info: Dict[str, Any],
                       overfitting_detection: Dict[str, Any],
                       timestamp: str):
        """保存JSON格式的数据"""
        data = {
            'timestamp': timestamp,
            'generated_at': datetime.now().isoformat(),
            'optimization_result': optimization_result,
            'model_info': model_info,
            'overfitting_detection': overfitting_detection,
            'config_snapshot': {
                'data_split': self.config.get('ai', {}).get('validation', {}),
                'early_stopping': self.config.get('ai', {}).get('early_stopping', {}),
                'strategy': self.config.get('strategy', {})
            }
        }
        
        json_path = self.reports_dir / f"optimization_data_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_charts(self, optimization_result: Dict[str, Any], timestamp: str):
        """生成图表"""
        try:
            charts_dir = self.reports_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 图表1: 优化历史（如果有的话）
            if 'optimization_history' in optimization_result:
                history = optimization_result['optimization_history']
                iterations = list(range(len(history)))
                scores = [h.get('score', 0) for h in history]
                
                ax1.plot(iterations, scores, 'b-', linewidth=2)
                ax1.set_title('优化历史', fontsize=14, fontweight='bold')
                ax1.set_xlabel('迭代次数')
                ax1.set_ylabel('得分')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, '暂无优化历史数据', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('优化历史', fontsize=14, fontweight='bold')
            
            # 图表2: 关键指标
            metrics = ['准确率', '成功率', '平均收益', '最终得分']
            values = [
                optimization_result.get('accuracy', 0) * 100,
                optimization_result.get('success_rate', 0) * 100,
                optimization_result.get('avg_return', optimization_result.get('avg_rise', 0)) * 100,
                optimization_result.get('best_score', 0) * 100
            ]
            
            bars = ax2.bar(metrics, values, color=['#2E8B57', '#4682B4', '#FF6347', '#FFD700'])
            ax2.set_title('关键指标', fontsize=14, fontweight='bold')
            ax2.set_ylabel('百分比 (%)')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # 图表3: 时间分解（如果有的话）
            if 'training_time_breakdown' in optimization_result:
                breakdown = optimization_result['training_time_breakdown']
                labels = list(breakdown.keys())
                sizes = list(breakdown.values())
                
                ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax3.set_title('训练时间分解', fontsize=14, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '暂无训练时间数据', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('训练时间分解', fontsize=14, fontweight='bold')
            
            # 图表4: 过拟合检测指标
            if 'overfitting_detection' in optimization_result:
                detection = optimization_result['overfitting_detection']
                metrics = detection.get('metrics', {})
                
                metric_names = []
                metric_values = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_names.append(key.replace('_', '\n'))
                        metric_values.append(value)
                
                if metric_names:
                    ax4.bar(metric_names, metric_values, color='#FF4500' if detection.get('overfitting_detected', False) else '#32CD32')
                    ax4.set_title('过拟合检测指标', fontsize=14, fontweight='bold')
                    ax4.set_ylabel('数值')
                    
                    # 旋转x轴标签以避免重叠
                    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax4.text(0.5, 0.5, '暂无过拟合检测数据', ha='center', va='center', 
                            transform=ax4.transAxes, fontsize=12)
            else:
                ax4.text(0.5, 0.5, '暂无过拟合检测数据', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('过拟合检测指标', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = charts_dir / f"optimization_charts_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 优化图表已生成: {chart_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 图表生成失败: {e}")
    
    def create_summary_report(self, days_back: int = 7) -> str:
        """
        创建最近几天的优化汇总报告
        
        参数:
        days_back: 回溯天数
        
        返回:
        str: 汇总报告路径
        """
        # 查找最近的报告文件
        cutoff_date = datetime.now().timestamp() - (days_back * 24 * 3600)
        recent_reports = []
        
        for json_file in self.reports_dir.glob("optimization_data_*.json"):
            try:
                # 从文件名提取时间戳
                timestamp_str = json_file.stem.replace('optimization_data_', '')
                file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').timestamp()
                
                if file_time >= cutoff_date:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        recent_reports.append(data)
            except Exception as e:
                self.logger.warning(f"无法解析报告文件 {json_file}: {e}")
        
        # 生成汇总报告
        if recent_reports:
            summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.reports_dir / f"summary_report_{summary_timestamp}.md"
            
            summary_content = self._generate_summary_content(recent_reports, days_back)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            self.logger.info(f"✅ 汇总报告已生成: {summary_path}")
            return str(summary_path)
        else:
            self.logger.warning(f"最近{days_back}天内没有找到优化报告")
            return ""
    
    def _generate_summary_content(self, reports: List[Dict[str, Any]], days_back: int) -> str:
        """生成汇总报告内容"""
        
        summary = f"""# AI优化汇总报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**统计周期**: 最近 {days_back} 天  
**报告数量**: {len(reports)}

---

## 📊 整体趋势

"""
        
        # 计算趋势统计
        scores = []
        accuracies = []
        overfitting_count = 0
        
        for report in reports:
            opt_result = report.get('optimization_result', {})
            scores.append(opt_result.get('best_score', 0))
            accuracies.append(opt_result.get('accuracy', 0))
            
            overfitting = report.get('overfitting_detection', {})
            if overfitting.get('overfitting_detected', False):
                overfitting_count += 1
        
        if scores:
            summary += f"""
### 得分趋势
- **最高得分**: {max(scores):.4f}
- **最低得分**: {min(scores):.4f}
- **平均得分**: {np.mean(scores):.4f}
- **得分标准差**: {np.std(scores):.4f}

### 准确率趋势
- **最高准确率**: {max(accuracies) * 100:.2f}%
- **最低准确率**: {min(accuracies) * 100:.2f}%
- **平均准确率**: {np.mean(accuracies) * 100:.2f}%

### 过拟合情况
- **过拟合次数**: {overfitting_count} / {len(reports)}
- **过拟合比例**: {overfitting_count / len(reports) * 100:.1f}%

"""
        
        # 添加详细记录
        summary += "## 📋 详细记录\n\n"
        summary += "| 时间 | 得分 | 准确率 | 过拟合 | 耗时 |\n"
        summary += "|------|------|--------|--------|------|\n"
        
        for report in sorted(reports, key=lambda x: x.get('timestamp', ''), reverse=True):
            timestamp = report.get('timestamp', 'N/A')
            opt_result = report.get('optimization_result', {})
            overfitting = report.get('overfitting_detection', {})
            
            time_str = timestamp[:8] + ' ' + timestamp[9:].replace('_', ':') if len(timestamp) >= 15 else timestamp
            score = opt_result.get('best_score', 0)
            accuracy = opt_result.get('accuracy', 0) * 100
            overfitting_status = '🚨' if overfitting.get('overfitting_detected', False) else '✅'
            duration = opt_result.get('total_time', 0)
            
            summary += f"| {time_str} | {score:.4f} | {accuracy:.1f}% | {overfitting_status} | {duration:.1f}s |\n"
        
        summary += f"""

---

## 💡 总结与建议

"""
        
        # 根据统计结果生成建议
        if overfitting_count > len(reports) * 0.5:
            summary += """
### ⚠️ 过拟合问题严重
- 过拟合发生频率过高
- 建议进一步降低模型复杂度
- 考虑增加更多训练数据
- 加强正则化措施
"""
        elif overfitting_count > 0:
            summary += """
### ⚠️ 偶有过拟合
- 偶尔出现过拟合现象
- 当前配置基本合理
- 建议持续监控
- 可考虑微调参数
"""
        else:
            summary += """
### ✅ 过拟合控制良好
- 未检测到明显过拟合
- 当前配置效果良好
- 可继续使用当前设置
- 建议定期评估
"""
        
        if scores and np.std(scores) > 0.1:
            summary += """
### 📈 得分波动较大
- 模型稳定性需要改善
- 考虑增加训练数据
- 检查特征工程一致性
- 评估参数搜索范围
"""
        
        summary += f"""

---

*汇总报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary


def create_optimization_report(optimization_result: Dict[str, Any], 
                             config: Dict[str, Any],
                             model_info: Dict[str, Any] = None,
                             overfitting_detection: Dict[str, Any] = None) -> str:
    """
    创建优化报告的便捷函数
    
    参数:
    optimization_result: 优化结果
    config: 配置信息
    model_info: 模型信息
    overfitting_detection: 过拟合检测结果
    
    返回:
    str: 报告文件路径
    """
    reporter = OptimizationReporter(config)
    return reporter.generate_report(optimization_result, model_info, overfitting_detection)