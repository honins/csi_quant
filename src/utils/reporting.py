# -*- coding: utf-8 -*-
"""
统一的回测结果输出格式化工具。
提供在 CLI 和脚本中复用的消息格式，避免重复拼接字符串。
"""
import os
from typing import Dict, Optional


def format_backtest_summary(result: Dict, project_root: Optional[str] = None) -> str:
    """
    将回测结果格式化为统一的摘要字符串。

    参数:
        result: run_rolling_backtest 返回的结果字典，包含 metrics 和 report_path。
        project_root: 项目根路径，用于将报告路径转换为相对路径，便于输出。

    返回:
        统一格式的摘要字符串（含关键指标与报告路径）。
    """
    metrics = result.get('metrics', {}) or {}
    success_rate = float(metrics.get('success_rate', 0.0) or 0.0)
    total_predictions = int(metrics.get('total_predictions', 0) or 0)
    f1_score = float(metrics.get('f1', 0.0) or 0.0)
    recall = float(metrics.get('recall', 0.0) or 0.0)
    precision = float(metrics.get('precision', 0.0) or 0.0)

    report_path = result.get('report_path')
    msg = (
        f"✅ 滚动回测完成: 成功率 {success_rate:.1%}, 预测数 {total_predictions}, "
        f"F1 {f1_score:.3f}, Recall {recall:.3f}, Precision {precision:.3f}"
    )

    if report_path:
        rel = (
            os.path.relpath(report_path, project_root)
            if project_root else report_path
        )
        msg += f"\n📄 报告: {rel}"

    return msg