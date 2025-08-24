#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
final_threshold 网格测试脚本

这是一个无侵入的网格测试脚本，使用临时配置文件的方式来测试不同的 final_threshold 值，
而不修改项目的原有配置文件。
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
import yaml
import logging

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from examples.run_rolling_backtest import run_rolling_backtest


def create_temp_config(base_config_dir: str, final_threshold: float) -> str:
    """
    创建包含指定 final_threshold 的临时配置文件
    
    Args:
        base_config_dir: 基础配置目录
        final_threshold: 要测试的 final_threshold 值
    
    Returns:
        str: 临时配置文件路径
    """
    # 创建临时文件
    temp_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml', prefix='grid_test_')
    
    try:
        # 只覆盖 final_threshold，其他配置保持不变
        temp_config = {
            'confidence_weights': {
                'final_threshold': final_threshold
            }
        }
        
        # 写入临时配置文件
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return temp_config_path
        
    finally:
        os.close(temp_fd)


def run_grid_test(start_date: str, end_date: str, threshold_values: list):
    """
    运行 final_threshold 网格测试
    
    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期  
        threshold_values: 要测试的 final_threshold 值列表
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("GridTest")
    
    project_root = Path(__file__).parent.parent
    base_config_dir = project_root / 'config'
    
    results = []
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 final_threshold 网格测试")
    logger.info(f"{'='*80}")
    logger.info(f"📅 测试期间: {start_date} 至 {end_date}")
    logger.info(f"🎯 测试阈值: {threshold_values}")
    logger.info(f"{'='*80}")
    
    for threshold in threshold_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 测试 final_threshold = {threshold:.3f}")
        logger.info(f"{'='*60}")
        
        # 创建临时配置文件
        temp_config_path = create_temp_config(str(base_config_dir), threshold)
        
        try:
            # 设置环境变量，让配置加载器读取临时配置
            original_env = os.environ.get('CSI_CONFIG_PATH')
            os.environ['CSI_CONFIG_PATH'] = temp_config_path
            
            # 运行回测并直接获取结果
            from examples.run_rolling_backtest import run_rolling_backtest_with_return
            
            # 运行回测并直接获取结果（禁用报告生成，避免大量文件）
            backtest_result = run_rolling_backtest_with_return(start_date, end_date, generate_report=False)
            
            if backtest_result and 'metrics' in backtest_result:
                metrics = backtest_result['metrics']
                total_predictions = metrics.get('total_predictions', 0)
                correct_predictions = metrics.get('correct_predictions', 0)
                pred_pos = metrics.get('pred_positive', 0)
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1 = metrics.get('f1', 0.0)
                success_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
            else:
                # 后备方案：如果函数没返回结果，设为0
                total_predictions = 0
                correct_predictions = 0
                pred_pos = 0
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                success_rate = 0.0
            
            results.append({
                'threshold': threshold,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'success_rate': success_rate,
                'pred_positive': pred_pos,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            logger.info(f"✅ 测试完成: 成功率={success_rate:.2%}, 预测正类={pred_pos}, P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
            
        except Exception as e:
            logger.error(f"❌ 测试失败 (threshold={threshold}): {e}")
            results.append({
                'threshold': threshold,
                'total_predictions': 0,
                'correct_predictions': 0,
                'success_rate': 0.0,
                'pred_positive': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': str(e)
            })
            
        finally:
            # 恢复环境变量
            if original_env is not None:
                os.environ['CSI_CONFIG_PATH'] = original_env
            elif 'CSI_CONFIG_PATH' in os.environ:
                del os.environ['CSI_CONFIG_PATH']
            
            # 删除临时配置文件
            try:
                os.unlink(temp_config_path)
            except:
                pass
    
    # 输出汇总结果
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 网格测试汇总结果")
    logger.info(f"{'='*80}")
    logger.info(f"{'Threshold':<10} {'Success%':<8} {'Pred+':<6} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    logger.info(f"{'-'*60}")
    
    best_f1 = 0
    best_threshold = None
    
    for result in results:
        if 'error' not in result:
            logger.info(f"{result['threshold']:<10.3f} {result['success_rate']:<8.2%} {result['pred_positive']:<6} {result['precision']:<10.3f} {result['recall']:<8.3f} {result['f1']:<8.3f}")
            
            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_threshold = result['threshold']
        else:
            logger.info(f"{result['threshold']:<10.3f} {'ERROR':<8} {'N/A':<6} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
    
    if best_threshold is not None:
        logger.info(f"\n🏆 最佳阈值: {best_threshold:.3f} (F1={best_f1:.3f})")
        logger.info(f"💡 建议将 config/optimized_params.yaml 中的 final_threshold 设置为 {best_threshold}")
    else:
        logger.info(f"\n❌ 未能找到有效的最佳阈值")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python grid_test_final_threshold.py <start_date> <end_date> [threshold1,threshold2,...]")
        print("示例: python grid_test_final_threshold.py 2023-01-01 2023-03-31 0.42,0.45,0.48,0.50,0.55")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    # 默认测试阈值
    if len(sys.argv) >= 4:
        threshold_values = [float(x.strip()) for x in sys.argv[3].split(',')]
    else:
        threshold_values = [0.42, 0.45, 0.48, 0.50, 0.55]
    
    run_grid_test(start_date, end_date, threshold_values)