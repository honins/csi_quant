#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析最近3个月高置信度(>0.70)信号的历史表现
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_module import DataModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.utils.config_loader import load_config
from src.utils.common import LoggerManager

def run_analysis():
    # 1. 初始化
    LoggerManager.setup_logging(level=logging.INFO)
    logger = logging.getLogger("Analysis")
    
    config = load_config()
    
    # 获取阈值 (优先从配置读取，默认0.25)
    threshold = config.get('confidence_weights', {}).get('final_threshold', 0.25)
    logger.info(f"使用置信度阈值: {threshold}")
    
    data_module = DataModule(config)
    ai_optimizer = AIOptimizerImproved(config)
    
    if not ai_optimizer._load_model():
        logger.error("❌ 未找到已训练模型，请先运行训练！")
        return

    # 2. 获取数据 (近3个月)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90) # 3个月
    
    logger.info(f"正在加载数据: {start_date.date()} 至 {end_date.date()}")
    
    # 为了保证特征计算（如MA60），需要额外获取一些前置数据
    fetch_start = start_date - timedelta(days=100)
    
    full_data = data_module.get_history_data(
        fetch_start.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    if full_data.empty:
        logger.error("数据为空")
        return
        
    full_data = data_module.preprocess_data(full_data)
    
    # 过滤出分析区间的数据索引
    analysis_data = full_data[full_data['date'] >= start_date]
    if analysis_data.empty:
        logger.error("分析区间内无数据")
        return
        
    start_idx = analysis_data.index[0]
    end_idx = full_data.index[-1]
    
    logger.info(f"开始扫描 {len(analysis_data)} 个交易日...")
    
    signals = []
    
    # 3. 逐日扫描
    window_size = 60 # 足够计算 MA60 等指标
    
    # 扫描到最近一天
    for i in range(start_idx, end_idx + 1): # 包括最后一天
        current_date = full_data.loc[i, 'date']
        
        # 构造历史窗口
        window_start = max(0, i - window_size)
        history_slice = full_data.iloc[window_start : i+1].copy()
        
        # 预测
        # 禁用日志以防刷屏
        logging.getLogger("AIOptimizer").setLevel(logging.WARNING)
        
        try:
            res = ai_optimizer.predict_low_point(history_slice)
            confidence = res.get('confidence', 0.0)
            
            # 记录所有信号，最后再排序筛选
            # 记录信号
            
            # T+1 开盘买入
            if i + 1 < len(full_data):
                entry_price = full_data.loc[i+1, 'open']
                
                # 尝试计算 T+10 收益，如果不足10天，则计算到最新一天
                target_days = 10
                exit_idx = i + 1 + target_days
                
                if exit_idx >= len(full_data):
                    exit_idx = len(full_data) - 1 # 取最后一天
                
                days_held = exit_idx - (i + 1)
                
                if days_held > 0:
                    exit_price = full_data.loc[exit_idx, 'close']
                    # 转换成浮点数计算，防止pandas series计算问题
                    entry_val = float(entry_price)
                    exit_val = float(exit_price)
                    ret = (exit_val - entry_val) / entry_val
                    
                    signals.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'confidence': confidence,
                        'entry_price': entry_val,
                        'exit_price': exit_val,
                        'days_held': days_held,
                        'return_pct': ret
                    })
                else:
                    # T+1 就是最后一天，还没有收盘
                    pass
            else:
                # i 是最后一天，无法买入
                signals.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'confidence': confidence,
                    'entry_price': 0,
                    'exit_price': 0,
                    'days_held': 0,
                    'return_pct': 0
                })

        except Exception as e:
            # logger.error(f"Error on {current_date}: {e}")
            pass

    # 4. 统计结果
    print("\n" + "="*40)
    print(f"📊 最近3个月高置信度信号 (阈值 >= {threshold})")
    print("="*40)
    
    if not signals:
        print(f"未发现置信度 >= {threshold} 的信号。")
        return

    df = pd.DataFrame(signals)
    
    # 过滤阈值
    df = df[df['confidence'] >= threshold]
    
    if df.empty:
        print(f"未发现置信度 >= {threshold} 的信号。")
        return
        
    df = df.sort_values('date', ascending=False) # 按日期倒序
    
    # 格式化输出列
    output_df = df[['date', 'confidence', 'return_pct', 'days_held']].copy()
    output_df['confidence'] = output_df['confidence'].apply(lambda x: f"{x:.4f}")
    
    # 对于未持有的（待买入），显示 --
    def format_ret(row):
        if row['days_held'] == 0:
            return "待买入 (最新信号)"
        return f"{row['return_pct']:+.2%}"
        
    output_df['status'] = output_df.apply(format_ret, axis=1)
    
    print("-" * 40)
    print("详细列表 (按日期倒序):")
    print(output_df[['date', 'confidence', 'status', 'days_held']].to_string(index=False))

if __name__ == "__main__":
    run_analysis()
