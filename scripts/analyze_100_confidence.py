#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析高置信度信号的历史表现
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
    data_module = DataModule(config)
    ai_optimizer = AIOptimizerImproved(config)
    
    if not ai_optimizer._load_model():
        logger.error("❌ 未找到已训练模型，请先运行训练！")
        return

    # 2. 获取数据 (近2年)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730) # 2年
    
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
    # 注意：我们不能使用未来数据。对于每一天 T，只能使用 T 及之前的数据进行预测。
    # predict_low_point 接受的是一个 DataFrame，它通常取最后一行作为 T。
    
    # 这里的循环可能会比较慢，因为每天都要切片。
    # 优化：不需每次切片所有历史，只需保留足够计算特征的窗口即可。
    # 但 predict_low_point 内部可能依赖 rolling 计算，如果传入太短可能不准。
    # 不过 full_data 已经是预处理过的（特征已计算好），所以可以直接取单行？
    # AIOptimizerImproved.predict_low_point 内部逻辑：
    # latest_features = features[-1:].reshape(1, -1)
    # 它会重新提取特征。如果传入的是单行，特征提取可能会失败（因为缺历史）。
    # 但是，我们已经对 full_data 做了 preprocess_data，特征列（如 rsi, ma5）已经存在了。
    # 检查 predict_low_point 源码：它会调用 self.feature_engineer.create_features(data)。
    # 如果 data 已经有特征列，create_features 会怎么处理？
    # 通常它会重新计算。如果只传一行，rolling 计算会变成 NaN。
    
    # 所以必须传入历史窗口。
    
    window_size = 60 # 足够计算 MA60 等指标
    
    for i in range(start_idx, end_idx - 10): # 留10天看结果
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
            
            # 降低阈值，分析新模型的高置信度区域
            if confidence >= 0.50: 
                # 记录信号
                # T+1 开盘买入
                entry_price = full_data.loc[i+1, 'open']
                # T+10 收盘卖出 (或者持有10天后的收盘价)
                exit_idx = i + 1 + 10
                if exit_idx < len(full_data):
                    exit_price = full_data.loc[exit_idx, 'close']
                    ret = (exit_price - entry_price) / entry_price
                    
                    signals.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'confidence': confidence,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': ret
                    })
                    print(f"✅ 发现信号: {current_date.strftime('%Y-%m-%d')} | 置信度: {confidence:.2f} | 收益: {ret:+.2%}")
                    
        except Exception as e:
            # logger.error(f"Error on {current_date}: {e}")
            pass

    # 4. 统计结果
    print("\n" + "="*40)
    print("📊 高置信度信号历史表现 (Confidence >= 0.50)")
    print("="*40)
    
    if not signals:
        print("未发现任何置信度 >= 0.50 的信号。")
        return

    df = pd.DataFrame(signals)
    df = df.sort_values('confidence', ascending=False) # 按置信度排序
    
    win_rate = len(df[df['return'] > 0]) / len(df)
    avg_ret = df['return'].mean()
    max_ret = df['return'].max()
    min_ret = df['return'].min()
    
    print(f"信号总数: {len(df)}")
    print(f"胜率:     {win_rate:.2%}")
    print(f"平均收益: {avg_ret:+.2%} (10天持有)")
    print(f"最大收益: {max_ret:+.2%}")
    print(f"最大亏损: {min_ret:+.2%}")
    print("-" * 40)
    print("详细记录 (Top 20):")
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    run_analysis()
