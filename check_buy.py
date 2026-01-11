#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简易查买点工具 (生产级版本)
用法: python check_buy.py [YYYY-MM-DD]
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. 环境与路径设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from src.utils.config_loader import load_config
    from src.utils.common import LoggerManager
    from src.data.data_module import DataModule
    from src.strategy.strategy_module import StrategyModule
    from src.ai.ai_optimizer_improved import AIOptimizerImproved
    from src.prediction.prediction_utils import predict_and_validate
    from src.utils.trade_date import is_trading_day
except ImportError as e:
    print(f"❌ 环境错误: 无法导入核心模块。\n详细错误: {e}")
    sys.exit(1)

def get_latest_trading_date(target_date: datetime) -> datetime:
    """寻找小于等于 target_date 的最近交易日"""
    check_date = target_date
    for _ in range(20):
        if is_trading_day(check_date.date()):
            return check_date
        check_date -= timedelta(days=1)
    return target_date

def check_buy(date_str=None):
    # -------------------------------------------------------------------------
    # 2. 初始化
    # -------------------------------------------------------------------------
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    logger = logging.getLogger("CheckBuy")
    
    try:
        config = load_config()
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        if not ai_optimizer._load_model():
            print("❌ 未找到已训练模型！请先运行训练。")
            return

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # -------------------------------------------------------------------------
    # 3. 日期处理
    # -------------------------------------------------------------------------
    if not date_str:
        target_date = datetime.now()
        user_specified = False
    else:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            user_specified = True
        except ValueError:
            print(f"❌ 日期格式错误: {date_str} (应为 YYYY-MM-DD)")
            return

    final_date = get_latest_trading_date(target_date)
    
    if user_specified and final_date.date() != target_date.date():
        print(f"⚠️  {target_date.strftime('%Y-%m-%d')} 不是交易日，已自动调整为: {final_date.strftime('%Y-%m-%d')}")

    print(f"🔍 正在分析 {final_date.strftime('%Y-%m-%d')} 的市场信号...")

    # -------------------------------------------------------------------------
    # 4. 执行预测
    # -------------------------------------------------------------------------
    try:
        result = predict_and_validate(
            predict_date=final_date,
            data_module=data_module,
            strategy_module=strategy_module,
            ai_optimizer=ai_optimizer,
            config=config,
            logger=logger,
            force_retrain=False,
            only_use_trained_model=True
        )

        if not result or result.predicted_low_point is None:
            print("⚠️  无法获取数据或预测失败")
            return

        # ---------------------------------------------------------------------
        # 5. 格式化输出 (增强版)
        # ---------------------------------------------------------------------
        print("-" * 45)
        print(f"📅 信号日期: {result.date.strftime('%Y-%m-%d')}")
        
        close_price = result.predict_price
        if close_price:
            print(f"💰 收盘价格: {close_price:.2f}")

        # 获取技术指标
        indicators = result.strategy_indicators or {}
        rsi = indicators.get('rsi')
        
        # 打印技术面快照
        tech_status = []
        if rsi:
            rsi_desc = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            tech_status.append(f"RSI={rsi:.1f}({rsi_desc})")
        
        # 简单的均线判断 (如果有MA数据)
        # 这里假设 indicators 里可能有 ma5, ma20 等，如果没有就不显示
        
        if tech_status:
             print(f"📊 技术状态: {', '.join(tech_status)}")

        print("-" * 45)

        # 核心建议
        conf = result.confidence if result.confidence is not None else 0.0
        
        if result.predicted_low_point:
            print(f"🚀 【买入建议】: 强烈推荐 (BUY)")
            print(f"🔥 AI置信度:  {conf:.2%}")
            
            if result.strategy_reasons:
                print(f"📝 策略依据:  {'; '.join(result.strategy_reasons)}")
            
            print(f"💡 操作执行:  次日开盘买入")
            
            # 计算止盈止损
            sl_pct = config.get('strategy', {}).get('backtest', {}).get('stop_loss_pct', 0.04)
            tp_pct = config.get('strategy', {}).get('backtest', {}).get('take_profit_pct', 0.06)
            
            if close_price:
                stop_loss = close_price * (1 - sl_pct)
                take_profit = close_price * (1 + tp_pct)
                print(f"🛑 建议止损:  {stop_loss:.2f} (-{sl_pct:.1%})")
                print(f"🎯 建议止盈:  {take_profit:.2f} (+{tp_pct:.1%})")
                
        else:
            print(f"✋ 【买入建议】: 观望 (WAIT)")
            print(f"❄️ AI置信度:  {conf:.2%}")
            
            if result.strategy_reasons:
                 # 即使不买，也看看策略说了啥（通常是负面理由）
                 print(f"📝 市场状态:  {'; '.join(result.strategy_reasons)}")
            
            if conf > 0.3:
                print(f"   (注: 置信度未达到买入阈值)")

        print("-" * 45)

    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    check_buy(target)
