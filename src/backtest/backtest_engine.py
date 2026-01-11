import pandas as pd
import numpy as np
import logging
import traceback

class BacktestEngine:
    """
    资金回测引擎 (Refactored)
    支持:
    1. 基于 ATR 的动态仓位管理 (Kelly / Volatility Scaling)
    2. 基于 ATR 的移动止损 (Chandelier Exit)
    3. 传统的固定比例止盈止损
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger('BacktestEngine')
        
        # 默认配置
        self.initial_capital = 100000.0
        self.commission_rate = 0.0003
        self.slippage = 0.001
        
        # 风控配置
        self.use_atr_risk = True        # 是否启用 ATR 风控
        self.risk_per_trade = 0.02      # 单笔交易风险 (占总资金的百分比)
        self.atr_multiplier = 2.0       # ATR 倍数 (用于止损距离)
        self.atr_period = 14            # ATR 周期
        
        # 传统配置 (Fallback)
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.10     # 提高止盈，让利润奔跑
        self.max_holding_days = 20      # 延长持仓，捕捉波段
        
        if config and 'strategy' in config and 'backtest' in config['strategy']:
            bt_conf = config['strategy']['backtest']
            self.initial_capital = bt_conf.get('initial_capital', self.initial_capital)
            self.commission_rate = bt_conf.get('commission_rate', self.commission_rate)
            self.slippage = bt_conf.get('slippage', self.slippage)
            
            # 风控参数覆盖
            self.use_atr_risk = bt_conf.get('use_atr_risk', True)
            self.risk_per_trade = bt_conf.get('risk_per_trade', 0.02)
            self.atr_multiplier = bt_conf.get('atr_multiplier', 2.0)
            
            self.stop_loss_pct = bt_conf.get('stop_loss_pct', self.stop_loss_pct)
            self.take_profit_pct = bt_conf.get('take_profit_pct', self.take_profit_pct)
            self.max_holding_days = bt_conf.get('max_holding_days', self.max_holding_days)
            
        self.reset()
        
    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.shares = 0
        self.equity_curve = []
        self.trades = []
        self.position = None
        
    def run(self, daily_data, signals):
        """运行回测"""
        self.reset()
        
        df = daily_data.copy()
        if 'date' not in df.columns:
            df = df.reset_index()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        # 确保有 ATR 数据
        if 'atr' not in df.columns and self.use_atr_risk:
            self._calculate_atr(df)
            
        # 准备信号
        if isinstance(signals, pd.Series):
            sig_df = signals.to_frame(name='signal')
        else:
            sig_df = signals.copy()
            if 'signal' not in sig_df.columns and len(sig_df.columns) > 0:
                sig_df.rename(columns={sig_df.columns[0]: 'signal'}, inplace=True)
                
        if 'date' not in sig_df.columns:
            sig_df = sig_df.reset_index()
            # 简单尝试重命名
            if 'index' in sig_df.columns:
                sig_df.rename(columns={'index': 'date'}, inplace=True)
                
        sig_df['date'] = pd.to_datetime(sig_df['date']).dt.normalize()
        sig_subset = sig_df[['date', 'signal']].rename(columns={'signal': 'trade_signal'})
        
        df = pd.merge(df, sig_subset, on='date', how='left')
        df['trade_signal'] = df['trade_signal'].fillna(False).astype(bool)
        
        # 逐日回测
        for idx, row in df.iterrows():
            current_date = row['date']
            # 使用开高低收
            ohlc = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'atr': row.get('atr', 0.0)
            }
            
            signal = row['trade_signal']
            
            # 1. 检查退出
            if self.position:
                self._check_exit(current_date, ohlc)
                
            # 2. 检查买入
            if not self.position and signal:
                self._execute_buy(current_date, ohlc)
                
            # 3. 记录资产
            current_equity = self.cash + (self.shares * ohlc['close'])
            self.equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'cash': self.cash,
                'shares': self.shares,
                'price': ohlc['close']
            })
            
        return pd.DataFrame(self.equity_curve)

    def _calculate_atr(self, df):
        """计算 ATR (如果数据中没有)"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean().fillna(0)

    def _execute_buy(self, date, ohlc):
        """执行买入 (支持动态仓位)"""
        price = ohlc['open'] # 假设开盘买入 (实际可能是收盘，这里保守一点用开盘或者收盘)
        # 为了更真实，如果信号是基于收盘价产生的，通常是次日开盘买入。
        # 这里假设 `run` 的逻辑是：当日信号，当日处理？
        # 原逻辑是：信号是 `predicted_low_point`，通常是基于 T 日收盘后预测 T+1。
        # 这里的 `row` 是 T 日的数据。如果是基于 T-1 的信号在 T 日买入，那么应该读取 `trade_signal`。
        # 假设 df['trade_signal'] 是 "今日是否应该买入"。
        
        buy_price = price * (1 + self.slippage)
        atr = ohlc['atr']
        
        # 动态仓位计算
        shares_to_buy = 0
        
        if self.use_atr_risk and atr > 0:
            # 风险资金 = 当前总资产 * 单笔风险比例 (例如 2%)
            current_equity = self.cash # 此时无持仓，权益=现金
            risk_money = current_equity * self.risk_per_trade
            
            # 止损距离 = ATR * 倍数 (例如 2.0)
            stop_distance = atr * self.atr_multiplier
            
            # 买入股数 = 风险资金 / 单股止损金额
            # 确保 stop_distance 不为 0
            if stop_distance < price * 0.005: # 最小止损 0.5%
                stop_distance = price * 0.005
                
            raw_shares = risk_money / stop_distance
            shares_to_buy = int(raw_shares / 100) * 100
            
            # 初始止损价
            initial_stop = buy_price - stop_distance
        else:
            # 全仓模式 (Aggressive Mode)
            # 计算最大可买股数 (考虑佣金预留)
            # 预留 1% 现金防止佣金计算误差
            available_cash = self.cash * 0.99 
            shares_to_buy = int(available_cash / buy_price / 100) * 100
            
            # 使用固定比例止损作为兜底
            initial_stop = buy_price * (1 - self.stop_loss_pct)
            
        # 检查资金是否足够 (Double Check)
        cost = shares_to_buy * buy_price * (1 + self.commission_rate)
        if cost > self.cash:
            shares_to_buy = int(self.cash / (buy_price * (1 + self.commission_rate)) / 100) * 100
            
        if shares_to_buy == 0:
            return

        total_cost = shares_to_buy * buy_price * (1 + self.commission_rate)
        self.cash -= total_cost
        self.shares = shares_to_buy
        
        self.position = {
            'entry_date': date,
            'entry_price': buy_price,
            'shares': shares_to_buy,
            'cost': total_cost,
            'days_held': 0,
            'stop_price': initial_stop,  # 动态止损线
            'highest_price': buy_price   # 用于计算回撤
        }
        
        self.logger.debug(f"Buy at {date}: {shares_to_buy} shares @ {buy_price:.2f}, Stop: {initial_stop:.2f}")

    def _check_exit(self, date, ohlc):
        """检查退出 (ATR 移动止损)"""
        pos = self.position
        pos['days_held'] += 1
        
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        atr = ohlc['atr']
        
        # 更新最高价
        if high > pos['highest_price']:
            pos['highest_price'] = high
            
        # 更新移动止损 (Chandelier Exit)
        # 只有当价格上涨时，止损线才上移
        if self.use_atr_risk and atr > 0:
            # 止损线挂在最高价下方 N * ATR 处
            new_stop = pos['highest_price'] - (atr * self.atr_multiplier)
            if new_stop > pos['stop_price']:
                pos['stop_price'] = new_stop
        
        exit_price = None
        reason = None
        
        # 1. 触发止损 (价格跌破 stop_price)
        if low <= pos['stop_price']:
            exit_price = min(ohlc['open'], pos['stop_price']) # 如果开盘就低开，按开盘价
            reason = 'stop_loss' # 包含移动止损
            
        # 2. 止盈 (如果配置了固定止盈) - 可选，Trend Following 通常不设固定止盈
        # 这里保留一个较大的硬止盈作为保护
        tp_price = pos['entry_price'] * (1 + self.take_profit_pct)
        if not exit_price and high >= tp_price:
            exit_price = max(ohlc['open'], tp_price)
            reason = 'take_profit'
            
        # 3. 超时 (Time Exit)
        if not exit_price and pos['days_held'] >= self.max_holding_days:
            exit_price = close
            reason = 'time_exit'
            
        if exit_price:
            self._execute_sell(date, exit_price, reason)

    def _execute_sell(self, date, price, reason):
        """执行卖出"""
        real_price = price * (1 - self.slippage)
        revenue = real_price * self.shares
        fee = revenue * self.commission_rate
        net_revenue = revenue - fee
        
        pnl = net_revenue - self.position['cost']
        pnl_pct = pnl / self.position['cost']
        
        self.cash += net_revenue
        
        self.trades.append({
            'entry_date': self.position['entry_date'],
            'entry_price': self.position['entry_price'],
            'exit_date': date,
            'exit_price': real_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'days_held': self.position['days_held']
        })
        
        self.shares = 0
        self.position = None
        
        self.logger.debug(f"Sell at {date} ({reason}): PnL {pnl:.2f}")

    def get_performance_metrics(self, benchmark_data=None):
        """计算回测指标"""
        if not self.equity_curve:
            return {}
            
        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        initial_equity = self.initial_capital
        final_equity = df['equity'].iloc[-1]
        
        total_return = (final_equity - initial_equity) / initial_equity
        
        # 最大回撤
        df['max_equity'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['max_equity']) / df['max_equity']
        max_drawdown = df['drawdown'].min()
        
        # 交易统计
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 平均收益
        avg_return = np.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0.0
        
        metrics = {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'trades': self.trades
        }
        
        # 基准对比逻辑保持不变...
        if benchmark_data is not None and not benchmark_data.empty:
            bm = benchmark_data.copy()
            if 'date' in bm.columns:
                bm['date'] = pd.to_datetime(bm['date']).dt.normalize()
                bm = bm.set_index('date')
            start = df.index[0]
            end = df.index[-1]
            bm = bm[(bm.index >= start) & (bm.index <= end)]
            if not bm.empty:
                bm_ret = (bm['close'].iloc[-1] - bm['close'].iloc[0]) / bm['close'].iloc[0]
                
                # 基准最大回撤
                bm['max_close'] = bm['close'].cummax()
                bm['drawdown'] = (bm['close'] - bm['max_close']) / bm['max_close']
                bm_max_drawdown = bm['drawdown'].min()
                
                metrics['benchmark_return'] = bm_ret
                metrics['benchmark_max_drawdown'] = bm_max_drawdown
                metrics['excess_return'] = total_return - bm_ret
                
        return metrics
