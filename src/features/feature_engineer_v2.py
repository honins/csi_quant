import pandas as pd
import numpy as np
import logging

class FeatureEngineerV2:
    """
    Feature Engineer V2: 多周期特征与高级指标计算
    
    包含:
    1. 基础日线特征 (ATR, ADX, OBV, RSI, MACD)
    2. 周线级别特征 (Weekly Trend)
    3. 波动率归一化特征
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger("FeatureEngineerV2")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有特征"""
        if df.empty:
            return df
            
        data = df.copy()
        
        # 1. 基础指标计算
        data = self._calculate_volatility_features(data)  # ATR, NATR
        data = self._calculate_trend_features(data)       # ADX, DMI
        data = self._calculate_volume_features(data)      # OBV, MFI
        data = self._calculate_basic_features(data)       # RSI, MACD, MA
        
        # 2. 周线特征 (重点：防止未来函数，使用 shift)
        data = self._calculate_weekly_features(data)
        
        # 3. 目标变量构建 (用于训练，预测时可忽略)
        # 预测未来5天夏普比率: Mean_Ret_5d / Std_Ret_5d
        # data = self._create_targets(data) 
        
        # 清理 NaN (由于 rolling 和 diff 产生)
        data.dropna(inplace=True)
        
        return data

    def _calculate_volatility_features(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """计算波动率指标 (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (Simple Moving Average of TR)
        df['atr'] = tr.rolling(window=period).mean()
        
        # NATR (Normalized ATR): ATR / Close * 100
        # 反映相对波动率，用于跨价格比较
        df['natr'] = (df['atr'] / close) * 100
        
        # TR (用于后续计算)
        df['tr'] = tr
        
        return df

    def _calculate_trend_features(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """计算趋势强度指标 (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # +DM, -DM
        up = high.diff()
        down = -low.diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Smoothed TR, +DM, -DM
        # 使用 Wilder's Smoothing (alpha = 1/period) 或者是简单的 Rolling mean
        # 这里使用 Rolling mean 模拟，为了简单和速度
        tr_smooth = df['tr'].rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_smooth)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # 趋势状态分类: 
        # ADX > 25: 强趋势
        # ADX < 20: 震荡
        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标 (OBV)"""
        close = df['close']
        volume = df['volume']
        
        # OBV
        change = close.diff()
        direction = np.where(change > 0, 1, np.where(change < 0, -1, 0))
        obv = (direction * volume).cumsum()
        
        df['obv'] = obv
        
        # OBV Slope (5日趋势)
        df['obv_slope'] = obv.diff(5)
        
        return df

    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础均线和震荡指标"""
        close = df['close']
        
        # MA
        for window in [5, 10, 20, 60]:
            df[f'ma{window}'] = close.rolling(window=window).mean()
            # 乖离率
            df[f'bias_{window}'] = (close - df[f'ma{window}']) / df[f'ma{window}']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
        
        return df

    def _calculate_weekly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算周线级别特征，并合并回日线
        注意：必须使用 shift(1) 避免未来数据泄露
        """
        # 确保有日期索引
        if 'date' in df.columns:
            temp_df = df.set_index('date').copy()
        else:
            temp_df = df.copy()
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                # 尝试转换
                try:
                    temp_df.index = pd.to_datetime(temp_df.index)
                except:
                    self.logger.warning("无法转换索引为日期，跳过周线特征")
                    return df

        # 重采样为周线 (每周五结束)
        weekly_logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 仅保留存在的列
        agg_dict = {k: v for k, v in weekly_logic.items() if k in temp_df.columns}
        
        w_df = temp_df.resample('W-FRI').agg(agg_dict)
        
        # 计算周线指标
        w_df['w_ma20'] = w_df['close'].rolling(20).mean()
        w_df['w_macd'] = w_df['close'].ewm(span=12).mean() - w_df['close'].ewm(span=26).mean()
        
        # 判断周线趋势
        # 1: 牛市 (价格在20周均线上方)
        # -1: 熊市
        w_df['w_trend'] = np.where(w_df['close'] > w_df['w_ma20'], 1, -1)
        
        # === 关键：Shift(1) ===
        # 本周五计算出的指标，只能用于下周一到下周五的预测
        w_features = w_df[['w_ma20', 'w_macd', 'w_trend']].shift(1)
        
        # 将周线特征合并回日线 (使用 ffill 填充一周内的值)
        # reindex 会自动对齐日期
        merged_df = temp_df.join(w_features, how='left')
        merged_df.ffill(inplace=True)
        
        # 恢复原始索引/列结构
        if 'date' in df.columns:
            merged_df.reset_index(inplace=True)
            # 确保 date 列名正确 (reset_index 可能会叫 index)
            if 'index' in merged_df.columns and 'date' not in merged_df.columns:
                merged_df.rename(columns={'index': 'date'}, inplace=True)
                
        return merged_df
