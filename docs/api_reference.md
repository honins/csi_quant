# API文档

## 数据模块 (DataModule)

### 类初始化

```python
DataModule(config: Dict[str, Any])
```

**参数:**
- `config`: 配置字典

### 方法

#### get_history_data

```python
get_history_data(start_date: str, end_date: str) -> pd.DataFrame
```

获取中证500指数历史数据。

**参数:**
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)

**返回:**
- `pd.DataFrame`: 历史数据

**示例:**
```python
data = data_module.get_history_data("2024-01-01", "2024-12-31")
```

#### preprocess_data

```python
preprocess_data(data: pd.DataFrame) -> pd.DataFrame
```

预处理数据，添加技术指标。

**参数:**
- `data`: 原始数据

**返回:**
- `pd.DataFrame`: 预处理后的数据

#### validate_data

```python
validate_data(data: pd.DataFrame) -> bool
```

验证数据质量。

**参数:**
- `data`: 数据

**返回:**
- `bool`: 数据是否有效

## 策略模块 (StrategyModule)

### 类初始化

```python
StrategyModule(config: Dict[str, Any])
```

**参数:**
- `config`: 配置字典

### 方法

#### identify_relative_low

```python
identify_relative_low(data: pd.DataFrame) -> Dict[str, Any]
```

识别相对低点。

**参数:**
- `data`: 市场数据

**返回:**
- `dict`: 识别结果
  - `date`: 日期
  - `price`: 价格
  - `is_low_point`: 是否为相对低点
  - `confidence`: 置信度
  - `reasons`: 识别原因列表

**示例:**
```python
result = strategy_module.identify_relative_low(data)
print(f"是否为相对低点: {result["is_low_point"]}")
print(f"置信度: {result["confidence"]:.2%}")
```

#### backtest

```python
backtest(data: pd.DataFrame) -> pd.DataFrame
```

回测策略。

**参数:**
- `data`: 历史数据

**返回:**
- `pd.DataFrame`: 回测结果

#### evaluate_strategy

```python
evaluate_strategy(backtest_results: pd.DataFrame) -> Dict[str, Any]
```

评估策略。

**参数:**
- `backtest_results`: 回测结果

**返回:**
- `dict`: 评估结果
  - `total_points`: 识别点数
  - `success_rate`: 成功率
  - `avg_rise`: 平均涨幅
  - `avg_days`: 平均天数
  - `score`: 综合得分

#### update_params

```python
update_params(params: Dict[str, Any]) -> None
```

更新策略参数。

**参数:**
- `params`: 新参数
  - `rise_threshold`: 上涨阈值
  - `max_days`: 最大交易日数

#### get_params

```python
get_params() -> Dict[str, Any]
```

获取当前策略参数。

**返回:**
- `dict`: 当前参数

## AI优化模块 (AIOptimizer)

### 类初始化

```python
AIOptimizer(config: Dict[str, Any])
```

**参数:**
- `config`: 配置字典

### 方法

#### optimize_strategy_parameters

```python
optimize_strategy_parameters(strategy_module, data: pd.DataFrame) -> Dict[str, Any]
```

优化策略参数。

**参数:**
- `strategy_module`: 策略模块实例
- `data`: 历史数据

**返回:**
- `dict`: 优化后的参数

#### train_prediction_model

```python
train_prediction_model(data: pd.DataFrame, strategy_module) -> Dict[str, Any]
```

训练预测模型。此方法现在支持时间序列感知的训练集/测试集划分和数据有效性加权，以更好地缓解过拟合问题。模型评估在测试集上进行。

**参数:**
- `data`: 历史数据
- `strategy_module`: 策略模块实例

**返回:**
- `dict`: 训练结果
  - `success`: 是否成功
  - `accuracy`: 准确率 (测试集)
  - `precision`: 精确率 (测试集)
  - `recall`: 召回率 (测试集)
  - `f1_score`: F1得分 (测试集)

#### predict_low_point

```python
predict_low_point(data: pd.DataFrame) -> Dict[str, Any]
```

预测相对低点。

**参数:**
- `data`: 市场数据

**返回:**
- `dict`: 预测结果
  - `is_low_point`: 是否为相对低点
  - `confidence`: 置信度

#### get_feature_importance

```python
get_feature_importance() -> Optional[Dict[str, float]]
```

获取特征重要性。

**返回:**
- `dict`: 特征重要性，如果模型未训练返回None

#### run_genetic_algorithm

```python
run_genetic_algorithm(evaluate_func, population_size: int = 20, generations: int = 10) -> Dict[str, Any]
```

运行遗传算法优化。

**参数:**
- `evaluate_func`: 评估函数
- `population_size`: 种群大小
- `generations`: 迭代代数

**返回:**
- `dict`: 最优参数

## 脚本 (Scripts)

### predict_single_day.py

单日相对低点预测脚本。允许用户输入日期，预测该日期是否为相对低点，并验证结果。

**用法:**

```bash
python predict_single_day.py <YYYY-MM-DD>
# 示例：python predict_single_day.py 2024-06-01
```

**参数:**
- `<YYYY-MM-DD>`: 要预测的日期字符串。

**输出:**
- 预测结果（是否为相对低点，置信度）。
- 验证结果（实际是否为相对低点，未来最大涨幅，达到目标涨幅所需天数）。

### llm_strategy_optimizer.py

LLM驱动的策略优化脚本。利用LLM的能力自动调整策略参数并优化策略。

**用法:**

```bash
python llm_strategy_optimizer.py
```

**参数:**
- 无需命令行参数，通过 `config/config.yaml` 配置。

**输出:**
- 每次迭代中LLM建议的参数、回测得分以及最佳策略参数和得分。

## 通知模块 (NotificationModule)

### 类初始化

```python
NotificationModule(config: Dict[str, Any])
```

**参数:**
- `config`: 配置字典

### 方法

#### send_low_point_notification

```python
send_low_point_notification(result: Dict[str, Any]) -> bool
```

发送相对低点通知。

**参数:**
- `result`: 识别结果

**返回:**
- `bool`: 是否发送成功

#### get_notification_history

```python
get_notification_history(days: int = 30) -> List[Dict[str, Any]]
```

获取通知历史。

**参数:**
- `days`: 查询天数

**返回:**
- `list`: 通知历史列表

## 工具函数

### setup_logging

```python
setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None
```

设置日志配置。

**参数:**
- `log_level`: 日志级别
- `log_file`: 日志文件路径

### load_config

```python
load_config(config_path: str) -> Dict[str, Any]
```

加载配置文件。

**参数:**
- `config_path`: 配置文件路径

**返回:**
- `dict`: 配置字典

### save_config

```python
save_config(config: Dict[str, Any], config_path: str) -> bool
```

保存配置文件。

**参数:**
- `config`: 配置字典
- `config_path`: 配置文件路径

**返回:**
- `bool`: 是否保存成功

### get_trading_days

```python
get_trading_days(start_date: str, end_date: str) -> list
```

获取交易日列表。

**参数:**
- `start_date`: 开始日期
- `end_date`: 结束日期

**返回:**
- `list`: 交易日列表

### calculate_returns

```python
calculate_returns(prices: list) -> list
```

计算收益率序列。

**参数:**
- `prices`: 价格序列

**返回:**
- `list`: 收益率序列

### calculate_volatility

```python
calculate_volatility(returns: list, annualize: bool = True) -> float
```

计算波动率。

**参数:**
- `returns`: 收益率序列
- `annualize`: 是否年化

**返回:**
- `float`: 波动率

### calculate_sharpe_ratio

```python
calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.03) -> float
```

计算夏普比率。

**参数:**
- `returns`: 收益率序列
- `risk_free_rate`: 无风险利率

**返回:**
- `float`: 夏普比率

### calculate_max_drawdown

```python
calculate_max_drawdown(prices: list) -> Dict[str, Any]
```

计算最大回撤。

**参数:**
- `prices`: 价格序列

**返回:**
- `dict`: 最大回撤信息

## 数据结构

### 历史数据格式

```python
{
    "date": pd.Timestamp,      # 日期
    "open": float,             # 开盘价
    "high": float,             # 最高价
    "low": float,              # 最低价
    "close": float,            # 收盘价
    "volume": int,             # 成交量
    "amount": float,           # 成交额
    # 技术指标
    "ma5": float,              # 5日移动平均
    "ma10": float,             # 10日移动平均
    "ma20": float,             # 20日移动平均
    "ma60": float,             # 60日移动平均
    "rsi": float,              # 相对强弱指标
    "macd": float,             # MACD
    "signal": float,           # MACD信号线
    "hist": float,             # MACD柱状图
    "bb_upper": float,         # 布林带上轨
    "bb_lower": float,         # 布林带下轨
    # 其他指标...
}
```

### 识别结果格式

```python
{
    "date": str,               # 日期
    "price": float,            # 价格
    "is_low_point": bool,      # 是否为相对低点
    "confidence": float,       # 置信度 (0-1)
    "reasons": List[str],      # 识别原因
    "technical_indicators": {  # 技术指标
        "ma5": float,
        "ma10": float,
        "rsi": float,
        # ...
    }
}
```

### 回测结果格式

```python
{
    # 原始数据列...
    "is_low_point": bool,      # 是否为相对低点
    "future_max_rise": float,  # 未来最大涨幅
    "days_to_rise": int,       # 达到目标涨幅天数
    "max_rise_date": str,      # 最大涨幅日期
}
```

### 策略评估格式

```python
{
    "total_points": int,       # 识别点数
    "success_rate": float,     # 成功率
    "avg_rise": float,         # 平均涨幅
    "avg_days": float,         # 平均天数
    "max_rise": float,         # 最大涨幅
    "min_rise": float,         # 最小涨幅
    "score": float,            # 综合得分
    "rise_threshold": float,   # 上涨阈值
    "max_days": int,           # 最大交易日数
}
```




### AI预测模型滚动回测脚本 (run_rolling_backtest.py)

该脚本用于执行AI预测模型的滚动回测，模拟模型在不同时间点进行训练和预测，并统计预测的成功率，以更真实地评估模型的长期表现。

**用法:**

```bash
python run_rolling_backtest.py <start_date> <end_date> <training_window_days>
# 示例：python run_rolling_backtest.py 2023-01-01 2023-03-31 365
```

**参数:**
- `<start_date>`: 回测开始日期 (YYYY-MM-DD)。
- `<end_date>`: 回测结束日期 (YYYY-MM-DD)。
- `<training_window_days>`: 训练数据的时间窗口大小（天数）。

**输出:**
- 每次预测的详细日志。
- 最终的预测成功率统计。
- 生成 `results/rolling_backtest_results.png`，可视化预测结果与实际低点。


