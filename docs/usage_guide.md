# 使用指南

## 快速开始

### 1. 环境准备

确保您的Python环境满足以下要求：
- Python 3.7+
- pip包管理器

### 2. 安装依赖

在项目根目录下运行：

```bash
pip install -r requirements.txt
```

### 3. 配置系统

编辑 `config/config.yaml` 文件，根据您的需求调整配置参数。

### 4. 运行示例

- **运行基础回测**

  ```bash
  python run.py basic
  ```

- **运行AI优化回测**

  ```bash
  python run.py ai_optimization
  ```

- **单日低点预测**

  ```bash
  python predict_single_day.py <YYYY-MM-DD>
  # 示例：python predict_single_day.py 2024-06-01
  ```

- **LLM驱动策略优化**

  ```bash
  python llm_strategy_optimizer.py
  ```

## 详细使用说明

### 数据模块使用

数据模块现在支持从单一CSV文件加载中证500指数历史数据，文件路径可在 `config/config.yaml` 中配置。

```python
from src.data.data_module import DataModule
from src.utils.utils import load_config

# 加载配置
config = load_config("config/config.yaml")

# 创建数据模块
data_module = DataModule(config)

# 获取历史数据 (将从配置的data_file_path中读取指定日期范围的数据)
data = data_module.get_history_data("2024-01-01", "2024-12-31")

# 预处理数据
processed_data = data_module.preprocess_data(data)
```

### 策略模块使用

```python
from src.strategy.strategy_module import StrategyModule

# 创建策略模块
strategy_module = StrategyModule(config)

# 识别相对低点
result = strategy_module.identify_relative_low(processed_data)

# 运行回测
backtest_results = strategy_module.backtest(processed_data)

# 评估策略
evaluation = strategy_module.evaluate_strategy(backtest_results)

# 可视化结果
chart_path = strategy_module.visualize_backtest(backtest_results)
```

### AI优化模块使用

AI优化模块现在支持时间序列感知的训练集/测试集划分和数据有效性加权，以更好地缓解过拟合问题。

```python
from src.ai.ai_optimizer import AIOptimizer

# 创建AI优化器
ai_optimizer = AIOptimizer(config)

# 优化策略参数 (此方法可能内部调用train_prediction_model)
# optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, processed_data)

# 训练预测模型
training_result = ai_optimizer.train_prediction_model(processed_data, strategy_module)

# 预测相对低点
prediction = ai_optimizer.predict_low_point(processed_data)
```

### 单日预测脚本使用

`predict_single_day.py` 脚本允许您输入一个日期，预测该日期是否为相对低点，并提供验证结果。

```python
# 在命令行中运行
python predict_single_day.py 2024-06-01
```

### LLM驱动策略优化使用

`llm_strategy_optimizer.py` 脚本利用LLM的能力自动调整策略参数并优化策略。

```python
# 在命令行中运行
python llm_strategy_optimizer.py
```

## 配置说明

### 系统配置

- `mode`: 运行模式
  - `backtest`: 回测模式
  - `simulation`: 模拟模式
  - `live`: 实盘模式

- `log_level`: 日志级别
  - `DEBUG`: 调试信息
  - `INFO`: 一般信息
  - `WARNING`: 警告信息
  - `ERROR`: 错误信息

### 策略配置

- `rise_threshold`: 上涨阈值，默认0.04（4%）
- `max_days`: 最大交易日数，默认20天

### AI配置

- `enable`: 是否启用AI优化
- `model_type`: 模型类型
  - `machine_learning`: 机器学习模型
  - `deep_learning`: 深度学习模型
  - `reinforcement_learning`: 强化学习模型
- `train_test_split_ratio`: 训练集占总数据的比例，剩余为测试集 (例如: 0.8)
- `data_decay_rate`: 数据有效性衰减率，用于计算样本权重 (例如: 0.001)

### 通知配置

- `methods`: 通知方式
  - `console`: 控制台输出
  - `email`: 邮件通知
  - `sms`: 短信通知（需要额外配置）

## 常见问题

### Q: 如何修改相对低点的定义？

A: 修改 `config/config.yaml` 中的 `strategy.rise_threshold` 和 `strategy.max_days` 参数。

### Q: 如何添加新的技术指标？

A: 在 `src/data/data_module.py` 的 `_calculate_technical_indicators` 方法中添加新的指标计算。

### Q: 如何自定义通知内容？

A: 修改 `src/notification/notification_module.py` 的 `_generate_notification_content` 方法。

### Q: 如何使用真实的股票数据？

A: 需要集成真实的数据源API，如掘金量化、聚宽等，替换 `src/data/data_module.py` 中的模拟数据生成部分。

## 性能优化建议

1. **数据缓存**: 系统会自动缓存历史数据，避免重复获取。

2. **参数调优**: 使用AI优化功能自动寻找最优参数。

3. **特征选择**: 根据特征重要性分析，选择最有效的技术指标。

4. **模型更新**: 定期重新训练AI模型，适应市场变化。

## 扩展开发

### 添加新的优化算法

1. 在 `src/ai/ai_optimizer.py` 中添加新的优化方法
2. 在配置文件中添加相应的参数
3. 在测试文件中添加相应的测试用例

### 添加新的通知方式

1. 在 `src/notification/notification_module.py` 中添加新的通知方法
2. 在配置文件中添加相应的配置项
3. 测试新的通知功能

### 集成实时数据源

1. 修改 `src/data/data_module.py`，添加实时数据获取功能
2. 配置相应的API密钥和参数
3. 测试实时数据获取和处理

## 许可证

本项目采用MIT许可证，详见LICENSE文件。




### AI预测模型滚动回测

`run_rolling_backtest.py` 脚本用于执行AI预测模型的滚动回测，以评估模型在不同时间段的真实表现。

```bash
python run_rolling_backtest.py <start_date> <end_date> <training_window_days>
# 示例：python run_rolling_backtest.py 2023-01-01 2023-03-31 365
```

**参数:**
- `<start_date>`: 回测开始日期 (YYYY-MM-DD)。
- `<end_date>`: 回测结束日期 (YYYY-MM-DD)。
- `<training_window_days>`: 训练数据的时间窗口大小（天数）。


