# 配置文件说明文档

本文档详细说明CSI量化交易系统中三个主要YAML配置文件的参数和作用。

## 1. strategy.yaml - 策略配置文件

### 文件作用
- 定义量化交易策略的核心参数
- 提供AI优化的参数范围
- 配置策略评分和验证规则
- 作为系统的基础配置文件

### 主要配置段落

#### 1.1 advanced_optimization - 高级优化配置
```yaml
advanced_optimization:
  enabled: true                    # 启用高级优化
  high_precision_mode: true        # 高精度模式
  use_hierarchical: true           # 使用分层优化
  use_scipy: true                  # 使用scipy优化器
  use_time_series_cv: true         # 使用时间序列交叉验证
```

#### 1.2 ai_scoring - AI评分配置
```yaml
ai_scoring:
  # AI模型评分相关参数
```

#### 1.3 confidence_weights - 置信度权重参数
```yaml
confidence_weights:
  # 详细的置信度计算权重参数
  # 包含60+个技术指标和市场情绪参数
  dynamic_confidence_adjustment: 0.1    # 动态置信度调整
  final_threshold: 0.5                  # 最终置信度阈值
  market_sentiment_weight: 0.15         # 市场情绪权重
  # ... 更多参数
```

#### 1.4 optimization_ranges - 优化参数范围
```yaml
optimization_ranges:
  # 定义每个参数的优化范围
  final_threshold:
    max: 0.6
    min: 0.25
    step: 0.02
  # ... 更多参数范围
```

#### 1.5 overfitting_threshold - 过拟合检测
```yaml
overfitting_threshold: 0.9    # 过拟合检测阈值（90%）
```

#### 1.6 default_strategy - 默认策略参数
```yaml
default_strategy:
  confidence_weights:             # 简化版置信度权重
    dynamic_confidence_adjustment: 0.1
    final_threshold: 0.5
    market_sentiment_weight: 0.15
    rsi_low_threshold: 40
    rsi_oversold_threshold: 30
    trend_strength_weight: 0.15
  max_days: 20                   # 最大持有天数
  rise_threshold: 0.04           # 涨幅阈值（4%）
```

#### 1.7 strategy_scoring - 策略评分配置
```yaml
strategy_scoring:
  days_benchmark: 10.0           # 天数基准（10天）
  days_weight: 0.2               # 天数权重（20%）
  rise_benchmark: 0.1            # 涨幅基准（10%）
  rise_weight: 0.3               # 涨幅权重（30%）
  success_weight: 0.5            # 成功率权重（50%）
```

#### 1.8 validation - 验证配置
```yaml
validation:
  test_ratio: 0.10               # 测试集比例（10%）
  train_ratio: 0.75              # 训练集比例（75%）
  validation_ratio: 0.15         # 验证集比例（15%）
  walk_forward:                  # 前向验证配置
    enabled: true                # 启用前向验证
    step_size: 63                # 步长（约3个月）
    window_size: 252             # 窗口大小（约1年）
```

#### 1.9 zero_confidence_threshold - 零置信度阈值
```yaml
zero_confidence_threshold: 0.5  # 零置信度检测阈值（50%）
```

---

## 2. optimized_params.yaml - 优化参数文件

### 文件作用
- 存储AI优化后的最佳参数值
- 覆盖strategy.yaml中的对应参数
- 记录优化历史和元数据
- 提供参数分类管理

### 主要配置段落

#### 2.1 metadata - 元数据
```yaml
metadata:
  last_updated: "2025-08-02T21:59:41"  # 最后更新时间
  optimization_count: 15               # 优化次数
  best_score: 0.7234                   # 最佳得分
  data_period: "2020-01-01 to 2024-12-31"  # 数据周期
```

#### 2.2 strategy_params - 策略参数
```yaml
strategy_params:
  rise_threshold: 0.04           # 涨幅阈值
  max_days: 20                   # 最大持有天数
```

#### 2.3 confidence_weights - 优化后的置信度权重
```yaml
confidence_weights:
  # AI优化后的最佳置信度权重参数
  final_threshold: 0.45          # 优化后的最终阈值
  # ... 其他优化参数
```

#### 2.4 technical_indicators - 技术指标参数
```yaml
technical_indicators:
  rsi_oversold_threshold: 28     # RSI超卖阈值
  rsi_low_threshold: 35          # RSI低位阈值
  # ... 其他技术指标参数
```

#### 2.5 volume_params - 成交量参数
```yaml
volume_params:
  volume_panic_threshold: 1.45   # 成交量恐慌阈值
  volume_surge_threshold: 1.25   # 成交量激增阈值
  # ... 其他成交量参数
```

#### 2.6 market_params - 市场参数
```yaml
market_params:
  market_sentiment_weight: 0.18  # 市场情绪权重
  trend_strength_weight: 0.12    # 趋势强度权重
  # ... 其他市场参数
```

---

## 3. system.yaml - 系统配置文件

### 文件作用
- 配置系统级别的参数
- 定义数据源和API设置
- 配置日志和通知
- 设置系统运行环境

### 主要配置段落

#### 3.1 data - 数据配置
```yaml
data:
  source: "akshare"               # 数据源
  cache_enabled: true            # 启用缓存
  cache_duration: 3600           # 缓存时长（秒）
  update_interval: 86400         # 更新间隔（秒）
```

#### 3.2 logging - 日志配置
```yaml
logging:
  level: "INFO"                  # 日志级别
  file_enabled: true             # 启用文件日志
  console_enabled: true          # 启用控制台日志
  max_file_size: 10485760        # 最大文件大小（10MB）
  backup_count: 5                # 备份文件数量
```

#### 3.3 notification - 通知配置
```yaml
notification:
  enabled: false                 # 启用通知
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    # ... 邮件配置
```

#### 3.4 ai - AI配置
```yaml
ai:
  model_path: "models/"          # 模型路径
  auto_retrain: true             # 自动重训练
  retrain_threshold: 0.1         # 重训练阈值
  validation:                    # 验证配置
    train_ratio: 0.6
    validation_ratio: 0.25
    test_ratio: 0.15
```

#### 3.5 backtest - 回测配置
```yaml
backtest:
  start_date: "2020-01-01"       # 回测开始日期
  end_date: "2024-12-31"         # 回测结束日期
  initial_capital: 100000        # 初始资金
  commission_rate: 0.0003        # 手续费率
```

#### 3.6 results - 结果配置
```yaml
results:
  save_charts: true              # 保存图表
  save_reports: true             # 保存报告
  output_format: "both"          # 输出格式（json/csv/both）
```

---

## 配置文件加载优先级

系统按以下顺序加载配置文件：

1. **system.yaml** - 系统基础配置
2. **strategy.yaml** - 策略配置（覆盖system.yaml中的相同参数）
3. **optimized_params.yaml** - 优化参数（覆盖前两个文件中的相同参数）

### 参数覆盖规则
- 后加载的文件会覆盖先加载文件中的相同参数
- `optimized_params.yaml` 具有最高优先级
- 如果某个参数在后续文件中不存在，则使用前面文件中的值

---

## 配置文件管理建议

### 1. 修改原则
- **system.yaml**: 仅修改系统级配置（数据源、日志等）
- **strategy.yaml**: 修改策略基础参数和优化范围
- **optimized_params.yaml**: 通常由AI自动生成，避免手动修改

### 2. 备份策略
- 系统会自动备份配置文件到 `config/backups/` 目录
- 重要修改前建议手动备份
- 可使用 `scripts/reset_strategy_params.py` 重置参数

### 3. 参数验证
- 系统启动时会自动验证配置参数
- 使用 `param_validator.py` 进行参数合法性检查
- 不合法的参数会使用默认值并记录警告

### 4. 调试技巧
- 查看日志文件了解参数加载过程
- 使用 `config_loader.py` 的调试模式
- 通过 `interactive_guide.py` 交互式配置参数

---

## 常见问题

### Q1: 为什么修改了strategy.yaml但参数没有生效？
A: 检查是否存在 `optimized_params.yaml` 文件，该文件的参数会覆盖 `strategy.yaml` 中的相同参数。

### Q2: 如何重置所有参数到默认值？
A: 运行 `python scripts/reset_strategy_params.py` 脚本。

### Q3: 配置文件损坏怎么办？
A: 从 `config/backups/` 目录恢复备份文件，或删除损坏文件让系统使用默认配置。

### Q4: 如何添加新的配置参数？
A: 在相应的配置文件中添加参数，并在 `param_config.py` 中定义参数规则。

---

*最后更新: 2025-01-03*