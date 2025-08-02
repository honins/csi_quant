# 📚 API参考文档

本文档详细介绍了中证500指数量化交易系统的所有命令行接口和配置选项。

## 🎯 命令行接口

### 基本语法

```bash
python run.py <command> [options]
```

### 全局选项

| 选项 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--verbose` | `-v` | 启用详细输出 | False |
| `--quiet` | `-q` | 静默模式 | False |
| `--no-timer` | 无 | 禁用执行时间统计 | False |
| `--config` | 无 | 指定配置文件路径 | config/system.yaml |
| `--log-level` | 无 | 设置日志级别 | INFO |

## 📋 命令详细说明

### 1. help - 帮助信息

```bash
python run.py help
```

**功能**：显示所有可用命令和选项的帮助信息。

**输出**：命令列表和使用说明。

### 2. ai - AI优化训练

```bash
python run.py ai [options]
```

**功能**：运行AI参数优化和模型训练。

**选项**：
- `--mode` / `-m`：优化模式
  - `optimize`：完整优化（默认）
  - `incremental`：增量训练
  - `full`：完全重训练
  - `demo`：演示预测
- `--iter` / `-i`：迭代次数（默认：20）

**示例**：
```bash
# 标准AI优化
python run.py ai

# 增量训练模式
python run.py ai -m incremental

# 指定迭代次数
python run.py ai -i 50
```

**输出**：
- 优化进度
- 最终策略得分
- 成功率统计
- 模型保存路径

### 3. basic - 基础策略测试

```bash
python run.py basic
```

**功能**：运行基础策略测试，验证系统功能。

**输出**：
- 策略测试结果
- 基本性能指标
- 系统状态检查

### 4. predict - 单日预测

```bash
python run.py predict [date]
```

**功能**：对指定日期或最新交易日进行相对低点预测。

**参数**：
- `date`：预测日期（格式：YYYY-MM-DD），可选

**示例**：
```bash
# 预测最新交易日
python run.py predict

# 预测指定日期
python run.py predict 2024-01-15
```

**输出**：
- 预测结果（是/否）
- 置信度分数
- 技术指标分析
- 建议操作

### 5. backtest - 滚动回测

```bash
python run.py backtest [start_date] [end_date]
```

**功能**：在指定时间范围内进行滚动回测。

**参数**：
- `start_date`：开始日期（格式：YYYY-MM-DD），可选
- `end_date`：结束日期（格式：YYYY-MM-DD），可选

**示例**：
```bash
# 默认时间范围回测
python run.py backtest

# 指定时间范围
python run.py backtest 2023-01-01 2023-12-31
```

**输出**：
- 回测统计结果
- 成功率分析
- 收益率分布
- 可视化图表

### 6. config - 配置查看

```bash
python run.py config
```

**功能**：显示当前系统配置信息。

**输出**：
- 策略参数
- 系统设置
- 优化配置
- 文件路径

### 7. fetch - 数据获取

```bash
python run.py fetch
```

**功能**：获取最新的市场数据。

**输出**：
- 数据获取状态
- 更新的数据范围
- 数据质量检查

### 8. status - 系统状态

```bash
python run.py status
```

**功能**：检查系统状态和健康度。

**输出**：
- 模型状态
- 数据状态
- 配置状态
- 依赖检查

### 9. test - 单元测试

```bash
python run.py test
```

**功能**：运行系统单元测试。

**输出**：
- 测试结果
- 覆盖率报告
- 错误详情

### 10. all - 全部测试

```bash
python run.py all
```

**功能**：运行完整的系统测试流程。

**输出**：
- 各模块测试结果
- 综合性能报告
- 系统健康度评估

## ⚙️ 配置文件API

### 配置文件结构

```
config/
├── system.yaml          # 系统基础配置
├── strategy.yaml        # 策略优化配置
└── optimized_params.yaml # AI优化后的参数
```

### system.yaml 配置项

#### 数据配置

```yaml
data:
  symbol: "000905.SH"           # 指数代码
  start_date: "2020-01-01"      # 数据开始日期
  end_date: null                # 数据结束日期（null为最新）
  data_source: "akshare"        # 数据源
  frequency: "1d"               # 数据频率
```

#### AI配置

```yaml
ai_optimization:
  population_size: 70           # 遗传算法种群大小
  generations: 20               # 遗传算法代数
  mutation_rate: 0.1            # 变异率
  crossover_rate: 0.8           # 交叉率
  early_stopping_patience: 5    # 早停耐心值
```

#### 系统配置

```yaml
system:
  models_dir: "models"          # 模型保存目录
  results_dir: "results"        # 结果保存目录
  log_level: "INFO"             # 日志级别
  enable_notifications: false   # 是否启用通知
```

### strategy.yaml 配置项

#### 策略参数

```yaml
strategy:
  rise_threshold: 0.04          # 涨幅阈值
  max_days: 20                  # 最大持有天数
```

#### 置信度权重

```yaml
confidence_weights:
  rsi_oversold_threshold: 30.0
  rsi_low_threshold: 45.0
  ma_all_below: 0.35
  dynamic_confidence_adjustment: 0.2384
  market_sentiment_weight: 0.1767
  trend_strength_weight: 0.16
  volume_panic_threshold: 1.3
  volume_panic_bonus: 0.12
  volume_surge_bonus: 0.06
  volume_shrink_penalty: 0.68
  bb_near_threshold: 1.01049
  recent_decline: 0.2
  macd_negative: 0.1
  price_decline_threshold: -0.018
  final_threshold: 0.25
```

#### 优化范围

```yaml
optimization_ranges:
  rsi_oversold_threshold: [25, 35]
  rsi_low_threshold: [40, 50]
  ma_all_below: [0.2, 0.4]
  # ... 其他参数范围
```

## 🔧 工具脚本API

### reset_strategy_params.py

```bash
python reset_strategy_params.py [options]
```

**选项**：
- `--all`：重置所有参数
- `--strategy`：仅重置策略参数
- `--confidence`：仅重置置信度权重
- `--optimization`：仅重置优化配置
- `--backup`：创建备份
- `--force`：强制重置，不询问确认
- `--show`：显示当前参数

**示例**：
```bash
# 重置所有参数并强制执行
python reset_strategy_params.py --all --force

# 仅重置策略参数
python reset_strategy_params.py --strategy

# 显示当前参数
python reset_strategy_params.py --show
```

## 📊 输出格式

### JSON格式输出

某些命令支持JSON格式输出，使用 `--format json` 选项：

```bash
python run.py predict --format json
```

输出示例：
```json
{
  "date": "2024-01-15",
  "prediction": true,
  "confidence": 0.75,
  "indicators": {
    "rsi": 28.5,
    "ma_below": true,
    "volume_ratio": 1.45
  }
}
```

### 表格格式输出

默认使用表格格式，便于阅读：

```
日期        | 预测结果 | 置信度 | RSI  | 建议
2024-01-15 | 是       | 0.75   | 28.5 | 关注买入
```

## 🚨 错误代码

| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| 1 | 配置文件错误 | 检查配置文件格式 |
| 2 | 数据获取失败 | 检查网络连接 |
| 3 | 模型文件不存在 | 运行AI优化生成模型 |
| 4 | 参数验证失败 | 检查参数范围 |
| 5 | 内存不足 | 增加系统内存或减少数据量 |

## 🔍 调试选项

### 详细日志

```bash
python run.py ai --log-level DEBUG
```

### 性能分析

```bash
python run.py ai --profile
```

### 内存监控

```bash
python run.py ai --memory-monitor
```

## 🌐 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `CSI_CONFIG_PATH` | 自定义配置文件路径 | config/system.yaml |
| `CSI_DATA_DIR` | 数据目录 | data/ |
| `CSI_MODELS_DIR` | 模型目录 | models/ |
| `CSI_LOG_LEVEL` | 日志级别 | INFO |
| `CSI_ENABLE_GPU` | 启用GPU加速 | false |

**使用示例**：
```bash
# Linux/Mac
export CSI_CONFIG_PATH=/path/to/custom.yaml
python run.py ai

# Windows
set CSI_CONFIG_PATH=C:\path\to\custom.yaml
python run.py ai
```

## 📝 API使用最佳实践

### 1. 命令组合

```bash
# 完整的优化流程
python run.py fetch && python run.py ai && python run.py backtest
```

### 2. 批处理脚本

```bash
#!/bin/bash
# daily_routine.sh
source venv/bin/activate
python run.py fetch
python run.py predict
deactivate
```

### 3. 错误处理

```bash
python run.py ai || echo "AI优化失败，请检查日志"
```

### 4. 输出重定向

```bash
# 保存输出到文件
python run.py ai > optimization_log.txt 2>&1

# 仅保存错误信息
python run.py ai 2> error_log.txt
```

---

**注意**：所有命令都应在激活的虚拟环境中运行，以确保依赖正确加载。