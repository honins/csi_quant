# 中证500指数量化交易系统

## 📋 项目简介

这是一个**务实的中证500指数波段交易辅助工具**。

我们的目标非常简单：**利用 AI 技术从历史数据中发现高胜率的买入机会，并通过科学的风控规则帮助交易者拿住利润。**

我们不追求复杂的全自动黑盒交易，而是致力于打造一个透明、可解释的**“智能投顾助手”**，为您提供客观的决策参考。

## 🎯 核心目标

1.  **发现买点**：在市场恐慌或调整到位时，识别出未来上涨概率较高的反弹机会。
2.  **控制回撤**：在单边下跌或震荡市中，通过严格的止损逻辑保护本金。
3.  **扩大利润**：在单边上涨市中，通过激进的持仓策略最大化收益。

## 🏗️ 系统模块

系统由三个核心模块组成，逻辑清晰，易于理解与扩展：

### 1. 信号挖掘 (Signal Discovery)
*   **核心逻辑**：基于机器学习模型，学习历史上的“黄金坑”特征。
*   **输入**：日线量价数据 + 周线趋势特征 + 资金流向指标 (OBV)。
*   **输出**：每日生成一个 **0~1 的置信度评分**。评分越高，未来上涨概率越大。

### 2. 策略过滤 (Strategy Filter)
*   **趋势共振**：结合周线趋势（牛/熊）自动调整买入标准。
    *   *牛市*：放宽标准，不错过任何上车机会。
    *   *熊市*：收紧标准，宁可踏空不做炮灰。
*   **指标验证**：引入 RSI、均线等传统指标进行二次确认。

### 3. 风控执行 (Risk Control)
*   **ATR 动态止损**：使用“平均真实波幅”计算止损位。市场波动大时止损宽一点，波动小时止损紧一点。
*   **移动止盈**：当产生利润后，止损线随价格上涨自动上移，锁定胜果。

## ✨ 主要功能

- 🔍 **买点预测**：每天收盘后运行，自动扫描次日是否值得买入。
- 🛡️ **动态风控**：根据当前市场波动率，给出具体的止损价格建议。
- 📊 **历史回测**：提供真实、严谨的历史回测报告，不画大饼，用数据说话。
- 🚀 **模式切换**：
    *   **激进模式**：适合牛市，全仓出击，追求高收益。
    *   **稳健模式**：适合震荡市，仓位管理，追求低回撤。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 建议使用虚拟环境

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```bash
# 查看帮助
python run.py help

# 运行AI优化
python run.py ai

# 基础策略测试
python run.py basic

# 滚动回测
python run.py backtest

# 单日预测
python run.py predict

# 直接运行滚动回测（支持更多参数）
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31 --verbose
```

## 📊 主要功能

### 1. AI参数优化

系统使用先进的优化算法自动调整策略参数：

- **遗传算法**：全局搜索最优参数组合
- **贝叶斯优化**：高效的参数空间探索
- **过拟合检测**：防止模型过度拟合
- **交叉验证**：确保参数的泛化能力

### 2. 策略参数

系统优化以下核心参数：

- `rise_threshold`: 涨幅阈值
- `max_days`: 最大持有天数
- `rsi_oversold_threshold`: RSI超卖阈值
- `final_threshold`: 最终置信度阈值
- 更多参数详见配置文件

### 3. 技术指标

- **RSI指标**：相对强弱指数，判断超买超卖
- **布林带**：价格通道，识别价格异常
- **成交量分析**：量价关系，确认信号强度
- **移动平均线**：趋势判断

## 📁 项目结构

```
csi_quant/
├── config/                 # 配置文件
│   ├── strategy.yaml       # 策略参数配置
│   ├── system.yaml         # 系统配置
│   └── optimized_params.yaml # 优化后的参数
├── src/                    # 源代码
│   ├── ai/                 # AI优化模块
│   ├── strategy/           # 策略模块
│   ├── data/               # 数据处理模块
│   └── utils/              # 工具模块
├── models/                 # 训练好的模型
├── data/                   # 历史数据
├── results/                # 结果输出
└── examples/               # 示例代码
    ├── run_rolling_backtest.py  # 滚动回测脚本
    ├── predict_single_day.py    # 单日预测示例
    └── basic_test.py            # 基础测试示例
```

## 🔧 配置说明

### 策略配置 (config/strategy.yaml)

包含所有策略参数和优化范围：

```yaml
default_strategy:
  rise_threshold: 0.04      # 涨幅阈值
  max_days: 20              # 最大持有天数
  confidence_weights:
    rsi_oversold_threshold: 30
    final_threshold: 0.5
  
optimization_ranges:
  rise_threshold: [0.02, 0.08]
  max_days: [10, 30]
```

### 系统配置 (config/system.yaml)

系统级别的配置参数：

```yaml
data:
  symbol: "000905.SH"        # 中证500指数代码
  start_date: "2020-01-01"
  
ai_optimization:
  population_size: 70
  generations: 20
```

## 📈 使用示例

### 完整的优化流程

```bash
# 1. 重置参数（可选）
python reset_strategy_params.py --all --force

# 2. 运行AI优化
python run.py ai

# 3. 查看优化结果
python run.py config

# 4. 运行回测验证
python run.py backtest
```

### 单日预测

```bash
# 预测指定日期
python run.py predict --date 2024-01-15

# 预测最新交易日
python run.py predict
```

### 滚动回测

```bash
# 基本滚动回测
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31

# 带详细输出的回测
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31 --verbose

# 自定义训练窗口和重训练间隔
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31 --training_window_days 252 --retrain_interval_days 30

# 复用现有模型（不重新训练）
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31 --reuse_model

# 指定报告输出目录
python examples/run_rolling_backtest.py --start_date 2025-02-01 --end_date 2025-08-31 --report_dir results/custom_backtest
```

## 📊 性能指标

系统评估指标包括：

- **成功率**：识别正确的相对低点比例
- **平均涨幅**：成功识别后的平均收益
- **策略得分**：综合评分指标
- **夏普比率**：风险调整后收益
- **最大回撤**：最大损失幅度

## 📋 输出报告

### 滚动回测报告

系统会生成两类报告文件：

1. **回测报告** (`backtest_report_YYYYMMDD_HHMMSS.md`)
   - 整体性能指标
   - 成功率统计
   - 平均收益分析
   - 详细的交易信号记录

2. **每日详情CSV** (`daily_details_rolling_backtest_YYYYMMDD_HHMMSS.csv`)
   - 包含每个交易日的详细信息
   - **趋势状态** (`trend_regime`)：bull（牛市）、bear（熊市）、sideways（横盘）
   - 预测价格、置信度、实际涨跌幅
   - 技术指标数值

### 趋势状态分析

系统会自动识别市场趋势状态：
- **bull（牛市）**：上升趋势，适合持有策略
- **bear（熊市）**：下降趋势，谨慎操作
- **sideways（横盘）**：震荡行情，短线操作

## 🛠️ 高级功能

### 参数重置

```bash
# 重置所有参数
python reset_strategy_params.py --all

# 仅重置策略参数
python reset_strategy_params.py --strategy

# 仅重置置信度权重
python reset_strategy_params.py --confidence
```

### 模型管理

- 自动保存优化后的模型
- 模型版本管理
- 模型性能对比

## 🔍 故障排除

### 常见问题

1. **模型文件不存在**
   - 检查 `models/latest_improved_model.txt` 文件
   - 重新运行AI优化生成新模型

2. **配置文件错误**
   - 使用重置脚本恢复默认配置
   - 检查YAML文件格式

3. **数据获取失败**
   - 检查网络连接
   - 确认数据源可用性

4. **滚动回测无输出**
   - 确认日期范围在数据文件覆盖范围内
   - 使用 `--verbose` 参数查看详细输出
   - 检查数据文件是否存在且格式正确

5. **CSV报告缺少趋势状态**
   - 确保使用最新版本的滚动回测脚本
   - 检查 `strategy_indicators` 列是否包含趋势信息

### 日志查看

系统日志保存在运行目录，包含详细的执行信息和错误诊断。

## 📝 开发说明

### 代码规范

- 使用Python类型提示
- 遵循PEP 8代码风格
- 完整的文档字符串
- 单元测试覆盖

### 扩展开发

- 新增技术指标：在 `src/strategy/` 目录添加
- 优化算法：在 `src/ai/` 目录扩展
- 数据源：在 `src/data/` 目录集成

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📞 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**免责声明**：本系统仅供学习和研究使用，不构成投资建议。投资有风险，决策需谨慎。


