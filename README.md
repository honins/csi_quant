# 中证500指数量化交易系统

## 📋 项目简介

这是一个基于中证500指数的**全链路智能量化交易系统**。

不同于传统的单一指标策略，本系统采用**“AI预测 + 动态风控”**的双轮驱动模式。通过 Feature Engineering V2 构建日线与周线的多周期特征，利用机器学习模型识别高胜率买点，并结合 **ATR (平均真实波幅)** 进行动态仓位管理和移动止损。

系统的核心目标是：**在单边牛市中通过激进策略最大化收益，在震荡与下跌市中通过波动率风控严格控制回撤。**

## 🏗️ 系统架构

本系统采用模块化的分层架构设计，确保各个组件的高效协作：

### 1. 数据与特征层 (Data & Alpha)
*   **多周期特征工程 (Feature Engineer V2)**：不仅关注日线，更引入**周线 (Weekly)** 级别的趋势特征，赋予 AI 宏观视野。
*   **高级指标集成**：计算 ATR (波动率)、ADX (趋势强度)、OBV (资金流向)、NATR (归一化波动率) 等深度因子。

### 2. 模型与预测层 (Model & Prediction)
*   **趋势识别模型**：基于 Random Forest 识别当前市场状态（牛/熊/震荡），自动调整策略偏好。
*   **收益预测模型**：预测未来 10-20 天的风险调整后收益，输出 0~1 的置信度评分。

### 3. 策略与决策层 (Strategy & Decision)
*   **动态阈值机制**：根据市场情绪自动调整买入阈值（如牛市降低至 0.25 以增加交易频率，熊市提高至 0.50 以过滤噪音）。
*   **信号共振**：结合 RSI 超卖、均线支撑与 AI 评分进行多因子验证。

### 4. 交易与风控层 (Execution & Risk)
*   **ATR 动态风控**：实现 **Chandelier Exit (吊灯止损)**，止损线随价格上涨动态上移，跌破即离场，拒绝回撤。
*   **双模式资金管理**：
    *   🛡️ **稳健模式**：基于波动率倒数的仓位管理，风险可控。
    *   🚀 **激进模式**：全仓复利策略，专为牛市打造，最大化资金利用率。

## ✨ 核心特性

- 🧠 **AI 驱动决策**：使用遗传算法和机器学习自动寻找最佳交易参数。
- 🛡️ **ATR 动态止损**：告别固定比例止损，根据市场波动率动态调整离场点，更能“拿住”利润。
- 📊 **多周期共振**：日线入场，周线定势，极大提高了信号的胜率。
- 🎯 **灵活的交易模式**：支持一键切换“稳健”与“激进”模式，适应不同风险偏好的投资者。
- 🔧 **参数自适应**：系统会根据回测结果自动推荐最优的置信度阈值。
- 📈 **全方位回测**：支持滚动回测、多周期对比（1/3/6/12个月），生成详细的胜率与盈亏报告。

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


