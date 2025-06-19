# 中证1000指数相对低点识别量化系统

一个基于Python的量化交易系统，旨在识别中证1000指数的相对低点，支持AI优化、LLM驱动策略、可视化回测和多种通知方式。

## 功能特点

- 🎯 **相对低点识别**：基于技术指标和自定义规则识别相对低点
- 🤖 **AI自优化**：支持机器学习、遗传算法等多种AI优化方法
- 🧠 **LLM驱动策略优化**：自动调整策略参数，提升策略表现
- 📊 **全面回测与可视化**：支持滚动回测、单日预测、结果表格美化和多种图表输出
- 📧 **实时通知**：支持控制台和邮件通知
- �� **模块化设计**：便于扩展和维护
- ⚡ **简化命令**：提供简洁的命令行界面，快速运行各种功能

## 环境要求

- Python 3.8+
- 推荐使用虚拟环境

## 🏁 快速开始

1. 安装依赖（建议使用虚拟环境）

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. 配置参数

编辑 `config/config.yaml`，可自定义数据、策略、AI、优化等参数。

3. 运行基础与AI优化

```bash
# 基础策略测试
python run.py b

# AI优化（推荐，自动分层优化+参数持久化）
python run.py ai

# 生成图表
python run.py ai plot
```

4. 运行示例脚本

```bash
# 高级优化演示
python examples/advanced_optimization_demo.py
# 完整AI优化测试
python examples/ai_optimization_test.py
# 滚动回测
python examples/run_rolling_backtest.py
```

5. 验证优化效果

```bash
# 查看优化后的参数
python examples/predict_single_day.py 2024-06-03
# 输出：策略模块初始化完成，参数: rise_threshold=0.0300, max_days=30
```

## ⚙️ 主要命令说明

| 命令         | 说明                       |
|--------------|----------------------------|
| b            | 基础策略测试               |
| a            | AI测试（含训练与预测）     |
| ai           | 高级AI优化（分层优化+参数持久化） |
| r            | 回测                       |
| s            | 单日预测                   |
| opt          | 策略参数优化               |
| all          | 全部测试                   |

**ai命令特点**：
- 🏗️ 自动分层优化
- 💾 参数自动持久化到配置文件
- 🌐 全局生效，所有脚本自动使用优化参数
- 📊 显示优化效果对比

## 🧠 分层优化与高级AI优化

- 避免循环依赖，先用技术指标识别低点，再用未来涨幅验证
- 多目标评估（成功率、涨幅、速度、风险）
- 时间序列交叉验证，防止未来数据泄漏
- 支持遗传算法、scipy数值优化
- **参数自动持久化**：优化后的参数自动保存到配置文件，全局生效
- 配置项详见 `config/config.yaml`，可灵活开关各优化模块

## 📝 配置文件说明

- `data`：数据源、频率、缓存等
- `strategy`：涨幅阈值、最大天数、技术指标参数
- `ai`：模型类型、优化参数、遗传算法、分层优化开关
- `backtest`：回测窗口、步长、起止日期
- `logging`/`results`/`notification`：日志、结果、通知等

## ❓ 常见问题

- **Q: 如何只用AI预测，不重新训练？**
  A: 只要模型已保存，`predict_single_day.py`会自动加载，无需重复训练。

- **Q: 优化后参数如何应用到其他脚本？**
  A: `run.py ai`会自动将优化后的参数保存到配置文件，所有脚本都会自动使用。

- **Q: 如何切换优化方式？**
  A: 修改`config.yaml`中`ai.advanced_optimization`相关开关。

- **Q: 依赖缺失怎么办？**
  A: 激活虚拟环境并`pip install -r requirements.txt`。

- **Q: 优化效果如何？**
  A: 分层优化可提升成功率77.7%，平均涨幅39.9%，综合得分33.0%。

- **Q: 参数会丢失吗？**
  A: 不会，优化后的参数会自动保存到配置文件，重启后仍然有效。

---

如需详细字段、图表、评估体系说明，请参见下文"生成图片字段说明"及代码注释。

## 项目结构

```
csi1000_quant/
├── src/                    # 源代码
│   ├── data/              # 数据获取与处理
│   ├── strategy/          # 策略与回测
│   ├── ai/                # AI优化
│   ├── notification/      # 通知
│   └── utils/             # 工具
├── examples/              # 典型用法脚本
├── config/                # 配置文件
├── data/                  # 历史数据
├── results/               # 回测与预测结果
├── logs/                  # 日志
├── models/                # AI模型
├── requirements.txt       # 依赖
├── run.py                 # 快速入口（简化命令）
├── predict_single_day.py  # 单日预测
├── llm_strategy_optimizer.py # LLM策略优化
├── QUICKSTART.md          # 快速开始指南
└── README.md              # 项目说明
```

## 配置说明

主要配置文件：`config/config.yaml`
- 策略参数：`strategy.rise_threshold`（涨幅阈值）、`strategy.max_days`（最大交易日数）
- AI参数：`ai.model_type`、`ai.enable`
- 通知参数：`notification.methods`、`notification.email`
- 回测参数：`backtest.default_start_date`、`backtest.default_end_date`

## 结果解读

### 生成图片字段详解

#### 1. Prediction Details 表格字段

系统会生成一个详细的预测结果表格，包含以下字段：

| 字段名 | 英文名 | 说明 | 颜色标识 |
|--------|--------|------|----------|
| **Date** | Date | 预测日期 | 白色 |
| **Predict Price** | Predict Price | 预测当日的收盘价格 | 淡蓝色 (#e3f2fd) |
| **Predicted** | Predicted | AI预测是否为相对低点<br/>• Yes: 预测为相对低点<br/>• No: 预测非相对低点 | 淡黄色 (#fff9c4)<br/>Yes: 淡绿色<br/>No: 淡红色 |
| **Confidence** | Confidence | AI预测的置信度<br/>• 范围: 0.00-1.00<br/>• 表示AI认为该日期是相对低点的概率<br/>• 0.00 = 100%确定不是相对低点<br/>• 1.00 = 100%确定是相对低点<br/>• 0.50 = 50%的概率是相对低点 | 淡紫色 (#ede7f6) |
| **Actual** | Actual | 实际是否为相对低点<br/>• Yes: 实际为相对低点<br/>• No: 实际非相对低点<br/>• Insufficient Data: 数据不足无法验证 | 淡橙色 (#ffe0b2)<br/>Yes: 淡绿色<br/>No: 淡红色 |
| **Max Future Rise** | Max Future Rise | 未来最大涨幅<br/>• 格式: 百分比 (如 5.23%)<br/>• 相对于预测日价格的最大涨幅 | 淡绿色 (#e8f5e9) |
| **Days to Target Rise** | Days to Target Rise | 达到目标涨幅所需天数<br/>• 整数天数<br/>• 从预测日开始计算 | 淡灰色 (#f5f5f5) |
| **Prediction Correct** | Prediction Correct | 预测是否正确<br/>• Yes: 预测正确<br/>• No: 预测错误<br/>• Insufficient Data: 数据不足无法验证 | 白色<br/>Yes: 淡绿色<br/>No: 淡红色 |

#### 2. 回测结果图表

系统会生成多个图表文件：

##### 2.1 滚动回测结果图 (`rolling_backtest_results_*.png`)
- **预测点标记**: 蓝色圆点表示AI预测的相对低点
- **实际点标记**: 红色叉号表示实际的相对低点
- **正确预测**: 绿色圆点表示预测正确的点
- **错误预测**: 红色圆点表示预测错误的点
- **成功率统计**: 图表底部显示总体预测成功率

##### 2.2 回测分析图 (`backtest_analysis_*.png`)
包含4个子图：
- **价格走势与相对低点**: 显示价格曲线和识别出的相对低点
- **涨幅分布**: 相对低点后的最大涨幅分布直方图
- **天数分布**: 达到目标涨幅所需天数分布
- **策略评估指标**: 成功率、平均涨幅、平均天数、综合得分的柱状图

#### 3. 颜色编码说明

- **淡绿色 (#e8f5e9)**: 正面结果 (Yes, 正确预测)
- **淡红色 (#ffebee)**: 负面结果 (No, 错误预测)
- **淡蓝色 (#e3f2fd)**: 价格相关数据
- **淡黄色 (#fff9c4)**: 预测结果
- **淡紫色 (#ede7f6)**: 置信度数据
- **淡橙色 (#ffe0b2)**: 实际结果
- **淡绿色 (#e8f5e9)**: 涨幅数据
- **淡灰色 (#f5f5f5)**: 天数数据

#### 4. 数据验证说明

- **可验证数据**: 有足够的历史数据来验证预测结果
- **数据不足**: 显示 "Insufficient Data"，表示无法验证预测结果
- **成功率计算**: 仅基于可验证的数据计算预测成功率

#### 5. Confidence值详细解释

**Confidence（置信度）** 是AI模型预测的核心指标，表示模型认为某个日期是相对低点的概率：

- **数值含义**：
  - `0.00` = 0% 概率是相对低点（100%确定不是相对低点）
  - `0.25` = 25% 概率是相对低点
  - `0.50` = 50% 概率是相对低点（模型最不确定）
  - `0.75` = 75% 概率是相对低点
  - `1.00` = 100% 概率是相对低点（100%确定是相对低点）

- **技术原理**：
  - 基于机器学习模型的 `predict_proba()` 方法
  - 使用 `prediction_proba[1]` 获取正类（相对低点）的概率
  - 模型通过历史数据训练，学习识别相对低点的特征模式

- **使用建议**：
  - **高置信度 (>0.8)**: 模型非常确信，建议重点关注
  - **中等置信度 (0.5-0.8)**: 模型有一定把握，需要结合其他指标
  - **低置信度 (<0.5)**: 模型不确定，建议谨慎对待

- **与Predicted字段的关系**：
  - 当 Confidence ≥ 0.5 时，Predicted 显示 "Yes"
  - 当 Confidence < 0.5 时，Predicted 显示 "No"
  - Confidence 提供了比简单 Yes/No 更细粒度的信息

### 结果文件位置

所有生成的图片文件保存在 `results/` 目录下：
- `prediction_details_YYYYMMDD_HHMMSS.png`: 预测详情表格
- `rolling_backtest_results_YYYYMMDD_HHMMSS.png`: 滚动回测结果图
- `backtest_analysis_YYYYMMDD_HHMMSS.png`: 回测分析图

## 最新更新

### v2.1.0 (最新)
- ✨ **简化命令系统**：将长命令简化为单字母命令，提高使用效率
- 🔄 **表格列顺序优化**：将Confidence字段移到Predicted后面，更符合逻辑顺序
- 🐛 **修复pandas警告**：解决FutureWarning，使用推荐的新方法
- 📚 **完善文档**：更新QUICKSTART.md，提供详细的使用指南

### v2.0.0
- 🎨 **表格美化**：Prediction Details表格自动高亮Yes/No
- 📊 **增强可视化**：多种图表输出，结果更直观
- 🤖 **AI优化**：支持多种机器学习算法
- 📧 **通知系统**：支持邮件和控制台通知

## 常见问题

- **Q: 如何只训练一次模型？**
  A: 默认每个预测日都重新训练，保证回测严谨。若需只训练一次，可在examples/run_rolling_backtest.py中调整训练逻辑。
- **Q: 如何自定义表格配色？**
  A: 修改run_rolling_backtest.py中table.get_celld()的set_facecolor部分。
- **Q: 如何集成真实数据？**
  A: 修改src/data/data_module.py的数据获取逻辑。
- **Q: 如何配置邮件通知？**
  A: 配置config/config.yaml的notification.email，并在notification_module.py中启用邮件功能。
- **Q: 简化命令支持哪些参数？**
  A: 支持 `-v` (详细输出)、`-i` (迭代次数)、位置参数(日期)等，详见QUICKSTART.md。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 更新日志

详见 `CHANGELOG.md`。

## 🚀 高级优化功能

### 分层优化策略

系统实现了分层优化策略，避免循环依赖问题：

1. **第一层：策略参数优化**
   - 使用基准策略生成固定标签
   - 基于固定标签优化策略参数
   - 避免策略优化和AI训练的循环依赖

2. **第二层：AI模型训练**
   - 基于优化后的策略训练AI模型
   - 使用样本权重平衡正负样本

3. **第三层：时间序列交叉验证**
   - 按时间顺序分割数据
   - 在训练集上优化，在测试集上验证
   - 避免未来信息泄露

4. **第四层：高级优化**
   - 使用scipy进行数值优化
   - 多目标优化考虑成功率、涨幅、速度、风险

### 参数持久化与全局生效

✅ **一次优化，全局生效**：
- 优化后的参数自动保存到 `config/config.yaml`
- 所有脚本（如 `predict_single_day.py`）自动使用优化后的参数
- 无需手动更新每个脚本的参数

✅ **参数持久化示例**：
```bash
# 运行AI优化
python run.py ai
# 输出：✅ 参数已保存: {'rise_threshold': 0.03, 'max_days': 30}

# 其他脚本自动使用优化后的参数
python examples/predict_single_day.py 2024-06-03
# 输出：策略模块初始化完成，参数: rise_threshold=0.0300, max_days=30
```

### 评估指标改进

新的评估体系包含多个维度：

- **成功率权重 (40%)**: 是否达到目标涨幅
- **涨幅权重 (30%)**: 相对于10%的基准的实际涨幅
- **速度权重 (20%)**: 达到目标涨幅所需天数
- **风险调整 (10%)**: 避免过度冒险的惩罚机制

### 使用方法

```bash
# 一键AI优化（推荐）
python run.py ai

# 运行高级优化演示
python examples/advanced_optimization_demo.py

# 运行完整AI优化测试
python examples/ai_optimization_test.py
```

### 配置选项

在 `config/config.yaml` 中可以配置高级优化选项：

```yaml
ai:
  advanced_optimization:
    enabled: true
    use_scipy: true
    use_hierarchical: true
    use_time_series_cv: true
  genetic_algorithm:
    population_size: 20
    generations: 10
    mutation_rate: 0.1
    crossover_rate: 0.8
```

### 优化效果对比

| 优化方法 | 成功率提升 | 平均涨幅 | 综合得分 |
|---------|-----------|----------|----------|
| 基准策略 | 25.75% | 4.11% | 0.4520 |
| 分层优化 | 45.75% | 5.75% | 0.6014 |
| 改进幅度 | +77.7% | +39.9% | +33.0% |

## 📊 生成图片字段说明


