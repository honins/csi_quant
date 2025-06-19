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

## 快速开始

### 1. 创建并激活虚拟环境

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置系统

编辑 `config/config.yaml`，设置AI、策略、回测、通知等参数。

### 4. 简化命令使用

#### 基础命令
```bash
# 基础测试
python run.py b

# AI优化测试  
python run.py a

# 单元测试
python run.py t

# 策略优化 (默认10次迭代)
python run.py opt

# 策略优化 (自定义迭代次数)
python run.py opt -i 20
```

#### 日期相关命令
```bash
# 单日预测
python run.py s 2023-12-01

# 回测 (需要开始和结束日期)
python run.py r 2023-01-01 2023-12-31
```

#### 完整运行
```bash
# 运行所有测试 (不包含日期相关功能)
python run.py all

# 运行所有测试 (包含回测和单日预测)
python run.py all 2023-01-01 2023-12-31
```

### 5. 命令对照表

| 简化命令 | 原命令 | 功能 |
|---------|--------|------|
| `b` | `basic` | 基础测试 |
| `a` | `ai` | AI优化测试 |
| `t` | `test` | 单元测试 |
| `r` | `rolling` | 回测 |
| `s` | `single` | 单日预测 |
| `opt` | `strategy` | 策略优化 |
| `all` | `all` | 全部运行 |

### 6. 高级用法

- **滚动回测（含表格美化）**
  ```bash
  python run.py r 2023-01-01 2023-06-30
  # 结果表格Prediction Details自动高亮Yes/No，Confidence字段紧跟在Predicted后面
  ```
- **单日低点预测**
  ```bash
  python run.py s 2024-06-01
  ```
- **LLM驱动策略优化**
  ```bash
  python run.py opt -i 50
  ```
- **运行所有测试**
  ```bash
  python run.py all 2023-01-01 2023-12-31
  ```

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


