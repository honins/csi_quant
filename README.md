# 中证1000指数相对低点识别量化系统

一个基于Python的量化交易系统，旨在识别中证1000指数的相对低点，支持AI优化、LLM驱动策略、可视化回测和多种通知方式。

## 功能特点

- 🎯 **相对低点识别**：基于技术指标和自定义规则识别相对低点
- 🤖 **AI自优化**：支持机器学习、遗传算法等多种AI优化方法
- 🧠 **LLM驱动策略优化**：自动调整策略参数，提升策略表现
- 📊 **全面回测与可视化**：支持滚动回测、单日预测、结果表格美化和多种图表输出
- 📧 **实时通知**：支持控制台和邮件通知
- 🔧 **模块化设计**：便于扩展和维护

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

### 4. 典型用法

- **基础回测**
  ```bash
  python run.py basic
  ```
- **AI优化回测**
  ```bash
  python run.py ai
  ```
- **滚动回测（含表格美化）**
  ```bash
  python examples/run_rolling_backtest.py 2023-01-01 2023-06-30
  # 结果表格Prediction Details自动高亮Yes/No
  ```
- **单日低点预测**
  ```bash
  python predict_single_day.py 2024-06-01
  ```
- **LLM驱动策略优化**
  ```bash
  python llm_strategy_optimizer.py
  ```
- **运行所有测试**
  ```bash
  python run.py all
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
├── run.py                 # 快速入口
├── predict_single_day.py  # 单日预测
├── llm_strategy_optimizer.py # LLM策略优化
└── README.md              # 项目说明
```

## 配置说明

主要配置文件：`config/config.yaml`
- 策略参数：`strategy.rise_threshold`（涨幅阈值）、`strategy.max_days`（最大交易日数）
- AI参数：`ai.model_type`、`ai.enable`
- 通知参数：`notification.methods`、`notification.email`
- 回测参数：`backtest.default_start_date`、`backtest.default_end_date`

## 结果解读

- **Prediction Details表格**：所有"Yes"单元格为淡绿，"No"为淡红，直观展示预测正确性
- **回测图表**：自动保存到results目录，含预测点、实际点、成功率等
- **日志**：详细记录每一步执行与异常，便于排查

## 常见问题

- **Q: 如何只训练一次模型？**
  A: 默认每个预测日都重新训练，保证回测严谨。若需只训练一次，可在examples/run_rolling_backtest.py中调整训练逻辑。
- **Q: 如何自定义表格配色？**
  A: 修改run_rolling_backtest.py中table.get_celld()的set_facecolor部分。
- **Q: 如何集成真实数据？**
  A: 修改src/data/data_module.py的数据获取逻辑。
- **Q: 如何配置邮件通知？**
  A: 配置config/config.yaml的notification.email，并在notification_module.py中启用邮件功能。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 更新日志

详见 `CHANGELOG.md`。


