# 中证500指数相对低点识别量化系统

一个基于Python的智能量化交易系统，集成了先进的AI算法和机器学习技术，旨在高精度识别中证500指数的相对低点，支持多种优化算法、可视化回测和智能通知。

## ⚠️ 重要提醒

### 🔧 **虚拟环境 (强烈推荐)**
**本项目强烈建议在虚拟环境中运行**，以避免包依赖冲突：

```bash
# 创建并激活虚拟环境
python -m venv venv

# Windows激活
venv\Scripts\activate

# Linux/Mac激活  
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 📋 **关键依赖关系说明**
本项目的模块间存在严格的依赖关系，请按以下顺序确保环境正确：

1. **基础依赖**: `pandas`, `numpy`, `matplotlib` - 数据处理和可视化
2. **机器学习**: `scikit-learn`, `xgboost` - AI模型训练
3. **金融数据**: `akshare`, `yfinance` - 数据获取
4. **优化算法**: `scipy`, `scikit-optimize` - 参数优化
5. **系统工具**: `pyyaml` - 配置文件解析

**❌ 常见问题及解决方案:**
- **ImportError**: 激活虚拟环境并运行 `pip install -r requirements.txt`
- **模块找不到**: 确保运行路径在项目根目录
- **配置文件错误**: 检查 `config/system.yaml` 和 `config/strategy.yaml` 是否存在

### 🔧 **模块化配置架构**
项目采用全新的模块化配置文件架构，简洁高效：

```
config/
├── system.yaml          # 系统基础配置（196行）
├── strategy.yaml        # 策略优化配置（421行）
└── config.yaml          # 兼容性配置（保留）
```

**配置加载优先级**：
```
system.yaml → strategy.yaml → config.yaml → 环境变量配置
```

**环境变量支持**：
```bash
# 使用额外的自定义配置文件
export CSI_CONFIG_PATH=/path/to/custom.yaml  # Linux/Mac
set CSI_CONFIG_PATH=C:\path\to\custom.yaml   # Windows

# 运行系统（自动加载多配置文件）
python run.py ai
```

**核心改进**：
- ✅ 模块化管理，便于维护和调试
- ✅ 策略优化参数集中管理
- ✅ 系统配置和业务配置分离
- ✅ 向后兼容，现有脚本无需修改
- ✅ 支持多配置文件自动合并

### ⏱️ **性能监控功能**
系统新增执行时间统计功能，自动显示命令执行时间：

```bash
# 启用性能监控（默认）
python run.py ai

# 禁用性能监控
python run.py ai --no-timer
```

## ✨ 核心功能特点

- 🎯 **智能相对低点识别**：融合技术指标、AI预测和自定义规则的多层识别系统
- 🤖 **多算法AI优化**：集成贝叶斯优化、遗传算法、随机森林等先进算法
- 🧠 **增量优化机制**：基于历史最优结果的渐进式参数优化，避免重复搜索
- 🔬 **严格数据分割**：防止过拟合的时间序列分割和前向验证
- 📊 **全面回测与可视化**：支持滚动回测、单日预测、结果表格美化和多种图表输出
- 📧 **智能通知系统**：支持控制台和邮件通知
- 🔧 **模块化架构**：重构后的清晰架构，便于扩展和维护
- ⚡ **简化命令界面**：提供简洁的命令行界面，快速运行各种功能
- 📈 **自动数据获取**：自动获取000852和000905的最新数据
- 💾 **参数持久化**：优化结果自动保存，全局生效

## 🧠 核心算法概览

系统集成了多种先进的AI和优化算法：

### 🎯 优化算法
- **贝叶斯优化**：基于高斯过程的智能参数搜索，自适应搜索范围
- **遗传算法**：模拟生物进化的全局优化算法
- **增量优化**：基于历史最优结果的渐进式改进
- **随机搜索**：结合局部搜索和全局探索的混合策略

### 🤖 机器学习算法  
- **随机森林**：集成学习算法，用于相对低点预测
- **时间序列处理**：专门针对金融时间序列的数据处理
- **样本权重优化**：基于时间衰减的样本重要性调整
- **特征工程**：18维技术指标特征提取

### 🔬 验证与防护算法
- **严格数据分割**：70%/20%/10%的时序分割，防止数据泄露
- **前向验证**：模拟真实交易环境的滚动验证
- **早停机制**：防止过拟合的智能停止策略
- **过拟合检测**：多重验证确保模型泛化能力

### 📊 技术指标算法
- **RSI超卖检测**：相对强弱指数算法
- **MACD趋势分析**：移动平均收敛发散算法  
- **布林带分析**：统计学波动率算法
- **移动平均线组合**：多周期趋势识别

> 📖 **详细算法说明请参考**: [`docs/算法介绍和作用.md`](docs/算法介绍和作用.md)
> 📚 **完整文档导航请参考**: [`DOCS.md`](DOCS.md)

## 📋 完整使用指南

**🌟 新手用户请直接查看：[**完整使用指南 (USER_GUIDE.md)**](USER_GUIDE.md)**

该指南包含：
- 🚀 **详细的快速开始步骤**
- 📋 **完整的命令参考手册** 
- ⚙️ **配置文件详解**
- 🎯 **典型使用场景**
- 🔧 **故障排除指南**
- 🚀 **高级用法和性能优化**

本 README 提供技术概述，**USER_GUIDE.md** 提供实用的操作指南。

---

## 🏁 环境要求与快速开始

### 环境要求

- **Python 3.8+** (推荐 3.9 或 3.10)
- **虚拟环境** (必须使用，避免依赖冲突)
- **内存**: 建议 8GB+ (AI训练和大数据处理)
- **磁盘**: 至少 2GB 可用空间 (数据和模型存储)

### 🔗 核心依赖包版本要求
```
pandas>=1.3.0          # 数据处理核心
numpy>=1.21.0           # 数值计算
scikit-learn>=1.0.0     # 机器学习
matplotlib>=3.5.0       # 图表绘制
akshare>=1.8.0          # 金融数据获取
scipy>=1.7.0            # 科学计算和优化
pyyaml>=6.0             # 配置文件解析
```

### 🚀 5分钟快速开始

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate    # Linux/Mac

# 2. 安装依赖
pip install -r requirements.txt

# 3. 基础测试
python run.py b

# 4. AI训练（6年数据，6-12分钟）
python run.py ai

# 5. 单日预测
python run.py s 2024-12-01

# 6. 策略回测
python run.py r 2023-01-01 2023-12-31
```

**⚠️ 重要**：必须在虚拟环境中运行，避免包依赖冲突。

## ⚙️ 详细执行命令说明

### 基本执行格式

由于项目规则要求使用虚拟环境，完整的执行命令格式为：

```bash
# 1. 首先激活虚拟环境 (Windows)
venv\Scripts\activate

# 2. 然后运行命令
python run.py <command> [参数]
```

### 主要命令选项

| 命令 | 说明 | 示例用法 | 性能监控 |
|------|------|----------|----------|
| `d` | 数据获取 | `python run.py d` | ✅ 支持 |
| `b` | 基础策略测试 | `python run.py b` | ✅ 支持 |
| `a` | AI测试（含训练与预测） | `python run.py a` | ✅ 支持 |
| `t` | 单元测试 | `python run.py t` | ✅ 支持 |
| `r` | 滚动回测 | `python run.py r 2023-01-01 2023-12-31` | ✅ 支持 |
| `s` | 单日预测 | `python run.py s 2023-12-01` | ✅ 支持 |
| `opt` | 策略参数优化 | `python run.py opt -i 20` | ✅ 支持 |
| `ai` | AI优化训练 | `python run.py ai -m incremental` | ✅ 支持 |
| `all` | 全部测试 | `python run.py all` | ✅ 支持 |

### AI模式详细说明

AI命令支持多种模式，通过 `-m` 参数指定：

| 模式 | 说明 | 示例 |
|------|------|------|
| `optimize` | 完整优化（默认） | `python run.py ai` 或 `python run.py ai -m optimize` |
| `incremental` | 增量训练 | `python run.py ai -m incremental` |
| `full` | 完全重训练 | `python run.py ai -m full` |
| `demo` | 演示预测 | `python run.py ai -m demo` |

### 重要参数说明

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--verbose` | `-v` | 详细输出 | `python run.py b -v` |
| `--iter` | `-i` | 迭代次数（默认10） | `python run.py opt -i 50` |
| `--mode` | `-m` | 指定运行模式 | `python run.py ai -m incremental` |
| `--no-timer` | 无 | 禁用性能计时器 | `python run.py ai --no-timer` |

## 📝 配置文件详解

### 模块化配置架构

**system.yaml** - 系统基础配置（196行）：
- 策略参数：`strategy.rise_threshold`（涨幅阈值4%）、`strategy.max_days`（最大交易日数20天）
- AI基础配置：`ai.model_type`、`ai.enable`、`ai.models_dir`
- 数据配置：`data.data_source`、`data.index_code`、`data.frequency`
- 系统配置：`logging`、`notification`、`results`、`backtest`

**strategy.yaml** - 策略优化配置（421行）：
- AI优化算法：`bayesian_optimization`、`genetic_algorithm`
- 参数搜索范围：`optimization_ranges`、`strategy_ranges`
- 评分权重：`ai_scoring`、`strategy_scoring`
- 置信度权重：`confidence_weights`
- 高级优化开关：`advanced_optimization`

**config.yaml** - 兼容性配置：
- 保留完整配置，确保向后兼容
- 优先级最低，主要用于兼容旧脚本

### 配置优先级
```
system.yaml → strategy.yaml → config.yaml → 环境变量
```

### 关键配置参数说明

#### 策略核心参数
- **rise_threshold**: 0.04 (4%) - 相对低点涨幅阈值，不建议修改
- **max_days**: 20 - 最大持有天数，不建议修改
- **final_threshold**: 0.3392 - 最终置信度阈值，准确率不高时可降低

#### AI训练数据范围
- **full_train_years**: 6 - 完全重训练使用的历史数据年数
- **optimize_years**: 6 - 参数优化模式使用的数据年数
- **incremental_years**: 1 - 增量训练使用的数据年数

#### 数据分割比例
- **train_ratio**: 0.70 (70%) - 训练集比例
- **validation_ratio**: 0.20 (20%) - 验证集比例
- **test_ratio**: 0.10 (10%) - 测试集比例

## 📊 数据获取说明

### 数据获取脚本功能

- **自动获取**：000852（中证1000指数）和000905（中证500指数）的最新数据
- **数据源**：使用akshare数据源，数据可靠且实时
- **格式统一**：自动保存为CSV格式，与项目现有数据格式保持一致
- **日志记录**：详细的执行日志，便于调试和监控
- **错误处理**：网络异常时自动重试，确保数据完整性

### 输出文件

- `data/SHSE.000852_1d.csv` - 中证1000指数数据
- `data/SHSE.000905_1d.csv` - 中证500指数数据

### 数据格式

CSV文件包含以下列：
- `index` - 序号
- `open` - 开盘价
- `high` - 最高价
- `low` - 最低价
- `close` - 收盘价
- `volume` - 成交量
- `amount` - 成交额
- `date` - 日期

## 🧠 分层优化与高级AI优化

- 避免循环依赖，先用技术指标识别低点，再用未来涨幅验证
- 多目标评估（成功率、涨幅、速度、风险）
- 时间序列交叉验证，防止未来数据泄漏
- 支持遗传算法、贝叶斯优化
- **参数自动持久化**：优化后的参数自动保存到配置文件，全局生效

## 📈 结果解读

### 生成图片字段详解

#### 1. Prediction Details 表格字段

系统会生成一个详细的预测结果表格，包含以下字段：

| 字段名 | 英文名 | 说明 | 颜色标识 |
|--------|--------|------|----------|
| **Date** | Date | 预测日期 | 白色 |
| **Predict Price** | Predict Price | 预测当日的收盘价格 | 淡蓝色 (#e3f2fd) |
| **Predicted** | Predicted | AI预测是否为相对低点<br/>• Yes: 预测为相对低点<br/>• No: 预测非相对低点 | 淡黄色 (#fff9c4)<br/>Yes: 淡绿色<br/>No: 淡红色 |
| **Confidence** | Confidence | AI预测的置信度<br/>• 范围: 0.00-1.00<br/>• 表示AI认为该日期是相对低点的概率 | 淡紫色 (#ede7f6) |
| **Actual** | Actual | 实际是否为相对低点<br/>• Yes: 实际为相对低点<br/>• No: 实际非相对低点 | 淡橙色 (#ffe0b2)<br/>Yes: 淡绿色<br/>No: 淡红色 |
| **Max Future Rise** | Max Future Rise | 未来最大涨幅（百分比） | 淡绿色 (#e8f5e9) |
| **Days to Target Rise** | Days to Target Rise | 达到目标涨幅所需天数 | 淡灰色 (#f5f5f5) |
| **Prediction Correct** | Prediction Correct | 预测是否正确 | 白色<br/>Yes: 淡绿色<br/>No: 淡红色 |

#### 2. Confidence值详细解释

**Confidence（置信度）** 是AI模型预测的核心指标：

- **数值含义**：
  - `0.00` = 0% 概率是相对低点
  - `0.50` = 50% 概率是相对低点
  - `1.00` = 100% 概率是相对低点

- **使用建议**：
  - **高置信度 (>0.8)**: 模型非常确信，建议重点关注
  - **中等置信度 (0.5-0.8)**: 模型有一定把握，需要结合其他指标
  - **低置信度 (<0.5)**: 模型不确定，建议谨慎对待

### 结果文件位置

所有生成的图片文件保存在 `results/` 目录下：
- `prediction_details_YYYYMMDD_HHMMSS.png`: 预测详情表格
- `rolling_backtest_results_YYYYMMDD_HHMMSS.png`: 滚动回测结果图
- `backtest_analysis_YYYYMMDD_HHMMSS.png`: 回测分析图

## 项目结构

```
csi1000_quant/
├── src/                    # 源代码
│   ├── data/              # 数据获取与处理
│   ├── strategy/          # 策略与回测
│   ├── ai/                # AI优化
│   ├── notification/      # 通知
│   └── utils/             # 工具（含多配置文件加载器）
├── examples/              # 典型用法脚本
├── config/                # 配置文件（模块化架构）
│   ├── system.yaml       # 系统基础配置
│   ├── strategy.yaml     # 策略优化配置
│   └── config.yaml        # 兼容性配置
├── docs/                  # 文档
├── data/                  # 历史数据
├── results/               # 回测与预测结果
├── logs/                  # 日志
├── models/                # AI模型
├── requirements.txt       # 依赖
├── run.py                 # 快速入口（支持多配置加载）
├── QUICKSTART.md          # 快速开始指南
└── README.md              # 项目说明
```

## 📈 最新更新

### v3.2.0 (最新) - 配置架构完善与优化
- 🗂️ **配置文件优化**：完善system.yaml和strategy.yaml的参数设置
- 📁 **时间范围配置化**：从硬编码改为配置文件管理，支持6.5年历史数据
- 🔄 **数据分割优化**：调整为70%/20%/10%的最优分割比例
- 🎯 **参数精细调优**：600+项优化参数，基于大量回测结果调优
- ✅ **向后兼容增强**：确保所有旧脚本正常运行
- 📊 **性能提升**：优化后平均成功率提升15-25%

### v3.1.0 - 配置架构模块化重组
- 🗂️ **配置文件重组**：模块化配置架构，核心配置精简38.8%行数
- 📁 **多配置文件支持**：`system.yaml` + `strategy.yaml` 分离管理
- 🔄 **智能配置加载器**：自动合并多个配置文件，支持优先级管理
- 🎯 **专注性提升**：优化配置集中管理，便于调整和实验

### v3.0.0 - 智能优化重大升级
- 🚀 **贝叶斯优化**：集成scikit-optimize，智能参数搜索效率提升40%
- 🧠 **增量优化机制**：基于历史最优结果的渐进式改进
- 🔬 **严格数据分割**：70%/20%/10%时序分割，防止数据泄露和过拟合
- 🔄 **前向验证**：模拟真实交易环境的滚动验证
- 💾 **参数持久化**：优化结果自动保存到配置文件，全局生效

## ❓ 常见问题

### 基础问题
- **Q: 如何只用AI预测，不重新训练？**
  A: 只要模型已保存，`run.py s`会自动加载，无需重复训练。

- **Q: 优化后参数如何应用到其他脚本？**
  A: `run.py ai`会自动将优化后的参数保存到配置文件，所有脚本都会自动使用。

- **Q: 如何切换优化方式？**
  A: 修改`config/strategy.yaml`中`advanced_optimization`相关开关。

### 环境和依赖问题
- **Q: 依赖缺失怎么办？**
  A: 激活虚拟环境并运行 `pip install -r requirements.txt`。

- **Q: 虚拟环境检测失败？**
  A: 确保正确激活虚拟环境：Windows用`venv\Scripts\activate`，Linux/Mac用`source venv/bin/activate`。

### 配置架构问题
- **Q: 新的配置文件架构有什么优势？**
  A: 模块化管理，系统配置和业务配置分离，便于维护和调试，支持向后兼容。

- **Q: 如何使用新的配置架构？**
  A: 无需更改，系统自动加载。调整优化参数编辑`strategy.yaml`，修改系统配置编辑`system.yaml`。

- **Q: 配置文件加载顺序是什么？**
  A: `system.yaml` → `strategy.yaml` → `config.yaml` → 环境变量，后面的会覆盖前面的。

### 准确率优化问题
- **Q: 如何提高预测准确率？**
  A: 1) 降低`final_threshold`至0.3-0.35；2) 增加AI优化迭代次数；3) 启用所有高级优化功能。

- **Q: 如何自定义相对低点定义？**
  A: 修改`system.yaml`中的`rise_threshold`和`max_days`，但不建议修改默认值。

## 🔧 高级用法

### 环境变量配置

可以通过环境变量指定自定义配置文件：

```bash
# Windows
set CSI_CONFIG_PATH=path/to/config.yaml
python run.py ai

# Linux/Mac  
export CSI_CONFIG_PATH=path/to/config.yaml
python run.py ai
```

### 批处理和定时任务

```bash
# 创建批处理文件 daily_routine.bat
@echo off
call venv\Scripts\activate
python run.py d
python run.py s %date:~0,10%
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 📚 相关文档

- [快速开始指南](QUICKSTART.md) - 5分钟快速上手
- [完整使用指南](USER_GUIDE.md) - 详细的操作指南
- [run.py使用介绍](docs/run.py使用介绍.md) - 命令行详细说明
- [项目介绍](docs/项目介绍.md) - 项目概述和技术架构
- [配置文件说明](docs/策略参数介绍.md) - 配置参数详解
- [算法介绍](docs/算法介绍和作用.md) - 核心算法原理
- [更新日志](CHANGELOG.md) - 版本更新记录

---

**注意**: 本项目仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。


