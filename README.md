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
5. **系统工具**: `pyyaml`, `schedule` - 配置和定时任务

**❌ 常见问题及解决方案:**
- **ImportError**: 激活虚拟环境并运行 `pip install -r requirements.txt`
- **Mod块找不到**: 确保运行路径在项目根目录
- **配置文件错误**: 检查 `config/config.yaml` 是否存在

### 🔧 **模块化配置架构 (v3.1.0新增)**
项目采用全新的模块化配置文件架构，简洁高效：

```
config/
├── config_core.yaml     # 核心系统配置（104行）
├── optimization.yaml    # 优化相关配置（194行）
└── config.yaml          # 兼容性配置（保留）
```

**配置加载优先级**：
```
config_core.yaml → optimization.yaml → config.yaml → 环境变量配置
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
- ✅ 核心配置减少 **38.8%** 行数（171行→104行）
- ✅ 优化参数集中管理，便于调整和实验
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
- 🔬 **严格数据分割**：防止过拟合的时间序列分割和走前验证
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
- **严格数据分割**：65%/20%/15%的时序分割，防止数据泄露
- **走前验证**：模拟真实交易环境的滚动验证
- **早停机制**：防止过拟合的智能停止策略
- **过拟合检测**：多重验证确保模型泛化能力

### 📊 技术指标算法
- **RSI超卖检测**：相对强弱指数算法
- **MACD趋势分析**：移动平均收敛发散算法  
- **布林带分析**：统计学波动率算法
- **移动平均线组合**：多周期趋势识别

> 📖 **详细算法说明请参考**: [`docs/algorithms_overview.md`](docs/algorithms_overview.md)
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

## 环境要求

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

## 🏁 快速开始

### 基础使用流程

```bash
# 1. 激活虚拟环境 (Windows)
venv\Scripts\activate

# 2. 基础测试
python run.py b

# 3. AI训练 (6年数据，6-12分钟)
python run.py ai -m full

# 4. 单日预测
python run.py s 2024-12-01

# 5. 策略回测
python run.py r 2023-01-01 2023-12-31
```

### 完整安装步骤

如果您是新用户，请查看 **[完整使用指南](USER_GUIDE.md)** 获取详细的安装和配置说明。

简化步骤：
1. 创建并激活虚拟环境
2. 安装依赖 `pip install -r requirements.txt`
3. 运行基础测试 `python run.py b`
4. 开始使用系统

**⚠️ 重要**：必须在虚拟环境中运行，避免包依赖冲突。

## 📈 数据获取说明

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

### 使用建议

- 建议在交易时间后运行，以获取完整的数据
- 可以设置定时任务，实现自动数据更新
- 详细使用说明请参考 `docs/fetch_data_guide.md`

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
| `b` | 基础策略测试 | `python run.py b` | ✅ 支持 |
| `a` | AI测试（含训练与预测） | `python run.py a` | ✅ 支持 |
| `t` | 单元测试 | `python run.py t` | ✅ 支持 |
| `r` | 滚动回测 | `python run.py r 2023-01-01 2023-12-31` | ✅ 支持 |
| `s` | 单日预测 | `python run.py s 2023-12-01` | ✅ 支持 |
| `opt` | 策略参数优化 | `python run.py opt -i 20` | ✅ 支持 |
| `ai` | AI优化训练 | `python run.py ai -m incremental` | ✅ 支持 |
| `bot` | 交易机器人 | `python run.py bot -m run` | ✅ 支持 |
| `all` | 全部测试 | `python run.py all` | ✅ 支持 |

### 常用执行命令示例

#### 1. 快速开始
```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行基础测试
python run.py b
```

#### 2. AI模型训练
```bash
# 增量训练
python run.py ai -m incremental

# 完全重训练  
python run.py ai -m full

# 演示预测
python run.py ai -m demo

# 完整优化（默认）
python run.py ai
```

#### 3. 预测功能
```bash
# 单日预测
python run.py s 2024-01-15

# 滚动回测
python run.py r 2023-01-01 2023-12-31
```

#### 4. 交易机器人
```bash
# 单次运行
python run.py bot -m run

# 定时执行
python run.py bot -m schedule

# 查看状态
python run.py bot -m status
```

### 重要参数说明

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--verbose` | `-v` | 详细输出 | `python run.py b -v` |
| `--iter` | `-i` | 迭代次数（默认10） | `python run.py opt -i 50` |
| `--mode` | `-m` | 指定运行模式 | `python run.py ai -m incremental` |
| `--no-timer` | 无 | 禁用性能计时器 | `python run.py ai --no-timer` |

### AI模式详细说明

AI命令支持多种模式，通过 `-m` 参数指定：

| 模式 | 说明 | 示例 |
|------|------|------|
| `optimize` | 完整优化（默认） | `python run.py ai` 或 `python run.py ai -m optimize` |
| `incremental` | 增量训练 | `python run.py ai -m incremental` |
| `full` | 完全重训练 | `python run.py ai -m full` |
| `demo` | 演示预测 | `python run.py ai -m demo` |

### 机器人模式详细说明

交易机器人支持多种运行模式：

| 模式 | 说明 | 示例 |
|------|------|------|
| `run` | 单次运行（默认） | `python run.py bot` 或 `python run.py bot -m run` |
| `schedule` | 定时执行 | `python run.py bot -m schedule` |
| `status` | 查看状态 | `python run.py bot -m status` |

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

### 实际使用场景示例

#### 日常使用流程
```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 快速测试系统状态
python run.py b

# 3. 进行单日预测
python run.py s 2024-01-15

# 4. 查看机器人状态
python run.py bot -m status
```

#### 模型训练流程
```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 完整优化训练
python run.py ai

# 3. 验证训练效果
python run.py s 2024-01-15

# 4. 运行回测验证
python run.py r 2023-01-01 2023-12-31
```

#### 系统全面测试
```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行全套测试（包含所有功能）
python run.py all 2023-01-01 2023-12-31
```

**新增功能特点**：
- 🏗️ 自动分层优化
- 💾 参数自动持久化到配置文件
- 🌐 全局生效，所有脚本自动使用优化参数
- 📊 显示优化效果对比
- ⏱️ **性能监控**: 自动统计执行时间
- 🔧 **环境变量配置**: 支持 CSI_CONFIG_PATH
- 🛡️ **虚拟环境检测**: 自动检测并提醒

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

### 基础问题
- **Q: 如何只用AI预测，不重新训练？**
  A: 只要模型已保存，`predict_single_day.py`会自动加载，无需重复训练。

- **Q: 优化后参数如何应用到其他脚本？**
  A: `run.py ai`会自动将优化后的参数保存到配置文件，所有脚本都会自动使用。

- **Q: 如何切换优化方式？**
  A: 修改`config.yaml`中`ai.advanced_optimization`相关开关。

- **Q: 优化效果如何？**
  A: 分层优化可提升成功率77.7%，平均涨幅39.9%，综合得分33.0%。

- **Q: 参数会丢失吗？**
  A: 不会，优化后的参数会自动保存到配置文件，重启后仍然有效。

### 环境和依赖问题
- **Q: 依赖缺失怎么办？**
  A: 激活虚拟环境并运行 `pip install -r requirements.txt`。

- **Q: 虚拟环境检测失败？**
  A: 确保正确激活虚拟环境：Windows用`venv\Scripts\activate`，Linux/Mac用`source venv/bin/activate`。

- **Q: 提示找不到模块？**
  A: 确保在项目根目录运行命令，且已安装所有依赖包。

### 配置架构问题 (v3.1.0新增)
- **Q: 新的配置文件架构有什么优势？**
  A: 模块化管理，核心配置精简38.8%，优化配置集中管理，便于调整和实验，支持向后兼容。

- **Q: 如何使用新的配置架构？**
  A: 无需更改，系统自动加载。调整优化参数编辑`optimization.yaml`，修改系统配置编辑`config_core.yaml`。

- **Q: 旧的config.yaml还能用吗？**
  A: 能用！保持100%向后兼容，旧脚本无需修改。新架构会自动合并所有配置文件。

- **Q: 配置文件加载顺序是什么？**
  A: `config_core.yaml` → `optimization.yaml` → `config.yaml` → 环境变量，后面的会覆盖前面的。

### 新增功能问题
- **Q: 如何使用自定义配置文件？**
  A: 设置环境变量：`export CSI_CONFIG_PATH=/path/to/custom.yaml` (Linux/Mac) 或 `set CSI_CONFIG_PATH=C:\path\to\custom.yaml` (Windows)

- **Q: 性能监控显示的时间准确吗？**
  A: 是的，包括完整的命令执行时间，从开始到结束。可用 `--no-timer` 禁用。

- **Q: 如何查看配置文件合并结果？**
  A: 运行测试脚本会显示配置摘要，或查看日志输出的配置加载信息。

- **Q: 为什么建议使用虚拟环境？**
  A: 避免包版本冲突，特别是scikit-learn、pandas等核心依赖的版本兼容性问题。

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
│   └── utils/             # 工具（含多配置文件加载器）
├── examples/              # 典型用法脚本
├── config/                # 配置文件（模块化架构）
│   ├── config_core.yaml  # 核心系统配置
│   ├── optimization.yaml # 优化相关配置
│   └── config.yaml        # 兼容性配置
├── docs/                  # 文档
│   ├── config_reorganization_guide.md # 配置重组指南
│   └── ...                # 其他文档
├── data/                  # 历史数据
├── results/               # 回测与预测结果
├── logs/                  # 日志
├── models/                # AI模型
├── requirements.txt       # 依赖
├── run.py                 # 快速入口（支持多配置加载）
├── QUICKSTART.md          # 快速开始指南
└── README.md              # 项目说明
```

## 📋 配置文件说明

### 模块化配置架构

**config_core.yaml** - 核心系统配置：
- 策略参数：`strategy.rise_threshold`（涨幅阈值4%）、`strategy.max_days`（最大交易日数）
- AI基础配置：`ai.model_type`、`ai.enable`、`ai.models_dir`
- 数据配置：`data.data_source`、`data.index_code`、`data.frequency`
- 系统配置：`logging`、`notification`、`results`、`backtest`

**optimization.yaml** - 优化配置：
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
config_core.yaml → optimization.yaml → config.yaml → 环境变量
```

### 使用建议
- **调整优化参数**：编辑 `optimization.yaml`
- **修改系统配置**：编辑 `config_core.yaml`
- **保持兼容性**：保留 `config.yaml`

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

## 📈 最新更新

### v3.1.0 (最新) - 配置架构模块化重组
- 🗂️ **配置文件重组**：模块化配置架构，核心配置减少38.8%行数
- 📁 **多配置文件支持**：`config_core.yaml` + `optimization.yaml` 分离管理
- 🔄 **智能配置加载器**：自动合并多个配置文件，支持优先级管理
- 🔧 **配置验证机制**：智能验证配置完整性，提供详细错误信息
- 🎯 **专注性提升**：优化配置集中管理，便于调整和实验
- ✅ **向后兼容**：保留`config.yaml`，现有脚本无需修改
- 📊 **配置摘要显示**：可视化配置结构，便于理解和调试
- 🌐 **环境变量增强**：支持多种配置文件路径指定方式

### v3.0.0 - 智能优化重大升级
- 🚀 **贝叶斯优化**：集成scikit-optimize，智能参数搜索效率提升40%
- 🧠 **增量优化机制**：基于历史最优结果的渐进式改进，避免重复搜索
- 🔬 **严格数据分割**：65%/20%/15%时序分割，防止数据泄露和过拟合
- 🔄 **走前验证**：模拟真实交易环境的滚动验证
- ⏹️ **早停机制**：智能防护过拟合，提升模型泛化能力
- 🏗️ **模块化重构**：代码从1773行重构为5个专业模块
- 💾 **参数持久化**：优化结果自动保存到配置文件，全局生效
- 📊 **过拟合检测**：多重验证确保模型稳定性
- 🎯 **自适应搜索空间**：基于当前参数动态调整搜索范围

### v2.1.0
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

## 📊 AI优化日志功能

### 实时进度监控

系统现在提供详细的AI优化进度日志，让您能够实时了解：

- **优化阶段**: 清晰显示4个优化阶段（基准策略、固定参数、搜索范围、参数迭代）
- **进度百分比**: 每10次迭代显示进度（6.7%, 13.3%, 20.0%...）
- **时间估算**: 显示已用时间和预计剩余时间
- **实时得分**: 显示当前最佳参数组合的得分
- **参数详情**: 显示当前最佳参数的具体值

### 分层优化日志

支持四层优化结构，每层都有详细的日志输出：

1. **第一层**: 策略参数优化
2. **第二层**: AI模型训练
3. **第三层**: 时间序列交叉验证
4. **第四层**: 高级优化

### 示例日志输出

```
🚀 开始AI策略参数优化
============================================================
📊 阶段1: 获取基准策略识别结果...
✅ 基准策略识别点数: 357
🔧 阶段2: 设置固定参数...
✅ 固定参数设置完成:
   - rise_threshold: 0.04
   - max_days: 20
📋 阶段3: 配置参数搜索范围...
✅ 可优化参数搜索范围:
   - rsi_oversold_threshold: 25 - 35, 步长: 1
   - rsi_low_threshold: 35 - 45, 步长: 1
   ...
📈 总搜索组合数: 1,045,440
🎯 使用随机采样，最大迭代次数: 150
🔄 阶段4: 开始参数优化迭代...
--------------------------------------------------
🎉 发现更好的参数组合 (第1次改进, 迭代1):
   📈 得分提升: 0.0000 → 0.2597
   🔧 参数详情:
      - RSI超卖阈值: 28
      - RSI低值阈值: 41
      - 最终置信度: 0.450
      - 动态调整系数: 0.210
      - 市场情绪权重: 0.160
      - 趋势强度权重: 0.160
--------------------------------------------------
📊 进度: 6.7% (10/150)
⏱️  已用时间: 8.9s, 预计剩余: 124.5s
🏆 当前最佳得分: 0.2597
🎯 当前最佳参数: RSI超卖=28, RSI低值=41, 置信度=0.450
------------------------------
```

### 测试日志功能

运行以下命令查看详细的AI优化日志：

```bash
python test_ai_optimization_logs.py
```

## 🛠️ 安装和配置

### 环境要求

- Python 3.8+
- 虚拟环境（推荐）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd csi1000_quant
```

2. **创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置参数**
编辑 `config/config.yaml` 文件，设置您的参数。

## 📈 使用方法

### 基础使用

```python
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer

# 加载配置和数据
config = load_config()
data_module = DataModule(config)
data = data_module.get_history_data('2023-01-01', '2025-06-21')
data = data_module.preprocess_data(data)

# 初始化策略和AI优化器
strategy_module = StrategyModule(config)
ai_optimizer = AIOptimizer(config)

# 运行AI优化（会显示详细进度日志）
optimized_params = ai_optimizer.optimize_strategy_parameters(strategy_module, data)

# 更新策略参数
strategy_module.update_params(optimized_params)

# 运行回测
backtest_results = strategy_module.backtest(data)
evaluation = strategy_module.evaluate_strategy(backtest_results)
```

### 分层优化

```python
# 运行完整的分层优化
result = ai_optimizer.hierarchical_optimization(data)

# 获取优化结果
best_params = result['params']
best_score = result['best_score']
total_time = result['total_time']
```

### 快速开始

```bash
# 运行基础测试
python examples/basic_test.py

# 运行AI优化测试
python test_ai_optimization_logs.py

# 运行滚动回测
python examples/run_rolling_backtest.py
```

## 📁 项目结构

```
csi1000_quant/
├── config/                 # 配置文件（模块化架构）
│   ├── config_core.yaml  # 核心系统配置
│   ├── optimization.yaml # 优化相关配置
│   └── config.yaml        # 兼容性配置
├── src/                   # 源代码
│   ├── ai/               # AI优化模块
│   ├── data/             # 数据处理模块
│   ├── strategy/         # 策略模块
│   ├── utils/            # 工具函数
│   └── notification/     # 通知模块
├── data/                 # 数据文件
├── results/              # 结果输出
├── logs/                 # 日志文件
├── examples/             # 示例脚本
├── tests/                # 测试文件
└── docs/                 # 文档
```

## 🔧 配置说明

### 主要配置项

- **策略参数**: 涨幅阈值、最大持仓天数等
- **技术指标**: RSI、MACD、布林带等参数
- **AI优化**: 优化范围、迭代次数、评分权重等
- **日志设置**: 日志级别、文件路径等

### 新增AI优化参数

系统新增了3个AI优化参数，每次优化后自动更新：

1. **dynamic_confidence_adjustment**: 动态置信度调整系数
2. **market_sentiment_weight**: 市场情绪权重
3. **trend_strength_weight**: 趋势强度权重

详细说明请参考 [AI优化参数文档](docs/ai_optimization_params.md)。

## 📊 策略特性

### 技术指标组合

- **移动平均线**: 5日、10日、20日、60日均线
- **RSI指标**: 相对强弱指数
- **MACD指标**: 移动平均收敛发散
- **布林带**: 价格通道分析
- **成交量分析**: 市场情绪判断

### 智能识别逻辑

1. **多条件筛选**: 结合多个技术指标进行综合判断
2. **置信度评分**: 使用加权评分系统评估买入信号
3. **动态调整**: 根据市场环境自动调整参数
4. **风险控制**: 内置止损和持仓时间控制

## 📈 性能分析

### 回测结果

系统提供完整的回测分析功能：

- **成功率统计**: 信号准确率分析
- **收益分析**: 平均涨幅、最大回撤等
- **风险指标**: 夏普比率、信息比率等
- **可视化图表**: 收益曲线、信号分布等

### 优化效果

通过AI优化，策略性能得到显著提升：

- **参数自适应**: 根据历史数据自动优化参数
- **多目标优化**: 平衡成功率、收益和风险
- **时间序列验证**: 确保参数在不同时期都有效

## 🔍 监控和日志

### 日志系统

- **详细进度**: AI优化过程的实时进度显示
- **性能统计**: 优化时间、迭代次数、改进次数等
- **错误追踪**: 清晰的错误信息和异常处理
- **文件管理**: 自动日志轮转和备份

### 日志文件

- **主日志**: `logs/system.log`
- **AI优化日志**: `logs/ai_optimization.log`
- **回测日志**: `logs/backtest.log`

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python test_ai_optimization_params.py
python test_ai_optimization_logs.py
```

### 测试覆盖

- **参数优化测试**: 验证AI优化功能
- **策略回测测试**: 验证策略逻辑
- **数据处理测试**: 验证数据预处理
- **日志功能测试**: 验证日志输出

## 📚 文档

- [快速开始指南](QUICKSTART.md)
- [配置文件重组指南](docs/config_reorganization_guide.md) ⭐ 新增
- [AI优化参数说明](docs/ai_optimization_params.md)
- [API参考文档](docs/api_reference.md)
- [使用指南](docs/usage_guide.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 项目讨论区

---

**注意**: 本项目仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。


