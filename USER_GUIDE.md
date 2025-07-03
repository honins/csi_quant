# 📊 中证500指数相对低点识别系统 - 完整使用指南

## 🎯 系统概述

中证500指数相对低点识别系统是一个基于AI机器学习的量化交易辅助工具，专门用于识别中证500指数的相对低点，为投资决策提供数据支持。

### 🌟 主要功能
- 🤖 **AI模型训练**: 基于6年历史数据训练机器学习模型
- 📈 **相对低点预测**: 预测指定日期是否为相对低点
- 📊 **策略回测**: 验证策略在历史数据上的表现
- 🔧 **参数优化**: 自动优化策略参数
- 🤖 **自动交易机器人**: 定时执行预测和通知
- 📋 **详细报告**: 生成预测报告和可视化图表

### 🎨 技术特点
- **直接置信度使用**: 完整保留AI模型原始判断，零信息损失
- **增量学习**: 支持模型的增量更新
- **严格数据分割**: 防止数据泄露，确保结果可靠
- **多配置支持**: 灵活的配置管理
- **全自动化**: 从数据获取到预测的全流程自动化

---

## 🚀 快速开始

### 🔧 环境准备

**检查Python版本** (需要Python 3.8+):
```bash
python --version
```

**激活虚拟环境**:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### ⚡ 首次运行

**步骤1: 基础测试**
```bash
python run.py b
```
验证系统基本功能是否正常。

**步骤2: 完全AI训练**
```bash
python run.py ai -m full
```
使用6年历史数据训练AI模型（预计6-12分钟）。

**步骤3: 测试预测**
```bash
python run.py s 2024-06-15
```
测试单日预测功能。

### ✅ 验证安装

如果所有命令都成功执行，系统就可以正常使用了！

---

## 📋 完整命令参考

### 🎯 命令格式
```bash
python run.py <command> [options] [arguments]
```

### 📚 命令列表

#### 🧪 测试命令
```bash
# 基础功能测试
python run.py b

# AI功能测试  
python run.py a

# 单元测试
python run.py t

# 运行所有测试
python run.py all
```

#### 🤖 AI训练命令
```bash
# 完全重训练 (推荐，使用6年数据)
python run.py ai -m full

# 增量训练 (基于已有模型更新)
python run.py ai -m incremental

# 演示预测 (加载模型进行预测演示)
python run.py ai -m demo

# 完整优化 (参数优化 + 模型训练)
python run.py ai
python run.py ai -m optimize
```

#### 📈 预测和回测命令
```bash
# 单日预测
python run.py s 2024-12-01
python run.py s 2024-06-15

# 历史回测
python run.py r 2023-01-01 2023-12-31
python run.py r 2024-01-01 2024-06-30
```

#### 🔧 策略优化命令
```bash
# 策略参数优化 (默认10次迭代)
python run.py opt

# 指定迭代次数
python run.py opt -i 20
python run.py opt --iter 50
```

#### 🤖 交易机器人命令
```bash
# 单次运行
python run.py bot -m run

# 定时执行 (持续运行)
python run.py bot -m schedule

# 查看状态
python run.py bot -m status
```

### ⚙️ 通用选项
```bash
-v              # 详细输出模式
-m <模式>       # 指定运行模式
-i <数字>       # 迭代次数
--no-timer      # 禁用性能计时器
-h, --help      # 显示帮助信息
```

---

## ⚙️ 配置文件详解

### 📁 配置文件位置
- `config/config.yaml` - 主配置文件
- `config/config_core.yaml` - 核心配置文件
- `config/optimization.yaml` - 优化配置文件

### 🔧 主要配置项

#### AI模型配置
```yaml
ai:
  enable: true
  model_type: machine_learning
  
  # 训练数据时间范围
  training_data:
    full_train_years: 6      # 完全训练使用6年数据
    optimize_years: 6        # 优化模式使用6年数据
    incremental_years: 1     # 增量训练使用1年数据
    
  # 训练参数
  train_test_split_ratio: 0.8
  data_decay_rate: 0.4
  
  # 置信度配置（已简化，无平滑处理）
  # 注：置信度平滑功能已废弃，现在直接使用AI模型原始输出
```

#### 策略配置
```yaml
strategy:
  # 核心参数
  rise_threshold: 0.04      # 涨幅阈值 (4%)
  max_days: 20              # 最大观察天数
  
  # 技术指标参数
  rsi_period: 14
  bb_period: 20
  bb_std: 2
  
  # 置信度权重
  confidence_weights:
    final_threshold: 0.5
    rsi_oversold_threshold: 30
    rsi_low_threshold: 40
```

#### 数据配置
```yaml
data:
  data_file_path: data/SHSE.000905_1d.csv
  data_source: akshare
  index_code: SHSE.000905
  history_days: 1000
```

### 🔄 自定义配置

**使用环境变量指定配置文件**:
```bash
CSI_CONFIG_PATH=path/to/custom_config.yaml python run.py ai -m full
```

**修改默认配置**:
```yaml
# 在 config/config.yaml 中修改
ai:
  training_data:
    full_train_years: 8     # 改为8年数据
    
strategy:
  rise_threshold: 0.05      # 改为5%涨幅阈值
```

---

## 🎯 典型使用场景

### 📈 场景1: 日常预测

**目标**: 预测明天是否为相对低点

```bash
# 1. 确保模型是最新的 (可选)
python run.py ai -m incremental

# 2. 进行预测 (替换为实际日期)
python run.py s 2024-12-02

# 3. 查看结果文件
# results/single_predictions/prediction_2024-12-02_*.json
# results/reports/report_2024-12-02_*.md
```

### 🔄 场景2: 模型重训练

**目标**: 定期重新训练模型

```bash
# 方式1: 增量训练 (快速，推荐)
python run.py ai -m incremental

# 方式2: 完全重训练 (彻底，耗时较长)
python run.py ai -m full

# 方式3: 完整优化 (参数优化 + 训练)
python run.py ai -m optimize
```

### 📊 场景3: 策略回测

**目标**: 验证策略在历史数据上的表现

```bash
# 最近一年回测
python run.py r 2023-07-01 2024-07-01

# 疫情期间回测
python run.py r 2020-01-01 2021-12-31

# 查看结果文件
# results/charts/rolling_backtest/
```

### 🤖 场景4: 自动化运行

**目标**: 设置自动化交易机器人

```bash
# 单次运行 (测试)
python run.py bot -m run

# 定时运行 (生产环境)
python run.py bot -m schedule

# 检查状态
python run.py bot -m status
```

### 🔧 场景5: 参数优化

**目标**: 优化策略参数以提升表现

```bash
# 快速优化 (10次迭代)
python run.py opt

# 深度优化 (50次迭代)
python run.py opt -i 50

# 完整AI优化 (包含参数优化和模型训练)
python run.py ai -m optimize
```

---

## 📊 输出文件说明

### 📁 文件结构
```
results/
├── single_predictions/          # 单日预测结果
│   └── prediction_YYYY-MM-DD_*.json
├── reports/                     # 预测报告
│   └── report_YYYY-MM-DD_*.md
├── history/                     # 历史记录
│   └── prediction_history.json
└── charts/                      # 图表文件
    └── rolling_backtest/        # 回测图表
        ├── rolling_backtest_results_*.png
        └── prediction_details_*.png

models/                          # 模型文件
├── improved_model_*.pkl         # AI模型文件
├── latest_improved_model.txt    # 最新模型指针
└── confidence_history.json     # 置信度历史

logs/                           # 日志文件
└── system.log                 # 系统日志
```

### 📄 结果文件格式

**预测结果 JSON** (`results/single_predictions/`):
```json
{
  "date": "2024-12-01",
  "prediction": {
    "is_low_point": true,
    "confidence": 0.7234,
    "final_confidence": 0.7234
  },
  "model_info": {
    "model_type": "RandomForestClassifier",
    "training_date": "2024-11-30"
  }
}
```

**预测报告 Markdown** (`results/reports/`):
```markdown
# 相对低点预测报告 - 2024-12-01

## 预测结果
- 📈 **预测**: 相对低点
- 📊 **AI置信度**: 72.34% (原始模型输出)
- 🤖 **模型**: RandomForest

## 技术指标分析
- RSI: 28.5 (超卖)
- 均线位置: 价格低于所有均线
- 成交量: 放量下跌
```

---

## 🔧 故障排除

### ❌ 常见问题

#### 问题1: 模块导入错误
```bash
ImportError: No module named 'src.xxx'
```
**解决方案**:
```bash
# 确保在项目根目录运行
cd /path/to/csi1000_quant
python run.py b
```

#### 问题2: 配置文件加载失败
```bash
❌ 无法加载配置文件
```
**解决方案**:
```bash
# 检查配置文件是否存在
ls config/config.yaml

# 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

#### 问题3: 虚拟环境问题
```bash
⚠️ 您似乎没有在虚拟环境中运行
```
**解决方案**:
```bash
# 激活虚拟环境
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

#### 问题4: 数据文件不存在
```bash
❌ 数据文件不存在: data/SHSE.000905_1d.csv
```
**解决方案**:
```bash
# 运行数据获取脚本
python src/data/fetch_latest_data.py

# 或手动下载数据文件到 data/ 目录
```

#### 问题5: 模型文件不存在
```bash
❌ 未找到已训练的模型
```
**解决方案**:
```bash
# 进行完全训练
python run.py ai -m full
```

### 🔍 调试模式

**启用详细输出**:
```bash
python run.py ai -m full -v
```

**查看日志文件**:
```bash
tail -f logs/system.log
```

**检查系统状态**:
```bash
python run.py b -v
```

---

## 🚀 高级用法

### 🔧 自定义配置

**创建自定义配置文件**:
```bash
cp config/config.yaml config/my_config.yaml
# 编辑 my_config.yaml
CSI_CONFIG_PATH=config/my_config.yaml python run.py ai -m full
```

**临时修改训练参数**:
```yaml
# config/my_config.yaml
ai:
  training_data:
    full_train_years: 10    # 使用10年数据
    
strategy:
  rise_threshold: 0.06      # 提高到6%涨幅阈值
```

### 📊 批量操作

**批量预测多个日期**:
```bash
# 创建批量预测脚本
for date in 2024-06-01 2024-06-02 2024-06-03; do
    python run.py s $date
done
```

**批量回测不同时间段**:
```bash
# 年度回测
python run.py r 2020-01-01 2020-12-31
python run.py r 2021-01-01 2021-12-31
python run.py r 2022-01-01 2022-12-31
```

### 🔄 定时任务设置

**Linux/Mac cron设置**:
```bash
# 编辑crontab
crontab -e

# 添加定时任务 (每个交易日9:30执行)
30 9 * * 1-5 cd /path/to/csi1000_quant && python run.py bot -m run
```

**Windows任务计划程序**:
```bat
@echo off
cd /d G:\repo\csi1000_quant
call venv\Scripts\activate
python run.py bot -m run
```

### 📈 性能优化

**提升训练速度**:
```yaml
# config/config.yaml
ai:
  training_data:
    full_train_years: 4     # 减少到4年数据
    
  optimization:
    global_iterations: 200  # 减少优化次数
```

**内存优化**:
```yaml
# 对于内存有限的环境
data:
  history_days: 500         # 减少历史数据天数
  cache_enabled: false      # 禁用缓存
```

---

## 📞 技术支持

### 📚 文档资源
- `README.md` - 项目概述
- `QUICKSTART.md` - 快速开始指南
- `RESET_GUIDE.md` - 参数重置指南
- `DATA_ANALYSIS.md` - 数据范围分析
- `DOCS.md` - 详细技术文档

### 🔍 问题排查流程
1. 检查 `logs/system.log` 日志文件
2. 运行 `python run.py b -v` 进行基础测试
3. 确认虚拟环境和依赖包安装
4. 检查配置文件语法和路径
5. 确认数据文件是否存在

### 💡 最佳实践
- 定期运行 `python run.py ai -m incremental` 更新模型
- 使用 `python run.py r` 验证策略表现
- 保持配置文件备份
- 定期清理旧的结果文件
- 监控系统日志

---

## 🏆 总结

本系统提供了完整的AI驱动的相对低点识别解决方案，从数据获取到模型训练，从单日预测到策略回测，为量化投资提供了强大的技术支持。

**核心优势**:
- 🤖 **智能化**: AI模型自动学习市场规律
- ⚡ **高效率**: 6年数据训练仅需6-12分钟
- 🎯 **准确性**: 严格数据分割确保结果可靠
- 🔄 **可扩展**: 支持增量学习和参数优化
- 📊 **可视化**: 详细报告和图表展示

开始您的量化投资之旅吧！🚀 