# run.py 使用介绍

`run.py` 是本项目的主要入口文件，提供了简洁的命令行界面来运行系统的各种功能。本文档详细介绍所有命令的使用方法和参数说明。

## 前置要求

### 虚拟环境
**强烈建议**在虚拟环境中运行，避免包依赖冲突：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 基本格式
```bash
python run.py <命令> [参数]
```

## 基础命令

### 1. 数据获取 (`d`)

#### 功能说明
自动获取中证500和中证1000指数的最新数据。

#### 使用方法
```bash
python run.py d
```

#### 执行内容
- 通过akshare接口获取SHSE.000905（中证500）数据
- 通过akshare接口获取SHSE.000852（中证1000）数据
- 自动保存为CSV格式到data/目录
- 显示数据更新状态和统计信息

#### 输出文件
- `data/SHSE.000905_1d.csv`
- `data/SHSE.000852_1d.csv`

### 2. 基础测试 (`b`)

#### 功能说明
运行基础策略测试，验证系统基本功能。

#### 使用方法
```bash
python run.py b
```

#### 执行内容
- 加载历史数据
- 运行相对低点识别算法
- 进行基础回测
- 显示策略性能指标

#### 性能监控
自动显示执行时间和系统状态。

### 3. 单元测试 (`t`)

#### 功能说明
运行项目的单元测试套件。

#### 使用方法
```bash
python run.py t
```

#### 执行内容
- 自动发现tests/目录下的所有测试文件
- 运行所有test_*.py文件
- 显示测试结果和覆盖率

## AI 功能命令

### 4. AI优化训练 (`ai`)

#### 功能说明
运行AI优化系统，支持多种训练模式。

#### 基本使用
```bash
python run.py ai
```

#### 模式参数 (`-m`, `--mode`)

##### 完整优化模式（默认）
```bash
python run.py ai
python run.py ai -m optimize
```
- 运行完整的AI优化流程
- 包含参数优化、模型训练、性能评估
- 执行时间：6-12分钟

##### 增量训练模式
```bash
python run.py ai -m incremental
```
- 基于现有模型进行增量更新
- 使用最近1年的数据进行训练
- 适用于模型的快速更新

##### 完全重训练模式
```bash
python run.py ai -m full
```
- 从零开始训练新模型
- 使用完整的历史数据（默认6年）
- 适用于模型完全重建

##### 演示预测模式
```bash
python run.py ai -m demo
```
- 使用现有模型进行演示预测
- 显示预测过程和结果分析
- 快速验证模型效果

#### 输出内容
- 训练进度和性能指标
- 优化参数对比
- 模型评估结果
- 保存的模型文件路径

### 5. AI测试 (`a`)

#### 功能说明
运行AI测试，包含模型训练和预测验证。

#### 使用方法
```bash
python run.py a
```

#### 执行内容
- 加载配置文件
- 运行改进版AI优化
- 显示训练和预测结果

## 预测命令

### 6. 单日预测 (`s`)

#### 功能说明
对指定日期进行相对低点预测。

#### 使用方法
```bash
python run.py s <日期>
```

#### 参数说明
- `<日期>`：预测日期，格式为YYYY-MM-DD

#### 使用示例
```bash
# 预测2024年1月15日
python run.py s 2024-01-15

# 预测今天（使用当前日期）
python run.py s 2024-12-08
```

#### 输出内容
- 预测结果（是否为相对低点）
- 置信度分析
- 技术指标详情
- 预测依据说明
- 生成的图表和报告文件

#### 输出文件
- `results/single_predictions/prediction_<日期>_<时间戳>.json`
- `results/reports/report_<日期>_<时间戳>.md`
- `results/history/prediction_history.json`

### 7. 滚动回测 (`r`)

#### 功能说明
在指定时间段内进行滚动回测，验证策略历史表现。

#### 使用方法
```bash
python run.py r <开始日期> <结束日期>
```

#### 参数说明
- `<开始日期>`：回测开始日期，格式为YYYY-MM-DD
- `<结束日期>`：回测结束日期，格式为YYYY-MM-DD

#### 使用示例
```bash
# 回测2023年全年
python run.py r 2023-01-01 2023-12-31

# 回测最近半年
python run.py r 2024-06-01 2024-12-01
```

#### 训练策略选择
系统会自动选择保守训练模式（10天重训练间隔），也可以通过交互方式选择：
1. 智能训练（30天间隔）
2. 保守训练（10天间隔）
3. 传统模式（每日训练）

#### 输出内容
- 回测统计结果
- 成功率和平均涨幅
- 生成的可视化图表
- 详细的预测记录

#### 输出文件
- `results/charts/rolling_backtest/rolling_backtest_results_<时间戳>.png`
- `results/charts/rolling_backtest/prediction_details_<时间戳>.png`

## 优化命令

### 8. 策略优化 (`opt`)

#### 功能说明
运行策略参数优化，寻找最优参数组合。

#### 使用方法
```bash
python run.py opt [选项]
```

#### 参数选项

##### 迭代次数 (`-i`, `--iter`)
```bash
python run.py opt -i 50
```
- 指定优化迭代次数
- 默认值：10
- 建议值：20-100（取决于时间要求）

##### 详细输出 (`-v`, `--verbose`)
```bash
python run.py opt -v
```
- 启用详细输出模式
- 显示优化过程的详细信息

#### 使用示例
```bash
# 基础优化（10次迭代）
python run.py opt

# 中等强度优化（50次迭代）
python run.py opt -i 50

# 高强度详细优化（100次迭代，详细输出）
python run.py opt -i 100 -v
```

#### 执行内容
- 使用遗传算法或贝叶斯优化
- 搜索最优策略参数组合
- 自动保存优化结果到配置文件
- 显示优化前后的性能对比



## 综合命令

### 10. 全部测试 (`all`)

#### 功能说明
运行完整的系统测试，包含所有主要功能。

#### 使用方法
```bash
python run.py all <开始日期> <结束日期>
```

#### 参数说明
- `<开始日期>`：测试开始日期
- `<结束日期>`：测试结束日期

#### 使用示例
```bash
python run.py all 2023-01-01 2023-12-31
```

#### 执行内容
1. 数据获取测试
2. 基础策略测试
3. AI模型测试
4. 单日预测测试
5. 滚动回测
6. 单元测试

## 全局选项

### 性能监控

#### 启用监控（默认）
```bash
python run.py <命令>
```
- 自动显示执行时间
- 显示详细的性能统计

#### 禁用监控
```bash
python run.py <命令> --no-timer
```
- 禁用性能监控显示
- 适用于脚本自动化调用

### 详细输出
```bash
python run.py <命令> -v
python run.py <命令> --verbose
```
- 启用详细输出模式
- 显示更多调试信息

## 环境变量配置

### 自定义配置文件
```bash
# Windows
set CSI_CONFIG_PATH=path\to\config.yaml
python run.py ai

# Linux/Mac
export CSI_CONFIG_PATH=path/to/config.yaml
python run.py ai
```

### 虚拟环境检测
系统会自动检测虚拟环境状态：
- ✅ 在虚拟环境中：正常执行
- ⚠️ 不在虚拟环境中：显示警告但继续执行

## 常用工作流程

### 日常使用流程
```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 获取最新数据
python run.py d

# 3. 运行基础测试
python run.py b

# 4. 进行当日预测
python run.py s 2024-12-08


```

### 模型训练流程
```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 运行完整AI优化
python run.py ai

# 3. 验证模型效果
python run.py s 2024-12-08

# 4. 进行历史回测
python run.py r 2023-01-01 2024-12-01
```

### 参数优化流程
```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 运行策略优化
python run.py opt -i 50

# 3. 运行AI优化
python run.py ai

# 4. 验证优化效果
python run.py r 2023-01-01 2024-12-01
```

## 常见问题和解决方案

### 1. 导入错误
**问题**：ImportError 或模块找不到
**解决**：
```bash
# 确保激活虚拟环境
venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件错误
**问题**：配置文件加载失败
**解决**：
- 检查config/目录下的配置文件是否存在
- 使用默认配置重新生成
- 检查YAML语法格式

### 3. 数据获取失败
**问题**：网络错误或数据源问题
**解决**：
- 检查网络连接
- 重试数据获取命令
- 检查akshare接口状态

### 4. 内存不足
**问题**：大数据处理时内存不足
**解决**：
- 减少数据时间范围
- 分批处理数据
- 增加系统内存

### 5. 模型文件缺失
**问题**：预测时提示模型未训练
**解决**：
```bash
# 重新训练模型
python run.py ai -m full
```

## 高级用法

### 批处理脚本
```bash
# 创建批处理文件 daily_routine.bat
@echo off
call venv\Scripts\activate
python run.py d
python run.py s %date:~0,10%
```

### 定时任务设置
使用系统定时任务（Windows任务计划程序或Linux crontab）设置自动执行：
```bash
# 每日9:30执行
30 9 * * * cd /path/to/project && source venv/bin/activate && python run.py s
```

### 结果分析脚本
```python
# 分析脚本示例
import json
with open('results/daily_trading/trading_history.json') as f:
    history = json.load(f)
    # 分析交易历史
```

## 性能优化建议

1. **数据缓存**：启用数据缓存以提高重复访问速度
2. **并行处理**：在多核系统上启用并行计算
3. **内存管理**：及时清理不必要的数据
4. **定期维护**：定期清理日志和临时文件

## 总结

`run.py` 提供了完整的命令行界面，涵盖了从数据获取到模型训练、预测分析、自动化交易的全流程。通过合理使用这些命令，可以高效地完成各种量化交易任务。建议从基础命令开始熟悉，逐步掌握高级功能。 