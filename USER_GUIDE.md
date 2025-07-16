# 中证500指数相对低点识别系统 - 完整使用指南

> 📖 **欢迎使用**：本指南提供详细的操作步骤、故障排除和高级用法，帮助您充分利用本量化交易系统。

## 📋 目录

- [快速开始](#快速开始)
- [详细安装指南](#详细安装指南)
- [命令参考手册](#命令参考手册)
- [配置文件详解](#配置文件详解)
- [典型使用场景](#典型使用场景)
- [故障排除指南](#故障排除指南)
- [高级用法](#高级用法)
- [性能优化](#性能优化)

## 🚀 快速开始

### 前置要求
- **Python 3.8+** (推荐 3.9 或 3.10)
- **Git** (可选，用于克隆项目)
- **8GB+ 内存** (AI训练推荐)
- **2GB+ 可用磁盘空间**

### 3分钟快速体验
```bash
# 1. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python run.py b
```

### 首次完整体验流程
```bash
# 1. 获取最新数据（30秒）
python run.py d

# 2. 基础功能测试（1分钟）
python run.py b

# 3. AI模型训练（6-12分钟）
python run.py ai

# 4. 单日预测验证（30秒）
python run.py s 2024-12-01

# 5. 历史回测验证（2-5分钟）
python run.py r 2023-01-01 2023-12-31
```

## 📦 详细安装指南

### Windows 安装

#### 方法1：使用命令提示符
```cmd
# 1. 打开命令提示符（管理员模式推荐）
# 2. 导航到项目目录
cd C:\path\to\csi1000_quant

# 3. 创建虚拟环境
python -m venv venv

# 4. 激活虚拟环境
venv\Scripts\activate

# 5. 升级pip（可选但推荐）
python -m pip install --upgrade pip

# 6. 安装依赖
pip install -r requirements.txt

# 7. 验证安装
python run.py b
```

#### 方法2：使用PowerShell
```powershell
# 如果出现执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后按照方法1的步骤执行
```

### Linux/Mac 安装
```bash
# 1. 确保有Python 3.8+
python3 --version

# 2. 创建虚拟环境
python3 -m venv venv

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python run.py b
```

### 依赖问题解决

#### 常见依赖安装问题
```bash
# 问题1：pip版本过旧
python -m pip install --upgrade pip

# 问题2：某些包安装失败
pip install --upgrade setuptools wheel

# 问题3：网络连接问题
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 问题4：权限问题（Linux/Mac）
pip install --user -r requirements.txt
```

#### 核心依赖验证
```python
# 在Python中验证关键依赖
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import akshare as ak
print("所有核心依赖安装成功！")
```

## 📋 命令参考手册

### 基础命令

#### 数据管理命令

##### `python run.py d` - 数据获取
**功能**：自动获取中证500和中证1000指数的最新数据

**使用场景**：
- 每日数据更新
- 初次安装后获取历史数据
- 数据缺失时重新获取

**执行时间**：30-60秒（取决于网络状况）

**输出文件**：
- `data/SHSE.000905_1d.csv` - 中证500指数数据
- `data/SHSE.000852_1d.csv` - 中证1000指数数据

**示例输出**：
```
📊 开始获取最新数据...
✅ 成功获取中证500数据：1584条记录
✅ 成功获取中证1000数据：1584条记录
💾 数据已保存到 data/ 目录
⏱️ 执行时间：45.2秒
```

##### `python run.py b` - 基础测试
**功能**：验证系统基本功能，包括数据加载、策略计算、结果输出

**使用场景**：
- 系统安装后验证
- 配置文件修改后测试
- 日常健康检查

**执行时间**：30-90秒

**检查项目**：
- 配置文件加载
- 数据文件读取
- 技术指标计算
- 相对低点识别
- 结果输出

**示例输出**：
```
🔧 配置文件加载完成
📊 历史数据加载：1584条记录
🔍 相对低点识别：45个点位
📈 成功率：62.2%，平均涨幅：4.8%
✅ 基础测试通过
```

#### AI和优化命令

##### `python run.py ai` - AI优化训练
**功能**：运行完整的AI优化流程，包括参数优化、模型训练、性能验证

**模式选项**：
- `optimize`（默认）：完整优化流程
- `incremental`：增量训练模式
- `full`：完全重训练模式
- `demo`：演示预测模式

**执行时间**：6-12分钟（取决于数据量和参数设置）

**详细用法**：
```bash
# 完整优化（推荐）
python run.py ai

# 增量训练（快速更新）
python run.py ai -m incremental

# 完全重训练（彻底重建）
python run.py ai -m full

# 演示模式（仅预测，不训练）
python run.py ai -m demo

# 详细输出模式
python run.py ai -v
```

**优化过程**：
1. **数据准备**：加载历史数据，计算技术指标
2. **参数搜索**：使用贝叶斯优化和遗传算法
3. **模型训练**：训练随机森林预测模型
4. **性能验证**：前向验证和交叉验证
5. **结果保存**：优化参数自动保存到配置文件

##### `python run.py opt` - 策略参数优化
**功能**：专门优化策略参数，不涉及AI模型训练

**参数选项**：
- `-i N`：指定优化迭代次数（默认10）
- `-v`：详细输出模式

**执行时间**：2-10分钟（取决于迭代次数）

**使用示例**：
```bash
# 基础优化（10次迭代）
python run.py opt

# 高强度优化（50次迭代）
python run.py opt -i 50

# 详细输出优化
python run.py opt -i 30 -v
```

#### 预测和回测命令

##### `python run.py s <日期>` - 单日预测
**功能**：对指定日期进行相对低点预测

**参数格式**：YYYY-MM-DD

**使用示例**：
```bash
# 预测2024年12月1日
python run.py s 2024-12-01

# 预测历史日期
python run.py s 2023-06-15

# 预测多个日期（批量）
python run.py s 2024-01-01
python run.py s 2024-01-02
python run.py s 2024-01-03
```

**输出内容**：
- AI预测结果（Yes/No）
- 预测置信度（0.00-1.00）
- 技术指标详情
- 预测依据说明
- 生成的图表和报告

**输出文件**：
- `results/single_predictions/prediction_<日期>_<时间戳>.json`
- `results/reports/report_<日期>_<时间戳>.md`

##### `python run.py r <开始日期> <结束日期>` - 滚动回测
**功能**：在指定时间段内进行滚动回测，验证策略历史表现

**使用示例**：
```bash
# 回测2023年全年
python run.py r 2023-01-01 2023-12-31

# 回测最近半年
python run.py r 2024-06-01 2024-12-01

# 回测特定月份
python run.py r 2024-01-01 2024-01-31
```

**训练策略选择**：
系统会自动选择保守训练模式，也可以交互选择：
1. 智能训练（30天重训练间隔）
2. 保守训练（10天重训练间隔）
3. 传统模式（每日训练）

**输出内容**：
- 回测统计结果（成功率、平均涨幅、平均天数）
- 预测详情表格（美化版）
- 可视化图表
- 详细的预测记录

#### 测试和维护命令

##### `python run.py t` - 单元测试
**功能**：运行所有单元测试，验证系统各模块功能

**执行时间**：1-3分钟

**测试模块**：
- 配置加载测试
- 数据处理测试
- 策略逻辑测试
- AI优化测试
- 结果输出测试

##### `python run.py all` - 全面测试
**功能**：运行完整的系统测试，包含所有主要功能

**使用示例**：
```bash
python run.py all 2023-01-01 2023-12-31
```

**执行内容**：
1. 数据获取测试
2. 基础策略测试
3. AI模型测试
4. 单日预测测试
5. 滚动回测
6. 单元测试

### 命令参数详解

#### 全局参数
- `-v, --verbose`：启用详细输出模式
- `-i N, --iter N`：指定迭代次数
- `-m MODE, --mode MODE`：指定运行模式
- `--no-timer`：禁用性能计时器

#### 使用示例
```bash
# 详细输出的基础测试
python run.py b -v

# 50次迭代的参数优化
python run.py opt -i 50

# 增量模式的AI训练
python run.py ai -m incremental

# 禁用计时器的AI训练
python run.py ai --no-timer
```

## ⚙️ 配置文件详解

### 配置架构概览

本系统采用模块化配置架构：

```
config/
├── system.yaml       # 系统基础配置（196行）
├── strategy.yaml     # 策略优化配置（421行）
└── config.yaml       # 兼容性配置（保留）
```

**加载优先级**：`system.yaml` → `strategy.yaml` → `config.yaml` → 环境变量

### system.yaml 详解

#### 核心策略参数
```yaml
strategy:
  rise_threshold: 0.04      # 涨幅阈值（4%）
  max_days: 20              # 最大持有天数
```

**rise_threshold 解释**：
- **定义**：相对低点的最小涨幅要求
- **作用**：从识别日开始，未来20天内最高涨幅需达到4%
- **调整建议**：**不建议修改**，此值经过大量回测验证
- **影响**：调高会减少信号但提高质量，调低会增加信号但可能降低质量

**max_days 解释**：
- **定义**：观察期的最大天数
- **作用**：从识别日开始的最长观察期
- **调整建议**：**不建议修改**，20天是经过优化的最佳值
- **影响**：调短可能错过慢涨股票，调长可能引入更多噪音

#### AI配置参数
```yaml
ai:
  enable: true                    # 启用AI功能
  model_type: machine_learning    # 模型类型
  retrain_interval_days: 30       # 重训练间隔
  
  training_data:
    full_train_years: 6           # 完全重训练使用6年数据
    optimize_years: 6             # 参数优化使用6年数据
    incremental_years: 1          # 增量训练使用1年数据
```

#### 数据配置参数
```yaml
data:
  data_file_path: data/SHSE.000905_1d.csv  # 数据文件路径
  data_source: akshare                      # 数据源
  time_range:
    start_date: "2019-01-01"                # 数据开始日期
    end_date: "2025-07-15"                  # 数据结束日期
```

### strategy.yaml 详解

#### 优化算法配置
```yaml
optimization:
  global_iterations: 600          # 全局优化迭代次数
  incremental_iterations: 1200    # 增量优化迭代次数
  enable_incremental: true        # 启用增量优化
```

#### 贝叶斯优化配置
```yaml
bayesian_optimization:
  enabled: true                   # 启用贝叶斯优化
  n_calls: 120                    # 优化调用次数
  n_initial_points: 25            # 初始随机点数量
  xi: 0.008                       # 探索参数
```

#### 遗传算法配置
```yaml
genetic_algorithm:
  enabled: true                   # 启用遗传算法
  population_size: 50             # 种群大小
  generations: 30                 # 进化代数
  crossover_rate: 0.8             # 交叉概率
  mutation_rate: 0.15             # 变异概率
```

#### 置信度权重配置
```yaml
confidence_weights:
  final_threshold: 0.3392         # 最终置信度阈值
  ma_all_below: 0.32              # 价格低于所有均线权重
  rsi_oversold: 0.32              # RSI超卖权重
  volume_panic_bonus: 0.12        # 恐慌性放量奖励
```

**final_threshold 调优指南**：
- **当前值**：0.3392（经过优化的最佳值）
- **准确率不高时**：可降至0.30-0.35
- **过度保守时**：可降至0.25-0.30
- **追求质量时**：可升至0.35-0.40

### 配置文件修改指南

#### 安全修改原则
1. **不要修改**：`rise_threshold`、`max_days`（核心定义参数）
2. **可以调整**：`final_threshold`（置信度阈值）
3. **谨慎修改**：优化算法参数（迭代次数、种群大小）
4. **自由修改**：日志、通知、路径配置

#### 修改步骤
1. **备份原配置**：`cp config/strategy.yaml config/strategy.yaml.backup`
2. **小幅调整**：每次只修改1-2个参数
3. **测试验证**：`python run.py b` 验证配置正确性
4. **效果评估**：`python run.py s 2024-12-01` 测试预测效果

#### 常见修改场景

##### 提高预测准确率
```yaml
# 降低置信度阈值
confidence_weights:
  final_threshold: 0.30           # 从0.3392降至0.30

# 增加优化强度
optimization:
  global_iterations: 800          # 从600增至800
  
bayesian_optimization:
  n_calls: 150                    # 从120增至150
```

##### 加快训练速度
```yaml
# 减少迭代次数
optimization:
  global_iterations: 400          # 从600减至400
  
genetic_algorithm:
  population_size: 30             # 从50减至30
  generations: 20                 # 从30减至20
```

## 🎯 典型使用场景

### 场景1：日常投资辅助

#### 每日操作流程
```bash
# 1. 激活环境（每次启动终端后）
venv\Scripts\activate

# 2. 获取最新数据（每日一次）
python run.py d

# 3. 当日预测（工作日）
python run.py s 2024-12-08

# 4. 查看结果
# 查看 results/single_predictions/ 目录下的最新文件
```

#### 自动化脚本（Windows）
创建 `daily_check.bat`：
```batch
@echo off
cd /d "C:\path\to\csi1000_quant"
call venv\Scripts\activate
python run.py d
python run.py s %date:~0,10%
pause
```

#### 自动化脚本（Linux/Mac）
创建 `daily_check.sh`：
```bash
#!/bin/bash
cd /path/to/csi1000_quant
source venv/bin/activate
python run.py d
python run.py s $(date +%Y-%m-%d)
```

### 场景2：策略研究和回测

#### 历史策略验证
```bash
# 1. 训练模型
python run.py ai

# 2. 多期回测
python run.py r 2022-01-01 2022-12-31  # 2022年
python run.py r 2023-01-01 2023-12-31  # 2023年
python run.py r 2024-01-01 2024-12-01  # 2024年

# 3. 特定事件回测
python run.py r 2023-03-01 2023-04-30  # 银行业危机期间
python run.py r 2024-01-01 2024-02-29  # 春节前后
```

#### 参数敏感性分析
```bash
# 1. 记录当前配置效果
python run.py s 2024-01-15 > baseline.log

# 2. 修改final_threshold为0.30，测试效果
# 编辑config/strategy.yaml
python run.py s 2024-01-15 > test_030.log

# 3. 修改final_threshold为0.35，测试效果
# 编辑config/strategy.yaml
python run.py s 2024-01-15 > test_035.log

# 4. 对比结果，选择最佳参数
```

### 场景3：模型开发和优化

#### 模型重训练流程
```bash
# 1. 获取最新数据
python run.py d

# 2. 完全重训练（使用最新数据）
python run.py ai -m full

# 3. 增量训练（快速更新）
python run.py ai -m incremental

# 4. 验证新模型效果
python run.py r 2024-01-01 2024-12-01
```

#### 参数优化流程
```bash
# 1. 运行策略参数优化
python run.py opt -i 50

# 2. 运行AI模型优化
python run.py ai

# 3. 验证优化效果
python run.py s 2024-01-15

# 4. 对比优化前后效果
# 查看results/目录下的对比报告
```

### 场景4：系统维护和监控

#### 定期健康检查
```bash
# 每周运行一次
python run.py t        # 单元测试
python run.py b        # 基础功能测试
python run.py d        # 数据更新测试
```

#### 性能监控
```bash
# 启用详细输出，监控系统性能
python run.py ai -v --no-timer
python run.py r 2024-01-01 2024-02-01 -v
```

#### 日志分析
```bash
# 查看系统日志
tail -f logs/system.log

# 查看错误日志
grep ERROR logs/system.log

# 清理旧日志（可选）
find logs/ -name "*.log" -mtime +30 -delete
```

## 🔧 故障排除指南

### 常见错误及解决方案

#### 1. ImportError: No module named 'xxx'

**错误现象**：
```
ImportError: No module named 'pandas'
ImportError: No module named 'sklearn'
```

**解决方案**：
```bash
# 1. 确认虚拟环境已激活
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. 重新安装依赖
pip install -r requirements.txt

# 3. 如果某个包特别顽固
pip install pandas
pip install scikit-learn

# 4. 使用国内镜像（网络问题）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 2. FileNotFoundError: config/xxx.yaml

**错误现象**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/system.yaml'
```

**解决方案**：
```bash
# 1. 确认当前目录
pwd  # Linux/Mac
cd   # Windows

# 2. 确认在项目根目录
ls config/  # 应该看到 system.yaml, strategy.yaml

# 3. 如果配置文件丢失，检查备份
ls config/backups/

# 4. 如果完全丢失，重新获取项目
```

#### 3. 数据获取失败

**错误现象**：
```
网络连接超时
数据源返回空结果
akshare接口错误
```

**解决方案**：
```bash
# 1. 检查网络连接
ping baidu.com

# 2. 重试数据获取
python run.py d

# 3. 使用备份数据（如果有）
cp data/backup/SHSE.000905_1d.csv data/

# 4. 手动下载数据
# 从金融数据网站下载CSV文件，放入data/目录
```

#### 4. AI训练内存不足

**错误现象**：
```
MemoryError
OOM killed
系统变慢，无响应
```

**解决方案**：
```bash
# 1. 减少训练数据量
# 编辑config/system.yaml
ai:
  training_data:
    full_train_years: 3  # 从6减到3

# 2. 使用增量训练
python run.py ai -m incremental

# 3. 关闭其他程序，释放内存

# 4. 降低优化强度
# 编辑config/strategy.yaml
optimization:
  global_iterations: 300  # 从600减到300
```

#### 5. 虚拟环境问题

**错误现象**：
```
'venv' is not recognized as an internal or external command
虚拟环境激活失败
```

**解决方案**：

**Windows：**
```cmd
# 1. 检查Python安装
python --version

# 2. 重新创建虚拟环境
python -m venv venv

# 3. 如果激活失败，尝试绝对路径
C:\path\to\project\venv\Scripts\activate

# 4. PowerShell执行策略问题
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac：**
```bash
# 1. 检查Python3
python3 --version
which python3

# 2. 重新创建虚拟环境
python3 -m venv venv

# 3. 权限问题
chmod +x venv/bin/activate
source venv/bin/activate
```

### 性能问题诊断

#### 1. AI训练过慢

**诊断步骤**：
```bash
# 1. 启用详细输出，观察瓶颈
python run.py ai -v

# 2. 检查数据量
python -c "
import pandas as pd
data = pd.read_csv('data/SHSE.000905_1d.csv')
print(f'数据行数: {len(data)}')
"

# 3. 监控系统资源
# Windows: 任务管理器
# Linux: top, htop
# Mac: Activity Monitor
```

**优化方案**：
```yaml
# 减少优化强度（strategy.yaml）
optimization:
  global_iterations: 300       # 从600减到300
  
bayesian_optimization:
  n_calls: 80                  # 从120减到80
  
genetic_algorithm:
  population_size: 30          # 从50减到30
  generations: 20              # 从30减到20
```

#### 2. 预测速度过慢

**诊断步骤**：
```bash
# 1. 测试单日预测时间
time python run.py s 2024-12-01  # Linux/Mac
# Windows: 使用 Measure-Command { python run.py s 2024-12-01 }

# 2. 检查模型文件大小
ls -lh models/  # 模型文件过大可能影响加载速度

# 3. 查看详细执行过程
python run.py s 2024-12-01 -v
```

**优化方案**：
```bash
# 1. 重新训练精简模型
python run.py ai -m incremental

# 2. 清理缓存
rm -rf cache/*  # 清理数据缓存

# 3. 检查磁盘空间
df -h  # Linux/Mac
```

### 数据问题诊断

#### 1. 数据异常检查

```python
# 创建数据检查脚本 check_data.py
import pandas as pd
import numpy as np

def check_data():
    try:
        data = pd.read_csv('data/SHSE.000905_1d.csv')
        print(f"数据行数: {len(data)}")
        print(f"数据列: {list(data.columns)}")
        print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
        print(f"空值检查: {data.isnull().sum().sum()}")
        print(f"重复行: {data.duplicated().sum()}")
        
        # 检查价格数据合理性
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                print(f"{col}: {data[col].min():.2f} - {data[col].max():.2f}")
                
    except Exception as e:
        print(f"数据检查失败: {e}")

if __name__ == "__main__":
    check_data()
```

```bash
# 运行数据检查
python check_data.py
```

#### 2. 配置文件验证

```python
# 创建配置检查脚本 check_config.py
import yaml
import os

def check_config():
    config_files = [
        'config/system.yaml',
        'config/strategy.yaml',
        'config/config.yaml'
    ]
    
    for file_path in config_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {file_path} 加载成功")
                
                # 检查关键参数
                if 'strategy' in config:
                    strategy = config['strategy']
                    print(f"   rise_threshold: {strategy.get('rise_threshold')}")
                    print(f"   max_days: {strategy.get('max_days')}")
                    
            except Exception as e:
                print(f"❌ {file_path} 加载失败: {e}")
        else:
            print(f"⚠️ {file_path} 文件不存在")

if __name__ == "__main__":
    check_config()
```

```bash
# 运行配置检查
python check_config.py
```

## 🚀 高级用法

### 自定义配置文件

#### 创建自定义配置
```yaml
# 创建 config/custom.yaml
strategy:
  rise_threshold: 0.035        # 降低涨幅要求
  max_days: 25                 # 延长观察期

confidence_weights:
  final_threshold: 0.30        # 降低置信度阈值

ai:
  training_data:
    full_train_years: 8        # 增加训练数据
```

#### 使用自定义配置
```bash
# Windows
set CSI_CONFIG_PATH=config/custom.yaml
python run.py ai

# Linux/Mac
export CSI_CONFIG_PATH=config/custom.yaml
python run.py ai

# 一次性使用
CSI_CONFIG_PATH=config/custom.yaml python run.py ai
```

### 批量处理和自动化

#### 批量日期预测
```python
# 创建批量预测脚本 batch_predict.py
import subprocess
import pandas as pd
from datetime import datetime, timedelta

def batch_predict(start_date, end_date):
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    results = []
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        print(f"预测日期: {date_str}")
        
        # 运行预测命令
        try:
            result = subprocess.run(
                ['python', 'run.py', 's', date_str],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                results.append({
                    'date': date_str,
                    'status': 'success',
                    'output': result.stdout
                })
            else:
                results.append({
                    'date': date_str,
                    'status': 'failed',
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            results.append({
                'date': date_str,
                'status': 'timeout',
                'error': 'Prediction timed out'
            })
        
        current += timedelta(days=1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('batch_prediction_results.csv', index=False)
    print(f"批量预测完成，结果保存至 batch_prediction_results.csv")

if __name__ == "__main__":
    # 预测2024年1月的所有交易日
    batch_predict('2024-01-01', '2024-01-31')
```

#### 定时任务设置

**Windows 任务计划程序：**
1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器（每日、工作日等）
4. 设置操作：
   - 程序：`C:\path\to\project\venv\Scripts\python.exe`
   - 参数：`run.py d`
   - 起始位置：`C:\path\to\project`

**Linux Crontab：**
```bash
# 编辑crontab
crontab -e

# 添加定时任务
# 每个工作日早上9:30获取数据并预测
30 9 * * 1-5 cd /path/to/project && source venv/bin/activate && python run.py d && python run.py s $(date +\%Y-\%m-\%d)

# 每周日凌晨2点重新训练模型
0 2 * * 0 cd /path/to/project && source venv/bin/activate && python run.py ai
```

### 结果分析和报告

#### 自动报告生成
```python
# 创建报告生成脚本 generate_report.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def generate_monthly_report(year, month):
    # 查找指定月份的所有预测结果
    pattern = f"results/single_predictions/prediction_{year}-{month:02d}-*"
    files = glob.glob(pattern)
    
    if not files:
        print(f"未找到{year}年{month}月的预测结果")
        return
    
    results = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    
    # 统计分析
    df = pd.DataFrame(results)
    
    # 基本统计
    total_predictions = len(df)
    positive_predictions = len(df[df['predicted'] == True])
    accuracy = df['prediction_correct'].mean() if 'prediction_correct' in df.columns else None
    
    # 生成报告
    report = f"""
# {year}年{month}月预测报告

## 基本统计
- 预测总数: {total_predictions}
- 正向预测: {positive_predictions}
- 预测准确率: {accuracy:.2%} (如有实际结果)

## 详细结果
"""
    
    for _, row in df.iterrows():
        report += f"- {row['date']}: {row['predicted']} (置信度: {row['confidence']:.3f})\n"
    
    # 保存报告
    report_file = f"monthly_report_{year}_{month:02d}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"月度报告已生成: {report_file}")

if __name__ == "__main__":
    # 生成当前月份报告
    now = datetime.now()
    generate_monthly_report(now.year, now.month)
```

### 性能监控和调优

#### 系统性能监控
```python
# 创建性能监控脚本 monitor_performance.py
import time
import psutil
import subprocess
import json
from datetime import datetime

def monitor_command(command):
    """监控命令执行的性能指标"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    start_cpu = psutil.cpu_percent()
    
    # 执行命令
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    end_cpu = psutil.cpu_percent()
    
    # 计算指标
    execution_time = end_time - start_time
    memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
    avg_cpu = (start_cpu + end_cpu) / 2
    
    performance_data = {
        'command': command,
        'execution_time': execution_time,
        'memory_usage_mb': memory_usage,
        'avg_cpu_percent': avg_cpu,
        'timestamp': datetime.now().isoformat(),
        'success': result.returncode == 0
    }
    
    return performance_data, result

def run_performance_test():
    commands = [
        'python run.py b',
        'python run.py d', 
        'python run.py s 2024-01-15',
        'python run.py ai -m demo'
    ]
    
    results = []
    for cmd in commands:
        print(f"监控命令: {cmd}")
        perf_data, cmd_result = monitor_command(cmd)
        results.append(perf_data)
        
        print(f"执行时间: {perf_data['execution_time']:.2f}秒")
        print(f"内存使用: {perf_data['memory_usage_mb']:.2f}MB")
        print(f"CPU使用: {perf_data['avg_cpu_percent']:.1f}%")
        print(f"执行结果: {'成功' if perf_data['success'] else '失败'}")
        print("-" * 50)
    
    # 保存性能数据
    with open('performance_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("性能测试完成，结果保存至 performance_report.json")

if __name__ == "__main__":
    run_performance_test()
```

## 📊 性能优化

### 系统级优化

#### 1. 硬件优化建议
- **内存**：建议16GB+，最低8GB
- **存储**：SSD优先，确保有5GB+可用空间
- **CPU**：多核CPU可提升并行计算性能
- **网络**：稳定的互联网连接用于数据获取

#### 2. Python环境优化
```bash
# 使用更快的Python解释器
pypy3 -m venv venv_pypy  # 如果可用

# 优化pip安装
pip install --upgrade pip setuptools wheel

# 使用编译加速的包
pip install numpy --config-settings=setup-args="-Dallow-noblas=false"
```

#### 3. 系统配置优化

**Windows：**
```cmd
# 设置高性能电源计划
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 增加虚拟内存
# 控制面板 -> 系统 -> 高级系统设置 -> 性能设置 -> 高级 -> 虚拟内存
```

**Linux：**
```bash
# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 调整内核参数
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### 应用级优化

#### 1. 配置参数优化

**快速训练配置**（适合日常使用）：
```yaml
# config/fast.yaml
optimization:
  global_iterations: 200          # 减少迭代
  incremental_iterations: 400
  
bayesian_optimization:
  n_calls: 50                     # 减少调用次数
  n_initial_points: 10
  
genetic_algorithm:
  population_size: 20             # 减少种群
  generations: 15                 # 减少代数
```

**高精度配置**（适合重要决策）：
```yaml
# config/accurate.yaml
optimization:
  global_iterations: 1000         # 增加迭代
  incremental_iterations: 2000
  
bayesian_optimization:
  n_calls: 200                    # 增加调用次数
  n_initial_points: 40
  
genetic_algorithm:
  population_size: 100            # 增加种群
  generations: 50                 # 增加代数
```

#### 2. 数据优化

```bash
# 定期清理缓存
rm -rf cache/*

# 压缩历史日志
gzip logs/*.log

# 清理过期结果
find results/ -name "*.png" -mtime +30 -delete
find results/ -name "*.json" -mtime +90 -delete
```

#### 3. 并行优化

修改配置启用并行处理：
```yaml
# strategy.yaml
execution:
  parallel_jobs: 4                # 使用4个并行作业
  use_multiprocessing: true       # 启用多进程

bayesian_optimization:
  n_jobs: 4                       # 贝叶斯优化并行
```

### 监控和维护

#### 1. 定期维护脚本
```bash
# 创建维护脚本 maintenance.sh
#!/bin/bash

echo "开始系统维护..."

# 更新数据
python run.py d

# 清理缓存
rm -rf cache/*

# 压缩日志
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# 清理过期结果
find results/ -name "*.png" -mtime +30 -delete

# 系统健康检查
python run.py b

echo "维护完成"
```

#### 2. 性能基准测试
```python
# benchmark.py
import time
import subprocess

def benchmark_command(cmd, runs=3):
    times = []
    for i in range(runs):
        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True)
        end = time.time()
        
        if result.returncode == 0:
            times.append(end - start)
        else:
            print(f"命令失败: {cmd}")
            return None
    
    avg_time = sum(times) / len(times)
    print(f"{cmd}: 平均耗时 {avg_time:.2f}秒")
    return avg_time

if __name__ == "__main__":
    commands = [
        'python run.py b',
        'python run.py s 2024-01-15',
        'python run.py opt -i 10'
    ]
    
    for cmd in commands:
        benchmark_command(cmd)
```

---

## 🎯 总结

本使用指南涵盖了系统的所有主要功能和使用场景。关键要点：

1. **必须使用虚拟环境**，避免依赖冲突
2. **从基础测试开始**，逐步学习各项功能
3. **配置文件谨慎修改**，重点调整置信度阈值
4. **定期维护和监控**，确保系统稳定运行
5. **遇到问题先查日志**，多数问题都有明确的错误信息

通过本指南，您应该能够熟练使用系统的所有功能，并根据自己的需求进行定制和优化。如有其他问题，请参考项目文档或提交Issue。 