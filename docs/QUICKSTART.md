# 🚀 快速开始指南

本指南将帮助您在5分钟内快速上手中证500指数量化交易系统。

## 📋 前置要求

- Python 3.8 或更高版本
- 至少 4GB 可用内存
- 稳定的网络连接（用于数据获取）

## 🔧 环境设置

### 1. 克隆项目

```bash
git clone <repository-url>
cd csi_quant
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## ⚡ 快速体验

### 第一步：基础测试

```bash
# 运行基础策略测试
python run.py basic
```

这将验证系统基本功能是否正常。

### 第二步：AI优化

```bash
# 运行AI参数优化（需要5-10分钟）
python run.py ai
```

系统将：
- 自动获取历史数据
- 训练AI模型
- 优化策略参数
- 保存最优配置

### 第三步：查看结果

```bash
# 查看优化后的配置
python run.py config

# 运行回测验证
python run.py backtest
```

## 📊 核心命令

| 命令 | 功能 | 用时 |
|------|------|------|
| `python run.py help` | 查看帮助 | 即时 |
| `python run.py basic` | 基础策略测试 | 1-2分钟 |
| `python run.py ai` | AI优化训练 | 5-10分钟 |
| `python run.py predict` | 单日预测 | 10-30秒 |
| `python run.py backtest` | 滚动回测 | 2-5分钟 |
| `python run.py config` | 查看配置 | 即时 |

## 🎯 典型使用流程

### 新用户首次使用

```bash
# 1. 基础测试
python run.py basic

# 2. AI优化
python run.py ai

# 3. 验证结果
python run.py backtest
```

### 日常使用

```bash
# 获取最新预测
python run.py predict

# 定期重新优化（建议每月）
python run.py ai
```

### 参数调优

```bash
# 重置参数
python reset_strategy_params.py --all --force

# 重新优化
python run.py ai
```

## 📈 理解输出结果

### AI优化输出

```
策略得分: 0.4669
成功率: 30.67%
平均涨幅: 4.23%
平均天数: 8.5天
```

- **策略得分**：综合评分，越高越好
- **成功率**：识别正确的相对低点比例
- **平均涨幅**：成功识别后的平均收益
- **平均天数**：达到目标涨幅的平均时间

### 预测输出

```
日期: 2024-01-15
预测: 相对低点
置信度: 0.75
建议: 关注买入机会
```

- **置信度 > 0.7**：高置信度，建议重点关注
- **置信度 0.5-0.7**：中等置信度，谨慎考虑
- **置信度 < 0.5**：低置信度，不建议操作

## 🔧 配置调整

### 基本配置文件

- `config/strategy.yaml`：策略参数配置
- `config/system.yaml`：系统配置
- `config/optimized_params.yaml`：AI优化后的参数

### 常用调整

```yaml
# config/strategy.yaml
default_strategy:
  rise_threshold: 0.04    # 涨幅阈值（4%）
  max_days: 20           # 最大持有天数
  confidence_weights:
    final_threshold: 0.5   # 最终置信度阈值
```

## ❗ 常见问题

### 1. 依赖安装失败

```bash
# 升级pip
pip install --upgrade pip

# 重新安装依赖
pip install -r requirements.txt
```

### 2. 数据获取失败

- 检查网络连接
- 稍后重试
- 确认防火墙设置

### 3. 模型文件不存在

```bash
# 重新运行AI优化
python run.py ai
```

### 4. 配置文件错误

```bash
# 重置配置
python reset_strategy_params.py --all --force
```

## 🎓 进阶使用

### 自定义参数范围

编辑 `config/strategy.yaml` 中的 `optimization_ranges` 部分：

```yaml
optimization_ranges:
  rise_threshold: [0.02, 0.08]  # 涨幅阈值范围
  max_days: [10, 30]            # 持有天数范围
```

### 批量回测

```bash
# 指定时间范围回测
python run.py backtest --start 2023-01-01 --end 2023-12-31
```

### 模型管理

```bash
# 查看模型信息
cat models/latest_improved_model.txt

# 备份当前模型
cp models/latest_improved_model.txt models/backup_model.txt
```

## 📞 获取帮助

- 查看详细文档：`README.md`
- 命令帮助：`python run.py help`
- 参数说明：`python run.py config`
- 问题反馈：GitHub Issues

## 🎯 下一步

完成快速开始后，建议：

1. 阅读完整的 `README.md` 文档
2. 了解各个配置参数的含义
3. 尝试不同的优化策略
4. 定期更新和重新优化模型

---

**提示**：首次运行AI优化可能需要较长时间，请耐心等待。系统会自动保存优化结果，后续使用会更快。