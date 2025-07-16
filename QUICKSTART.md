# 快速开始指南

> 📋 **完整使用指南**：查看 [**USER_GUIDE.md**](USER_GUIDE.md) 获取最详细的使用说明、故障排除和高级用法。

## 🚀 5分钟快速上手

### ⚠️ 重要：必须使用虚拟环境

本项目**强烈要求**在虚拟环境中运行，避免依赖冲突：

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 🎯 核心命令（5分钟体验）

```bash
# 基础测试（1分钟）
python run.py b

# AI训练（6-12分钟）
python run.py ai

# 单日预测（30秒）
python run.py s 2024-12-01

# 策略回测（2-5分钟）
python run.py r 2023-01-01 2023-12-31
```

## 📋 主要命令速查

| 命令 | 功能 | 耗时 | 说明 |
|------|------|------|------|
| `d` | 数据获取 | 30秒 | 获取最新的中证500数据 |
| `b` | 基础测试 | 1分钟 | 验证系统基本功能 |
| `ai` | AI优化 | 6-12分钟 | 完整的AI模型训练 |
| `s <日期>` | 单日预测 | 30秒 | 预测指定日期是否为低点 |
| `r <开始> <结束>` | 滚动回测 | 2-10分钟 | 回测策略历史表现 |
| `t` | 单元测试 | 2分钟 | 运行所有测试用例 |

## 🔧 环境要求

- **Python 3.8+** (推荐 3.9 或 3.10)
- **虚拟环境** (必须使用)
- **内存**: 建议 8GB+
- **磁盘**: 至少 2GB 可用空间

## ❓ 常见问题快速解决

### Q: ImportError 或模块找不到
```bash
# 确保激活虚拟环境
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 重新安装依赖
pip install -r requirements.txt
```

### Q: 如何查看详细输出？
```bash
# 添加 -v 参数
python run.py b -v
python run.py ai -v
```

### Q: 如何禁用性能监控？
```bash
# 添加 --no-timer 参数
python run.py ai --no-timer
```

### Q: 配置文件在哪里？
- `config/system.yaml` - 系统基础配置
- `config/strategy.yaml` - 策略优化配置
- `config/config.yaml` - 兼容性配置

## 📈 典型使用流程

### 新手入门流程
```bash
# 1. 设置环境
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. 基础测试
python run.py b

# 3. 获取数据
python run.py d

# 4. 开始预测
python run.py s 2024-12-01
```

### 日常使用流程
```bash
# 1. 激活环境
venv\Scripts\activate

# 2. 更新数据
python run.py d

# 3. 当日预测
python run.py s 2024-12-08

# 4. 查看结果
# 查看 results/ 目录下的图表和报告
```

### 模型训练流程
```bash
# 1. 激活环境
venv\Scripts\activate

# 2. 完整AI训练
python run.py ai

# 3. 验证效果
python run.py s 2024-12-01

# 4. 历史回测
python run.py r 2023-01-01 2024-12-01
```

## 📊 项目结构（重点目录）

```
csi1000_quant/
├── config/                # 配置文件
│   ├── system.yaml       # 系统配置
│   └── strategy.yaml     # 策略配置
├── data/                  # 数据文件
├── results/               # 结果输出
│   ├── charts/           # 图表
│   ├── reports/          # 报告
│   └── single_predictions/ # 预测结果
├── logs/                  # 日志文件
├── models/                # AI模型
├── run.py                 # 主入口（最重要）
└── requirements.txt       # 依赖列表
```

## 🔧 VS Code 使用（可选）

### 打开项目
1. 启动VS Code
2. 选择"文件" -> "打开文件夹"
3. 选择项目文件夹

### 可用的调试配置
- **运行基础测试**: 测试系统基本功能
- **运行AI优化测试**: 测试AI优化功能
- **运行快速脚本**: 使用命令行参数运行

## ⚙️ 配置说明（重要参数）

### system.yaml 关键参数
```yaml
strategy:
  rise_threshold: 0.04      # 涨幅阈值（4%）- 不建议修改
  max_days: 20             # 最大持有天数 - 不建议修改

ai:
  enable: true             # 启用AI功能
  full_train_years: 6      # 训练数据年数
```

### strategy.yaml 关键参数
```yaml
confidence_weights:
  final_threshold: 0.3392  # 置信度阈值（准确率不高时可降至0.3）
  
genetic_algorithm:
  population_size: 50      # 遗传算法种群大小
  generations: 30          # 进化代数
```

## 📈 结果解读

### 预测结果表格字段
- **Date**: 预测日期
- **Predicted**: AI预测结果（Yes/No）
- **Confidence**: 置信度（0.00-1.00）
- **Actual**: 实际结果（Yes/No）
- **Max Future Rise**: 未来最大涨幅（%）
- **Prediction Correct**: 预测是否正确

### 置信度解释
- **>0.8**: 高置信度，重点关注
- **0.5-0.8**: 中等置信度，结合其他指标
- **<0.5**: 低置信度，谨慎对待

## 🧪 测试验证

### 运行所有测试
```bash
python run.py t
```

### 测试特定功能
```bash
# 测试AI优化
python run.py ai -m demo

# 测试数据获取
python run.py d

# 测试配置加载
python run.py b -v
```

## 🚀 进阶用法

### 自定义配置文件
```bash
# Windows
set CSI_CONFIG_PATH=C:\path\to\custom.yaml
python run.py ai

# Linux/Mac
export CSI_CONFIG_PATH=/path/to/custom.yaml
python run.py ai
```

### 批处理脚本
```bash
# 创建 daily.bat (Windows)
@echo off
call venv\Scripts\activate
python run.py d
python run.py s %date:~0,10%
```

### 增加迭代次数
```bash
# 提高优化精度
python run.py opt -i 50
python run.py ai -v
```

## 📚 更多资源

- [**详细使用指南**](USER_GUIDE.md) - 完整操作手册
- [**项目介绍**](docs/项目介绍.md) - 技术架构详解
- [**run.py命令详解**](docs/run.py使用介绍.md) - 所有命令说明
- [**配置参数详解**](docs/策略参数介绍.md) - 参数调优指南
- [**算法原理**](docs/算法介绍和作用.md) - 核心算法介绍

## ⚠️ 注意事项

1. **必须使用虚拟环境**，否则可能出现依赖冲突
2. **第一次AI训练需要6-12分钟**，请耐心等待
3. **配置文件不要随意修改**，除非您了解参数含义
4. **网络问题可能影响数据获取**，多试几次即可
5. **本项目仅供学习研究**，不构成投资建议

---

🎯 **快速开始总结**：
1. 创建虚拟环境 → 2. 安装依赖 → 3. 基础测试 → 4. AI训练 → 5. 开始预测

💡 **遇到问题**：先查看 [USER_GUIDE.md](USER_GUIDE.md)，再查看具体文档。