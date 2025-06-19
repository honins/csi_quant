# 快速开始指南

## 🚀 5分钟快速上手

### 1. 解压项目

将 `csi1000_quant_final.zip` 解压到您的工作目录。

### 2. 安装依赖

在项目根目录下打开终端，运行：

```bash
pip install -r requirements.txt
```

### 3. 运行示例

- **运行基础回测**

  ```bash
  python run.py basic
  ```

- **运行AI优化回测**

  ```bash
  python run.py ai
  ```

- **单日低点预测**

  ```bash
  python predict_single_day.py <YYYY-MM-DD>
  # 示例：python predict_single_day.py 2024-06-01
  ```

- **LLM驱动策略优化**

  ```bash
  python llm_strategy_optimizer.py
  ```

### 4. 运行所有测试

```bash
python run.py all
```

## 📁 项目结构

```
csi1000_quant/
├── src/                    # 源代码
│   ├── data/              # 数据模块
│   ├── strategy/          # 策略模块
│   ├── ai/                # AI优化模块
│   ├── notification/      # 通知模块
│   └── utils/             # 工具模块
├── tests/                 # 单元测试
├── examples/              # 示例代码
├── config/                # 配置文件
├── docs/                  # 文档
├── .vscode/               # VS Code配置
├── cache/                 # 数据缓存
├── logs/                  # 日志文件
├── results/               # 结果文件
├── charts/                # 图表文件
├── models/                # AI模型文件
├── README.md              # 项目说明
├── requirements.txt       # 依赖列表
├── setup.py               # 安装脚本
├── run.py                 # 快速运行脚本
├── predict_single_day.py  # 单日预测脚本
└── llm_strategy_optimizer.py # LLM驱动策略优化脚本
```

## 🔧 VS Code 使用

### 打开项目

1. 启动VS Code
2. 选择"文件" -> "打开文件夹"
3. 选择解压后的 `csi1000_quant` 文件夹

### 运行和调试

项目已配置好VS Code的调试环境，您可以：

1. 按 `F5` 选择要运行的配置
2. 在代码中设置断点进行调试
3. 使用集成终端运行命令

### 可用的调试配置

- **运行基础测试**: 测试系统基本功能
- **运行AI优化测试**: 测试AI优化功能
- **运行快速脚本**: 使用命令行参数运行
- **调试单元测试**: 调试特定模块的测试

## 📊 功能说明

### 基础功能

- **数据获取**: 获取中证1000指数历史数据
- **技术指标**: 计算MA、RSI、MACD等技术指标
- **相对低点识别**: 根据定义识别相对低点
- **策略回测**: 评估策略历史表现
- **结果可视化**: 生成图表展示结果

### AI优化功能

- **参数优化**: 自动寻找最优策略参数
- **机器学习**: 训练模型预测相对低点
- **遗传算法**: 使用进化算法优化策略
- **特征重要性**: 分析哪些指标最重要

### 通知功能

- **控制台通知**: 在终端显示结果
- **邮件通知**: 发送邮件提醒（需配置）
- **通知历史**: 查看历史通知记录

## ⚙️ 配置说明

主要配置文件：`config/config.yaml`

### 关键配置项

```yaml
# 策略参数
strategy:
  rise_threshold: 0.05    # 上涨阈值（5%）
  max_days: 20           # 最大交易日数

# AI配置
ai:
  enable: true           # 是否启用AI优化
  model_type: "machine_learning"  # 模型类型

# 通知配置
notification:
  methods: ["console"]   # 通知方式
```

## 🧪 测试说明

### 运行单元测试

```bash
python run.py test
```

### 测试覆盖的模块

- **数据模块测试**: 测试数据获取和预处理
- **策略模块测试**: 测试相对低点识别和回测
- **AI模块测试**: 测试AI优化功能

## 📈 结果解读

### 策略评估指标

- **识别点数**: 识别出的相对低点数量
- **成功率**: 实际符合定义的比例
- **平均涨幅**: 相对低点后的平均涨幅
- **平均天数**: 达到目标涨幅的平均天数
- **综合得分**: 综合评估得分

### AI模型指标

- **准确率**: 预测正确的比例
- **精确率**: 预测为正样本中实际为正的比例
- **召回率**: 实际正样本中被预测为正的比例
- **F1得分**: 精确率和召回率的调和平均

## 🔍 常见问题

### Q: 如何修改相对低点的定义？

A: 修改 `config/config.yaml` 中的 `strategy.rise_threshold` 和 `strategy.max_days`。

### Q: 如何使用真实数据？

A: 需要集成真实的数据源API，修改 `src/data/data_module.py` 中的数据获取部分。

### Q: 如何添加新的技术指标？

A: 在 `src/data/data_module.py` 的 `_calculate_technical_indicators` 方法中添加。

### Q: 如何配置邮件通知？

A: 修改 `config/config.yaml` 中的 `notification.email` 配置，并在 `src/notification/notification_module.py` 中启用实际的邮件发送代码。

## 📞 技术支持

如果您在使用过程中遇到问题，请：

1. 查看 `logs/` 目录下的日志文件
2. 检查配置文件是否正确
3. 确保所有依赖都已正确安装
4. 参考 `docs/` 目录下的详细文档

## 🎯 下一步

1. **熟悉基础功能**: 运行基础测试，了解系统工作原理
2. **尝试AI优化**: 运行AI优化测试，体验智能优化功能
3. **自定义配置**: 根据需求调整配置参数
4. **集成真实数据**: 连接真实的数据源
5. **扩展功能**: 根据需要添加新的功能模块

祝您使用愉快！🎉

## 简化命令使用

### 基础命令

```bash
# 基础测试
python run.py b

# AI测试（包含训练和预测）
python run.py a

# 单元测试
python run.py t

# 回测
python run.py r

# 单日预测
python run.py s

# 策略优化
python run.py opt

# 高级AI优化（推荐）
python run.py ai

# 全部测试
python run.py all
```

### 高级AI优化

新增的 `ai` 命令提供了更强大的优化功能：

```bash
# 使用分层优化策略
python run.py ai

# 生成图表
python run.py ai plot
```

**分层优化特点：**
- 避免循环依赖问题
- 使用时间序列交叉验证
- 多目标优化（成功率、涨幅、速度、风险）
- 自动选择最佳优化方法

### 示例脚本

```bash
# 运行高级优化演示
python examples/advanced_optimization_demo.py

# 运行完整AI优化测试
python examples/ai_optimization_test.py

# 运行滚动回测
python examples/run_rolling_backtest.py
```

## AI模型训练与应用

### 1. 训练AI模型
```bash
# 运行AI优化测试，会训练并保存模型
python run.py a
```

### 2. 使用已训练模型进行预测
```bash
# 使用已训练模型预测单日
python run.py s 2024-06-01

# 或直接使用脚本
python examples/predict_single_day.py 2024-06-01
```

### 3. 强制重新训练模型
```bash
# 重新训练模型进行预测
python examples/predict_single_day.py 2024-06-01 --retrain
```

### 4. 模型文件位置
训练好的模型保存在 `models/` 目录下：
- `model_YYYYMMDD_HHMMSS.pkl`: 模型文件
- `features_YYYYMMDD_HHMMSS.json`: 特征名称文件
- `latest_model.txt`: 最新模型路径记录

## 命令对照表

| 简化命令 | 原命令 | 功能 |
|---------|--------|------|
| `b` | `basic` | 基础测试 |
| `a` | `ai` | AI优化测试 |
| `t` | `test` | 单元测试 |
| `r` | `rolling` | 回测 |
| `s` | `single` | 单日预测 |
| `opt` | `strategy` | 策略优化 |
| `all` | `all` | 全部运行 |

## 参数说明

- `-v`: 详细输出
- `-i` 或 `--iter`: 设置迭代次数 (默认: 10)
- 位置参数: 直接输入日期，无需 `--start-date` 等长参数

## 示例

```bash
# 快速运行基础测试
python run.py b

# 运行回测并查看详细输出
python run.py r 2023-01-01 2023-12-31 -v

# 运行策略优化，设置50次迭代
python run.py opt -i 50
```

## AI模型工作流程

### 完整流程
1. **训练阶段**: `python run.py a`
   - 训练AI模型
   - 保存模型到 `models/` 目录
   - 输出训练和验证结果

2. **应用阶段**: `python run.py s 2024-06-01`
   - 自动加载已训练模型
   - 使用模型进行预测
   - 输出预测结果和置信度

3. **验证阶段**: 查看预测结果与实际走势的对比

### 模型更新
- 每次运行 `python run.py a` 都会重新训练并保存新模型
- 新模型会自动覆盖旧模型路径记录
- 如需保留旧模型，请手动备份 `models/` 目录

### 注意事项
- 首次运行单日预测时，如果没有已训练模型，会自动重新训练
- 模型训练需要较长时间，建议先运行AI优化测试
- 模型质量取决于训练数据的质量和数量

## 环境准备

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 一键AI优化推荐

```bash
python run.py ai
```
- 自动分层优化，自动选择最佳参数和AI模型
- **参数自动持久化**：优化后的参数自动保存到配置文件
- **全局生效**：所有脚本自动使用优化后的参数
- 支持生成图表：`python run.py ai plot`

## 验证优化效果

```bash
# 查看优化后的参数是否生效
python examples/predict_single_day.py 2024-06-03
# 输出：策略模块初始化完成，参数: rise_threshold=0.0300, max_days=30
```

## 常用命令

| 命令         | 说明                       |
|--------------|----------------------------|
| b            | 基础策略测试               |
| a            | AI测试（含训练与预测）     |
| ai           | 高级AI优化（分层优化）     |
| r            | 回测                       |
| s            | 单日预测                   |
| opt          | 策略参数优化               |
| all          | 全部测试                   |

## 配置说明

- 编辑 `config/config.yaml` 可自定义数据、策略、AI、优化等参数
- `ai.advanced_optimization` 控制分层/高级优化开关
- `ai.genetic_algorithm` 控制遗传算法参数

## 常见问题

- **如何只用AI预测，不重新训练？**
  - 已有模型会自动加载，无需重复训练
- **优化后参数如何应用到其他脚本？**
  - `run.py ai`会自动将优化后的参数保存到配置文件，所有脚本都会自动使用
- **依赖缺失怎么办？**
  - 激活虚拟环境并`pip install -r requirements.txt`
- **优化效果如何？**
  - 分层优化可提升成功率77.7%，平均涨幅39.9%，综合得分33.0%
- **参数会丢失吗？**
  - 不会，优化后的参数会自动保存到配置文件，重启后仍然有效

---

详细字段、图表、评估体系说明请参见README和代码注释。


