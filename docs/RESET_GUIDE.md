# 🔄 参数重置和重新训练指南


### ✅ 参数重置：
- ✓ final_threshold: 0.618 → 0.5
- ✓ rsi_oversold_threshold: 34 → 30  
- ✓ rsi_low_threshold: 38 → 40
- ✓ dynamic_confidence_adjustment: 0.1 → 0.05
- ✓ market_sentiment_weight: 0.15 → 0.16
- ✓ trend_strength_weight: 0.12 → 0.16

### ✅ 清理的模型文件：
- 删除了所有旧的训练模型（*.pkl）
- 删除了特征文件（*.json）  
- 删除了模型指针文件（*.txt）
- 清理了缓存文件

## 🚀 重新训练步骤

### 1. 激活虚拟环境
```bash
# Windows
venv\Scripts\activate

# Linux/Mac  
source venv/bin/activate
```

### 2. 完整重训练（推荐）
```bash
python run.py ai -m full
```

### 3. 或使用增量训练模式
```bash
python run.py ai -m incremental
```

### 4. 验证训练效果
```bash
python run.py ai -m demo
```

## 📊 新的训练特性

- **严格数据分割**: 65%训练 / 20%验证 / 15%测试
- **防止数据泄露**: 测试集完全隔离
- **改进的AI优化器**: 统一使用 `ai_optimizer_improved.py`
- **置信度平滑**: 更稳定的预测结果

## 🎯 重训练后的预期效果

1. **模型性能**: 在干净的测试集上评估，结果更可靠
2. **参数优化**: 基于严格分割的数据进行优化
3. **减少过拟合**: 防止模型"记忆"测试数据

## ⚠️ 注意事项

- 重训练可能需要5-15分钟（取决于硬件配置）
- 新模型的表现可能与之前不同（这是正常的，因为之前可能存在数据泄露）
- 建议在重训练前备份重要的策略配置

## 🔍 训练完成后的验证

运行基础测试确认系统正常：
```bash
python run.py test
```

运行单日预测测试：
```bash
python run.py predict --date 2024-12-01
```

## 📁 项目文件结构（简化后）

```
src/
├── ai/
│   └── ai_optimizer_improved.py  # 唯一的AI优化器
├── data/
│   ├── data_module.py
│   └── fetch_latest_data.py
├── strategy/
│   └── strategy_module.py
├── utils/
│   ├── config_loader.py
│   ├── utils.py
│   └── trade_date.py
└── prediction/
    └── prediction_utils.py

examples/
├── basic_test.py
├── run_rolling_backtest.py
├── predict_single_day.py
└── llm_strategy_optimizer.py

scripts/
├── reset_parameters.py  # 参数重置工具
└── bot_core.py
```

这次清理让项目更加简洁和统一，所有组件现在都使用改进版的AI优化器，并且消除了数据泄露风险。

---

## 📋 `python run.py ai -m full` 完整执行流程解析

### 🚀 概述
这个命令执行**完全重训练模式**，从头开始训练AI模型，不依赖任何以前的训练结果。

### 📊 详细执行步骤

#### 1️⃣ **系统初始化阶段**
```bash
============================================================
中证500指数相对低点识别系统
============================================================
```
- ✅ 检查虚拟环境状态
- ✅ 加载配置文件（多配置文件支持）
- ✅ 初始化日志系统
- ✅ 启动性能计时器

#### 2️⃣ **AI模块初始化**
```bash
🤖 AI模型训练系统
============================================================
📋 训练配置:
   🎯 训练模式: full
```
- 📦 导入 `AIOptimizerImproved` 模块
- 🔧 初始化置信度平滑器 (`ConfidenceSmoother`)
- 📁 设置模型保存目录 (`models/`)
- ⚙️ 配置增量学习参数

#### 3️⃣ **数据准备阶段**
```bash
🔄 开始完全重训练...
```
- 📅 **数据时间范围**: 当前日期往前推 **6年** (365*6 = 2190天)
- 📊 **数据获取**: 调用 `data_module.get_history_data()`
- 🔄 **数据预处理**: 调用 `data_module.preprocess_data()`
- 📈 计算技术指标（均线、RSI、MACD、布林带等）

#### 4️⃣ **特征工程阶段** (`prepare_features_improved`)
**🎯 改进的特征准备:**

**长期趋势指标（高权重）:**
- `ma20`, `ma60` - 长期均线
- `trend_strength_20`, `trend_strength_60` - 趋势强度
- `price_position_20`, `price_position_60` - 价格在均线系统中的位置

**中期趋势指标（正常权重）:**
- `ma10`, `dist_ma10`, `dist_ma20` - 中期均线
- `rsi`, `macd`, `signal` - 技术指标
- `bb_upper`, `bb_lower` - 布林带
- `volatility_normalized` - 标准化波动率

**短期指标（降低权重）:**
- `ma5`, `dist_ma5`, `hist` - 短期指标
- `price_change_5d`, `price_change_10d` - 短期价格变化

**成交量指标（平衡权重）:**
- `volume_trend`, `volume_strength` - 成交量趋势

#### 5️⃣ **标签准备阶段** (`_prepare_labels`)
- 🎯 调用策略模块的 `backtest()` 方法
- 📊 生成"相对低点"标签 (0 或 1)
- ⚖️ 计算样本权重（时间衰减）

#### 6️⃣ **模型训练阶段** (`full_train`)
**🔄 数据分割:**
- 训练集：80% (约1200条样本，约4.8年)
- 测试集：20% (约300条样本，约1.2年)

**🤖 模型配置:**
```python
RandomForestClassifier(
    n_estimators=150,      # 150棵决策树
    max_depth=12,          # 最大深度12
    min_samples_split=8,   # 最小分割样本数
    min_samples_leaf=3,    # 最小叶子节点样本数
    class_weight='balanced', # 类别平衡
    warm_start=True,       # 支持增量学习
    n_jobs=-1             # 并行训练
)
```

**🔧 特征处理:**
- 📊 标准化处理 (`StandardScaler`)
- ⚖️ 应用时间衰减权重
- 🏗️ 构建 `Pipeline` (标准化 + 分类器)

#### 7️⃣ **模型保存阶段** (`_save_model`)
- 💾 保存到 `models/improved_model_YYYYMMDD_HHMMSS.pkl`
- 📝 更新 `models/latest_improved_model.txt` 指针
- 📋 保存模型元信息（特征名称、增量计数等）

#### 8️⃣ **结果输出阶段**
```bash
📊 训练结果:
   ✅ 训练状态: 成功
   📊 训练样本数: XXX
   🎯 模型准确率: X.XXXX
```

### 🎯 关键特性

#### 🔄 **严格数据分割**
- 65% 训练 / 20% 验证 / 15% 测试（在完整优化中）
- 防止数据泄露，确保测试集完全独立

#### 🧠 **改进的AI模型**
- **置信度平滑**: 避免预测结果的剧烈波动
- **动态权重调整**: 根据市场状况调整特征权重
- **增量学习支持**: 支持后续的增量更新

#### ⚡ **高效特征工程**
- **权重化特征**: 长期指标高权重，短期指标低权重
- **趋势确认指标**: 基于线性回归的趋势强度
- **标准化处理**: 确保不同量级特征的公平比较

### 💡 预期结果

**✅ 成功情况下的输出:**
- 新的模型文件保存到 `models/` 目录
- 重置增量训练计数器
- 可立即用于预测和增量学习

**⚠️ 可能的执行时间:**
- **6年数据集** (约1500条): 6-12分钟 **[当前配置]**
- **8年数据集** (约2000条): 8-15分钟  
- **10年数据集** (约2500条): 12-20分钟

### 🔧 配置参数影响

关键配置项会影响训练过程：
- `ai.training_data.full_train_years`: 完全训练数据年数（默认6年）
- `ai.training_data.optimize_years`: 优化模式数据年数（默认6年）  
- `ai.training_data.incremental_years`: 增量训练数据年数（默认1年）
- `ai.train_test_split_ratio`: 训练测试分割比例
- `ai.data_decay_rate`: 时间衰减率
- `ai.confidence_smoothing`: 置信度平滑配置
- `strategy.confidence_weights.final_threshold`: 最终预测阈值

### 📝 训练后的文件结构

训练完成后，`models/` 目录将包含：
```
models/
├── improved_model_20240701_123456.pkl  # 新训练的模型文件
├── latest_improved_model.txt           # 最新模型指针
└── confidence_history.json            # 置信度历史记录
```

这个完整的训练流程确保了模型的可靠性和性能，为后续的预测和增量学习奠定了坚实的基础。

## ⚙️ **训练数据范围配置**

### 📊 **为什么选择6年数据？**

根据实际使用考虑，6年是更平衡的选择：

1. **时效性优势**: 覆盖2018-2024年，更贴近当前市场环境
2. **样本充足性**: 约1500条样本，满足23个特征的统计要求（每特征65个样本）
3. **周期完整性**: 包含贸易摩擦、疫情、政策调整、经济复苏等完整周期
4. **训练效率**: 6-12分钟的训练时间，实用性更强

### 🔧 **如何调整数据范围？**

在 `config/config.yaml` 中修改：

```yaml
ai:
  training_data:
    full_train_years: 6     # 调整为你想要的年数（当前默认6年）
    optimize_years: 6       # 优化模式的年数（当前默认6年）
    incremental_years: 1    # 增量训练的年数
```

### 📈 **不同数据范围的适用场景**

- **6年**: **推荐配置**，平衡时效性和样本充足性
- **8年**: 追求更高稳健性，可容忍较长训练时间
- **10-12年**: 追求最大样本量，主要用于研究分析

详细分析请参考 `DATA_ANALYSIS.md` 文档。 