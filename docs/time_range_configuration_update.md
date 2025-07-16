# 时间范围配置化更新文档

## 📅 更新概述

本次更新将系统中硬编码的时间范围改为配置文件管理，并调整了数据分割比例，提高了系统的灵活性和可配置性。

## 🔧 主要修改

### 1. 时间范围配置化

**修改前**：硬编码时间范围
- `run.py` 中硬编码：`'2022-01-01'` ~ `'2024-12-31'`
- 测试文件中硬编码：`'2023-01-01'` ~ `'2024-12-31'`

**修改后**：配置文件管理
- 新增 `config/system.yaml` 中的 `data.time_range` 配置
- 统一时间范围：`2019-01-01` ~ `2025-07-15`（6.5年完整数据）

### 2. 数据分割比例调整

**修改前**：65% / 20% / 15%
- 训练集：65%
- 验证集：20% 
- 测试集：15%

**修改后**：70% / 20% / 10%
- 训练集：70%（增加训练数据）
- 验证集：20%（保持不变）
- 测试集：10%（减少测试集，增强训练效果）

## 📊 配置文件变更

### config/system.yaml 新增配置

```yaml
data:
  # 数据时间范围配置
  time_range:
    start_date: "2019-01-01"       # 数据开始日期（包含约6.5年完整数据）
    end_date: "2025-07-15"         # 数据结束日期（当前日期）
    # 说明：此范围包含多个完整的市场周期，有利于模型训练和验证
    # 调整建议：开始日期不宜早于2015年，结束日期应为当前或最新交易日

ai:
  # 数据验证和分割配置
  validation:
    train_ratio: 0.70             # 训练集比例：70%（用于参数优化）
    validation_ratio: 0.20        # 验证集比例：20%（用于模型验证和过拟合检测）
    test_ratio: 0.10              # 测试集比例：10%（用于最终评估）
    # 说明：三个比例总和必须等于1.0
    # 训练集：较大比例确保充足的学习数据
    # 验证集：适中比例用于参数调优和过拟合检测
    # 测试集：较小比例用于最终无偏评估
    
    overfitting_threshold: 0.85   # 过拟合检测阈值
                                  # 验证集得分应至少为训练集得分的85%
    min_test_samples: 50          # 测试集最小样本数
                                  # 确保测试结果的统计意义
```

## 🔄 代码变更摘要

### 1. run.py
```python
# 修改前
backtest_config = config.get('backtest', {})
start_date = backtest_config.get('start_date', '2022-01-01')
end_date = backtest_config.get('end_date', '2024-12-31')

# 修改后
data_config = config.get('data', {})
time_range = data_config.get('time_range', {})
start_date = time_range.get('start_date', '2019-01-01')
end_date = time_range.get('end_date', '2025-07-15')
```

### 2. src/ai/ai_optimizer_improved.py
```python
# 修改前
train_ratio = validation_config.get('train_ratio', 0.65)
val_ratio = validation_config.get('validation_ratio', 0.2) 
test_ratio = validation_config.get('test_ratio', 0.15)

# 修改后
train_ratio = validation_config.get('train_ratio', 0.70)
val_ratio = validation_config.get('validation_ratio', 0.20) 
test_ratio = validation_config.get('test_ratio', 0.10)
```

### 3. tests/test_performance_anomaly_analysis.py
```python
# 修改前
data = data_module.get_history_data('2023-01-01', '2024-12-31')

# 修改后
data_config = config.get('data', {})
time_range = data_config.get('time_range', {})
start_date = time_range.get('start_date', '2019-01-01')
end_date = time_range.get('end_date', '2025-07-15')
data = data_module.get_history_data(start_date, end_date)
```

## 📈 更新效果验证

### 时间范围测试结果
```
📅 测试数据时间范围配置:
   开始日期: 2019-01-01  
   结束日期: 2025-07-15  
   ✅ 时间范围配置正确
```

### 数据分割测试结果
```
📊 测试数据分割比例配置:
   训练集比例: 70.0%     
   验证集比例: 20.0%     
   测试集比例: 10.0%     
   总和: 100.0%
   ✅ 数据分割比例配置正确
```

### 数据量对比

| 指标 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| **数据时间范围** | 2023-2024 (2年) | 2019-2025 (6.5年) | **+4.5年** |
| **总数据量** | 484条 | 1584条 | **+1100条** |
| **训练集** | 314条 (65%) | 1108条 (70%) | **+794条** |
| **验证集** | 97条 (20%) | 317条 (20%) | **+220条** |
| **测试集** | 73条 (15%) | 159条 (10%) | **+86条** |

## 🎯 新时间范围下的数据分布

### 时间分割结果
```
训练集: 2019-01-02 ~ 2023-07-26 (4.5年)
验证集: 2023-07-27 ~ 2024-11-18 (1.3年) 
测试集: 2024-11-19 ~ 2025-07-15 (0.7年)
```

### 市场环境分析
- **训练集**：经历完整牛熊周期，总收益率 +45.2%
- **验证集**：市场震荡期，总收益率 -2.1%
- **测试集**：近期市场，总收益率 +1.4%

## 💡 配置优势

### 1. 灵活性提升
- **可配置时间范围**：用户可根据需要调整数据范围
- **可配置分割比例**：支持不同的训练/验证/测试比例
- **向后兼容**：保留默认值，现有代码无需修改

### 2. 数据质量改善
- **更长时间序列**：6.5年数据包含更多市场周期
- **更多训练数据**：训练集从314条增加到1108条
- **更稳定测试**：测试集样本数量充足（159条 > 50条最小要求）

### 3. 维护性增强
- **消除硬编码**：所有时间配置集中管理
- **配置验证**：自动检查配置完整性和合理性
- **文档完备**：配置项都有详细注释说明

## 🔧 使用建议

### 1. 时间范围调整
```yaml
# 获取更多历史数据（如10年）
data:
  time_range:
    start_date: "2015-01-01"
    end_date: "2025-07-15"

# 或仅使用最近数据（如3年）
data:
  time_range:
    start_date: "2022-01-01" 
    end_date: "2025-07-15"
```

### 2. 分割比例调整
```yaml
# 增加训练数据比例
ai:
  validation:
    train_ratio: 0.80    # 80%训练
    validation_ratio: 0.15   # 15%验证
    test_ratio: 0.05     # 5%测试

# 增加测试数据比例  
ai:
  validation:
    train_ratio: 0.60    # 60%训练
    validation_ratio: 0.20   # 20%验证
    test_ratio: 0.20     # 20%测试
```

## ⚠️ 注意事项

1. **时间序列顺序**：必须保持训练→验证→测试的时间顺序
2. **比例总和**：三个分割比例总和必须等于1.0
3. **最小样本数**：确保测试集不少于50个样本
4. **数据可用性**：确保配置的时间范围内有足够的历史数据

## 🧪 测试验证

运行测试脚本验证配置：
```bash
# 验证时间范围和分割比例配置
python tests/test_config_time_range.py

# 验证性能表现
python tests/test_performance_anomaly_analysis.py
```

## 📚 相关文档

- [配置文件重组指南](config_reorganization_guide.md)
- [AI优化参数说明](ai_optimization_params.md)
- [项目使用指南](../USER_GUIDE.md)

---

**总结**：本次更新通过配置化时间范围和优化数据分割比例，显著提升了系统的灵活性和数据质量，为更好的模型训练和评估奠定了基础。 