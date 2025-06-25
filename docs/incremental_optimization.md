# AI增量优化功能说明

## 概述

AI优化器现在支持增量优化功能，这意味着每次运行 `run.py ai` 时，系统会：

1. **检查历史参数**：自动加载之前优化好的参数
2. **智能选择模式**：根据是否有历史参数选择优化模式
3. **增量优化**：基于历史参数进行局部搜索，而不是每次都重新开始
4. **保存优化结果**：将每次优化的结果保存到历史记录中

## 🔄 优化模式对比

### 🌐 全局优化模式（首次运行）
- **触发条件**：没有历史参数文件
- **搜索策略**：从随机参数开始全局搜索
- **搜索范围**：使用配置文件中定义的完整搜索范围
- **迭代次数**：150次
- **适用场景**：首次运行、重置优化历史

### 🎯 增量优化模式（后续运行）
- **触发条件**：存在历史最优参数文件
- **搜索策略**：基于历史最优参数进行局部搜索
- **搜索范围**：以历史参数为中心，搜索范围收缩到原来的30%
- **迭代次数**：100次
- **适用场景**：日常优化、参数微调

## 📁 文件结构

### 参数历史文件
```
models/
├── parameter_history.json    # 参数优化历史记录
├── best_parameters.json      # 当前最优参数
├── latest_model.txt         # 最新模型路径
├── model_20250626_143022.pkl # AI模型文件
└── features_20250626_143022.json # 特征名称文件
```

### 文件内容示例

#### `best_parameters.json`
```json
{
  "timestamp": "2025-06-26T14:30:22.123456",
  "parameters": {
    "rise_threshold": 0.04,
    "max_days": 20,
    "rsi_oversold_threshold": 28,
    "rsi_low_threshold": 41,
    "final_threshold": 0.45,
    "dynamic_confidence_adjustment": 0.21,
    "market_sentiment_weight": 0.16,
    "trend_strength_weight": 0.16,
    "volume_weight": 0.25,
    "price_momentum_weight": 0.20
  },
  "score": 0.2597
}
```

#### `parameter_history.json`
```json
[
  {
    "timestamp": "2025-06-26T14:30:22.123456",
    "parameters": {
      "rise_threshold": 0.04,
      "max_days": 20,
      "rsi_oversold_threshold": 28,
      "rsi_low_threshold": 41,
      "final_threshold": 0.45,
      "dynamic_confidence_adjustment": 0.21,
      "market_sentiment_weight": 0.16,
      "trend_strength_weight": 0.16,
      "volume_weight": 0.25,
      "price_momentum_weight": 0.20
    },
    "score": 0.2597
  },
  {
    "timestamp": "2025-06-26T15:45:33.456789",
    "parameters": {
      "rise_threshold": 0.04,
      "max_days": 20,
      "rsi_oversold_threshold": 27,
      "rsi_low_threshold": 40,
      "final_threshold": 0.44,
      "dynamic_confidence_adjustment": 0.22,
      "market_sentiment_weight": 0.17,
      "trend_strength_weight": 0.15,
      "volume_weight": 0.26,
      "price_momentum_weight": 0.19
    },
    "score": 0.2612
  }
]
```

## 🚀 增量优化算法

### 搜索范围计算
```python
# 增量搜索范围计算逻辑
contraction_factor = 0.3  # 搜索范围收缩到原来的30%

for param_name, base_value in base_params.items():
    # 获取完整搜索范围
    min_val = param_range.get('min', 0)
    max_val = param_range.get('max', 1)
    
    # 计算增量搜索范围
    range_width = max_val - min_val
    incremental_width = range_width * contraction_factor
    
    # 以基础值为中心，向两边扩展
    new_min = max(min_val, base_value - incremental_width / 2)
    new_max = min(max_val, base_value + incremental_width / 2)
```

### 示例：RSI超卖阈值
- **完整范围**：25 - 35
- **历史最优值**：28
- **增量范围**：26.5 - 29.5（以28为中心，±1.5）

## 📊 优化效果对比

### 首次运行（全局优化）
```
🚀 开始AI策略参数优化
📋 阶段3: 检查历史参数...
🆕 没有历史参数，使用全局搜索模式
🌐 使用全局搜索范围:
   - rsi_oversold_threshold: 25 - 35, 步长: 1
   - rsi_low_threshold: 35 - 45, 步长: 1
   - final_threshold: 0.30 - 0.70, 步长: 0.05
🌐 全局优化模式，最大迭代次数: 150
```

### 后续运行（增量优化）
```
🚀 开始AI策略参数优化
📋 阶段3: 检查历史参数...
🔄 发现历史最优参数，启用增量优化模式
🎯 使用增量搜索范围（基于历史最优参数）:
   - rsi_oversold_threshold: 26.5 - 29.5 (基于 28.0)
   - rsi_low_threshold: 38.5 - 41.5 (基于 40.0)
   - final_threshold: 0.42 - 0.48 (基于 0.45)
🎯 历史最优参数作为起点，得分: 0.2597
🎯 增量优化模式，最大迭代次数: 100
```

## 🎯 使用建议

### 1. 首次使用
- 运行 `python run.py ai` 进行全局优化
- 系统会自动保存最优参数
- 优化时间较长（约2-3分钟）

### 2. 日常优化
- 定期运行 `python run.py ai` 进行增量优化
- 基于历史参数进行微调
- 优化时间较短（约1-2分钟）

### 3. 重置优化历史
如果需要重新开始全局优化：
```bash
# 删除历史参数文件
rm models/best_parameters.json
rm models/parameter_history.json

# 重新运行全局优化
python run.py ai
```

### 4. 监控优化效果
- 查看 `models/parameter_history.json` 了解优化历史
- 观察得分变化趋势
- 如果得分长期没有提升，考虑重置优化历史

## 🔧 配置选项

### 增量搜索收缩比例
在 `src/ai/ai_optimizer.py` 中可以调整：
```python
# 定义增量搜索的收缩比例
contraction_factor = 0.3  # 搜索范围收缩到原来的30%
```

### 历史记录保留数量
```python
# 只保留最近100条记录
if len(history) > 100:
    history = history[-100:]
```

## 📈 性能优势

### 时间效率
- **全局优化**：150次迭代，约2-3分钟
- **增量优化**：100次迭代，约1-2分钟
- **时间节省**：约40-50%

### 优化效果
- **全局优化**：探索完整参数空间，可能找到全局最优
- **增量优化**：基于历史最优进行微调，更稳定
- **适用场景**：增量优化适合日常维护，全局优化适合重大调整

## 🚨 注意事项

1. **参数文件权限**：确保 `models/` 目录有写入权限
2. **磁盘空间**：历史记录文件会逐渐增长，定期清理
3. **网络环境**：首次运行需要下载数据，确保网络连接
4. **内存使用**：大量历史记录可能影响内存使用

## 🔍 故障排除

### 问题：增量优化没有生效
**解决方案**：
1. 检查 `models/best_parameters.json` 文件是否存在
2. 确认文件格式正确
3. 查看日志输出确认优化模式

### 问题：优化效果不理想
**解决方案**：
1. 删除历史参数文件，重新进行全局优化
2. 调整配置文件中的参数搜索范围
3. 增加迭代次数

### 问题：历史记录文件损坏
**解决方案**：
1. 备份并删除损坏的文件
2. 重新运行全局优化
3. 检查文件权限和磁盘空间 