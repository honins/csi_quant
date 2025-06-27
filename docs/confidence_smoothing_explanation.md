# 置信度平滑机制说明

## 什么是置信度平滑？

置信度平滑是一种技术手段，用于减少AI模型预测置信度的剧烈波动，通过时间序列平滑算法来稳定置信度输出。

## 主要平滑方法

### 1. 移动平均平滑
```python
# 3日移动平均
smoothed_confidence = (confidence_t + confidence_t-1 + confidence_t-2) / 3

# 指数移动平均（EMA）
smoothed_confidence = alpha * confidence_t + (1-alpha) * smoothed_confidence_t-1
```

### 2. 最大变化幅度限制
```python
# 限制单日最大变化为±0.2
max_daily_change = 0.2
if abs(new_confidence - old_confidence) > max_daily_change:
    if new_confidence > old_confidence:
        new_confidence = old_confidence + max_daily_change
    else:
        new_confidence = old_confidence - max_daily_change
```

### 3. 置信度区间平滑
```python
# 根据置信度区间设置不同的平滑强度
if confidence < 0.3:  # 低置信度区间，较少平滑
    smooth_factor = 0.1
elif confidence > 0.7:  # 高置信度区间，较少平滑
    smooth_factor = 0.1
else:  # 中等置信度区间，较多平滑
    smooth_factor = 0.3
```

## 平滑机制的作用

### 1. **减少噪音影响**
- 过滤掉由于数据噪音导致的异常波动
- 避免单日异常数据对模型判断的过度影响

### 2. **提高决策稳定性**
- 防止频繁的买卖信号
- 减少追涨杀跌的风险

### 3. **增强实用性**
- 使AI预测结果更适合实际交易应用
- 提供更可靠的决策依据

### 4. **风险控制**
- 避免极端置信度变化带来的决策风险
- 保持策略的一致性

## 应用场景

### ✅ 适用情况
- 市场波动较大时
- 模型对短期噪音敏感时
- 需要稳定交易信号时

### ❌ 不适用情况
- 市场出现重大突发事件时
- 需要快速响应市场变化时
- 明确的趋势转折点时

## 参数调节原则

1. **平滑强度**：根据市场波动性调整
2. **时间窗口**：通常3-7个交易日
3. **变化限制**：一般设置为±0.15-0.25
4. **自适应调整**：根据市场状态动态调整平滑参数 