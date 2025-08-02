# optimize_parameters 参数优化功能说明文档

## 概述

`optimize_parameters` 是 CSI Quant 项目中的核心参数优化功能，位于 `src/ai/parameter_optimizer.py` 文件中的 `ParameterOptimizer` 类。该功能提供了多种优化算法来自动寻找策略的最佳参数组合，以提升交易策略的表现。

## 功能特性

### 🎯 支持的优化算法

1. **网格搜索 (Grid Search)**
   - 系统性地遍历所有参数组合
   - 适用于参数空间较小的情况
   - 保证找到全局最优解（在搜索范围内）

2. **随机搜索 (Random Search)**
   - 随机采样参数组合进行评估
   - 适用于高维参数空间
   - 计算效率高，适合快速探索

3. **贝叶斯优化 (Bayesian Optimization)**
   - 基于高斯过程的智能优化算法
   - 利用历史评估结果指导下一步搜索
   - 在有限迭代次数内找到较优解
   - 需要安装 `scikit-optimize` 库

4. **自适应随机搜索 (Adaptive Random Search)**
   - 贝叶斯优化的备选方案
   - 根据历史表现调整参数生成策略
   - 无需额外依赖库

## 方法签名

```python
def optimize_parameters(self, 
                       strategy_module,
                       data,
                       param_ranges: Dict[str, Any],
                       method: str = 'random',
                       max_iterations: int = 100) -> Dict[str, Any]:
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `strategy_module` | 策略模块实例 | - | 需要优化的策略模块对象 |
| `data` | DataFrame | - | 用于回测的历史数据 |
| `param_ranges` | Dict[str, Any] | - | 参数搜索范围配置 |
| `method` | str | 'random' | 优化方法：'grid', 'random', 'bayesian' |
| `max_iterations` | int | 100 | 最大迭代次数（网格搜索除外） |

### 参数范围配置格式

```python
param_ranges = {
    'parameter_name': {
        'min': 0.1,      # 最小值
        'max': 0.9,      # 最大值
        'type': 'float'  # 参数类型：'float' 或 'int'
    },
    # 更多参数...
}
```

### 返回值

返回一个包含优化结果的字典：

```python
{
    'success': bool,                    # 优化是否成功
    'method': str,                      # 使用的优化方法
    'best_params': Dict[str, Any],      # 最佳参数组合
    'best_score': float,                # 最佳得分
    'best_metrics': Dict[str, Any],     # 最佳参数的详细指标
    'iterations': int,                  # 实际迭代次数
    'improvements': int,                # 改进次数
    'optimization_time': float,         # 优化耗时（秒）
    'timestamp': str,                   # 优化完成时间戳
    'error': str                        # 错误信息（如果失败）
}
```

## 评分机制

### 评分指标

优化器使用多维度评分机制来评估参数组合的表现：

1. **成功率得分** (`success_rate`)
   - 权重：默认 40%
   - 计算：预测成功的比例

2. **平均涨幅得分** (`avg_rise`)
   - 权重：默认 30%
   - 计算：相对于基准涨幅（4%）的表现

3. **平均天数得分** (`avg_days`)
   - 权重：默认 20%
   - 计算：持仓天数越少得分越高

4. **风险惩罚** (`risk_penalty`)
   - 权重：默认 10%
   - 计算：信号数量过少时的惩罚

### 得分计算公式

```python
total_score = (success_rate * 0.4) + 
              (min(avg_rise/0.04, 2.0) * 0.3) + 
              (max(0, (20-avg_days)/20) * 0.2) - 
              (risk_penalty * 0.1)
```

## 使用示例

### 基本用法

```python
from src.ai.parameter_optimizer import ParameterOptimizer

# 初始化优化器
optimizer = ParameterOptimizer()

# 定义参数搜索范围
param_ranges = {
    'confidence_threshold': {'min': 0.1, 'max': 0.9, 'type': 'float'},
    'volume_threshold': {'min': 1.0, 'max': 5.0, 'type': 'float'},
    'lookback_days': {'min': 5, 'max': 30, 'type': 'int'}
}

# 执行随机搜索优化
result = optimizer.optimize_parameters(
    strategy_module=my_strategy,
    data=historical_data,
    param_ranges=param_ranges,
    method='random',
    max_iterations=200
)

# 检查结果
if result['success']:
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳得分: {result['best_score']:.4f}")
    print(f"优化耗时: {result['optimization_time']:.2f}秒")
else:
    print(f"优化失败: {result['error']}")
```

### 贝叶斯优化示例

```python
# 使用贝叶斯优化（需要安装 scikit-optimize）
result = optimizer.optimize_parameters(
    strategy_module=my_strategy,
    data=historical_data,
    param_ranges=param_ranges,
    method='bayesian',
    max_iterations=100
)
```

### 网格搜索示例

```python
# 网格搜索（适用于参数空间较小的情况）
result = optimizer.optimize_parameters(
    strategy_module=my_strategy,
    data=historical_data,
    param_ranges=param_ranges,
    method='grid'
    # max_iterations 参数对网格搜索无效
)
```

## 配置选项

### 贝叶斯优化配置

可以通过修改 `bayesian_config` 来调整贝叶斯优化的行为：

```python
optimizer.bayesian_config = {
    'n_calls': 120,              # 总调用次数
    'n_initial_points': 25,      # 初始随机点数量
    'acq_func': 'EI',           # 采集函数：'EI', 'PI', 'LCB'
    'xi': 0.01,                 # 探索参数
    'kappa': 1.96,              # 置信参数
    'random_state': 42          # 随机种子
}
```

### 评分权重配置

可以调整评分权重来适应不同的优化目标：

```python
optimizer.scoring_weights = {
    'success_rate': 0.5,    # 更重视成功率
    'avg_rise': 0.3,        # 涨幅权重
    'avg_days': 0.1,        # 天数权重
    'risk_penalty': 0.1     # 风险惩罚权重
}
```

## 性能优化建议

### 1. 选择合适的优化方法

- **参数数量 < 5，搜索空间小**：使用网格搜索
- **参数数量 5-15，中等搜索空间**：使用随机搜索
- **参数数量 > 15，大搜索空间**：使用贝叶斯优化

### 2. 合理设置迭代次数

- **随机搜索**：100-500 次迭代
- **贝叶斯优化**：50-200 次迭代
- **网格搜索**：自动计算，注意组合数量

### 3. 数据量要求

- 最少需要 100 条历史数据
- 推荐使用 500+ 条数据以获得稳定结果
- 数据质量比数量更重要

### 4. 参数范围设置

- 基于业务经验设置合理的参数范围
- 避免过大的搜索空间导致效率低下
- 可以先用粗粒度搜索，再细化范围

## 输出和日志

### 优化过程日志

优化器会输出详细的进度信息：

```
[INFO] 开始参数优化，方法: random，最大迭代: 200
[INFO] 随机搜索进度: 10.0% (20/200), 已用时: 15.3秒
[INFO] 发现更好参数 (第3次改进, 迭代45): 得分=7.2341
[INFO] 随机搜索进度: 50.0% (100/200), 已用时: 78.1秒
[INFO] 优化完成，最佳得分: 7.8923，总耗时: 156.7秒
```

### 结果保存

优化结果会自动保存到：
- **文件位置**：`results/optimization_{method}_{timestamp}.json`
- **历史记录**：内存中保留最近的优化历史
- **最佳参数**：自动更新全局最佳参数记录

## 错误处理

### 常见错误及解决方案

1. **数据验证失败**
   ```
   错误：输入验证失败: ['数据量不足，需要至少100条，实际50条']
   解决：增加历史数据量或降低最小数据要求
   ```

2. **参数配置错误**
   ```
   错误：参数 confidence_threshold 缺少 max 配置
   解决：检查 param_ranges 配置格式
   ```

3. **贝叶斯优化依赖缺失**
   ```
   警告：scikit-optimize未安装，回退到自适应随机搜索
   解决：pip install scikit-optimize
   ```

4. **策略模块接口不兼容**
   ```
   错误：strategy_module 缺少必要的方法
   解决：确保策略模块实现了 update_params, backtest, evaluate_strategy 方法
   ```

## 扩展和自定义

### 添加新的优化算法

1. 在 `optimize_parameters` 方法中添加新的分支
2. 实现对应的私有方法（如 `_custom_optimization`）
3. 遵循统一的返回格式

### 自定义评分函数

重写 `_calculate_score` 方法来实现自定义的评分逻辑：

```python
def _calculate_score(self, metrics: Dict[str, Any]) -> float:
    # 自定义评分逻辑
    custom_score = your_scoring_function(metrics)
    return custom_score
```

### 添加新的评估指标

在策略模块的 `evaluate_strategy` 方法中添加新的指标，优化器会自动使用这些指标进行评分。

## 最佳实践

1. **渐进式优化**：先用粗粒度参数范围快速探索，再细化优化
2. **交叉验证**：使用不同时间段的数据验证优化结果的稳定性
3. **参数合理性检查**：优化后的参数应该符合业务逻辑
4. **性能监控**：关注优化过程的内存和CPU使用情况
5. **结果分析**：不仅关注最佳得分，还要分析参数分布和改进趋势

## 相关文件

- **主要实现**：`src/ai/parameter_optimizer.py`
- **参数配置**：`src/utils/param_config.py`
- **优化结果保存**：`src/utils/optimized_params_saver.py`
- **配置文件**：`config/optimized_params.yaml`
- **使用示例**：`examples/` 目录下的相关文件

---

*本文档最后更新时间：2025年1月*