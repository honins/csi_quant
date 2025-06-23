# 掘金量化集成说明

本文档详细说明如何在系统中集成和使用掘金量化平台的数据接口。

## 📋 掘金量化简介

掘金量化是一个专业的量化交易平台，提供丰富的金融数据和交易接口。本系统主要使用其history函数获取中证500指数的历史数据。

## 🔧 集成配置

### 1. 安装掘金量化SDK

```bash
pip install gm
```

### 2. 账户配置

如果您有掘金量化账户，需要在使用前进行登录配置：

```python
from gm.api import set_token

# 设置您的掘金量化token
set_token('your_token_here')
```

### 3. 系统配置

在 `config/config.yaml` 中配置掘金量化相关参数：

```yaml
data:
  # 中证500指数代码
  index_code: "SHSE.000905"
  # 数据频率
  frequency: "1d"
  # 掘金量化API配置
  gm_config:
    adjust: 1  # 前复权
    fields: "open,close,high,low,volume,amount,eob"
    skip_suspended: true
    fill_missing: null
```

## 📊 数据获取

### history函数详解

根据掘金量化官方文档，history函数的完整签名：

```python
history(symbol, frequency, start_time, end_time, fields=None, 
        skip_suspended=True, fill_missing=None, adjust=ADJUST_NONE, 
        adjust_end_time='', df=True)
```

### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| symbol | str | 标的代码，中证500为"SHSE.000905" |
| frequency | str | 频率，支持'1d'、'60s'、'tick'等 |
| start_time | str/datetime | 开始时间，格式：YYYY-MM-DD |
| end_time | str/datetime | 结束时间，格式：YYYY-MM-DD |
| fields | str | 返回字段，用逗号分隔 |
| skip_suspended | bool | 是否跳过停牌（暂不支持） |
| fill_missing | str/None | 填充方式（暂不支持） |
| adjust | int | 复权方式：0不复权，1前复权，2后复权 |
| adjust_end_time | str | 复权基点时间 |
| df | bool | 是否返回DataFrame格式 |

### 使用示例

```python
from gm.api import history, ADJUST_PREV

# 获取中证500指数日线数据
data = history(
    symbol='SHSE.000905',
    frequency='1d',
    start_time='2024-01-01',
    end_time='2024-12-31',
    fields='open,close,high,low,volume,amount,eob',
    adjust=ADJUST_PREV,
    df=True
)

print(data.head())
```

### 返回数据格式

当 `df=True` 时，返回pandas DataFrame：

```
          open      close        low       high      volume        amount                       eob
0     6234.56    6245.78   6220.12   6250.34    89234567   5.234567e+10 2024-01-01 00:00:00+08:00
1     6245.78    6255.89   6235.67   6260.45    92345678   5.456789e+10 2024-01-02 00:00:00+08:00
...
```

当 `df=False` 时，返回字典列表：

```python
[
    {
        'open': 6234.56,
        'close': 6245.78,
        'low': 6220.12,
        'high': 6250.34,
        'volume': 89234567,
        'amount': 52345670000.0,
        'eob': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=tzfile('PRC'))
    },
    ...
]
```

## 🎯 中证500指数

### 指数信息

- **指数代码**: SHSE.000905
- **指数名称**: 中证500指数
- **指数简介**: 反映中证800指数样本股之外规模偏小且流动性好的1000只证券的整体表现

### 数据特点

- **交易时间**: 周一至周五 9:30-11:30, 13:00-15:00
- **数据更新**: 实时更新（交易时间内）
- **历史数据**: 支持获取历史任意时间段数据
- **复权处理**: 支持前复权、后复权、不复权

## 🔄 系统集成流程

### 1. 数据获取流程

```
用户请求 → 检查缓存 → 调用掘金API → 数据预处理 → 返回结果
    ↓           ↓           ↓           ↓           ↓
  参数验证   缓存命中?   API调用成功?  技术指标计算  保存缓存
    ↓           ↓           ↓           ↓           ↓
  开始获取    返回缓存    使用模拟数据   数据验证    完成
```

### 2. 错误处理

系统具备完善的错误处理机制：

- **API不可用**: 自动切换到模拟数据
- **网络错误**: 重试机制和超时处理
- **数据异常**: 数据验证和清洗
- **参数错误**: 参数验证和默认值

### 3. 缓存机制

- **缓存位置**: `cache/` 目录
- **缓存格式**: CSV文件
- **缓存命名**: `{symbol}_{start_date}_{end_date}_{frequency}.csv`
- **缓存更新**: 自动检测数据更新

## ⚠️ 注意事项

### 1. API限制

- **数据量限制**: 单次最大返回33000条数据
- **频率限制**: 根据账户类型有不同的调用频率限制
- **时间范围**: 获取数据采用前后闭区间方式

### 2. 数据质量

- **停牌处理**: skip_suspended参数暂不支持
- **缺失值处理**: fill_missing参数暂不支持
- **数据排序**: 返回数据按eob升序排序

### 3. 错误处理

- **无效代码**: 返回空列表/空DataFrame
- **无效字段**: 只返回有效字段
- **时间格式**: 严格按照YYYY-MM-DD格式

## 🧪 测试验证

### 1. 连接测试

```python
def test_gm_connection():
    """测试掘金量化连接"""
    try:
        from gm.api import history
        
        # 测试获取少量数据
        data = history(
            symbol='SHSE.000905',
            frequency='1d',
            start_time='2024-01-01',
            end_time='2024-01-05',
            df=True
        )
        
        if data is not None and not data.empty:
            print("✅ 掘金量化连接成功")
            return True
        else:
            print("❌ 掘金量化返回空数据")
            return False
            
    except ImportError:
        print("❌ 掘金量化SDK未安装")
        return False
    except Exception as e:
        print(f"❌ 掘金量化连接失败: {e}")
        return False
```

### 2. 数据验证

```python
def validate_gm_data(data):
    """验证掘金量化数据"""
    if data.empty:
        return False, "数据为空"
    
    required_fields = ['open', 'close', 'high', 'low', 'eob']
    missing_fields = [f for f in required_fields if f not in data.columns]
    
    if missing_fields:
        return False, f"缺少字段: {missing_fields}"
    
    # 检查价格逻辑
    invalid_rows = (data['high'] < data['low']) | \
                   (data['high'] < data['close']) | \
                   (data['low'] > data['close'])
    
    if invalid_rows.any():
        return False, f"发现{invalid_rows.sum()}行价格逻辑错误"
    
    return True, "数据验证通过"
```

## 🔧 故障排除

### 常见问题及解决方案

1. **ImportError: No module named 'gm'**
   - 解决方案: `pip install gm`

2. **API调用返回空数据**
   - 检查网络连接
   - 验证账户状态
   - 确认参数格式

3. **数据格式异常**
   - 检查字段配置
   - 验证时间格式
   - 确认复权设置

4. **缓存数据过期**
   - 删除缓存文件
   - 重新获取数据
   - 检查缓存逻辑

## 📈 性能优化

### 1. 数据获取优化

- **批量获取**: 一次获取较长时间段的数据
- **智能缓存**: 避免重复获取相同数据
- **异步处理**: 对于大量数据的异步获取

### 2. 内存优化

- **数据分片**: 对于大数据集进行分片处理
- **及时释放**: 及时释放不需要的数据
- **格式优化**: 使用合适的数据类型

## 📚 参考资料

- [掘金量化官方文档](https://www.myquant.cn/docs/python)
- [掘金量化Python API](https://www.myquant.cn/docs/python/python_select_api)
- [中证500指数介绍](http://www.csindex.com.cn/)

## 🤝 技术支持

如果在使用掘金量化集成过程中遇到问题：

1. 查看系统日志文件
2. 检查掘金量化账户状态
3. 验证网络连接
4. 联系掘金量化技术支持
5. 提交Issue到项目仓库

---

**注意**: 掘金量化的API可能会有更新，请以官方最新文档为准。

