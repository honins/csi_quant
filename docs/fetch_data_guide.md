# 数据获取脚本使用指南

## 概述

本脚本用于获取中证500指数（000905）和中证1000指数（000852）的最新数据，并保存到`data`目录下的CSV文件中。

## 功能特性

- 自动获取000852（中证1000指数）和000905（中证500指数）的最新数据
- 使用akshare数据源，数据可靠且实时
- 自动保存为CSV格式，与项目现有数据格式保持一致
- 支持日志记录，便于调试和监控
- 返回JSON格式的执行结果

## 安装依赖

### 1. 激活虚拟环境

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法一：直接运行脚本

```bash
python fetch_latest_data.py
```

### 方法二：运行模块脚本

```bash
python src/data/fetch_latest_data.py
```

## 输出文件

脚本运行后会在`data`目录下生成以下文件：

- `SHSE.000852_1d.csv` - 中证1000指数数据
- `SHSE.000905_1d.csv` - 中证500指数数据

## 数据格式

CSV文件包含以下列：

| 列名 | 说明 |
|------|------|
| index | 序号 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| volume | 成交量 |
| amount | 成交额 |
| date | 日期 |

## 返回结果

脚本执行完成后会返回JSON格式的结果：

```json
{
    "code": 200,
    "msg": "数据获取完成",
    "data": {
        "results": {
            "000852": true,
            "000905": true
        },
        "timestamp": "2024-01-01 12:00:00"
    }
}
```

## 错误处理

- 如果网络连接失败，脚本会记录错误信息并继续处理其他指数
- 如果某个指数数据获取失败，对应的结果会标记为`false`
- 所有错误信息都会记录在日志中

## 注意事项

1. 确保网络连接正常，脚本需要访问akshare数据源
2. 建议在交易时间后运行，以获取完整的数据
3. 脚本会自动创建`data`目录（如果不存在）
4. 每次运行都会覆盖现有的CSV文件

## 日志

脚本运行时会输出详细的日志信息，包括：
- 数据获取进度
- 成功/失败状态
- 错误信息
- 数据统计信息

## 配置

脚本会自动读取`config/config.yaml`配置文件，支持以下配置项：

- 数据源配置
- 日志配置
- 缓存配置

## 故障排除

### 常见问题

1. **ImportError: No module named 'akshare'**
   - 解决方案：运行 `pip install akshare`

2. **网络连接超时**
   - 解决方案：检查网络连接，稍后重试

3. **权限错误**
   - 解决方案：确保对`data`目录有写权限

4. **配置文件不存在**
   - 解决方案：确保`config/config.yaml`文件存在

### 调试模式

如需更详细的日志信息，可以修改脚本中的日志级别：

```python
logger.setLevel(logging.DEBUG)
```

## 自动化

可以将此脚本添加到定时任务中，实现自动数据更新：

### Windows 计划任务

```cmd
schtasks /create /tn "FetchData" /tr "python C:\path\to\fetch_latest_data.py" /sc daily /st 18:00
```

### Linux Cron

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天18:00执行）
0 18 * * * cd /path/to/project && python fetch_latest_data.py
```

## 联系支持

如有问题，请查看日志文件或联系开发团队。 