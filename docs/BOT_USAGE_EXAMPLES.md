# 🤖 增强版交易机器人使用示例

## 🚀 快速体验

### 1. 安装依赖
```bash
pip install psutil GitPython schedule
```

### 2. 单次运行测试
```bash
# 运行单次交易流程
python run.py bot -m run
```

### 3. 查看状态
```bash
# 查看机器人状态
python run.py bot -m status
```

### 4. 健康检查
```bash
# 系统健康检查
python run.py bot -m health
```

## 🛠️ 生产部署

### Linux/Mac 用户

```bash
# 启动守护进程
./scripts/bot_daemon.sh start

# 查看运行状态
./scripts/bot_daemon.sh status

# 查看实时日志
./scripts/bot_daemon.sh logs

# 停止守护进程
./scripts/bot_daemon.sh stop
```

### Windows 用户

```cmd
:: 启动守护进程
scripts\bot_daemon.bat start

:: 查看运行状态
scripts\bot_daemon.bat status

:: 查看实时日志
scripts\bot_daemon.bat logs

:: 停止守护进程
scripts\bot_daemon.bat stop
```

## 📊 监控管理

### 状态监控
```bash
# 详细状态报告
python run.py bot -m status
```

输出示例：
```
📊 增强版机器人状态报告:
==================================================
🤖 机器人状态: 正常
📅 运行开始: 2024-12-01 09:30:00
⏱️ 运行时长: 2天 3小时 15分钟

📊 执行统计:
   总预测次数: 45
   成功预测: 42
   训练次数: 3
   数据拉取次数: 5
   备份次数: 1
   连续错误: 0

🕐 最后执行时间:
   训练: 2024-12-01 09:30:00
   预测: 2024-12-01 15:05:00
   数据拉取: 2024-12-01 15:05:00
   备份: 2024-11-30 02:00:00

🏥 系统健康检查:
   整体状态: ✅ 健康
   CPU使用率: 15.2%
   内存使用率: 45.8%
   磁盘使用率: 23.1%
==================================================
```

### 健康检查
```bash
# 系统健康检查
python run.py bot -m health
```

输出示例：
```
🏥 执行系统健康检查...

系统健康状态: ✅ 健康
检查时间: 2024-12-01 16:30:00

📊 系统指标:
   CPU使用率: 15.2%
   内存使用率: 45.8%
   磁盘使用率: 23.1%

✅ 未发现问题
```

## 💾 数据管理

### 备份操作
```bash
# 手动备份
python run.py bot -m backup
```

输出示例：
```
💾 执行手动数据备份...
✅ 备份完成！
   备份路径: results/backup/backup_20241201_163000
   备份大小: 156.7MB
   备份项目: 8个
```

### 恢复操作
```bash
# 查看备份列表
ls results/backup/

# 从备份恢复
python run.py bot -m restore --backup-timestamp 20241201_163000
```

输出示例：
```
🔄 从备份恢复数据: 20241201_163000
✅ 恢复完成！
   恢复项目: data, models, config, results, logs
```

## 📅 定时任务

机器人会自动执行以下定时任务：

| 时间 | 任务 | 说明 |
|------|------|------|
| 每天 15:05 | 数据拉取 | 自动获取最新数据并提交到Git |
| 每天 09:30 | 日常交易流程 | 执行预测、信号生成、结果记录 |
| 每天 01:00 | 健康检查 | 系统性能监控和告警 |
| 每周日 02:00 | 数据备份 | 自动备份重要数据 |
| 每小时 | 性能指标收集 | 收集系统性能数据 |

## 🔧 配置示例

### 基础配置
```yaml
# config/config_core.yaml
bot:
  daemon_mode: true
  auto_data_fetch: true
  backup_interval_days: 7
  health_check_interval_hours: 1
  
  # 定时任务配置
  schedules:
    data_fetch: "15:05"      # 数据拉取时间
    daily_workflow: "09:30"  # 日常交易流程
    health_check: "01:00"    # 健康检查
    backup: "02:00"          # 备份（每周日）
```

### 高级配置
```yaml
# 性能监控配置
monitoring:
  cpu_threshold: 80          # CPU使用率告警阈值
  memory_threshold: 85       # 内存使用率告警阈值
  disk_threshold: 90         # 磁盘使用率告警阈值
  error_threshold: 5         # 连续错误告警阈值
  
# 备份配置
backup:
  keep_count: 10             # 保留备份数量
  include_dirs:              # 备份目录
    - data
    - models
    - config
    - results
    - logs
```

## 🚨 故障排除

### 常见问题

1. **守护进程启动失败**
   ```bash
   # 检查依赖
   pip install psutil GitPython schedule
   
   # 检查虚拟环境
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **数据拉取失败**
   ```bash
   # 手动测试
   python run.py fetch
   
   # 检查网络连接
   ping api.example.com
   ```

3. **备份失败**
   ```bash
   # 检查磁盘空间
   df -h  # Linux/Mac
   dir    # Windows
   ```

### 调试模式
```bash
# 详细输出
python run.py bot -m run -v

# 查看日志
tail -f logs/enhanced_trading_bot.log
```

## 📈 性能优化

### 系统资源监控
```bash
# 查看性能指标
cat results/daily_trading/performance_metrics.json
```

### 日志分析
```bash
# 查看错误日志
grep "ERROR" logs/enhanced_trading_bot.log

# 查看性能日志
grep "performance" logs/enhanced_trading_bot.log
```

## 🎯 最佳实践

1. **首次部署**：先运行单次模式测试
2. **生产环境**：使用守护进程模式
3. **定期监控**：每天检查状态和健康
4. **数据安全**：定期备份重要数据
5. **日志管理**：定期清理旧日志文件

## 📞 技术支持

- 查看完整文档：[ENHANCED_BOT_GUIDE.md](ENHANCED_BOT_GUIDE.md)
- 查看使用指南：[USER_GUIDE.md](USER_GUIDE.md)
- 查看项目主页：[README.md](README.md)

---

**🎉 开始使用增强版交易机器人，享受全自动的交易体验！** 