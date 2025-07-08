# 日常交易机器人使用指南

## 🚀 快速开始

### Windows用户
```cmd
scripts\bot_daemon.bat start
```

### Linux/Mac用户
```bash
chmod +x scripts/bot_daemon.sh
# 启动守护进程
./scripts/bot_daemon.sh start
```

## 🎯 核心功能

### 增强版交易机器人特性
- ✅ **守护进程模式** - 24/7无人值守运行
- ✅ **自动数据更新** - 每天15:05自动拉取最新数据
- ✅ **性能监控** - 实时监控系统资源使用情况
- ✅ **健康检查** - 自动检测系统问题并告警
- ✅ **数据备份** - 自动备份重要数据
- ✅ **完整日志** - 详细记录所有操作

### 使用命令

#### 基础命令
```bash
# 启动守护进程
./scripts/bot_daemon.sh start

# 查看运行状态
./scripts/bot_daemon.sh status

# 停止守护进程
./scripts/bot_daemon.sh stop

# 重启守护进程
./scripts/bot_daemon.sh restart

# 查看实时日志
./scripts/bot_daemon.sh logs
```

#### 高级命令
```bash
# 系统健康检查
./scripts/bot_daemon.sh health

# 手动数据备份
./scripts/bot_daemon.sh backup

# 显示帮助信息
./scripts/bot_daemon.sh help
```

## 📊 定时任务配置

增强版机器人自动执行以下定时任务：
- **每天15:05** - 自动数据拉取和提交
- **每天09:30** - 日常交易流程执行  
- **每天01:00** - 系统健康检查
- **每周日02:00** - 数据备份
- **每小时** - 性能指标收集

## 📋 状态监控

### 查看详细状态
```bash
./scripts/bot_daemon.sh status
```

状态报告包含：
- 📊 执行统计
- 🕐 最后执行时间  
- 🏥 系统健康状态
- 💾 备份信息
- 📈 性能指标

### 日志文件位置
- 主日志：`logs/enhanced_trading_bot.log`
- 守护进程日志：`logs/daemon.log`
- 性能监控日志：`logs/performance_monitor.log`

## 🔧 故障排除

### 常见问题
1. **依赖包安装**
   ```bash
   pip install -r requirements.txt
   ```

2. **权限问题**
   ```bash
   chmod +x scripts/bot_daemon.sh
   ```

3. **查看错误日志**
   ```bash
   tail -f logs/enhanced_trading_bot.log
   ```

## 📚 更多信息

详细使用指南请参考：
- `ENHANCED_BOT_QUICKSTART.md` - 快速开始指南
- `ENHANCED_BOT_GUIDE.md` - 完整使用手册

---

**注意**：如果您之前使用旧版的 `start_trading_bot.sh/bat` 脚本，请迁移到新的增强版脚本以获得更好的功能和稳定性。 