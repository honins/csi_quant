# 🔄 脚本迁移通知

## 📢 **重要更新**

**旧版启动脚本已被删除，请使用增强版脚本！**

### ❌ **已删除的文件**
- `scripts/start_trading_bot.sh` (Linux/Mac旧版)
- `scripts/start_trading_bot.bat` (Windows旧版)

### ✅ **新版增强脚本**
- `scripts/bot_daemon.sh` (Linux/Mac增强版)
- `scripts/bot_daemon.bat` (Windows增强版)
- `scripts/bot_core.py` (增强版机器人核心)

## 🚀 **快速迁移指南**

### Linux/Mac用户
```bash
# 旧版方式（已不可用）
# ./scripts/start_trading_bot.sh

# 新版方式（立即使用）
./scripts/bot_daemon.sh start
```

### Windows用户
```cmd
:: 旧版方式（已不可用）
:: scripts\start_trading_bot.bat

:: 新版方式（立即使用）
scripts\bot_daemon.bat start
```

## 🎯 **新版优势**

### 🛡️ **更强功能**
- ✅ **守护进程管理** - 完整的进程生命周期管理
- ✅ **健康监控** - 自动系统资源监控和告警
- ✅ **自动备份** - 定期数据备份和快速恢复
- ✅ **实时日志** - 详细的运行日志和错误追踪
- ✅ **性能优化** - 更高的稳定性和执行效率

### 📋 **可用命令**
```bash
# 启动守护进程
./scripts/bot_daemon.sh start

# 查看运行状态  
./scripts/bot_daemon.sh status

# 查看实时日志
./scripts/bot_daemon.sh logs

# 停止守护进程
./scripts/bot_daemon.sh stop

# 重启守护进程
./scripts/bot_daemon.sh restart

# 系统健康检查
./scripts/bot_daemon.sh health

# 手动数据备份
./scripts/bot_daemon.sh backup

# 显示帮助信息
./scripts/bot_daemon.sh help
```

## 📚 **详细文档**

获取完整使用指南：
- **快速开始**：`ENHANCED_BOT_QUICKSTART.md`
- **完整手册**：`ENHANCED_BOT_GUIDE.md`  
- **日常使用**：`docs/daily_trading_guide.md`

## 🤝 **需要帮助？**

如果您在迁移过程中遇到问题：

1. **查看文档**：上述三个文档文件包含详细说明
2. **检查日志**：`logs/enhanced_trading_bot.log`
3. **运行健康检查**：`./scripts/bot_daemon.sh health`
4. **查看状态**：`./scripts/bot_daemon.sh status`

---

## 🎉 **立即开始使用增强版！**

**新版本提供更强大、更稳定、更智能的交易机器人体验！**

```bash
# 一键启动，享受全新体验
./scripts/bot_daemon.sh start
``` 