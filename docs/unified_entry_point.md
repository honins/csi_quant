# 统一入口点设计文档

## 概述

本文档描述了CSI1000量化交易系统的统一入口点设计，通过 `run.py` 脚本和增强版启动器提供一致的用户体验。

## 🚀 启动方式

### 1. 增强版启动器（推荐）

#### **Windows启动器 (`scripts\bot_daemon.bat`)：**
- 提供完整的守护进程管理功能
- 支持启动、停止、重启、状态查看
- 包含健康检查和备份功能
- 适合长期无人值守运行

#### **Linux/Mac启动器 (`scripts/bot_daemon.sh`)：**
- 兼容sh和bash的跨平台脚本
- 完整的进程生命周期管理
- 自动环境检测和依赖安装
- 支持实时日志查看

### 2. 直接命令行

通过 `run.py` 提供的统一接口：

```bash
# AI相关命令
python run.py ai -m train          # 训练模型
python run.py ai -m demo           # 演示预测
python run.py ai -m optimize       # 参数优化

# 预测命令
python run.py s 2025-06-28         # 单日预测
python run.py r 2025-06-01 2025-06-30  # 滚动回测

# 机器人命令
python run.py bot -m run           # 单次执行
python run.py bot -m daemon        # 守护进程模式
python run.py bot -m status        # 状态查看

# 数据命令
python run.py fetch               # 获取数据
```

## 📋 功能对比

| 功能特性 | 增强版启动器 | 直接命令行 |
|----------|-------------|------------|
| 守护进程管理 | ✅ | ✅ |
| 进程监控 | ✅ | ❌ |
| 健康检查 | ✅ | ✅ |
| 自动备份 | ✅ | ✅ |
| 实时日志 | ✅ | ❌ |
| 交互友好 | ✅ | ❌ |
| 脚本自动化 | ✅ | ✅ |

## 🎯 使用建议

### 日常用户
推荐使用**增强版启动器**：
```bash
# 一键启动所有功能
./scripts/bot_daemon.sh start

# 查看运行状态
./scripts/bot_daemon.sh status
```

### 开发者/高级用户
推荐使用**直接命令行**：
```bash
# 精确控制具体功能
python run.py ai -m optimize
python run.py s 2025-06-28
```

### CI/CD和自动化
推荐使用**直接命令行**：
```bash
# 脚本化执行
python run.py bot -m run --config production.yaml
```

## 📁 文件结构

```
scripts/
├── bot_daemon.sh      # Linux/Mac增强版启动器
├── bot_daemon.bat     # Windows增强版启动器
├── bot_core.py              # 增强版机器人核心
└── ...

run.py                                # 统一入口点
```

## 🔄 迁移指南

如果您之前使用旧版启动器（`start_trading_bot.sh/bat`），请按以下步骤迁移：

### 迁移步骤
1. **停止旧版进程**
2. **使用新版启动器**：
   ```bash
   # 旧版方式（已废弃）
   # ./scripts/start_trading_bot.sh
   
   # 新版方式（推荐）
   ./scripts/bot_daemon.sh start
   ```
3. **享受新功能**：守护进程管理、健康监控、自动备份等

### 新功能优势
- 🛡️ **更强的稳定性**：守护进程模式，自动故障恢复
- 📊 **完整的监控**：系统资源、健康状态实时监控
- 💾 **数据安全**：自动备份，支持快速恢复
- 📝 **详细日志**：完整的操作记录和错误追踪

## 📚 更多信息

详细使用说明请参考：
- `ENHANCED_BOT_QUICKSTART.md` - 快速开始指南
- `ENHANCED_BOT_GUIDE.md` - 完整功能手册
- `docs/daily_trading_guide.md` - 日常使用指南 