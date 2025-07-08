# 增强版交易机器人快速开始指南

## 🚀 概述

增强版交易机器人是一个全功能的自动化交易系统，支持无人值守运行、自动数据更新、性能监控和数据备份等功能。

## ✨ 新特性

### 🎯 核心功能
- ✅ **每天15:05自动拉取最新数据并提交** - 确保数据始终是最新的
- ✅ **守护进程模式常驻运行** - 无需手动干预，24/7运行
- ✅ **系统性能监控和告警** - 实时监控CPU、内存、磁盘使用情况
- ✅ **自动数据备份和恢复** - 自动备份重要数据，支持快速恢复
- ✅ **健康检查和故障恢复** - 自动检测问题并发送告警
- ✅ **完整的日志记录系统** - 详细记录所有操作日志

### 📅 定时任务
- **每天15:05** - 自动数据拉取和提交
- **每天09:30** - 日常交易流程执行
- **每天01:00** - 系统健康检查
- **每周日02:00** - 数据备份
- **每小时** - 性能指标收集

## 📦 依赖包安装

```bash
# 安装新增的依赖包
pip install psutil GitPython schedule

# 或者安装完整的依赖
pip install -r requirements.txt
```

## 🛠️ 快速开始

### 方法一：使用管理脚本（推荐）

#### Linux/Mac
```bash
# 启动守护进程
./scripts/start_enhanced_bot_daemon.sh start

# 查看状态
./scripts/start_enhanced_bot_daemon.sh status

# 查看日志
./scripts/start_enhanced_bot_daemon.sh logs

# 停止守护进程
./scripts/start_enhanced_bot_daemon.sh stop
```

#### Windows
```cmd
:: 启动守护进程
scripts\start_enhanced_bot_daemon.bat start

:: 查看状态
scripts\start_enhanced_bot_daemon.bat status

:: 查看日志
scripts\start_enhanced_bot_daemon.bat logs

:: 停止守护进程
scripts\start_enhanced_bot_daemon.bat stop
```

### 方法二：使用run.py命令

```bash
# 运行单次交易流程
python run.py bot -m run

# 启动守护进程模式
python run.py bot -m daemon --daemon

# 查看机器人状态
python run.py bot -m status

# 执行数据备份
python run.py bot -m backup

# 系统健康检查
python run.py bot -m health

# 从备份恢复数据
python run.py bot -m restore --backup-timestamp 20240101_120000
```

## 📊 监控和管理

### 状态查看
```bash
# 详细状态报告
python run.py bot -m status
```

状态报告包含：
- 📊 执行统计（预测次数、成功次数等）
- 🕐 最后执行时间
- 🏥 系统健康检查
- 💾 备份信息

### 日志管理
```bash
# 实时查看日志
./scripts/start_enhanced_bot_daemon.sh logs   # Linux/Mac
scripts\start_enhanced_bot_daemon.bat logs    # Windows

# 日志文件位置
logs/enhanced_trading_bot.log      # 主日志
logs/daemon.log                   # 守护进程日志
logs/performance_monitor.log      # 性能监控日志
```

### 备份管理
```bash
# 手动备份
python run.py bot -m backup

# 查看备份文件
ls results/backup/

# 恢复数据（需要指定时间戳）
python run.py bot -m restore --backup-timestamp 20240101_120000
```

## 🔧 配置说明

### 主要配置文件
- `config/config_core.yaml` - 核心系统配置
- `config/optimization.yaml` - 优化配置

### 关键配置项
```yaml
# 机器人配置示例
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

## 📁 文件结构

```
csi1000_quant/
├── scripts/
│   ├── daily_trading_bot.py              # 增强版机器人核心
│   ├── start_enhanced_bot_daemon.sh      # Linux/Mac守护进程管理
│   └── start_enhanced_bot_daemon.bat     # Windows守护进程管理
├── results/
│   ├── daily_trading/
│   │   ├── bot_state.json                # 机器人状态
│   │   ├── trading_history.json          # 交易历史
│   │   ├── performance_metrics.json      # 性能指标
│   │   └── enhanced_trading_bot.pid      # 进程ID文件
│   └── backup/                           # 备份目录
└── logs/
    ├── enhanced_trading_bot.log           # 主日志
    ├── daemon.log                        # 守护进程日志
    └── performance_monitor.log           # 性能监控日志
```

## 🚨 健康监控

### 自动监控项目
- **CPU使用率** - 超过80%告警
- **内存使用率** - 超过85%告警
- **磁盘使用率** - 超过90%告警
- **连续错误次数** - 超过5次告警
- **数据新鲜度** - 超过48小时告警

### 手动健康检查
```bash
python run.py bot -m health
```

## 📈 性能优化

### 系统资源监控
机器人会自动收集系统性能指标：
- CPU使用率
- 内存使用率
- 磁盘使用率
- 进程数量

### 性能指标文件
```bash
# 查看性能指标
cat results/daily_trading/performance_metrics.json
```

## 🔄 故障恢复

### 自动重启机制
- 守护进程支持自动重启
- 异常退出时自动记录错误日志
- 支持远程重启信号

### 手动重启
```bash
# 重启守护进程
./scripts/start_enhanced_bot_daemon.sh restart   # Linux/Mac
scripts\start_enhanced_bot_daemon.bat restart    # Windows
```

### 数据恢复
```bash
# 列出所有备份
ls results/backup/

# 恢复到指定备份
python run.py bot -m restore --backup-timestamp 20240101_120000
```

## 🎛️ 高级功能

### Git集成
- 自动提交数据更新到Git仓库
- 支持自定义提交信息
- 自动检测数据变更

### 通知系统
- 系统异常自动告警
- 交易信号通知
- 健康检查报告

### 备份策略
- 自动备份重要目录（data、models、config、results、logs）
- 保留最近10个备份
- 支持手动和自动备份

## 📞 故障排除

### 常见问题

1. **守护进程启动失败**
   ```bash
   # 检查依赖是否安装
   pip install -r requirements.txt
   
   # 检查虚拟环境
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **数据拉取失败**
   ```bash
   # 手动测试数据拉取
   python run.py fetch
   
   # 检查网络连接
   # 检查API配置
   ```

3. **备份失败**
   ```bash
   # 检查磁盘空间
   df -h  # Linux/Mac
   dir    # Windows
   
   # 检查目录权限
   ```

### 调试模式
```bash
# 使用详细输出
python run.py bot -m run -v

# 查看详细日志
tail -f logs/enhanced_trading_bot.log
```

## 📚 更多信息

- 查看完整文档：`docs/`
- 配置指南：`docs/config_reorganization_guide.md`
- API文档：`docs/usage_guide.md`
- 更新日志：`CHANGELOG.md`

## 🤝 支持

如果遇到问题，请：
1. 查看日志文件
2. 运行健康检查
3. 检查系统资源
4. 查阅文档

---

**🎉 现在你可以享受全自动的交易机器人服务了！** 