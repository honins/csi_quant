# 项目初始化指南

## 概述

本项目提供了自动化的初始化脚本，可以一键完成项目环境的搭建，包括虚拟环境创建、依赖安装、数据获取等。

## 快速开始

### Windows用户

1. **双击运行**：
   ```
   setup.bat
   ```

2. **命令行运行**：
   ```cmd
   python setup.py
   ```

### Linux/Mac用户

1. **添加执行权限**：
   ```bash
   chmod +x setup.sh
   ```

2. **运行脚本**：
   ```bash
   ./setup.sh
   ```

3. **或者直接运行Python脚本**：
   ```bash
   python3 setup.py
   ```

## 初始化流程

setup脚本会自动执行以下步骤：

### 1. Python版本检查
- 检查Python版本是否为3.8或更高
- 如果版本过低，会提示安装新版本

### 2. 虚拟环境创建
- 自动创建`venv`虚拟环境
- 如果虚拟环境已存在，会询问是否重新创建

### 3. 依赖安装
- 自动安装`requirements.txt`中的所有依赖包
- 包括akshare、pandas、numpy等核心依赖

### 4. 目录创建
- 自动创建项目所需的目录结构：
  - `data/` - 数据文件目录
  - `logs/` - 日志文件目录
  - `results/` - 结果文件目录
  - `models/` - 模型文件目录
  - `cache/` - 缓存文件目录
  - `docs/` - 文档目录

### 5. 配置文件检查
- 检查`config/config.yaml`配置文件是否存在
- 确保项目配置正确

### 6. 数据获取
- 自动获取000852和000905的最新交易数据
- 使用增量更新，不会覆盖历史数据

### 7. 项目测试
- 自动运行项目中的测试文件
- 验证项目功能是否正常

## 环境要求

### 系统要求
- **操作系统**：Windows 10+ / Linux / macOS
- **Python版本**：3.8或更高版本
- **内存**：建议4GB以上
- **磁盘空间**：建议2GB以上可用空间

### 网络要求
- 需要网络连接以下载依赖包
- 需要网络连接以获取交易数据

## 常见问题

### Q: 虚拟环境创建失败
**A**: 检查Python是否正确安装，确保有足够的磁盘空间和权限。

### Q: 依赖安装失败
**A**: 
1. 检查网络连接
2. 尝试使用国内镜像源：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

### Q: 数据获取失败
**A**: 
1. 检查网络连接
2. 确认akshare库安装成功
3. 检查是否有防火墙阻止

### Q: 权限错误
**A**: 
- Windows：以管理员身份运行
- Linux/Mac：使用sudo运行或检查目录权限

## 手动初始化

如果自动初始化失败，可以手动执行以下步骤：

### 1. 创建虚拟环境
```bash
# Windows
python -m venv venv

# Linux/Mac
python3 -m venv venv
```

### 2. 激活虚拟环境
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 获取数据
```bash
python src/data/fetch_latest_data.py
```

## 初始化后操作

初始化完成后，您可以：

1. **激活虚拟环境**：
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **运行项目**：
   ```bash
   python run.py
   ```

3. **获取最新数据**：
   ```bash
   python src/data/fetch_latest_data.py
   ```

4. **查看项目文档**：
   - `README.md` - 项目说明
   - `docs/` - 详细文档
   - `QUICKSTART.md` - 快速开始指南

## 自动化部署

### Windows计划任务
```cmd
schtasks /create /tn "ProjectSetup" /tr "python C:\path\to\setup.py" /sc once /st 09:00
```

### Linux Cron
```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天9点执行）
0 9 * * * cd /path/to/project && python3 setup.py
```

## 联系支持

如果遇到问题，请：
1. 查看错误日志
2. 检查网络连接
3. 确认Python版本
4. 联系开发团队 