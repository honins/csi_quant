# 🔧 故障排除指南

本指南帮助您快速诊断和解决中证500指数量化交易系统的常见问题。

## 🚨 紧急问题快速解决

### 系统无法启动

```bash
# 1. 检查虚拟环境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. 重新安装依赖
pip install -r requirements.txt

# 3. 检查Python版本
python --version  # 需要3.8+

# 4. 基础测试
python run.py basic
```

### AI优化失败

```bash
# 1. 重置所有参数
python reset_strategy_params.py --all --force

# 2. 清理缓存
rm -rf data/cache/*  # Linux/Mac
del data\cache\*     # Windows

# 3. 重新运行优化
python run.py ai
```

### 数据获取失败

```bash
# 1. 检查网络连接
ping baidu.com

# 2. 手动获取数据
python run.py fetch

# 3. 检查数据文件
ls -la data/  # Linux/Mac
dir data\     # Windows
```

## 📋 常见问题分类

### 🔧 环境和依赖问题

#### 问题1：ImportError - 模块找不到

**错误信息**：
```
ImportError: No module named 'pandas'
ModuleNotFoundError: No module named 'src'
```

**解决方案**：
```bash
# 确认虚拟环境已激活
which python  # 应该指向venv目录

# 重新安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 检查安装状态
pip list | grep pandas
pip list | grep scikit-learn
```

**预防措施**：
- 始终在虚拟环境中运行
- 定期更新依赖包
- 使用requirements.txt管理依赖

#### 问题2：Python版本不兼容

**错误信息**：
```
SyntaxError: invalid syntax
TypeError: unsupported operand type(s)
```

**解决方案**：
```bash
# 检查Python版本
python --version

# 如果版本低于3.8，升级Python
# 或创建新的虚拟环境
python3.9 -m venv venv39
source venv39/bin/activate
```

#### 问题3：内存不足

**错误信息**：
```
MemoryError
OSError: [Errno 12] Cannot allocate memory
```

**解决方案**：
```bash
# 检查内存使用
free -h  # Linux
top      # 查看进程内存使用

# 减少数据量
# 编辑config/system.yaml
data:
  train_years: 3  # 减少到3年
  batch_size: 500 # 减少批次大小
```

### 📊 数据相关问题

#### 问题4：数据文件不存在

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/SHSE.000905_1d.csv'
```

**解决方案**：
```bash
# 检查数据目录
ls -la data/

# 手动获取数据
python run.py fetch

# 如果仍然失败，检查网络
curl -I https://www.baidu.com
```

#### 问题5：数据格式错误

**错误信息**：
```
ValueError: could not convert string to float
KeyError: 'close'
```

**解决方案**：
```bash
# 删除损坏的数据文件
rm data/SHSE.000905_1d.csv

# 重新获取数据
python run.py fetch

# 检查数据格式
head -5 data/SHSE.000905_1d.csv
```

#### 问题6：数据时间范围不足

**错误信息**：
```
ValueError: Insufficient data for training
```

**解决方案**：
```yaml
# 编辑config/system.yaml
data:
  start_date: "2018-01-01"  # 扩大时间范围
  
ai_optimization:
  train_years: 4  # 减少训练年数
```

### 🤖 AI和模型问题

#### 问题7：模型文件不存在

**错误信息**：
```
FileNotFoundError: Model file not found
```

**解决方案**：
```bash
# 检查模型文件
ls -la models/
cat models/latest_improved_model.txt

# 重新训练模型
python run.py ai

# 检查模型路径
grep -r "models/" config/
```

#### 问题8：AI优化收敛失败

**错误信息**：
```
Warning: Optimization did not converge
```

**解决方案**：
```yaml
# 编辑config/strategy.yaml
ai_optimization:
  generations: 50        # 增加迭代次数
  population_size: 100   # 增加种群大小
  early_stopping_patience: 10  # 增加耐心值
```

#### 问题9：预测置信度过低

**现象**：所有预测的置信度都很低（<0.3）

**解决方案**：
```bash
# 1. 重置参数
python reset_strategy_params.py --all --force

# 2. 调整阈值
# 编辑config/strategy.yaml
confidence_weights:
  final_threshold: 0.2  # 降低阈值

# 3. 重新优化
python run.py ai
```

### ⚙️ 配置文件问题

#### 问题10：YAML格式错误

**错误信息**：
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解决方案**：
```bash
# 检查YAML语法
python -c "import yaml; yaml.safe_load(open('config/system.yaml'))"

# 重置配置文件
python reset_strategy_params.py --all --force

# 使用在线YAML验证器检查格式
```

#### 问题11：配置参数超出范围

**错误信息**：
```
ValueError: Parameter value out of range
```

**解决方案**：
```yaml
# 检查config/strategy.yaml中的optimization_ranges
optimization_ranges:
  rsi_oversold_threshold: [25, 35]  # 确保在合理范围内
  final_threshold: [0.1, 0.8]       # 确保范围合理
```

### 🔄 运行时问题

#### 问题12：程序卡死或运行缓慢

**现象**：程序长时间无响应

**解决方案**：
```bash
# 1. 检查系统资源
top
htop  # 如果已安装

# 2. 减少计算量
# 编辑config/system.yaml
ai_optimization:
  population_size: 30   # 减少种群大小
  generations: 10       # 减少代数

# 3. 启用详细日志
python run.py ai --verbose
```

#### 问题13：权限错误

**错误信息**：
```
PermissionError: [Errno 13] Permission denied
```

**解决方案**：
```bash
# 检查文件权限
ls -la config/
ls -la models/

# 修复权限
chmod 755 config/
chmod 644 config/*.yaml

# 确保目录可写
chmod 755 models/ results/ data/
```

## 🔍 诊断工具

### 系统健康检查

```bash
# 运行系统状态检查
python run.py status

# 检查配置
python run.py config

# 运行基础测试
python run.py test
```

### 日志分析

```bash
# 查看最新日志
tail -f logs/system.log

# 搜索错误信息
grep -i error logs/system.log
grep -i warning logs/system.log

# 按时间查看日志
grep "2024-08-02" logs/system.log
```

### 性能监控

```bash
# 启用性能监控
python run.py ai --verbose

# 内存使用监控
ps aux | grep python

# 磁盘空间检查
df -h
du -sh data/ models/ results/
```

## 🛠️ 高级故障排除

### 数据库重建

```bash
# 完全重置数据
rm -rf data/processed/
rm -rf data/cache/
mkdir -p data/processed data/cache

# 重新获取数据
python run.py fetch
```

### 配置重置

```bash
# 备份当前配置
cp config/strategy.yaml config/strategy_backup.yaml
cp config/system.yaml config/system_backup.yaml

# 完全重置
python reset_strategy_params.py --all --force

# 恢复备份（如需要）
cp config/strategy_backup.yaml config/strategy.yaml
```

### 环境重建

```bash
# 删除虚拟环境
rm -rf venv/

# 重新创建
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📞 获取帮助

### 自助诊断清单

- [ ] Python版本是否为3.8+
- [ ] 虚拟环境是否已激活
- [ ] 依赖包是否完整安装
- [ ] 配置文件格式是否正确
- [ ] 数据文件是否存在
- [ ] 磁盘空间是否充足
- [ ] 内存是否充足
- [ ] 网络连接是否正常

### 收集诊断信息

```bash
# 生成诊断报告
echo "=== 系统信息 ===" > diagnostic_report.txt
python --version >> diagnostic_report.txt
which python >> diagnostic_report.txt

echo "\n=== 依赖信息 ===" >> diagnostic_report.txt
pip list >> diagnostic_report.txt

echo "\n=== 配置信息 ===" >> diagnostic_report.txt
python run.py config >> diagnostic_report.txt

echo "\n=== 系统状态 ===" >> diagnostic_report.txt
python run.py status >> diagnostic_report.txt

echo "\n=== 错误日志 ===" >> diagnostic_report.txt
tail -50 logs/system.log >> diagnostic_report.txt
```

### 联系支持

如果问题仍未解决，请：

1. **GitHub Issues**：提交详细的问题描述
   - 包含错误信息
   - 附上诊断报告
   - 说明复现步骤

2. **问题模板**：
   ```
   **问题描述**：
   简要描述遇到的问题
   
   **环境信息**：
   - 操作系统：
   - Python版本：
   - 项目版本：
   
   **复现步骤**：
   1. 执行命令...
   2. 出现错误...
   
   **错误信息**：
   ```
   粘贴完整的错误信息
   ```
   
   **已尝试的解决方案**：
   列出已经尝试过的解决方法
   ```

3. **紧急问题**：对于严重影响使用的问题，可以标记为高优先级

---

**提示**：大多数问题都可以通过重置配置和重新安装依赖来解决。在寻求帮助前，请先尝试基础的故障排除步骤。