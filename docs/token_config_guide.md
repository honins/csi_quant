# 掘金量化Token配置指南

## 📋 概述

掘金量化token是访问掘金量化API的身份凭证。本系统支持多种token配置方式，您可以根据需要选择最适合的方式。

## 🔑 获取Token

### 1. 注册掘金量化账户

1. 访问掘金量化官网：https://www.myquant.cn/
2. 注册账户并完成实名认证
3. 登录到您的账户

### 2. 获取API Token

1. 登录后进入"个人中心"
2. 找到"API管理"或"开发者设置"页面
3. 生成新的API Token或查看现有Token
4. 复制Token字符串（通常是一长串字符）

## ⚙️ 配置方式

系统支持4种token配置方式，按优先级排序：

### 方式1: 环境变量（推荐）

#### Linux/macOS:
```bash
# 临时设置（当前会话有效）
export GM_TOKEN="your_gm_token_here"

# 永久设置（添加到~/.bashrc或~/.zshrc）
echo 'export GM_TOKEN="your_gm_token_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows:
```cmd
# 临时设置
set GM_TOKEN=your_gm_token_here

# 永久设置（通过系统设置）
setx GM_TOKEN "your_gm_token_here"
```

#### Python中验证:
```python
import os
print(os.environ.get('GM_TOKEN'))  # 应该输出您的token
```

### 方式2: .env文件（推荐）

#### 创建.env文件:
```bash
# 在项目根目录创建.env文件
cd csi1000_quant
cp .env.example .env
```

#### 编辑.env文件:
```bash
# 使用文本编辑器打开.env文件
nano .env
# 或
vim .env
# 或在VS Code中打开
code .env
```

#### .env文件内容:
```
GM_TOKEN=your_gm_token_here
```

#### 注意事项:
- .env文件已添加到.gitignore，不会被提交到版本控制
- 确保.env文件权限设置正确（仅当前用户可读）

### 方式3: 用户配置文件

#### 创建用户配置文件:
```bash
# 在用户主目录创建.gm_token文件
echo "your_gm_token_here" > ~/.gm_token

# 设置文件权限（仅当前用户可读）
chmod 600 ~/.gm_token
```

#### 验证配置:
```bash
cat ~/.gm_token  # 应该输出您的token
```

### 方式4: 配置文件（不推荐）

#### 修改config.yaml:
```yaml
data:
  gm_config:
    token: "your_gm_token_here"  # 取消注释并填入token
```

#### 安全警告:
- ⚠️ 不推荐在配置文件中直接写入token
- ⚠️ 容易误提交到版本控制系统
- ⚠️ 存在安全风险

## 🧪 测试配置

### 运行测试脚本:
```bash
python run.py basic
```

### 查看日志输出:
- ✅ "从环境变量GM_TOKEN获取掘金量化token" - 环境变量配置成功
- ✅ "从.env文件获取掘金量化token" - .env文件配置成功
- ✅ "从用户配置文件获取掘金量化token" - 用户配置文件成功
- ✅ "掘金量化连接测试成功" - token有效且连接正常
- ❌ "未找到掘金量化token配置" - 未配置token
- ❌ "掘金量化连接测试失败" - token无效或网络问题

### Python代码测试:
```python
from src.data.data_module import DataModule
import yaml

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建数据模块
data_module = DataModule(config)

# 检查连接状态
if data_module.gm_initialized:
    print("✅ 掘金量化连接成功")
else:
    print("❌ 掘金量化连接失败，使用模拟数据")
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. "未找到掘金量化token配置"
**原因**: 没有配置token或配置位置错误
**解决方案**:
- 检查环境变量：`echo $GM_TOKEN`
- 检查.env文件是否存在且内容正确
- 检查用户配置文件：`cat ~/.gm_token`

#### 2. "掘金量化连接测试失败"
**原因**: token无效、网络问题或API限制
**解决方案**:
- 验证token是否正确（重新从官网复制）
- 检查网络连接
- 确认账户状态是否正常
- 检查API调用频率是否超限

#### 3. "掘金量化SDK未安装"
**原因**: 没有安装掘金量化SDK
**解决方案**:
```bash
pip install gm
```

#### 4. "ImportError: No module named 'gm'"
**原因**: 掘金量化SDK安装失败或环境问题
**解决方案**:
```bash
# 重新安装
pip uninstall gm
pip install gm

# 或使用conda
conda install -c conda-forge gm
```

#### 5. Token格式错误
**原因**: token包含多余的空格、引号或特殊字符
**解决方案**:
- 确保token字符串没有前后空格
- 不要包含引号（除非是token的一部分）
- 检查是否有换行符或其他隐藏字符

### 调试技巧

#### 1. 启用详细日志:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 手动测试API:
```python
from gm.api import set_token, history

# 设置token
set_token("your_token_here")

# 测试API调用
try:
    data = history(
        symbol='SHSE.000852',
        frequency='1d',
        start_time='2024-01-01',
        end_time='2024-01-02',
        df=True
    )
    print("API调用成功:", len(data))
except Exception as e:
    print("API调用失败:", str(e))
```

#### 3. 检查token有效性:
```python
import requests

# 注意：这只是示例，实际API端点可能不同
def check_token_validity(token):
    # 具体实现需要参考掘金量化官方文档
    pass
```

## 🔒 安全最佳实践

### 1. Token保护
- ✅ 使用环境变量或.env文件
- ✅ 设置适当的文件权限
- ✅ 定期更换token
- ❌ 不要在代码中硬编码token
- ❌ 不要将token提交到版本控制

### 2. 访问控制
- 限制token的使用范围
- 监控API调用频率
- 及时撤销不需要的token

### 3. 备份和恢复
- 保存token的安全备份
- 准备token失效时的应急方案

## 📚 相关文档

- [掘金量化官方文档](https://www.myquant.cn/docs)
- [掘金量化Python API](https://www.myquant.cn/docs/python)
- [项目README](../README.md)
- [快速开始指南](../QUICKSTART.md)

## 💡 提示

1. **首次使用**: 建议先使用环境变量方式配置，简单快捷
2. **生产环境**: 推荐使用.env文件或用户配置文件
3. **开发调试**: 可以临时使用配置文件方式，但记得及时清理
4. **团队协作**: 使用.env.example文件分享配置模板，但不要分享实际token

## 🆘 获取帮助

如果您在配置过程中遇到问题：

1. 查看系统日志文件：`logs/`目录
2. 参考掘金量化官方文档
3. 检查网络连接和防火墙设置
4. 联系掘金量化技术支持
5. 在项目仓库提交Issue

---

**注意**: 请妥善保管您的token，不要泄露给他人。如果怀疑token泄露，请立即在掘金量化官网重新生成。

