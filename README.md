# 中证1000指数相对低点识别量化系统

一个基于掘金量化平台的量化交易程序，旨在识别中证1000指数的相对低点，并在当天或第二天向用户发出提示。

## 功能特点

- 🎯 **相对低点识别**：基于定义和技术指标识别相对低点
- 🤖 **AI自优化**：支持多种AI优化方法提高识别准确率，引入时间序列数据有效性加权，并优化训练集/测试集划分
- 🧠 **LLM驱动策略优化**：新增LLM驱动的策略参数自动调整和优化功能
- 📊 **全面回测**：支持多种回测方式验证策略效果，并提供测试集评估以缓解过拟合
- 📧 **实时通知**：在识别到相对低点时及时通知用户
- 🔧 **模块化设计**：便于扩展和维护

## 环境要求

- Python 3.8+
- 虚拟环境（推荐使用）

## 快速开始

### 1. 创建并激活虚拟环境

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置系统

编辑 `config/config.yaml` 文件，设置您的参数，包括：
- AI优化参数
- 数据有效性配置
- 回测参数
- 通知设置

### 4. 运行示例

- **运行基本回测**

  ```bash
  python run.py basic
  ```

- **运行AI优化回测**

  ```bash
  python run.py ai_optimization
  ```

- **单日低点预测**

  ```bash
  python predict_single_day.py <YYYY-MM-DD>
  # 示例：python predict_single_day.py 2024-06-01
  ```

- **LLM驱动策略优化**

  ```bash
  python llm_strategy_optimizer.py
  ```

## 项目结构

```
csi1000_quant/
├── src/                    # 源代码
│   ├── data/              # 数据获取模块
│   ├── strategy/          # 策略执行模块
│   ├── ai/                # AI优化模块
│   ├── notification/      # 通知模块
│   └── utils/             # 工具模块
├── tests/                 # 测试代码
├── config/                # 配置文件
├── docs/                  # 文档
├── examples/              # 示例代码
├── data/                  # 数据存储目录
├── logs/                  # 日志文件
├── models/                # 模型存储目录
├── requirements.txt       # 依赖列表
├── predict_single_day.py  # 单日预测脚本
├── llm_strategy_optimizer.py # LLM驱动策略优化脚本
└── README.md             # 项目说明
```

## 相对低点定义

从当天起到20个交易日内，直至某一天指数能够上涨5%，则当天被认为是该指数的相对低点。

## 主要依赖

- **基础数据处理**：numpy, pandas, scipy
- **数据可视化**：matplotlib, seaborn
- **机器学习**：scikit-learn, xgboost, lightgbm
- **配置管理**：PyYAML
- **网络请求**：requests
- **开发工具**：pytest, black, flake8

## 使用说明

详细使用说明请参考：
- `docs/` 目录下的文档
- `QUICKSTART.md` 快速入门指南
- `examples/` 目录下的示例代码

## 开发指南

1. 克隆仓库
2. 创建并激活虚拟环境
3. 安装开发依赖
4. 运行测试确保环境正常
5. 开始开发

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 更新日志

详细更新历史请参考 `CHANGELOG.md`。


