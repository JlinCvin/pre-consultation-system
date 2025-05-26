# 预诊系统 (Pre-consultation System)

这是一个基于 FastAPI 和 LangChain 开发的智能预诊系统，旨在通过人工智能技术提供初步的医疗咨询和诊断建议。

## 功能特点

- 🤖 智能对话：基于 LangChain 的智能对话系统
- 🗣️ 语音交互：支持语音输入和输出
- 📊 报告生成：自动生成诊断报告
- 📝 患者信息管理：完整的患者信息记录和管理
- 🔄 历史记录：保存咨询历史记录
- 🎨 现代化界面：美观的用户界面设计

## 系统要求

- Python 3.8+
- MySQL 8.0+
- 足够的系统内存（建议 8GB 以上）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd pre-consultation-system
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
创建 `.env` 文件并设置必要的环境变量：
```env
DATABASE_URL=mysql://user:password@localhost:3306/dbname
OPENAI_API_KEY=your_api_key
```

## 运行系统

### 开发环境

```bash
python run.py
```

### 生产环境

Windows:
```bash
start_prod.bat
```

Linux/Mac:
```bash
./start_prod.sh
```

## 项目结构

```
pre-consultation-system/
├── main.py              # 主应用程序入口
├── config.py            # 配置文件
├── models.py            # 数据模型
├── db.py               # 数据库操作
├── voice_chat.py       # 语音交互功能
├── agents.py           # AI 代理实现
├── report.py           # 报告生成
├── routers/            # API 路由
├── static/             # 静态文件
└── images/             # 图片资源
```

## API 文档

启动服务后，访问以下地址查看 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request
