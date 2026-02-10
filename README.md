# AI 相面 - 春晚特别版

> 利用 AI 识别面相并给出吉利判词的互动小游戏，专为春节联欢晚会设计。

## 功能亮点

- 📷 摄像头实时人脸追踪（支持多人）
- 🔮 AI 面相分析（6-10 秒，分阶段展示分析过程）
- 🎯 五维特征识别（印堂 / 眉眼 / 鼻翼 / 嘴角下巴 / 气色）
- 🖼️ 特征圈注图（标注识别区域，可验真）
- 🧧 20-30 字吉利判词 + 硅谷科技梗

## 快速启动

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 AI_BUILDER_TOKEN（必填）
```

### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 启动后端服务

```bash
cd backend
# 若使用 .env，需先加载：source ../.env 或使用 python-dotenv
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问应用

打开浏览器访问：`http://localhost:8000`

前端页面由 FastAPI 静态文件服务自动托管。

## 项目结构

```
AI 相面/
├── frontend/           # 前端（纯 HTML + JS）
│   └── index.html
├── backend/            # 后端（Python FastAPI）
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # 主服务
│       ├── services/
│       │   ├── fortune_pipeline.py   # MCP 推理编排
│       │   └── feature_annotator.py  # 特征圈注
│       └── data/
│           └── face_rules.json       # 知识库（程序可读）
├── docs/
│   └── face-reading-core.md          # 知识库（人类可读）
├── logs/
│   └── ai_trace.log                  # 验真日志
├── outputs/                          # 圈注图输出
├── tests/                            # 测试脚本
├── 人脸测试.png                       # 测试用人脸图片
└── README.md
```

## 技术栈

- **前端**：纯 HTML + Vanilla JS + CSS（无构建依赖）
- **后端**：Python FastAPI
- **人脸追踪**：MediaPipe Tasks Vision (FaceDetector)
- **AI 推理**：AI Builders MCP（`chat/completions` 多模态接口）
- **图像标注**：Pillow

## AI 能力调用

所有模型推理通过 AI Builders MCP 完成：
- 认证：`AI_BUILDER_TOKEN`
- 接口：`/v1/chat/completions`（多模态 image_url / data URI）
- 模型：`grok-4-fast`（默认，速度快）
