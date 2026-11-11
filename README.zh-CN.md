# EchoMind

[English](README.md) · **简体中文**

一个具备多 Agent 路由、持久化对话记忆和真实工具调用的 AI 客服系统，基于 FastAPI、LangGraph、Redis、ChromaDB 和 MCP Python SDK 构建。

## 核心能力

- **请求路由**：由通用、技术和账单 Agent 处理请求，通过 Markdown Skills 加载业务规则。
- **上下文记忆**：Redis 保存工作记忆，ChromaDB 保存会话摘要和用户画像，SQLite 保存问题处理记录。
- **工具调用**：Agent 自行决定何时检索知识、查询订单、准备退货或记录交接，由独立 MCP 进程执行。
- **知识检索**：查询改写、向量与 BM25 召回、RRF 融合和 LLM 重排。
- **中断恢复**：LangGraph SQLite checkpoint 保存节点进度，结合请求编号支持重试，并记录工具执行证据。
- **零售评测**：接入固定版本的官方 τ²-bench，通过页面运行任务、查看进度和结果。

```text
HTTP /chat 或 CLI
  → LangGraph 工作流：记忆 → 问题记录 → 意图 → Agent
  → 模型选择工具 → MCP Server → 知识库 / 业务 SQLite
  → 回复、工具证据、记忆更新与 checkpoint
```

## 快速开始

需要 Docker、Docker Compose，以及支持 Anthropic 协议的模型 API 凭证。以下命令均在项目根目录执行。

```bash
cp .env.example .env
```

编辑 `.env`。以下使用 DeepSeek，与内置评测页面的默认配置一致；请将 Key 和模型占位符替换为账号实际可用的值：

```env
ANTHROPIC_API_KEY=your_deepseek_api_key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=your_deepseek_model
```

主应用也支持官方 Anthropic API：不设置 `ANTHROPIC_BASE_URL`，并使用对应的 Key 和模型即可。评测页面必须显式设置 `ANTHROPIC_MODEL`，用户模拟器和 Judge 默认使用 DeepSeek；其他供应商需单独配置评测。详见[配置指南](docs/guide.zh-CN.md#section-2)。

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

Compose 会启动主应用、τ 评测服务、Redis、ChromaDB、Prometheus 和 Nginx。首次构建还会下载模型资源和固定版本的评测源码。

| 入口 | 用途 |
| --- | --- |
| [API 文档](http://localhost:8000/docs) | 浏览和调用 HTTP 接口 |
| [评测页面](http://localhost:8000/eval) | 运行官方零售任务、查看结果 |
| [Prometheus](http://localhost:9090) | 查看监控指标 |

## 试用

发送一条消息：

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"我想查询订单","user_id":"u1001","conv_id":"demo-chat","client_request_id":"message-1"}'
```

后续消息保持相同的 `user_id` 和 `conv_id`。每条新消息使用新的 `client_request_id`，重试同一条消息时沿用原编号。响应包含回复、路由信息、工作流进度和工具执行证据。

上传演示知识：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F 'file=@fixtures/knowledge/sample_knowledge.json'
```

也可以使用交互式 CLI：

```bash
docker compose run --rm echomind python api/main.py --cli
```

## 评测

打开[评测页面](http://localhost:8000/eval)，选择题集和重复次数后启动。接入调试使用 `train`，正式评测使用 `test`。结果保存在 `data/eval/tau/<run-id>/`。

只生成计划、不调用模型：

```bash
docker compose build evaluation
docker compose run --rm evaluation --split test --plan-only
```

独立 CLI 需要为用户模拟器和 Judge 单独配置供应商凭证，不会自动复用主应用的 Key。配置、判分、测试与历史接入记录见[评测指南](evaluation/README.md)。已记录的运行是训练题接入验证，不是完整测试集成绩。

## 文档

| 指南 | 内容 |
| --- | --- |
| [完整使用指南](docs/guide.zh-CN.md) | 部署、API 参考、知识导入、记忆查看、监控与排障 |
| [评测指南](evaluation/README.md) | τ²-bench 配置、凭证、判分、产物与测试 |
| [Skills 指南](skills/README.md) | 业务规则编写与热加载 |

## 当前范围

仓库提供 HTTP API、CLI 和评测页面，未包含独立聊天前端。业务工具操作本地 SQLite 演示数据库，订单为虚构数据；尚未对接支付系统和真实人工客服。

HTTP API 尚无账户登录鉴权，会话锁仅在单个应用进程内生效，当前部署面向单实例运行。
