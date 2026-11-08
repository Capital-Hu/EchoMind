# EchoMind 完整使用指南

本文档说明 EchoMind 的部署、启动、API 调用、知识库使用、ChromaDB 数据查看、监控评测和常见排障。

EchoMind 是一个企业级智能客服系统，核心链路为：

```text
用户请求
  -> FastAPI /chat
  -> LangGraph 工作流（每一步完成后写入 SQLite checkpoint）
  -> MemoryManager 读取 Redis 工作记忆 + ChromaDB 情景记忆 + 用户画像
  -> IntentRecognizer 识别意图
  -> AgentOrchestrator 路由到 General/Technical/Billing Agent
  -> LLM 自行决定工具调用，接收实际结果后继续决策或生成回复
  -> 写入 Redis，并异步更新 ChromaDB 用户画像
```

## 1. 项目结构

```text
EchoMind/
├── api/main.py                    # FastAPI 入口，/chat /search /knowledge /monitor
├── core/intent_recognizer.py      # 三路融合意图识别
├── agents/agent_orchestrator.py   # 多 Agent 路由编排
├── agents/chat_workflow.py        # LangGraph 节点、持久化与中断恢复
├── agents/tool_calling.py         # 原生 tool_use / tool_result 多轮循环
├── memory/conversation_memory.py  # Redis + ChromaDB 记忆管理
├── memory/problem_record.py       # 按问题保存用户反馈、已尝试方法及接手 Agent
├── tooling/tool_manager.py        # MCP 工具注册、范围检查、缓存与请求转发
├── tooling/knowledge_service.py   # MCP Server 调用的知识库业务实现与完整增强检索
├── tooling/business_store.py      # MCP Server 持有的业务 SQLite、事务、归属与确认检查
├── tooling/business_schema.sql    # 客户、商品、订单、退货、交接与操作记录表
├── fixtures/business_demo.json    # 项目编写的虚构演示业务数据
├── tooling/mcp_client.py          # 官方 SDK 的 stdio 客户端与子进程生命周期
├── tooling/mcp_server.py          # 独立 MCP 工具进程
├── tooling/mcp_protocol.py        # 工具 Schema、调用范围与上下文合同
├── tooling/knowledge_base.py      # ChromaDB RAG 知识库
├── monitor/performance_monitor.py # Agent/工具在线监控
├── evaluation/run.py              # τ 官方零售评测
├── data/demo_docs/                # 演示知识库文档
├── docker-compose.yml             # Docker 全栈编排
├── Dockerfile
├── requirements.txt
└── .env
```

## 2. 环境准备

### 2.1 必需依赖

- Docker
- Docker Compose
- Anthropic API Key，或兼容 Anthropic 协议的第三方 API Key

### 2.2 配置 `.env`

复制示例文件：

```bash
cp .env.example .env
```

最少需要配置：

```env
ANTHROPIC_API_KEY=your_api_key
```

如果使用 DeepSeek 这类 Anthropic 兼容接口，可以配置：

```env
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-pro
ANTHROPIC_API_KEY=your_deepseek_key
```

所有模型调用的输出上限统一在 `.env` 配置，默认 4096，必须是正整数：

```env
CLAUDE_MAX_TOKENS=4096
```

意图识别、查询改写、检索重排、用户画像、摘要生成与合并、Agent 回复均使用
这个值，问题处理记录的提取也共用此配置。它只控制输出额度，思考模式仍沿用当前
模型设置；摘要最终仍限制在 800 字内。

修改 `.env` 后，本地运行需重新启动后端进程。Docker Compose 需重新创建应用容器
以加载新环境变量，仅执行 `docker compose restart` 不会重新读取 `.env`：

```bash
docker compose up -d --no-deps --force-recreate echomind
```

Docker Compose 场景下，Redis 和 ChromaDB 的连接由 `docker-compose.yml` 覆盖为容器内地址。通常不需要手动改：

```env
REDIS_PASSWORD=your_redis_password
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

### 2.3 全栈部署和 run 开发模式的区别

EchoMind 常用两种 Docker 启动方式：`docker compose up` 全栈部署，以及 `docker run` 开发模式。两者最大的区别是：**全栈部署会同时启动应用和依赖服务；run 开发模式通常只手动运行一个应用容器，依赖服务需要提前启动**。

| 对比项 | Docker Compose 全栈部署 | Docker run 开发模式 |
|--------|--------------------------|----------------------|
| 启动命令 | `docker compose up -d --build` | `docker run ... echomind ...` |
| 启动内容 | EchoMind、Redis、ChromaDB、Prometheus、Nginx | 只启动你指定的单个容器 |
| Redis/ChromaDB | 自动启动并加入同一网络 | 必须先执行 `docker compose up -d redis chromadb` |
| 容器网络 | Compose 自动创建并管理 | 需要手动指定 `--network echomind_echomind-network` |
| 服务名解析 | 应用可直接访问 `redis`、`chromadb` | 只有加入同一网络后才可访问 `redis`、`chromadb` |
| 代码更新 | 通常需要 rebuild 或重启服务 | 挂载 `-v "$(pwd):/workspace"` 后，代码修改可直接生效，重启容器即可 |
| 适合场景 | 演示、联调、完整部署、HTTP API 服务 | 本地开发、调试 CLI、临时覆盖环境变量 |
| 常见问题 | API Key 或依赖健康检查失败 | 忘记启动 Redis/ChromaDB，导致 `redis:6379 Name or service not known` |

选择建议：

- 想完整体验 HTTP API、Swagger、Nginx、Prometheus：用 **Docker Compose 全栈部署**。
- 想调试源码或 CLI，并且希望本地改代码后快速重跑：用 **Docker run 开发模式**。
- 如果只是跑 CLI，最省心的方式是 `docker compose run --rm echomind python api/main.py --cli`，它会自动使用 Compose 网络。

## 3. Docker Compose 全栈部署

推荐使用此方式启动完整服务。

```bash
docker compose up -d --build
```

查看服务状态：

```bash
docker compose ps
```

查看应用日志：

```bash
docker compose logs -f echomind
```

看到 EchoMind 启动日志并且健康检查通过后，服务可用。

启动后的端口：

| 服务 | 容器名 | 宿主机端口 | 容器内端口 | 用途 |
|------|--------|------------|------------|------|
| EchoMind API | `echomind-app` | `8000` | `8000` | 主 API 服务 |
| Nginx | `echomind-nginx` | `80` | `80` | 反向代理 |
| ChromaDB | `echomind-chromadb` | `8001` | `8000` | 向量数据库 |
| Redis | `echomind-redis` | `6379` | `6379` | 工作记忆 |
| Prometheus | `echomind-prometheus` | `9090` | `9090` | 监控数据 |

健康检查：

```bash
curl http://localhost:8000/health
```

Swagger 文档：

```text
http://localhost:8000/docs
```

也可以通过 Nginx 访问：

```bash
curl http://localhost/health
```

## 4. Docker Run 开发模式

开发时可以只用 Compose 启动依赖，然后用 `docker run` 挂载当前代码目录。

先启动 Redis 和 ChromaDB：

```bash
docker compose up -d redis chromadb
```

构建镜像：

```bash
docker compose build --no-cache echomind
```

启动 HTTP 服务：

```bash
docker run -it --rm \
  --network echomind_echomind-network \
  -p 8000:8000 \
  -e ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic" \
  -e ANTHROPIC_API_KEY="your_key" \
  -e ANTHROPIC_MODEL="deepseek-v4-pro" \
  -e REDIS_URL="redis://:your_redis_password@redis:6379/0" \
  -e CHROMA_HOST="chromadb" \
  -e CHROMA_PORT="8000" \
  -e CHROMA_PERSIST_DIRECTORY="/workspace/data/chroma" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  echomind
```

CLI 交互模式：

```bash
docker run -it --rm \
  --network echomind_echomind-network \
  -e ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic" \
  -e ANTHROPIC_API_KEY="your_key" \
  -e ANTHROPIC_MODEL="deepseek-v4-pro" \
  -e REDIS_URL="redis://:your_redis_password@redis:6379/0" \
  -e CHROMA_HOST="chromadb" \
  -e CHROMA_PORT="8000" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  echomind \
  python api/main.py --cli
```

## 5. Swagger 和接口总览

EchoMind 基于 FastAPI 构建，启动 HTTP 服务后可以直接在浏览器访问 Swagger UI 调用接口。

本地 Swagger 地址：

```text
http://localhost:8000/docs
```

如果使用 Nginx 反向代理：

```text
http://localhost/docs
```

打开 Swagger 后，可以点击任意接口右侧的 **Try it out**，填写参数后点 **Execute** 直接调用本地服务。常用调试顺序：

```text
1. GET /health                确认服务是否就绪
2. POST /chat                 测试主对话链路
3. GET /knowledge/stats       查看知识库是否已有数据
4. POST /knowledge/upload     上传演示知识库文件
5. POST /search               测试知识库检索、查询改写和重排
6. GET /monitor               查看 Agent 和工具运行指标
7. GET /skills                查看已加载 Skills
8. POST /skills/reload        重新加载 Skills
9. τ retail                  独立运行官方零售评测
```

### 5.1 接口总览

| 方法 | 路径 | 参数位置 | 作用 | 适合场景 |
|------|------|----------|------|----------|
| `GET` | `/health` | 无 | 健康检查，返回服务状态和 Agent 统计 | 启动后确认服务可用 |
| `POST` | `/chat` | JSON Body | 主对话接口，完成记忆读取、意图识别、Agent 路由、回复生成、记忆写入 | 业务主链路 |
| `GET` | `/workflow/status` | Query 参数 | 查看当前请求已保存的进度与下一步 | 排查中断位置 |
| `POST` | `/workflow/resume` | JSON Body | 从持久化进度继续最后一条未完成请求 | 重启或异常后恢复 |
| `GET` | `/monitor` | 无 | 查看 Agent/工具统计、告警和优化建议 | 观察在线表现 |
| `POST` | `/search` | Query 参数 | 查询改写、向量与 BM25 召回、RRF 融合、LLM 重排 | 测试 RAG 检索 |
| `GET` | `/skills` | 无 | 查看当前加载的 Skills、匹配关键词和解析错误 | 确认动态能力是否生效 |
| `POST` | `/skills/reload` | 无 | 运行时重新扫描 Skill 目录 | 修改业务规则后热加载 |
| `POST` | `/knowledge/add` | JSON Body | 批量导入文档到 ChromaDB 知识库 | 程序化导入文档 |
| `POST` | `/knowledge/upload` | Form File | 上传 `.txt`、`.md`、`.json` 文件导入知识库 | 手动上传知识库文件 |
| `GET` | `/knowledge/stats` | 无 | 查看知识库文档片段总数 | 确认知识库是否有数据 |
| `GET` | `/business/stats` | 无 | 通过 MCP 查看业务 SQLite 路径、来源与表数量 | 核对业务库和持久化结果 |
| `GET` | `/docs` | 浏览器访问 | Swagger UI | 浏览和调试所有接口 |

### 5.2 Skills 动态能力加载

EchoMind 支持从目录加载 Skills，用来把业务流程、客服话术、排障 SOP 等规则动态注入 Agent。

默认配置：

```env
ECHOMIND_SKILLS_DIR=./skills
ECHOMIND_SKILLS_MAX_PROMPT_CHARS=5000
```

推荐结构：

```text
skills/refund/SKILL.md
skills/customer_support/SKILL.md
```

`SKILL.md` 示例：

```markdown
---
name: 退款处理流程
description: 退款场景的客服处理规则
keywords: 退款,退费,refund
agents: billing,general
enabled: true
---

# 退款处理流程

- 先确认订单号和支付方式。
- 涉及实际退款操作时转人工审核。
```

查看加载结果：

```bash
curl http://localhost:8000/skills
```

修改 Skill 文件后热加载：

```bash
curl -X POST http://localhost:8000/skills/reload
```

### 5.3 `/health`

用途：确认服务是否初始化完成。

```bash
curl http://localhost:8000/health
```

响应示例：

```json
{
  "status": "ok",
  "agents": {
    "general_0": {
      "total": 0,
      "success_rate": 1.0,
      "avg_ms": 0.0,
      "monitor_penalty": 0.0,
      "routing_score": 1.0
    }
  }
}
```

### 5.4 `/chat`

用途：主对话接口。

请求体：

```json
{
  "message": "我要退款",
  "user_id": "user_001",
  "conv_id": "session_001"
}
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `message` | 是 | 用户输入 |
| `user_id` | 否 | 用户 ID，默认 `anonymous` |
| `conv_id` | 否 | 会话 ID，不传则自动生成 |
| `client_request_id` | 否 | 同一消息重试时沿用同一编号，恢复进度或返回原结果 |

返回字段：

| 字段 | 说明 |
|------|------|
| `conv_id` | 会话 ID |
| `response` | Agent 回复 |
| `intent` | 意图识别结果 |
| `agent_type` | 实际处理请求的 Agent |
| `escalated` | 是否触发升级 |
| `latency_ms` | 端到端耗时 |
| `workflow` | 请求编号、checkpoint 编号、执行路径及恢复/缓存标记 |
| `tools_used` | 本轮实际执行过或命中缓存的工具名称 |
| `tool_traces` | 模型调用编号、实际参数、执行结果、错误、耗时和缓存标记 |

### 原生 function calling

API 与 CLI 的 Agent 使用 Anthropic Messages 原生 `tools` 参数，模型通过 `tool_use`
给出工具名和参数；后端调用已注册的真实 handler，再按相同 `tool_use_id` 回传 `tool_result`。
模型可以根据结果继续调用工具，或直接回答。实现参考
[Anthropic 工具定义](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)与
[DeepSeek Anthropic 兼容协议](https://api-docs.deepseek.com/guides/anthropic_api/)。

当前提供知识检索和 SQLite 业务工具，模型可查询当前用户订单、准备并提交退货、查询进度和记录交接。聊天不再按关键词提前检索：
模型自行决定是否检索、搜索什么、取多少条，以及是否调整查询再次检索。
前端演示 `/search` 和模型的 `knowledge_search` 都把完整请求交给同一个 MCP Server，
由工具子进程内唯一的 `KnowledgeService` 实例完成改写、召回、去重与重排，再一次性返回结果。
主程序只转发调用，不创建检索实例。现有对话记忆、问题记录和 Skills 继续提供背景。

```text
模型收到工具定义与用户问题
  -> 返回 tool_use（例如 knowledge_search，query="退款审核及到账时间"）
  -> 后端校验参数，把完整搜索请求交给 MCP Server
  -> Server 内调用 search_with_rewrite：改写、向量与 BM25 召回、RRF 融合与重排
  -> 回传 tool_result（真实文档或错误）
  -> 模型继续决策，最终生成回复
```

每个 Agent 最多执行 4 轮工具响应、处理 8 个工具请求，达到上限后另做一轮仅允许回答的模型调用。
工具未注册、未向该 Agent 开放、参数不合法或执行超时，都会返回明确错误。
供应商返回的完整 assistant 内容（包括思考块）会带回下一轮，执行证据只记录工具信息。
工具超时或降级内容不标为已核实知识，`knowledge_used` 根据实际成功检索结果计算。

在 Swagger 的 `/chat` 中可以测试：

```json
{
  "user_id": "function-demo",
  "conv_id": "function-demo-chat",
  "client_request_id": "function-demo-1",
  "message": "平台退款申请审核和到账分别需要多久？"
}
```

查看响应中的 `tools_used` 和 `tool_traces`。每个 trace 包含模型返回的 `tool_use_id`、
`input` 参数、`executed`（是否进入 handler）、`success`、`result`、`error`、`latency_ms`
与 `cached`。`retrievals` 列出每次召回的查询、通道（`vector` 或 `bm25`）、结果数量和执行 PID，顶层 `receipt` 对应整次 MCP 工具调用。
重排未执行或失败时 `reranked=false`，按 RRF 顺序返回。
缓存保存 Server 返回的最终检索结果与调用回执。知识导入成功后，索引和搜索缓存都会失效。
这些证据也随 Agent 结果存入 checkpoint，成功请求重试时直接返回原证据。

工具通过真正的 MCP stdio Client/Server 执行。业务库操作真实持久化；支付退款系统和专用前端证据面板尚未接入。

### MCP stdio 工具服务

后端启动时使用官方 `mcp==1.26.0` SDK 启动 `python -m tooling.mcp_server` 子进程，
完成 `initialize` 与 `tools/list`，核对工具集合及 Schema，再把已发现的工具提供给 Agent。
本地模块统一使用 `tooling` 名称，官方 SDK 使用 `mcp`，不保留旧导入路径。

```text
模型 tool_use
  -> 后端检查 Agent 可用范围与参数
  -> MCP Client 发送一次完整的 tools/call（stdin/stdout JSON-RPC）
  -> 独立 MCP Server 再检查会话凭据、调用范围和 JSON Schema
  -> Server 完成改写、向量与 BM25 召回、RRF 融合和重排
  -> 完整增强检索结果及服务端回执
  -> 模型 tool_result，继续决策
```

| MCP 工具 | 允许的调用范围 | 用途 |
|---|---|---|
| `knowledge_search` | general / technical / billing | 聊天工具，Server 完整执行增强检索 |
| `knowledge_add` | admin | 现有知识导入、文件上传接口 |
| `knowledge_stats` | admin | 启动检查、现有知识统计接口 |
| `customer_lookup` | general / technical / billing | 定位当前后台用户关联的客户 |
| `order_list` / `order_get` | general / technical / billing | 当前用户订单列表与明细 |
| `product_get` | general / technical / billing | 商品介绍、标价 |
| `return_prepare` / `return_submit` | general / billing | 准备退货、用户下一轮确认后正式提交 |
| `return_status` | general / technical / billing | 读取实际退货进度和当前聊天待确认明细 |
| `handoff_create` / `handoff_status` | general / technical / billing | 持久化交接申请和查询记录 |
| `business_stats` | admin | 业务库路径、来源与表数量 |

共 13 个 MCP 工具，general / billing 模型可见其中 10 个，technical 不开放准备和提交退货。
调用范围、用户编号、会话编号、请求编号和模型工具调用编号，
由后端通过 MCP `_meta` 传递，模型参数不能覆盖。子进程会验证每次启动生成的私有会话凭据；
Server 通过显式白名单接收现有模型配置（API Key、模型名、Base URL、CLAUDE_MAX_TOKENS），
用于自己发起改写和重排的模型请求；不继承 Redis 配置。这是后端与工具进程的范围检查，现有 HTTP 接口尚未提供账户登录鉴权。

三个 MCP 知识工具由子进程执行，`/knowledge/add`、`/knowledge/upload`、`/knowledge/stats` 也通过 MCP。
前端演示 `/search` 同样调用 MCP 的 `knowledge_search`，使用与聊天工具相同的服务实例和知识库连接。
主进程的工具管理器只负责定义、范围检查、缓存和转发，不代 Server 做改写或重排。
`mcp_server.py` 负责 MCP 协议、权限检查、超时和调用日志；`knowledge_service.py` 负责具体的知识库业务流程，
由 Server 调用，不是另一个独立服务。演示每次都执行真实 MCP 调用，聊天工具仍可复用缓存结果。
导入每次最多 100 篇，每篇正文最多 50000 字；上传文件仍先检查 10MB 大小限制。

执行证据新增 `transport=stdio`、`execution_status` 和 `receipt`。
回执包含 `server_pid`、`call_id`、`request_id`、`tool_use_id`，可与服务端日志相互核对。
缓存命中时不会再次发起 MCP 调用，`cached=true`、`executed=false`，保留原调用回执。
成功请求的工具证据仍随 checkpoint 保存。

### 业务 SQLite 与退货链路

业务操作全部在 MCP Server 中执行固定的参数化 SQL，主程序不直接打开业务 SQLite，也不向模型提供任意 SQL 执行工具。
`knowledge_*` 交给 `KnowledgeService`，其余业务工具交给 `BusinessStore`。启动时自动建表和导入一次演示数据。

| 项目 | 路径或说明 |
|---|---|
| 宿主机业务库 | `./data/business/business.sqlite3` |
| 容器内路径 | `/app/data/business/business.sqlite3` |
| Docker 挂载 | `./data/business:/app/data/business`，SQLite、WAL 和 SHM 一起保留 |
| 本地配置 | `BUSINESS_DB_PATH`，默认指向项目 `data/business`；Docker compose 固定为容器内路径 |
| 表结构 | `tooling/business_schema.sql` |
| 数据来源 | `fixtures/business_demo.json`，项目自行编写的虚构数据，不来自真实用户、支付系统或参考项目的数据集 |
| 导入规则 | 首次初始化记录来源版本、SHA256、时间；再次启动不重置订单或覆盖退货记录 |

业务表为 `customers`、`products`、`orders`、`order_items`、`return_preparations`、`returns`、`return_events`、`handoffs`，
另有 `business_meta` 保存来源、`business_operations` 保存每次成功业务工具操作的请求、调用编号、执行 PID 和变更行数。
工具结果中的 `data.database` 可与 MCP 回执、`logs/mcp-tools.jsonl` 和 SQLite 操作记录相互核对。
`rows_changed` 不计算操作日志自身；重复提交会增加调用日志，但不会增加退货单、事件或已退数量。

演示账号与订单：

| API / 前端 user_id | 订单 | 首次初始化状态 |
|---|---|---|
| `u1001`（前端默认） | `ORD-DEMO-1001` | 已签收，耳机 1 件，实付 199.00 元 |
| `u1001` | `ORD-DEMO-1002` | 已签收，保温杯 2 件，每件实付 79.00 元；帆布包 1 件 |
| `u1001` | `ORD-DEMO-1003` | 已发货未签收，用于验证拒绝自助退货 |
| `u1001` | `ORD-DEMO-1004` | 已超过自助期限，用于验证交接处理 |
| `u1002` | `ORD-DEMO-2001` | 另一用户的已签收订单，用于验证归属隔离 |

日期按第一次导入时间生成后固定保存，之后会正常过期。金额按整数“分”存储，退货按订单实付而非商品标价计算。
当前自助范围为已签收且购买后7天内；超期和质量争议记录交接申请，由人工核实。只记录交接，不连接真实人工。
用户归属来自后端 `user_id`，业务工具参数不能覆盖；当前仍沿用演示接口身份，尚无真实账户登录鉴权。

完整流程：

```text
用户提出退货
  -> 模型选择 customer_lookup / order_list / order_get / product_get
  -> 模型调用 return_prepare，SQLite 保存待确认明细（30分钟有效）
  -> 模型完整原样展示 preparation.display_text，等待下一轮
用户明确确认
  -> 模型调用 return_status 找回 preparation_id，再调用 return_submit
  -> Server 核对后端传来的本轮用户原话、上一轮展示内容、用户与会话、实付金额和剩余数量
  -> 单个 BEGIN IMMEDIATE 事务写退货单和事件，更新已退数量、订单退货状态、待确认状态
  -> 模型再次调用 return_status，核对落库结果再回答
重复提交同一 preparation_id
  -> 返回同一 return_id、replayed=true、rows_changed=0
```

确认信息只由后端通过私有 `_meta` 提供，模型没有 `confirmed=true`、确认原话或身份覆盖参数。
Server 要求上一轮完整展示明细，拒绝准备与提交发生在同一请求；“不确认”、引用确认话术等不会触发提交。
一份待确认明细时可以回复“确认提交”；多份明细并存时需回复对应的 `confirmation_text`。
已提交的同一编号可安全重试，数据库唯一约束和事务同时防止并发重复写入；写入途中失败会整体回滚。
业务工具不缓存，每次查询都读取 SQLite 最新状态。超时或连接中断仍标记结果未确认，不能由模型推断成功。

Swagger `/chat` 可使用以下第一轮请求：

```json
{"user_id":"u1001","conv_id":"business-demo-1","client_request_id":"business-prepare-1","message":"我想退 ORD-DEMO-1001 的耳机，不需要了，请帮我准备退货。"}
```

看到完整明细后，使用同一 `user_id`、`conv_id` 和新的 `client_request_id` 发送“确认提交”。
查看 `tool_traces` 中的真实参数、`result.data.database`、`receipt.server_pid` 及退货编号；
再发送“查询这笔退货进度”，或使用 `GET /business/stats` 查看表数量。退款状态保持 `not_issued`，未调用真实支付系统。

可以在 Swagger 或以下接口查看实时连接状态：

```bash
curl http://localhost:8000/mcp/status
```

`/health` 也会检测 MCP 连通性。子进程与后端在同一个容器里，关闭后端会回收子进程。
启动/握手默认超时 30 秒，Server 内整次增强检索与主进程演示均限时 90 秒，导入执行超时 60 秒。
断链、传输超时或结果合同错误会报告未确认/不可用，不自动重试，不改走本地直调；重启后端可建立新连接。

日志保存到已有 Docker 日志挂载目录：

- `logs/mcp-server.log`：进程启动、诊断及工具执行日志；协议独占 stdout，日志写 stderr。
- `logs/mcp-tools.jsonl`：服务端收到/拒绝/完成调用的记录，含原始参数、调用上下文、内部子查询、重排标记、结果数量、错误及耗时；不记录会话凭据。

MCP stdio 的运行方式参考 [官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.26.0)。

### LangGraph 进度保存与恢复

`/chat` 和 CLI 已使用真正的 LangGraph `StateGraph`，通过 `AsyncSqliteSaver` 保存状态。
现阶段是单实例应用，使用独立 SQLite 文件，后续多实例部署再考虑 PostgreSQL。

```text
读取记忆 -> 更新问题记录 -> 识别意图
        -> Agent（原生工具循环与回复） -> 保存问题回复 -> 写入本轮聊天
```

每一步完成后同步保存 checkpoint，再执行下一步。保存的是这一轮的输入、已完成步骤、
中间结果及下一步位置；它与面向客服的问题处理记录、Chroma 向量记忆用途不同。
例如 Agent 节点抛出异常，恢复会从 Agent 继续，前面的问题提取和意图识别不会再次调用模型。
知识检索统一在 Agent 工具循环中执行，由模型决定是否调用。
当前循环整体属于 `run_agent` 节点，尚未逐次工具调用保存 checkpoint。
进行到一半的节点可能重新执行；当前聊天写入使用 Redis 原子去重，问题记录更新也带请求编号去重。

存储位置：

- 本地 `.env`：`WORKFLOW_CHECKPOINT_PATH=./data/workflow/checkpoints.sqlite3`
- Docker：`/app/data/workflow/checkpoints.sqlite3`，挂载宿主机 `./data/workflow`
- 重建容器保留进度；数据库文件不打包进镜像，也不提交到 Git。
- checkpoint 会保存聊天上下文，目前没有自动清理历史版本的策略。

发送消息时带上 `client_request_id`，同一条消息重试时继续使用同一编号：

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","conv_id":"chat-1","client_request_id":"request-1","message":"无法登录"}'
```

成功响应新增 `workflow` 字段，包含 `request_id`、`checkpoint_id`、`node_path`、
`resumed`（本次是否恢复）、`cached`（是否直接返回已有结果）。已成功的请求重试直接返回原结果。
前端已经会为 Python 聊天请求发送此编号，并在请求失败后的同页重试中沿用。
如果首条消息未带 `conv_id`，相同用户与请求编号仍会对应同一会话，以便第一次请求失败后恢复。

查询保存到哪一步：

```bash
curl 'http://localhost:8000/workflow/status?user_id=demo&conv_id=chat-1'
```

中断时返回 `phase=interrupted`、`resumable=true` 和 `next`（下一步）。
显式恢复最后一条未完成请求，也可以直接重发原消息：

```bash
curl -X POST http://localhost:8000/workflow/resume \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","conv_id":"chat-1","request_id":"request-1"}'
```

恢复由请求触发，重启后不会自动执行所有未完成任务。同一用户的同一会话串行执行，
有未完成请求时发送另一条消息返回 HTTP 409，避免新消息覆盖进度。
CLI 可通过 `python -m api.main --cli --conv-id chat-1` 接续会话，输入 `/resume` 恢复。
用户画像仍在回复后后台更新，不属于这条可恢复流程。

聊天流程保存节点进度，业务写操作在独立 MCP Server 中执行，并通过业务唯一约束实现重复提交去重。
工具循环仍作为整个 Agent 节点保存，checkpoint 本身不能保证外部操作只执行一次；退货提交需使用原 preparation_id 重试和查询核对。
当前锁只覆盖单个应用进程，不支持多个应用实例同时处理同一会话。

### 5.5 `/search`

用途：主动调用 MCP `knowledge_search`，运行知识检索和 RAG 检索优化。

Query 参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | 是 | 无 | 用户检索问题 |
| `top_k` | 否 | `5` | 返回结果数量 |

示例：

```bash
curl -X POST "http://localhost:8000/search?query=退款多久到账&top_k=3"
```

### 5.6 `/knowledge/add`

用途：通过 JSON 批量导入知识库。

请求体：

```json
{
  "documents": [
    {
      "title": "退款政策",
      "content": "用户在购买后 7 天内可以申请无理由退款..."
    }
  ]
}
```

### 5.7 `/knowledge/upload`

用途：上传文件导入知识库。

支持格式：

| 格式 | 说明 |
|------|------|
| `.txt` | 整个文件作为一篇文档 |
| `.md` | 整个文件作为一篇文档 |
| `.json` | JSON 数组，格式为 `[{ "title": "...", "content": "..." }]` |

示例：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@data/demo_docs/sample_knowledge.json"
```

### 5.8 `/knowledge/stats`

用途：查看知识库片段数量。

```bash
curl http://localhost:8000/knowledge/stats
```

### 5.9 `/monitor`

用途：查看 Agent 和工具在线指标。

```bash
curl http://localhost:8000/monitor
```

返回内容包括：

| 字段 | 说明 |
|------|------|
| `agent_stats` | Agent 调用次数、成功率、延迟、routing_score |
| `tool_stats` | 工具调用次数、成功率、延迟、熔断状态 |
| `active_alerts` | 最近告警 |
| `suggestions` | 优化建议 |

### 5.10 τ 零售评测

打开 [评测页面](http://localhost:8000/eval)，选择题集、题目和重复次数，点击「开始评测」。
支持进度查看、停止、刷新后找回运行，以及逐题对话、官方分数和结果下载。
启动 `docker compose up -d --build echomind` 会自动带起评测服务及 Redis、ChromaDB。

评测直接采用固定版本 τ²-bench 的零售任务、模拟用户、工具环境与官方评分器。
EchoMind 通过真实 HTTP、工作流、记忆和 MCP 处理任务，详见 [评测说明](evaluation/README.md)。

```bash
docker compose build evaluation
docker compose run --rm evaluation --split test --plan-only
```

## 6. 使用项目

### 6.1 主对话接口

请求：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我的订单什么时候到？",
    "user_id": "user_001",
    "conv_id": "session_001"
  }'
```

响应示例：

```json
{
  "conv_id": "session_001",
  "response": "请提供订单号，我可以帮您查询订单状态和物流进度。",
  "intent": "query",
  "agent_type": "general",
  "escalated": false,
  "latency_ms": 1234.5
}
```

字段说明：

| 字段 | 含义 |
|------|------|
| `message` | 用户输入 |
| `user_id` | 用户唯一标识，用于隔离记忆和用户画像 |
| `conv_id` | 会话 ID，相同 `conv_id` 表示同一轮多轮对话 |
| `intent` | 识别出的意图 |
| `agent_type` | 实际处理请求的 Agent |
| `escalated` | 是否触发升级/转人工 |
| `latency_ms` | 端到端延迟 |

### 6.2 多轮对话

多轮对话只需要保持同一个 `user_id` 和 `conv_id`。

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "订单号是 A123456",
    "user_id": "user_001",
    "conv_id": "session_001"
  }'
```

系统会从 Redis 读取当前会话最近消息，并从 ChromaDB 读取相关历史和用户画像，拼成上下文传给 Agent。

### 6.3 技术问题示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "应用登录一直报 401 错误",
    "user_id": "user_tech",
    "conv_id": "tech_001"
  }'
```

预期会路由到 `technical` Agent。

### 6.4 账单问题示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "为什么这个月重复扣款了？我要退款",
    "user_id": "user_bill",
    "conv_id": "bill_001"
  }'
```

预期会路由到 `billing` Agent。

### 6.5 复合问题示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "登录报错 401，而且这个月还重复扣款了",
    "user_id": "user_mix",
    "conv_id": "mix_001"
  }'
```

这类问题会触发多 Agent 并行协作，由技术 Agent 和账单 Agent 分别处理后合并回复。

## 7. 知识库使用

EchoMind 的知识库由 `tooling/knowledge_base.py` 管理，底层使用 ChromaDB collection：

```text
knowledge_base
```

首次启动时，如果知识库为空，会自动导入默认客服文档，包括退款政策、订单查询、账户安全、技术故障排查、会员积分、配送说明。

### 7.1 查看知识库统计

```bash
curl http://localhost:8000/knowledge/stats
```

响应示例：

```json
{
  "total_chunks": 18
}
```

### 7.2 批量导入文档

```bash
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "title": "退换货政策",
        "content": "用户在购买后 7 天内可以申请无理由退货，审核通过后 5-7 个工作日退款。"
      },
      {
        "title": "会员权益",
        "content": "金卡会员享受 9 折优惠，生日当月可获得双倍积分。"
      }
    ]
  }'
```

系统会把长文档切成 500 字左右的片段，并写入 ChromaDB。

### 7.3 上传文件导入知识库

上传 Markdown：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@data/demo_docs/troubleshooting.md"
```

上传 JSON：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@data/demo_docs/sample_knowledge.json"
```

JSON 格式必须是数组：

```json
[
  {
    "title": "文档标题",
    "content": "文档内容"
  }
]
```

### 7.4 检索知识库

```bash
curl -X POST "http://localhost:8000/search?query=退款需要多久到账&top_k=3"
```

响应示例：

```json
{
  "query": "退款需要多久到账",
  "results": [
    {
      "title": "退款政策",
      "content": "审核通过后，款项将在 5-7 个工作日内退回原支付账户。",
      "score": 0.82,
      "chunk": 0,
      "chunk_id": "示例切片 ID",
      "rrf_score": 0.03265,
      "channel_ranks": {"vector": [1, 2], "bm25": 1}
    }
  ],
  "reranked": true
}
```

前端演示 `/search` 和聊天模型的 `knowledge_search` 统一交给 MCP Server，在其中执行完整检索流程：

```text
原始查询           -> BM25 关键词召回
原始查询与改写查询 -> ChromaDB 向量召回
两路榜单           -> 按切片 ID 做 RRF 融合 -> LLM 重排 -> 返回 Top-K
```

BM25 索引在 MCP 进程内按需从 ChromaDB 的切片构建。RRF 按各榜名次计分，每张榜最多取 20 条，向量榜共同分摊一份权重。
`channel_ranks.vector` 按原始查询、改写查询的顺序记录名次。`null` 表示该榜未命中。
`score` 是向量召回分数，只有 BM25 命中时为 `null`。`rrf_score` 对应融合排序。

## 8. ChromaDB 在项目中的用途

EchoMind 使用了三个 ChromaDB collection：

| Collection | 模块 | 作用 |
|------------|------|------|
| `knowledge_base` | `tooling/knowledge_base.py` | RAG 知识库文档片段 |
| `episodic` | `memory/conversation_memory.py` | 压缩后的历史对话摘要 |
| `user_profile` | `memory/conversation_memory.py` | 用户画像，包含偏好和关键实体 |

数据写入时机：

| 数据 | 写入时机 |
|------|----------|
| `knowledge_base` | 启动时自动导入默认文档，或调用 `/knowledge/add`、`/knowledge/upload` |
| `episodic` | 当前会话工作记忆超过阈值后自动压缩并写入 |
| `user_profile` | 每次 `/chat` 回复后异步提炼并更新 |

## 9. 在 Docker 中查看 ChromaDB 内容

Compose 中 ChromaDB 容器名是：

```text
echomind-chromadb
```

宿主机访问端口是：

```text
http://localhost:8001
```

容器内部端口是：

```text
http://localhost:8000
```

### 9.1 查看 ChromaDB 是否存活

宿主机执行：

```bash
curl http://localhost:8001/api/v1/heartbeat
```

容器内执行：

```bash
docker exec -it echomind-chromadb curl http://localhost:8000/api/v1/heartbeat
```

### 9.2 查看所有 collection

```bash
curl http://localhost:8001/api/v1/collections
```

如果 ChromaDB 版本返回 tenant/database 相关错误，可以使用 Python 客户端查看，见下一节。

### 9.3 用 Python 客户端查看 collections

进入应用容器：

```bash
docker exec -it echomind-app bash
```

在容器里执行：

```bash
python - <<'PY'
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
print("heartbeat:", client.heartbeat())

collections = client.list_collections()
print("collections:")
for c in collections:
    print("-", c.name, "count=", c.count())
PY
```

预期可以看到：

```text
collections:
- knowledge_base count= ...
- episodic count= ...
- user_profile count= ...
```

### 9.4 查看 `knowledge_base` 文档内容

```bash
docker exec -it echomind-app bash
```

执行：

```bash
python - <<'PY'
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("knowledge_base")

data = col.get(limit=10, include=["documents", "metadatas"])
for i, doc_id in enumerate(data["ids"]):
    print("=" * 80)
    print("id:", doc_id)
    print("metadata:", data["metadatas"][i])
    print("document:", data["documents"][i][:500])
PY
```

### 9.5 查询 `knowledge_base`

```bash
docker exec -it echomind-app bash
```

执行：

```bash
python - <<'PY'
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("knowledge_base")

result = col.query(
    query_texts=["退款多久到账"],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(
    result["documents"][0],
    result["metadatas"][0],
    result["distances"][0],
):
    print("=" * 80)
    print("title:", meta.get("title"))
    print("distance:", dist)
    print("content:", doc[:300])
PY
```

### 9.6 查看用户画像 `user_profile`

先多调用几次 `/chat`，让系统异步生成用户画像：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我经常咨询会员积分和退款问题，回答请简洁一点", "user_id": "profile_user", "conv_id": "profile_session"}'
```

等待几秒后查看：

```bash
docker exec -it echomind-app bash
```

```bash
python - <<'PY'
import json
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("user_profile")

data = col.get(
    where={"user_id": "profile_user"},
    include=["documents", "metadatas"],
)

for i, doc in enumerate(data["documents"]):
    print("=" * 80)
    print("metadata:", data["metadatas"][i])
    print(json.dumps(json.loads(doc), ensure_ascii=False, indent=2))
PY
```

### 9.7 查看情景记忆 `episodic`

情景记忆只有在当前会话消息数量达到压缩阈值后才会写入。默认阈值在 `MemoryManager.COMPRESS_AT` 中，目前是 15 条消息。

可以连续发送多条消息触发压缩：

```bash
for i in $(seq 1 16); do
  curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"这是第 $i 条测试消息，我想咨询退款和订单问题\", \"user_id\": \"episodic_user\", \"conv_id\": \"episodic_session\"}" > /dev/null
done
```

查看情景记忆：

```bash
docker exec -it echomind-app bash
```

```bash
python - <<'PY'
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("episodic")

data = col.get(
    where={"user_id": "episodic_user"},
    include=["documents", "metadatas"],
)

for i, doc in enumerate(data["documents"]):
    print("=" * 80)
    print("metadata:", data["metadatas"][i])
    print("summary:", doc)
PY
```

### 9.8 查看 ChromaDB 持久化文件

ChromaDB 的持久化卷在 Compose 中定义为：

```yaml
volumes:
  chromadb-data:
```

查看 Docker volume：

```bash
docker volume ls | grep chromadb
docker volume inspect echomind_chromadb-data
```

查看容器内数据目录：

```bash
docker exec -it echomind-chromadb sh
ls -lah /chroma/chroma
find /chroma/chroma -maxdepth 2 -type f | head
```

注意：不建议直接修改这些底层文件。查看和管理数据应优先使用 ChromaDB API 或 Python 客户端。

### 9.9 清空 ChromaDB 数据

谨慎操作。停止服务并删除 volume：

```bash
docker compose down
docker volume rm echomind_chromadb-data
docker compose up -d --build
```

如果只想删除某个 collection，可以用 Python 客户端：

```bash
docker exec -it echomind-app bash
```

```bash
python - <<'PY'
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
client.delete_collection("knowledge_base")
print("deleted knowledge_base")
PY
```

删除后重启应用，`KnowledgeBase` 会在 collection 为空时重新导入默认文档。

## 10. Redis 工作记忆查看

Redis 容器名：

```text
echomind-redis
```

进入 Redis：

```bash
docker exec -it echomind-redis redis-cli -a your_redis_password
```

查看 key：

```redis
KEYS *
```

工作记忆 key 格式：

```text
wm:{user_id}:{conv_id}
```

会话摘要 key 格式：

```text
summary:{user_id}:{conv_id}
```

查看某个会话最近消息：

```redis
LRANGE wm:user_001:session_001 0 -1
```

查看 TTL：

```redis
TTL wm:user_001:session_001
```

默认 TTL 是 24 小时。

## 11. 查看工作记忆压缩内容

### 通用问题处理记录

API 和 CLI 在每次回答前，会读取当前问题记录，并用本轮用户消息更新它。
该步骤独立调用一次模型，判断是在继续当前问题、提出新问题，还是仅闲聊。
记录包括问题标题、处理状态、用户报告的信息、已尝试的方法与结果、反馈、
未解决事项、下一步待办和接手过的 Agent；所有 Agent 共用这份上下文。

用户“试过了但无效”和客服“建议尝试”分开保存，用户信息和已做步骤必须附带
来自本轮用户原话的引用。上一轮客服建议只从实际展示的回复中提取，不作为已执行
操作。`resolved` 表示用户反馈问题已解决；`handoff_requested` 只表示用户请求人工，
不代表已真正接通人工。退款到账等业务结果也不能由处理记录证明。

切换到独立问题时创建新记录，旧问题仍保留；同一用户同一会话整轮串行处理。
记录写在现有 `CHROMA_PERSIST_DIRECTORY` 下独立的 `problem_records.sqlite3` 文件中，
复用已有数据挂载，无需新增依赖或环境变量。它不依赖 Redis 的 24 小时有效期，
也不会因聊天原文压缩而删除。每类明细最多保存 12 项，角色输入使用较小的记录视图；
这里保留的是有限的问题状态，并非完整聊天原文。

如果模型输出截断、JSON 格式错误或保存失败，原记录保留，聊天仍继续，响应的
`problem_record_error` 会说明本轮记录未更新。`problem_record` 返回本轮对应记录，
前端的“运行详情”中可直接查看。“客服建议与上次回复”保留实际回复；其中的结构化
历史建议会在下一轮提取。

聊天响应中的 `tool_traces` 是前端展示执行证据的数据源，Python 前端不再另查追踪接口。
每条回复和“运行详情”共用同一展示组件，可展开实际参数、完整返回结果、错误、执行
Agent、MCP 调用编号、进程 PID 和执行时间。业务工具还展示 SQLite 路径、操作编号、
读写类型和业务变更行数（不包含调用日志）；重复提交返回已有记录时显示未重复写入。
缓存、拒绝执行、执行失败和结果未确认分别展示。

问题记录中的 `facts` 继续只保存带用户原话的描述，`verified_facts` 单独保存成功 MCP
调用的参数、原始数据和回执。它由后端根据真实调用结果直接写入，不由模型生成，
也不从客服回复推断。失败、空结果、未确认结果不进入核实结果；缓存复用不重复记账。
同一 `call_id` 只保留一份，重试 checkpoint 不重复增加；切换问题不继承旧问题的证据。
每个问题最多保留最近 12 次核实结果，每次结果数据最多 6000 字节，超出时明确标记
为节选；完整返回值仍在对应聊天响应和 checkpoint 的工具追踪中。核实结果标明观察
时间，下一轮可作背景，业务状态变动前仍应重新查询。返回数据中的用户退货原因、
客服交接摘要仍是相应来源的描述，申请提交也不证明支付退款已到账。

查看同一用户同一会话最近 50 个问题记录：

```bash
curl 'http://localhost:8000/memory/problems?user_id=user_001&conv_id=session_001'
```

`active=true` 标识最近选中的问题，其状态可能已经解决。问题记录与历史摘要、用户
画像不同，不会自动跨会话续接，也不会从部署前的历史聊天自动回填。

运行记忆和问题处理记录检查（无需真实模型或数据库服务）：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

工作记忆压缩发生在 `memory/conversation_memory.py` 中。默认配置：

```text
WORKING_MAX = 20
COMPRESS_AT = 15
```

当同一个 `user_id + conv_id` 的工作记忆达到 15 条消息时，系统会：

```text
旧消息 -> LLM 摘要 -> Redis summary
旧消息摘要 -> ChromaDB episodic
最近 5 条消息 -> 继续保留在 Redis wm 列表
```

同一服务进程内，同一会话的消息写入和压缩会串行执行；压缩直接保留列表中最新的
5 条消息，读取时仍按时间顺序排列。摘要生成、摘要合并或历史保存失败时，不裁剪
原消息；恢复后会处理完整积压，而非仅处理最近 20 条。

新旧摘要会重新合并，Redis 中的会话摘要最多 800 字。历史保存成功后，摘要更新和
消息裁剪通过 Redis 事务一起提交。用户画像按用户统一更新，并带入已有画像；旧版
按会话保存的画像仍可读取，首次更新以其中最新的一份为基础生成统一画像。

历史检索优先当前会话，再补充同一用户的其他会话，并移除重复文本。在 `.env` 中设置
`MEMORY_CROSS_CONV_HISTORY=false` 可关闭跨会话补充；API 和 CLI 均支持，默认开启。
直接使用 `MemoryManager` 时可传 `cross_conv_history=False`。

会话锁和画像锁作用于单个服务进程；部署多个实例或 worker 时需要共享锁协调。

记忆回归检查（无需真实模型或数据库）：

```bash
python -m unittest discover -s tests -p 'test_memory.py' -v
```

日志示例：

```text
工作记忆压缩完成: cli_user/5a076f2b-b607-4339-9e9f-f0399862d366，摘要 19 字
```

其中：

```text
user_id = cli_user
conv_id = 5a076f2b-b607-4339-9e9f-f0399862d366
```

### 11.1 查看 Redis 中的会话摘要

进入 Redis：

```bash
docker exec -it echomind-redis redis-cli -a your_redis_password
```

查询摘要：

```redis
GET summary:cli_user:5a076f2b-b607-4339-9e9f-f0399862d366
```

一条命令快速查看：

```bash
docker exec -it echomind-redis redis-cli -a your_redis_password \
  GET summary:cli_user:5a076f2b-b607-4339-9e9f-f0399862d366
```

### 11.2 查看压缩后仍保留的最近 5 条工作记忆

进入 Redis 后执行：

```redis
LRANGE wm:cli_user:5a076f2b-b607-4339-9e9f-f0399862d366 0 -1
```

一条命令快速查看：

```bash
docker exec -it echomind-redis redis-cli -a your_redis_password \
  LRANGE wm:cli_user:5a076f2b-b607-4339-9e9f-f0399862d366 0 -1
```

说明：

- Redis 使用 `LPUSH` 写入，最新消息在列表前面。
- 代码读取时会 `reversed(raws)` 还原时间顺序。
- 压缩后 Redis 工作记忆列表只保留最近 5 条；更早的内容会以摘要形式进入 Redis summary 和 ChromaDB `episodic`。

### 11.3 查看 ChromaDB 中的情景记忆摘要

如果是全栈部署，应用容器名通常是：

```text
echomind-app
```

进入应用容器：

```bash
docker exec -it echomind-app bash
```

如果你是用 `docker run --rm` 跑 CLI，容器名可能是随机的。先查看：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Networks}}\t{{.Status}}'
```

进入对应容器：

```bash
docker exec -it <容器名> bash
```

执行 Python 脚本查询 `episodic`：

```bash
python - <<'PY'
import chromadb

user_id = "cli_user"
conv_id = "5a076f2b-b607-4339-9e9f-f0399862d366"

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("episodic")

data = col.get(
    where={"user_id": user_id},
    include=["documents", "metadatas"],
)

for i, doc in enumerate(data["documents"]):
    meta = data["metadatas"][i]
    if meta.get("conv_id") == conv_id:
        print("=" * 80)
        print("metadata:", meta)
        print("summary:", doc)
        print("full_text_preview:", meta.get("full_text"))
PY
```

字段说明：

| 字段 | 含义 |
|------|------|
| `documents[i]` | LLM 生成的历史对话摘要 |
| `metadata.user_id` | 用户 ID |
| `metadata.conv_id` | 会话 ID |
| `metadata.ts` | 写入时间 |
| `metadata.full_text` | 被压缩的原始旧消息前 500 字预览 |

### 11.4 如果只想看某个用户的所有情景记忆

```bash
docker exec -it echomind-app bash
```

```bash
python - <<'PY'
import chromadb

user_id = "cli_user"

client = chromadb.HttpClient(host="chromadb", port=8000)
col = client.get_collection("episodic")

data = col.get(
    where={"user_id": user_id},
    include=["documents", "metadatas"],
)

for i, doc in enumerate(data["documents"]):
    print("=" * 80)
    print("metadata:", data["metadatas"][i])
    print("summary:", doc)
PY
```

### 11.5 Redis summary 和 ChromaDB episodic 的区别

| 位置 | 保存内容 | 用途 |
|------|----------|------|
| Redis `summary:{user_id}:{conv_id}` | 当前会话压缩摘要 | 下一次同会话请求直接拼入 prompt |
| ChromaDB `episodic` | 压缩摘要 + metadata | 跨会话按语义检索相关历史 |
| Redis `wm:{user_id}:{conv_id}` | 最近 5 条消息 | 保持当前对话连贯性 |

## 12. Monitor 在线监控

查看监控摘要：

```bash
curl http://localhost:8000/monitor
```

响应包含：

```json
{
  "agent_stats": {
    "general_0": {
      "total": 10,
      "success_rate": 1.0,
      "avg_ms": 1200.3,
      "monitor_penalty": 0.0,
      "routing_score": 0.836
    }
  },
  "tool_stats": {
    "knowledge_search": {
      "total": 5,
      "success_rate": 1.0,
      "avg_latency_ms": 80.2,
      "consecutive_fails": 0,
      "circuit_state": "closed"
    }
  },
  "active_alerts": [],
  "suggestions": []
}
```

指标含义：

| 指标 | 含义 |
|------|------|
| `total` | 调用次数 |
| `success_rate` | 成功率 |
| `avg_ms` / `avg_latency_ms` | 平均延迟 |
| `routing_score` | Agent 路由评分 |
| `monitor_penalty` | Monitor 根据在线表现写回的降权系数 |
| `consecutive_fails` | 工具连续失败次数 |
| `circuit_state` | 工具熔断器状态，可能是 `closed`、`open`、`half_open` |

Prometheus 页面：

```text
http://localhost:9090
```

## 13. 运行 τ 零售评测

页面入口：[http://localhost:8000/eval](http://localhost:8000/eval)。
默认跑一道训练题；正式评测选择「测试集」和「整个题集」。

详见 [运行方法、官方判分与产物说明](evaluation/README.md)。

```bash
docker compose build evaluation
docker compose run --rm evaluation --split train --task-id 0 \
  --user-model deepseek/deepseek-flash --judge-model deepseek/deepseek-flash
```

用户模型和 Judge 需要对应供应商凭证。正式评测使用 `--split test --trials 3`。
官方评分器比较实际工具执行后的数据库状态与任务目标，并检查任务指定的自然语言要求。
结果保存在 `data/eval/tau/<run-id>/`，同时记录失败、执行错误和未运行任务。

## 14. 停止、重启和清理

停止服务：

```bash
docker compose stop
```

重启服务：

```bash
docker compose restart echomind
```

停止并删除容器，但保留数据卷：

```bash
docker compose down
```

停止并删除容器和数据卷：

```bash
docker compose down -v
```

重新构建并启动：

```bash
docker compose up -d --build
```

## 15. 常见问题

### 15.1 `/health` 返回 503

查看应用日志：

```bash
docker compose logs -f echomind
```

重点检查：

- `.env` 是否配置 `ANTHROPIC_API_KEY`
- Redis 是否健康
- ChromaDB 是否健康
- 应用容器是否正在反复重启

### 15.2 ChromaDB 连接失败

查看 ChromaDB 状态：

```bash
docker compose ps chromadb
docker compose logs -f chromadb
curl http://localhost:8001/api/v1/heartbeat
```

应用容器内测试：

```bash
docker exec -it echomind-app bash
python - <<'PY'
import chromadb
client = chromadb.HttpClient(host="chromadb", port=8000)
print(client.heartbeat())
PY
```

### 15.3 Redis 认证失败

确认 `.env` 和 `docker-compose.yml` 中使用的密码一致。默认密码是：

```text
your_redis_password
```

测试连接：

```bash
docker exec -it echomind-redis redis-cli -a your_redis_password ping
```

### 15.4 `/search` 没有结果

先确认知识库中有数据：

```bash
curl http://localhost:8000/knowledge/stats
```

如果是 0，可以重新导入演示文档：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@data/demo_docs/sample_knowledge.json"
```

再测试：

```bash
curl -X POST "http://localhost:8000/search?query=API如何接入&top_k=3"
```

### 15.5 用户画像查不到

用户画像是异步更新的，并且依赖 LLM 调用成功。排查步骤：

1. 先调用 `/chat`，使用固定 `user_id`
2. 等待几秒
3. 查看 `docker compose logs -f echomind` 是否出现 `用户画像已更新`
4. 使用第 8.6 节的 Python 脚本查询 `user_profile`

### 15.6 情景记忆查不到

情景记忆不是每次对话都写入。只有当前会话消息数达到压缩阈值后才写入。默认阈值：

```text
MemoryManager.COMPRESS_AT = 15
```

连续发 16 条以上消息后再查看 `episodic`。

## 16. 推荐验证流程

完整验证可以按这个顺序执行：

```bash
# 1. 启动
docker compose up -d --build

# 2. 健康检查
curl http://localhost:8000/health

# 3. 主对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，我想了解退款政策", "user_id": "demo_user", "conv_id": "demo_conv"}'

# 4. 知识库统计
curl http://localhost:8000/knowledge/stats

# 5. 导入演示知识库
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@data/demo_docs/sample_knowledge.json"

# 6. 检索
curl -X POST "http://localhost:8000/search?query=EchoMind如何接入API&top_k=3"

# 7. 监控
curl http://localhost:8000/monitor

# 8. Skills
curl http://localhost:8000/skills

# 9. 评测
docker compose run --rm evaluation --split test --plan-only
```
