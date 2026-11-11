# EchoMind

**English** · [简体中文](README.zh-CN.md)

An AI customer support system with specialized agents, persistent conversation memory, and real tool execution. Built with FastAPI, LangGraph, Redis, ChromaDB, and the MCP Python SDK.

## What it does

- **Routes requests** to general, technical, and billing agents, with Markdown Skills for business instructions.
- **Remembers context** through Redis working memory, ChromaDB conversation summaries and user profiles, and SQLite problem records.
- **Calls tools** through a separate MCP process. Agents choose when to search knowledge, inspect orders, prepare returns, or record handoffs.
- **Searches knowledge** with query rewriting, vector and BM25 retrieval, RRF fusion, and LLM reranking.
- **Resumes conversations** from LangGraph SQLite checkpoints, with request IDs for retries and recorded tool traces.
- **Evaluates retail tasks** using a pinned official τ²-bench version, with a web interface for runs, progress, and results.

```text
HTTP /chat or CLI
  → LangGraph workflow: memory → problem record → intent → agent
  → Model-selected tools → MCP server → knowledge / business SQLite
  → Reply, tool traces, memory updates, and checkpoints
```

## Quick start

You need Docker with Docker Compose and credentials for an Anthropic-compatible model API. Run these commands from the repository root.

```bash
cp .env.example .env
```

Edit `.env`. The following configuration uses DeepSeek to match the bundled evaluation page defaults; replace the key and model placeholders with values available to your account:

```env
ANTHROPIC_API_KEY=your_deepseek_api_key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=your_deepseek_model
```

The main application also supports the official Anthropic API: omit `ANTHROPIC_BASE_URL` and use your Anthropic key and model. The evaluation page requires an explicit `ANTHROPIC_MODEL` and defaults its user/Judge models to DeepSeek; other providers need separate evaluation configuration. See the [configuration guide (Chinese)](docs/guide.zh-CN.md#section-2).

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

Compose starts the application, τ evaluation service, Redis, ChromaDB, Prometheus, and Nginx. The initial build also downloads model assets and the pinned benchmark source.

| Entry point | Purpose |
| --- | --- |
| [API docs](http://localhost:8000/docs) | Explore and call the HTTP endpoints |
| [Evaluation](http://localhost:8000/eval) | Run official retail tasks and inspect results |
| [Prometheus](http://localhost:9090) | Inspect metrics |

## Try it

Send a message:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How can I check my order?","user_id":"u1001","conv_id":"demo-chat","client_request_id":"message-1"}'
```

Keep `user_id` and `conv_id` for follow-up messages. Use a new `client_request_id` for each new message, and reuse it when retrying that same message. Responses include the reply, routing information, workflow progress, and tool traces.

Upload sample knowledge:

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F 'file=@fixtures/knowledge/sample_knowledge.json'
```

Or start an interactive CLI:

```bash
docker compose run --rm echomind python api/main.py --cli
```

## Evaluation

Open [the evaluation page](http://localhost:8000/eval), choose a task set and repeat count, then start a run. Use `train` for integration checks and `test` for benchmark runs. Results are stored in `data/eval/tau/<run-id>/`.

To generate a plan without calling models:

```bash
docker compose build evaluation
docker compose run --rm evaluation --split test --plan-only
```

The standalone CLI requires separate provider credentials for its user simulator and Judge; it does not automatically reuse the main application's key. See the [evaluation guide (Chinese)](evaluation/README.md) for configuration, scoring, tests, and recorded integration runs. A single-trial run of the 40 official test tasks passed 36 (pass^1 = 0.90) with `deepseek-flash` as system, user simulator, and Judge; the guide records the run directories and failed tasks.

## Documentation

Detailed guides are currently in Chinese.

| Guide | Contents |
| --- | --- |
| [Usage guide](docs/guide.zh-CN.md) | Deployment, API reference, knowledge import, memory inspection, monitoring, and troubleshooting |
| [Evaluation guide](evaluation/README.md) | τ²-bench setup, credentials, scoring, artifacts, and tests |
| [Skills guide](skills/README.md) | Business instructions and hot reload |

## Current scope

The repository provides HTTP APIs, a CLI, and an evaluation page; it does not include a standalone chat UI. Business tools persist changes to a local SQLite demo database with fictional orders. Payments and live human handoffs are not connected.

The HTTP API has no account authentication, and conversation locks are local to one application process. The current deployment is intended for a single application instance.
