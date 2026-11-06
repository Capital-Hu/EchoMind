"""
EchoMind 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from jsonschema import ValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from core.llm_utils import get_max_tokens
from agents.chat_workflow import ChatWorkflow, WorkflowConflict, WorkflowExecutionError
from tooling.mcp_protocol import CATALOG, validate_arguments

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   EchoMind  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_workflow = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    get_max_tokens()  # 启动时校验，避免配置错误被后续各阶段降级掩盖。
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _tool_context(scope, source):
    return {"user_id": "system", "conv_id": source, "request_id": uuid.uuid4().hex, "agent_type": scope}


async def _create_tool_manager(cfg):
    from tooling.mcp_client import ToolMCPClient
    from tooling.tool_manager import MCPToolManager

    manager = MCPToolManager()
    manager.mcp_client = ToolMCPClient(llm_config=cfg)
    try:
        await manager.mcp_client.start()
        manager.attach_mcp(manager.mcp_client)
        stats = await manager.call("knowledge_stats", {}, _tool_context("admin", "startup"), use_cache=False)
        if not stats.success:
            raise RuntimeError(f"MCP 知识服务未就绪: {stats.error}")
        business = await manager.call("business_stats", {}, _tool_context("admin", "startup"), use_cache=False)
        if not business.success:
            raise RuntimeError(f"MCP 业务库未就绪: {business.error}")
        logger.info("MCP 业务库: %s，订单: %s", business.data["path"], business.data["counts"]["orders"])
        logger.info("MCP stdio 已连接: %s，知识片段: %s", manager.mcp_client.status(), stats.data["total_chunks"])
        return manager
    except BaseException:
        await manager.close()
        raise


async def _admin_tool_call(name, params):
    if _tool_manager is None:
        raise HTTPException(503, "工具服务未初始化")
    result = await _tool_manager.call(name, params, _tool_context("admin", "knowledge-api"), use_cache=False)
    if not result.success:
        raise HTTPException(503, result.error or "工具服务调用失败")
    return result.data


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _workflow

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")
    logger.info(f"模型统一输出上限: {get_max_tokens()} tokens")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        cross_conv_history=os.getenv("MEMORY_CROSS_CONV_HISTORY", "true").lower() == "true",
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    async with AsyncExitStack() as resources:
        resources.push_async_callback(_memory.close)
        _tool_manager = await _create_tool_manager(cfg)
        resources.push_async_callback(_tool_manager.close)
        _orchestrator.set_tool_manager(_tool_manager)

        # 性能监控（可选启动 Prometheus）
        prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
        _monitor = PerformanceMonitor(
            orchestrator=_orchestrator,
            tool_manager=_tool_manager,
            interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
            prometheus_port=prom_port,
        )
        resources.push_async_callback(_monitor.stop)
        await _monitor.start()

        # 评测器
        _evaluator = EndToEndEvaluator(
            orchestrator=_orchestrator,
            recognizer=recognizer,
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
            baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        )

        _workflow = ChatWorkflow(_memory, _orchestrator)
        resources.push_async_callback(_workflow.close)
        await _workflow.start()
        logger.info("LangGraph checkpoint: %s", _workflow.path)
        logger.info("EchoMind 已就绪")
        yield
    _workflow = None
    logger.info("EchoMind 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoMind 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None
    client_request_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")


class WorkflowResumeRequest(BaseModel):
    user_id: str = "anonymous"
    conv_id: str
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    problem_record: Dict[str, Any] = Field(default_factory=dict)
    problem_record_error: str = ""
    workflow: Dict[str, Any] = Field(default_factory=dict)
    tools_used: List[str] = Field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    status = await _tool_manager.mcp_client.check_status()
    if not status["connected"]:
        raise HTTPException(503, "MCP 工具服务不可用")
    return {"status": "ok", "agents": _orchestrator.get_stats(), "mcp": status}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """LangGraph 持久化执行；用同一 client_request_id 重试可恢复或返回已完成结果。"""
    if _workflow is None:
        raise HTTPException(503, "服务未就绪")
    conv_id = req.conv_id or (
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"echomind:{req.user_id}:{req.client_request_id}"))
        if req.client_request_id else str(uuid.uuid4())
    )
    return await _workflow_response(_workflow.handle_turn(
        req.message, req.user_id, conv_id, req.client_request_id,
    ))


async def _workflow_response(call):
    try:
        return ChatResponse(**await call)
    except WorkflowConflict as ex:
        raise HTTPException(409, str(ex)) from ex
    except WorkflowExecutionError as ex:
        raise HTTPException(503, str(ex)) from ex


@app.get("/workflow/status", tags=["Workflow"])
async def workflow_status(user_id: str, conv_id: str):
    if _workflow is None:
        raise HTTPException(503, "服务未就绪")
    return await _workflow.status(user_id, conv_id)


@app.post("/workflow/resume", response_model=ChatResponse, tags=["Workflow"])
async def workflow_resume(req: WorkflowResumeRequest):
    if _workflow is None:
        raise HTTPException(503, "服务未就绪")
    return await _workflow_response(_workflow.resume(req.user_id, req.conv_id, req.request_id))


@app.get("/memory/problems", tags=["Memory"])
async def problem_records(user_id: str, conv_id: str):
    """查看同一用户同一会话的问题处理记录，包含已切换的旧问题。"""
    if _memory is None:
        raise HTTPException(503, "服务未就绪")
    try:
        records = await asyncio.to_thread(_memory.problem_records.store.list, user_id, conv_id)
    except Exception:
        raise HTTPException(503, "问题处理记录暂时无法读取")
    return {"records": records}


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示混合检索：查询改写 → 向量与 BM25 召回 → RRF 融合 → 重排。
    此接口主动调用 MCP knowledge_search；聊天里由模型选择同一工具。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    params = {"query": query, "top_k": top_k}
    try:
        validate_arguments(CATALOG["knowledge_search"].schema, params)
    except (ValidationError, ValueError) as ex:
        raise HTTPException(422, ex.message if isinstance(ex, ValidationError) else str(ex)) from ex
    result = await _tool_manager.call("knowledge_search", params, _tool_context("general", "search-api"), use_cache=False)
    if not result.success:
        raise HTTPException(503, result.error or "知识检索失败")
    return {"query": query, "results": result.data, "reranked": result.reranked,
            "retrievals": result.retrievals, "execution_pid": result.receipt["server_pid"],
            "receipt": result.receipt}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    data = await _admin_tool_call("knowledge_add", {"documents": [d.model_dump() for d in body.documents]})
    return {"message": f"成功导入 {data['added_chunks']} 个文档片段", **data}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    data = await _admin_tool_call("knowledge_add", {"documents": docs})
    return {
        "message": f"文件 {filename} 导入成功",
        **data,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    return await _admin_tool_call("knowledge_stats", {})


@app.get("/business/stats", tags=["业务库"])
async def business_stats():
    """通过 MCP 查看业务 SQLite 路径、演示数据来源和表数量。"""
    return await _admin_tool_call("business_stats", {})


@app.get("/mcp/status", tags=["MCP"])
async def mcp_status():
    if _tool_manager is None:
        raise HTTPException(503, "工具服务未初始化")
    return await _tool_manager.mcp_client.check_status()


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli(user_id: str = "cli_user", conv_id: Optional[str] = None):
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出，/resume 恢复未完成请求\n")

    from agents.agent_orchestrator import AgentOrchestrator
    from memory.conversation_memory import MemoryManager
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        cross_conv_history=os.getenv("MEMORY_CROSS_CONV_HISTORY", "true").lower() == "true",
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    async with AsyncExitStack() as resources:
        resources.push_async_callback(mem.close)
        tools = await _create_tool_manager(cfg)
        resources.push_async_callback(tools.close)
        orch.set_tool_manager(tools)
        conv_id = conv_id or str(uuid.uuid4())
        print(f"当前会话：{conv_id}（重启时可用 --conv-id 接续）")
        workflow = ChatWorkflow(mem, orch)
        resources.push_async_callback(workflow.close)
        await workflow.start()
        while True:
            try:
                msg = (await asyncio.to_thread(input, "你: ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg or msg.lower() in ("quit", "exit", "退出"):
                break
            try:
                result = await workflow.resume(user_id, conv_id) if msg == "/resume" else await workflow.handle_turn(msg, user_id, conv_id)
                print(f"\nEchoMind [{result['agent_type']}]: {result['response']}\n")
                if result["problem_record_error"]:
                    print(f"处理记录：{result['problem_record_error']}\n")
            except (WorkflowConflict, WorkflowExecutionError) as ex:
                print(f"处理提示：{ex}\n")
    print("再见 ʕ•ᴥ•ʔ")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cli", action="store_true")
        parser.add_argument("--user-id", default="cli_user")
        parser.add_argument("--conv-id")
        args = parser.parse_args()
        asyncio.run(_cli(args.user_id, args.conv_id))
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
