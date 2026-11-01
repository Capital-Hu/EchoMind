"""LangGraph 聊天工作流：节点完成后同步保存，按用户和会话恢复。"""
import asyncio
import copy
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from agents.agent_orchestrator import Request
from core.intent_recognizer import IntentCategory, UrgencyLevel
from memory.problem_record import render_problem_record

logger = logging.getLogger(__name__)


def checkpoint_path() -> Path:
    default = Path(__file__).resolve().parent.parent / "data" / "workflow" / "checkpoints.sqlite3"
    return Path(os.getenv("WORKFLOW_CHECKPOINT_PATH", str(default))).expanduser().resolve()


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class ChatState(TypedDict):
    user_id: str
    conv_id: str
    request_id: str
    message: str
    phase: str
    node_path: List[str]
    context: str
    recent: List[Dict[str, str]]
    history: List[Dict[str, str]]
    problem_record: Dict[str, Any]
    problem_record_error: str
    intent: Dict[str, Any]
    agent_result: Dict[str, Any]
    response: Dict[str, Any]


class WorkflowConflict(ValueError):
    pass


class WorkflowExecutionError(RuntimeError):
    pass


class ChatWorkflow:
    NODES = (
        "load_context", "prepare_problem", "recognize_intent",
        "run_agent", "save_problem_reply", "save_exchange",
    )

    def __init__(self, memory, orchestrator, path=None):
        self.memory = memory
        self.orchestrator = orchestrator
        self.path = Path(path).resolve() if path is not None else checkpoint_path()
        if self.path == Path(memory.problem_records.store.path).resolve():
            raise ValueError("checkpoint 必须与问题处理记录使用不同数据库文件")
        self._connection = None
        self._graph = None
        self._start_lock = asyncio.Lock()
        self._locks = {}
        self._running = set()
        self._background = set()
        self._closed = False

    def config(self, user_id: str, conv_id: str, request_id=None):
        identity = json.dumps(["echomind-chat-v1", user_id, conv_id], ensure_ascii=True)
        config = {"configurable": {"thread_id": hashlib.sha256(identity.encode()).hexdigest()}}
        if request_id is not None:
            config["metadata"] = {"request_id": request_id}
        return config

    async def start(self):
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("工作流已关闭")
            if self._graph is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(self.path))
            try:
                await connection.execute("PRAGMA journal_mode=WAL")
                await connection.execute("PRAGMA busy_timeout=30000")
                saver = AsyncSqliteSaver(
                    connection, serde=JsonPlusSerializer(allowed_msgpack_modules=[]),
                )
                await saver.setup()
                builder = StateGraph(ChatState)
                for name in self.NODES:
                    builder.add_node(name, getattr(self, "_" + name))
                builder.add_edge(START, self.NODES[0])
                for before, after in zip(self.NODES, self.NODES[1:]):
                    builder.add_edge(before, after)
                builder.add_edge(self.NODES[-1], END)
                self._graph = builder.compile(checkpointer=saver)
                self._connection = connection
            except BaseException:
                await connection.close()
                raise

    async def handle_turn(self, message: str, user_id: str, conv_id: str, request_id=None):
        await self.start()
        config = self.config(user_id, conv_id)
        thread_id = config["configurable"]["thread_id"]
        async with self._locks.setdefault(thread_id, asyncio.Lock()):
            with tracing_context(enabled=False):
                # 已成功的旧请求也可直接返回，不能因后续有新消息而重新执行。
                if request_id:
                    async for prior in self._graph.aget_state_history(
                        config, filter={"request_id": request_id}, limit=1,
                    ):
                        if prior.values.get("message") != message:
                            raise WorkflowConflict("同一个请求编号不能用于不同消息")
                        if not prior.next and prior.values.get("response"):
                            return self._decorate(prior, resumed=False, cached=True)
                snapshot = await self._graph.aget_state(config)
                if snapshot.next:
                    if snapshot.values.get("message") != message:
                        raise WorkflowConflict("上一条消息尚未完成，请先恢复该请求，或新建对话")
                    if request_id and request_id != snapshot.values.get("request_id"):
                        raise WorkflowConflict("当前会话有未完成请求，请使用原请求编号重试或调用恢复接口")
                    return await self._invoke(None, snapshot.values, resumed=True)
                request_id = request_id or str(uuid.uuid4())
                state = {
                    "user_id": user_id, "conv_id": conv_id, "request_id": request_id,
                    "message": message, "phase": "started", "node_path": [],
                    "context": "", "recent": [], "history": [], "problem_record": {},
                    "problem_record_error": "", "intent": {},
                    "agent_result": {}, "response": {},
                }
                return await self._invoke(state, state, resumed=False)

    async def resume(self, user_id: str, conv_id: str, request_id=None):
        await self.start()
        config = self.config(user_id, conv_id)
        thread_id = config["configurable"]["thread_id"]
        async with self._locks.setdefault(thread_id, asyncio.Lock()):
            with tracing_context(enabled=False):
                snapshot = await self._graph.aget_state(config)
                if not snapshot.values:
                    raise WorkflowConflict("该会话没有可恢复的进度")
                if request_id and snapshot.values.get("request_id") != request_id:
                    raise WorkflowConflict("请求编号与待恢复进度不一致")
                if not snapshot.next:
                    return self._decorate(snapshot, resumed=False, cached=True)
                return await self._invoke(None, snapshot.values, resumed=True)

    async def _invoke(self, value, state, resumed: bool):
        config = self.config(state["user_id"], state["conv_id"], state["request_id"])
        thread_id = config["configurable"]["thread_id"]
        self._running.add(thread_id)
        try:
            # 每个节点的 checkpoint 落盘后才进入下一个节点。
            await self._graph.ainvoke(value, config, durability="sync")
            snapshot = await self._graph.aget_state(config)
            result = self._decorate(snapshot, resumed=resumed, cached=False)
            task = asyncio.create_task(self.memory.update_profile(state["user_id"], state["conv_id"]))
            self._background.add(task)
            task.add_done_callback(self._profile_done)
            return result
        except Exception as ex:
            logger.exception("聊天工作流中断 request_id=%s", state["request_id"])
            raise WorkflowExecutionError("处理已中断；可查询已保存的进度，重试原消息或调用恢复接口") from ex
        finally:
            self._running.discard(thread_id)

    def _profile_done(self, task):
        self._background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("后台画像更新失败: %s", task.exception())

    async def status(self, user_id: str, conv_id: str):
        await self.start()
        config = self.config(user_id, conv_id)
        with tracing_context(enabled=False):
            snapshot = await self._graph.aget_state(config)
        if not snapshot.values:
            return {"exists": False, "resumable": False, "node_path": [], "next": []}
        running = config["configurable"]["thread_id"] in self._running
        return {
            "exists": True, "request_id": snapshot.values["request_id"],
            "phase": "running" if running else "interrupted" if snapshot.next else "complete",
            "last_completed_phase": snapshot.values.get("phase"),
            "node_path": snapshot.values.get("node_path", []), "next": list(snapshot.next),
            "resumable": bool(snapshot.next) and not running,
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "updated_at": snapshot.created_at,
        }

    @staticmethod
    def _decorate(snapshot, resumed: bool, cached: bool):
        result = copy.deepcopy(snapshot.values["response"])
        result["workflow"] = {
            "mode": "langgraph", "phase": "complete", "resumed": resumed, "cached": cached,
            "request_id": snapshot.values["request_id"],
            "node_path": snapshot.values["node_path"], "next": list(snapshot.next),
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        }
        return result

    @staticmethod
    def _step(state, name, phase, **values):
        return {"node_path": state["node_path"] + [name], "phase": phase, **values}

    async def _load_context(self, state):
        context = await self.memory.get_context(state["user_id"], state["conv_id"], query=state["message"])
        recent = [{"role": message.role.value, "content": message.content}
                  for message in context.recent_messages[-5:]]
        return self._step(state, "load_context", "context_loaded", context=context.to_prompt_text(), recent=recent)

    async def _prepare_problem(self, state):
        prepared = await self.memory.problem_records.prepare(
            state["user_id"], state["conv_id"], state["message"], state["recent"][-4:],
            operation_id=state["request_id"],
        )
        record_text = render_problem_record(prepared.record)
        history = state["recent"]
        if record_text:
            history = [{"role": "system", "content": record_text}] + history[-2:]
        context = "\n\n".join(part for part in (state["context"], record_text) if part)
        return self._step(state, "prepare_problem", "problem_prepared", context=context, history=history,
                          problem_record=prepared.record, problem_record_error=prepared.error)

    async def _recognize_intent(self, state):
        result = await self.orchestrator.recognize_intent(state["message"], history=state["history"] or None)
        return self._step(state, "recognize_intent", "intent_recognized", intent={
            "intent": result.intent.value, "urgency": result.urgency.value,
            "intent_group": result.intent_group, "confidence": result.confidence,
            "entities": result.entities, "source_scores": result.source_scores,
        })

    async def _run_agent(self, state):
        intent = state["intent"]
        request = Request(
            message=state["message"], user_id=state["user_id"], conv_id=state["conv_id"],
            request_id=state["request_id"], context=state["context"],
            history=state["history"] or None, entities=intent["entities"],
            intent=IntentCategory(intent["intent"]), intent_group=intent["intent_group"],
            urgency=UrgencyLevel(intent["urgency"]), intent_confidence=intent["confidence"],
        )
        result = await self.orchestrator.run(request)
        return self._step(state, "run_agent", "answered", agent_result=_json_value(asdict(result)))

    async def _save_problem_reply(self, state):
        result = state["agent_result"]
        recorded = await self.memory.problem_records.finish(
            state["user_id"], state["conv_id"], state["problem_record"],
            result["response"], result["agent_type"], result["agent_types"],
            tool_traces=result["tool_traces"], request_id=state["request_id"],
        )
        return self._step(state, "save_problem_reply", "problem_saved", problem_record=recorded.record,
                          problem_record_error=state["problem_record_error"] or recorded.error)

    async def _save_exchange(self, state):
        result, intent = state["agent_result"], state["intent"]
        await self.memory.save_exchange_once(
            state["user_id"], state["conv_id"], state["message"], result["response"], state["request_id"],
        )
        response = {
            "conv_id": state["conv_id"], "response": result["response"],
            "intent": result["intent"] or "other", "intent_group": intent["intent_group"],
            "agent_type": result["agent_type"], "agent_types": result["agent_types"],
            "primary_agent": result["primary_agent"] or result["agent_type"],
            "supporting_agents": result["supporting_agents"], "routing_reason": result["routing_reason"],
            "routing_confidence": result["routing_confidence"], "escalated": result["escalated"],
            "latency_ms": round(result["latency_ms"], 1),
            "knowledge_used": result["knowledge_used"],
            "tools_used": result["tools_used"], "tool_traces": result["tool_traces"],
            "entities": intent["entities"], "intent_confidence": round(intent["confidence"], 4),
            "intent_source_scores": intent["source_scores"], "problem_record": state["problem_record"],
            "problem_record_error": state["problem_record_error"],
        }
        return self._step(state, "save_exchange", "complete", response=response)

    async def close(self):
        self._closed = True
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._graph = None
