"""用确定的模型响应和真实注册表验证 function calling 协议与执行边界。"""
import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from anthropic import AsyncAnthropic

from agents.agent_orchestrator import AgentOrchestrator, AgentType, GeneralAgent, Request, RoutingDecision
from agents.chat_workflow import ChatWorkflow
from agents.tool_calling import NativeToolLoop, ToolCallingError
from core.intent_recognizer import IntentCategory, UrgencyLevel
from mcp.tool_manager import MCPToolManager, Tool
from memory.conversation_memory import MemoryContext
from memory.problem_record import ProblemRecordManager
from test_problem_record import delta, response as problem_response


SCHEMA = {"type": "object", "properties": {
    "query": {"type": "string", "minLength": 1, "maxLength": 2000},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
}, "required": ["query"], "additionalProperties": False}
CONTEXT = {"user_id": "user", "conv_id": "chat", "request_id": "r1", "agent_type": "general"}


def tool_call(call_id="t1", name="knowledge_search", params=None):
    return {"type": "tool_use", "id": call_id, "name": name,
            "input": {"query": "退款规则"} if params is None else params}


def model_response(*blocks, stop_reason=None):
    if not blocks:
        blocks = ({"type": "text", "text": "这是根据工具结果整理的回答。"},)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason or (
        "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"))


def registry(handler=None, **kwargs):
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._tools, manager._cache = {}, {}
    handler = handler or AsyncMock(return_value=[{"title": "退款规则", "content": "申请后审核"}])
    manager.register(Tool("knowledge_search", "检索真实知识库", handler, copy.deepcopy(SCHEMA),
                          agent_types=("general", "technical", "billing"), **kwargs))
    return manager, handler


class NativeToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager, self.handler = registry()
        self.create = AsyncMock()
        self.client = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.loop = NativeToolLoop(self.client, "test-model", self.manager)

    async def run_loop(self, **kwargs):
        return await self.loop.run(messages=[{"role": "user", "content": "退款规则是什么"}],
                                   system="测试客服", context=kwargs.get("context", CONTEXT))

    async def test_model_can_answer_without_using_any_tool(self):
        self.create.return_value = model_response({"type": "text", "text": "你好"})
        result = await self.run_loop()
        self.handler.assert_not_awaited()
        self.assertEqual(result.tool_traces, [])
        self.assertFalse(result.knowledge_used)
        self.assertEqual(self.create.call_args.kwargs["tools"][0]["input_schema"], SCHEMA)
        self.assertEqual(self.create.call_args.kwargs["tool_choice"], {"type": "auto"})

    async def test_continues_after_tool_result_and_can_choose_a_second_query(self):
        thinking = {"type": "thinking", "thinking": "保留的测试思考块", "signature": "test-signature"}
        self.create.side_effect = [model_response(thinking, tool_call()),
                                   model_response(tool_call("t2", params={"query": "退款到账时间", "top_k": 2})),
                                   model_response()]
        result = await self.run_loop()
        self.assertEqual(self.handler.await_count, 2)
        self.assertEqual(self.handler.call_args_list[1].args[0]["query"], "退款到账时间")
        self.assertEqual(self.handler.call_args_list[0].args[1], CONTEXT)
        second = self.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(second[-2]["content"], [thinking, tool_call()])
        self.assertEqual(second[-1]["role"], "user")
        self.assertEqual(second[-1]["content"][0]["tool_use_id"], "t1")
        self.assertTrue(json.loads(second[-1]["content"][0]["content"])["success"])
        self.assertEqual(result.tools_used, ["knowledge_search"])
        self.assertEqual(len(result.tool_traces), 2)
        self.assertTrue(result.knowledge_used)

    async def test_all_calls_in_a_batch_get_results_including_unknown_tool(self):
        self.create.side_effect = [model_response(tool_call(), tool_call("t2", "refund_submit")), model_response()]
        result = await self.run_loop()
        returned = self.create.call_args_list[1].kwargs["messages"][-1]["content"]
        self.assertEqual([row["tool_use_id"] for row in returned], ["t1", "t2"])
        self.assertTrue(returned[1]["is_error"])
        self.assertFalse(result.tool_traces[1]["executed"])
        self.handler.assert_awaited_once()

    async def test_model_can_fix_invalid_params_after_receiving_an_error(self):
        self.create.side_effect = [model_response(tool_call(params={"query": "退款", "top_k": True})),
                                   model_response(tool_call("t2", params={"query": "退款", "top_k": 2})),
                                   model_response()]
        result = await self.run_loop()
        error = self.create.call_args_list[1].kwargs["messages"][-1]["content"][0]
        self.assertTrue(error["is_error"])
        self.assertFalse(json.loads(error["content"])["success"])
        self.handler.assert_awaited_once()
        self.assertFalse(result.tool_traces[0]["executed"])
        self.assertTrue(result.tool_traces[1]["executed"])

    async def test_timeout_fallback_is_returned_as_error_not_verified_knowledge(self):
        self.manager._tools["knowledge_search"].timeout_s = 0.01
        self.manager._tools["knowledge_search"].fallback = lambda *args: [{"fallback": True, "content": "服务暂不可用"}]
        async def hang(*args):
            await asyncio.Event().wait()
        self.handler.side_effect = hang
        self.create.side_effect = [model_response(tool_call()), model_response()]
        result = await self.run_loop()
        returned = self.create.call_args_list[1].kwargs["messages"][-1]["content"][0]
        self.assertTrue(returned["is_error"])
        self.assertIn("执行超时", returned["content"])
        self.assertFalse(result.knowledge_used)
        self.assertFalse(result.tool_traces[0]["success"])

    async def test_truncated_tool_request_is_never_executed(self):
        self.create.return_value = model_response(tool_call(), stop_reason="max_tokens")
        with self.assertRaises(ToolCallingError):
            await self.run_loop()
        self.handler.assert_not_awaited()

    async def test_duplicate_tool_ids_are_rejected_before_executing_the_batch(self):
        self.create.return_value = model_response(tool_call(), tool_call())
        with self.assertRaises(ToolCallingError):
            await self.run_loop()
        self.handler.assert_not_awaited()

    async def test_text_that_looks_like_a_call_is_not_executed(self):
        self.create.return_value = model_response({"type": "text", "text": 'knowledge_search({"query":"退款"})'})
        await self.run_loop()
        self.handler.assert_not_awaited()

    async def test_round_limit_finishes_with_tool_choice_none(self):
        self.create.side_effect = [model_response(tool_call(f"t{i}")) for i in range(4)] + [model_response()]
        result = await self.run_loop()
        self.assertEqual(self.handler.await_count, 4)
        self.assertEqual(self.create.await_count, 5)
        self.assertEqual(self.create.call_args.kwargs["tool_choice"], {"type": "none"})
        self.assertEqual(len(result.tool_traces), 4)

    async def test_batch_limit_returns_error_for_each_excess_call(self):
        self.create.side_effect = [model_response(*[tool_call(f"t{i}") for i in range(10)]), model_response()]
        result = await self.run_loop()
        self.assertEqual(self.handler.await_count, 8)
        self.assertEqual(len(result.tool_traces), 10)
        returned = self.create.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(len(returned), 10)
        self.assertTrue(all(row["is_error"] for row in returned[-2:]))
        self.assertEqual(self.create.call_args.kwargs["tool_choice"], {"type": "none"})

    async def test_model_ignoring_final_none_does_not_execute_more_tools(self):
        self.create.side_effect = [model_response(tool_call(f"t{i}")) for i in range(5)]
        with self.assertRaises(ToolCallingError) as raised:
            await self.run_loop()
        self.assertEqual(self.handler.await_count, 4)
        self.assertEqual(len(raised.exception.tool_traces), 4)

    async def test_agent_failure_keeps_completed_tool_evidence(self):
        self.create.side_effect = [model_response(tool_call()), RuntimeError("模型网络中断")]
        agent = GeneralAgent(self.client, "test-model", tool_manager=self.manager)
        result = await agent.handle(Request("退款规则是什么", "user", "chat"))
        self.assertFalse(result.success)
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.tools_used, ["knowledge_search"])
        self.assertEqual(agent.stats.success, 0)

    async def test_existing_anthropic_sdk_preserves_thinking_in_http_payload(self):
        bodies = []
        thinking = {"type": "thinking", "thinking": "测试思考", "signature": "test-signature"}
        def respond(request):
            bodies.append(json.loads(request.content))
            blocks = [thinking, tool_call()] if len(bodies) == 1 else [{"type": "text", "text": "已核实规则"}]
            return httpx.Response(200, json={"id": "msg_test", "type": "message", "role": "assistant",
                "model": "test-model", "content": blocks,
                "stop_reason": "tool_use" if len(bodies) == 1 else "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 10}})
        async with AsyncAnthropic(api_key="test", base_url="http://model.test",
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
            agent = GeneralAgent(client, "test-model", tool_manager=self.manager)
            with patch.dict(os.environ, {"CLAUDE_MAX_TOKENS": "4096"}), patch.object(NativeToolLoop, "MAX_TOOL_ROUNDS", 1):
                result = await agent.handle(Request("退款规则是什么", "user", "chat", context="原有记忆和问题记录"))
        self.assertTrue(result.success)
        self.assertEqual(bodies[1]["messages"][-2]["content"], [thinking, tool_call()])
        self.assertEqual(bodies[1]["messages"][-1]["content"][0]["tool_use_id"], "t1")
        self.assertEqual(bodies[0]["max_tokens"], 4096)
        self.assertEqual(bodies[1]["tool_choice"], {"type": "none"})
        self.assertIn("原有记忆和问题记录", bodies[0]["messages"][0]["content"])
        self.assertEqual(result.content, "已核实规则")

    async def test_concurrent_agent_calls_keep_separate_tool_results(self):
        async def respond(**kwargs):
            last = kwargs["messages"][-1]["content"]
            await asyncio.sleep(0)
            if isinstance(last, str):
                return model_response(tool_call("tool-" + last, params={"query": last}))
            data = json.loads(last[0]["content"])["data"]
            return model_response({"type": "text", "text": data[0]["content"]})
        async def search(params, context):
            await asyncio.sleep(0)
            return [{"content": params["query"]}]
        self.handler.side_effect = search
        self.create.side_effect = respond
        agent = GeneralAgent(self.client, "test-model", tool_manager=self.manager)
        first, second = await asyncio.gather(agent.handle(Request("请求A", "user", "a")),
                                             agent.handle(Request("请求B", "user", "b")))
        self.assertEqual(first.content, "请求A")
        self.assertEqual(second.content, "请求B")
        self.assertEqual(first.tool_traces[0]["input"], {"query": "请求A"})
        self.assertEqual(second.tool_traces[0]["input"], {"query": "请求B"})


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_model_params_never_reach_handler_or_fallback(self):
        manager, handler = registry()
        fallback = AsyncMock(return_value="降级")
        manager._tools["knowledge_search"].fallback = fallback
        for params in ({}, {"query": ""}, {"query": "x", "top_k": True}, {"query": "x", "top_k": 0},
                       {"query": "x", "top_k": 9}, {"query": "x", "user_id": "victim"}, [], {"query": "x" * 2001}):
            with self.subTest(params=str(params)[:80]):
                result = await manager.call("knowledge_search", params)
                self.assertFalse(result.success)
                self.assertFalse(result.executed)
        handler.assert_not_awaited()
        fallback.assert_not_awaited()
        self.assertEqual(manager._tools["knowledge_search"].breaker.fail_count, 0)

    async def test_agent_tools_are_explicitly_exposed_and_schema_is_copied(self):
        manager, _ = registry()
        manager.register(Tool("internal", "内部工具", AsyncMock(), {"type": "object"}))
        definitions = manager.definitions("general")
        self.assertEqual([row["name"] for row in definitions], ["knowledge_search"])
        definitions[0]["input_schema"]["required"].clear()
        self.assertEqual(manager.definitions("general")[0]["input_schema"]["required"], ["query"])
        self.assertEqual(manager.definitions("unknown"), [])

    async def test_cached_results_are_isolated_by_execution_context(self):
        handler = AsyncMock(side_effect=lambda params, context: {"user": context["user_id"]})
        manager, _ = registry(handler, cache_ttl=60)
        first = await manager.call("knowledge_search", {"query": "same"}, {"user_id": "a"})
        cached = await manager.call("knowledge_search", {"query": "same"}, {"user_id": "a"})
        other = await manager.call("knowledge_search", {"query": "same"}, {"user_id": "b"})
        self.assertEqual(handler.await_count, 2)
        self.assertTrue(first.executed)
        self.assertTrue(cached.cached)
        self.assertFalse(cached.executed)
        self.assertEqual(other.data, {"user": "b"})


class ToolWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_specialist_fallback_keeps_tools_already_executed(self):
        manager, handler = registry()
        orchestrator = AgentOrchestrator(api_key="test", tool_manager=manager)
        technical = orchestrator._pool[AgentType.TECHNICAL][0]
        general = orchestrator._pool[AgentType.GENERAL][0]
        technical._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
            model_response(tool_call()), RuntimeError("模型调用失败"),
        ])))
        general._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=model_response())))
        result = await orchestrator._execute(Request("无法登录", "user", "chat"), AgentType.TECHNICAL)
        self.assertTrue(result.success)
        self.assertEqual(result.agent_type, AgentType.GENERAL)
        self.assertEqual(result.tools_used, ["knowledge_search"])
        self.assertEqual(result.tool_traces[0]["agent_type"], "technical")
        handler.assert_awaited_once()

    async def test_parallel_agents_keep_both_sets_of_tool_evidence(self):
        manager, handler = registry()
        orchestrator = AgentOrchestrator(api_key="test", tool_manager=manager)
        for name in (AgentType.TECHNICAL, AgentType.BILLING):
            agent = orchestrator._pool[name][0]
            agent._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
                model_response(tool_call(name.value, params={"query": name.value})), model_response(),
            ])))
        request = Request("登录报错还重复扣款", "user", "chat", intent=IntentCategory.TECHNICAL_LOGIN)
        result = await orchestrator.run_parallel(request, RoutingDecision(AgentType.TECHNICAL, [AgentType.BILLING]))
        self.assertEqual(result.tools_used, ["knowledge_search"])
        self.assertEqual({trace["agent_type"] for trace in result.tool_traces}, {"technical", "billing"})
        self.assertTrue(result.knowledge_used)
        self.assertEqual(handler.await_count, 2)

    async def test_native_tool_evidence_reaches_http_response_and_survives_checkpoint_retry(self):
        import api.main as api
        with tempfile.TemporaryDirectory() as temp:
            manager, handler = registry()
            model = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
                model_response(tool_call(params={"query": "退款申请规则", "top_k": 2})), model_response(),
            ])))
            orchestrator = AgentOrchestrator(api_key="test", tool_manager=manager)
            for agents in orchestrator._pool.values():
                for agent in agents:
                    agent._client = model
            orchestrator.recognize_intent = AsyncMock(return_value=SimpleNamespace(
                intent=IntentCategory.REFUND, urgency=UrgencyLevel.LOW, intent_group="billing",
                confidence=0.9, entities={}, source_scores={},
            ))
            records = ProblemRecordManager(str(Path(temp) / "problems.sqlite3"),
                SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=problem_response(delta())))), "test")
            memory = SimpleNamespace(problem_records=records,
                get_context=AsyncMock(return_value=MemoryContext([], [], {}, "已有聊天背景")),
                save_exchange_once=AsyncMock(), update_profile=AsyncMock())
            path = Path(temp) / "checkpoints.sqlite3"
            workflow = ChatWorkflow(memory, orchestrator, path=path)
            payload = {"message": "退款规则是什么", "user_id": "user", "conv_id": "chat", "client_request_id": "native-1"}
            try:
                with patch.object(api, "_workflow", workflow):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                        result = await client.post("/chat", json=payload)
                self.assertEqual(result.status_code, 200, result.text)
                data = result.json()
                self.assertEqual(data["tools_used"], ["knowledge_search"])
                self.assertTrue(data["knowledge_used"])
                self.assertEqual(data["tool_traces"][0]["input"], {"query": "退款申请规则", "top_k": 2})
                self.assertIn("已有聊天背景", model.messages.create.call_args_list[0].kwargs["messages"][0]["content"])
                self.assertIn("[问题处理记录]", model.messages.create.call_args_list[0].kwargs["messages"][0]["content"])
                handler.assert_awaited_once()
                await workflow.close()
                workflow = ChatWorkflow(memory, orchestrator, path=path)
                cached = await workflow.handle_turn(payload["message"], "user", "chat", "native-1")
                self.assertTrue(cached["workflow"]["cached"])
                self.assertEqual(cached["tool_traces"], data["tool_traces"])
                handler.assert_awaited_once()
                self.assertEqual(model.messages.create.await_count, 2)
            finally:
                await workflow.close()
