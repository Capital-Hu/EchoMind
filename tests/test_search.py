"""前端演示和模型工具统一由 MCP Server 执行完整检索流程。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from agents.agent_orchestrator import GeneralAgent, Request
from tooling.knowledge_service import KnowledgeService
from tooling.mcp_client import ToolMCPClient
from tooling.tool_manager import MCPToolManager
from tests.fixture_mcp_server import TestKnowledge
from test_tool_calling import model_response, tool_call

CONTEXT = {"user_id": "test-user", "conv_id": "search-chat", "request_id": "search-test", "agent_type": "general"}


def text_response(text):
    return model_response({"type": "text", "text": text})


class EnhancedSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = ToolMCPClient(log_dir=self.temp.name, server_args=["-m", "tests.fixture_mcp_server"],
                                   server_env={"FIXTURE_REWRITES": '["query-b", "query-c"]', "FIXTURE_RANK": '[2, 0, 1]'})
        await self.client.start()
        self.manager = MCPToolManager()
        self.manager.attach_mcp(self.client)
        self.model = AsyncMock(side_effect=[text_response('["query-b", "query-c"]'), text_response('[2, 0, 1]')])
        self.service = KnowledgeService({"api_key": "test", "model": "test"},
            client=SimpleNamespace(messages=SimpleNamespace(create=self.model), close=AsyncMock()), knowledge=TestKnowledge())

    async def asyncTearDown(self):
        await self.manager.close()
        await self.service.close()
        self.temp.cleanup()

    def model_events(self):
        path = Path(self.temp.name) / "fixture-model.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    async def test_model_and_demo_search_both_run_entirely_in_same_mcp_server(self):
        import api.main as api
        agent_model = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
            model_response(tool_call(params={"query": "query-a", "top_k": 1})), model_response(),
        ])))
        with patch.object(self.service, "search_with_rewrite", side_effect=AssertionError("模型工具不能使用主进程检索实例")):
            chat = await GeneralAgent(agent_model, "test", tool_manager=self.manager).handle(
                Request("query-a", "test-user", "search-chat", request_id="chat-search"))
        self.model.assert_not_awaited()
        before = self.client.call_count
        with patch.object(api, "_tool_manager", self.manager), \
             patch("tooling.knowledge_service.KnowledgeService", side_effect=AssertionError("演示不能在主进程创建检索实例")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                manual = await client.post("/search", params={"query": "query-a", "top_k": 1})
        self.assertEqual(self.client.call_count, before + 1)
        self.assertTrue(chat.success)
        trace = chat.tool_traces[0]
        self.assertTrue(trace["result"]["reranked"])
        self.assertEqual(manual.status_code, 200, manual.text)
        self.assertTrue(manual.json()["reranked"])
        self.assertEqual(manual.json()["execution_pid"], self.client.server_pid)
        self.assertNotEqual(manual.json()["execution_pid"], os.getpid())
        self.assertEqual(manual.json()["receipt"]["server_pid"], trace["receipt"]["server_pid"])
        self.assertNotEqual(manual.json()["receipt"]["call_id"], trace["receipt"]["call_id"])
        self.assertEqual(trace["result"]["data"], manual.json()["results"])
        self.assertEqual(manual.json()["results"][0]["content"], "query-c")
        self.assertEqual([r["query"] for r in trace["retrievals"]], ["query-a", "query-b", "query-c"])
        self.assertTrue(all(r["execution_pid"] == trace["receipt"]["server_pid"] != os.getpid() for r in trace["retrievals"]))
        self.assertTrue(all(r["execution_pid"] == self.client.server_pid for r in manual.json()["retrievals"]))
        self.assertEqual(self.client.call_count, 2)
        self.assertEqual([r["stage"] for r in self.model_events()], ["rewrite", "rerank", "rewrite", "rerank"])
        self.assertTrue(all(r["pid"] == self.client.server_pid for r in self.model_events()))
        self.model.assert_not_awaited()

    async def test_demo_rejects_invalid_arguments_before_mcp_dispatch(self):
        import api.main as api
        with patch.object(api, "_tool_manager", self.manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                for params in ({"query": " "}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 9},
                               {"query": "x" * 2001}):
                    with self.subTest(params=params):
                        response = await client.post("/search", params=params)
                        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.call_count, 0)
        self.assertEqual(self.model_events(), [])

    async def test_demo_reports_mcp_search_failure(self):
        import api.main as api
        with patch.object(api, "_tool_manager", self.manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                response = await client.post("/search", params={"query": "__error__"})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("部分知识检索失败", response.json()["detail"])
        self.assertEqual(self.client.call_count, 1)
        self.model.assert_not_awaited()

    async def test_demo_reports_uninitialized_tool_service(self):
        import api.main as api
        with patch.object(api, "_tool_manager", None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                response = await client.post("/search", params={"query": "query-a"})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(self.client.call_count, 0)

    async def test_direct_mcp_call_contains_rewrite_retrieval_and_rerank(self):
        result = await self.client.call("knowledge_search", {"query": "query-a", "top_k": 1}, CONTEXT)
        self.assertTrue(result["success"])
        self.assertTrue(result["reranked"])
        self.assertEqual(result["data"][0]["content"], "query-c")
        self.assertEqual(len(result["retrievals"]), 3)
        self.assertEqual(self.client.call_count, 1)
        self.model.assert_not_awaited()

    async def test_cache_reuses_complete_server_result_without_new_mcp_or_model_calls(self):
        first = await self.manager.call("knowledge_search", {"query": "query-a", "top_k": 1}, {**CONTEXT, "tool_use_id": "first"})
        again = await self.manager.call("knowledge_search", {"query": "query-a", "top_k": 1}, {**CONTEXT, "tool_use_id": "again"})
        self.assertTrue(again.cached)
        self.assertFalse(again.executed)
        self.assertTrue(again.reranked)
        self.assertEqual(first.data, again.data)
        self.assertEqual(first.retrievals, again.retrievals)
        self.assertEqual(len(self.model_events()), 2)
        self.assertEqual(self.client.call_count, 1)
        self.model.assert_not_awaited()

    async def test_original_params_are_checked_before_dispatch(self):
        result = await self.manager.call("knowledge_search", {"query": "x", "top_k": 0}, CONTEXT)
        self.assertFalse(result.success)
        self.assertEqual(self.model_events(), [])
        self.assertEqual(self.client.call_count, 0)

    async def test_one_failed_subquery_is_not_reported_as_completed_search(self):
        self.model.side_effect = [text_response('["__error__"]')]
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertFalse(result.success)
        self.assertEqual(len(result.retrievals), 2)
        self.assertFalse(result.retrievals[1]["success"])
        self.assertFalse(result.reranked)
        self.assertEqual(self.model.await_count, 1)

    async def test_same_document_with_different_scores_is_deduplicated(self):
        document = {"title": "退款规则", "content": "测试规则", "chunk": 0}
        with patch.object(self.service, "rewrite_query", AsyncMock(return_value=["a", "b"])), \
             patch.object(self.service, "_retrieve", AsyncMock(side_effect=[
                 [{**document, "score": 0.9}], [{**document, "score": 0.6}],
             ])):
            result = await self.service.search_with_rewrite("a", 5)
        self.assertEqual(len(result.data), 1)
        self.assertFalse(result.reranked)
        self.model.assert_not_awaited()

    async def test_bad_rerank_output_does_not_claim_reranked(self):
        self.model.side_effect = [text_response('["query-b", "query-c"]'), text_response('[-1, "bad"]')]
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertTrue(result.success)
        self.assertFalse(result.reranked)
        self.assertEqual(result.data[0]["content"], "query-a")

    async def test_bad_rewrite_output_uses_original_query(self):
        self.model.side_effect = None
        self.model.return_value = text_response('invalid output')
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertTrue(result.success)
        self.assertEqual([r["query"] for r in result.retrievals], ["query-a"])
        self.assertFalse(result.reranked)

    async def test_duplicate_rank_indices_do_not_duplicate_documents(self):
        self.model.side_effect = [text_response('["query-b", "query-c"]'), text_response('[2, 2]')]
        result = await self.service.search_with_rewrite("query-a", 2)
        self.assertTrue(result.reranked)
        self.assertEqual([d["content"] for d in result.data], ["query-c", "query-a"])
