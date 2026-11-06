"""前端演示和模型工具统一由 MCP Server 执行完整检索流程。"""
import asyncio
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


class KnowledgeIndexTests(unittest.TestCase):
    def test_bm25_reads_existing_chroma_chunks_and_refreshes_after_add(self):
        import chromadb
        from tooling.knowledge_base import KnowledgeBase

        class FixedEmbedding:
            def __call__(self, input):
                return [[1.0, 0.0] for _ in input]

        with tempfile.TemporaryDirectory() as path:
            client = chromadb.PersistentClient(path=path, settings=chromadb.Settings(anonymized_telemetry=False))
            collection = client.get_or_create_collection(KnowledgeBase.COLLECTION_NAME, embedding_function=FixedEmbedding())
            collection.add(ids=["existing"], documents=["ERR_AUTH_403 表示权限不足。"],
                           metadatas=[{"title": "权限排障", "chunk_index": 0}])
            backend = SimpleNamespace(heartbeat=lambda: None, get_or_create_collection=lambda **_: collection)
            with patch("tooling.knowledge_base.chromadb.HttpClient", return_value=backend):
                knowledge = KnowledgeBase()
            self.assertEqual(knowledge.search_bm25("权限排障")[0]["chunk_id"], "existing")
            self.assertEqual(knowledge.search_bm25("err_auth_401"), [])

            knowledge.add_documents([{"title": "认证手册", "content": "ERR_AUTH_401 表示凭据失效。"}])
            lexical = knowledge.search_bm25("err_auth_401")
            vector = {row["chunk_id"]: row for row in knowledge.search("登录问题", 2)}
            self.assertEqual(len(lexical), 1)
            self.assertEqual(lexical[0]["content"], vector[lexical[0]["chunk_id"]]["content"])


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
        self.assertEqual([(r["channel"], r["query"]) for r in trace["retrievals"]],
                         [("vector", "query-a"), ("vector", "query-b"), ("vector", "query-c"), ("bm25", "query-a")])
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
        self.assertEqual(len(result["retrievals"]), 4)
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

    async def test_upload_invalidates_cache_even_when_an_old_search_is_in_flight(self):
        params = {"query": "ERR_AUTH_401", "top_k": 5}
        await self.manager.call("knowledge_search", params, CONTEXT)
        old_ready = asyncio.Event()
        release_old = asyncio.Event()
        run_handler = self.manager._run_handler

        async def hold_old_search(tool, args, context):
            result = await run_handler(tool, args, context)
            if context.get("tool_use_id") == "old-search":
                old_ready.set()
                await release_old.wait()
            return result

        with patch.object(self.manager, "_run_handler", side_effect=hold_old_search):
            old_task = asyncio.create_task(self.manager.call(
                "knowledge_search", params, {**CONTEXT, "tool_use_id": "old-search"}, use_cache=False))
            try:
                await asyncio.wait_for(old_ready.wait(), timeout=5)
                added = await self.manager.call("knowledge_add", {"documents": [
                    {"title": "认证排障", "content": "ERR_AUTH_401 表示凭据失效。"},
                ]}, {**CONTEXT, "agent_type": "admin"})
                self.assertTrue(added.success, added.error)
                fresh = await self.manager.call("knowledge_search", params, CONTEXT)
                self.assertTrue(fresh.success, fresh.error)
                self.assertFalse(fresh.cached)
                self.assertEqual(fresh.data[0]["chunk_id"], "uploaded:0")
            finally:
                release_old.set()
                old = await old_task

        self.assertNotIn("uploaded:0", [row["chunk_id"] for row in old.data])
        cached = await self.manager.call("knowledge_search", params, CONTEXT)
        self.assertTrue(cached.cached)
        self.assertEqual(cached.data, fresh.data)

    async def test_original_params_are_checked_before_dispatch(self):
        result = await self.manager.call("knowledge_search", {"query": "x", "top_k": 0}, CONTEXT)
        self.assertFalse(result.success)
        self.assertEqual(self.model_events(), [])
        self.assertEqual(self.client.call_count, 0)

    async def test_one_failed_subquery_is_not_reported_as_completed_search(self):
        self.model.side_effect = [text_response('["__error__"]')]
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertFalse(result.success)
        self.assertEqual(len(result.retrievals), 3)
        self.assertFalse(result.retrievals[1]["success"])
        self.assertFalse(result.reranked)
        self.assertEqual(self.model.await_count, 1)

    async def test_same_document_with_different_scores_is_deduplicated(self):
        document = {"chunk_id": "refund", "title": "退款规则", "content": "测试规则", "chunk": 0}
        with patch.object(self.service, "rewrite_query", AsyncMock(return_value=["a", "b"])), \
             patch.object(self.service, "_retrieve", AsyncMock(side_effect=[
                 [{**document, "score": 0.9}], [{**document, "score": 0.6}],
             ])), patch.object(self.service, "_retrieve_bm25", AsyncMock(return_value=[document])):
            result = await self.service.search_with_rewrite("a", 5)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]["channel_ranks"], {"vector": [1, 1], "bm25": 1})
        self.assertAlmostEqual(result.data[0]["rrf_score"], 2 / 61)
        self.assertEqual(result.data[0]["score"], 0.9)
        self.assertFalse(result.reranked)
        self.model.assert_not_awaited()

    async def test_bad_rerank_output_does_not_claim_reranked(self):
        self.model.side_effect = [text_response('["query-b", "query-c"]'), text_response('[-1, "bad"]')]
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertTrue(result.success)
        self.assertFalse(result.reranked)
        self.assertEqual(result.data[0]["content"], "query-a")

    async def test_rerank_timeout_returns_fused_results(self):
        async def model_call(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            if prompt.startswith("将以下用户查询"):
                return text_response('["query-b"]')
            await asyncio.Event().wait()

        self.model.side_effect = model_call
        with patch("tooling.knowledge_service._RERANK_TIMEOUT", 0.01):
            result = await self.service.search_with_rewrite("query-a", 1)
        self.assertTrue(result.success)
        self.assertFalse(result.reranked)
        self.assertEqual(result.data[0]["content"], "query-a")

    async def test_exact_error_code_missing_from_vector_results_survives_rerank_failure(self):
        knowledge = await self.service.get()
        await knowledge.add_documents_async([
            {"title": "认证手册", "content": "ERR_AUTH_401 表示凭据失效。"},
            {"title": "认证手册", "content": "ERR_AUTH_403 表示权限不足。"},
        ])
        self.model.side_effect = [text_response('["登录失败"]'), text_response('invalid ranking')]
        with patch.object(self.service, "_retrieve", AsyncMock(side_effect=[
            [{"chunk_id": "session", "title": "会话过期", "content": "登录后会话失效"}],
            [{"chunk_id": "network", "title": "网络问题", "content": "网络断开导致无法登录"}],
        ])):
            result = await self.service.search_with_rewrite("登录报 err_auth_401", 1)
        self.assertTrue(result.success)
        self.assertFalse(result.reranked)
        self.assertIn("ERR_AUTH_401", result.data[0]["content"])
        self.assertEqual(result.data[0]["channel_ranks"], {"vector": [None, None], "bm25": 1})
        self.assertIsNone(result.data[0]["score"])
        self.assertEqual(result.retrievals[-1]["result_count"], 1)

    async def test_bad_rewrite_output_uses_original_query(self):
        self.model.side_effect = None
        self.model.return_value = text_response('invalid output')
        result = await self.service.search_with_rewrite("query-a", 1)
        self.assertTrue(result.success)
        self.assertEqual([(r["channel"], r["query"]) for r in result.retrievals],
                         [("vector", "query-a"), ("bm25", "query-a")])
        self.assertFalse(result.reranked)

    async def test_duplicate_rank_indices_do_not_duplicate_documents(self):
        self.model.side_effect = [text_response('["query-b", "query-c"]'), text_response('[2, 2]')]
        result = await self.service.search_with_rewrite("query-a", 2)
        self.assertTrue(result.reranked)
        self.assertEqual([d["content"] for d in result.data], ["query-c", "query-a"])
