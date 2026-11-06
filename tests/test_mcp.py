"""真实官方 SDK、stdio、子进程；不连接外部模型或实际业务库。"""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import mcp.types as types
from agents.agent_orchestrator import GeneralAgent, Request
from tooling.mcp_client import ToolMCPClient, MCPProtocolError, MCPTransportError
from tooling.tool_manager import MCPToolManager
from tooling.mcp_protocol import CATALOG
from test_tool_calling import model_response, tool_call

CONTEXT = {"user_id": "test-user", "conv_id": "chat", "request_id": "mcp-1", "agent_type": "general"}


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = self.make_client()
        await self.client.start()
        self.manager = MCPToolManager()
        self.manager.attach_mcp(self.client)

    def make_client(self, **kwargs):
        return ToolMCPClient(log_dir=self.temp.name, startup_timeout=5, call_timeout=3,
                             server_args=["-m", "tests.fixture_mcp_server"], **kwargs)

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    async def call(self, query="退款规则", **context):
        return await self.client.call("knowledge_search", {"query": query}, {**CONTEXT, **context})

    async def test_handshake_discovery_and_receipt_come_from_another_process(self):
        result = await self.call()
        self.assertTrue(result["success"])
        self.assertNotEqual(result["receipt"]["server_pid"], os.getpid())
        self.assertEqual(result["receipt"]["request_id"], CONTEXT["request_id"])
        self.assertEqual(self.client.status()["server_name"], "echomind-tools")
        self.assertEqual({t.name for t in self.client.tools}, set(CATALOG))
        self.assertEqual({t["name"] for t in self.manager.definitions("general")},
                         {name for name, policy in CATALOG.items() if "general" in policy.scopes})
        self.assertTrue((await self.client.check_status())["connected"])

    async def test_server_rejects_admin_tools_from_chat_scope(self):
        result = await self.client.call("knowledge_add", {"documents": [{"title": "x", "content": "x"}]}, CONTEXT)
        self.assertFalse(result["success"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["execution_status"], "rejected")

    async def test_server_validates_params_even_when_client_manager_is_bypassed(self):
        for args in ({"query": "x", "top_k": True}, {"query": "x", "top_k": 0},
                     {"query": "x", "user_id": "victim"}, {"query": " "}):
            result = await self.client.call("knowledge_search", args, CONTEXT)
            self.assertFalse(result["success"])
            self.assertFalse(result["executed"])

    async def test_server_requires_private_session_token(self):
        request = types.ClientRequest(types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(
            name="knowledge_search", arguments={"query": "x"},
            _meta={"echomind": {**CONTEXT, "call_id": "test", "session_token": "wrong"}},
        )))
        result = await self.client._session.send_request(request, types.CallToolResult)
        self.assertTrue(result.isError)
        self.assertFalse(result.structuredContent["executed"])

    async def test_manager_checks_scope_and_params_before_transport(self):
        start = self.client.call_count
        invalid = await self.manager.call("knowledge_search", {"query": "x", "top_k": False}, CONTEXT)
        forbidden = await self.manager.call("knowledge_add", {"documents": []}, CONTEXT)
        self.assertFalse(invalid.success)
        self.assertFalse(forbidden.success)
        self.assertEqual(self.client.call_count, start)

    async def test_cache_preserves_original_receipt_and_does_not_dispatch_again(self):
        first = await self.manager.call("knowledge_search", {"query": "x"}, {**CONTEXT, "tool_use_id": "a"})
        calls = self.client.call_count
        second = await self.manager.call("knowledge_search", {"query": "x"}, {**CONTEXT, "tool_use_id": "b"})
        self.assertTrue(second.cached)
        self.assertFalse(second.executed)
        self.assertEqual(second.receipt, first.receipt)
        self.assertEqual(self.client.call_count, calls)

    async def test_concurrent_calls_keep_user_and_request_context_separate(self):
        results = await asyncio.gather(*[self.call(str(i), user_id=f"user-{i}", request_id=f"r{i}") for i in range(8)])
        for i, result in enumerate(results):
            self.assertEqual(result["receipt"]["request_id"], f"r{i}")
            self.assertEqual(result["receipt"]["user_id"], f"user-{i}")

    async def test_model_loop_reports_stdio_receipt_and_server_audit(self):
        model = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[model_response(tool_call()), model_response()])))
        result = await GeneralAgent(model, "test", tool_manager=self.manager).handle(Request(
            "退款规则是什么", "test-user", "chat", request_id="native-mcp"))
        trace = result.tool_traces[0]
        self.assertTrue(result.success)
        self.assertEqual(trace["transport"], "stdio")
        self.assertNotEqual(trace["receipt"]["server_pid"], os.getpid())
        self.assertEqual(trace["receipt"]["tool_use_id"], "t1")
        rows = [json.loads(line) for line in (Path(self.temp.name) / "mcp-tools.jsonl").read_text().splitlines()]
        received = [row for row in rows if row["event"] == "received"]
        self.assertEqual(received[0]["context"]["request_id"], "native-mcp")
        self.assertEqual(received[0]["arguments"], {"query": "退款规则"})
        self.assertNotIn("session_token", json.dumps(rows))

    async def test_handler_error_is_a_failed_result_not_a_transport_failure(self):
        result = await self.call("__error__")
        self.assertFalse(result["success"])
        self.assertTrue(result["executed"])
        self.assertEqual(result["execution_status"], "failed")
        self.assertTrue((await self.call())["success"])

    async def test_process_exit_is_unknown_and_never_runs_a_local_fallback(self):
        result = await self.manager.call("knowledge_search", {"query": "__exit__"}, CONTEXT)
        self.assertFalse(result.success)
        self.assertEqual(result.transport, "stdio")
        self.assertEqual(result.execution_status, "unknown")
        self.assertFalse((await self.client.check_status())["connected"])

    async def test_transport_timeout_is_unknown_without_automatic_retry(self):
        self.client.call_timeout = 0.02
        result = await self.manager.call("knowledge_search", {"query": "__sleep__"}, CONTEXT)
        self.assertFalse(result.success)
        self.assertEqual(result.execution_status, "unknown")
        self.assertEqual(self.client.call_count, 1)

    async def test_close_from_another_task_reaps_the_process(self):
        pid = (await self.call())["receipt"]["server_pid"]
        await asyncio.create_task(self.client.close())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_child_does_not_inherit_model_keys_or_redis_secrets(self):
        await self.client.close()
        self.client = self.make_client()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sentinel", "REDIS_PASSWORD": "sentinel", "REDIS_URL": "sentinel"}):
            await self.client.start()
        result = await self.call("__env__")
        for key in ("ANTHROPIC_API_KEY", "REDIS_PASSWORD", "REDIS_URL"):
            self.assertIs(result["data"][0][key], False)
        await self.client.close()

    async def test_model_config_is_forwarded_only_when_explicitly_supplied(self):
        client = self.make_client(llm_config={"api_key": "test-key", "model": "test-model"})
        try:
            with patch.dict(os.environ, {"REDIS_PASSWORD": "sentinel", "REDIS_URL": "sentinel"}):
                await client.start()
            result = await client.call("knowledge_search", {"query": "__env__"}, CONTEXT)
            self.assertTrue(result["data"][0]["ANTHROPIC_API_KEY"])
            self.assertFalse(result["data"][0]["REDIS_PASSWORD"])
            self.assertFalse(result["data"][0]["REDIS_URL"])
        finally:
            await client.close()

    async def test_discovery_rejects_wrong_tool_contract(self):
        client = self.make_client(server_env={"FIXTURE_MODE": "wrong_tools"})
        try:
            with self.assertRaises(MCPTransportError):
                await client.start()
        finally:
            await client.close()

    async def test_process_exiting_before_handshake_fails_startup(self):
        client = ToolMCPClient(log_dir=self.temp.name, startup_timeout=2,
                               server_args=["-c", "raise SystemExit(4)"])
        try:
            with self.assertRaises(MCPTransportError):
                await client.start()
            self.assertFalse(client.status()["connected"])
        finally:
            await client.close()

    async def test_inconsistent_structured_result_is_rejected(self):
        client = self.make_client(server_env={"FIXTURE_MODE": "wrong_result"})
        try:
            await client.start()
            with self.assertRaises(MCPProtocolError):
                await client.call("knowledge_search", {"query": "x"}, CONTEXT)
        finally:
            await client.close()

    async def test_http_knowledge_management_also_uses_the_mcp_process(self):
        import api.main as api
        with patch.object(api, "_tool_manager", self.manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                added = await client.post("/knowledge/add", json={"documents": [{"title": "测试", "content": "测试文本"}]})
                self.assertEqual(added.json()["added_chunks"], 1)
                uploaded = await client.post("/knowledge/upload", files={"file": ("demo.txt", b"demo content", "text/plain")})
                self.assertEqual(uploaded.json()["total_chunks"], 2)
                stats = await client.get("/knowledge/stats")
                self.assertEqual(stats.json()["total_chunks"], 2)
        self.assertEqual(self.client.call_count, 3)
