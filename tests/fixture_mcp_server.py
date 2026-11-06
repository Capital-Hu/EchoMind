"""仅测试使用的服务后端；协议、权限和日志复用生产 MCP Server。"""
import asyncio
import os
import sys
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mcp.types as types
from mcp.server.stdio import stdio_server
from tooling.hybrid_retrieval import BM25Index
from tooling.mcp_server import build_server
from tooling.knowledge_service import KnowledgeService
from tooling.business_store import BusinessStore


class TestKnowledge:
    def __init__(self):
        self.documents = []

    async def doc_count_async(self):
        return len(self.documents)

    async def add_documents_async(self, documents):
        self.documents.extend({**doc, "chunk_id": f"uploaded:{i}", "chunk": 0}
                              for i, doc in enumerate(documents, start=len(self.documents)))
        return len(documents)

    async def search_bm25_async(self, query, top_k):
        return BM25Index(self.documents).search(query, top_k)

    async def search_async(self, query, top_k):
        if query == "__exit__":
            os._exit(73)
        if query == "__sleep__":
            await asyncio.sleep(2)
        if query == "__error__":
            raise RuntimeError("测试后端异常")
        if query == "__env__":
            return [{"chunk_id": "env", **{key: key in os.environ for key in (
                "ANTHROPIC_API_KEY", "REDIS_PASSWORD", "REDIS_URL")}}]
        return [{"chunk_id": f"vector:{query}", "title": "演示知识", "content": query}, *self.documents][:top_k]


class TestModel:
    def __init__(self):
        self.messages = self

    async def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        stage = "rewrite" if prompt.startswith("将以下用户查询") else "rerank"
        path = Path(os.environ["ECHOMIND_MCP_AUDIT_PATH"]).parent / "fixture-model.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({"stage": stage, "pid": os.getpid()}) + "\n")
        default = "[]" if stage == "rewrite" else json.dumps(list(range(len(re.findall(r"^\d+\. ", prompt, re.M)))))
        text = os.getenv("FIXTURE_REWRITES" if stage == "rewrite" else "FIXTURE_RANK", default)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")

    async def close(self):
        pass


async def main():
    service = KnowledgeService({"api_key": "test", "model": "test"}, client=TestModel(), knowledge=TestKnowledge())
    business = BusinessStore(Path(os.environ["ECHOMIND_MCP_AUDIT_PATH"]).parent / "business.sqlite3")
    server = build_server(service, business, os.environ["ECHOMIND_MCP_TOKEN"], os.environ["ECHOMIND_MCP_AUDIT_PATH"])
    if os.getenv("FIXTURE_MODE") == "wrong_tools":
        async def wrong(request):
            return types.ServerResult(types.ListToolsResult(tools=[]))
        server.request_handlers[types.ListToolsRequest] = wrong
    if os.getenv("FIXTURE_MODE") == "wrong_result":
        original = server.request_handlers[types.CallToolRequest]
        async def changed(request):
            result = await original(request)
            result.root.structuredContent["data"] = "被篡改的测试结果"
            return result
        server.request_handlers[types.CallToolRequest] = changed
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
