"""官方 MCP stdio 客户端，连接的上下文由独立任务统一持有和关闭。"""
import asyncio
import json
import logging
import os
import secrets
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tooling.mcp_protocol import CATALOG

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


class MCPTransportError(RuntimeError):
    pass


class MCPProtocolError(RuntimeError):
    pass


class ToolMCPClient:
    def __init__(self, *, llm_config=None, log_dir=None, startup_timeout=30, call_timeout=None, server_args=None, server_env=None):
        self.log_dir = Path(log_dir) if log_dir else ROOT / "logs"
        self.startup_timeout, self.call_timeout = startup_timeout, call_timeout
        self.server_args = server_args or ["-m", "tooling.mcp_server"]
        self.server_env = server_env or {}
        self.llm_config = dict(llm_config or {})
        self._token = secrets.token_urlsafe(32)
        self._session = None
        self._task = None
        self._stop = asyncio.Event()
        self._ready = None
        self._closed = False
        self._broken = False
        self.tools = []
        self.server_pid = None
        self.server_name = ""
        self.protocol_version = ""
        self.call_count = 0

    def _env(self):
        config = {key: os.environ[key] for key in ("CHROMA_HOST", "CHROMA_PORT", "CHROMA_PERSIST_DIRECTORY", "LOG_LEVEL", "BUSINESS_DB_PATH")
                  if key in os.environ}
        if self.llm_config:
            config.update({"ANTHROPIC_API_KEY": self.llm_config["api_key"], "ANTHROPIC_MODEL": self.llm_config["model"]})
            if self.llm_config.get("base_url"):
                config["ANTHROPIC_BASE_URL"] = self.llm_config["base_url"]
            config["CLAUDE_MAX_TOKENS"] = os.getenv("CLAUDE_MAX_TOKENS", "4096")
        return {**config, **self.server_env, "ECHOMIND_MCP_TOKEN": self._token,
                "ECHOMIND_MCP_AUDIT_PATH": str((self.log_dir / "mcp-tools.jsonl").resolve()),
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}

    async def start(self):
        if self._closed or self._broken:
            raise MCPTransportError("MCP 客户端已关闭或连接失效")
        if self._task is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._ready = asyncio.get_running_loop().create_future()
            self._task = asyncio.create_task(self._owner())
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), self.startup_timeout)
        except BaseException:
            await self.close()
            raise

    async def _owner(self):
        try:
            params = StdioServerParameters(command=sys.executable, args=self.server_args, cwd=str(ROOT), env=self._env())
            with (self.log_dir / "mcp-server.log").open("a", encoding="utf-8") as error_log:
                async with stdio_client(params, errlog=error_log) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.startup_timeout)) as session:
                        initialized = await session.initialize()
                        listed = await session.list_tools()
                        found = {tool.name: tool for tool in listed.tools}
                        if set(found) != set(CATALOG) or len(listed.tools) != len(CATALOG):
                            raise MCPProtocolError("MCP 服务工具集合与当前合同不一致")
                        for name, policy in CATALOG.items():
                            if found[name].inputSchema != policy.schema or found[name].description != policy.description:
                                raise MCPProtocolError(f"MCP 工具合同不一致: {name}")
                        self.tools = listed.tools
                        self.server_name = initialized.serverInfo.name
                        self.protocol_version = initialized.protocolVersion
                        self._session = session
                        self._ready.set_result(None)
                        await self._stop.wait()
        except BaseException as ex:
            self._broken = True
            if not self._ready.done():
                self._ready.set_exception(MCPTransportError(f"MCP 服务启动失败: {type(ex).__name__}"))
            elif not isinstance(ex, asyncio.CancelledError):
                logger.exception("MCP 连接结束")
        finally:
            self._session = None

    async def call(self, name, arguments, context):
        if self._session is None or self._broken or self._closed:
            raise MCPTransportError("MCP 工具连接不可用")
        call_id = uuid.uuid4().hex
        meta = {"echomind": {**context, "call_id": call_id, "session_token": self._token}}
        request = types.ClientRequest(types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(
            name=name, arguments=arguments, _meta=meta,
        )))
        self.call_count += 1
        try:
            result = await self._session.send_request(
                request, types.CallToolResult,
                request_read_timeout_seconds=timedelta(seconds=self.call_timeout or CATALOG[name].timeout + 2),
            )
        except Exception as ex:
            self._broken = True
            logger.warning("MCP 传输失败 tool=%s call_id=%s type=%s", name, call_id, type(ex).__name__)
            raise MCPTransportError("MCP 调用中断或超时，执行结果未确认") from ex
        payload = result.structuredContent
        texts = [block.text for block in result.content if isinstance(block, types.TextContent)]
        if not isinstance(payload, dict) or len(texts) != 1:
            self._broken = True
            raise MCPProtocolError("MCP 服务返回格式错误")
        try:
            matches = json.loads(texts[0]) == payload
        except ValueError:
            matches = False
        receipt = payload.get("receipt", {})
        if (not matches or not isinstance(receipt, dict) or "data" not in payload or
                payload.get("execution_status") not in ("succeeded", "failed", "rejected", "unknown") or
                not isinstance(payload.get("error"), str) or type(payload.get("success")) is not bool or
                type(payload.get("reranked")) is not bool or not isinstance(payload.get("retrievals"), list) or
                type(payload.get("executed")) is not bool or result.isError != (not payload["success"]) or
                receipt.get("call_id") != call_id or receipt.get("request_id") != context.get("request_id") or
                any(receipt.get(key, "") != context.get(key, "") for key in ("user_id", "conv_id", "agent_type", "tool_use_id")) or
                receipt.get("transport") != "stdio" or
                receipt.get("tool_name") != name or type(receipt.get("server_pid")) is not int or
                not isinstance(receipt.get("completed_at"), str) or not receipt["completed_at"]):
            self._broken = True
            raise MCPProtocolError("MCP 服务返回结果或回执与请求不一致")
        if payload["execution_status"] == "unknown":
            self._broken = True
        self.server_pid = receipt["server_pid"]
        return payload

    async def check_status(self):
        if self._session is not None and not self._broken:
            try:
                await asyncio.wait_for(self._session.send_ping(), 2)
            except Exception:
                self._broken = True
        return self.status()

    def status(self):
        return {"transport": "stdio", "connected": self._session is not None and not self._broken and not self._closed,
                "server_name": self.server_name, "server_pid": self.server_pid,
                "protocol_version": self.protocol_version, "tools": [tool.name for tool in self.tools],
                "call_count": self.call_count}

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._task is not None:
            if not self._ready.done():
                self._task.cancel()
            await self._task
            if self._ready.done() and not self._ready.cancelled():
                self._ready.exception()
