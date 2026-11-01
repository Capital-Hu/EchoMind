"""独立 stdio MCP 工具进程，stdout 专用于协议消息。"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mcp.types as types
from jsonschema import ValidationError
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from tooling.mcp_protocol import CATALOG, parse_meta, validate_arguments
from tooling.knowledge_service import KnowledgeService, SearchResult
from tooling.business_store import BusinessStore

logger = logging.getLogger("echomind.mcp.server")


def build_server(knowledge, business, token, audit_path):
    server = Server("echomind-tools", version="1.0.0")
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    def audit(event, **values):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
               "server_pid": os.getpid(), **values}
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=name, description=p.description, inputSchema=p.schema)
                for name, p in CATALOG.items()]

    async def call_tool(request):
        name, arguments = request.params.name, request.params.arguments
        context, public_context, executed, data, error, unknown = {}, {}, False, None, "", False
        started = time.monotonic()
        reranked, retrievals = False, []
        try:
            context = parse_meta(server.request_context.meta, token)
            public_context = {key: value for key, value in context.items() if key not in ("user_message", "previous_reply")}
            policy = CATALOG.get(name)
            if policy is None:
                raise ValueError("工具不存在")
            if context["agent_type"] not in policy.scopes:
                raise PermissionError("当前调用范围不允许使用此工具")
            validate_arguments(policy.schema, arguments)
            audit("received", tool_name=name, context=public_context, arguments=arguments)
            executed = True
            service = knowledge if name.startswith("knowledge_") else business
            data = await asyncio.wait_for(service.call(name, arguments, context), timeout=policy.timeout)
            if isinstance(data, SearchResult):
                reranked, retrievals, error = data.reranked, data.retrievals, data.error
                data = data.data
        except (ValidationError, ValueError, PermissionError) as ex:
            error = str(ex.message if isinstance(ex, ValidationError) else ex)
        except asyncio.TimeoutError:
            error = "工具服务执行超时，执行结果未确认，请勿自动重复写操作"
            unknown = True
        except Exception:
            logger.exception("工具执行失败 tool=%s request_id=%s", name, context.get("request_id"))
            error = "工具服务执行失败"
        elapsed = round((time.monotonic() - started) * 1000, 1)
        receipt = {"server_pid": os.getpid(), "tool_name": name, "transport": "stdio", **public_context,
                   "completed_at": datetime.now(timezone.utc).isoformat()}
        state = "unknown" if unknown else "rejected" if not executed else "failed" if error else "succeeded"
        payload = {"success": not bool(error), "data": data, "error": error, "executed": executed, "execution_status": state,
                   "reranked": reranked, "retrievals": retrievals,
                   "receipt": receipt, "latency_ms": elapsed}
        audit("completed" if executed else "rejected", tool_name=name, context=public_context,
              arguments=arguments,
              success=payload["success"], error=error, latency_ms=elapsed,
              reranked=reranked, retrievals=retrievals,
              database=data.get("database") if isinstance(data, dict) else None,
              result_count=len(data) if isinstance(data, list) else None)
        logger.info("tool=%s call_id=%s request_id=%s success=%s pid=%s",
                    name, context.get("call_id"), context.get("request_id"), not bool(error), os.getpid())
        return types.ServerResult(types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload, isError=bool(error),
        ))

    server.request_handlers[types.CallToolRequest] = call_tool
    return server


async def main():
    logging.basicConfig(stream=sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s mcp-server %(levelname)s %(message)s")
    token = os.environ["ECHOMIND_MCP_TOKEN"]
    config = {"api_key": os.environ["ANTHROPIC_API_KEY"], "model": os.environ["ANTHROPIC_MODEL"]}
    if os.getenv("ANTHROPIC_BASE_URL"):
        config["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    service = KnowledgeService(config)
    business = BusinessStore()
    server = build_server(service, business, token, os.environ["ECHOMIND_MCP_AUDIT_PATH"])
    logger.info("started pid=%s tools=%s", os.getpid(), list(CATALOG))
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
