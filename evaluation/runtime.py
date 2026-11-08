"""Construct an isolated instance of the normal HTTP chat path."""
import asyncio
import os
import socket
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
import httpx
from anthropic import AsyncAnthropic

from agents.agent_orchestrator import AgentOrchestrator
from agents.chat_workflow import ChatWorkflow
from tau2.domains.retail.environment import get_environment
from evaluation.retail import catalog
from evaluation.chroma import connect, database_name
from evaluation.meter import MeteredClient
from memory.conversation_memory import MemoryManager
from tooling.mcp_client import ToolMCPClient
from tooling.tool_manager import MCPToolManager

@asynccontextmanager
async def runtime(directory, call_limit):
    import api.main as api
    cfg = api._anthropic_cfg()
    client_cfg = {k: v for k, v in cfg.items() if k != "model"}
    client_cfg.update(max_retries=0, timeout=90)
    async with AsyncExitStack() as stack:
        client = MeteredClient(AsyncAnthropic(**client_cfg), directory / "usage.sqlite3", "system", call_limit)
        stack.push_async_callback(client.close)
        environment = get_environment()
        orchestrator = AgentOrchestrator(**cfg)
        original_clients = {agent._client for pool in orchestrator._pool.values() for agent in pool}
        original_clients.add(orchestrator._intent_recognizer.client)
        for original in original_clients:
            await original.close()
        orchestrator._intent_recognizer.client = client
        for pool in orchestrator._pool.values():
            for agent in pool:
                agent._client = client
        database = database_name(directory)
        chroma = await asyncio.to_thread(connect, database, create=True)
        prefix = f"eval:{directory.parent.name}:{directory.name}:"
        memory = MemoryManager(redis_url=os.environ["ECHOMIND_EVAL_REDIS_URL"],
                               chroma_path=str(directory / "memory"), chroma_client=chroma,
                               key_prefix=prefix, **cfg)
        stack.push_async_callback(memory.close)
        await memory._client.close()
        memory._client = client
        memory.problem_records._client = client
        await memory._redis.ping()
        manager = MCPToolManager(instructions=environment.policy)
        manager.mcp_client = ToolMCPClient(llm_config=cfg, log_dir=directory,
            catalog=catalog(environment), server_args=["-m", "evaluation.retail"],
            server_env={"ECHOMIND_EVAL_DIR": str(directory), "TAU2_DATA_DIR": os.environ["TAU2_DATA_DIR"]})
        stack.push_async_callback(manager.close)
        await manager.mcp_client.start()
        manager.attach_mcp(manager.mcp_client)
        orchestrator.set_tool_manager(manager)
        workflow = ChatWorkflow(memory, orchestrator, directory / "checkpoints.sqlite3")
        stack.push_async_callback(workflow.close)
        await workflow.start()
        # Each evaluation runs in a separate worker process. Bind the existing routes.
        api._workflow, api._memory, api._orchestrator, api._tool_manager = workflow, memory, orchestrator, manager
        endpoint = await stack.enter_async_context(serve_http(api.app))
        http = await stack.enter_async_context(httpx.AsyncClient(base_url=endpoint, timeout=300))
        yield http, workflow, {"redis_prefix": prefix, "chroma_database": database,
            "http_endpoint": endpoint, "model": cfg["model"], "system_call_limit": call_limit,
            "policy": environment.policy, "tools": sorted(catalog(environment))}
        if workflow._background:
            await asyncio.gather(*list(workflow._background))


@asynccontextmanager
async def serve_http(app):
    # Components have already been initialized with this task's isolated configuration.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("HTTP server exited before startup")
                await asyncio.sleep(0.05)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=10)
        finally:
            listener.close()
