"""τ 原版 Retail 环境，通过项目已有 MCP 服务执行。"""
import asyncio
import os
from pathlib import Path

from mcp.server.stdio import stdio_server
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.domains.retail.environment import get_environment

from tooling.mcp_protocol import ToolPolicy
from tooling.mcp_server import build_server


def catalog(environment):
    return {
        tool.name: ToolPolicy(
            tool.openai_schema['function']['description'],
            tool.openai_schema['function']['parameters'],
            ('general', 'billing', 'technical', 'escalation'),
        )
        for tool in environment.get_tools()
    }


class Retail:
    def __init__(self, directory):
        self.environment = get_environment()
        self.directory = Path(directory)
        (self.directory / 'tools.jsonl').touch()
        self.save('before.json')

    def save(self, name):
        (self.directory / name).write_text(self.environment.tools.db.model_dump_json(indent=2))

    async def call(self, name, arguments, context):
        call = ToolCall(id=context['tool_use_id'], name=name, arguments=arguments)
        response = self.environment.get_response(call)
        with (self.directory / 'tools.jsonl').open('a') as stream:
            stream.write(AssistantMessage(role='assistant', tool_calls=[call]).model_dump_json() + '\n')
            stream.write(response.model_dump_json() + '\n')
        self.save('after.json')
        if response.error:
            raise ValueError(response.content)
        return response.content


async def main():
    retail = Retail(os.environ['ECHOMIND_EVAL_DIR'])
    retail.save('after.json')
    server = build_server(None, retail, os.environ['ECHOMIND_MCP_TOKEN'],
                          os.environ['ECHOMIND_MCP_AUDIT_PATH'], catalog=catalog(retail.environment))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
