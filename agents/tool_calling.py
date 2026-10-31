"""Anthropic Messages 原生工具循环；工具定义与执行通过现有注册表提供。"""
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.llm_utils import extract_text_content, get_max_tokens

logger = logging.getLogger(__name__)


@dataclass
class AgentCallResult:
    content: str
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_used: bool = False


class ToolCallingError(RuntimeError):
    def __init__(self, message, traces):
        super().__init__(message)
        self.tool_traces = copy.deepcopy(traces)


def _content_blocks(content):
    """完整保留 thinking/signature 和 tool_use，供下一轮原样回传。"""
    blocks = []
    for block in content or []:
        if isinstance(block, dict):
            blocks.append(copy.deepcopy(block))
        elif hasattr(block, "model_dump"):
            blocks.append(block.model_dump(mode="json", exclude_none=True))
        else:
            blocks.append(copy.deepcopy(vars(block)))
    return blocks


class NativeToolLoop:
    MAX_TOOL_ROUNDS = 4
    MAX_TOOL_CALLS = 8

    def __init__(self, client, model, tool_manager=None):
        self.client, self.model, self.tool_manager = client, model, tool_manager

    async def run(self, *, messages, system, context):
        messages = copy.deepcopy(messages)
        definitions = self.tool_manager.definitions(context["agent_type"]) if self.tool_manager else []
        allowed = {tool["name"] for tool in definitions}
        traces, seen_ids = [], set()
        count = 0
        try:
            # 最后一轮只允许总结，避免工具不断自循环。
            for round_index in range(self.MAX_TOOL_ROUNDS + 1):
                final_round = round_index == self.MAX_TOOL_ROUNDS or count >= self.MAX_TOOL_CALLS
                kwargs = {
                    "model": self.model, "max_tokens": get_max_tokens(),
                    "system": system, "messages": copy.deepcopy(messages),
                }
                if definitions:
                    kwargs["tools"] = copy.deepcopy(definitions)
                    kwargs["tool_choice"] = {"type": "none" if final_round else "auto"}
                response = await self.client.messages.create(**kwargs)
                blocks = _content_blocks(response.content)
                calls = [block for block in blocks if block.get("type") == "tool_use"]
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason == "max_tokens":
                    raise RuntimeError("模型输出达到上限，未执行本轮未完成的工具请求")
                if not calls:
                    text = extract_text_content(blocks).strip()
                    if stop_reason == "tool_use" or not text:
                        raise RuntimeError("模型未返回完整的回复或工具调用")
                    return AgentCallResult(
                        content=text,
                        tools_used=list(dict.fromkeys(trace["tool_name"] for trace in traces
                                                      if trace["executed"] or trace["cached"])),
                        tool_traces=traces,
                        knowledge_used=any(
                            trace["tool_name"] == "knowledge_search" and trace["success"]
                            and isinstance(trace["result"]["data"], list) and bool(trace["result"]["data"])
                            for trace in traces
                        ),
                    )
                if final_round:
                    raise RuntimeError("模型超过工具调用上限，本轮新增调用未执行")
                ids = [call.get("id") for call in calls]
                if any(not isinstance(value, str) or not value for value in ids):
                    raise RuntimeError("模型工具调用缺少有效编号")
                if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                    raise RuntimeError("模型重复使用工具调用编号，本轮新增调用未执行")
                seen_ids.update(ids)
                messages.append({"role": "assistant", "content": blocks})
                results = []
                for call in calls:
                    name, params = call.get("name"), call.get("input")
                    started = time.monotonic()
                    executed = False
                    transport, execution_status, receipt, retrievals = "none", "rejected", {}, []
                    if count >= self.MAX_TOOL_CALLS:
                        result = {"success": False, "data": None, "error": "工具调用数量达到上限，本次未执行"}
                    elif not isinstance(name, str) or name not in allowed:
                        result = {"success": False, "data": None, "error": "工具不在当前 Agent 可用列表中"}
                    elif not isinstance(params, dict):
                        result = {"success": False, "data": None, "error": "工具参数必须是 JSON 对象"}
                    else:
                        try:
                            output = await self.tool_manager.call(name, params, context={**context, "tool_use_id": call["id"]})
                            executed = output.executed
                            transport, execution_status, receipt = output.transport, output.execution_status, output.receipt
                            retrievals = output.retrievals
                            result = {
                                "success": bool(output.success and not output.error), "data": output.data,
                                "error": output.error or "", "cached": output.cached, "reranked": output.reranked,
                                "execution_status": output.execution_status,
                            }
                        except Exception:
                            logger.exception("工具执行异常 request_id=%s tool=%s", context["request_id"], name)
                            result = {"success": False, "data": None, "error": "工具执行异常，请重试或调整请求"}
                    count += 1
                    # 证据与模型收到的结果保持一致；不把模型思考内容作为执行证据。
                    try:
                        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    except (TypeError, ValueError):
                        result = {"success": False, "data": None, "error": "工具结果无法序列化为 JSON"}
                        encoded = json.dumps(result, ensure_ascii=False)
                    trace = {
                        "agent_type": context["agent_type"], "tool_name": name, "tool_use_id": call["id"],
                        "input": copy.deepcopy(params), "executed": executed, "success": result["success"],
                        "result": copy.deepcopy(result), "error": result.get("error", ""),
                        "cached": result.get("cached", False),
                        "transport": transport, "execution_status": execution_status, "receipt": receipt,
                        "retrievals": retrievals,
                        "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    }
                    traces.append(trace)
                    logger.info("模型工具调用 request_id=%s agent=%s tool=%s success=%s cached=%s",
                                context["request_id"], context["agent_type"], name, trace["success"], trace["cached"])
                    results.append({
                        "type": "tool_result", "tool_use_id": call["id"],
                        "content": encoded, "is_error": not result["success"],
                    })
                messages.append({"role": "user", "content": results})
        except Exception as ex:
            raise ToolCallingError(str(ex), traces) from ex
