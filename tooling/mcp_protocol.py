"""stdio 两端共用的工具合同和调用上下文。"""
import hmac
import json
from dataclasses import dataclass

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ToolPolicy:
    description: str
    schema: dict
    scopes: tuple[str, ...]
    timeout: float = 30
    cache_ttl: float = 0


CATALOG = {
    "knowledge_search": ToolPolicy(
        "搜索平台知识库，核实政策、业务规则和办理要求。返回文档标题、正文和相关度；不能查询用户订单或执行退款。",
        {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000, "pattern": r"\S"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        }, "required": ["query"], "additionalProperties": False},
        ("general", "technical", "billing"), timeout=90, cache_ttl=300,
    ),
    "knowledge_add": ToolPolicy(
        "知识管理接口使用：向知识库导入文档。",
        {"type": "object", "properties": {"documents": {
            "type": "array", "minItems": 1, "maxItems": 100,
            "items": {"type": "object", "properties": {
                "title": {"type": "string", "maxLength": 500},
                "content": {"type": "string", "minLength": 1, "maxLength": 50000},
            }, "required": ["title", "content"], "additionalProperties": False},
        }}, "required": ["documents"], "additionalProperties": False},
        ("admin",), timeout=60,
    ),
    "knowledge_stats": ToolPolicy(
        "知识管理接口使用：读取知识库文档片段数量。",
        {"type": "object", "properties": {}, "additionalProperties": False}, ("admin",),
    ),
}


def validate_arguments(schema, arguments):
    try:
        Draft202012Validator(schema).validate(arguments)
        json.dumps(arguments, allow_nan=False)
    except (ValueError, TypeError) as ex:
        raise ValueError("工具参数不是合法 JSON") from ex


def parse_meta(meta, expected_token):
    data = meta if isinstance(meta, dict) else meta.model_dump() if meta is not None else {}
    context = data.get("echomind", {})
    token = context.get("session_token") if isinstance(context, dict) else None
    if not isinstance(token, str) or not hmac.compare_digest(token, expected_token):
        raise PermissionError("工具调用缺少有效的进程会话凭据")
    clean = {}
    for key in ("user_id", "conv_id", "request_id", "agent_type", "call_id"):
        value = context.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"工具调用上下文字段无效: {key}")
        clean[key] = value
    tool_use_id = context.get("tool_use_id", "")
    if not isinstance(tool_use_id, str) or len(tool_use_id) > 512:
        raise ValueError("工具调用编号无效")
    return {**clean, "tool_use_id": tool_use_id}
