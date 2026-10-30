"""LLM configuration and response helpers for Anthropic-compatible providers."""
import os
from typing import Any, Iterable, List


def get_max_tokens() -> int:
    """统一读取已有的 CLAUDE_MAX_TOKENS，未配置时默认 4096。"""
    try:
        value = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
    except ValueError as ex:
        raise ValueError("CLAUDE_MAX_TOKENS 必须是正整数") from ex
    if value <= 0:
        raise ValueError("CLAUDE_MAX_TOKENS 必须是正整数")
    return value


def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)
