"""MCP Server 使用的知识库业务实现：查询改写、检索、去重与重排。"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from anthropic import AsyncAnthropic
from jsonschema import ValidationError
from core.llm_utils import extract_text_content, get_max_tokens
from tooling.mcp_protocol import CATALOG, validate_arguments

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    success: bool
    data: List[Any]
    error: str = ""
    reranked: bool = False
    retrievals: List[Dict[str, Any]] = field(default_factory=list)


class KnowledgeService:
    def __init__(self, config, *, client=None, knowledge=None):
        self._config = dict(config)
        self._client = client
        self._model = config["model"]
        self._kb = knowledge
        self._lock = asyncio.Lock()

    async def get(self):
        async with self._lock:
            if self._kb is None:
                from tooling.knowledge_base import KnowledgeBase
                self._kb = await asyncio.to_thread(
                    KnowledgeBase, chroma_host=os.getenv("CHROMA_HOST", "localhost"),
                    chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
                    chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"),
                )
            return self._kb

    def _llm(self):
        if self._client is None:
            self._client = AsyncAnthropic(**{k: v for k, v in self._config.items() if k != "model"})
        return self._client

    async def _retrieve(self, query, top_k):
        return await (await self.get()).search_async(query, top_k)

    async def call(self, name, arguments, context):
        if name == "knowledge_search":
            return await self.search_with_rewrite(arguments["query"], arguments.get("top_k", 5))
        kb = await self.get()
        if name == "knowledge_stats":
            return {"total_chunks": await kb.doc_count_async()}
        count = await kb.add_documents_async(arguments["documents"])
        return {"added_chunks": count, "total_chunks": await kb.doc_count_async()}

    async def close(self):
        if self._client is not None:
            await self._client.close()

    @staticmethod
    def _clean_text(value):
        return str(value).encode("utf-8", errors="ignore").decode("utf-8")

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._llm().messages.create(
                model=self._model, max_tokens=get_max_tokens(), temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise ValueError("查询改写输出被截断")
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            if not isinstance(queries, list):
                raise ValueError("查询改写结果必须是数组")
            queries = [q.strip()[:2000] for q in queries if isinstance(q, str) and q.strip()][:n]
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排 → Top-K

        这是解决"检索不全、召回不好"的完整方案。
        """
        try:
            validate_arguments(CATALOG["knowledge_search"].schema, {"query": query, "top_k": top_k})
        except ValidationError as ex:
            raise ValueError(ex.message) from ex
        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索
        recall_k = max(top_k, 5)
        tasks = [
            self._retrieve(q, recall_k)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        retrievals = [{"query": q, "success": not isinstance(r, BaseException),
                       "error": str(r) if isinstance(r, BaseException) else "",
                       "result_count": len(r) if isinstance(r, list) else 0,
                       "execution_pid": os.getpid()}
                      for q, r in zip(sub_queries, results)]
        if any(isinstance(r, BaseException) for r in results):
            return SearchResult(False, [], error="部分知识检索失败，本次增强检索未完成", retrievals=retrievals)

        # 3. 合并去重（按文档内容去重）
        seen, merged = set(), []
        for r in results:
            if isinstance(r, list):
                for item in r:
                    # 同一文档在不同查询下分数会变化，去重不包含 score。
                    identity = {key: item.get(key) for key in ("title", "content", "chunk")} if isinstance(item, dict) else item
                    key = json.dumps(identity, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            return SearchResult(True, [], retrievals=retrievals)

        # 4. 重排：用 LLM 对合并结果按相关性打分，取 Top-K
        data, reranked = await self._rerank(query, merged, top_k)
        return SearchResult(True, data, reranked=reranked, retrievals=retrievals)

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> Tuple[List[Any], bool]:
        """
        用 LLM 对召回结果重新打分排序。

        解决问题：向量检索的相似度分数不等于"对用户有用"，
        LLM 能理解语义相关性，重排后 Top-K 质量显著提升。
        """
        if len(items) <= top_k:
            return items, False

        # 将结果序列化为文本供 LLM 评分
        items_text = "\n".join(f"{i}. {json.dumps(item, ensure_ascii=False)[:200]}"
                               for i, item in enumerate(items))
        prompt = f"""根据用户查询，对以下检索结果按相关性打分（0-10），返回 JSON 数组。
用户查询: "{query}"
检索结果:
{items_text}

返回格式（按相关性降序排列的索引列表）: [最相关的索引, ..., 最不相关的索引]
只返回 JSON 数组，不要其他文字。"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self._llm().messages.create(
                model=self._model, max_tokens=get_max_tokens(), temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise ValueError("重排输出被截断")
            s, e = raw.find("["), raw.rfind("]") + 1
            order: List[int] = json.loads(raw[s:e])
            if not isinstance(order, list) or not order or any(type(i) is not int or not 0 <= i < len(items) for i in order):
                raise ValueError("重排结果不是有效的文档索引")
            indices = list(dict.fromkeys(order + list(range(len(items)))))
            return [items[i] for i in indices[:top_k]], True
        except Exception as ex:
            logger.warning(f"重排失败，返回原始顺序: {ex}")
            return items[:top_k], False

