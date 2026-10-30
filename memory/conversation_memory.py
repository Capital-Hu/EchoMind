"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 优先检索当前会话，可选补充同一用户的跨会话历史，并去重
  - 工作记忆超过阈值时自动压缩；摘要或保存失败时保留原消息
  - 新旧摘要合并后不超过 800 字；Embedding 使用 ChromaDB 内置模型
"""
import hashlib
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import redis.asyncio as redis
from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content, get_max_tokens
from memory.problem_record import ProblemRecordManager, render_problem_record

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）
    problem_record: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.problem_record:
            parts.append(render_problem_record(self.problem_record))
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 读取上下文的窗口；压缩失败的积压仍完整保留
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    SUMMARY_MAX_CHARS = 800

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        cross_conv_history: bool = True,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._cross_conv_history = cross_conv_history
        # 锁覆盖同一服务进程内的写入和压缩；画像按用户串行合并。
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._profile_locks: Dict[str, asyncio.Lock] = {}
        self._turn_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        # 复用现有持久化目录；记录独立保存，不随 Redis 过期或摘要压缩而丢失。
        self.problem_records = ProblemRecordManager(
            str(Path(chroma_path) / "problem_records.sqlite3"), self._client, self._model,
        )

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic")
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection("user_profile")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        async with self._lock_for(user_id, conv_id):
            # 追加到 Redis 列表（左推，最新在前）
            await self._redis.lpush(key, json.dumps({
                "role":      msg.role.value,
                "content":   msg.content,
                "ts":        msg.timestamp.isoformat(),
                "metadata":  msg.metadata,
            }))
            await self._redis.expire(key, 86400)  # 24h TTL

            if await self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress_locked(user_id, conv_id)

    async def save_exchange_once(self, user_id: str, conv_id: str, message: str,
                                 reply: str, request_id: str) -> bool:
        """一轮消息与去重标记原子写入；checkpoint 恢复时不重复追加。"""
        user_id, conv_id = self._safe_text(user_id), self._safe_text(conv_id)
        identity = json.dumps([user_id, conv_id, request_id], ensure_ascii=True)
        marker = "exchange_once:" + hashlib.sha256(identity.encode()).hexdigest()
        timestamp = datetime.now().isoformat()
        rows = [json.dumps({"role": role.value, "content": self._safe_text(content),
                            "ts": timestamp, "metadata": {"request_id": request_id}})
                for role, content in ((MsgRole.USER, message), (MsgRole.ASSISTANT, reply))]
        script = """
            if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
            redis.call('LPUSH', KEYS[1], ARGV[1], ARGV[2])
            redis.call('EXPIRE', KEYS[1], ARGV[3])
            redis.call('SET', KEYS[2], '1', 'EX', ARGV[3])
            return 1
        """
        key = self._wm_key(user_id, conv_id)
        async with self._lock_for(user_id, conv_id):
            added = await self._redis.eval(script, 2, key, marker, rows[0], rows[1], 86400)
            if await self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress_locked(user_id, conv_id)
        return bool(added)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。
        用 LLM 提炼偏好，然后存入 ChromaDB（ChromaDB 内置 embedding，不依赖外部 API）。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        async with self._profile_locks.setdefault(user_id, asyncio.Lock()):
            await self._update_profile_locked(user_id, conv_id)

    async def _update_profile_locked(self, user_id: str, conv_id: str) -> None:
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        current_profile = await self._get_profile(user_id)
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        prompt = f"""结合已有用户画像和以下对话，更新用户偏好和关键实体，返回 JSON。
保留未被修改的已有信息，去除重复信息；用户明确改口时以新说法为准。
只提炼用户表达的信息，不把助手的猜测当作用户事实。
已有画像:
{json.dumps(current_profile, ensure_ascii=False)}

对话:
{text}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=get_max_tokens(), temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])
            if (not isinstance(profile_data, dict)
                    or not isinstance(profile_data.get("preferences"), list)
                    or not isinstance(profile_data.get("entities"), dict)):
                raise ValueError("用户画像格式不正确")

            doc_id = self._profile_doc_id(user_id)
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            # 同一个用户始终更新同一份画像，失败时不会先删除旧画像。
            await asyncio.to_thread(
                self._profile.upsert,
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "updated_at": datetime.now().isoformat()}],
            )
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(
            user_id, conv_id, query or (recent[-1].content if recent else "")
        )

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        async with self._lock_for(user_id, conv_id):
            await self._compress_locked(user_id, conv_id)

    async def _compress_locked(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要，合并已有摘要
          2. 先确认旧消息摘要存入情景记忆（ChromaDB）
          3. 通过 Redis 事务更新摘要并保留最近 5 条消息
          4. 摘要或保存失败时保留原消息，等待下次重试
        """
        # 压缩必须包含失败期间积压的全部消息，不能只处理上下文窗口。
        messages = await self._read_working_messages(user_id, conv_id, limit=None)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条

        # LLM 摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=get_max_tokens(), temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
            if not summary:
                raise ValueError("摘要为空")

            skey = self._summary_key(user_id, conv_id)
            old_summary = await self._redis.get(skey) or ""
            new_summary = await self._merge_summary(old_summary, summary)

            # 确认历史保存成功后，才缩短工作记忆。
            if not await self._store_episodic(user_id, conv_id, text, summary):
                return

            key = self._wm_key(user_id, conv_id)
            # 原列表已按最新在前排列，直接截取避免重建列表时颠倒顺序。
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.setex(skey, 86400, new_summary)
                pipe.ltrim(key, 0, 4)
                pipe.expire(key, 86400)
                await pipe.execute()
        except Exception as ex:
            logger.warning(f"压缩失败，保留工作记忆: {ex}")
            return
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        return await self._read_working_messages(user_id, conv_id, limit=self.WORKING_MAX)

    async def _read_working_messages(
        self, user_id: str, conv_id: str, limit: Optional[int]
    ) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = await self._redis.lrange(key, 0, -1 if limit is None else limit - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, conv_id: str, query: str) -> List[str]:
        """先查当前会话，再按开关补充同一用户的其他会话，最后去重。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = await self._query_episodic(
                query_text,
                where={"$and": [{"user_id": user_id}, {"conv_id": conv_id}]},
            )
            docs = self._dedupe_docs(results)
        except Exception as ex:
            logger.warning(f"当前会话记忆检索失败: {ex}")
            docs = []

        if self._cross_conv_history and len(docs) < self.HISTORY_TOP_K:
            try:
                results = await self._query_episodic(
                    query_text,
                    where={"$and": [{"user_id": user_id}, {"conv_id": {"$ne": conv_id}}]},
                )
                docs = list(dict.fromkeys(docs + self._dedupe_docs(results)))
            except Exception as ex:
                logger.warning(f"跨会话记忆检索失败，保留当前会话结果: {ex}")
        return docs[:self.HISTORY_TOP_K]

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> bool:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
            return True
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")
            return False

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """优先读取用户统一画像；兼容旧数据，按更新时间取最新一份。"""
        try:
            results = await asyncio.to_thread(self._profile.get, ids=[self._profile_doc_id(user_id)])
            if results["documents"]:
                return json.loads(results["documents"][0])
            results = await asyncio.to_thread(self._profile.get, where={"user_id": user_id})
            docs = results.get("documents") or []
            metadata = results.get("metadatas") or [{} for _ in docs]
            candidates = sorted(
                zip(docs, metadata),
                key=lambda item: str((item[1] or {}).get("updated_at") or (item[1] or {}).get("ts") or ""),
                reverse=True,
            )
            for doc, _ in candidates:
                try:
                    return json.loads(doc)
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
        return {}

    def _lock_for(self, user_id: str, conv_id: str) -> asyncio.Lock:
        return self._locks.setdefault((user_id, conv_id), asyncio.Lock())

    def conversation_lock(self, user_id: str, conv_id: str) -> asyncio.Lock:
        """同一进程内整轮处理串行，避免问题记录更新互相覆盖。"""
        return self._turn_locks.setdefault((user_id, conv_id), asyncio.Lock())

    @staticmethod
    def _profile_doc_id(user_id: str) -> str:
        return f"user_profile:{user_id}"

    async def _query_episodic(self, query: str, where: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._episodic.query,
            query_texts=[query], n_results=self.HISTORY_TOP_K, where=where,
        )

    @classmethod
    def _dedupe_docs(cls, results: Dict[str, Any]) -> List[str]:
        documents = results.get("documents") or []
        docs = documents[0] if documents else []
        return list(dict.fromkeys(
            cls._safe_text(doc).strip() for doc in docs
            if isinstance(doc, str) and doc.strip()
        ))

    async def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        if not old_summary.strip():
            return new_summary[:self.SUMMARY_MAX_CHARS]
        prompt = self._safe_text(f"""把旧摘要与新增摘要合并成不超过 {self.SUMMARY_MAX_CHARS} 字的会话摘要。
保留用户偏好、关键实体、已尝试的方法及结果、约束条件、待办事项与未解决的问题。
去除重复信息，用户明确更正的内容以新说法为准。只输出摘要正文。

旧摘要:
{old_summary}

新增摘要:
{new_summary}""")
        resp = await self._client.messages.create(
            model=self._model, max_tokens=get_max_tokens(), temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        merged = self._safe_text(extract_text_content(resp.content)).strip()
        if not merged:
            raise ValueError("合并摘要为空")
        return merged[:self.SUMMARY_MAX_CHARS]

    async def close(self) -> None:
        """关闭异步 Redis 连接。"""
        await self._redis.aclose()

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
