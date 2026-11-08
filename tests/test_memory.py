"""记忆行为回归检查：用替身模拟模型和存储，不调用外部服务。"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from memory.conversation_memory import MemoryManager, MsgRole


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.values = {}
        self.expires = {}
        self.fail_commit = False

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def expire(self, key, ttl):
        self.expires[key] = ttl

    async def get(self, key):
        return self.values.get(key)

    def pipeline(self, transaction=True):
        if not transaction:
            raise AssertionError("压缩必须通过事务提交")
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def setex(self, key, ttl, value):
        self.commands.append(("setex", key, ttl, value))

    def ltrim(self, key, start, end):
        self.commands.append(("ltrim", key, start, end))

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))

    async def execute(self):
        if self.redis.fail_commit:
            raise RuntimeError("Redis unavailable before commit")
        for command in self.commands:
            kind, key, *args = command
            if kind == "setex":
                ttl, value = args
                self.redis.values[key] = value
                self.redis.expires[key] = ttl
            elif kind == "ltrim":
                start, end = args
                self.redis.lists[key] = self.redis.lists[key][start:end + 1]
            else:
                self.redis.expires[key] = args[0]


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.fail_write = False
        self.query_calls = []
        self.query_results = []

    def add(self, ids, documents, metadatas):
        self.upsert(ids, documents, metadatas)

    def upsert(self, ids, documents, metadatas):
        if self.fail_write:
            raise RuntimeError("Chroma unavailable")
        for doc_id, doc, metadata in zip(ids, documents, metadatas):
            self.rows[doc_id] = (doc, metadata)

    def get(self, ids=None, where=None):
        rows = [row for key, row in self.rows.items()
                if (ids is None or key in ids)
                and (where is None or all(row[1].get(k) == v for k, v in where.items()))]
        return {"documents": [r[0] for r in rows], "metadatas": [r[1] for r in rows]}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        result = self.query_results.pop(0) if self.query_results else []
        if isinstance(result, Exception):
            raise result
        return {"documents": [result]}


def response(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def make_manager(cross_conv=True):
    manager = MemoryManager.__new__(MemoryManager)
    manager._key_prefix = ""
    manager._redis = FakeRedis()
    manager._episodic = FakeCollection()
    manager._profile = FakeCollection()
    manager._client = SimpleNamespace(messages=SimpleNamespace(
        create=AsyncMock(return_value=response("本次摘要"))
    ))
    manager._model = "test-model"
    manager._locks = {}
    manager._profile_locks = {}
    manager._cross_conv_history = cross_conv
    return manager


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def fill(self, manager, count, conv="chat", start=0):
        for index in range(start, start + count):
            await manager.add_message("user", conv, MsgRole.USER, f"m{index}")

    async def contents(self, manager, conv="chat", all_messages=False):
        messages = await manager._read_working_messages(
            "user", conv, limit=None if all_messages else manager.WORKING_MAX
        )
        return [m.content for m in messages]

    async def test_compression_keeps_latest_five_in_time_order(self):
        manager = make_manager()
        await self.fill(manager, 15)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(10, 15)])
        self.assertEqual(len(manager._episodic.rows), 1)
        self.assertEqual(await manager._redis.get("summary:user:chat"), "本次摘要")

    async def test_append_during_compression_waits_and_is_not_lost(self):
        manager = make_manager()
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_summary(**kwargs):
            started.set()
            await release.wait()
            return response("摘要")

        manager._client.messages.create.side_effect = slow_summary
        await self.fill(manager, 14)
        compressing = asyncio.create_task(self.fill(manager, 1, start=14))
        await asyncio.wait_for(started.wait(), 1)
        appending = asyncio.create_task(self.fill(manager, 1, start=15))
        await asyncio.sleep(0)
        self.assertFalse(appending.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(compressing, appending), 1)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(10, 16)])

    async def test_other_conversations_are_not_blocked(self):
        manager = make_manager()
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_summary(**kwargs):
            started.set()
            await release.wait()
            return response("摘要")

        manager._client.messages.create.side_effect = slow_summary
        await self.fill(manager, 14)
        compressing = asyncio.create_task(self.fill(manager, 1, start=14))
        await asyncio.wait_for(started.wait(), 1)
        try:
            await asyncio.wait_for(self.fill(manager, 1, conv="other"), 1)
        finally:
            release.set()
            await compressing
        self.assertEqual(await self.contents(manager, "other"), ["m0"])

    async def test_summary_error_or_empty_output_preserves_messages(self):
        for failure in (RuntimeError("model unavailable"), response("  ")):
            with self.subTest(failure=failure):
                manager = make_manager()
                manager._client.messages.create.side_effect = [failure]
                await self.fill(manager, 15)
                self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(15)])
                self.assertFalse(manager._episodic.rows)
                self.assertFalse(manager._redis.values)

    async def test_history_storage_failure_preserves_messages_and_old_summary(self):
        manager = make_manager()
        manager._redis.values["summary:user:chat"] = "旧摘要"
        manager._episodic.fail_write = True
        await self.fill(manager, 15)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(15)])
        self.assertEqual(await manager._redis.get("summary:user:chat"), "旧摘要")

    async def test_redis_commit_failure_preserves_messages(self):
        manager = make_manager()
        manager._redis.fail_commit = True
        await self.fill(manager, 15)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(15)])
        self.assertFalse(manager._redis.values)

    async def test_recovery_compresses_entire_backlog_beyond_context_window(self):
        manager = make_manager()
        manager._client.messages.create.side_effect = RuntimeError("model unavailable")
        await self.fill(manager, 25)
        self.assertEqual(len(await self.contents(manager, all_messages=True)), 25)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(5, 25)])
        manager._client.messages.create.side_effect = None
        await self.fill(manager, 1, start=25)
        self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(21, 26)])
        prompt = manager._client.messages.create.call_args.kwargs["messages"][0]["content"]
        for index in range(21):
            self.assertIn(f"user: m{index}\n", prompt + "\n")

    async def test_new_and_merged_summaries_never_exceed_800_characters(self):
        manager = make_manager()
        manager._client.messages.create.return_value = response("新" * 1000)
        await self.fill(manager, 15)
        self.assertEqual(len(await manager._redis.get("summary:user:chat")), 800)
        manager._client.messages.create.side_effect = [response("补充信息"), response("合" * 900)]
        await self.fill(manager, 10, start=15)
        summary = await manager._redis.get("summary:user:chat")
        self.assertEqual(summary, "合" * 800)
        prompt = manager._client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("新" * 800, prompt)
        self.assertIn("补充信息", prompt)

    async def test_merge_failure_keeps_old_summary_and_all_messages(self):
        for failure in (RuntimeError("merge failed"), response("")):
            with self.subTest(failure=failure):
                manager = make_manager()
                manager._redis.values["summary:user:chat"] = "已有摘要"
                manager._client.messages.create.side_effect = [response("新增摘要"), failure]
                await self.fill(manager, 15)
                self.assertEqual(await self.contents(manager), [f"m{i}" for i in range(15)])
                self.assertEqual(await manager._redis.get("summary:user:chat"), "已有摘要")
                self.assertFalse(manager._episodic.rows)

    async def test_history_prioritizes_current_chat_and_deduplicates(self):
        manager = make_manager()
        manager._episodic.query_results = [[" 本次问题 ", "本次问题"], ["本次问题", "旧问题", "旧问题"]]
        docs = await manager._search_episodic("user", "chat", "问题")
        self.assertEqual(docs, ["本次问题", "旧问题"])
        calls = manager._episodic.query_calls
        self.assertEqual(calls[0]["where"], {"$and": [{"user_id": "user"}, {"conv_id": "chat"}]})
        self.assertEqual(calls[1]["where"], {"$and": [{"user_id": "user"}, {"conv_id": {"$ne": "chat"}}]})

    async def test_disabled_cross_chat_search_never_queries_other_chats(self):
        manager = make_manager(cross_conv=False)
        manager._episodic.query_results = [["本次问题"]]
        self.assertEqual(await manager._search_episodic("user", "chat", "问题"), ["本次问题"])
        self.assertEqual(len(manager._episodic.query_calls), 1)

    async def test_full_current_chat_results_do_not_need_cross_chat_search(self):
        manager = make_manager()
        manager._episodic.query_results = [[f"摘要{i}" for i in range(5)]]
        self.assertEqual(len(await manager._search_episodic("user", "chat", "问题")), 5)
        self.assertEqual(len(manager._episodic.query_calls), 1)

    async def test_cross_chat_search_failure_keeps_current_chat_results(self):
        manager = make_manager()
        manager._episodic.query_results = [["本次问题"], RuntimeError("search failed")]
        self.assertEqual(await manager._search_episodic("user", "chat", "问题"), ["本次问题"])

    async def test_current_chat_search_failure_can_fall_back_to_user_history(self):
        manager = make_manager()
        manager._episodic.query_results = [RuntimeError("search failed"), ["其他聊天"]]
        self.assertEqual(await manager._search_episodic("user", "chat", "问题"), ["其他聊天"])

    async def test_blank_query_does_not_search(self):
        manager = make_manager()
        self.assertEqual(await manager._search_episodic("user", "chat", "  "), [])
        self.assertFalse(manager._episodic.query_calls)

    async def test_profile_updates_merge_existing_data_into_one_user_document(self):
        manager = make_manager()
        old = {"preferences": ["简洁"], "entities": {}}
        new = {"preferences": ["简洁", "中文"], "entities": {"产品": ["耳机"]}}
        manager._client.messages.create.side_effect = [response(json.dumps(old)), response(json.dumps(new))]
        await self.fill(manager, 1)
        await manager.update_profile("user", "chat")
        await self.fill(manager, 1, conv="other")
        await manager.update_profile("user", "other")
        self.assertEqual(list(manager._profile.rows), ["user_profile:user"])
        self.assertEqual(await manager._get_profile("user"), new)
        prompt = manager._client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn(json.dumps(old, ensure_ascii=False), prompt)

    async def test_profile_read_selects_latest_legacy_record_for_this_user(self):
        manager = make_manager()
        for doc_id, user, ts, pref in (
            ("old", "user", "2026-01-01", "旧偏好"),
            ("new", "user", "2026-10-30", "新偏好"),
            ("someone", "someone", "2026-11-01", "别人"),
        ):
            manager._profile.rows[doc_id] = (
                json.dumps({"preferences": [pref], "entities": {}}),
                {"user_id": user, "ts": ts},
            )
        latest = await manager._get_profile("user")
        self.assertEqual(latest["preferences"], ["新偏好"])
        await self.fill(manager, 1)
        manager._client.messages.create.return_value = response(json.dumps(latest))
        await manager.update_profile("user", "chat")
        prompt = manager._client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("新偏好", prompt)
        self.assertIn("user_profile:user", manager._profile.rows)

    async def test_profile_failure_or_invalid_output_does_not_destroy_old_profile(self):
        for invalid in (RuntimeError("model unavailable"), response("not json"), response('{"preferences": 1}')):
            with self.subTest(invalid=invalid):
                manager = make_manager()
                old = {"preferences": ["简洁"], "entities": {}}
                manager._profile.rows["user_profile:user"] = (json.dumps(old), {"user_id": "user"})
                await self.fill(manager, 1)
                manager._client.messages.create.side_effect = [invalid]
                await manager.update_profile("user", "chat")
                self.assertEqual(await manager._get_profile("user"), old)

    async def test_profile_storage_failure_does_not_delete_old_profile(self):
        manager = make_manager()
        old = {"preferences": ["简洁"], "entities": {}}
        manager._profile.rows["user_profile:user"] = (json.dumps(old), {"user_id": "user"})
        manager._profile.fail_write = True
        manager._client.messages.create.return_value = response('{"preferences": ["中文"], "entities": {}}')
        await self.fill(manager, 1)
        await manager.update_profile("user", "chat")
        self.assertEqual(await manager._get_profile("user"), old)

    async def test_concurrent_profile_updates_use_the_preceding_saved_profile(self):
        manager = make_manager()
        started, release = asyncio.Event(), asyncio.Event()
        first = {"preferences": ["简洁"], "entities": {}}
        second = {"preferences": ["简洁", "中文"], "entities": {}}
        prompts = []

        async def model(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            if len(prompts) == 1:
                started.set()
                await release.wait()
                return response(json.dumps(first))
            return response(json.dumps(second))

        manager._client.messages.create.side_effect = model
        await self.fill(manager, 1)
        await self.fill(manager, 1, conv="other")
        updating = asyncio.create_task(manager.update_profile("user", "chat"))
        await asyncio.wait_for(started.wait(), 1)
        another = asyncio.create_task(manager.update_profile("user", "other"))
        await asyncio.sleep(0)
        self.assertFalse(another.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(updating, another), 1)
        self.assertIn(json.dumps(first, ensure_ascii=False), prompts[1])
        self.assertEqual(await manager._get_profile("user"), second)

    async def test_context_combines_recent_messages_history_profile_and_summary(self):
        manager = make_manager(cross_conv=False)
        await self.fill(manager, 1)
        manager._redis.values["summary:user:chat"] = "旧摘要"
        manager._episodic.query_results = [["相关历史"]]
        profile = {"preferences": ["简洁"], "entities": {}}
        manager._profile.rows["user_profile:user"] = (json.dumps(profile), {"user_id": "user"})
        context = await manager.get_context("user", "chat", "新问题")
        self.assertEqual([m.content for m in context.recent_messages], ["m0"])
        self.assertEqual(context.relevant_history, ["相关历史"])
        self.assertEqual(context.user_profile, profile)
        self.assertEqual(context.summary, "旧摘要")


if __name__ == "__main__":
    unittest.main()
