import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory.problem_record import ProblemRecordManager, ProblemRecordStore, render_problem_record


def delta(**changes):
    data = {
        "relation": "new", "title": "登录失败", "domain": "technical",
        "status": "in_progress", "status_evidence": "",
        "facts": [{"text": "无法登录", "evidence": "无法登录"}],
        "taken_steps": [], "feedback": [], "previous_suggestions": [],
        "unresolved": ["登录问题未解决"], "next_step": "等待用户反馈",
    }
    data.update(changes)
    return data


def response(data, stop_reason="end_turn"):
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason)


class ProblemRecordFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "problem_records.sqlite3")
        self.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response(delta()))))
        self.manager = ProblemRecordManager(self.path, self.client, "test-model")

    def tearDown(self):
        self.temp.cleanup()

    async def begin(self):
        result = await self.manager.prepare("user", "chat", "我无法登录")
        self.assertFalse(result.error)
        return result.record


class ProblemRecordTests(ProblemRecordFixture):
    async def test_record_survives_restart_and_is_isolated_by_user_and_conversation(self):
        record = await self.begin()
        restarted = ProblemRecordStore(self.path)
        self.assertEqual(restarted.active("user", "chat"), record)
        self.assertEqual(restarted.active("other", "chat"), {})
        self.assertEqual(restarted.active("user", "other"), {})
        self.assertEqual(restarted.list("other", "chat"), [])

    async def test_assistant_suggestion_is_not_a_user_attempt(self):
        record = await self.begin()
        saved = await self.manager.finish("user", "chat", record, "请尝试重启应用。", "technical")
        self.assertEqual(saved.record["taken_steps"], [])
        self.assertEqual(saved.record["last_reply"], "请尝试重启应用。")
        self.assertEqual(saved.record["status"], "awaiting_user")

    async def test_feedback_records_tried_method_and_passes_it_to_the_next_agent(self):
        record = await self.begin()
        await self.manager.finish("user", "chat", record, "请尝试重启应用。", "technical")
        self.client.messages.create.return_value = response(delta(
            relation="continue", facts=[],
            taken_steps=[{"action": "重启应用", "result": "failed", "evidence": "重启应用后还是不行"}],
            feedback=[{"text": "重启无效", "evidence": "还是不行"}],
            previous_suggestions=[{"text": "重启应用", "evidence": "请尝试重启应用。"}],
        ))
        continued = await self.manager.prepare("user", "chat", "重启应用后还是不行")
        self.assertEqual(continued.record["problem_id"], record["problem_id"])
        self.assertEqual(continued.record["taken_steps"][0]["result"], "failed")
        self.assertEqual(continued.record["taken_steps"][0]["source"], "user_message")
        self.assertEqual(continued.record["suggestions"][0]["source"], "assistant_reply")
        prompt = render_problem_record(continued.record)
        self.assertIn("重启应用", prompt)
        self.assertIn("failed", prompt)
        saved = await self.manager.finish("user", "chat", continued.record, "继续核对账户状态。", "general")
        self.assertEqual(saved.record["agent_history"], ["technical", "general"])

    async def test_new_topic_keeps_old_record_without_mixing_steps(self):
        old = await self.begin()
        await self.manager.finish("user", "chat", old, "请尝试重启应用。", "technical")
        self.client.messages.create.return_value = response(delta(
            title="退款咨询", domain="billing", facts=[{"text": "用户申请退款", "evidence": "我要退款"}],
            previous_suggestions=[{"text": "重启应用", "evidence": "请尝试重启应用。"}],
            unresolved=["退款申请未处理"],
        ))
        new = await self.manager.prepare("user", "chat", "换个问题，我要退款")
        self.assertNotEqual(new.record["problem_id"], old["problem_id"])
        self.assertEqual(new.record["suggestions"], [])
        self.assertEqual(new.record["taken_steps"], [])
        records = self.manager.store.list("user", "chat")
        self.assertEqual(len(records), 2)
        self.assertEqual(sum(record["active"] for record in records), 1)
        self.assertEqual(self.manager.store.active("user", "chat")["title"], "退款咨询")

    async def test_only_entries_with_quotes_from_the_actual_user_are_accepted(self):
        self.client.messages.create.return_value = response(delta(
            facts=[{"text": "退款已到账", "evidence": "退款已到账"}],
            taken_steps=[{"action": "重启", "result": "succeeded", "evidence": "请尝试重启"}],
        ))
        result = await self.manager.prepare("user", "chat", "无法登录")
        self.assertEqual(result.record["facts"], [])
        self.assertEqual(result.record["taken_steps"], [])

    async def test_recent_context_can_explain_a_followup_but_is_not_user_execution_evidence(self):
        self.client.messages.create.return_value = response(delta(
            relation="continue",
            facts=[{"text": "无法登录", "evidence": "无法登录"}],
            taken_steps=[{"action": "重启", "result": "failed", "evidence": "建议重启"}],
        ))
        result = await self.manager.prepare("user", "chat", "还是不行", [
            {"role": "user", "content": "我无法登录"},
            {"role": "assistant", "content": "建议重启"},
        ])
        prompt = self.client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertFalse(result.error)
        self.assertIn("我无法登录", prompt)
        self.assertEqual(result.record["facts"], [])
        self.assertEqual(result.record["taken_steps"], [])

    async def test_user_confirmation_can_resolve_but_missing_evidence_cannot(self):
        record = await self.begin()
        self.client.messages.create.return_value = response(delta(
            relation="continue", status="resolved", status_evidence="已经好了", facts=[],
        ))
        result = await self.manager.prepare("user", "chat", "已经好了")
        self.assertEqual(result.record["status"], "resolved")
        self.assertEqual(result.record["unresolved"], [])
        self.client.messages.create.return_value = response(delta(
            relation="continue", status="resolved", status_evidence="已经好了", facts=[],
        ))
        result = await self.manager.prepare("user", "chat", "还是无法登录")
        self.assertEqual(result.record["status"], "in_progress")
        self.assertEqual(result.record["problem_id"], record["problem_id"])

    async def test_negative_feedback_cannot_mark_the_problem_resolved(self):
        await self.begin()
        self.client.messages.create.return_value = response(delta(
            relation="continue", status="resolved", status_evidence="没有解决", facts=[],
        ))
        result = await self.manager.prepare("user", "chat", "没有解决")
        self.assertEqual(result.record["status"], "in_progress")

    async def test_handoff_request_does_not_claim_actual_handoff(self):
        await self.begin()
        self.client.messages.create.return_value = response(delta(
            relation="continue", status="handoff_requested", status_evidence="我要人工", facts=[],
        ))
        result = await self.manager.prepare("user", "chat", "我要人工")
        saved = await self.manager.finish("user", "chat", result.record, "建议联系人工客服。", "general")
        self.assertEqual(saved.record["status"], "handoff_requested")
        self.assertNotIn("handoff_completed", saved.record)

    async def test_supporting_agents_are_included_in_the_record(self):
        record = await self.begin()
        result = await self.manager.finish(
            "user", "chat", record, "联合处理建议。", "technical", ["technical", "billing"],
        )
        self.assertEqual(result.record["agent_history"], ["technical", "billing"])

    async def test_greeting_does_not_create_or_overwrite_a_problem(self):
        before = await self.begin()
        self.client.messages.create.return_value = response({"relation": "none"})
        result = await self.manager.prepare("user", "chat", "你好")
        self.assertEqual(result.record, {})
        self.assertEqual(self.manager.store.active("user", "chat"), before)

    async def test_parse_model_or_storage_failures_keep_the_existing_record(self):
        before = await self.begin()
        cases = [RuntimeError("model unavailable"), response(""), response("{broken"),
                 response(delta(relation="continue"), "max_tokens"), response({"relation": "continue"})]
        for value in cases:
            with self.subTest(value=value):
                self.client.messages.create.side_effect = [value]
                result = await self.manager.prepare("user", "chat", "还是不行")
                self.assertTrue(result.error)
                self.assertEqual(result.record, {})
                self.assertEqual(self.manager.store.active("user", "chat"), before)
        self.client.messages.create.side_effect = None
        self.client.messages.create.return_value = response(delta(relation="continue"))
        with patch.object(self.manager.store, "save", side_effect=RuntimeError("storage unavailable")):
            result = await self.manager.prepare("user", "chat", "还是不行")
        self.assertTrue(result.error)
        self.assertEqual(self.manager.store.active("user", "chat"), before)

    async def test_finish_storage_failure_preserves_prepared_record(self):
        before = await self.begin()
        with patch.object(self.manager.store, "save", side_effect=RuntimeError("disk unavailable")):
            result = await self.manager.finish("user", "chat", before, "请重启。", "technical")
        self.assertTrue(result.error)
        self.assertEqual(self.manager.store.active("user", "chat"), before)

    async def test_repeated_action_updates_its_result_instead_of_duplicating(self):
        await self.begin()
        for outcome in ("unknown", "failed"):
            self.client.messages.create.return_value = response(delta(
                relation="continue", facts=[],
                taken_steps=[{"action": "重启", "result": outcome, "evidence": "重启"}],
            ))
            result = await self.manager.prepare("user", "chat", "重启仍然失败")
        self.assertEqual(len(result.record["taken_steps"]), 1)
        self.assertEqual(result.record["taken_steps"][0]["result"], "failed")

    async def test_record_lists_and_prompt_are_bounded(self):
        await self.begin()
        for index in range(16):
            text = f"尝试方法{index}"
            self.client.messages.create.return_value = response(delta(
                relation="continue", facts=[],
                taken_steps=[{"action": text + "很长" * 500, "result": "failed", "evidence": text}],
            ))
            result = await self.manager.prepare("user", "chat", text)
        self.assertEqual(len(result.record["taken_steps"]), 12)
        self.assertLess(len(render_problem_record(result.record)), 6500)


class ProblemRecordApiTests(ProblemRecordFixture):
    async def test_chat_injects_record_and_returns_the_saved_agent_reply(self):
        import api.main as api
        from agents.agent_orchestrator import AgentType, OrchestratorResult
        from core.intent_recognizer import IntentCategory, UrgencyLevel
        from memory.conversation_memory import MemoryContext, MemoryManager

        memory = MemoryManager.__new__(MemoryManager)
        memory._turn_locks = {}
        memory.problem_records = self.manager
        memory.get_context = AsyncMock(return_value=MemoryContext([], [], {}, ""))
        memory.add_message = AsyncMock()
        memory.update_profile = AsyncMock()
        intent = SimpleNamespace(intent=IntentCategory.TECHNICAL_LOGIN, intent_group="technical",
                                 urgency=UrgencyLevel.LOW, confidence=0.9, entities={}, source_scores={})
        orchestrator = SimpleNamespace(
            recognize_intent=AsyncMock(return_value=intent),
            run=AsyncMock(return_value=OrchestratorResult(
                request_id="test", response="请尝试重启应用。", agent_type=AgentType.TECHNICAL,
                intent=IntentCategory.TECHNICAL_LOGIN,
            )),
        )
        with patch.object(api, "_memory", memory), patch.object(api, "_orchestrator", orchestrator), \
             patch.object(api, "_build_knowledge_context", AsyncMock(return_value=("", False))):
            result = await api.chat(api.ChatRequest(user_id="user", conv_id="chat", message="我无法登录"))
            viewed = await api.problem_records(user_id="user", conv_id="chat")
        self.assertIn("[问题处理记录]", orchestrator.run.call_args.args[0].context)
        history = orchestrator.recognize_intent.call_args.kwargs["history"]
        self.assertIn("登录失败", history[0]["content"])
        self.assertEqual(result.problem_record["last_agent"], "technical")
        self.assertEqual(result.problem_record["last_reply"], "请尝试重启应用。")
        self.assertEqual(viewed["records"][0]["problem_id"], result.problem_record["problem_id"])
        self.assertFalse(result.problem_record_error)
        await asyncio.sleep(0)  # 让画像替身后台任务结束。


if __name__ == "__main__":
    unittest.main()
