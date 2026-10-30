"""真实 SQLite checkpointer；替换外部模型和记忆服务，验证恢复边界。"""
import asyncio
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.agent_orchestrator import AgentType, OrchestratorResult
from agents.chat_workflow import ChatWorkflow, WorkflowConflict, WorkflowExecutionError
from core.intent_recognizer import IntentCategory, UrgencyLevel
from memory.conversation_memory import MemoryContext
from memory.problem_record import ProblemRecordManager
from test_problem_record import delta, response


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "checkpoints.sqlite3"
        self.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response(delta()))))
        self.records = ProblemRecordManager(str(Path(self.temp.name) / "problems.sqlite3"), self.client, "test")
        self.exchanges = {}
        self.memory = SimpleNamespace(
            problem_records=self.records,
            get_context=AsyncMock(return_value=MemoryContext([], [], {}, "私有上下文")),
            save_exchange_once=AsyncMock(side_effect=self.save_exchange), update_profile=AsyncMock(),
        )
        intent = SimpleNamespace(intent=IntentCategory.TECHNICAL_LOGIN, intent_group="technical",
                                 urgency=UrgencyLevel.LOW, confidence=0.9, entities={}, source_scores={"llm": 0.9})
        self.orchestrator = SimpleNamespace(
            recognize_intent=AsyncMock(return_value=intent),
            run=AsyncMock(return_value=OrchestratorResult(
                request_id="test", response="请重启应用。", agent_type=AgentType.TECHNICAL,
                intent=IntentCategory.TECHNICAL_LOGIN, agent_types=[AgentType.TECHNICAL],
            )),
        )
        self.knowledge = AsyncMock(return_value=("知识", True))
        self.instances = []
        self.workflow = self.new_workflow()

    def new_workflow(self):
        workflow = ChatWorkflow(self.memory, self.orchestrator, self.knowledge, path=self.path)
        self.instances.append(workflow)
        return workflow

    async def asyncTearDown(self):
        for workflow in self.instances:
            await workflow.close()
        self.temp.cleanup()

    async def save_exchange(self, user, conv, message, reply, request_id):
        key = (user, conv, request_id)
        if key in self.exchanges:
            return False
        self.exchanges[key] = (message, reply)
        return True

    async def turn(self, request_id="r1", message="我无法登录", user="user", conv="chat"):
        return await self.workflow.handle_turn(message, user, conv, request_id)

    async def test_completed_state_survives_new_instance_and_cached_retry(self):
        result = await self.turn()
        self.assertEqual(result["workflow"]["node_path"], list(ChatWorkflow.NODES))
        self.assertTrue(result["knowledge_used"])
        self.assertEqual(result["problem_record"]["last_reply"], result["response"])
        await self.workflow.close()
        self.workflow = self.new_workflow()
        status = await self.workflow.status("user", "chat")
        self.assertEqual(status["phase"], "complete")
        self.assertEqual(status["checkpoint_id"], result["workflow"]["checkpoint_id"])
        again = await self.turn()
        self.assertTrue(again["workflow"]["cached"])
        self.assertEqual(again["response"], result["response"])
        self.orchestrator.run.assert_awaited_once()
        self.assertEqual(len(self.exchanges), 1)
        self.assertNotIn("私有上下文", str(status))

    async def test_resume_after_restart_skips_completed_model_calls(self):
        self.knowledge.side_effect = RuntimeError("检索中断")
        with self.assertRaises(WorkflowExecutionError):
            await self.turn()
        status = await self.workflow.status("user", "chat")
        self.assertEqual(status["next"], ["retrieve_knowledge"])
        self.assertTrue(status["resumable"])
        await self.workflow.close()
        self.workflow = self.new_workflow()
        self.knowledge.side_effect = None
        result = await self.workflow.resume("user", "chat", "r1")
        self.assertTrue(result["workflow"]["resumed"])
        self.memory.get_context.assert_awaited_once()
        self.client.messages.create.assert_awaited_once()
        self.orchestrator.recognize_intent.assert_awaited_once()
        self.orchestrator.run.assert_awaited_once()
        self.assertEqual(result["workflow"]["node_path"], list(ChatWorkflow.NODES))

    async def test_write_succeeded_before_node_failed_does_not_duplicate_messages(self):
        async def fail_after_write(*args):
            await self.save_exchange(*args)
            raise RuntimeError("写入后断开")
        self.memory.save_exchange_once.side_effect = fail_after_write
        with self.assertRaises(WorkflowExecutionError):
            await self.turn()
        self.assertEqual((await self.workflow.status("user", "chat"))["next"], ["save_exchange"])
        self.memory.save_exchange_once.side_effect = self.save_exchange
        await self.workflow.close()
        self.workflow = self.new_workflow()
        result = await self.turn()
        self.assertTrue(result["workflow"]["resumed"])
        self.assertEqual(len(self.exchanges), 1)
        self.orchestrator.run.assert_awaited_once()

    async def test_problem_write_before_checkpoint_does_not_increment_twice(self):
        original = self.records.prepare
        async def fail_after_prepare(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("问题记录已写入，节点未返回")
        with patch.object(self.records, "prepare", side_effect=fail_after_prepare):
            with self.assertRaises(WorkflowExecutionError):
                await self.turn()
        before = self.records.store.active("user", "chat")
        result = await self.turn()
        self.assertEqual(result["problem_record"]["turn_count"], before["turn_count"])
        self.client.messages.create.assert_awaited_once()

    async def test_older_completed_request_is_cached_after_a_new_turn(self):
        first = await self.turn()
        await self.turn("r2", "另外一个问题")
        again = await self.turn()
        self.assertTrue(again["workflow"]["cached"])
        self.assertEqual(again["workflow"]["checkpoint_id"], first["workflow"]["checkpoint_id"])
        self.assertEqual(self.orchestrator.run.await_count, 2)
        self.assertEqual((await self.workflow.status("user", "chat"))["request_id"], "r2")

    async def test_same_request_id_cannot_be_reused_for_other_text(self):
        await self.turn()
        with self.assertRaises(WorkflowConflict):
            await self.turn(message="另一条内容")

    async def test_pending_request_blocks_new_message_but_not_other_users_or_chats(self):
        self.orchestrator.run.side_effect = RuntimeError("中断")
        with self.assertRaises(WorkflowExecutionError):
            await self.turn()
        with self.assertRaises(WorkflowConflict):
            await self.turn("r2", "新内容")
        with self.assertRaises(WorkflowConflict):
            await self.workflow.resume("user", "chat", "wrong")
        with self.assertRaises(WorkflowConflict):
            await self.workflow.resume("another", "chat")
        self.orchestrator.run.side_effect = None
        await self.turn(user="another")
        await self.turn(conv="another")
        self.assertTrue((await self.workflow.status("user", "chat"))["resumable"])
        await self.turn()
        self.assertEqual(len(self.exchanges), 3)

    async def test_cancelled_node_can_be_resumed_from_disk(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def hanging_agent(request):
            entered.set()
            await release.wait()
        self.orchestrator.run.side_effect = hanging_agent
        task = asyncio.create_task(self.turn())
        await asyncio.wait_for(entered.wait(), 5)
        self.assertEqual((await self.workflow.status("user", "chat"))["phase"], "running")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.workflow.close()
        self.workflow = self.new_workflow()
        self.assertEqual((await self.workflow.status("user", "chat"))["next"], ["run_agent"])
        self.orchestrator.run.side_effect = None
        await self.workflow.resume("user", "chat")
        self.orchestrator.recognize_intent.assert_awaited_once()

    async def test_abrupt_process_exit_preserves_the_last_completed_checkpoint(self):
        # 子进程在 Agent 节点直接退出，既不关闭连接，也不运行 finally。
        script = '''
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, "tests")
from test_chat_workflow import WorkflowTests
from memory.problem_record import ProblemRecordManager
async def main():
    case = WorkflowTests()
    case.setUp()
    case.path = Path(sys.argv[1])
    case.memory.problem_records = ProblemRecordManager(sys.argv[2], case.client, "test")
    workflow = case.new_workflow()
    async def crash(request):
        os._exit(73)
    case.orchestrator.run.side_effect = crash
    await workflow.handle_turn("我无法登录", "user", "chat", "r1")
asyncio.run(main())
'''
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, str(self.path), str(self.records.store.path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        output, error = await asyncio.wait_for(process.communicate(), 15)
        self.assertEqual(process.returncode, 73, error.decode())
        self.assertEqual((await self.workflow.status("user", "chat"))["next"], ["run_agent"])
        result = await self.workflow.resume("user", "chat", "r1")
        self.assertTrue(result["workflow"]["resumed"])
        self.memory.get_context.assert_not_awaited()
        self.client.messages.create.assert_not_awaited()
        self.orchestrator.recognize_intent.assert_not_awaited()
        self.knowledge.assert_not_awaited()
        self.orchestrator.run.assert_awaited_once()

    async def test_same_conversation_is_serial_and_other_conversation_can_run(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.orchestrator.run.return_value
        async def blocking_agent(request):
            if request.request_id == "r1":
                entered.set()
                await release.wait()
            return original
        self.orchestrator.run.side_effect = blocking_agent
        first = asyncio.create_task(self.turn())
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(self.turn("r2", "第二条"))
        other = await asyncio.wait_for(self.turn("r3", conv="other"), 5)
        self.assertEqual(other["workflow"]["phase"], "complete")
        self.assertFalse(second.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        self.assertEqual((await self.workflow.status("user", "chat"))["request_id"], "r2")

    async def test_http_retry_without_conv_id_recovers_same_first_conversation(self):
        import httpx
        import api.main as api
        payload = {"user_id": "user", "message": "我无法登录", "client_request_id": "r1"}
        with patch.object(api, "_workflow", self.workflow):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                self.knowledge.side_effect = RuntimeError("中断")
                failed = await client.post("/chat", json=payload)
                self.assertEqual(failed.status_code, 503)
                self.knowledge.side_effect = None
                recovered = await client.post("/chat", json=payload)
                self.assertEqual(recovered.status_code, 200, recovered.text)
                data = recovered.json()
                self.assertTrue(data["workflow"]["resumed"])
                cached = (await client.post("/chat", json=payload)).json()
                self.assertEqual(cached["conv_id"], data["conv_id"])
                self.assertTrue(cached["workflow"]["cached"])
                status = await client.get("/workflow/status", params={"user_id": "user", "conv_id": data["conv_id"]})
                self.assertEqual(status.json()["phase"], "complete")
                resumed = await client.post("/workflow/resume", json={"user_id": "user", "conv_id": data["conv_id"]})
                self.assertEqual(resumed.status_code, 200)
                invalid = await client.post("/workflow/resume", json={"user_id": "other", "conv_id": data["conv_id"]})
                self.assertEqual(invalid.status_code, 409)
        self.orchestrator.recognize_intent.assert_awaited_once()

