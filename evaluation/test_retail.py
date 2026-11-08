"""验证官方工具实际写入、严格重放及判分；不调用模型，不充当评测题库。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from pydantic import TypeAdapter
from tau2.data_model.message import Message, ToolMessage
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from evaluation.retail import catalog
from evaluation.run import report
from tooling.mcp_client import ToolMCPClient


class RetailTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_mcp_state_matches_strict_replay_and_official_goal(self):
        task = next(t for t in get_tasks('train') if t.id == '0')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            client = ToolMCPClient(log_dir=directory, catalog=catalog(get_environment()),
                server_args=['-m', 'evaluation.retail'],
                server_env={'ECHOMIND_EVAL_DIR': temp, 'TAU2_DATA_DIR': os.environ['TAU2_DATA_DIR']})
            try:
                await client.start()
                # Execute official reference actions only to verify plumbing and scoring.
                for i, action in enumerate(task.evaluation_criteria.actions):
                    response = await client.call(action.name, action.arguments, dict(
                        user_id='test', conv_id='test', request_id='test',
                        agent_type='general', tool_use_id=f'test-{i}'))
                    self.assertTrue(response['success'], response['error'])
                self.assertNotEqual(client.server_pid, os.getpid())
            finally:
                await client.close()
            messages = [TypeAdapter(Message).validate_json(line)
                        for line in (directory/'tools.jsonl').read_text().splitlines()]
            replay = get_environment()
            replay.set_state(initialization_data=None, initialization_actions=None,
                             message_history=messages, strict=True)
            actual = json.loads((directory/'after.json').read_text())
            self.assertEqual(replay.tools.db.model_dump(mode='json'), actual)
            self.assertNotEqual(json.loads((directory/'before.json').read_text()), actual)
            reward = EnvironmentEvaluator.calculate_reward(get_environment, task, messages, strict_replay=True)
            self.assertEqual(reward.reward, 1)
            wrong = EnvironmentEvaluator.calculate_reward(get_environment, task, [], strict_replay=True)
            self.assertEqual(wrong.reward, 0)
            tampered = [m.model_copy(deep=True) for m in messages]
            self.assertIsInstance(tampered[-1], ToolMessage)
            tampered[-1].content = 'fabricated success'
            with self.assertRaises(ValueError):
                EnvironmentEvaluator.calculate_reward(get_environment, task, tampered, strict_replay=True)


class ReportTests(unittest.TestCase):
    def test_incomplete_run_does_not_publish_official_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = [dict(task_id=str(i), trial=0, slot=str(i), status=status)
                    for i, status in enumerate(('completed', 'error', 'not_run'))]
            rows[0]['reward'] = 1
            report(directory, rows)
            summary = json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['success_planned'], 1/3)
            self.assertEqual(summary['errors'], 1)
            self.assertEqual(summary['not_run'], 1)
            self.assertIsNone(summary['pass_hat_k'])

    def test_pass_hat_k_measures_repeatability_not_best_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = [dict(task_id='0', trial=i, slot=str(i), status='completed', reward=r)
                    for i, r in enumerate((1, 1, 0))]
            report(directory, rows)
            summary = json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['pass_hat_k'], {'1': 2/3, '2': 1/3, '3': 0})


if __name__ == '__main__':
    unittest.main()
