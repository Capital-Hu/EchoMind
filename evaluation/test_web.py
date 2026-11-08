"""任务控制测试；业务题目仍然来自官方数据，不调用模型。"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from evaluation import run as runner, web


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        for target, value in [('evaluation.run.ROOT', self.root),
                              ('evaluation.web.RUNS', self.root/'data/eval/tau'),
                              ('evaluation.web.process', None), ('evaluation.web.active_directory', None)]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = {**web.RunInput(user_model='deepseek/test', judge_model='deepseek/test',
                                      task_id=['0']).model_dump(), 'plan_only': False}

    def test_official_splits_and_invalid_task_never_launch_a_run(self):
        with TestClient(web.app) as client:
            for split, size in [('train', 74), ('test', 40), ('base', 114)]:
                response = client.get('/tasks', params={'split': split})
                self.assertEqual(len(response.json()), size)
            for ids in ([], ['not-a-task']):
                response = client.post('/runs', json={**self.config, 'task_id': ids})
                self.assertEqual(response.status_code, 422)
            self.assertEqual(client.get('/runs').json(), [])

    def test_page_defaults_use_the_running_system_model(self):
        with patch.dict(os.environ, {'ANTHROPIC_MODEL': 'deepseek-flash'}), TestClient(web.app) as client:
            config = client.get('/config').json()
            self.assertEqual(config['system_model'], 'deepseek-flash')
            self.assertEqual(config['user_model'], 'deepseek/deepseek-flash')
            self.assertEqual(config['judge_model'], 'deepseek/deepseek-flash')

    def test_restart_marks_unfinished_run_interrupted_without_retry(self):
        directory = runner.prepare(self.config)
        (directory/'runner.log').touch()
        with TestClient(web.app) as client:
            run = client.get('/runs/'+directory.name).json()
            self.assertEqual(run['status'], 'interrupted')
            self.assertEqual(run['summary']['not_run'], 1)
            self.assertIsNone(run['summary']['pass_hat_k'])
            self.assertEqual(client.post('/runs/'+directory.name+'/cancel').status_code, 409)
            self.assertEqual(client.get('/runs/'+directory.name+'/tasks/not-a-slot').status_code, 404)
            self.assertEqual(client.get('/runs/'+directory.name+'/files/.env').status_code, 422)

    def test_restart_does_not_leave_an_active_task_running(self):
        directory = runner.prepare(self.config)
        (directory/'runner.log').touch()
        (directory/'0--0').mkdir()
        rows = web.read(directory/'results.json')
        rows[0]['status'] = 'running'
        runner.report(directory, rows)
        runner.write(directory/'state.json', dict(status='running', active_slot='0--0'))
        with TestClient(web.app) as client:
            result = client.get('/runs/'+directory.name).json()
            self.assertEqual(result['status'], 'interrupted')
            self.assertEqual(result['summary']['running'], 0)
            self.assertEqual(result['summary']['errors'], 1)
            self.assertFalse(result['can_cancel'])

    def test_plan_only_is_not_an_active_run(self):
        directory = runner.prepare({**self.config, 'plan_only': True})
        with TestClient(web.app) as client:
            self.assertEqual(client.get('/runs/'+directory.name).json()['status'], 'plan_only')

    def test_cancel_before_first_task_keeps_all_slots_not_run(self):
        directory = runner.prepare({**self.config, 'task_id': ['0', '2']})
        (directory/'cancel').touch()
        self.assertEqual(runner.run(directory), 130)
        summary = web.read(directory/'summary.json')
        self.assertEqual(summary['not_run'], 2)
        self.assertEqual(summary['cancelled'], 0)
        self.assertEqual(web.read(directory/'state.json')['status'], 'cancelled')

    def test_worker_process_crash_is_not_a_zero_or_success_reward(self):
        directory = runner.prepare(self.config)
        popen = subprocess.Popen
        with patch('evaluation.run.subprocess.Popen', side_effect=lambda args, **kw:
                   popen([sys.executable, '-c', 'raise SystemExit(7)'], **kw)):
            self.assertEqual(runner.run(directory), 1)
        result = web.read(directory/'results.json')[0]
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('reward', result)
        self.assertEqual(web.read(directory/'summary.json')['errors'], 1)
        self.assertIsNone(web.read(directory/'summary.json')['pass_hat_k'])

    def test_timeout_reaps_worker_group_and_keeps_future_tasks_not_run(self):
        directory = runner.prepare({**self.config, 'task_id': ['0', '2'], 'timeout': 1})
        popen, children = subprocess.Popen, []
        def worker(args, **kwargs):
            child = popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            children.append(child)
            return child
        with patch('evaluation.run.subprocess.Popen', side_effect=worker):
            self.assertEqual(runner.run(directory), 1)
        with self.assertRaises(ProcessLookupError):
            os.killpg(children[0].pid, 0)
        summary = web.read(directory/'summary.json')
        self.assertEqual((summary['errors'], summary['not_run']), (1, 1))
        self.assertEqual(web.read(directory/'state.json')['status'], 'error')


if __name__ == '__main__':
    unittest.main()
