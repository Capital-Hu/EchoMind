"""页面使用的 τ 评测服务；与 CLI 共用 prepare/run，不实现另一套评分。"""
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import uvicorn
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tau2.domains.retail.environment import get_tasks

from evaluation.run import ROOT, REVISION, prepare, report, write

RUNS = ROOT / 'data/eval/tau'
process = None
active_directory = None


def read(path):
    return json.loads(path.read_text())


def directory_for(run_id):
    if not re.fullmatch(r'\d{8}T\d{6}Z-[a-f0-9]{8}', run_id):
        raise HTTPException(404, '运行记录不存在')
    directory = RUNS / run_id
    if not (directory/'state.json').is_file():
        raise HTTPException(404, '运行记录不存在')
    return directory


def refresh_process():
    if process is not None and process.poll() is not None:
        state = read(active_directory/'state.json')
        if state['status'] in ('planned', 'running'):
            interrupted(active_directory, 'error', f'评测进程异常退出（{process.returncode}），请查看运行日志')


def interrupted(directory, status, reason):
    rows = read(directory/'results.json')
    for row in rows:
        if row['status'] == 'running':
            row.update(status='error', error=reason)
            write(directory/row['slot']/'result.json', row)
    report(directory, rows)
    write(directory/'state.json', dict(status=status, active_slot=None, error=reason))


def snapshot(directory):
    refresh_process()
    manifest = read(directory/'manifest.json')
    return dict(id=directory.name, **read(directory/'state.json'),
                can_cancel=active_directory == directory and process is not None and process.poll() is None,
                summary=read(directory/'summary.json'), config=manifest['config'], model=manifest['model'])


@asynccontextmanager
async def lifespan(app):
    RUNS.mkdir(parents=True, exist_ok=True)
    # 容器重启后的未完成任务如实标记中断，不自动重跑。
    for log in RUNS.glob('*/runner.log'):
        directory = log.parent
        if read(directory/'state.json')['status'] in ('planned', 'running'):
            interrupted(directory, 'interrupted', '评测服务已重启，本次运行中断')
    yield
    if process is not None and process.poll() is None:
        (active_directory/'cancel').touch()
        if read(active_directory/'state.json')['status'] == 'running':
            process.send_signal(signal.SIGINT)
        await asyncio.to_thread(process.wait, timeout=30)


app = FastAPI(title='τ retail evaluation', lifespan=lifespan)


class RunInput(BaseModel):
    split: Literal['train', 'test', 'base'] = 'train'
    task_id: list[str] | None = None
    trials: int = Field(default=1, ge=1, le=10)
    user_model: str = Field(min_length=1, max_length=120, pattern=r'^\S+$')
    judge_model: str = Field(min_length=1, max_length=120, pattern=r'^\S+$')
    max_turns: int = Field(default=30, ge=1, le=100)
    max_tokens: int = Field(default=8192, ge=1, le=32768)
    call_limit: int = Field(default=200, ge=1, le=1000)
    timeout: int = Field(default=1200, ge=1, le=7200)
    seed: int = Field(default=300, ge=0)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/config')
async def config():
    model = os.environ['ANTHROPIC_MODEL']
    return dict(revision=REVISION, system_model=model,
                user_model='deepseek/' + model, judge_model='deepseek/' + model)


@app.get('/tasks')
async def tasks(split: Literal['train', 'test', 'base'] = 'train'):
    return [dict(id=t.id, scenario=str(t.user_scenario)) for t in get_tasks(split)]


@app.get('/runs')
async def list_runs():
    return [snapshot(path.parent) for path in sorted(RUNS.glob('*/state.json'), reverse=True)[:50]]


@app.post('/runs', status_code=202)
async def start_run(body: RunInput):
    global process, active_directory
    refresh_process()
    if process is not None and process.poll() is None:
        raise HTTPException(409, f'已有评测正在运行：{active_directory.name}')
    valid_ids = {task.id for task in get_tasks(body.split)}
    if body.task_id is not None and (not body.task_id or not set(body.task_id) <= valid_ids):
        raise HTTPException(422, '请选择当前题集中的有效题目；运行全部题目时 task_id 传 null')
    directory = prepare({**body.model_dump(), 'plan_only': False})
    with (directory/'runner.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'evaluation.run', '--run-directory', str(directory)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    active_directory = directory
    return snapshot(directory)


@app.get('/runs/{run_id}')
async def get_run(run_id: str):
    directory = directory_for(run_id)
    return {**snapshot(directory), 'rows': read(directory/'results.json')}


@app.post('/runs/{run_id}/cancel', status_code=202)
async def cancel_run(run_id: str):
    directory = directory_for(run_id)
    refresh_process()
    if active_directory != directory or process is None or process.poll() is not None:
        raise HTTPException(409, '这次评测没有正在运行的进程')
    (directory/'cancel').touch()
    if read(directory/'state.json')['status'] == 'running':
        process.send_signal(signal.SIGINT)
    return {'id': run_id, 'message': '正在停止，已完成的结果会保留'}


def task_directory(run_id, slot):
    directory = directory_for(run_id)
    if slot not in {row['slot'] for row in read(directory/'results.json')}:
        raise HTTPException(404, '题目记录不存在')
    return directory/slot


@app.get('/runs/{run_id}/tasks/{slot}')
async def task_detail(run_id: str, slot: str):
    directory = task_directory(run_id, slot)
    return dict(result=read(directory/'result.json') if (directory/'result.json').exists() else None,
                trajectory=read(directory/'trajectory.json') if (directory/'trajectory.json').exists() else [],
                turns=len(list(directory.glob('turn-*.json'))),
                artifacts=[name for name in ('result.json', 'trajectory.json', 'simulation.json', 'tools.jsonl',
                           'before.json', 'after.json', 'worker.log') if (directory/name).is_file()])


@app.get('/runs/{run_id}/tasks/{slot}/files/{name}')
async def task_file(run_id: str, slot: str,
                    name: Literal['result.json', 'trajectory.json', 'simulation.json', 'tools.jsonl',
                                  'before.json', 'after.json', 'worker.log']):
    path = task_directory(run_id, slot)/name
    if not path.is_file():
        raise HTTPException(404, '文件尚未生成')
    return FileResponse(path, filename=name)


@app.get('/runs/{run_id}/files/{name}')
async def run_file(run_id: str, name: Literal['manifest.json', 'report.md', 'results.json', 'summary.json', 'runner.log']):
    path = directory_for(run_id)/name
    if not path.is_file():
        raise HTTPException(404, '文件尚未生成')
    return FileResponse(path, filename=name)


if __name__ == '__main__':
    # 当前项目使用 DeepSeek 的 Anthropic 接口；τ 的 LiteLLM 使用同一个容器内凭证。
    # 在容器启动后取值，避免 Compose 插值被宿主机的模型或凭证覆盖。
    os.environ['DEEPSEEK_API_KEY'] = os.environ['ANTHROPIC_API_KEY']
    uvicorn.run(app, host='0.0.0.0', port=8002)
