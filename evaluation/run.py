"""Run EchoMind against unchanged τ retail tasks, user and evaluators."""
import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REVISION = 'fc0055dc4e0a316c3f83133267fbd6faaa770992'
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def report(directory, rows):
    write(directory / 'results.json', rows)
    successful = sum(row.get('reward') == 1 for row in rows)
    completed = sum(row['status'] == 'completed' for row in rows)
    summary = dict(planned=len(rows), successful=successful, completed=completed,
                   errors=sum(row['status'] == 'error' for row in rows),
                   not_run=sum(row['status'] == 'not_run' for row in rows),
                   running=sum(row['status'] == 'running' for row in rows),
                   cancelled=sum(row['status'] == 'cancelled' for row in rows),
                   success_planned=successful / len(rows))
    summary['pass_hat_k'] = None
    if completed == len(rows):
        from tau2.metrics.agent_metrics import pass_hat_k
        tasks = {}
        for row in rows:
            tasks.setdefault(row['task_id'], []).append(row['reward'] == 1)
        summary['pass_hat_k'] = {
            str(k): sum(pass_hat_k(len(trials), sum(trials), k) for trials in tasks.values()) / len(tasks)
            for k in range(1, min(map(len, tasks.values())) + 1)
        }
    write(directory / 'summary.json', summary)
    lines = ['# EchoMind · τ retail', '',
             f"成功 {successful}/{len(rows)}；有效判分 {completed}；错误 {summary['errors']}；"
             f"运行中 {summary['running']}；已停止 {summary['cancelled']}；未运行 {summary['not_run']}。", '',
             '| Task | Trial | Status | Official reward | Error |',
             '| --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| [{row['task_id']}]({row['slot']}/result.json) | {row['trial']} | {row['status']} | {row.get('reward', '—')} | {row.get('error', '')} |")
    lines.extend(['', f"官方 pass^k：{summary['pass_hat_k']}（全部完成判分后计算）"])
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')


async def execute(directory, job):
    from pydantic import TypeAdapter
    from tau2.data_model.message import AssistantMessage, Message
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.domains.retail.environment import get_tasks
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.llm_utils import set_llm_log_dir, set_llm_log_mode

    from evaluation.meter import usage_summary
    from evaluation.runtime import runtime

    task = next(task for task in get_tasks(job['split']) if task.id == job['task_id'])
    write(directory / 'task.json', task.model_dump(mode='json'))
    llm_args = dict(temperature=0, max_tokens=job['max_tokens'], num_retries=0, timeout=90)
    # Configure the model constants used by the upstream NL evaluator.
    nl.DEFAULT_LLM_NL_ASSERTIONS = job['judge_model']
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args
    set_llm_log_dir(directory / 'llm')
    set_llm_log_mode('all')
    user = UserSimulator(llm=job['user_model'], instructions=str(task.user_scenario), llm_args=llm_args)
    user.set_seed(job['seed'])
    user_state = user.get_init_state()
    reply = DEFAULT_FIRST_AGENT_MESSAGE.model_copy(deep=True)
    messages = [reply]
    message_type = TypeAdapter(Message)
    tool_lines_seen = 0
    started = time.monotonic()
    start_time = datetime.now(timezone.utc).isoformat()
    result = dict(task_id=task.id, trial=job['trial'], status='error')
    try:
        async with runtime(directory, job['call_limit']) as (http, workflow, effective):
            write(directory / 'effective.json', effective)
            termination = TerminationReason.MAX_STEPS
            for turn in range(job['max_turns']):
                user_message, user_state = await asyncio.to_thread(user.generate_next_message, reply, user_state)
                messages.append(user_message)
                write(directory / 'trajectory.json', [m.model_dump(mode='json') for m in messages])
                if user.is_stop(user_message):
                    termination = TerminationReason.USER_STOP
                    break
                response = await http.post('/chat', json=dict(message=user_message.content,
                    user_id='tau-session', conv_id=directory.name, client_request_id=f'turn-{turn}'))
                response.raise_for_status()
                if workflow._background:
                    await asyncio.gather(*list(workflow._background))
                data = response.json()
                write(directory / f'turn-{turn}.json', data)
                tool_lines = (directory / 'tools.jsonl').read_text().splitlines()
                messages.extend(message_type.validate_json(line) for line in tool_lines[tool_lines_seen:])
                tool_lines_seen = len(tool_lines)
                reply = AssistantMessage(role='assistant', content=data['response'])
                messages.append(reply)
                write(directory / 'trajectory.json', [m.model_dump(mode='json') for m in messages])
        simulation = SimulationRun(id=directory.name, task_id=task.id, start_time=start_time,
            end_time=datetime.now(timezone.utc).isoformat(), duration=time.monotonic()-started,
            termination_reason=termination, messages=messages, trial=job['trial'], seed=job['seed'])
        write(directory / 'simulation.json', simulation.model_dump(mode='json'))
        reward = await asyncio.to_thread(evaluate_simulation, simulation=simulation, task=task,
            evaluation_type=EvaluationType.ALL, solo_mode=False, domain='retail', strict_replay=True)
        simulation.reward_info = reward
        write(directory / 'simulation.json', simulation.model_dump(mode='json'))
        result.update(status='completed', reward=reward.reward,
                      reward_info=reward.model_dump(mode='json'), termination_reason=termination.value)
    except Exception as error:
        traceback.print_exc()
        result['error'] = f'{type(error).__name__}: {error}'
    result['duration_seconds'] = time.monotonic()-started
    result['system_usage'] = usage_summary(directory / 'usage.sqlite3')
    write(directory / 'result.json', result)
    return result


def prepare(config):
    from tau2.domains.retail.environment import get_tasks
    from tau2.utils import DATA_DIR
    from core.llm_utils import get_max_tokens

    tasks = get_tasks(config['split'])
    if config['task_id']:
        by_id = {task.id: task for task in tasks}
        tasks = [by_id[task_id] for task_id in dict.fromkeys(config['task_id'])]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    directory = ROOT / 'data/eval/tau' / run_id
    directory.mkdir(parents=True)
    rows = [dict(task_id=task.id, trial=trial, slot=f'{task.id}--{trial}', status='not_run')
            for task in tasks for trial in range(config['trials'])]
    sources = [p for folder in ('evaluation', 'agents', 'memory', 'tooling', 'core', 'api')
               for p in (ROOT/folder).glob('*.py')]
    assets = [DATA_DIR/'tau2/domains/retail'/name for name in ('db.json', 'tasks.json', 'split_tasks.json', 'policy.md')]
    write(directory / 'manifest.json', dict(revision=REVISION, config=config,
        model=os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'),
        system_max_tokens=get_max_tokens(), system_base_url=os.getenv('ANTHROPIC_BASE_URL'), slots=rows,
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        assets={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}))
    report(directory, rows)
    write(directory / 'state.json', dict(status='plan_only' if config['plan_only'] else 'planned', active_slot=None))
    return directory


def run(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    config, rows = manifest['config'], manifest['slots']
    process, row = None, None
    write(directory/'state.json', dict(status='running', active_slot=None))
    try:
        for index, row in enumerate(rows):
            if (directory/'cancel').exists():
                raise KeyboardInterrupt
            slot = directory / row['slot']
            slot.mkdir()
            row['status'] = 'running'
            report(directory, rows)
            write(directory/'state.json', dict(status='running', active_slot=row['slot']))
            write(slot/'job.json', {**config, **row, 'seed': config['seed'] + row['trial']})
            with (slot/'worker.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-m', 'evaluation.run', '--job', str(slot/'job.json')],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=config['timeout'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    row.update(status='error', error='单题执行超时')
                    write(slot/'result.json', row)
                    report(directory, rows)
                    write(directory/'state.json', dict(status='error', active_slot=None, error=row['error']))
                    return 1
            if process.returncode:
                row.update(status='error', error=f'worker exited {process.returncode}; see worker.log')
                write(slot/'result.json', row)
            else:
                row.update(json.loads((slot/'result.json').read_text()))
            report(directory, rows)
            print(f"{index+1}/{len(rows)} task={row['task_id']} trial={row['trial']} {row['status']} reward={row.get('reward')}", flush=True)
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if row is not None and row['status'] == 'running':
            row.update(status='cancelled', error='用户停止运行')
            write(directory/row['slot']/'result.json', row)
        report(directory, rows)
        write(directory/'state.json', dict(status='cancelled', active_slot=None, error='用户停止运行'))
        return 130
    except Exception as error:
        write(directory/'state.json', dict(status='error', active_slot=None, error=f'{type(error).__name__}: {error}'))
        raise
    write(directory/'state.json', dict(status='error' if any(row['status']=='error' for row in rows) else 'completed',
                                      active_slot=None))
    return 0 if all(row.get('reward') == 1 for row in rows) else 1


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'test', 'base'), default='test')
    parser.add_argument('--task-id', action='append')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--user-model')
    parser.add_argument('--judge-model')
    parser.add_argument('--max-turns', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--call-limit', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=300)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--run-directory', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        asyncio.run(execute(args.job.parent, json.loads(args.job.read_text())))
        return
    if args.run_directory:
        raise SystemExit(run(args.run_directory))
    if min(args.trials, args.max_turns, args.max_tokens, args.call_limit, args.timeout) < 1:
        parser.error('limits must be positive')
    if not args.plan_only and not (args.user_model and args.judge_model):
        parser.error('--user-model and --judge-model are required')
    config = vars(args).copy()
    config.pop('job')
    config.pop('run_directory')
    directory = prepare(config)
    print(f'Run directory: {directory}', flush=True)
    if not args.plan_only:
        raise SystemExit(run(directory))


if __name__ == '__main__':
    main()
