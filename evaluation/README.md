# τ retail 评测

只使用 τ²-bench 官方零售任务、政策、数据库、UserSimulator 和官方 evaluator。
固定版本：`fc0055dc4e0a316c3f83133267fbd6faaa770992`（tau2 1.0.1）。
源码：[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992)。
官方划分为 train 74 题、test 40 题；接入调试用 train，正式评测用 test。

## 运行

### 页面运行

```bash
docker compose up -d --build echomind
```

打开 [τ 评测页面](http://localhost:8000/eval)。Compose 会同时启动 `tau-eval`、Redis 和 ChromaDB。
页面默认选择 train 的一道题；选择题集、题目和重复次数后点击「开始评测」。
选择 test 和「整个题集」即可运行全部 40 道测试题。

页面提供实时进度、历史运行、逐题结果、对话与工具轨迹、官方判分详情和文件下载。
刷新或关闭页面不停止实验；「停止运行」会终止当前题的进程，保留已经完成的结果，
其余题保持未运行。服务重启时中断的实验会明确标记，不自动恢复或重新调用模型。
默认一次只启动一个页面实验，重复启动返回 409。

页面通过主应用 `/eval/api/*` 调用内网 `tau-eval` 服务；不把 Docker 控制接口挂进应用。
`tau-eval` 和 CLI 使用相同镜像、`prepare/run`、官方数据和评分器，没有另一套页面评测逻辑。
评测服务不直接开放宿主机端口。

当前 Compose 配置针对项目正在使用的 DeepSeek：系统、用户模拟器和 Judge 显式共用
`ANTHROPIC_API_KEY` 中的 DeepSeek 凭证，用户/Judge 默认 `deepseek/${ANTHROPIC_MODEL}`。
页面默认模型从容器实际配置读取，不用宿主机的同名变量插值。
页面可以指定用户/Judge 模型，使用的供应商凭证必须已在服务端配置；不会自动更换模型或供应商。

### 命令行运行

评测镜像直接安装官方源码及原始数据，不需要转换任务或下载本项目自建题集。

```bash
docker compose build evaluation
docker compose run --rm evaluation --split test --plan-only
```

真实运行需要项目已有的 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL` 配置用于 EchoMind。
用户模拟器和官方 NL Judge 直接使用 τ 的 LiteLLM 调用。
独立 CLI 通过供应商环境变量取凭证。例如将 DeepSeek 凭证放在 `.env` 的
`DEEPSEEK_API_KEY` 后运行下列命令；复用已有凭证的命令见后文接入验证记录。

```bash
# 接入验证：官方训练题 0，保持原任务和判分要求。
docker compose run --rm evaluation --split train --task-id 0 \
  --user-model deepseek/deepseek-flash --judge-model deepseek/deepseek-flash

# 正式测试：40 道官方测试题，各运行 3 次。
docker compose run --rm evaluation --split test --trials 3 \
  --user-model deepseek/deepseek-flash --judge-model deepseek/deepseek-flash
```

Compose 会启动真实 Redis 和 Chroma；未运行 Docker 或依赖不可达时失败。
每题独立进程、Redis 前缀、Chroma database、问题记录和 checkpoint。
评测通过 TCP HTTP 调用现有 `/chat` 与 ChatWorkflow，再经 MCP 调用官方 RetailTools。
评测加载官方 policy，不加载本地业务 Skills；不会向 Agent 提供参考操作或真实业务用户 ID。
正常应用的本地订单业务不变，τ 数据保存在本题独立环境中，不映射到本地退货表。

## 判分

`evaluate_simulation(..., evaluation_type=ALL, domain='retail', strict_replay=True)`
直接计算官方 reward。参考操作仅用于官方判分器创建目标终态，不作为 Agent 的执行脚本。
任务的 `reward_basis` 决定 DB、NL 等分量；没有 NL 要求时不额外调用自建 Judge。
模型调用或判分抛错会保存为 error，没有默认分数、自动补答或重试到通过。

默认单题最多 30 次用户交互、200 次 EchoMind 模型调用、1200 秒；
用户与 Judge 的输出上限为 8192 tokens，系统输出上限仍由 `CLAUDE_MAX_TOKENS` 指定。
用户/Judge 的 SDK 重试设为 0；官方 seed 原样传递，供应商是否实现确定性取决于其接口。
`--trials` 改变重复次数；`--max-turns`、`--call-limit`、`--timeout` 调整实验预算。
所有模型与预算在 manifest 中记录，不在运行中改动。

## 文件

- `run.py`：读取官方任务、用户对话循环、官方评分与结果汇总。
- `retail.py`：调用官方 Retail 环境，使用已有 MCP 服务传递原始工具参数与结果。
- `runtime.py`：装配真实 EchoMind HTTP 工作流及隔离存储。
- `chroma.py`：创建服务端独立数据库。
- `meter.py`：记录 EchoMind 各组件的模型调用与 usage。
- `web.py`：启动和停止实验、查询进度、读取结果文件。
- `api/evaluation.py`、`api/evaluation.html`：主应用的页面和 HTTP 入口。

产物位于 `data/eval/tau/<run-id>/`：

- `manifest.json`：版本、原始资产和源码哈希、配置、计划任务。
- `report.md`、`results.json`、`summary.json`：官方 reward、错误和未运行项。
- 每题的 `task.json`、`trajectory.json`、`simulation.json`：原题、τ 原生消息及官方结果。
- `before.json`、`after.json`：实际 MCP 进程中官方 RetailDB 的前后快照。
- `turn-*.json`、`tools.jsonl`、`mcp-tools.jsonl`：真实 HTTP 回复和工具执行证据。
- `usage.sqlite3`：EchoMind 模型请求、响应和 tokens；`llm/`：τ 原生用户/Judge 请求与响应记录。
- `worker.log`：执行日志。

失败仍在分母内；完整通过退出 0，任务失败或执行错误退出非 0。
只有所有计划任务完成官方判分后才发布 `pass_hat_k`，使用 τ 原版 `pass_hat_k` 计算。
例如一题三次成功两次，pass^1=2/3，pass^2=1/3，pass^3=0；不挑选最好的一次。
有错误或未运行项时该指标为 null，另外展示“成功数 / 计划数”，避免把不完整实验当成正式成绩。
这测量 EchoMind 在官方零售业务上的任务能力，不代表已证明 RAG、长期记忆或各组件的收益。
旧自建客服题集、自定义 scorer、消融开关和 `/eval/run` 已移除，不再维护兼容入口。

## 接入验证记录（2026-11-06）

实际使用 Compose 的 Redis、ChromaDB 和 evaluation 容器，以及真实 DeepSeek 模型：

| 官方 train task | 场景 | 官方结果 | 产物 |
| --- | --- | --- | --- |
| 0 | 身份验证、商品选择、确认后一次申请两件换货 | DB=1，reward=1；无 NL 断言 | `20261106T190048Z-5267c02e` |
| 2 | 查询商品规格、定位订单、确认退货，并回答可售 T 恤数量 | DB=1，NL=1，reward=1 | `20261106T190255Z-6f4ad0e9` |

这是两道训练题的接入验证，未运行完整 40 题 test，不作为正式 benchmark 成绩。
存储复核：两题分别写入 4/5 个 Redis key、1/1 条 Chroma 用户画像、24/32 个 checkpoint；
两题实际 RetailDB 快照均与官方严格重放完全一致，详见各题 `storage_verification.json`。
两题对话都未触发摘要压缩，因此没有验证情景记忆归档或长期记忆收益。
最终镜像通过现有 132 项测试及 3 项 τ 接入测试；主应用、Redis、ChromaDB 均健康，
主应用 `/health` 与 `/mcp/status` 返回 200，删除的 `/eval/run` 返回 404。
这次运行通过下列命令显式复用现有 DeepSeek 凭证，未修改 `.env` 或切换系统模型：

```bash
docker compose run --rm --entrypoint sh evaluation -c \
  'export DEEPSEEK_API_KEY="$ANTHROPIC_API_KEY"; python -m evaluation.run --split train --task-id 2 --user-model deepseek/deepseek-flash --judge-model deepseek/deepseek-flash'
```

仅在现有 `ANTHROPIC_API_KEY` 确实是 DeepSeek 凭证时使用这个命令；其他供应商按前文独立配置。

代码测试：

```bash
docker compose run --rm --no-deps --entrypoint python evaluation -m unittest discover -s tests
docker compose run --rm --no-deps --entrypoint python evaluation -m unittest evaluation.test_retail
docker compose run --rm --no-deps --entrypoint python evaluation -m unittest evaluation.test_web
```

接入测试只验证基础设施：真实 MCP 子进程修改官方 DB，重放结果等于实际快照，
正确操作得 1、未执行操作得 0、篡改工具返回值导致严格重放失败，以及 pass^k 计算。
这些测试不是新的自建评测集，参考操作只出现在接入测试中。

页面接入验证：浏览器实际选择 train task 0 并点击「开始评测」，
运行 `20261106T195506Z-fef828f5` 完成官方判分，reward=1；系统、用户和 Judge 均使用
项目当前的 `deepseek-flash`（用户/Judge 通过 LiteLLM 的 `deepseek/deepseek-flash` 调用）。
已验证 train/test/base 分别加载 74/40/114 题，刷新保持同一次运行，重复提交返回 409。
另一次有意点击停止的实验 `20261106T195855Z-ca18c10b` 记录为 1 题已停止、1 题未运行，
没有默认分数或 pass^k。8 项任务控制测试覆盖输入校验、模型配置、进程崩溃、超时回收和重启中断。
