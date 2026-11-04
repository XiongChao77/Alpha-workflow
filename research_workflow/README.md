# 运行与配置

项目概览见 [根目录 README](../README.md)。系统默认按 **Topic → Direction → Experiment** 研究，持续生成、回测和复盘；研究员也可在运行期间通过独立的 `chat` 命令讨论并排队新 Topic。完整流程及各模型提示词的输入输出见 [WORKFLOW.md](WORKFLOW.md)。

## 配置

使用 Python 3.11+，仅需标准库。复制 [config.env.example](config.env.example) 为 `research_workflow/config.env`，填写 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_KEY`、`WQ_BRAIN_EMAIL` 和 `WQ_BRAIN_PASSWORD`。模型提供方选择 `openai` 或 `deepseek` 之一。第三方兼容网关可设置 `LLM_BASE_URL`、`LLM_API_FORMAT=chat_completions` 或 `responses`；基础地址应类似 `https://.../v1`，不要手动追加 `/chat/completions`。`LLM_JSON_MODE=auto` 会在网关拒绝结构化格式时回退并进行本地 JSON 校验。

提示词都在 [prompts](prompts) 目录，启动后每次调用会重新读取。`config.env.example` 列出各提示词文件覆盖项。日志按启动时间分文件，记录模型调用阶段、提示词路径/哈希、探索方向、公式、BRAIN 指标和检查结果；密钥会遮盖。密钥不写入 SQLite 或 Alpha 归档。

重要预算：`ALPHA_MAX_CONCURRENT_SIMULATIONS=3`、`ALPHA_MAX_SIMULATIONS_DAY=20`、`ALPHA_MAX_DIRECTIONS_TOPIC=3`、`ALPHA_MAX_EXPERIMENTS_DIRECTION=8`、`ALPHA_FAMILY_MAX_STAGNANT=5`、`ALPHA_MAX_REPAIR_ATTEMPTS=2`、`ALPHA_PERIOD_GRID=30,60,90`。Period 阶段还会测试模型给出的一个值；只有 `abs(Sharpe) > 1.2` 才进入下一阶段，门槛可用 `ALPHA_SHARPE_GATE` 调整。负 Sharpe 需要反向公式真实回测，不能直接通过提交检查。连续 5 个批次没有改善会停止 Direction。平台限流响应优先于本地预算。

## 启动

在项目根目录：

```powershell
cd D:\Project\QR\WorldQuant-Alpha
if (-not (Test-Path research_workflow\config.env)) { Copy-Item research_workflow\config.env.example research_workflow\config.env }
# 编辑 config.env 后启动；如果文件已存在，无需再次复制
python -m research_workflow
```

可选：`python -m research_workflow check` 检查配置，`python -m research_workflow probe` 调用一次模型验证连接，`python -m research_workflow status` 查看任务。默认启动先处理优先级最高的已入队 Topic；没有待办时 Research Agent 自主生成新 Topic。按 Ctrl+C 停止，重启同一命令会从 SQLite 恢复。只运行一个 worker。

若从 `research_workflow` 文件夹启动，先回到上级目录，或在 PowerShell 设置 `PYTHONPATH=..` 再执行 `python -m research_workflow`。示例配置会从包目录自动读取。

## 研究员任务入口

直接入队已确认的研究问题：

```powershell
python -m research_workflow topic add "低成交量条件下的价格反转" --objective "寻找稳定、低相关的条件反转 Alpha" --priority 10
python -m research_workflow topic list
```

研究 worker 运行时，可在**另一个终端**执行 `python -m research_workflow chat`。输入研究想法，Agent 每轮给出讨论答复和结构化 Topic 草案；输入 `/queue` 才将草案以人工高优先级入库，`/quit` 退出但保留讨论。命令启动时打印讨论 ID，可用 `python -m research_workflow chat <讨论ID>` 恢复。原有 `topic chat`、`topic brainstorm` 仍可使用；`topic suggest` 仅生成建议、不入队。当前 Topic 会完整执行；其结束后才按优先级和入库时间选择下一 Topic，不中断在途回测。

## 实验与提交

每个 Direction 由若干结构不同的公式构成一个实验批次。程序先批量测试 Period；每个 Sharpe 小于 `-1.2` 的候选都会将公式取反，作为同族新候选送 BRAIN 重新回测，后续调参只使用实际测得 Sharpe 大于 `1.2` 的候选。模型随后看到同族原式、反向式、参数和完整结果；决定继续时在**同一次调用**返回优化方向和 1～2 个新假设及 FASTEXPR 模板，程序展开并回测。公式语法、单位、类型错误由 Alpha Agent 最多修复两次。所有候选，包括失败结果，都保留在 SQLite 和 `alpha_libraries/candidates/*.json`。

单因子批次结束时，系统还可从**同一 Direction** 选两个候选组合，按换手差异排序，不要求父因子的回测设置相同。组合使用当前目标配置重新回测，默认测试三个权重 `0.25/0.75`、`0.5/0.5`、`0.75/0.25`；每对因子的三个权重算一次组合尝试。组合最佳 Fitness 未超过某父因子本身的 Fitness，记该父因子一次失败；连续两次失败后不再选它参与新组合。不同设置下的父因子 Fitness 比较只用于配对退出启发式，组合是否合格以其自身的 BRAIN 检查为准。可用 `ALPHA_MAX_COMBINATION_PAIRS`、`ALPHA_COMBINATION_WEIGHTS` 控制组合预算。独立或组合正式提交会原子占用其基础因子，防止任何基础因子被提交两次。

所有通过 BRAIN 检查的因子统一保存在 `alpha_libraries/wq_<alpha_id>.md` 和 `eligible_alphas.json`，标记是否已提交。模型从合格且未提交的因子中选择当日一个；北京时间每天只占用一个正式提交槽位，即使提交响应不明确也不会当天重试另一个。每次向 BRAIN 发起模拟测试时，日志打印北京时间“今日第 N 次测试”和候选 ID；每 10 秒的统计日志打印合格总数、已提交数和今日提交数。

`python -m research_workflow export` 可从 SQLite 补齐结构化候选 JSON，不调用模型或平台。旧版 `create`、`run <run_id>`、`once <run_id>` 保留供旧任务使用；默认无参数命令使用新分层流程。已有且仍在运行的旧版 run 会在待办 Topic 处理完后自动纳入一个迁入批次，继续完成其在途候选并集中复盘。

## 限流与恢复

BRAIN 返回 429/503 时保存 `Retry-After` 冷却；未知模拟 POST 结果标记 `unknown`，不盲目重复提交。`WARNING` 且带 Alpha ID 时继续读取详情，有 IS 指标则进入检查，否则继续轮询。没有已证实适用于所有 BRAIN 账号的统一模拟日上限，`ALPHA_MAX_SIMULATIONS_DAY` 是本地安全预算，可按账户情况调整。
