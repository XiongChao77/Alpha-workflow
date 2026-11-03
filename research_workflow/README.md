# WorldQuant Alpha Research Workflow

独立于 `QuantGPT` 的持久化因子研究工作流。通过 API 调用 OpenAI/DeepSeek 和 WorldQuant BRAIN；可选通过 HTTP 调用现有 `QuantGPT` 本地回测。不会修改或导入 `QuantGPT` 源码。

## 工作方式

完整的 Mermaid 工作流图、五类模型调用的输入输出及各自使用的提示词文件，见 [WORKFLOW.md](WORKFLOW.md)。

1. 模型按研究方向生成带经济假设的 FASTEXPR 模板、一个固定的 `Period` 默认值，以及最多两个额外参数及其离散候选值。程序依次测试 `ALPHA_PERIOD_GRID`（默认 `30,60,90`）与模型给出的 Period 值，去重后逐个发送 BRAIN；所有其他参数先用默认值。
2. workflow 校验基本格式、跨轮去重，并可选调用 QuantGPT 本地回测。
3. BRAIN 模拟完成后读取其 IS 检查。若模拟接口返回 `WARNING` 但已提供 Alpha ID，程序会读取 Alpha 详情；详情已有 IS 结果时直接进入检查，否则继续轮询。所有现有 IS 检查均为 `PASS` 时，立即在 `alpha_libraries/wq_<alpha_id>.md` 保存思路、公式、设置和结果。
4. Period 测试全部完成后，若没有任何结果的 Sharpe **严格大于 1**，该族停止；否则选择 Sharpe 最高的 Period，固定它后探索第一个额外参数。第二维完成后，模型结合该族结果决定是否探索第三维。随后决策模型一次只看到**一个假设族**的完整历史（根候选及其参数搜索、`OPTIMIZE`/`TUNE` 子候选），返回 `NEW_DIRECTION`、`OPTIMIZE`、`TUNE` 或 `SUBMIT`。`OPTIMIZE` 一次可给出 1～2 组新假设与新公式，分别入队并利用 BRAIN 并发槽位；`TUNE` 沿用公式修改参数。
5. 所有通过 BRAIN 检查的 Alpha 统一写入 `alpha_libraries/eligible_alphas.json`，包含 `submitted`、提交状态和日期。模型从尚未提交的合格因子中选择当日一个候选。北京时间每天、整个工作流数据库最多占用一个正式提交槽位。发起提交后回查 BRAIN 状态，并更新同一份归档。即使结果不明确，也不会当天再次提交另一个 Alpha。

任务、候选、决策、提交占位和服务冷却时间写入 SQLite。服务重启后可继续 `run`。如果模拟 POST 的结果不明确，候选标为 `unknown`，需要在 BRAIN 核实后处理，避免重复创建。

每个已接收的候选还会同步写入 `alpha_libraries/candidates/<candidate_id>.json`，包含 `hypothesis`、`expression`、设置、状态、WorldQuant 指标与逐项检查、原始测试结果、提交状态、`family_id` 和来源候选 ID。未通过的候选也保留，便于后续分析和模型优化；通过检查或提交的 Alpha 仍在 `alpha_libraries/wq_<alpha_id>.md` 归档。重新启动工作流后，旧候选会在下一步自动补齐 JSON 文件。

也可在运行中的 worker 旁边执行 `python -m research_workflow export`，直接从 SQLite 补齐所有历史候选的 JSON；该命令不连接模型或 WorldQuant。

## 配置与启动

Python 3.11+，仅使用标准库。先复制示例，编辑 `research_workflow/config.env`，填入模型 API Key 和 BRAIN 账号。生成与复盘判断共用一组 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_KEY`：选择 `openai` 或 `deepseek`，不需要同时填写两家。程序会自动读取这个文件；显式 `--config <路径>` 可选用其他文件。已有进程环境变量优先于文件值。密钥不会写进 SQLite 或 Alpha 归档。`config.env` 已列入 `.gitignore`。

第三方 OpenAI 兼容网关可设置 `LLM_BASE_URL`（**基础地址**，如 `https://www.jiji.cc/v1`）和 `LLM_API_FORMAT=chat_completions`，程序会追加 `/chat/completions`。若该模型线路只支持 Responses，改为 `LLM_API_FORMAT=responses`，程序会追加 `/responses`。`LLM_MODEL` 必须填写该网关实际开放的模型 ID。`LLM_JSON_MODE=auto` 会先请求 JSON 格式；网关明确拒绝该格式时降级为普通输出，再做本地 JSON 校验。`LLM_DISABLE_RESPONSE_STORAGE=true` 会发送 `store:false`；若网关不支持此参数，需让服务商确认，程序不会悄悄改变存储设置。

生成、假设族决策、第三维探索决策、当日提交选择和连通性测试的系统提示词分别在 `research_workflow/prompts/generate.txt`、`decide.txt`、`parameter_decide.txt`、`select_submission.txt`、`probe.txt`。可直接编辑这些文件，下一次模型调用就会读取新内容；也可在配置中用对应的 `LLM_*_PROMPT_FILE` 指定自定义 UTF-8 文本文件。工作流把研究数据作为单独的 JSON 用户消息发送，字段名与返回格式仍须遵守程序的校验规则。每次调用都会在日志中记录提示词路径和摘要哈希，以便复盘使用了哪个版本。

模板仅支持 `{Period}` 和声明过的 `{参数名}` 占位符，程序只替换这些名字，不会误改公式里其他数字。每个搜索阶段、候选参数取值和已测试结果都写入 SQLite 和候选 JSON；重启后继续。额外参数每维最多 8 个离散值，当前默认每日模拟预算仍由 `ALPHA_MAX_SIMULATIONS_DAY` 控制。

模型 API 调用不保留对话会话。生成新研究方向时会用最近 30 个候选避免重复；**决策时只从 SQLite 构造当前假设族的完整历史**，不混入其他族。每条记录都有候选 ID、假设、公式、设置、状态和回测摘要，因此模型可通过 `target_id` 指向要改进的候选。`OPTIMIZE` 返回的新假设与公式会保存 `parent_candidate_id`，可追溯来源。

完成参数网格后，一个假设族若连续 **5 次模型提出的 `OPTIMIZE`/`TUNE` 候选**未改善，就停止该族，继续处理其他族；可用 `ALPHA_FAMILY_MAX_STAGNANT` 调整，设为 `0` 可关闭此限制。参数网格本身不计入这五次，但其最佳结果会成为比较基线。改善按 BRAIN 未通过检查项数优先、其次按这些检查超出阈值的相对幅度判断；每出现一次优于该族历史最佳的结果，就把连续未改善次数清零。无法得到有效 BRAIN 检查的后续候选计为未改善。停止事件写入决策日志和 SQLite。旧版没有父候选 ID 的调参记录，会在启动时按同公式的上一版本补齐关系。

可先运行 `python -m research_workflow probe` 发起**一次小型模型调用**检查网关连接、模型 ID 和响应格式；它不会连接 BRAIN，也不会提交 Alpha。该调用可能产生少量 API 费用。

日常启动只需一条命令。程序自动创建或恢复与当前目标匹配的研究任务，持续生成、模拟、判断和提交；任务 ID 只用于内部记录。`check` 是可选诊断命令。

Windows PowerShell（已安装 Python 3.11+）：

```powershell
cd D:\Project\QR\WorldQuant-Alpha
if (-not (Test-Path research_workflow\config.env)) { Copy-Item research_workflow\config.env.example research_workflow\config.env }
# 编辑 research_workflow\config.env，填入密钥和目标设置
python -m research_workflow
```

当前工作环境的 Windows 端没有可用的 Python，可从 PowerShell 调用 Ubuntu WSL 的 Python：

```powershell
cd D:\Project\QR\WorldQuant-Alpha
if (-not (Test-Path research_workflow\config.env)) { Copy-Item research_workflow\config.env.example research_workflow\config.env }
# 编辑 research_workflow\config.env
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/Project/QR/WorldQuant-Alpha && python3 -m research_workflow'
```

新开终端查看状态：`python -m research_workflow status`（WSL 环境同样包在 `wsl ... bash -lc` 中）。保持进程运行；按 Ctrl+C 停止，之后执行同一启动命令即可继续。不要为同一个任务同时启动两个 worker。`ALPHA_MAX_ROUNDS=0` 表示持续挖掘；`ALPHA_MAX_PENDING` 控制当日模拟预算耗尽后的候选积压上限。可选的 `ALPHA_OBJECTIVE`、`ALPHA_DIRECTION` 能在配置文件中指定初始研究目标和方向；留空时自动根据 BRAIN 目标生成。

`create`、`run <run_id>`、`once <run_id>` 保留给需要管理多个独立研究任务的高级用法。每日提交的数据库唯一约束会拦住多进程重复提交，但候选模拟流程目前按单 worker 设计。

## 运行日志

控制台按 `ALPHA_LOG_LEVEL` 输出进度（默认 `INFO`），包括当前探索方向、候选假设和公式、模型决策与理由、BRAIN 模拟指标和逐项检查、提交状态及服务退让时间。每次启动会在 `research_workflow/logs/` 新建独立文件，例如 `workflow_20261104_003000_123456_12345.log`，并在启动日志中打印实际路径；同一次运行的模型输入输出、BRAIN 完整结果仍写入该文件。单个文件达到 10 MB 后轮转，保留 5 份备份；`ALPHA_LOG_FILE` 用于指定目录和文件名前缀。配置中的模型密钥、BRAIN 密码和 QuantGPT token 会从日志文本中遮盖；日志可能包含研究目标、公式、模型回复和平台结果，应按私有研究数据保管。

持续运行时，独立统计线程每 10 秒打印一次“符合条件总数、已提交因子数、今日提交因子数”。已提交数指已收到成功提交响应或之后经 BRAIN 确认 ACTIVE 的因子；提交请求结果不明确的记录会单独标为 `uncertain`，仍占用当日唯一提交槽位。

模型返回的候选需要 `hypothesis` 和 `expression`。兼容网关把思路写成 `economic_hypothesis` 时会自动映射并记录；其他缺失字段会在控制台指出候选序号和字段。只有通过该本地校验后才会调用 BRAIN。

## QuantGPT 适配

配置 `QUANTGPT_URL` 后，每个候选先调用其 `/api/v1/auto_backtest`，再轮询 `/api/v1/tasks/{task_id}`。需要认证时设置 `QUANTGPT_TOKEN`。可设置 `QUANTGPT_UNIVERSE`、`QUANTGPT_MIN_SCORE`。默认**不启用**本地筛选，因为 QuantGPT 示例使用 A 股数据，而 BRAIN 默认目标是 USA/TOP3000；先确认数据和算子语义一致，再让本地分数阻断 BRAIN 模拟。

## 限流与恢复

对 BRAIN 的 429/503 读取 `Retry-After`（秒数或 HTTP 日期），将冷却时间持久化。模拟轮询也遵循响应头。没有公开证实适用于所有 BRAIN 账号的固定每日模拟额度，因此本地预算是自设上限，平台响应优先。提交 POST 的结果不明确时只回查，不盲目重试。

`ALPHA_MAX_CONCURRENT_SIMULATIONS` 控制同时在 BRAIN 运行的模拟数量，默认 **3**。一次工作流推进会依次填满可用槽位，并分别轮询到期任务；这是多个服务器端回测同时在途，HTTP 请求本身依次发送。单工作流进程下，槽位统计覆盖同一 SQLite 数据库中的所有研究任务；POST 结果不明确的 `unknown` 候选也暂占槽位，避免误超额。BRAIN 返回并发限流时仍按 `Retry-After` 退让。该默认值来自当前账户的实测，其他账户可在配置文件中调整。

首次连接真实账号前，需要用该账号核对可用 region、universe、字段、算子和提交检查。当前生成提示词仅列常见字段，BRAIN 仍是最终语法与结果判定来源。
