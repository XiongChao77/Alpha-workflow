# 使用项目模型配置启动 Codex CLI

项目根目录的 [codex_project.py](codex_project.py) 从已忽略的 `research_workflow/config.env` 读取 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，把密钥仅作为本次 Codex 子进程的环境变量传入。脚本不复制密钥到全局 Codex 配置，也不改变桌面版 Codex 的默认模型。需安装 Python 3 和 Codex CLI（`npm install -g @openai/codex`）。

在 Windows PowerShell 中运行（无需修改执行策略）：

```powershell
cd D:\Project\QR\WorldQuant-Alpha
python .\codex_project.py
```

在 Linux shell 中运行：

```bash
python3 ./codex_project.py
```

可传入 Codex CLI 子命令，例如非交互运行或恢复最近会话：

```powershell
python .\codex_project.py exec "只阅读项目，概述当前研究工作流"
python .\codex_project.py resume --last
```

如网关支持另一个模型的 Responses 接口，可临时指定模型，而无需修改研究工作流的模型：

```powershell
$env:CODEX_PROJECT_MODEL = '<网关支持的模型名>'
python .\codex_project.py
Remove-Item Env:CODEX_PROJECT_MODEL
```

脚本将 `LLM_BASE_URL` 作为自定义提供方地址，并通过 `env_key` 读取密钥。**Codex CLI 的自定义提供方仅支持 Responses API**，而研究工作流当前使用 `LLM_API_FORMAT=chat_completions`。这两个客户端可以共用 URL 和密钥，前提是第三方网关还支持同一模型的 `/responses` 接口；只支持 `/chat/completions` 的网关无法直接用于 Codex CLI。若提示模型不可用或返回 404，请向网关确认可用的 Responses 模型名。

脚本只给 Codex 子进程传入密钥，不修改当前终端环境。密钥不会出现在命令行参数、脚本文件或 Git 提交中；仍请保持 `research_workflow/config.env` 私有。
