# WorldQuant Alpha Research

本仓库管理独立的自动化因子研究工作流。配置、启动命令和运行说明见 [research_workflow/README.md](research_workflow/README.md)。

工作流流程图及每次模型调用使用的提示词见 [research_workflow/WORKFLOW.md](research_workflow/WORKFLOW.md)。

`QuantGPT/` 和 `wqb/` 是各自独立的第三方 Git 检出目录，不属于本仓库。工作流默认直接调用 WorldQuant BRAIN；只有配置 `QUANTGPT_URL` 时才通过 HTTP 使用 QuantGPT。

账户配置 `research_workflow/config.env`、SQLite 状态、日志和 `alpha_libraries/` 下的因子研究结果仅保存在本机。复制 `research_workflow/config.env.example` 后填写账户信息，再运行 `python -m research_workflow`。
