"""One auditable Markdown record per BRAIN alpha ID."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile

from .brain import is_checks_passed
from .store import CST, group_families, today


def export_candidate(directory: Path, candidate: dict, submission: dict | None = None,
                     search: dict | None = None) -> Path:
    """Atomically persist a machine-readable record for every accepted candidate."""
    candidate_id = candidate["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", candidate_id):
        raise ValueError("Invalid candidate ID")
    result = json.loads(candidate["result"] or "{}")
    is_data = result.get("is") or {}
    worldquant = {
        "alpha_id": candidate["alpha_id"],
        "simulation_url": candidate["simulation_url"],
        "simulated_at": candidate["simulated_at"],
        "platform_status": result.get("status"),
        "metrics": {key: is_data.get(key) for key in ("sharpe", "fitness", "returns", "turnover")},
        "checks": [{"name": check.get("name"), "result": check.get("result"),
                    "value": check.get("value"), "limit": check.get("limit")}
                   for check in (is_data.get("checks") or [])],
        "raw_result": result if candidate["simulated_at"] else None,
    }
    record = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "run_id": candidate["run_id"],
        "family_id": candidate.get("family_id") or candidate_id,
        "round": candidate["round"],
        "parent_candidate_id": candidate.get("parent_candidate_id"),
        "origin_action": candidate.get("origin_action") or "GENERATE",
        "parameter_values": json.loads(candidate.get("parameter_values") or "{}"),
        "parameter_search": ({"expression_template": search["expression_template"],
                              "period_default": search["period_default"],
                              "parameters": json.loads(search["parameters"]),
                              "stage": search["stage"],
                              "stage_candidate_ids": json.loads(search["stage_trials"]),
                              "selected_period": search["selected_period"],
                              "selected_params": json.loads(search["selected_params"])} if search else None),
        "created_at": candidate["created_at"],
        "status": candidate["status"],
        "hypothesis": candidate["hypothesis"],
        "expression": candidate["expression"],
        "settings": json.loads(candidate["settings"]),
        "quantgpt": {"task_id": candidate["quantgpt_task_id"],
                     "result": result.get("quantgpt") if isinstance(result, dict) else None},
        "worldquant": worldquant,
        "submission": ({"day": submission["day"], "status": submission["status"],
                        "detail": submission["detail"],
                        "submitted_at": submission.get("submitted_at")} if submission else None),
        "submitted": bool(submission and (submission.get("submitted_at") or
                                          submission["status"] in {"sent", "pending", "active"})),
        "submission_uncertain": bool(submission and submission["status"] == "uncertain"),
        "error": result.get("error") if isinstance(result, dict) else None,
    }
    path = Path(directory) / "candidates" / f"{candidate_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{candidate_id}_", delete=False) as stream:
        stream.write(content)
        temp_name = stream.name
    os.replace(temp_name, path)
    return path


def export_candidates(store, directory: Path, run_id: str | None = None) -> int:
    """Synchronize JSON files from the SQLite source of truth."""
    submissions = {row["candidate_id"]: row for row in store.all(
        "SELECT candidate_id,day,status,detail,submitted_at FROM submissions")}
    if run_id is None:
        candidates = store.all("SELECT * FROM candidates")
    else:
        candidates = store.all("SELECT * FROM candidates WHERE run_id=?", (run_id,))
    for root_id, members in group_families(candidates).items():
        for candidate in members:
            candidate["family_id"] = root_id
    searches = {row["family_id"]: row for row in store.all("SELECT * FROM parameter_searches")}
    for candidate in candidates:
        export_candidate(directory, candidate, submissions.get(candidate["id"]),
                         searches.get(candidate["family_id"]))
    write_eligible_index(store, directory)
    return len(candidates)


def eligible_records(store):
    """All candidates whose saved BRAIN checks passed, with durable submit status."""
    submissions = {row["alpha_id"]: row for row in store.all("SELECT * FROM submissions")}
    lineage = store.all("SELECT id,parent_candidate_id FROM candidates")
    family_ids = {member["id"]: root for root, members in group_families(lineage).items() for member in members}
    records = []
    seen = set()
    for candidate in store.all("SELECT * FROM candidates WHERE status='passed' AND alpha_id IS NOT NULL "
                               "ORDER BY created_at,id"):
        alpha_id = candidate["alpha_id"]
        if alpha_id in seen:
            continue
        result = json.loads(candidate["result"] or "{}")
        is_data = result.get("is") or {}
        if not is_checks_passed(is_data):
            continue
        seen.add(alpha_id)
        submission = submissions.get(alpha_id)
        submitted = _confirmed_submission(submission)
        records.append({
            "alpha_id": alpha_id, "candidate_id": candidate["id"], "run_id": candidate["run_id"],
            "family_id": family_ids.get(candidate["id"], candidate["id"]),
            "hypothesis": candidate["hypothesis"], "expression": candidate["expression"],
            "parameter_values": json.loads(candidate.get("parameter_values") or "{}"),
            "metrics": {key: is_data.get(key) for key in ("sharpe", "fitness", "returns", "turnover")},
            "checks": is_data.get("checks") or [], "submitted": submitted,
            "submission_status": submission["status"] if submission else "not_submitted",
            "submission_day": submission["day"] if submission else None,
            "submitted_at": submission.get("submitted_at") if submission else None,
            "submission_uncertain": bool(submission and submission["status"] == "uncertain"),
        })
    return records


def _confirmed_submission(submission):
    return bool(submission and (submission.get("submitted_at") or
                                submission["status"] in {"sent", "pending", "active"}))


def _submission_counts(store):
    submitted = [row for row in store.all("SELECT * FROM submissions") if _confirmed_submission(row)]
    return {"submitted_total": len({row["alpha_id"] for row in submitted}),
            "submitted_today": len({row["alpha_id"] for row in submitted if row["day"] == today()})}


def eligible_counts(store):
    records = eligible_records(store)
    return {"eligible_total": len(records), **_submission_counts(store)}


def write_eligible_index(store, directory: Path) -> Path:
    records = eligible_records(store)
    data = {"schema_version": 1, "eligible_total": len(records), **_submission_counts(store), "alphas": records}
    path = Path(directory) / "eligible_alphas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".eligible_", delete=False) as stream:
        stream.write(content)
        temp_name = stream.name
    os.replace(temp_name, path)
    return path


def archive(directory: Path, candidate: dict, platform: dict, stage: str, decision: dict | None = None) -> Path:
    alpha_id = candidate.get("alpha_id")
    if not alpha_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", alpha_id):
        raise ValueError("Invalid BRAIN alpha ID")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wq_{alpha_id}.md"
    result = json.loads(candidate["result"] or "{}")
    settings = json.loads(candidate["settings"])
    checks = (platform.get("is") or result.get("is") or {}).get("checks", [])
    summary = platform.get("is") or result.get("is") or {}
    lines = [f"# WorldQuant Alpha {alpha_id}", "", f"- 状态：{stage}",
             f"- 更新：{datetime.now(CST).isoformat(timespec='seconds')}",
             f"- 研究任务：{candidate['run_id']}", f"- 候选编号：{candidate['id']}", "",
             "## 因子思路", "", candidate["hypothesis"], "", "## FASTEXPR 公式", "", "```text",
             candidate["expression"], "```", "", "## BRAIN 设置", "", "```json",
             json.dumps(settings, ensure_ascii=False, indent=2), "```", "", "## 平台结果", "",
             f"- Sharpe：{summary.get('sharpe')}", f"- Fitness：{summary.get('fitness')}",
             f"- Returns：{summary.get('returns')}", f"- Turnover：{summary.get('turnover')}",
             f"- 平台状态：{platform.get('status', '')}", "", "### 检查项", ""]
    lines.extend(f"- {c.get('name', '?')}: {c.get('result', '?')} (value={c.get('value', '')}, limit={c.get('limit', '')})" for c in checks)
    if decision:
        lines.extend(["", "## 模型提交判断", "", decision.get("reason", "")])
    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, prefix=f".wq_{alpha_id}_", delete=False) as stream:
        stream.write(content)
        temp_name = stream.name
    os.replace(temp_name, path)
    return path
