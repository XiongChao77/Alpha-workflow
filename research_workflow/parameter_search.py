"""Safe named-parameter rendering and staged search selection."""

import json
import math
import re


PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Parameter values must be finite numbers")
    return int(value) if float(value).is_integer() else value


def normalize_plan(candidate):
    template = candidate.get("expression_template")
    period = candidate.get("period")
    parameters = candidate.get("parameters", [])
    if not isinstance(template, str) or not template.strip():
        raise ValueError("expression_template must be a nonempty string")
    if not isinstance(period, dict) or period.get("name") != "Period":
        raise ValueError("period must have name Period and one default value")
    period_default = period.get("default")
    if (isinstance(period_default, bool) or not isinstance(period_default, (int, float)) or
            not math.isfinite(period_default) or not float(period_default).is_integer() or
            period_default < 1 or period_default > 1000):
        raise ValueError("Period default must be an integer in 1..1000")
    period_default = int(period_default)
    if not isinstance(parameters, list) or len(parameters) > 2:
        raise ValueError("parameters must contain at most two additional dimensions")
    normalized = []
    names = {"Period"}
    for item in parameters:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not NAME.fullmatch(item["name"]):
            raise ValueError("Parameter name must be an ASCII identifier")
        name = item["name"]
        if name in names:
            raise ValueError(f"Duplicate parameter name: {name}")
        names.add(name)
        default = _number(item.get("default"))
        values = item.get("values")
        if not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise ValueError(f"{name} needs 1..8 explicit candidate values")
        values = list(dict.fromkeys(_number(value) for value in [default, *values]))
        normalized.append({"name": name, "default": default, "values": values})
    found = set(PLACEHOLDER.findall(template))
    if found != names or re.sub(PLACEHOLDER, "", template).find("{") >= 0 or re.sub(PLACEHOLDER, "", template).find("}") >= 0:
        raise ValueError(f"Template placeholders must match declared parameters; found={sorted(found)} expected={sorted(names)}")
    return {"expression_template": template.strip(),
            "period": {"name": "Period", "default": period_default}, "parameters": normalized}


def render_expression(template, values):
    expression = template
    for name, value in values.items():
        expression = expression.replace("{" + name + "}", str(_number(value)))
    if "{" in expression or "}" in expression:
        raise ValueError("Unresolved expression placeholder")
    return expression


def period_values(grid, model_default):
    values = list(dict.fromkeys([*grid, model_default]))
    if any(type(value) is not int or value < 1 or value > 1000 for value in values):
        raise ValueError("Period grid must contain positive integers <= 1000")
    return values


def sharpe(candidate):
    result = json.loads(candidate["result"] or "{}")
    value = (result.get("is") or {}).get("sharpe")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def best_candidate(candidates):
    eligible = [(sharpe(candidate), candidate) for candidate in candidates]
    eligible = [(score, candidate) for score, candidate in eligible if score is not None]
    return max(eligible, key=lambda item: abs(item[0]))[1] if eligible else None
