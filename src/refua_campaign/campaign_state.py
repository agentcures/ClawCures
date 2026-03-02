from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_STATE_SCHEMA_VERSION = "1.0"
_DEFAULT_MAX_RUNS = 250
_DEFAULT_MAX_FAILURES = 2000
_DEFAULT_MAX_NEGATIVE_RESULTS = 2000


def default_campaign_state_path() -> Path:
    env_value = os.getenv("REFUA_CAMPAIGN_STATE_PATH", "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".clawcures" / "campaign_state.json"


def load_campaign_state(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return _empty_state()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(payload, Mapping):
        return _empty_state()
    state = _empty_state()
    for key in ("schema_version", "updated_at", "runs", "failures", "negative_results"):
        value = payload.get(key)
        if isinstance(value, list):
            state[key] = [item for item in value if isinstance(item, Mapping)]
        elif isinstance(value, str):
            state[key] = value
    registry = payload.get("program_registry")
    if isinstance(registry, Mapping):
        state["program_registry"] = {
            str(key): dict(value)
            for key, value in registry.items()
            if isinstance(value, Mapping)
        }
    return state


def persist_campaign_state(
    *,
    objective: str,
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    promising_cures: list[Mapping[str, Any]],
    interesting_targets: list[Mapping[str, Any]],
    session_key: str | None = None,
    state_path: Path | None = None,
    max_runs: int = _DEFAULT_MAX_RUNS,
) -> dict[str, Any]:
    path = (state_path or default_campaign_state_path()).expanduser().resolve()
    state = load_campaign_state(path)
    now = _utc_now()
    run_id = _stable_run_id(objective=objective, timestamp=now, plan=plan)

    run_summary = {
        "run_id": run_id,
        "captured_at": now,
        "objective": str(objective),
        "session_key": session_key,
        "plan_calls": _plan_call_count(plan),
        "result_count": len(results),
        "promising_count": _count_promising(promising_cures),
        "interesting_target_count": len(interesting_targets),
        "failed_tool_calls": len(_extract_failures(results)),
    }

    runs = list(state.get("runs", []))
    runs.append(run_summary)
    state["runs"] = runs[-max(1, int(max_runs)) :]

    failures = list(state.get("failures", []))
    failures.extend(_extract_failures(results, run_id=run_id, objective=objective))
    state["failures"] = failures[-_DEFAULT_MAX_FAILURES:]

    negative_results = list(state.get("negative_results", []))
    negative_results.extend(
        _extract_negative_results(
            promising_cures,
            run_id=run_id,
            objective=objective,
        )
    )
    state["negative_results"] = negative_results[-_DEFAULT_MAX_NEGATIVE_RESULTS:]

    registry = dict(state.get("program_registry", {}))
    _update_program_registry(
        registry,
        interesting_targets=interesting_targets,
        promising_cures=promising_cures,
        captured_at=now,
    )
    state["program_registry"] = registry

    state["schema_version"] = _STATE_SCHEMA_VERSION
    state["updated_at"] = now
    _write_state(path, state)

    return {
        "state_path": str(path),
        "run_id": run_id,
        "runs_tracked": len(state["runs"]),
        "failures_tracked": len(state["failures"]),
        "negative_results_tracked": len(state["negative_results"]),
        "programs_tracked": len(state["program_registry"]),
    }


def build_failure_intelligence(
    *,
    results: list[Mapping[str, Any]],
    promising_cures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = _extract_failures(results)
    reasons = Counter(item.get("error") or "unknown_error" for item in failures)
    top_reasons = [
        {"reason": reason, "count": count}
        for reason, count in reasons.most_common(5)
    ]

    rejected = [
        item
        for item in promising_cures
        if isinstance(item, Mapping) and not bool(item.get("promising"))
    ]
    return {
        "failed_tool_calls": len(failures),
        "top_failure_reasons": top_reasons,
        "negative_candidate_count": len(rejected),
        "negative_candidates": [
            {
                "cure_id": item.get("cure_id"),
                "name": item.get("name"),
                "target": item.get("target"),
                "score": item.get("score"),
                "assessment": item.get("assessment"),
            }
            for item in rejected[:10]
        ],
    }


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "updated_at": None,
        "runs": [],
        "failures": [],
        "negative_results": [],
        "program_registry": {},
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_run_id(*, objective: str, timestamp: str, plan: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "objective": objective,
            "timestamp": timestamp,
            "plan_calls": _plan_call_count(plan),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def _plan_call_count(plan: Mapping[str, Any]) -> int:
    calls = plan.get("calls")
    if not isinstance(calls, list):
        return 0
    return len(calls)


def _extract_failures(
    results: list[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    objective: str | None = None,
) -> list[dict[str, Any]]:
    captured_at = _utc_now()
    failures: list[dict[str, Any]] = []
    for item in results:
        tool = str(item.get("tool") or "")
        output = item.get("output")
        args = item.get("args")
        output_map = output if isinstance(output, Mapping) else {}
        error = str(output_map.get("error") or "").strip()
        invalid = output_map.get("valid") is False
        if not error and not invalid:
            continue
        if not error and invalid:
            error = "validation_failed"
        failures.append(
            {
                "captured_at": captured_at,
                "run_id": run_id,
                "objective": objective,
                "tool": tool or "unknown_tool",
                "error": error or "unknown_error",
                "args": dict(args) if isinstance(args, Mapping) else {},
            }
        )
    return failures


def _extract_negative_results(
    cures: list[Mapping[str, Any]],
    *,
    run_id: str,
    objective: str,
) -> list[dict[str, Any]]:
    captured_at = _utc_now()
    rows: list[dict[str, Any]] = []
    for item in cures:
        if bool(item.get("promising")):
            continue
        rows.append(
            {
                "captured_at": captured_at,
                "run_id": run_id,
                "objective": objective,
                "cure_id": item.get("cure_id"),
                "name": item.get("name"),
                "target": item.get("target"),
                "score": item.get("score"),
                "assessment": item.get("assessment"),
            }
        )
    return rows


def _update_program_registry(
    registry: dict[str, dict[str, Any]],
    *,
    interesting_targets: list[Mapping[str, Any]],
    promising_cures: list[Mapping[str, Any]],
    captured_at: str,
) -> None:
    for item in interesting_targets:
        target = str(item.get("target") or "").strip()
        if not target:
            continue
        disease = str(item.get("disease") or "unknown").strip().lower()
        key = f"target::{disease}::{target.upper()}"
        row = registry.get(
            key,
            {
                "kind": "target",
                "disease": disease,
                "target": target.upper(),
                "first_seen_at": captured_at,
                "last_seen_at": captured_at,
                "mentions": 0,
                "max_score": 0.0,
                "evidence_urls": [],
            },
        )
        row["last_seen_at"] = captured_at
        row["mentions"] = int(row.get("mentions", 0)) + int(item.get("mentions") or 0)
        row["max_score"] = max(float(row.get("max_score", 0.0)), float(item.get("score") or 0.0))
        urls = set(row.get("evidence_urls", []))
        for url in item.get("source_urls", []) if isinstance(item.get("source_urls"), list) else []:
            urls.add(str(url))
        row["evidence_urls"] = sorted(urls)[:50]
        registry[key] = row

    for item in promising_cures:
        cure_id = str(item.get("cure_id") or "").strip()
        if not cure_id:
            continue
        key = f"cure::{cure_id}"
        row = registry.get(
            key,
            {
                "kind": "cure_candidate",
                "cure_id": cure_id,
                "name": item.get("name"),
                "target": item.get("target"),
                "first_seen_at": captured_at,
                "last_seen_at": captured_at,
                "max_score": 0.0,
                "promising_runs": 0,
                "total_runs": 0,
            },
        )
        row["last_seen_at"] = captured_at
        row["max_score"] = max(float(row.get("max_score", 0.0)), float(item.get("score") or 0.0))
        row["total_runs"] = int(row.get("total_runs", 0)) + 1
        if bool(item.get("promising")):
            row["promising_runs"] = int(row.get("promising_runs", 0)) + 1
        registry[key] = row


def _count_promising(cures: list[Mapping[str, Any]]) -> int:
    count = 0
    for item in cures:
        if bool(item.get("promising")):
            count += 1
    return count


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
