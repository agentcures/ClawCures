from __future__ import annotations

from pathlib import Path

from refua_campaign.campaign_state import (
    build_failure_intelligence,
    load_campaign_state,
    persist_campaign_state,
)


def test_persist_campaign_state_tracks_runs_failures_and_registry(tmp_path: Path) -> None:
    state_path = tmp_path / "campaign_state.json"
    summary = persist_campaign_state(
        objective="Find cures for all diseases.",
        plan={"calls": [{"tool": "refua_validate_spec", "args": {"entities": []}}]},
        results=[
            {
                "tool": "web_fetch",
                "args": {"url": "https://example.org"},
                "output": {"error": "timeout"},
            }
        ],
        promising_cures=[
            {
                "cure_id": "c1",
                "name": "candidate-1",
                "target": "EGFR",
                "score": 62.5,
                "promising": False,
                "assessment": "toxicity liability",
            }
        ],
        interesting_targets=[
            {
                "target": "EGFR",
                "disease": "lung cancer",
                "score": 88.0,
                "mentions": 3,
                "source_urls": ["https://example.org/egfr"],
            }
        ],
        session_key="session-main",
        state_path=state_path,
    )

    assert summary["runs_tracked"] == 1
    assert summary["failures_tracked"] == 1
    assert summary["negative_results_tracked"] == 1
    assert summary["programs_tracked"] >= 2

    payload = load_campaign_state(state_path)
    assert len(payload["runs"]) == 1
    assert len(payload["failures"]) == 1
    assert len(payload["negative_results"]) == 1
    assert isinstance(payload["program_registry"], dict)


def test_build_failure_intelligence_summarizes_failures_and_negatives() -> None:
    summary = build_failure_intelligence(
        results=[
            {
                "tool": "web_fetch",
                "args": {"url": "https://example.org"},
                "output": {"error": "network"},
            },
            {
                "tool": "refua_validate_spec",
                "args": {},
                "output": {"valid": False},
            },
        ],
        promising_cures=[
            {
                "cure_id": "c1",
                "name": "risk-candidate",
                "target": "KRAS",
                "score": 45.0,
                "promising": False,
                "assessment": "high risk",
            }
        ],
    )
    assert summary["failed_tool_calls"] == 2
    assert summary["negative_candidate_count"] == 1
    assert summary["top_failure_reasons"]
