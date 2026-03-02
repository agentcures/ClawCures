from __future__ import annotations

from refua_campaign.cli import (
    DEFAULT_OBJECTIVE,
    _append_cycle_memory_note,
    _build_state_memory_note,
    _compose_objective_with_cycle_memory,
    build_parser,
)
from refua_campaign.refua_mcp_adapter import DEFAULT_TOOL_LIST


def test_run_defaults_to_all_disease_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.objective == DEFAULT_OBJECTIVE
    assert args.max_cycles == 0


def test_run_autonomous_defaults_to_all_disease_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-autonomous", "--dry-run"])
    assert args.objective == DEFAULT_OBJECTIVE


def test_default_tool_list_includes_protein_properties() -> None:
    assert "refua_protein_properties" in DEFAULT_TOOL_LIST
    assert "refua_clinical_simulator" in DEFAULT_TOOL_LIST
    assert "refua_data_list" in DEFAULT_TOOL_LIST
    assert "refua_data_query" in DEFAULT_TOOL_LIST
    assert "web_search" in DEFAULT_TOOL_LIST
    assert "web_fetch" in DEFAULT_TOOL_LIST


def test_run_parser_accepts_native_tool_loop_and_session_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--max-cycles",
            "5",
            "--native-tool-loop",
            "--native-tool-max-rounds",
            "12",
            "--session-key",
            "campaign-main",
            "--store-responses",
            "--stream",
            "--stream-to-stderr",
            "--native-discovery-bootstrap-rounds",
            "2",
            "--native-tool-fail-fast",
            "--disable-native-parallel-tool-calls",
            "--native-tool-max-workers",
            "6",
            "--auto-web-fetch",
            "--auto-web-fetch-max-urls",
            "9",
            "--auto-web-fetch-max-chars",
            "7777",
            "--policy-max-calls",
            "18",
            "--enforce-stage-policy",
            "--require-evidence-before-hypothesis",
            "--strict-plan-policy",
            "--state-file",
            "artifacts/campaign_state.json",
            "--regulatory-bundle-dir",
            "artifacts/bundle",
            "--agent-model-map-json",
            '{"planner":"openclaw:planner"}',
            "--evidence-file",
            "docs/RESEARCH.md",
            "--evidence-max-chars",
            "1234",
        ]
    )
    assert args.max_cycles == 5
    assert args.native_tool_loop is True
    assert args.native_tool_max_rounds == 12
    assert args.session_key == "campaign-main"
    assert args.store_responses is True
    assert args.stream is True
    assert args.stream_to_stderr is True
    assert args.native_discovery_bootstrap_rounds == 2
    assert args.native_tool_fail_fast is True
    assert args.disable_native_parallel_tool_calls is True
    assert args.native_tool_max_workers == 6
    assert args.auto_web_fetch is True
    assert args.auto_web_fetch_max_urls == 9
    assert args.auto_web_fetch_max_chars == 7777
    assert args.policy_max_calls == 18
    assert args.enforce_stage_policy is True
    assert args.require_evidence_before_hypothesis is True
    assert args.strict_plan_policy is True
    assert str(args.state_file).endswith("campaign_state.json")
    assert str(args.regulatory_bundle_dir).endswith("artifacts/bundle")
    assert args.agent_model_map_json == '{"planner":"openclaw:planner"}'
    assert len(args.evidence_file) == 1
    assert args.evidence_max_chars == 1234


def test_run_autonomous_parser_accepts_session_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-autonomous",
            "--session-key",
            "campaign-main",
            "--store-responses",
            "--stream",
            "--auto-web-fetch",
            "--auto-web-fetch-max-urls",
            "8",
            "--auto-web-fetch-max-chars",
            "4567",
            "--enforce-stage-policy",
            "--require-evidence-before-hypothesis",
            "--state-file",
            "artifacts/campaign_state.json",
            "--regulatory-bundle-dir",
            "artifacts/bundle",
            "--agent-model-map-json",
            '{"critic":"openclaw:critic"}',
            "--evidence-file",
            "docs/ARCHITECTURE.md",
        ]
    )
    assert args.session_key == "campaign-main"
    assert args.store_responses is True
    assert args.stream is True
    assert args.auto_web_fetch is True
    assert args.auto_web_fetch_max_urls == 8
    assert args.auto_web_fetch_max_chars == 4567
    assert args.enforce_stage_policy is True
    assert args.require_evidence_before_hypothesis is True
    assert str(args.state_file).endswith("campaign_state.json")
    assert str(args.regulatory_bundle_dir).endswith("artifacts/bundle")
    assert args.agent_model_map_json == '{"critic":"openclaw:critic"}'
    assert len(args.evidence_file) == 1


def test_compose_objective_with_cycle_memory_includes_notes() -> None:
    objective = _compose_objective_with_cycle_memory(
        base_objective="Find cures",
        cycle_index=3,
        memory_notes=["alpha signal", "beta failure"],
    )
    assert "Find cures" in objective
    assert "cycle 3" in objective
    assert "alpha signal" in objective
    assert "beta failure" in objective


def test_append_cycle_memory_note_deduplicates_and_trims() -> None:
    notes = ["one", "two"]
    updated = _append_cycle_memory_note(notes, "two", max_notes=2)
    assert updated == ["one", "two"]
    updated = _append_cycle_memory_note(updated, "three", max_notes=2)
    assert updated == ["two", "three"]


def test_build_state_memory_note_includes_registry_and_failures() -> None:
    note = _build_state_memory_note(
        {
            "runs": [
                {
                    "plan_calls": 9,
                    "promising_count": 2,
                    "interesting_target_count": 5,
                }
            ],
            "failures": [
                {"error": "timeout"},
                {"error": "timeout"},
                {"error": "rate_limit"},
            ],
            "program_registry": {
                "target::x": {"kind": "target", "target": "EGFR", "mentions": 7},
                "cure::x": {
                    "kind": "cure_candidate",
                    "name": "compound-a",
                    "promising_runs": 2,
                    "total_runs": 4,
                },
            },
        }
    )
    assert "tracked 1 runs" in note
    assert "timeout (2)" in note
    assert "EGFR (7)" in note
    assert "compound-a (2/4)" in note
