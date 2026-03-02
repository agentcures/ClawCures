from __future__ import annotations

from refua_campaign.cli import DEFAULT_OBJECTIVE, build_parser
from refua_campaign.refua_mcp_adapter import DEFAULT_TOOL_LIST


def test_run_defaults_to_all_disease_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.objective == DEFAULT_OBJECTIVE


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
            "--auto-web-fetch",
            "--auto-web-fetch-max-urls",
            "9",
            "--auto-web-fetch-max-chars",
            "7777",
            "--agent-model-map-json",
            '{"planner":"openclaw:planner"}',
            "--evidence-file",
            "docs/RESEARCH.md",
            "--evidence-max-chars",
            "1234",
        ]
    )
    assert args.native_tool_loop is True
    assert args.native_tool_max_rounds == 12
    assert args.session_key == "campaign-main"
    assert args.store_responses is True
    assert args.stream is True
    assert args.stream_to_stderr is True
    assert args.native_discovery_bootstrap_rounds == 2
    assert args.native_tool_fail_fast is True
    assert args.auto_web_fetch is True
    assert args.auto_web_fetch_max_urls == 9
    assert args.auto_web_fetch_max_chars == 7777
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
    assert args.agent_model_map_json == '{"critic":"openclaw:critic"}'
    assert len(args.evidence_file) == 1
