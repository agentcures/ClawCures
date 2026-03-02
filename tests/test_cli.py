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
        ]
    )
    assert args.native_tool_loop is True
    assert args.native_tool_max_rounds == 12
    assert args.session_key == "campaign-main"
    assert args.store_responses is True


def test_run_autonomous_parser_accepts_session_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-autonomous",
            "--session-key",
            "campaign-main",
            "--store-responses",
        ]
    )
    assert args.session_key == "campaign-main"
    assert args.store_responses is True
