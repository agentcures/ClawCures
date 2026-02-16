from __future__ import annotations

from refua_campaign.cli import DEFAULT_OBJECTIVE, build_parser


def test_run_defaults_to_all_disease_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.objective == DEFAULT_OBJECTIVE


def test_run_autonomous_defaults_to_all_disease_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-autonomous", "--dry-run"])
    assert args.objective == DEFAULT_OBJECTIVE
