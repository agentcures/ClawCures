from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from refua_campaign.autonomy import (
    AutonomousPlanner,
    PlanPolicy,
    evaluate_plan_policy,
)
from refua_campaign.campaign_state import (
    build_failure_intelligence,
    default_campaign_state_path,
    load_campaign_state,
    persist_campaign_state,
)
from refua_campaign.clinical_trials import ClawCuresClinicalController
from refua_campaign.config import CampaignRunConfig, OpenClawConfig
from refua_campaign.evidence_quality import summarize_evidence_quality
from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.orchestrator import CampaignOrchestrator
from refua_campaign.portfolio import PortfolioWeights, rank_disease_programs
from refua_campaign.promising_cures import (
    extract_promising_cures,
    summarize_promising_cures,
)
from refua_campaign.prompts import load_system_prompt
from refua_campaign.regulatory_bridge import build_regulatory_bundle
from refua_campaign.refua_mcp_adapter import DEFAULT_TOOL_LIST, RefuaMcpAdapter
from refua_campaign.target_discovery import (
    extract_interesting_targets,
    summarize_interesting_targets,
)
from refua_campaign.translational_handoff import build_translational_handoff
from refua_campaign.web_evidence import expand_results_with_web_fetch

DEFAULT_OBJECTIVE = (
    "Find cures for all diseases by prioritizing the highest-burden conditions and "
    "researching the best drug design strategies for each."
)
_LOOP_MEMORY_WINDOW_CYCLES = 6
_LOOP_MEMORY_OBJECTIVE_CHAR_BUDGET = 8_000


class _StaticToolAdapter:
    def available_tools(self) -> list[str]:
        return list(DEFAULT_TOOL_LIST)

    def execute_plan(self, _plan: dict[str, object]) -> list[object]:
        raise RuntimeError(
            "Cannot execute plan because refua-mcp runtime dependencies are missing."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ClawCures",
        description=(
            "Campaign orchestration on top of OpenClaw planning and refua-mcp execution."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run",
        help="Run continuous plan+execute cycles (infinite by default).",
    )
    run_parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help=(
            "Campaign objective for the planner. Defaults to an all-disease cure "
            "mission focused on worst diseases and best drug-design strategies."
        ),
    )
    run_parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help=(
            "Number of run cycles before exit. 0 means loop forever (default). "
            "When --dry-run is set and --max-cycles is omitted, one cycle is used."
        ),
    )
    run_parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional override for the default campaign system prompt.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print plan without executing tools.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write run JSON output.",
    )
    run_parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="Optional JSON plan file. When set, OpenClaw planning is skipped.",
    )
    run_parser.add_argument(
        "--native-tool-loop",
        action="store_true",
        help=(
            "Use OpenClaw native function-calling loop instead of JSON plan parsing. "
            "Tools are executed turn-by-turn until a terminal assistant response."
        ),
    )
    run_parser.add_argument(
        "--native-tool-max-rounds",
        type=int,
        default=8,
        help="Maximum OpenClaw native function-calling rounds.",
    )
    run_parser.add_argument(
        "--session-key",
        default=None,
        help=(
            "Optional stable OpenClaw user/session key for memory continuity. "
            "Defaults to REFUA_CAMPAIGN_SESSION_KEY when set."
        ),
    )
    run_parser.add_argument(
        "--store-responses",
        action="store_true",
        help=(
            "Request OpenClaw response storage for cross-turn memory. "
            "Can also be set with REFUA_CAMPAIGN_STORE_RESPONSES."
        ),
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable OpenClaw streaming responses for planning/native loops.",
    )
    run_parser.add_argument(
        "--stream-to-stderr",
        action="store_true",
        help="When --stream is enabled, print streamed text deltas to stderr.",
    )
    run_parser.add_argument(
        "--native-discovery-bootstrap-rounds",
        type=int,
        default=0,
        help=(
            "During the first N native-tool rounds, constrain OpenClaw tools to "
            "web_search/web_fetch for target discovery."
        ),
    )
    run_parser.add_argument(
        "--native-tool-fail-fast",
        action="store_true",
        help=(
            "When set, stop immediately on native tool execution errors instead of "
            "returning recoverable tool-error payloads to the model."
        ),
    )
    run_parser.add_argument(
        "--disable-native-parallel-tool-calls",
        action="store_true",
        help=(
            "Disable OpenClaw parallel tool-call planning/execution in native loop mode."
        ),
    )
    run_parser.add_argument(
        "--native-tool-max-workers",
        type=int,
        default=4,
        help=(
            "Maximum worker threads when executing parallel-safe native tool calls."
        ),
    )
    run_parser.add_argument(
        "--auto-web-fetch",
        action="store_true",
        help=(
            "Automatically follow web_search result URLs with web_fetch to enrich "
            "target-discovery evidence."
        ),
    )
    run_parser.add_argument(
        "--auto-web-fetch-max-urls",
        type=int,
        default=6,
        help="Max auto-generated web_fetch calls from accumulated web_search results.",
    )
    run_parser.add_argument(
        "--auto-web-fetch-max-chars",
        type=int,
        default=20000,
        help="Max characters per auto-generated web_fetch call.",
    )
    run_parser.add_argument(
        "--agent-model-map-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping routing keys to OpenClaw models "
            "(e.g. planner:oncology, critic, default)."
        ),
    )
    run_parser.add_argument(
        "--agent-model-map-json",
        default=None,
        help="Optional JSON object string for agent model routing.",
    )
    run_parser.add_argument(
        "--evidence-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Path to literature/evidence text file to inject into OpenClaw input. "
            "May be specified multiple times."
        ),
    )
    run_parser.add_argument(
        "--evidence-max-chars",
        type=int,
        default=20000,
        help="Max characters to ingest per evidence file.",
    )
    run_parser.add_argument(
        "--policy-max-calls",
        type=int,
        default=24,
        help="Max call budget for optional run-policy checks.",
    )
    run_parser.add_argument(
        "--enforce-stage-policy",
        action="store_true",
        help="Enforce staged pipeline ordering checks on generated plans.",
    )
    run_parser.add_argument(
        "--require-evidence-before-hypothesis",
        action="store_true",
        help="Require evidence-gathering calls before design/admet/clinical calls.",
    )
    run_parser.add_argument(
        "--strict-plan-policy",
        action="store_true",
        help="Fail immediately when run-policy checks return errors.",
    )
    run_parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Optional campaign state JSON path for persistent memory/failure tracking."
        ),
    )
    run_parser.add_argument(
        "--disable-state-update",
        action="store_true",
        help="Disable persistent campaign state updates for this run.",
    )
    run_parser.add_argument(
        "--regulatory-bundle-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory to generate a refua-regulatory evidence bundle."
        ),
    )
    run_parser.set_defaults(handler=_cmd_run)

    loop_parser = sub.add_parser(
        "run-autonomous",
        help="Run planner/critic autonomous loop with policy checks.",
    )
    loop_parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help=(
            "Campaign objective for the planner. Defaults to an all-disease cure "
            "mission focused on worst diseases and best drug-design strategies."
        ),
    )
    loop_parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional override for the default campaign system prompt.",
    )
    loop_parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum planner/critic rounds.",
    )
    loop_parser.add_argument(
        "--max-calls",
        type=int,
        default=10,
        help="Maximum number of tool calls allowed in a plan.",
    )
    loop_parser.add_argument(
        "--allow-skip-validate-first",
        action="store_true",
        help="Disable policy warning that first call should be refua_validate_spec.",
    )
    loop_parser.add_argument(
        "--enforce-stage-policy",
        action="store_true",
        help="Enforce staged pipeline ordering checks in autonomous policy.",
    )
    loop_parser.add_argument(
        "--require-evidence-before-hypothesis",
        action="store_true",
        help="Require evidence-gathering calls before design/admet/clinical calls.",
    )
    loop_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute tools after approval; emit the final plan only.",
    )
    loop_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write run JSON output.",
    )
    loop_parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="Optional JSON plan file. When set, OpenClaw autonomous planning is skipped.",
    )
    loop_parser.add_argument(
        "--session-key",
        default=None,
        help=(
            "Optional stable OpenClaw user/session key for memory continuity. "
            "Defaults to REFUA_CAMPAIGN_SESSION_KEY when set."
        ),
    )
    loop_parser.add_argument(
        "--store-responses",
        action="store_true",
        help=(
            "Request OpenClaw response storage for cross-turn memory. "
            "Can also be set with REFUA_CAMPAIGN_STORE_RESPONSES."
        ),
    )
    loop_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable OpenClaw streaming responses for planner/critic loop.",
    )
    loop_parser.add_argument(
        "--auto-web-fetch",
        action="store_true",
        help=(
            "Automatically follow web_search result URLs with web_fetch after final "
            "plan execution."
        ),
    )
    loop_parser.add_argument(
        "--auto-web-fetch-max-urls",
        type=int,
        default=6,
        help="Max auto-generated web_fetch calls from accumulated web_search results.",
    )
    loop_parser.add_argument(
        "--auto-web-fetch-max-chars",
        type=int,
        default=20000,
        help="Max characters per auto-generated web_fetch call.",
    )
    loop_parser.add_argument(
        "--agent-model-map-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping routing keys to OpenClaw models "
            "(e.g. planner:oncology, critic, default)."
        ),
    )
    loop_parser.add_argument(
        "--agent-model-map-json",
        default=None,
        help="Optional JSON object string for agent model routing.",
    )
    loop_parser.add_argument(
        "--evidence-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Path to literature/evidence text file to inject into OpenClaw input. "
            "May be specified multiple times."
        ),
    )
    loop_parser.add_argument(
        "--evidence-max-chars",
        type=int,
        default=20000,
        help="Max characters to ingest per evidence file.",
    )
    loop_parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Optional campaign state JSON path for persistent memory/failure tracking."
        ),
    )
    loop_parser.add_argument(
        "--disable-state-update",
        action="store_true",
        help="Disable persistent campaign state updates for this run.",
    )
    loop_parser.add_argument(
        "--regulatory-bundle-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory to generate a refua-regulatory evidence bundle."
        ),
    )
    loop_parser.set_defaults(handler=_cmd_run_autonomous)

    prompt_parser = sub.add_parser(
        "print-default-prompt",
        help="Print the default campaign system prompt.",
    )
    prompt_parser.set_defaults(handler=_cmd_print_default_prompt)

    tools_parser = sub.add_parser(
        "list-tools",
        help="List supported execution tools (refua-mcp + web tools).",
    )
    tools_parser.set_defaults(handler=_cmd_list_tools)

    validate_parser = sub.add_parser(
        "validate-plan",
        help="Validate a JSON tool plan against autonomy policy.",
    )
    validate_parser.add_argument(
        "--plan-file",
        type=Path,
        required=True,
        help="Path to JSON plan file.",
    )
    validate_parser.add_argument(
        "--max-calls",
        type=int,
        default=10,
        help="Maximum number of calls allowed.",
    )
    validate_parser.add_argument(
        "--allow-skip-validate-first",
        action="store_true",
        help="Disable warning that first tool should be refua_validate_spec.",
    )
    validate_parser.add_argument(
        "--enforce-stage-policy",
        action="store_true",
        help="Enforce staged pipeline ordering checks.",
    )
    validate_parser.add_argument(
        "--require-evidence-before-hypothesis",
        action="store_true",
        help="Require evidence-gathering calls before design/admet/clinical calls.",
    )
    validate_parser.set_defaults(handler=_cmd_validate_plan)

    portfolio_parser = sub.add_parser(
        "rank-portfolio",
        help="Rank disease programs from a JSON list using weighted scoring.",
    )
    portfolio_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing a list of disease program objects.",
    )
    portfolio_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write ranking output JSON.",
    )
    portfolio_parser.add_argument("--w-burden", type=float, default=0.35)
    portfolio_parser.add_argument("--w-tractability", type=float, default=0.25)
    portfolio_parser.add_argument("--w-unmet-need", type=float, default=0.20)
    portfolio_parser.add_argument(
        "--w-translational-readiness", type=float, default=0.10
    )
    portfolio_parser.add_argument("--w-novelty", type=float, default=0.10)
    portfolio_parser.add_argument(
        "--total-budget",
        type=float,
        default=None,
        help="Optional total budget to allocate across ranked programs.",
    )
    portfolio_parser.add_argument(
        "--voi-weight",
        type=float,
        default=0.15,
        help="Weight multiplier for value-of-information signal when present.",
    )
    portfolio_parser.set_defaults(handler=_cmd_rank_portfolio)

    trial_list_parser = sub.add_parser(
        "trials-list",
        help="List managed clinical trials.",
    )
    trial_list_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_list_parser.set_defaults(handler=_cmd_trials_list)

    trial_get_parser = sub.add_parser(
        "trials-get",
        help="Get one managed clinical trial by id.",
    )
    trial_get_parser.add_argument("--trial-id", required=True)
    trial_get_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_get_parser.set_defaults(handler=_cmd_trials_get)

    trial_add_parser = sub.add_parser(
        "trials-add",
        help="Add a managed clinical trial.",
    )
    trial_add_parser.add_argument("--trial-id", default=None)
    trial_add_parser.add_argument("--config-file", type=Path, default=None)
    trial_add_parser.add_argument("--indication", default=None)
    trial_add_parser.add_argument("--phase", default=None)
    trial_add_parser.add_argument("--objective", default=None)
    trial_add_parser.add_argument("--status", default="planned")
    trial_add_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON object for trial metadata.",
    )
    trial_add_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_add_parser.set_defaults(handler=_cmd_trials_add)

    trial_update_parser = sub.add_parser(
        "trials-update",
        help="Apply partial updates to a managed trial.",
    )
    trial_update_parser.add_argument("--trial-id", required=True)
    trial_update_parser.add_argument(
        "--updates-json",
        required=True,
        help='JSON object patch, e.g. \'{"status":"active"}\'.',
    )
    trial_update_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_update_parser.set_defaults(handler=_cmd_trials_update)

    trial_remove_parser = sub.add_parser(
        "trials-remove",
        help="Remove a managed clinical trial.",
    )
    trial_remove_parser.add_argument("--trial-id", required=True)
    trial_remove_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_remove_parser.set_defaults(handler=_cmd_trials_remove)

    trial_enroll_parser = sub.add_parser(
        "trials-enroll",
        help="Enroll a patient (human or simulated) in a managed trial.",
    )
    trial_enroll_parser.add_argument("--trial-id", required=True)
    trial_enroll_parser.add_argument("--patient-id", default=None)
    trial_enroll_parser.add_argument("--source", default="human")
    trial_enroll_parser.add_argument("--arm-id", default=None)
    trial_enroll_parser.add_argument("--demographics-json", default=None)
    trial_enroll_parser.add_argument("--baseline-json", default=None)
    trial_enroll_parser.add_argument("--metadata-json", default=None)
    trial_enroll_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_enroll_parser.set_defaults(handler=_cmd_trials_enroll)

    trial_enroll_sim_parser = sub.add_parser(
        "trials-enroll-simulated",
        help="Enroll simulated patients in a managed trial.",
    )
    trial_enroll_sim_parser.add_argument("--trial-id", required=True)
    trial_enroll_sim_parser.add_argument("--count", type=int, required=True)
    trial_enroll_sim_parser.add_argument("--seed", type=int, default=None)
    trial_enroll_sim_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_enroll_sim_parser.set_defaults(handler=_cmd_trials_enroll_simulated)

    trial_result_parser = sub.add_parser(
        "trials-result",
        help="Add a patient result to a managed trial.",
    )
    trial_result_parser.add_argument("--trial-id", required=True)
    trial_result_parser.add_argument("--patient-id", required=True)
    trial_result_parser.add_argument(
        "--values-json",
        required=True,
        help="JSON object containing endpoint/result values.",
    )
    trial_result_parser.add_argument("--result-type", default="endpoint")
    trial_result_parser.add_argument("--visit", default=None)
    trial_result_parser.add_argument("--source", default=None)
    trial_result_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_result_parser.set_defaults(handler=_cmd_trials_result)

    trial_sim_parser = sub.add_parser(
        "trials-simulate",
        help="Run or refresh simulation for a managed trial.",
    )
    trial_sim_parser.add_argument("--trial-id", required=True)
    trial_sim_parser.add_argument("--replicates", type=int, default=None)
    trial_sim_parser.add_argument("--seed", type=int, default=None)
    trial_sim_parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional trial store file path override.",
    )
    trial_sim_parser.set_defaults(handler=_cmd_trials_simulate)

    return parser


def _cmd_print_default_prompt(_args: argparse.Namespace) -> int:
    print(load_system_prompt())
    return 0


def _cmd_list_tools(_args: argparse.Namespace) -> int:
    adapter, adapter_error = _build_adapter()
    if adapter_error is not None:
        names = list(DEFAULT_TOOL_LIST)
        print(f"warning: {adapter_error}", file=sys.stderr)
        print("warning: using static tool list fallback.", file=sys.stderr)
    else:
        names = adapter.available_tools()

    for name in names:
        print(name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    system_prompt = load_system_prompt(args.system_prompt_file)
    run_config = CampaignRunConfig(
        objective=args.objective,
        output_path=args.output,
        dry_run=bool(args.dry_run),
    )
    requested_max_cycles = int(args.max_cycles)
    if requested_max_cycles < 0:
        raise ValueError("--max-cycles must be >= 0.")
    # Keep dry-run workflows bounded unless user explicitly requests more cycles.
    effective_max_cycles = (
        1 if run_config.dry_run and requested_max_cycles == 0 else requested_max_cycles
    )

    adapter, adapter_error = _build_adapter()

    env_session_key = os.getenv("REFUA_CAMPAIGN_SESSION_KEY", "").strip() or None
    session_key = str(args.session_key).strip() if args.session_key else env_session_key
    if session_key == "":
        session_key = None

    store_from_env = _parse_optional_bool_env("REFUA_CAMPAIGN_STORE_RESPONSES")
    store_responses: bool | None
    if bool(args.store_responses):
        store_responses = True
    else:
        store_responses = store_from_env
    multi_cycle_mode = effective_max_cycles == 0 or effective_max_cycles > 1
    if multi_cycle_mode and session_key is None:
        session_key = _build_loop_session_key()
    if multi_cycle_mode and store_responses is None:
        store_responses = True
    agent_model_map = _load_agent_model_map(
        map_file=args.agent_model_map_file,
        map_json=args.agent_model_map_json,
    )
    evidence_items = _load_evidence_items(
        paths=list(args.evidence_file),
        max_chars=max(1, int(args.evidence_max_chars)),
    )

    openclaw = OpenClawClient(OpenClawConfig.from_env())
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw,
        refua_mcp=adapter,
        session_key=session_key,
        store_responses=store_responses,
        native_tool_max_rounds=max(1, int(args.native_tool_max_rounds)),
        agent_model_map=agent_model_map,
        stream_responses=bool(args.stream),
        stream_to_stderr=bool(args.stream_to_stderr),
        evidence_items=evidence_items,
        native_discovery_bootstrap_rounds=max(
            0, int(args.native_discovery_bootstrap_rounds)
        ),
        native_tool_fail_fast=bool(args.native_tool_fail_fast),
        native_parallel_tool_calls=not bool(args.disable_native_parallel_tool_calls),
        native_tool_max_workers=max(1, int(args.native_tool_max_workers)),
        auto_web_fetch=bool(args.auto_web_fetch),
        auto_web_fetch_max_urls=max(0, int(args.auto_web_fetch_max_urls)),
        auto_web_fetch_max_chars=max(1, int(args.auto_web_fetch_max_chars)),
    )

    cycle_index = 0
    loop_forever = effective_max_cycles == 0
    cycle_memory_notes: list[str] = []
    if multi_cycle_mode:
        state_snapshot = load_campaign_state(args.state_file or default_campaign_state_path())
        state_memory = _build_state_memory_note(state_snapshot)
        if state_memory:
            cycle_memory_notes.append(state_memory)
    try:
        while loop_forever or cycle_index < effective_max_cycles:
            cycle_index += 1
            cycle_objective = run_config.objective
            if multi_cycle_mode:
                cycle_objective = _compose_objective_with_cycle_memory(
                    base_objective=run_config.objective,
                    cycle_index=cycle_index,
                    memory_notes=cycle_memory_notes,
                )
            planner_text = ""
            plan_policy_payload: dict[str, Any] | None = None
            if bool(args.native_tool_loop):
                if args.plan_file is not None:
                    raise ValueError(
                        "--native-tool-loop cannot be used with --plan-file."
                    )
                if run_config.dry_run:
                    raise ValueError("--native-tool-loop cannot be used with --dry-run.")
                if adapter_error is not None:
                    raise RuntimeError(str(adapter_error))
                native_run = orchestrator.run_native_tool_loop(
                    objective=cycle_objective,
                    system_prompt=system_prompt,
                    max_rounds=max(1, int(args.native_tool_max_rounds)),
                )
                planner_text = native_run.planner_response_text
                plan = native_run.plan
                results = native_run.results
            elif args.plan_file is not None:
                plan_payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
                if not isinstance(plan_payload, dict):
                    raise ValueError("--plan-file must contain a JSON object.")
                plan = plan_payload
                planner_text = "Loaded from --plan-file"
                results = []
            else:
                planner_text, plan = orchestrator.plan(
                    objective=cycle_objective,
                    system_prompt=system_prompt,
                )
                results = []

            apply_policy = (
                bool(args.enforce_stage_policy)
                or bool(args.require_evidence_before_hypothesis)
                or bool(args.strict_plan_policy)
            )
            if apply_policy and not bool(args.native_tool_loop):
                plan_policy = PlanPolicy(
                    max_calls=max(1, int(args.policy_max_calls)),
                    require_validate_first=True,
                    enforce_stage_progression=bool(args.enforce_stage_policy),
                    require_evidence_before_hypothesis=bool(
                        args.require_evidence_before_hypothesis
                    ),
                )
                check = evaluate_plan_policy(
                    plan,
                    allowed_tools=adapter.available_tools(),
                    policy=plan_policy,
                )
                plan_policy_payload = {
                    "approved": bool(check.approved),
                    "errors": list(check.errors),
                    "warnings": list(check.warnings),
                    "config": {
                        "max_calls": int(plan_policy.max_calls),
                        "require_validate_first": bool(plan_policy.require_validate_first),
                        "enforce_stage_progression": bool(
                            plan_policy.enforce_stage_progression
                        ),
                        "require_evidence_before_hypothesis": bool(
                            plan_policy.require_evidence_before_hypothesis
                        ),
                    },
                }
                if bool(args.strict_plan_policy) and not check.approved:
                    raise ValueError(
                        "Run plan failed strict policy checks: "
                        + "; ".join(check.errors)
                    )

            if run_config.dry_run:
                payload = {
                    "objective": run_config.objective,
                    "system_prompt": system_prompt,
                    "planner_response_text": planner_text,
                    "plan": plan,
                    "dry_run": True,
                    "native_tool_loop": bool(args.native_tool_loop),
                }
                if agent_model_map:
                    payload["agent_model_map"] = agent_model_map
                if evidence_items:
                    payload["evidence_item_count"] = len(evidence_items)
                payload["stream"] = bool(args.stream)
                payload["auto_web_fetch"] = bool(args.auto_web_fetch)
                payload["native_parallel_tool_calls"] = not bool(
                    args.disable_native_parallel_tool_calls
                )
                payload["native_tool_max_workers"] = max(
                    1, int(args.native_tool_max_workers)
                )
                if plan_policy_payload is not None:
                    payload["plan_policy"] = plan_policy_payload
                if adapter_error is not None:
                    payload["warnings"] = [str(adapter_error)]
            else:
                if adapter_error is not None and not bool(args.native_tool_loop):
                    raise RuntimeError(str(adapter_error))
                if not bool(args.native_tool_loop):
                    results = orchestrator.execute_plan(plan)
                serialized_results = [
                    {
                        "tool": item.tool,
                        "args": item.args,
                        "output": item.output,
                    }
                    for item in results
                ]
                promising_cures = extract_promising_cures(serialized_results)
                interesting_targets = extract_interesting_targets(serialized_results)
                evidence_quality = summarize_evidence_quality(
                    results=serialized_results,
                    interesting_targets=interesting_targets,
                    promising_cures=promising_cures,
                )
                failure_intelligence = build_failure_intelligence(
                    results=serialized_results,
                    promising_cures=promising_cures,
                )
                translational_handoff = build_translational_handoff(
                    objective=run_config.objective,
                    interesting_targets=interesting_targets,
                    promising_cures=promising_cures,
                    evidence_quality=evidence_quality,
                )
                payload = {
                    "objective": run_config.objective,
                    "system_prompt": system_prompt,
                    "planner_response_text": planner_text,
                    "plan": plan,
                    "results": serialized_results,
                    "promising_cures": promising_cures,
                    "promising_cures_summary": summarize_promising_cures(promising_cures),
                    "interesting_targets": interesting_targets,
                    "interesting_targets_summary": summarize_interesting_targets(
                        interesting_targets
                    ),
                    "evidence_quality_summary": evidence_quality,
                    "failure_intelligence": failure_intelligence,
                    "translational_handoff": translational_handoff,
                    "dry_run": False,
                    "native_tool_loop": bool(args.native_tool_loop),
                    "stream": bool(args.stream),
                    "auto_web_fetch": bool(args.auto_web_fetch),
                    "native_parallel_tool_calls": not bool(
                        args.disable_native_parallel_tool_calls
                    ),
                    "native_tool_max_workers": max(
                        1, int(args.native_tool_max_workers)
                    ),
                }
                if plan_policy_payload is not None:
                    payload["plan_policy"] = plan_policy_payload
                if agent_model_map:
                    payload["agent_model_map"] = agent_model_map
                if evidence_items:
                    payload["evidence_item_count"] = len(evidence_items)

            if session_key is not None:
                payload["session_key"] = session_key
            if store_responses is not None:
                payload["store_responses"] = bool(store_responses)
            if multi_cycle_mode:
                payload["cycle_memory_enabled"] = True
                payload["cycle_objective_augmented"] = (
                    cycle_objective != run_config.objective
                )
                payload["memory_notes_used"] = len(cycle_memory_notes)

            if not bool(run_config.dry_run) and not bool(args.disable_state_update):
                state_file = args.state_file or default_campaign_state_path()
                try:
                    payload["campaign_state"] = persist_campaign_state(
                        objective=run_config.objective,
                        plan=plan,
                        results=[
                            item
                            for item in payload.get("results", [])
                            if isinstance(item, dict)
                        ],
                        promising_cures=[
                            item
                            for item in payload.get("promising_cures", [])
                            if isinstance(item, dict)
                        ],
                        interesting_targets=[
                            item
                            for item in payload.get("interesting_targets", [])
                            if isinstance(item, dict)
                        ],
                        session_key=session_key,
                        state_path=state_file,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload.setdefault("warnings", []).append(
                        f"Campaign state update failed: {exc}"
                    )

            if not bool(run_config.dry_run) and args.regulatory_bundle_dir is not None:
                try:
                    payload["regulatory_bundle"] = build_regulatory_bundle(
                        payload=payload,
                        bundle_dir=args.regulatory_bundle_dir,
                        campaign_run_path=None,
                        overwrite=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload.setdefault("warnings", []).append(
                        f"Regulatory bundle generation failed: {exc}"
                    )

            if multi_cycle_mode:
                cycle_memory_note = _build_cycle_memory_note(
                    payload=payload,
                    cycle_index=cycle_index,
                )
                if cycle_memory_note:
                    payload["cycle_memory_note"] = cycle_memory_note
                    cycle_memory_notes = _append_cycle_memory_note(
                        cycle_memory_notes,
                        cycle_memory_note,
                        max_notes=_LOOP_MEMORY_WINDOW_CYCLES,
                    )
                payload["memory_notes_stored"] = len(cycle_memory_notes)

            payload["cycle_index"] = cycle_index
            payload["max_cycles"] = int(effective_max_cycles)
            payload["loop_forever"] = bool(loop_forever)

            rendered = json.dumps(payload, indent=2)
            print(rendered)

            if run_config.output_path is not None:
                run_config.output_path.parent.mkdir(parents=True, exist_ok=True)
                run_config.output_path.write_text(rendered + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        print("Interrupted; stopping continuous run loop.", file=sys.stderr)
        return 130

    return 0


def _cmd_run_autonomous(args: argparse.Namespace) -> int:
    system_prompt = load_system_prompt(args.system_prompt_file)
    adapter, adapter_error = _build_adapter()
    env_session_key = os.getenv("REFUA_CAMPAIGN_SESSION_KEY", "").strip() or None
    session_key = str(args.session_key).strip() if args.session_key else env_session_key
    if session_key == "":
        session_key = None
    store_from_env = _parse_optional_bool_env("REFUA_CAMPAIGN_STORE_RESPONSES")
    store_responses = True if bool(args.store_responses) else store_from_env
    agent_model_map = _load_agent_model_map(
        map_file=args.agent_model_map_file,
        map_json=args.agent_model_map_json,
    )
    evidence_items = _load_evidence_items(
        paths=list(args.evidence_file),
        max_chars=max(1, int(args.evidence_max_chars)),
    )
    policy = PlanPolicy(
        max_calls=max(1, int(args.max_calls)),
        require_validate_first=not bool(args.allow_skip_validate_first),
        enforce_stage_progression=bool(args.enforce_stage_policy),
        require_evidence_before_hypothesis=bool(
            args.require_evidence_before_hypothesis
        ),
    )

    if args.plan_file is not None:
        plan_payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
        if not isinstance(plan_payload, dict):
            raise ValueError("--plan-file must contain a JSON object.")
        policy_check = evaluate_plan_policy(
            plan_payload,
            allowed_tools=adapter.available_tools(),
            policy=policy,
        )
        plan_result_payload = {
            "objective": str(args.objective),
            "system_prompt": system_prompt,
            "approved": bool(policy_check.approved),
            "iterations": [
                {
                    "round_index": 1,
                    "planner_text": "Loaded from --plan-file",
                    "plan": plan_payload,
                    "policy": {
                        "approved": policy_check.approved,
                        "errors": list(policy_check.errors),
                        "warnings": list(policy_check.warnings),
                    },
                    "critic_text": "Skipped (offline plan file mode).",
                    "critic": {"approved": policy_check.approved},
                }
            ],
            "final_plan": plan_payload,
        }
    else:
        openclaw = OpenClawClient(OpenClawConfig.from_env())
        planner = AutonomousPlanner(
            openclaw=openclaw,
            available_tools=adapter.available_tools(),
            policy=policy,
            session_key=session_key,
            store_responses=store_responses,
            agent_model_map=agent_model_map,
            stream_responses=bool(args.stream),
            evidence_items=evidence_items,
        )
        plan_result = planner.run(
            objective=str(args.objective),
            system_prompt=system_prompt,
            max_rounds=max(1, int(args.max_rounds)),
        )
        plan_result_payload = plan_result.to_json()

    payload = dict(plan_result_payload)
    payload["dry_run"] = bool(args.dry_run)
    payload["stream"] = bool(args.stream)
    payload["auto_web_fetch"] = bool(args.auto_web_fetch)
    if session_key is not None:
        payload["session_key"] = session_key
    if store_responses is not None:
        payload["store_responses"] = bool(store_responses)
    if agent_model_map:
        payload["agent_model_map"] = agent_model_map
    if evidence_items:
        payload["evidence_item_count"] = len(evidence_items)
    if adapter_error is not None:
        payload.setdefault("warnings", []).append(str(adapter_error))

    if bool(payload.get("approved")) and not bool(args.dry_run):
        if adapter_error is not None:
            raise RuntimeError(str(adapter_error))
        final_plan = payload.get("final_plan")
        if not isinstance(final_plan, dict):
            raise ValueError("Final plan is missing from autonomous payload.")
        results = adapter.execute_plan(final_plan)
        if bool(args.auto_web_fetch):
            results, generated_auto_fetch = expand_results_with_web_fetch(
                results=results,
                execute_tool=adapter.execute_tool,  # type: ignore[attr-defined]
                max_urls=max(0, int(args.auto_web_fetch_max_urls)),
                max_chars=max(1, int(args.auto_web_fetch_max_chars)),
            )
            payload["auto_web_fetch_generated"] = generated_auto_fetch
        serialized_results = [
            {
                "tool": item.tool,
                "args": item.args,
                "output": item.output,
            }
            for item in results
        ]
        promising_cures = extract_promising_cures(serialized_results)
        interesting_targets = extract_interesting_targets(serialized_results)
        evidence_quality = summarize_evidence_quality(
            results=serialized_results,
            interesting_targets=interesting_targets,
            promising_cures=promising_cures,
        )
        failure_intelligence = build_failure_intelligence(
            results=serialized_results,
            promising_cures=promising_cures,
        )
        translational_handoff = build_translational_handoff(
            objective=str(args.objective),
            interesting_targets=interesting_targets,
            promising_cures=promising_cures,
            evidence_quality=evidence_quality,
        )
        payload["results"] = serialized_results
        payload["promising_cures"] = promising_cures
        payload["promising_cures_summary"] = summarize_promising_cures(promising_cures)
        payload["interesting_targets"] = interesting_targets
        payload["interesting_targets_summary"] = summarize_interesting_targets(
            interesting_targets
        )
        payload["evidence_quality_summary"] = evidence_quality
        payload["failure_intelligence"] = failure_intelligence
        payload["translational_handoff"] = translational_handoff
    elif not bool(payload.get("approved")):
        payload.setdefault("warnings", []).append(
            "Autonomous loop finished without an approved plan."
        )

    if bool(payload.get("approved")) and not bool(args.dry_run) and not bool(
        args.disable_state_update
    ):
        state_file = args.state_file or default_campaign_state_path()
        final_plan_payload = payload.get("final_plan")
        try:
            payload["campaign_state"] = persist_campaign_state(
                objective=str(args.objective),
                plan=final_plan_payload if isinstance(final_plan_payload, dict) else {},
                results=[
                    item
                    for item in payload.get("results", [])
                    if isinstance(item, dict)
                ],
                promising_cures=[
                    item
                    for item in payload.get("promising_cures", [])
                    if isinstance(item, dict)
                ],
                interesting_targets=[
                    item
                    for item in payload.get("interesting_targets", [])
                    if isinstance(item, dict)
                ],
                session_key=session_key,
                state_path=state_file,
            )
        except Exception as exc:  # noqa: BLE001
            payload.setdefault("warnings", []).append(
                f"Campaign state update failed: {exc}"
            )

    if bool(payload.get("approved")) and not bool(args.dry_run) and args.regulatory_bundle_dir is not None:
        try:
            payload["regulatory_bundle"] = build_regulatory_bundle(
                payload=payload,
                bundle_dir=args.regulatory_bundle_dir,
                campaign_run_path=None,
                overwrite=True,
            )
        except Exception as exc:  # noqa: BLE001
            payload.setdefault("warnings", []).append(
                f"Regulatory bundle generation failed: {exc}"
            )

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _cmd_validate_plan(args: argparse.Namespace) -> int:
    plan_payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
    if not isinstance(plan_payload, dict):
        raise ValueError("--plan-file must contain a JSON object.")
    adapter, adapter_error = _build_adapter()

    policy = PlanPolicy(
        max_calls=max(1, int(args.max_calls)),
        require_validate_first=not bool(args.allow_skip_validate_first),
        enforce_stage_progression=bool(args.enforce_stage_policy),
        require_evidence_before_hypothesis=bool(
            args.require_evidence_before_hypothesis
        ),
    )
    check = evaluate_plan_policy(
        plan_payload,
        allowed_tools=adapter.available_tools(),
        policy=policy,
    )
    payload: dict[str, object] = {
        "approved": check.approved,
        "errors": list(check.errors),
        "warnings": list(check.warnings),
    }
    if adapter_error is not None:
        payload.setdefault("warnings", []).append(str(adapter_error))
    print(json.dumps(payload, indent=2))
    return 0


def _build_loop_session_key() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%SZ")
    return f"clawcures-loop-{stamp}-{uuid.uuid4().hex[:8]}"


def _build_state_memory_note(state: dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return ""

    lines: list[str] = []
    runs_raw = state.get("runs")
    runs = [item for item in runs_raw if isinstance(item, dict)] if isinstance(runs_raw, list) else []
    if runs:
        recent = runs[-1]
        lines.append(
            "Historical state: tracked "
            f"{len(runs)} runs; most recent had "
            f"{_as_int(recent.get('plan_calls'))} planned calls, "
            f"{_as_int(recent.get('promising_count'))} promising candidates, and "
            f"{_as_int(recent.get('interesting_target_count'))} interesting targets."
        )

    failures_raw = state.get("failures")
    failures = [item for item in failures_raw if isinstance(item, dict)] if isinstance(failures_raw, list) else []
    if failures:
        reason_counts = Counter(
            str(item.get("error") or "unknown_error")
            for item in failures[-200:]
        )
        top_reasons = ", ".join(
            f"{reason} ({count})"
            for reason, count in reason_counts.most_common(3)
        )
        if top_reasons:
            lines.append(f"Common prior tool failures: {top_reasons}.")

    registry = state.get("program_registry")
    if isinstance(registry, dict) and registry:
        target_rows: list[tuple[int, str]] = []
        cure_rows: list[tuple[int, int, str]] = []
        for entry in registry.values():
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind") or "").strip().lower()
            if kind == "target":
                target = str(entry.get("target") or "").strip()
                if target:
                    target_rows.append((_as_int(entry.get("mentions")), target))
            elif kind == "cure_candidate":
                name = str(entry.get("name") or entry.get("cure_id") or "").strip()
                if name:
                    cure_rows.append(
                        (
                            _as_int(entry.get("promising_runs")),
                            _as_int(entry.get("total_runs")),
                            name,
                        )
                    )

        if target_rows:
            target_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            top_targets = ", ".join(
                f"{name} ({mentions})"
                for mentions, name in target_rows[:5]
            )
            lines.append(f"Top retained targets: {top_targets}.")
        if cure_rows:
            cure_rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            top_cures = ", ".join(
                f"{name} ({promising}/{total})"
                for promising, total, name in cure_rows[:5]
            )
            lines.append(
                "Most repeated candidates (promising/total runs): "
                f"{top_cures}."
            )

    note = " ".join(lines).strip()
    return note[:1200]


def _compose_objective_with_cycle_memory(
    *,
    base_objective: str,
    cycle_index: int,
    memory_notes: list[str],
) -> str:
    trimmed_notes = [
        note.strip()
        for note in memory_notes[-_LOOP_MEMORY_WINDOW_CYCLES :]
        if str(note).strip()
    ]
    if not trimmed_notes:
        return base_objective

    guidance = (
        "Use this memory to avoid repeating completed steps, carry forward validated "
        "signals, and prioritize unresolved evidence gaps."
    )

    def _render(notes: list[str]) -> str:
        note_block = "\n".join(f"- {note}" for note in notes)
        return (
            f"{base_objective}\n\n"
            f"Cross-cycle memory for cycle {cycle_index}:\n"
            f"{note_block}\n\n"
            f"{guidance}"
        )

    rendered = _render(trimmed_notes)
    while (
        len(rendered) > _LOOP_MEMORY_OBJECTIVE_CHAR_BUDGET
        and len(trimmed_notes) > 1
    ):
        trimmed_notes = trimmed_notes[1:]
        rendered = _render(trimmed_notes)

    return rendered[:_LOOP_MEMORY_OBJECTIVE_CHAR_BUDGET]


def _build_cycle_memory_note(*, payload: dict[str, Any], cycle_index: int) -> str:
    parts: list[str] = [f"cycle {cycle_index}"]

    plan = payload.get("plan")
    if isinstance(plan, dict):
        calls = plan.get("calls")
        if isinstance(calls, list):
            parts.append(f"plan_calls={len(calls)}")

    if bool(payload.get("dry_run")):
        parts.append("dry_run")
    else:
        results = payload.get("results")
        if isinstance(results, list):
            parts.append(f"results={len(results)}")

    promising_summary = payload.get("promising_cures_summary")
    if isinstance(promising_summary, dict):
        parts.append(
            "promising="
            f"{_as_int(promising_summary.get('promising_count'))}/"
            f"{_as_int(promising_summary.get('total_candidates'))}"
        )

    target_summary = payload.get("interesting_targets_summary")
    if isinstance(target_summary, dict):
        total_targets = _as_int(target_summary.get("total_targets"))
        top_targets_raw = target_summary.get("top_targets")
        top_targets = (
            [str(item).strip() for item in top_targets_raw if str(item).strip()]
            if isinstance(top_targets_raw, list)
            else []
        )
        if top_targets:
            parts.append(
                f"targets={total_targets} top={','.join(top_targets[:3])}"
            )
        else:
            parts.append(f"targets={total_targets}")

    failures = payload.get("failure_intelligence")
    if isinstance(failures, dict):
        failed_calls = _as_int(failures.get("failed_tool_calls"))
        if failed_calls > 0:
            parts.append(f"failed_tool_calls={failed_calls}")

    planner_text = str(payload.get("planner_response_text") or "")
    if "Planner fallback plan was used" in planner_text:
        parts.append("planner_fallback_used")

    warnings_raw = payload.get("warnings")
    warnings = (
        [str(item).strip() for item in warnings_raw if str(item).strip()]
        if isinstance(warnings_raw, list)
        else []
    )
    if warnings:
        parts.append(f"warning={warnings[0][:80]}")

    return "; ".join(parts)[:900]


def _append_cycle_memory_note(
    memory_notes: list[str],
    note: str,
    *,
    max_notes: int,
) -> list[str]:
    clean_note = note.strip()
    if not clean_note:
        return memory_notes[-max(1, int(max_notes)) :]
    if memory_notes and memory_notes[-1] == clean_note:
        return memory_notes[-max(1, int(max_notes)) :]
    updated = [*memory_notes, clean_note]
    return updated[-max(1, int(max_notes)) :]


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_optional_bool_env(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value when set.")


def _load_agent_model_map(
    *,
    map_file: Path | None,
    map_json: str | None,
) -> dict[str, str]:
    payload: object = {}
    if map_file is not None:
        payload = json.loads(map_file.read_text(encoding="utf-8"))
    elif map_json:
        payload = json.loads(map_json)
    else:
        env_json = os.getenv("REFUA_CAMPAIGN_AGENT_MODEL_MAP_JSON", "").strip()
        if env_json:
            payload = json.loads(env_json)

    if not isinstance(payload, dict):
        if payload:
            raise ValueError("Agent model map must be a JSON object.")
        return {}
    resolved: dict[str, str] = {}
    for key, value in payload.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text or not value_text:
            continue
        resolved[key_text] = value_text
    return resolved


def _load_evidence_items(*, paths: list[Path], max_chars: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        clipped = text[:max_chars]
        items.append(
            {
                "type": "input_text",
                "text": (
                    f"[Evidence File: {path}]\n"
                    f"{clipped}"
                ),
            }
        )
    return items


def _cmd_rank_portfolio(args: argparse.Namespace) -> int:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--input must contain a JSON list of disease programs.")

    weights = PortfolioWeights(
        burden=float(args.w_burden),
        tractability=float(args.w_tractability),
        unmet_need=float(args.w_unmet_need),
        translational_readiness=float(args.w_translational_readiness),
        novelty=float(args.w_novelty),
    )
    ranked = rank_disease_programs(
        payload,
        weights=weights,
        total_budget=float(args.total_budget)
        if args.total_budget is not None
        else None,
        voi_weight=float(args.voi_weight),
    )
    rendered_payload = {
        "weights": {
            "burden": weights.burden,
            "tractability": weights.tractability,
            "unmet_need": weights.unmet_need,
            "translational_readiness": weights.translational_readiness,
            "novelty": weights.novelty,
        },
        "portfolio_constraints": {
            "total_budget": float(args.total_budget)
            if args.total_budget is not None
            else None,
            "voi_weight": float(args.voi_weight),
        },
        "ranked": [item.to_json() for item in ranked],
    }
    rendered = json.dumps(rendered_payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _cmd_trials_list(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    print(json.dumps(controller.list_trials(), indent=2))
    return 0


def _cmd_trials_get(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    print(json.dumps(controller.get_trial(str(args.trial_id)), indent=2))
    return 0


def _cmd_trials_add(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    config_payload = None
    if args.config_file is not None:
        config_payload = _load_mapping_file(args.config_file)
    metadata_payload = _parse_optional_json_object(
        args.metadata_json, flag="--metadata-json"
    )

    payload = controller.add_trial(
        trial_id=str(args.trial_id) if args.trial_id else None,
        config=config_payload,
        indication=str(args.indication) if args.indication else None,
        phase=str(args.phase) if args.phase else None,
        objective=str(args.objective) if args.objective else None,
        status=str(args.status) if args.status else None,
        metadata=metadata_payload,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_update(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    updates = _parse_required_json_object(args.updates_json, flag="--updates-json")
    payload = controller.update_trial(str(args.trial_id), updates=updates)
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_remove(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    payload = controller.remove_trial(str(args.trial_id))
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_enroll(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    payload = controller.enroll_patient(
        str(args.trial_id),
        patient_id=str(args.patient_id) if args.patient_id else None,
        source=str(args.source) if args.source else None,
        arm_id=str(args.arm_id) if args.arm_id else None,
        demographics=_parse_optional_json_object(
            args.demographics_json, flag="--demographics-json"
        ),
        baseline=_parse_optional_json_object(
            args.baseline_json, flag="--baseline-json"
        ),
        metadata=_parse_optional_json_object(
            args.metadata_json, flag="--metadata-json"
        ),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_enroll_simulated(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    payload = controller.enroll_simulated_patients(
        str(args.trial_id),
        count=max(1, int(args.count)),
        seed=int(args.seed) if args.seed is not None else None,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_result(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    values = _parse_required_json_object(args.values_json, flag="--values-json")
    payload = controller.add_result(
        str(args.trial_id),
        patient_id=str(args.patient_id),
        values=values,
        result_type=str(args.result_type) if args.result_type else "endpoint",
        visit=str(args.visit) if args.visit else None,
        source=str(args.source) if args.source else None,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_trials_simulate(args: argparse.Namespace) -> int:
    controller = _clinical_controller(args.store)
    payload = controller.simulate_trial(
        str(args.trial_id),
        replicates=int(args.replicates) if args.replicates is not None else None,
        seed=int(args.seed) if args.seed is not None else None,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _clinical_controller(store_path: Path | None) -> ClawCuresClinicalController:
    return ClawCuresClinicalController(store_path=store_path)


def _parse_optional_json_object(
    value: str | None, *, flag: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must be a JSON object.")
    return parsed


def _parse_required_json_object(value: str, *, flag: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must be a JSON object.")
    return parsed


def _load_mapping_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--config-file must contain a JSON object.")
    return payload


def _build_adapter() -> (
    tuple[RefuaMcpAdapter | _StaticToolAdapter, RuntimeError | None]
):
    try:
        return RefuaMcpAdapter(), None
    except RuntimeError as exc:
        return _StaticToolAdapter(), exc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
