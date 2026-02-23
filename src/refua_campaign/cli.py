from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refua_campaign.autonomy import (
    AutonomousPlanner,
    PlanPolicy,
    evaluate_plan_policy,
)
from refua_campaign.config import CampaignRunConfig, OpenClawConfig
from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.orchestrator import CampaignOrchestrator
from refua_campaign.portfolio import PortfolioWeights, rank_disease_programs
from refua_campaign.promising_cures import (
    extract_promising_cures,
    summarize_promising_cures,
)
from refua_campaign.prompts import load_system_prompt
from refua_campaign.refua_mcp_adapter import DEFAULT_TOOL_LIST, RefuaMcpAdapter

DEFAULT_OBJECTIVE = (
    "Find cures for all diseases by prioritizing the highest-burden conditions and "
    "researching the best drug design strategies for each."
)


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

    run_parser = sub.add_parser("run", help="Run one plan+execute cycle.")
    run_parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help=(
            "Campaign objective for the planner. Defaults to an all-disease cure "
            "mission focused on worst diseases and best drug-design strategies."
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
    loop_parser.set_defaults(handler=_cmd_run_autonomous)

    prompt_parser = sub.add_parser(
        "print-default-prompt",
        help="Print the default campaign system prompt.",
    )
    prompt_parser.set_defaults(handler=_cmd_print_default_prompt)

    tools_parser = sub.add_parser("list-tools", help="List supported refua-mcp tools.")
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
    portfolio_parser.add_argument("--w-translational-readiness", type=float, default=0.10)
    portfolio_parser.add_argument("--w-novelty", type=float, default=0.10)
    portfolio_parser.set_defaults(handler=_cmd_rank_portfolio)

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

    adapter, adapter_error = _build_adapter()

    openclaw = OpenClawClient(OpenClawConfig.from_env())
    orchestrator = CampaignOrchestrator(openclaw=openclaw, refua_mcp=adapter)

    planner_text = ""
    if args.plan_file is not None:
        plan_payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
        if not isinstance(plan_payload, dict):
            raise ValueError("--plan-file must contain a JSON object.")
        plan = plan_payload
        planner_text = "Loaded from --plan-file"
    else:
        planner_text, plan = orchestrator.plan(
            objective=run_config.objective,
            system_prompt=system_prompt,
        )

    if run_config.dry_run:
        payload = {
            "objective": run_config.objective,
            "system_prompt": system_prompt,
            "planner_response_text": planner_text,
            "plan": plan,
            "dry_run": True,
        }
        if adapter_error is not None:
            payload["warnings"] = [str(adapter_error)]
    else:
        if adapter_error is not None:
            raise RuntimeError(str(adapter_error))
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
        payload = {
            "objective": run_config.objective,
            "system_prompt": system_prompt,
            "planner_response_text": planner_text,
            "plan": plan,
            "results": serialized_results,
            "promising_cures": promising_cures,
            "promising_cures_summary": summarize_promising_cures(promising_cures),
            "dry_run": False,
        }

    rendered = json.dumps(payload, indent=2)
    print(rendered)

    if run_config.output_path is not None:
        run_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_config.output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0


def _cmd_run_autonomous(args: argparse.Namespace) -> int:
    system_prompt = load_system_prompt(args.system_prompt_file)
    adapter, adapter_error = _build_adapter()
    policy = PlanPolicy(
        max_calls=max(1, int(args.max_calls)),
        require_validate_first=not bool(args.allow_skip_validate_first),
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
        )
        plan_result = planner.run(
            objective=str(args.objective),
            system_prompt=system_prompt,
            max_rounds=max(1, int(args.max_rounds)),
        )
        plan_result_payload = plan_result.to_json()

    payload = dict(plan_result_payload)
    payload["dry_run"] = bool(args.dry_run)
    if adapter_error is not None:
        payload.setdefault("warnings", []).append(str(adapter_error))

    if bool(payload.get("approved")) and not bool(args.dry_run):
        if adapter_error is not None:
            raise RuntimeError(str(adapter_error))
        final_plan = payload.get("final_plan")
        if not isinstance(final_plan, dict):
            raise ValueError("Final plan is missing from autonomous payload.")
        results = adapter.execute_plan(final_plan)
        serialized_results = [
            {
                "tool": item.tool,
                "args": item.args,
                "output": item.output,
            }
            for item in results
        ]
        promising_cures = extract_promising_cures(serialized_results)
        payload["results"] = serialized_results
        payload["promising_cures"] = promising_cures
        payload["promising_cures_summary"] = summarize_promising_cures(promising_cures)
    elif not bool(payload.get("approved")):
        payload.setdefault("warnings", []).append(
            "Autonomous loop finished without an approved plan."
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
    ranked = rank_disease_programs(payload, weights=weights)
    rendered_payload = {
        "weights": {
            "burden": weights.burden,
            "tractability": weights.tractability,
            "unmet_need": weights.unmet_need,
            "translational_readiness": weights.translational_readiness,
            "novelty": weights.novelty,
        },
        "ranked": [item.to_json() for item in ranked],
    }
    rendered = json.dumps(rendered_payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _build_adapter() -> tuple[RefuaMcpAdapter | _StaticToolAdapter, RuntimeError | None]:
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
