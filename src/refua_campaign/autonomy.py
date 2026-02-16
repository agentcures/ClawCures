from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.orchestrator import _extract_first_json_object, _extract_json_plan
from refua_campaign.prompts import planner_suffix


@dataclass(frozen=True)
class PlanPolicy:
    max_calls: int = 10
    require_validate_first: bool = True


@dataclass(frozen=True)
class PolicyCheck:
    approved: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass
class AutonomyIteration:
    round_index: int
    planner_text: str
    plan: dict[str, Any]
    policy: PolicyCheck
    critic_text: str
    critic: dict[str, Any]


@dataclass
class AutonomousPlanResult:
    objective: str
    system_prompt: str
    iterations: list[AutonomyIteration]
    final_plan: dict[str, Any]
    approved: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "system_prompt": self.system_prompt,
            "approved": self.approved,
            "iterations": [
                {
                    "round_index": item.round_index,
                    "planner_text": item.planner_text,
                    "plan": item.plan,
                    "policy": {
                        "approved": item.policy.approved,
                        "errors": list(item.policy.errors),
                        "warnings": list(item.policy.warnings),
                    },
                    "critic_text": item.critic_text,
                    "critic": item.critic,
                }
                for item in self.iterations
            ],
            "final_plan": self.final_plan,
        }


def build_mission_milestones(objective: str) -> list[dict[str, str]]:
    _ = objective
    return [
        {
            "phase": "portfolio",
            "goal": "prioritize disease programs by burden, tractability, and unmet need",
        },
        {
            "phase": "targeting",
            "goal": "generate validated target hypotheses and assay strategies",
        },
        {
            "phase": "design",
            "goal": "produce structure-grounded candidate molecules or biologics",
        },
        {
            "phase": "screening",
            "goal": "score candidates on binding, confidence, and safety signals",
        },
        {
            "phase": "translation",
            "goal": "package reproducible evidence and regulatory-ready rationale",
        },
    ]


class AutonomousPlanner:
    def __init__(
        self,
        *,
        openclaw: OpenClawClient,
        available_tools: list[str],
        policy: PlanPolicy,
    ) -> None:
        self._openclaw = openclaw
        self._available_tools = sorted(available_tools)
        self._policy = policy

    def run(
        self,
        *,
        objective: str,
        system_prompt: str,
        max_rounds: int,
    ) -> AutonomousPlanResult:
        iterations: list[AutonomyIteration] = []
        feedback: list[str] = []
        final_plan: dict[str, Any] = {"calls": []}
        approved = False

        for idx in range(1, max(1, max_rounds) + 1):
            planner_text, plan = self._plan_once(
                objective=objective,
                system_prompt=system_prompt,
                feedback=feedback,
            )
            policy_check = evaluate_plan_policy(
                plan,
                allowed_tools=self._available_tools,
                policy=self._policy,
            )
            critic_text, critic = self._critic_once(
                objective=objective,
                plan=plan,
                policy_check=policy_check,
            )

            iteration = AutonomyIteration(
                round_index=idx,
                planner_text=planner_text,
                plan=plan,
                policy=policy_check,
                critic_text=critic_text,
                critic=critic,
            )
            iterations.append(iteration)
            final_plan = plan

            critic_approved = bool(critic.get("approved", False))
            if policy_check.approved and critic_approved:
                approved = True
                break

            new_feedback = _build_feedback(policy_check=policy_check, critic=critic)
            if not new_feedback:
                break
            feedback = new_feedback

        return AutonomousPlanResult(
            objective=objective,
            system_prompt=system_prompt,
            iterations=iterations,
            final_plan=final_plan,
            approved=approved,
        )

    def _plan_once(
        self,
        *,
        objective: str,
        system_prompt: str,
        feedback: list[str],
    ) -> tuple[str, dict[str, Any]]:
        milestone_payload = json.dumps(build_mission_milestones(objective), indent=2)
        feedback_block = ""
        if feedback:
            feedback_block = "\n\nPrevious issues to fix:\n- " + "\n- ".join(feedback)

        instructions = (
            system_prompt.strip()
            + "\n\n"
            + "Mission milestones (must be represented in your actions):\n"
            + milestone_payload
            + "\n\n"
            + planner_suffix(self._available_tools)
            + feedback_block
        )

        response = self._openclaw.create_response(
            user_input=objective,
            instructions=instructions,
            metadata={"component": "ClawCures", "phase": "plan-loop"},
        )
        plan = _extract_json_plan(response.text)
        return response.text, plan

    def _critic_once(
        self,
        *,
        objective: str,
        plan: dict[str, Any],
        policy_check: PolicyCheck,
    ) -> tuple[str, dict[str, Any]]:
        critic_prompt = {
            "objective": objective,
            "plan": plan,
            "policy": {
                "approved": policy_check.approved,
                "errors": list(policy_check.errors),
                "warnings": list(policy_check.warnings),
            },
            "required_output": {
                "approved": "boolean",
                "issues": ["string"],
                "suggested_fixes": ["string"],
            },
        }
        critic_payload = json.dumps(critic_prompt, ensure_ascii=True)

        response = self._openclaw.create_response(
            user_input=(
                "Critique this plan for scientific rigor, safety, and mission fit.\n"
                "Use this exact JSON payload as the review target:\n"
                f"{critic_payload}"
            ),
            instructions=(
                "Return JSON only with shape "
                '{"approved":bool,"issues":[...],"suggested_fixes":[...]}. '
                "Reject plans that are vague, unsafe, or non-executable."
            ),
            metadata={
                "component": "ClawCures",
                "phase": "critic-loop",
            },
        )

        parsed = _parse_critic_json(response.text)
        return response.text, parsed


def evaluate_plan_policy(
    plan: dict[str, Any],
    *,
    allowed_tools: list[str],
    policy: PlanPolicy,
) -> PolicyCheck:
    errors: list[str] = []
    warnings: list[str] = []

    calls = plan.get("calls")
    if not isinstance(calls, list):
        return PolicyCheck(
            approved=False,
            errors=("Plan must contain a 'calls' list.",),
            warnings=(),
        )

    if len(calls) == 0:
        errors.append("Plan has no tool calls.")

    if len(calls) > policy.max_calls:
        errors.append(
            f"Plan has {len(calls)} calls, exceeding policy max_calls={policy.max_calls}."
        )

    for idx, entry in enumerate(calls):
        if not isinstance(entry, dict):
            errors.append(f"Call #{idx + 1} is not an object.")
            continue
        tool = entry.get("tool")
        args = entry.get("args", {})
        if not isinstance(tool, str) or not tool:
            errors.append(f"Call #{idx + 1} has invalid tool name.")
            continue
        if tool not in allowed_tools:
            errors.append(f"Call #{idx + 1} uses unsupported tool '{tool}'.")
        if not isinstance(args, dict):
            errors.append(f"Call #{idx + 1} args must be an object.")

    if policy.require_validate_first and calls:
        first_tool = calls[0].get("tool") if isinstance(calls[0], dict) else None
        if first_tool != "refua_validate_spec":
            warnings.append(
                "First call is not refua_validate_spec; high-cost calls may fail later."
            )

    return PolicyCheck(
        approved=(len(errors) == 0),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _parse_critic_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _extract_first_json_object(stripped)

    if not isinstance(payload, dict):
        raise ValueError("Critic output must be a JSON object.")
    approved = payload.get("approved", False)
    payload["approved"] = approved if isinstance(approved, bool) else False

    issues = payload.get("issues")
    if not isinstance(issues, list):
        payload["issues"] = []
    else:
        payload["issues"] = [str(item).strip() for item in issues if str(item).strip()]

    suggested_fixes = payload.get("suggested_fixes")
    if not isinstance(suggested_fixes, list):
        payload["suggested_fixes"] = []
    else:
        payload["suggested_fixes"] = [
            str(item).strip() for item in suggested_fixes if str(item).strip()
        ]
    return payload


def _build_feedback(*, policy_check: PolicyCheck, critic: dict[str, Any]) -> list[str]:
    feedback: list[str] = []
    feedback.extend(policy_check.errors)
    feedback.extend(policy_check.warnings)

    issues = critic.get("issues")
    if isinstance(issues, list):
        feedback.extend(str(item) for item in issues if str(item).strip())

    fixes = critic.get("suggested_fixes")
    if isinstance(fixes, list):
        feedback.extend(str(item) for item in fixes if str(item).strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for item in feedback:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
