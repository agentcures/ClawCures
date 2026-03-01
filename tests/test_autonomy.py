from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from refua_campaign.autonomy import (
    AutonomousPlanner,
    PlanPolicy,
    PolicyCheck,
    _parse_critic_json,
)
from refua_campaign.openclaw_client import OpenClawResponse


@dataclass
class _CapturedCall:
    user_input: str
    instructions: str
    metadata: dict[str, Any] | None


class _FakeOpenClawClient:
    def __init__(self) -> None:
        self.calls: list[_CapturedCall] = []

    def create_response(
        self,
        *,
        user_input: str,
        instructions: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawResponse:
        self.calls.append(
            _CapturedCall(
                user_input=user_input,
                instructions=instructions,
                metadata=metadata,
            )
        )
        return OpenClawResponse(
            raw={"output_text": '{"approved":true,"issues":[],"suggested_fixes":[]}'},
            text='{"approved":true,"issues":[],"suggested_fixes":[]}',
        )


def test_critic_includes_plan_payload_in_input_not_metadata() -> None:
    fake_client = _FakeOpenClawClient()
    planner = AutonomousPlanner(
        openclaw=fake_client,
        available_tools=["refua_validate_spec"],
        policy=PlanPolicy(),
    )

    plan = {"calls": [{"tool": "refua_validate_spec", "args": {"name": "demo"}}]}
    planner._critic_once(  # noqa: SLF001
        objective="Assess KRAS candidate quality",
        plan=plan,
        policy_check=PolicyCheck(approved=True, errors=(), warnings=()),
    )

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call.metadata == {"component": "ClawCures", "phase": "critic-loop"}
    assert '"objective": "Assess KRAS candidate quality"' in call.user_input
    assert '"tool": "refua_validate_spec"' in call.user_input
    assert '"name": "demo"' in call.user_input

    payload = call.user_input.split("\n", maxsplit=2)[-1]
    parsed = json.loads(payload)
    assert parsed["plan"] == plan


def test_parse_critic_json_requires_boolean_approved() -> None:
    parsed = _parse_critic_json(
        '{"approved":"false","issues":["missing controls"],"suggested_fixes":[]}'
    )
    assert parsed["approved"] is False
    assert parsed["issues"] == ["missing controls"]


class _LoopOpenClawClient:
    def create_response(
        self,
        *,
        user_input: str,
        instructions: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawResponse:
        phase = (metadata or {}).get("phase")
        if phase == "plan-loop":
            return OpenClawResponse(
                raw={
                    "output_text": '{"calls":[{"tool":"refua_validate_spec","args":{}}]}'
                },
                text='{"calls":[{"tool":"refua_validate_spec","args":{}}]}',
            )
        if phase == "critic-loop":
            return OpenClawResponse(
                raw={
                    "output_text": '{"approved":"false","issues":["unsafe"],"suggested_fixes":[]}'
                },
                text='{"approved":"false","issues":["unsafe"],"suggested_fixes":[]}',
            )
        raise AssertionError(f"Unexpected phase: {phase}")


def test_autonomous_planner_rejects_non_boolean_critic_approved() -> None:
    planner = AutonomousPlanner(
        openclaw=_LoopOpenClawClient(),
        available_tools=["refua_validate_spec"],
        policy=PlanPolicy(),
    )
    result = planner.run(
        objective="Assess target safety",
        system_prompt="Return strict JSON plans.",
        max_rounds=1,
    )
    assert result.approved is False
    assert result.iterations[0].critic["approved"] is False
