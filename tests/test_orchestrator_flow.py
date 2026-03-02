from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from refua_campaign.openclaw_client import OpenClawResponse
from refua_campaign.orchestrator import CampaignOrchestrator


@dataclass
class _CapturedCall:
    user_input: str
    instructions: str
    metadata: dict[str, Any] | None


class _FakeOpenClawClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
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
        if not self._responses:
            raise AssertionError("No fake response remaining.")
        text = self._responses.pop(0)
        return OpenClawResponse(raw={"output_text": text}, text=text)


class _FakeAdapter:
    def __init__(self, tools: list[str]) -> None:
        self._tools = list(tools)

    def available_tools(self) -> list[str]:
        return list(self._tools)

    def execute_plan(self, _plan: dict[str, Any]) -> list[Any]:
        return []


def test_orchestrator_plan_repairs_invalid_first_response() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            "Please clarify your request.",
            (
                '{"calls":[{"tool":"validate_spec","arguments":{"entities":[{"type":"protein",'
                '"id":"target","sequence":"MKTAYI"}],"deep_validate":false}}]}'
            ),
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec"])
    orchestrator = CampaignOrchestrator(openclaw=openclaw, refua_mcp=adapter, max_plan_attempts=2)

    _planner_text, plan = orchestrator.plan(
        objective="Find cures for all diseases",
        system_prompt="Return strict JSON plans.",
    )

    assert len(openclaw.calls) == 2
    assert openclaw.calls[0].metadata == {"component": "ClawCures", "phase": "plan"}
    assert openclaw.calls[1].metadata is not None
    assert openclaw.calls[1].metadata.get("phase") == "plan-repair"
    assert plan["calls"][0]["tool"] == "refua_validate_spec"
    assert plan["calls"][0]["args"]["deep_validate"] is False


def test_orchestrator_plan_uses_mission_fallback_for_all_disease_objective() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            "I need more context first.",
            "Still need context before producing a tool plan.",
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec"])
    orchestrator = CampaignOrchestrator(openclaw=openclaw, refua_mcp=adapter, max_plan_attempts=2)

    planner_text, plan = orchestrator.plan(
        objective=(
            "Find cures for all diseases by prioritizing the highest-burden conditions "
            "and researching the best drug design strategies for each."
        ),
        system_prompt="Return strict JSON plans.",
    )

    assert "Planner fallback plan was used" in planner_text
    assert len(plan["calls"]) >= 1
    assert all(call["tool"] == "refua_validate_spec" for call in plan["calls"])


def test_orchestrator_plan_fallback_adds_web_search_when_available() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            "I need more context first.",
            "Still need context before producing a tool plan.",
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec", "web_search"])
    orchestrator = CampaignOrchestrator(openclaw=openclaw, refua_mcp=adapter, max_plan_attempts=2)

    _planner_text, plan = orchestrator.plan(
        objective=(
            "Find cures for all diseases by prioritizing the highest-burden conditions "
            "and researching the best drug design strategies for each."
        ),
        system_prompt="Return strict JSON plans.",
    )

    tools = [call["tool"] for call in plan["calls"]]
    assert "web_search" in tools
    assert "refua_validate_spec" in tools


def test_orchestrator_plan_falls_back_when_semantically_invalid_for_mission() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            (
                '{"calls":[{"tool":"refua_validate_spec","args":{"objective":"global cure '
                'roadmap"}},{"tool":"refua_job","args":{"action":"create_program"}}]}'
            ),
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec", "refua_job"])
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw, refua_mcp=adapter, max_plan_attempts=1
    )

    planner_text, plan = orchestrator.plan(
        objective=(
            "Find cures for all diseases by prioritizing the highest-burden conditions "
            "and researching the best drug design strategies for each."
        ),
        system_prompt="Return strict JSON plans.",
    )

    assert "Planner fallback plan was used" in planner_text
    assert len(plan["calls"]) >= 1
    assert all(call["tool"] == "refua_validate_spec" for call in plan["calls"])


def test_orchestrator_plan_raises_for_non_mission_objective_after_failures() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            "Not a JSON plan.",
            "Still not a JSON plan.",
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec"])
    orchestrator = CampaignOrchestrator(openclaw=openclaw, refua_mcp=adapter, max_plan_attempts=2)

    with pytest.raises(ValueError, match="Planner output did not contain a JSON object."):
        orchestrator.plan(
            objective="Build a focused EGFR plan.",
            system_prompt="Return strict JSON plans.",
        )
