from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from refua_campaign.openclaw_client import OpenClawFunctionCall, OpenClawResponse
from refua_campaign.orchestrator import CampaignOrchestrator
from refua_campaign.refua_mcp_adapter import ToolExecutionResult


@dataclass
class _CapturedCall:
    user_input: str
    instructions: str
    metadata: dict[str, Any] | None
    kwargs: dict[str, Any]


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
        **kwargs: Any,
    ) -> OpenClawResponse:
        self.calls.append(
            _CapturedCall(
                user_input=user_input,
                instructions=instructions,
                metadata=metadata,
                kwargs=kwargs,
            )
        )
        if not self._responses:
            raise AssertionError("No fake response remaining.")
        text = self._responses.pop(0)
        return OpenClawResponse(raw={"output_text": text}, text=text)


class _FakeNativeOpenClawClient:
    def __init__(self, responses: list[OpenClawResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[_CapturedCall] = []

    def create_response(
        self,
        *,
        user_input: str,
        instructions: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OpenClawResponse:
        self.calls.append(
            _CapturedCall(
                user_input=user_input,
                instructions=instructions,
                metadata=metadata,
                kwargs=kwargs,
            )
        )
        if not self._responses:
            raise AssertionError("No fake response remaining.")
        return self._responses.pop(0)


class _FakeAdapter:
    def __init__(self, tools: list[str]) -> None:
        self._tools = list(tools)
        self.native_execute_calls: list[tuple[str, dict[str, Any]]] = []

    def available_tools(self) -> list[str]:
        return list(self._tools)

    def openclaw_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Execute {name}.",
                    "parameters": {"type": "object", "additionalProperties": True},
                },
            }
            for name in self._tools
        ]

    def execute_tool(self, tool: str, args: dict[str, Any]) -> ToolExecutionResult:
        self.native_execute_calls.append((tool, dict(args)))
        return ToolExecutionResult(tool=tool, args=dict(args), output={"ok": True})

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


def test_orchestrator_native_tool_loop_executes_function_calls() -> None:
    openclaw = _FakeNativeOpenClawClient(
        responses=[
            OpenClawResponse(
                raw={"id": "resp_1"},
                text="",
                response_id="resp_1",
                function_calls=[
                    OpenClawFunctionCall(
                        call_id="call_1",
                        name="web_search",
                        arguments={
                            "query": "lung cancer actionable targets EGFR KRAS",
                            "count": 3,
                        },
                    )
                ],
            ),
            OpenClawResponse(
                raw={"id": "resp_2", "output_text": "Completed target discovery."},
                text="Completed target discovery.",
                response_id="resp_2",
                function_calls=[],
            ),
        ]
    )
    adapter = _FakeAdapter(["web_search"])
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw,
        refua_mcp=adapter,
        session_key="campaign-main",
        store_responses=True,
        native_tool_max_rounds=4,
    )

    run = orchestrator.run_native_tool_loop(
        objective="Find disease targets with web evidence.",
        system_prompt="Use tools.",
    )

    assert len(run.results) == 1
    assert run.plan["calls"] == [
        {
            "tool": "web_search",
            "args": {"query": "lung cancer actionable targets EGFR KRAS", "count": 3},
        }
    ]
    assert "Completed target discovery." in run.planner_response_text
    assert len(openclaw.calls) == 2
    assert openclaw.calls[0].kwargs["user"] == "campaign-main"
    assert openclaw.calls[0].kwargs["store"] is True
    assert openclaw.calls[1].kwargs["previous_response_id"] == "resp_1"
    assert isinstance(openclaw.calls[1].kwargs["input_items"], list)
