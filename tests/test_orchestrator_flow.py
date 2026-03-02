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


class _AutoFetchAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__(["web_search", "web_fetch"])

    def execute_plan(self, _plan: dict[str, Any]) -> list[Any]:
        return [
            ToolExecutionResult(
                tool="web_search",
                args={"query": "lung cancer targets"},
                output={
                    "results": [
                        {
                            "title": "EGFR review",
                            "url": "https://example.org/egfr",
                            "snippet": "EGFR target evidence",
                        }
                    ]
                },
            )
        ]

    def execute_tool(self, tool: str, args: dict[str, Any]) -> ToolExecutionResult:
        self.native_execute_calls.append((tool, dict(args)))
        if tool == "web_fetch":
            return ToolExecutionResult(
                tool="web_fetch",
                args=dict(args),
                output={"url": args.get("url"), "text": "EGFR is actionable."},
            )
        return super().execute_tool(tool, args)


class _FailingToolAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__(["web_fetch"])

    def execute_tool(self, tool: str, args: dict[str, Any]) -> ToolExecutionResult:
        self.native_execute_calls.append((tool, dict(args)))
        raise RuntimeError("simulated tool failure")


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


def test_orchestrator_plan_applies_agent_routing_and_evidence_items() -> None:
    openclaw = _FakeOpenClawClient(
        responses=[
            '{"calls":[{"tool":"refua_validate_spec","args":{"entities":[{"type":"protein","id":"target","sequence":"MKTAYI"}]}}]}',
        ]
    )
    adapter = _FakeAdapter(["refua_validate_spec"])
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw,
        refua_mcp=adapter,
        agent_model_map={"planner:oncology": "openclaw:oncology-planner"},
        evidence_items=[{"type": "input_text", "text": "paper evidence"}],
    )

    _planner_text, plan = orchestrator.plan(
        objective="Build a cancer target prioritization plan.",
        system_prompt="Return strict JSON plans.",
    )

    assert plan["calls"][0]["tool"] == "refua_validate_spec"
    assert len(openclaw.calls) == 1
    call = openclaw.calls[0]
    assert call.user_input == ""
    assert call.kwargs["model"] == "openclaw:oncology-planner"
    assert isinstance(call.kwargs["input_items"], list)
    assert any(item.get("text") == "paper evidence" for item in call.kwargs["input_items"])


def test_orchestrator_execute_plan_auto_expands_web_fetch_results() -> None:
    adapter = _AutoFetchAdapter()
    orchestrator = CampaignOrchestrator(
        openclaw=_FakeOpenClawClient(responses=[]),
        refua_mcp=adapter,
        auto_web_fetch=True,
        auto_web_fetch_max_urls=3,
        auto_web_fetch_max_chars=4000,
    )

    results = orchestrator.execute_plan({"calls": []})
    assert len(results) == 2
    assert results[0].tool == "web_search"
    assert results[1].tool == "web_fetch"
    assert results[1].args["url"] == "https://example.org/egfr"
    assert results[1].args["max_chars"] == 4000


def test_native_tool_loop_bootstraps_discovery_tools_first_round() -> None:
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
                        arguments={"query": "KRAS", "count": 1},
                    )
                ],
            ),
            OpenClawResponse(
                raw={"id": "resp_2", "output_text": "done"},
                text="done",
                response_id="resp_2",
                function_calls=[],
            ),
        ]
    )
    adapter = _FakeAdapter(["web_search", "web_fetch", "refua_validate_spec"])
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw,
        refua_mcp=adapter,
        native_discovery_bootstrap_rounds=1,
    )
    orchestrator.run_native_tool_loop(
        objective="Find targets",
        system_prompt="Use tools.",
    )

    first_round_tools = openclaw.calls[0].kwargs["tools"]
    second_round_tools = openclaw.calls[1].kwargs["tools"]
    first_names = {
        item["function"]["name"]
        for item in first_round_tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    second_names = {
        item["function"]["name"]
        for item in second_round_tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    assert first_names == {"web_search", "web_fetch"}
    assert "refua_validate_spec" in second_names


def test_native_tool_loop_returns_recoverable_tool_errors_when_not_fail_fast() -> None:
    openclaw = _FakeNativeOpenClawClient(
        responses=[
            OpenClawResponse(
                raw={"id": "resp_1"},
                text="",
                response_id="resp_1",
                function_calls=[
                    OpenClawFunctionCall(
                        call_id="call_1",
                        name="web_fetch",
                        arguments={"url": "http://localhost/test"},
                    )
                ],
            ),
            OpenClawResponse(
                raw={"id": "resp_2", "output_text": "Recovered"},
                text="Recovered",
                response_id="resp_2",
                function_calls=[],
            ),
        ]
    )
    adapter = _FailingToolAdapter()
    orchestrator = CampaignOrchestrator(
        openclaw=openclaw,
        refua_mcp=adapter,
        native_tool_fail_fast=False,
    )

    run = orchestrator.run_native_tool_loop(
        objective="Find targets",
        system_prompt="Use tools.",
    )

    assert len(run.results) == 1
    assert run.results[0].tool == "web_fetch"
    assert run.results[0].output["recoverable"] is True
    assert "simulated tool failure" in run.results[0].output["error"]
