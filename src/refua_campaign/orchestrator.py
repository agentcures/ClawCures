from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.prompts import planner_suffix
from refua_campaign.refua_mcp_adapter import RefuaMcpAdapter, ToolExecutionResult


@dataclass
class CampaignRun:
    objective: str
    system_prompt: str
    planner_response_text: str
    plan: dict[str, Any]
    results: list[ToolExecutionResult]

    def to_json(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "system_prompt": self.system_prompt,
            "planner_response_text": self.planner_response_text,
            "plan": self.plan,
            "results": [
                {
                    "tool": item.tool,
                    "args": item.args,
                    "output": item.output,
                }
                for item in self.results
            ],
        }


class CampaignOrchestrator:
    def __init__(self, openclaw: OpenClawClient, refua_mcp: RefuaMcpAdapter) -> None:
        self._openclaw = openclaw
        self._refua_mcp = refua_mcp

    def plan(self, *, objective: str, system_prompt: str) -> tuple[str, dict[str, Any]]:
        instructions = (
            system_prompt.strip()
            + "\n\n"
            + planner_suffix(self._refua_mcp.available_tools())
        )
        response = self._openclaw.create_response(
            user_input=objective,
            instructions=instructions,
            metadata={"component": "ClawCures", "phase": "plan"},
        )
        plan = _extract_json_plan(response.text)
        return response.text, plan

    def plan_and_execute(self, *, objective: str, system_prompt: str) -> CampaignRun:
        planner_text, plan = self.plan(objective=objective, system_prompt=system_prompt)
        results = self.execute_plan(plan)
        return CampaignRun(
            objective=objective,
            system_prompt=system_prompt,
            planner_response_text=planner_text,
            plan=plan,
            results=results,
        )

    def execute_plan(self, plan: dict[str, Any]) -> list[ToolExecutionResult]:
        return self._refua_mcp.execute_plan(plan)


def _extract_json_plan(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Planner returned empty output.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(text)

    if not isinstance(parsed, dict):
        raise ValueError("Planner output must be a JSON object.")
    calls = parsed.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Planner output must contain a 'calls' list.")
    return parsed


def _extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Planner output did not contain a JSON object.")
    snippet = text[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON payload is not an object.")
    return parsed
