from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.prompts import planner_suffix
from refua_campaign.refua_mcp_adapter import RefuaMcpAdapter, ToolExecutionResult

_PLAN_REPAIR_TEXT_LIMIT = 12_000
_ALL_DISEASE_OBJECTIVE_HINTS: tuple[str, ...] = (
    "find cures for all diseases",
    "all diseases",
    "all human disease",
    "solve all human disease",
)
_TOOL_ALIAS_MAP: dict[str, str] = {
    "validate_spec": "refua_validate_spec",
    "refua_validate": "refua_validate_spec",
    "protein_properties": "refua_protein_properties",
    "refua_protein_property": "refua_protein_properties",
    "clinical_simulator": "refua_clinical_simulator",
    "jobs": "refua_job",
    "websearch": "web_search",
    "webfetch": "web_fetch",
}
_MISSION_TARGET_DISCOVERY_QUERIES: tuple[dict[str, str], ...] = (
    {
        "disease_slug": "ischemic_heart_disease",
        "query": (
            "ischemic heart disease validated therapeutic targets "
            "PCSK9 LPA IL1B NLRP3 review"
        ),
    },
    {
        "disease_slug": "lung_cancer",
        "query": (
            "lung cancer actionable therapeutic targets "
            "EGFR ALK KRAS MET review"
        ),
    },
    {
        "disease_slug": "alzheimers_disease",
        "query": (
            "alzheimer disease therapeutic targets "
            "APP MAPT TREM2 APOE review"
        ),
    },
    {
        "disease_slug": "type_2_diabetes",
        "query": (
            "type 2 diabetes therapeutic targets "
            "GLP1R SGLT2 PPARG GIPR review"
        ),
    },
    {
        "disease_slug": "tuberculosis",
        "query": (
            "tuberculosis validated drug targets "
            "InhA DprE1 ATP synthase review"
        ),
    },
    {
        "disease_slug": "hiv",
        "query": (
            "HIV cure and functional cure targets "
            "CCR5 integrase reverse transcriptase review"
        ),
    },
)
_MISSION_BOOTSTRAP_PROGRAMS: tuple[dict[str, str], ...] = (
    {
        "disease_slug": "ischemic_heart_disease",
        "candidate_slug": "aspirin",
        "target_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    },
    {
        "disease_slug": "stroke_prevention",
        "candidate_slug": "clopidogrel",
        "target_sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQR",
        "smiles": "COC(=O)N[C@H](C1=CC=CC=C1Cl)SC2=NC=CC=C2",
    },
    {
        "disease_slug": "type_2_diabetes",
        "candidate_slug": "metformin",
        "target_sequence": "MNNKRTKQSLVLRQLESLKSNQNNRGLKQVEQ",
        "smiles": "CN(C)C(=N)N",
    },
    {
        "disease_slug": "tuberculosis",
        "candidate_slug": "isoniazid",
        "target_sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGV",
        "smiles": "NNC(=O)C1=CC=NC=C1",
    },
    {
        "disease_slug": "hiv",
        "candidate_slug": "dolutegravir",
        "target_sequence": "MNNRQILSMRDKKELKQLEEQLKQLEAELKQ",
        "smiles": "CC1=CC2=C(N1)N(C(=O)N2C)CC(C(=O)O)O",
    },
    {
        "disease_slug": "lung_cancer",
        "candidate_slug": "imatinib",
        "target_sequence": "MSDVAALRGCNQSLNERVKQLEAELQKQLEA",
        "smiles": "CC1=CC(=CC=C1NC(=O)C2=NC=CC(=N2)NCC3=CC=CC=C3)N",
    },
    {
        "disease_slug": "copd",
        "candidate_slug": "albuterol",
        "target_sequence": "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRL",
        "smiles": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
    },
    {
        "disease_slug": "alzheimers_disease",
        "candidate_slug": "donepezil",
        "target_sequence": "MENSDSPEKVSATPKKDKKTKQATPKKAAATK",
        "smiles": "COC1=CC2=C(C=C1)C(CC3=CC=CC=C3)N(CC4=CC=CC=C4)CC2",
    },
)


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
    def __init__(
        self,
        openclaw: OpenClawClient,
        refua_mcp: RefuaMcpAdapter,
        *,
        max_plan_attempts: int = 3,
        session_key: str | None = None,
        store_responses: bool | None = None,
        native_tool_max_rounds: int = 8,
    ) -> None:
        self._openclaw = openclaw
        self._refua_mcp = refua_mcp
        self._max_plan_attempts = max(1, int(max_plan_attempts))
        self._session_key = (session_key or "").strip() or None
        self._store_responses = store_responses
        self._native_tool_max_rounds = max(1, int(native_tool_max_rounds))

    def _openclaw_request_kwargs(
        self,
        *,
        phase: str,
        metadata_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"component": "ClawCures", "phase": phase}
        if metadata_extra:
            metadata.update(metadata_extra)
        if self._session_key:
            metadata["session_key"] = self._session_key

        kwargs: dict[str, Any] = {"metadata": metadata}
        if self._session_key:
            kwargs["user"] = self._session_key
        if self._store_responses is not None:
            kwargs["store"] = bool(self._store_responses)
        return kwargs

    def plan(self, *, objective: str, system_prompt: str) -> tuple[str, dict[str, Any]]:
        allowed_tools = self._refua_mcp.available_tools()
        instructions = (
            system_prompt.strip()
            + "\n\n"
            + planner_suffix(allowed_tools)
        )
        attempt_texts: list[str] = []
        last_error: ValueError | None = None

        for attempt in range(1, self._max_plan_attempts + 1):
            if attempt == 1:
                user_input = objective
                attempt_instructions = instructions
                request_kwargs = self._openclaw_request_kwargs(phase="plan")
            else:
                user_input = _build_plan_repair_input(
                    objective=objective,
                    prior_output=attempt_texts[-1] if attempt_texts else "",
                    error=last_error,
                )
                attempt_instructions = _build_plan_repair_instructions(allowed_tools)
                request_kwargs = self._openclaw_request_kwargs(
                    phase="plan-repair",
                    metadata_extra={"attempt": str(attempt)},
                )

            response = self._openclaw.create_response(
                user_input=user_input,
                instructions=attempt_instructions,
                **request_kwargs,
            )
            attempt_texts.append(response.text)

            try:
                plan = _extract_json_plan(response.text, allowed_tools=allowed_tools)
                return response.text, plan
            except ValueError as exc:
                last_error = exc

        fallback_plan = _build_default_objective_fallback_plan(
            objective=objective,
            allowed_tools=allowed_tools,
        )
        if fallback_plan is not None:
            fallback_message = (
                "Planner fallback plan was used after JSON/tool validation failures."
            )
            if last_error is not None:
                fallback_message += f" Last error: {last_error}"
            fallback_text = "\n\n".join([*attempt_texts, fallback_message]).strip()
            return fallback_text, fallback_plan

        if last_error is not None:
            raise last_error
        raise ValueError("Planner failed without producing a valid plan.")

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

    def run_native_tool_loop(
        self,
        *,
        objective: str,
        system_prompt: str,
        max_rounds: int | None = None,
    ) -> CampaignRun:
        rounds = (
            self._native_tool_max_rounds
            if max_rounds is None
            else max(1, int(max_rounds))
        )
        tool_schemas = self._refua_mcp.openclaw_tool_schemas()
        if not tool_schemas:
            raise RuntimeError("No OpenClaw function tool schemas are available.")

        transcript: list[str] = []
        executed_calls: list[dict[str, Any]] = []
        results: list[ToolExecutionResult] = []
        previous_response_id: str | None = None
        pending_input_items: list[dict[str, Any]] | None = None

        for round_index in range(1, rounds + 1):
            request_kwargs = self._openclaw_request_kwargs(
                phase="native-tool-loop",
                metadata_extra={"round": str(round_index)},
            )
            response = self._openclaw.create_response(
                user_input=objective if pending_input_items is None else "",
                input_items=pending_input_items,
                instructions=system_prompt.strip(),
                tools=tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=False,
                previous_response_id=previous_response_id,
                **request_kwargs,
            )
            if response.text.strip():
                transcript.append(response.text.strip())
            if response.response_id:
                previous_response_id = response.response_id

            if not response.function_calls:
                break

            pending_input_items = []
            for call in response.function_calls:
                result = self._refua_mcp.execute_tool(call.name, call.arguments)
                results.append(result)
                executed_calls.append({"tool": result.tool, "args": result.args})
                pending_input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result.output, ensure_ascii=True),
                    }
                )
        else:
            transcript.append(
                f"Native tool loop reached max_rounds={rounds} before completion."
            )

        return CampaignRun(
            objective=objective,
            system_prompt=system_prompt,
            planner_response_text="\n\n".join(transcript).strip(),
            plan={"calls": executed_calls},
            results=results,
        )


def _extract_json_plan(
    text: str,
    *,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Planner returned empty output.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(text)

    plan = _normalize_plan_payload(parsed)
    if allowed_tools is None:
        return plan

    canonical = _canonicalize_plan_tools(plan, allowed_tools=allowed_tools)
    _validate_plan_tools(canonical, allowed_tools=allowed_tools)
    _validate_plan_call_shapes(canonical)
    return canonical


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


def _normalize_plan_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = {"calls": payload}

    if not isinstance(payload, dict):
        raise ValueError("Planner output must be a JSON object.")

    calls = payload.get("calls")
    if not isinstance(calls, list):
        for nested_key in ("plan", "result", "output"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("calls"), list):
                calls = nested["calls"]
                break

    if not isinstance(calls, list) and isinstance(payload.get("tool_calls"), list):
        calls = payload["tool_calls"]

    if not isinstance(calls, list):
        raise ValueError("Planner output must contain a 'calls' list.")

    normalized_calls: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"Planner call #{index + 1} is not an object.")

        tool_raw = call.get("tool")
        function_block = call.get("function")
        if tool_raw is None and isinstance(function_block, dict):
            tool_raw = function_block.get("name")
        if tool_raw is None:
            tool_raw = call.get("name")

        if not isinstance(tool_raw, str) or not tool_raw.strip():
            raise ValueError(f"Planner call #{index + 1} has no valid tool name.")

        args_raw = call.get("args")
        if args_raw is None:
            args_raw = call.get("arguments")
        if args_raw is None and isinstance(function_block, dict):
            args_raw = function_block.get("arguments")
        if args_raw is None:
            args_raw = call.get("params", {})

        if isinstance(args_raw, str):
            stripped = args_raw.strip()
            if not stripped:
                args_raw = {}
            else:
                try:
                    args_raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Planner call #{index + 1} args string is not valid JSON."
                    ) from exc

        if args_raw is None:
            args_raw = {}
        if not isinstance(args_raw, dict):
            raise ValueError(f"Planner call #{index + 1} args must be an object.")

        normalized_calls.append(
            {
                "tool": tool_raw.strip(),
                "args": args_raw,
            }
        )

    return {"calls": normalized_calls}


def _canonicalize_plan_tools(
    plan: dict[str, Any],
    *,
    allowed_tools: list[str],
) -> dict[str, Any]:
    calls = plan.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Plan must contain a 'calls' list.")

    normalized_calls: list[dict[str, Any]] = []
    for entry in calls:
        if not isinstance(entry, dict):
            raise ValueError("Each call must be an object.")
        tool_raw = entry.get("tool")
        args_raw = entry.get("args", {})
        if not isinstance(tool_raw, str) or not tool_raw.strip():
            raise ValueError("Each call must define a non-empty 'tool'.")
        if not isinstance(args_raw, dict):
            raise ValueError("Each call args must be an object.")
        normalized_calls.append(
            {
                "tool": _canonicalize_tool_name(tool_raw, allowed_tools=allowed_tools),
                "args": args_raw,
            }
        )
    return {"calls": normalized_calls}


def _canonicalize_tool_name(tool: str, *, allowed_tools: list[str]) -> str:
    normalized = tool.strip()
    if not normalized:
        return normalized

    allowed_set = set(allowed_tools)
    if normalized in allowed_set:
        return normalized

    lowered = normalized.lower().replace("-", "_").replace(" ", "_")
    lower_lookup = {name.lower(): name for name in allowed_tools}
    if lowered in lower_lookup:
        return lower_lookup[lowered]

    if "." in lowered:
        tail = lowered.rsplit(".", maxsplit=1)[-1]
        if tail in lower_lookup:
            return lower_lookup[tail]
        lowered = tail

    alias_target = _TOOL_ALIAS_MAP.get(lowered)
    if alias_target is not None and alias_target in allowed_set:
        return alias_target

    fuzzy = difflib.get_close_matches(lowered, list(lower_lookup), n=1, cutoff=0.9)
    if fuzzy:
        return lower_lookup[fuzzy[0]]
    return normalized


def _validate_plan_tools(plan: dict[str, Any], *, allowed_tools: list[str]) -> None:
    calls = plan.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Plan must contain a 'calls' list.")

    allowed_set = set(allowed_tools)
    unsupported: set[str] = set()
    for entry in calls:
        if not isinstance(entry, dict):
            raise ValueError("Each call must be an object.")
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("Each call must define a non-empty 'tool'.")
        if tool not in allowed_set:
            unsupported.add(tool)

    if unsupported:
        allowed_csv = ", ".join(sorted(allowed_tools))
        unsupported_csv = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Planner used unsupported tool(s): {unsupported_csv}. "
            f"Allowed tools: {allowed_csv}."
        )


def _validate_plan_call_shapes(plan: dict[str, Any]) -> None:
    calls = plan.get("calls")
    if not isinstance(calls, list):
        return

    for index, entry in enumerate(calls, start=1):
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if not isinstance(tool, str):
            continue
        args = entry.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"Planner call #{index} args must be an object.")

        if tool in {"refua_validate_spec", "refua_fold", "refua_affinity"}:
            if "entities" not in args:
                raise ValueError(
                    f"Planner call #{index} for {tool} must include 'entities'."
                )

        if tool == "refua_job":
            job_id = args.get("job_id")
            if not isinstance(job_id, str) or not job_id.strip():
                if "action" in args:
                    raise ValueError(
                        f"Planner call #{index} for refua_job used workflow-style "
                        "arguments. refua_job expects a 'job_id'."
                    )
                raise ValueError(
                    f"Planner call #{index} for refua_job must include a non-empty "
                    "'job_id'."
                )

        if tool == "web_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                query = args.get("q")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Planner call #{index} for web_search must include a non-empty "
                    "'query'."
                )
            count = args.get("count")
            if count is not None and not isinstance(count, int):
                raise ValueError(
                    f"Planner call #{index} for web_search count must be an integer."
                )

        if tool == "web_fetch":
            url = args.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(
                    f"Planner call #{index} for web_fetch must include a non-empty "
                    "'url'."
                )


def _build_plan_repair_instructions(allowed_tools: list[str]) -> str:
    tools = ", ".join(sorted(allowed_tools))
    return (
        "Repair the planner output into a strict execution plan. "
        'Return only JSON with shape {"calls":[{"tool":"<name>","args":{...}}]}. '
        f"Allowed tools: {tools}. "
        "Use key 'args' (not 'arguments'). "
        'If context is insufficient, return {"calls":[]}. '
        "Never emit markdown, prose, or comments."
    )


def _build_plan_repair_input(
    *,
    objective: str,
    prior_output: str,
    error: Exception | None,
) -> str:
    clipped = prior_output.strip()
    if len(clipped) > _PLAN_REPAIR_TEXT_LIMIT:
        clipped = clipped[: _PLAN_REPAIR_TEXT_LIMIT - 3] + "..."
    reason = str(error) if error is not None else "Invalid planner output."
    return (
        "The previous planner output was invalid.\n"
        f"Objective: {objective}\n"
        f"Validation error: {reason}\n"
        "Rewrite the previous output into a valid tool plan.\n"
        f"Previous output:\n{clipped}"
    )


def _build_default_objective_fallback_plan(
    *,
    objective: str,
    allowed_tools: list[str],
) -> dict[str, Any] | None:
    if not _is_all_disease_objective(objective):
        return None
    allowed_set = set(allowed_tools)
    calls: list[dict[str, Any]] = []

    if "web_search" in allowed_set:
        for item in _MISSION_TARGET_DISCOVERY_QUERIES:
            calls.append(
                {
                    "tool": "web_search",
                    "args": {
                        "query": item["query"],
                        "count": 5,
                    },
                }
            )

    if "refua_validate_spec" not in allowed_set:
        return {"calls": calls}

    max_validation_calls = (
        4 if "web_search" in allowed_set else len(_MISSION_BOOTSTRAP_PROGRAMS)
    )
    for item in _MISSION_BOOTSTRAP_PROGRAMS[:max_validation_calls]:
        disease_slug = item["disease_slug"]
        candidate_slug = item["candidate_slug"]
        calls.append(
            {
                "tool": "refua_validate_spec",
                "args": {
                    "name": f"{disease_slug}_{candidate_slug}_bootstrap",
                    "action": "affinity",
                    "deep_validate": False,
                    "entities": [
                        {
                            "type": "protein",
                            "id": "target",
                            "sequence": item["target_sequence"],
                        },
                        {
                            "type": "ligand",
                            "id": "candidate",
                            "smiles": item["smiles"],
                        },
                    ],
                },
            }
        )
    return {"calls": calls}


def _is_all_disease_objective(objective: str) -> bool:
    lowered = objective.lower()
    return any(token in lowered for token in _ALL_DISEASE_OBJECTIVE_HINTS)
