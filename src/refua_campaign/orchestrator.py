from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass
from typing import Any

from refua_campaign.agent_routing import pick_model_for_phase
from refua_campaign.openclaw_client import OpenClawClient
from refua_campaign.prompts import planner_suffix
from refua_campaign.refua_mcp_adapter import RefuaMcpAdapter, ToolExecutionResult
from refua_campaign.web_evidence import expand_results_with_web_fetch

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
_ENTITY_REQUIRED_TOOLS = frozenset(
    {"refua_validate_spec", "refua_fold", "refua_affinity"}
)
_PROTEIN_SEQUENCE_ALIASES: tuple[str, ...] = (
    "sequence",
    "target_sequence",
    "protein_sequence",
    "receptor_sequence",
    "antigen_sequence",
)
_PROTEIN_ID_ALIASES: tuple[str, ...] = (
    "protein_id",
    "target_id",
)
_LIGAND_SMILES_ALIASES: tuple[str, ...] = (
    "smiles",
    "ligand_smiles",
    "compound_smiles",
    "candidate_smiles",
    "binder_smiles",
    "molecule_smiles",
)
_LIGAND_CCD_ALIASES: tuple[str, ...] = (
    "ccd",
    "ligand_ccd",
    "compound_ccd",
    "candidate_ccd",
    "binder_ccd",
)
_LIGAND_ID_ALIASES: tuple[str, ...] = (
    "ligand_id",
    "compound_id",
    "candidate_id",
    "binder_id",
    "ligand_name",
    "compound_name",
    "candidate_name",
    "binder_name",
    "ligand",
    "compound",
    "candidate",
    "binder",
    "molecule",
)
_PROTEIN_CONTAINER_KEYS: tuple[str, ...] = ("protein", "target", "receptor", "antigen")
_LIGAND_CONTAINER_KEYS: tuple[str, ...] = ("ligand", "compound", "candidate", "binder")
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
_MISSION_EVIDENCE_URLS: tuple[str, ...] = (
    "https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://www.who.int/news-room/fact-sheets/detail/alzheimer-disease-and-other-dementias",
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
_KRAS_G12D_BOOTSTRAP_PLAN: dict[str, Any] = {
    "objective_hints": ("kras", "g12d"),
    "target_sequence": (
        "GMTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETSLLDILDTAGQEEYSAMRDQYMRTGEGF"
        "LLVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKSDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTL"
        "VREIRKHKEK"
    ),
    "smiles": (
        "Oc1cc2ccc(F)c(C#C)c2c(c1)c3ncc4c("
        "nc(OC[C@@]56CCCN5C[C@H](F)C6)nc4c3F)N7C[C@H]8CC[C@@H](C7)N8"
    ),
    "evidence_queries": (
        "KRAS G12D MRTX-1133 preclinical evidence",
        "KRAS G12D inhibitor MRTX-1133 structure 7RPZ",
    ),
    "evidence_urls": (
        "https://www.rcsb.org/structure/7RPZ",
        "https://www.cancer.gov/research/key-initiatives/ras/news-events/dialogue-blog/2022/kemp-kras-still-leading-cancer-target",
    ),
}


def _stderr_stream_callback(chunk: str) -> None:
    if not chunk:
        return
    print(chunk, end="", file=sys.stderr, flush=True)


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
        agent_model_map: dict[str, str] | None = None,
        stream_responses: bool = False,
        stream_to_stderr: bool = False,
        evidence_items: list[dict[str, Any]] | None = None,
        planner_tools: list[str] | None = None,
        native_discovery_bootstrap_rounds: int = 0,
        native_tool_fail_fast: bool = False,
        native_parallel_tool_calls: bool = True,
        native_tool_max_workers: int = 4,
        auto_web_fetch: bool = False,
        auto_web_fetch_max_urls: int = 6,
        auto_web_fetch_max_chars: int = 20_000,
    ) -> None:
        self._openclaw = openclaw
        self._refua_mcp = refua_mcp
        self._max_plan_attempts = max(1, int(max_plan_attempts))
        self._session_key = (session_key or "").strip() or None
        self._store_responses = store_responses
        self._native_tool_max_rounds = max(1, int(native_tool_max_rounds))
        self._agent_model_map = dict(agent_model_map or {})
        self._stream_responses = bool(stream_responses)
        self._stream_to_stderr = bool(stream_to_stderr)
        self._evidence_items = list(evidence_items or [])
        normalized_planner_tools = [
            str(name).strip()
            for name in (planner_tools or [])
            if isinstance(name, str) and str(name).strip()
        ]
        self._planner_tools = (
            sorted(dict.fromkeys(normalized_planner_tools))
            if normalized_planner_tools
            else None
        )
        self._native_discovery_bootstrap_rounds = max(
            0,
            int(native_discovery_bootstrap_rounds),
        )
        self._native_tool_fail_fast = bool(native_tool_fail_fast)
        self._native_parallel_tool_calls = bool(native_parallel_tool_calls)
        self._native_tool_max_workers = max(1, int(native_tool_max_workers))
        self._auto_web_fetch = bool(auto_web_fetch)
        self._auto_web_fetch_max_urls = max(0, int(auto_web_fetch_max_urls))
        self._auto_web_fetch_max_chars = max(1, int(auto_web_fetch_max_chars))

    def _openclaw_request_kwargs(
        self,
        *,
        phase: str,
        objective: str,
        allow_evidence: bool = False,
        metadata_extra: dict[str, Any] | None = None,
        previous_response_id: str | None = None,
        input_items: list[dict[str, Any]] | None = None,
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
        model_override = pick_model_for_phase(
            phase=phase,
            objective=objective,
            model_map=self._agent_model_map,
        )
        if model_override is not None:
            kwargs["model"] = model_override
        if self._stream_responses:
            kwargs["stream"] = True
            if self._stream_to_stderr:
                kwargs["on_stream_text"] = _stderr_stream_callback
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        merged_input_items: list[dict[str, Any]] = []
        if input_items:
            merged_input_items.extend(input_items)
        if allow_evidence and self._evidence_items:
            merged_input_items.extend(self._evidence_items)
        if merged_input_items:
            kwargs["input_items"] = merged_input_items
        return kwargs

    def plan(self, *, objective: str, system_prompt: str) -> tuple[str, dict[str, Any]]:
        allowed_tools = list(self._planner_tools or self._refua_mcp.available_tools())
        instructions = (
            system_prompt.strip()
            + "\n\n"
            + planner_suffix(allowed_tools)
        )
        attempt_texts: list[str] = []
        last_error: ValueError | None = None
        planner_failure: Exception | None = None

        for attempt in range(1, self._max_plan_attempts + 1):
            if attempt == 1:
                user_input = objective
                attempt_instructions = instructions
                first_turn_items: list[dict[str, Any]] | None = None
                if self._evidence_items:
                    first_turn_items = [{"type": "input_text", "text": objective}]
                    user_input = ""
                request_kwargs = self._openclaw_request_kwargs(
                    phase="plan",
                    objective=objective,
                    allow_evidence=True,
                    input_items=first_turn_items,
                )
            else:
                user_input = _build_plan_repair_input(
                    objective=objective,
                    prior_output=attempt_texts[-1] if attempt_texts else "",
                    error=last_error,
                )
                attempt_instructions = _build_plan_repair_instructions(allowed_tools)
                request_kwargs = self._openclaw_request_kwargs(
                    phase="plan-repair",
                    objective=objective,
                    metadata_extra={"attempt": str(attempt)},
                )

            try:
                response = self._openclaw.create_response(
                    user_input=user_input,
                    instructions=attempt_instructions,
                    **request_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                planner_failure = exc
                break
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
                "Planner fallback plan was used after planner failure or "
                "JSON/tool validation failures."
            )
            if last_error is not None:
                fallback_message += f" Last error: {last_error}"
            if planner_failure is not None:
                fallback_message += f" Planner failure: {planner_failure}"
            fallback_text = "\n\n".join([*attempt_texts, fallback_message]).strip()
            return fallback_text, fallback_plan

        if last_error is not None:
            raise last_error
        if planner_failure is not None:
            raise planner_failure
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
        executed = self._refua_mcp.execute_plan(plan)
        if not self._auto_web_fetch:
            return executed
        expanded, _ = expand_results_with_web_fetch(
            results=executed,
            execute_tool=self._refua_mcp.execute_tool,
            max_urls=self._auto_web_fetch_max_urls,
            max_chars=self._auto_web_fetch_max_chars,
        )
        return expanded

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
        discovery_tool_schemas = _filter_native_discovery_tool_schemas(tool_schemas)

        transcript: list[str] = []
        executed_calls: list[dict[str, Any]] = []
        results: list[ToolExecutionResult] = []
        previous_response_id: str | None = None
        pending_input_items: list[dict[str, Any]] | None = None

        for round_index in range(1, rounds + 1):
            base_input_items: list[dict[str, Any]] = []
            turn_user_input = objective if pending_input_items is None else ""
            if pending_input_items is None and self._evidence_items:
                base_input_items.append({"type": "input_text", "text": objective})
                turn_user_input = ""
            if pending_input_items:
                base_input_items.extend(pending_input_items)
            turn_tool_schemas = tool_schemas
            if (
                round_index <= self._native_discovery_bootstrap_rounds
                and discovery_tool_schemas
            ):
                turn_tool_schemas = discovery_tool_schemas
            request_kwargs = self._openclaw_request_kwargs(
                phase="native-tool-loop",
                objective=objective,
                allow_evidence=(pending_input_items is None),
                metadata_extra={"round": str(round_index)},
                previous_response_id=previous_response_id,
                input_items=base_input_items or None,
            )
            response = self._openclaw.create_response(
                user_input=turn_user_input,
                instructions=system_prompt.strip(),
                tools=turn_tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=self._native_parallel_tool_calls,
                **request_kwargs,
            )
            if response.text.strip():
                transcript.append(response.text.strip())
            if response.response_id:
                previous_response_id = response.response_id

            if not response.function_calls:
                break

            pending_input_items = []
            round_results = self._execute_native_function_calls(response.function_calls)
            for call, result in zip(response.function_calls, round_results):
                results.append(result)
                executed_calls.append({"tool": result.tool, "args": result.args})
                pending_input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result.output, ensure_ascii=True),
                    }
                )
                if self._auto_web_fetch and result.tool == "web_search":
                    expanded, generated = expand_results_with_web_fetch(
                        results=results,
                        execute_tool=self._refua_mcp.execute_tool,
                        max_urls=self._auto_web_fetch_max_urls,
                        max_chars=self._auto_web_fetch_max_chars,
                    )
                    if generated > 0:
                        new_items = expanded[len(results) :]
                        results = expanded
                        for generated_item in new_items:
                            executed_calls.append(
                                    {"tool": generated_item.tool, "args": generated_item.args}
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

    def _execute_native_function_calls(
        self,
        function_calls: list[Any],
    ) -> list[ToolExecutionResult]:
        if not function_calls:
            return []

        if (
            self._native_parallel_tool_calls
            and len(function_calls) > 1
            and all(self._is_parallel_safe_tool(call.name) for call in function_calls)
        ):
            execute_parallel = getattr(self._refua_mcp, "execute_tools_parallel", None)
            if callable(execute_parallel):
                return execute_parallel(
                    [(call.name, call.arguments) for call in function_calls],
                    max_workers=self._native_tool_max_workers,
                    fail_fast=self._native_tool_fail_fast,
                )

        round_results: list[ToolExecutionResult] = []
        for call in function_calls:
            try:
                result = self._refua_mcp.execute_tool(call.name, call.arguments)
            except Exception as exc:  # noqa: BLE001
                if self._native_tool_fail_fast:
                    raise
                result = ToolExecutionResult(
                    tool=call.name,
                    args=dict(call.arguments),
                    output={
                        "error": str(exc),
                        "failed_tool": call.name,
                        "recoverable": True,
                    },
                )
            round_results.append(result)
        return round_results

    def _is_parallel_safe_tool(self, tool: str) -> bool:
        checker = getattr(self._refua_mcp, "is_parallel_safe_tool", None)
        if callable(checker):
            try:
                return bool(checker(tool))
            except Exception:
                return False
        return False


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
    canonical = _enrich_plan_call_shapes(canonical)
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


def _enrich_plan_call_shapes(plan: dict[str, Any]) -> dict[str, Any]:
    calls = plan.get("calls")
    if not isinstance(calls, list):
        return plan

    normalized_calls: list[dict[str, Any]] = []
    last_entities: Any = None
    entities_by_name: dict[str, Any] = {}
    for entry in calls:
        if not isinstance(entry, dict):
            normalized_calls.append(entry)
            continue

        tool = entry.get("tool")
        args = entry.get("args")
        if not isinstance(tool, str) or not isinstance(args, dict):
            normalized_calls.append(entry)
            continue

        enriched_args = dict(args)
        if tool in _ENTITY_REQUIRED_TOOLS and "entities" not in enriched_args:
            inferred_entities = _infer_plan_entities(
                enriched_args,
                last_entities=last_entities,
                entities_by_name=entities_by_name,
            )
            if inferred_entities is not None:
                enriched_args["entities"] = inferred_entities

        current_entities = enriched_args.get("entities")
        if current_entities is not None:
            last_entities = current_entities
            name_key = _plan_name_key(enriched_args.get("name"))
            if name_key is not None:
                entities_by_name[name_key] = current_entities

        normalized_calls.append({"tool": tool, "args": enriched_args})

    return {"calls": normalized_calls}


def _infer_plan_entities(
    args: dict[str, Any],
    *,
    last_entities: Any,
    entities_by_name: dict[str, Any],
) -> Any | None:
    direct = _infer_entities_from_args(args)
    if direct is not None:
        return direct

    name_key = _plan_name_key(args.get("name"))
    if name_key is not None:
        mapped = entities_by_name.get(name_key)
        if mapped is not None:
            return mapped

    if last_entities is not None:
        return last_entities
    return None


def _infer_entities_from_args(args: dict[str, Any]) -> list[dict[str, Any]] | None:
    protein = _infer_protein_entity(args)
    ligand = _infer_ligand_entity(args)
    if protein is None and ligand is None:
        return None

    entities: list[dict[str, Any]] = []
    if protein is not None:
        entities.append(protein)
    if ligand is not None:
        entities.append(ligand)
    return entities


def _infer_protein_entity(args: dict[str, Any]) -> dict[str, Any] | None:
    sequence = _pick_string(args, _PROTEIN_SEQUENCE_ALIASES)
    identifier = _pick_string(args, _PROTEIN_ID_ALIASES, fallback="A")
    if sequence is not None:
        return {
            "type": "protein",
            "id": identifier,
            "sequence": sequence,
        }

    for key in _PROTEIN_CONTAINER_KEYS:
        nested = args.get(key)
        if isinstance(nested, dict):
            entity = _protein_entity_from_mapping(nested, default_id="A")
            if entity is not None:
                return entity
    return None


def _infer_ligand_entity(args: dict[str, Any]) -> dict[str, Any] | None:
    smiles = _pick_string(args, _LIGAND_SMILES_ALIASES)
    ccd = _pick_string(args, _LIGAND_CCD_ALIASES)
    identifier = _pick_string(args, _LIGAND_ID_ALIASES, fallback="lig")
    if smiles is not None or ccd is not None:
        entity: dict[str, Any] = {
            "type": "ligand",
            "id": identifier,
        }
        if smiles is not None:
            entity["smiles"] = smiles
        else:
            entity["ccd"] = ccd
        return entity

    for key in _LIGAND_CONTAINER_KEYS:
        nested = args.get(key)
        if isinstance(nested, dict):
            entity = _ligand_entity_from_mapping(nested, default_id=key)
            if entity is not None:
                return entity
    return None


def _protein_entity_from_mapping(
    payload: dict[str, Any],
    *,
    default_id: str,
) -> dict[str, Any] | None:
    sequence = _pick_string(payload, _PROTEIN_SEQUENCE_ALIASES)
    if sequence is None:
        return None
    return {
        "type": "protein",
        "id": _pick_string(payload, _PROTEIN_ID_ALIASES, fallback=default_id),
        "sequence": sequence,
    }


def _ligand_entity_from_mapping(
    payload: dict[str, Any],
    *,
    default_id: str,
) -> dict[str, Any] | None:
    smiles = _pick_string(payload, _LIGAND_SMILES_ALIASES)
    ccd = _pick_string(payload, _LIGAND_CCD_ALIASES)
    if smiles is None and ccd is None:
        return None
    entity: dict[str, Any] = {
        "type": "ligand",
        "id": _pick_string(payload, _LIGAND_ID_ALIASES, fallback=default_id),
    }
    if smiles is not None:
        entity["smiles"] = smiles
    else:
        entity["ccd"] = ccd
    return entity


def _pick_string(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    fallback: str | None = None,
) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _plan_name_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    for suffix in (
        "_validate_spec",
        "_validation",
        "_validated",
        "_affinity",
        "_fold",
        "_bootstrap",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized or None


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

        if tool in _ENTITY_REQUIRED_TOOLS:
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

        if tool in {"refua_data_fetch", "refua_data_materialize", "refua_data_query"}:
            dataset_id = args.get("dataset_id")
            if not isinstance(dataset_id, str) or not dataset_id.strip():
                raise ValueError(
                    f"Planner call #{index} for {tool} must include a non-empty "
                    "'dataset_id'."
                )

        if tool == "refua_admet_profile":
            smiles = args.get("smiles")
            if not isinstance(smiles, str) or not smiles.strip():
                raise ValueError(
                    f"Planner call #{index} for refua_admet_profile must include a "
                    "non-empty 'smiles'."
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
    targeted_plan = _build_targeted_objective_fallback_plan(
        objective=objective,
        allowed_tools=allowed_tools,
    )
    if targeted_plan is not None:
        return targeted_plan
    if not _is_all_disease_objective(objective):
        return None
    allowed_set = set(allowed_tools)
    calls: list[dict[str, Any]] = []

    if "refua_data_list" in allowed_set:
        calls.append(
            {
                "tool": "refua_data_list",
                "args": {
                    "limit": 25,
                    "include_usage_notes": True,
                    "include_urls": True,
                },
            }
        )

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

    if "web_fetch" in allowed_set:
        for url in _MISSION_EVIDENCE_URLS[:3]:
            calls.append(
                {
                    "tool": "web_fetch",
                    "args": {
                        "url": url,
                        "extract_mode": "text",
                        "max_chars": 12000,
                    },
                }
            )

    if "refua_validate_spec" not in allowed_set:
        return {"calls": calls}

    max_programs = 3 if ("web_search" in allowed_set or "web_fetch" in allowed_set) else 5
    seed_programs = _MISSION_BOOTSTRAP_PROGRAMS[:max_programs]
    for item in seed_programs:
        disease_slug = item["disease_slug"]
        candidate_slug = item["candidate_slug"]
        entities = [
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
        ]
        calls.append(
            {
                "tool": "refua_validate_spec",
                "args": {
                    "name": f"{disease_slug}_{candidate_slug}_bootstrap",
                    "action": "affinity",
                    "deep_validate": False,
                    "entities": entities,
                },
            }
        )

        if "refua_affinity" in allowed_set:
            calls.append(
                {
                    "tool": "refua_affinity",
                    "args": {
                        "name": f"{disease_slug}_{candidate_slug}_affinity",
                        "entities": entities,
                    },
                }
            )
        elif "refua_fold" in allowed_set:
            calls.append(
                {
                    "tool": "refua_fold",
                    "args": {
                        "name": f"{disease_slug}_{candidate_slug}_fold",
                        "entities": entities,
                        "affinity": True,
                    },
                }
            )

        if "refua_admet_profile" in allowed_set:
            calls.append(
                {
                    "tool": "refua_admet_profile",
                    "args": {
                        "smiles": item["smiles"],
                    },
                }
            )

        if "refua_clinical_simulator" in allowed_set:
            calls.append(
                {
                    "tool": "refua_clinical_simulator",
                    "args": {
                        "trial_id": f"{disease_slug}_{candidate_slug}_phase2_sim",
                        "indication": disease_slug.replace("_", " "),
                        "phase": "Phase II",
                        "objective": (
                            "Assess translational potential for burden-prioritized "
                            "disease program."
                        ),
                        "include_workup": True,
                        "include_replicates": False,
                    },
                }
            )
    return {"calls": calls}


def _is_all_disease_objective(objective: str) -> bool:
    lowered = objective.lower()
    return any(token in lowered for token in _ALL_DISEASE_OBJECTIVE_HINTS)


def _build_targeted_objective_fallback_plan(
    *,
    objective: str,
    allowed_tools: list[str],
) -> dict[str, Any] | None:
    lowered = objective.lower()
    objective_hints = _KRAS_G12D_BOOTSTRAP_PLAN["objective_hints"]
    if not all(token in lowered for token in objective_hints):
        return None

    allowed_set = set(allowed_tools)
    calls: list[dict[str, Any]] = []

    if "web_search" in allowed_set:
        for query in _KRAS_G12D_BOOTSTRAP_PLAN["evidence_queries"]:
            calls.append(
                {
                    "tool": "web_search",
                    "args": {
                        "query": query,
                        "count": 5,
                    },
                }
            )

    if "web_fetch" in allowed_set:
        for url in _KRAS_G12D_BOOTSTRAP_PLAN["evidence_urls"]:
            calls.append(
                {
                    "tool": "web_fetch",
                    "args": {
                        "url": url,
                        "extract_mode": "text",
                        "max_chars": 12000,
                    },
                }
            )

    entities = [
        {
            "type": "protein",
            "id": "A",
            "sequence": _KRAS_G12D_BOOTSTRAP_PLAN["target_sequence"],
        },
        {
            "type": "ligand",
            "id": "candidate",
            "smiles": _KRAS_G12D_BOOTSTRAP_PLAN["smiles"],
        },
    ]
    program_name = "kras_g12d_mrtx1133"

    if "refua_validate_spec" in allowed_set:
        calls.append(
            {
                "tool": "refua_validate_spec",
                "args": {
                    "name": f"{program_name}_validate",
                    "action": "affinity",
                    "deep_validate": False,
                    "entities": entities,
                },
            }
        )

    if "refua_affinity" in allowed_set:
        calls.append(
            {
                "tool": "refua_affinity",
                "args": {
                    "name": f"{program_name}_affinity",
                    "entities": entities,
                    "binder": "candidate",
                },
            }
        )
    elif "refua_fold" in allowed_set:
        calls.append(
            {
                "tool": "refua_fold",
                "args": {
                    "name": f"{program_name}_fold",
                    "entities": entities,
                    "affinity": {"binder": "candidate"},
                },
            }
        )

    return {"calls": calls} if calls else None


def _filter_native_discovery_tool_schemas(
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    discovery_names = {"web_search", "web_fetch"}
    filtered: list[dict[str, Any]] = []
    for item in tool_schemas:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip() != "function":
            continue
        function_block = item.get("function")
        if not isinstance(function_block, dict):
            continue
        name = str(function_block.get("name") or "").strip()
        if name in discovery_names:
            filtered.append(item)
    return filtered
