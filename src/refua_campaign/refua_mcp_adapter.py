from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

DEFAULT_TOOL_LIST: tuple[str, ...] = (
    "refua_validate_spec",
    "refua_fold",
    "refua_affinity",
    "refua_antibody_design",
    "refua_protein_properties",
    "refua_clinical_simulator",
    "refua_data_list",
    "refua_data_fetch",
    "refua_data_materialize",
    "refua_data_query",
    "refua_job",
    "refua_admet_profile",
)


def _import_refua_mcp_server():
    first_error: ModuleNotFoundError | None = None
    try:
        from refua_mcp import server  # type: ignore

        return server
    except ModuleNotFoundError as exc:
        first_error = exc
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "refua-mcp" / "src"
        if local_src.exists():
            sys.path.insert(0, str(local_src))
            try:
                from refua_mcp import server  # type: ignore

                return server
            except ModuleNotFoundError as nested_exc:
                raise RuntimeError(
                    "Failed to import refua-mcp from local source. "
                    "Install refua-mcp dependencies first "
                    f"(missing module: {nested_exc.name})."
                ) from nested_exc
        missing = first_error.name if first_error else "refua_mcp"
        raise RuntimeError(
            "refua-mcp is not available. Install it with dependencies before running "
            f"campaign execution (missing module: {missing})."
        ) from exc


def _discover_tool_names(server: Any) -> list[str]:
    tool_manager = getattr(getattr(server, "mcp", None), "_tool_manager", None)
    if tool_manager is None:
        return []

    list_tools = getattr(tool_manager, "list_tools", None)
    if not callable(list_tools):
        return []

    try:
        tool_infos = list_tools()
    except Exception:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for info in tool_infos:
        name = getattr(info, "name", None)
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _load_tool_map() -> dict[str, Callable[..., Any]]:
    server = _import_refua_mcp_server()

    tool_map: dict[str, Callable[..., Any]] = {}

    for name in _discover_tool_names(server):
        fn = getattr(server, name, None)
        if callable(fn):
            tool_map[name] = fn

    if tool_map:
        return tool_map

    for name in DEFAULT_TOOL_LIST:
        fn = getattr(server, name, None)
        if callable(fn):
            tool_map[name] = fn

    if not tool_map:
        raise RuntimeError("No executable refua-mcp tools were discovered.")
    return tool_map


@dataclass
class ToolExecutionResult:
    tool: str
    args: dict[str, Any]
    output: Any


class RefuaMcpAdapter:
    def __init__(self) -> None:
        self._tools = _load_tool_map()

    def available_tools(self) -> list[str]:
        return sorted(self._tools)

    def execute_tool(self, tool: str, args: dict[str, Any]) -> ToolExecutionResult:
        if tool not in self._tools:
            raise ValueError(f"Unsupported tool: {tool}")
        fn = self._tools[tool]
        result = fn(**args)
        return ToolExecutionResult(
            tool=tool,
            args=dict(args),
            output=_to_plain_data(result),
        )

    def execute_plan(self, plan: dict[str, Any]) -> list[ToolExecutionResult]:
        calls = plan.get("calls")
        if not isinstance(calls, list):
            raise ValueError("Plan must contain a 'calls' list.")

        results: list[ToolExecutionResult] = []
        for entry in calls:
            if not isinstance(entry, dict):
                raise ValueError("Each plan call must be an object.")
            tool = entry.get("tool")
            args = entry.get("args", {})
            if not isinstance(tool, str) or not tool:
                raise ValueError("Each plan call must define a non-empty 'tool'.")
            if not isinstance(args, dict):
                raise ValueError("Each plan call 'args' must be an object.")
            results.append(self.execute_tool(tool, args))
        return results


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain_data(v) for v in value]
    return value
