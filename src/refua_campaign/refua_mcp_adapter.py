from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

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
    "web_search",
    "web_fetch",
)

_HTTP_TIMEOUT_SECONDS = 30.0
_DEFAULT_SEARCH_COUNT = 5
_MAX_SEARCH_COUNT = 10
_DEFAULT_MAX_FETCH_CHARS = 30_000
_MAX_FETCH_CHARS = 200_000
_HTTP_USER_AGENT = "ClawCures/1.0"
_ALLOW_PRIVATE_FETCH_ENV = "CLAWCURES_ALLOW_PRIVATE_WEB_FETCH"
_OPENCLAW_DEFAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
}
_OPENCLAW_TOOL_SCHEMA_OVERRIDES: dict[str, dict[str, Any]] = {
    "refua_validate_spec": {
        "type": "object",
        "properties": {
            "entities": {"type": ["array", "string"]},
            "action": {"type": "string", "enum": ["fold", "affinity"]},
            "name": {"type": "string"},
            "base_dir": {"type": "string"},
            "constraints": {"type": ["array", "string"]},
            "affinity": {"type": ["boolean", "object", "null"]},
            "run_boltz": {"type": "boolean"},
            "run_boltzgen": {"type": "boolean"},
            "boltz": {"type": "object"},
            "boltzgen": {"type": "object"},
            "admet": {
                "type": ["string", "boolean", "object", "null"],
            },
            "structure_output_path": {"type": "string"},
            "structure_output_format": {
                "type": ["string", "null"],
                "enum": ["cif", "mmcif", "bcif", None],
            },
            "feature_output_path": {"type": "string"},
            "feature_output_format": {
                "type": ["string", "null"],
                "enum": ["torch", "npz", "json", None],
            },
            "deep_validate": {"type": "boolean"},
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
    "refua_fold": {
        "type": "object",
        "properties": {
            "entities": {"type": ["array", "string"]},
            "name": {"type": "string"},
            "base_dir": {"type": "string"},
            "constraints": {"type": ["array", "string", "null"]},
            "affinity": {"type": ["boolean", "object", "null"]},
            "run_boltz": {"type": "boolean"},
            "run_boltzgen": {"type": "boolean"},
            "boltz": {"type": "object"},
            "boltzgen": {"type": "object"},
            "admet": {"type": ["string", "boolean", "object", "null"]},
            "structure_output_path": {"type": "string"},
            "structure_output_format": {
                "type": ["string", "null"],
                "enum": ["cif", "mmcif", "bcif", None],
            },
            "feature_output_path": {"type": "string"},
            "feature_output_format": {
                "type": ["string", "null"],
                "enum": ["torch", "npz", None],
            },
            "allow_exploratory_run": {"type": "boolean"},
            "async_mode": {"type": "boolean"},
            "queue_timeout_seconds": {"type": "number"},
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
    "refua_affinity": {
        "type": "object",
        "properties": {
            "entities": {"type": ["array", "string"]},
            "name": {"type": "string"},
            "base_dir": {"type": "string"},
            "binder": {"type": "string"},
            "boltz": {"type": "object"},
            "admet": {"type": ["string", "boolean", "object", "null"]},
            "async_mode": {"type": "boolean"},
            "queue_timeout_seconds": {"type": "number"},
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
    "refua_antibody_design": {
        "type": "object",
        "properties": {
            "antibody": {"type": ["object", "string"]},
            "context_entities": {"type": ["array", "string", "null"]},
            "name": {"type": "string"},
            "base_dir": {"type": "string"},
            "constraints": {"type": ["array", "string", "null"]},
            "affinity": {"type": ["boolean", "object", "null"]},
            "run_boltz": {"type": "boolean"},
            "run_boltzgen": {"type": "boolean"},
            "boltz": {"type": "object"},
            "boltzgen": {"type": "object"},
            "admet": {"type": ["string", "boolean", "object", "null"]},
            "structure_output_path": {"type": "string"},
            "structure_output_format": {
                "type": ["string", "null"],
                "enum": ["cif", "mmcif", "bcif", None],
            },
            "feature_output_path": {"type": "string"},
            "feature_output_format": {
                "type": ["string", "null"],
                "enum": ["torch", "npz", None],
            },
            "allow_exploratory_run": {"type": "boolean"},
            "async_mode": {"type": "boolean"},
            "queue_timeout_seconds": {"type": "number"},
        },
        "required": ["antibody"],
        "additionalProperties": False,
    },
    "refua_protein_properties": {
        "type": "object",
        "properties": {
            "sequence": {"type": "string"},
            "properties": {
                "type": ["array", "string", "null"],
                "items": {"type": "string"},
            },
            "groups": {
                "type": ["array", "string", "null"],
                "items": {"type": "string"},
            },
            "lazy": {"type": "boolean"},
            "sanitize": {"type": "boolean"},
            "include_catalog": {"type": "boolean"},
        },
        "required": ["sequence"],
        "additionalProperties": False,
    },
    "refua_clinical_simulator": {
        "type": "object",
        "properties": {
            "config": {"type": ["object", "null"]},
            "trial_id": {"type": ["string", "null"]},
            "indication": {"type": ["string", "null"]},
            "phase": {"type": ["string", "null"]},
            "objective": {"type": ["string", "null"]},
            "seed": {"type": ["integer", "null"]},
            "replicates": {"type": ["integer", "null"]},
            "include_replicates": {"type": "boolean"},
            "include_workup": {"type": "boolean"},
            "workup_options": {"type": ["object", "null"]},
            "admet_profile": {"type": ["object", "null"]},
            "refua_payload": {"type": ["object", "null"]},
            "apply_refua_payload": {"type": "boolean"},
            "refua_ligand_id": {"type": ["string", "null"]},
            "refua_max_candidate_arms": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    },
    "refua_data_list": {
        "type": "object",
        "properties": {
            "tag": {"type": ["string", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
            "include_usage_notes": {"type": "boolean"},
            "include_urls": {"type": "boolean"},
            "cache_root": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
    "refua_data_fetch": {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "force": {"type": "boolean"},
            "refresh": {"type": "boolean"},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
            "cache_root": {"type": ["string", "null"]},
            "include_metadata": {"type": "boolean"},
        },
        "required": ["dataset_id"],
        "additionalProperties": False,
    },
    "refua_data_materialize": {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "force": {"type": "boolean"},
            "refresh": {"type": "boolean"},
            "chunksize": {"type": "integer", "minimum": 1},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
            "cache_root": {"type": ["string", "null"]},
            "include_manifest": {"type": "boolean"},
        },
        "required": ["dataset_id"],
        "additionalProperties": False,
    },
    "refua_data_query": {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "filters": {"type": ["object", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
            "cache_root": {"type": ["string", "null"]},
            "materialize_if_missing": {"type": "boolean"},
            "force_materialize": {"type": "boolean"},
            "refresh": {"type": "boolean"},
            "chunksize": {"type": "integer", "minimum": 1},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["dataset_id"],
        "additionalProperties": False,
    },
    "refua_job": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "include_result": {"type": "boolean"},
            "wait_for_terminal_seconds": {"type": ["number", "null"]},
            "cancel": {"type": "boolean"},
        },
        "required": ["job_id"],
        "additionalProperties": False,
    },
    "refua_admet_profile": {
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "model_variant": {"type": "string"},
            "max_new_tokens": {"type": "integer", "minimum": 1},
            "include_scoring": {"type": "boolean"},
            "task_ids": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": ["smiles"],
        "additionalProperties": False,
    },
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "q": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": _MAX_SEARCH_COUNT},
        },
        "anyOf": [{"required": ["query"]}, {"required": ["q"]}],
        "additionalProperties": False,
    },
    "web_fetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "extract_mode": {"type": "string", "enum": ["markdown", "text"]},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": _MAX_FETCH_CHARS},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}
_OPENCLAW_TOOL_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "refua_validate_spec": (
        "Validate and normalize a typed Refua request before expensive inference."
    ),
    "refua_fold": "Run Refua fold/design workflows with typed entities and constraints.",
    "refua_affinity": "Run affinity-focused Refua predictions for typed entities.",
    "refua_antibody_design": "Run antibody-focused design/fold workflows with context.",
    "refua_protein_properties": "Compute protein property descriptors for a sequence.",
    "refua_clinical_simulator": (
        "Simulate trial outcomes and optional workups from trial configuration."
    ),
    "refua_data_list": "List datasets in the refua-data catalog.",
    "refua_data_fetch": "Fetch one refua-data dataset to local cache.",
    "refua_data_materialize": (
        "Materialize one refua-data dataset to parquet with manifest output."
    ),
    "refua_data_query": "Query rows from a materialized refua-data dataset.",
    "refua_job": "Poll, wait for, or cancel an async Refua job by job_id.",
    "refua_admet_profile": "Predict ADMET profile from a SMILES string.",
    "web_search": "Search public web sources for evidence relevant to a biomedical query.",
    "web_fetch": "Fetch and extract text/markdown from a specific public URL.",
}
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "refua_validate_spec",
        "refua_protein_properties",
        "refua_data_list",
        "refua_data_query",
        "web_search",
        "web_fetch",
    }
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

    if not tool_map:
        for name in DEFAULT_TOOL_LIST:
            fn = getattr(server, name, None)
            if callable(fn):
                tool_map[name] = fn

    refua_tools = {name for name in tool_map if name.startswith("refua_")}
    if not refua_tools:
        raise RuntimeError("No executable refua-mcp tools were discovered.")

    tool_map.update(_local_tool_map())
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

    def is_parallel_safe_tool(self, tool: str) -> bool:
        return tool in _PARALLEL_SAFE_TOOLS

    def parallel_safe_tools(self) -> list[str]:
        return sorted(name for name in self.available_tools() if name in _PARALLEL_SAFE_TOOLS)

    def execute_tools_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        max_workers: int = 4,
        fail_fast: bool = False,
    ) -> list[ToolExecutionResult]:
        if not calls:
            return []

        normalized_calls: list[tuple[str, dict[str, Any]]] = []
        for idx, (tool, args) in enumerate(calls):
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError(f"Call #{idx + 1} tool must be a non-empty string.")
            if not isinstance(args, dict):
                raise ValueError(f"Call #{idx + 1} args must be an object.")
            normalized_calls.append((tool.strip(), dict(args)))

        workers = max(1, int(max_workers))
        workers = min(workers, len(normalized_calls))
        ordered_results: list[ToolExecutionResult | None] = [None] * len(normalized_calls)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(self.execute_tool, tool, args): idx
                for idx, (tool, args) in enumerate(normalized_calls)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                tool, args = normalized_calls[idx]
                try:
                    ordered_results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    if fail_fast:
                        raise
                    ordered_results[idx] = ToolExecutionResult(
                        tool=tool,
                        args=args,
                        output={
                            "error": str(exc),
                            "failed_tool": tool,
                            "recoverable": True,
                        },
                    )

        return [item for item in ordered_results if item is not None]

    def openclaw_tool_schemas(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for name in self.available_tools():
            schema = _OPENCLAW_TOOL_SCHEMA_OVERRIDES.get(name, _OPENCLAW_DEFAULT_SCHEMA)
            description = _OPENCLAW_TOOL_DESCRIPTION_OVERRIDES.get(
                name,
                f"Execute {name} with typed arguments.",
            )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": schema,
                    },
                }
            )
        return tools

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


def _local_tool_map() -> dict[str, Callable[..., Any]]:
    return {
        "web_search": _web_search,
        "web_fetch": _web_fetch,
    }


def _web_search(
    *,
    query: str | None = None,
    q: str | None = None,
    count: int = _DEFAULT_SEARCH_COUNT,
    **_extras: Any,
) -> dict[str, Any]:
    query_value = (query or q or "").strip()
    if not query_value:
        raise ValueError("web_search requires a non-empty 'query'.")

    count_value = _normalize_count(count)
    brave_key = (
        os.getenv("BRAVE_API_KEY", "").strip()
        or os.getenv("TOOLS_WEB_SEARCH_API_KEY", "").strip()
    )
    if brave_key:
        return _web_search_brave(query=query_value, count=count_value, api_key=brave_key)

    instant = _web_search_duckduckgo(query=query_value, count=count_value)
    if _has_web_results(instant):
        return instant
    return _web_search_duckduckgo_html(query=query_value, count=count_value)


def _web_fetch(
    *,
    url: str | None = None,
    extract_mode: str = "markdown",
    extractMode: str | None = None,
    max_chars: int | None = None,
    maxChars: int | None = None,
    **_extras: Any,
) -> dict[str, Any]:
    url_value = (url or "").strip()
    if not url_value:
        raise ValueError("web_fetch requires a non-empty 'url'.")
    _validate_fetch_url(url_value)

    mode = (extractMode or extract_mode).strip().lower()
    if mode not in {"markdown", "text"}:
        raise ValueError("web_fetch extract_mode must be 'markdown' or 'text'.")

    max_chars_value = max_chars if max_chars is not None else maxChars
    char_limit = _normalize_max_chars(max_chars_value)
    raw_text, content_type, status_code = _http_get(url_value)

    extracted_text = (
        _html_to_text(raw_text)
        if "html" in content_type.lower() or "<html" in raw_text[:500].lower()
        else raw_text
    )
    if mode == "markdown":
        extracted_text = (
            f"# Source\n\n- URL: {url_value}\n\n# Extracted Content\n\n{extracted_text}"
        )

    trimmed_text = extracted_text[:char_limit]
    return {
        "provider": "builtin",
        "url": url_value,
        "status_code": status_code,
        "content_type": content_type,
        "extract_mode": mode,
        "truncated": len(trimmed_text) < len(extracted_text),
        "char_count": len(trimmed_text),
        "text": trimmed_text,
    }


def _web_search_brave(*, query: str, count: int, api_key: str) -> dict[str, Any]:
    url = (
        "https://api.search.brave.com/res/v1/web/search"
        f"?q={quote_plus(query)}&count={count}"
    )
    payload = _http_get_json(url, headers={"X-Subscription-Token": api_key})
    web_block = payload.get("web", {})
    raw_results = web_block.get("results", []) if isinstance(web_block, dict) else []

    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        item_url = str(item.get("url") or "").strip()
        if not item_url:
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": item_url,
                "snippet": str(item.get("description") or "").strip(),
            }
        )
        if len(results) >= count:
            break

    return {
        "provider": "brave",
        "query": query,
        "requested_count": count,
        "count": len(results),
        "results": results,
    }


def _web_search_duckduckgo(*, query: str, count: int) -> dict[str, Any]:
    url = (
        "https://api.duckduckgo.com/"
        f"?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=0"
    )
    payload = _http_get_json(url, headers={})

    results: list[dict[str, str]] = []
    abstract_url = str(payload.get("AbstractURL") or "").strip()
    abstract_text = str(payload.get("AbstractText") or "").strip()
    heading = str(payload.get("Heading") or "").strip()
    if abstract_url:
        results.append(
            {
                "title": heading or abstract_url,
                "url": abstract_url,
                "snippet": abstract_text,
            }
        )

    related = payload.get("RelatedTopics", [])
    if isinstance(related, list):
        _append_duckduckgo_related_results(related, results, max_results=count)

    if not results:
        results.append(
            {
                "title": "No direct result returned",
                "url": "",
                "snippet": "DuckDuckGo Instant Answer returned no structured results.",
            }
        )

    return {
        "provider": "duckduckgo_instant_answer",
        "query": query,
        "requested_count": count,
        "count": min(len(results), count),
        "results": results[:count],
        "warning": (
            "BRAVE_API_KEY not configured; using DuckDuckGo Instant Answer fallback."
        ),
    }


def _web_search_duckduckgo_html(*, query: str, count: int) -> dict[str, Any]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    html_body, _content_type, _status_code = _http_get(url, headers={"Accept": "text/html"})
    results = _parse_duckduckgo_html_results(html_body, count=count)

    if not results:
        return {
            "provider": "duckduckgo_html",
            "query": query,
            "requested_count": count,
            "count": 0,
            "results": [],
            "warning": (
                "BRAVE_API_KEY not configured and DuckDuckGo HTML search yielded no "
                "parseable results."
            ),
        }

    return {
        "provider": "duckduckgo_html",
        "query": query,
        "requested_count": count,
        "count": len(results),
        "results": results,
        "warning": "BRAVE_API_KEY not configured; using DuckDuckGo HTML fallback.",
    }


def _append_duckduckgo_related_results(
    related: list[Any],
    out: list[dict[str, str]],
    *,
    max_results: int,
) -> None:
    for entry in related:
        if len(out) >= max_results:
            return
        if not isinstance(entry, dict):
            continue
        nested_topics = entry.get("Topics")
        if isinstance(nested_topics, list):
            _append_duckduckgo_related_results(nested_topics, out, max_results=max_results)
            continue

        first_url = str(entry.get("FirstURL") or "").strip()
        text = str(entry.get("Text") or "").strip()
        if not first_url and not text:
            continue
        title = text.split(" - ", maxsplit=1)[0] if text else first_url
        out.append(
            {
                "title": title.strip(),
                "url": first_url,
                "snippet": text,
            }
        )


def _parse_duckduckgo_html_results(value: str, *, count: int) -> list[dict[str, str]]:
    title_matches = re.findall(
        r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>',
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_matches = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    results: list[dict[str, str]] = []
    for idx, (href, raw_title) in enumerate(title_matches):
        parsed_url = _decode_duckduckgo_redirect_url(href)
        title = _normalize_html_fragment(raw_title)
        snippet = _normalize_html_fragment(snippet_matches[idx]) if idx < len(snippet_matches) else ""
        if not parsed_url:
            continue
        if not title:
            title = parsed_url
        results.append(
            {
                "title": title,
                "url": parsed_url,
                "snippet": snippet,
            }
        )
        if len(results) >= count:
            break
    return results


def _normalize_html_fragment(value: str) -> str:
    no_tags = re.sub(r"(?is)<[^>]+>", " ", value)
    unescaped = html.unescape(no_tags)
    compact = re.sub(r"\s+", " ", unescaped).strip()
    return compact


def _decode_duckduckgo_redirect_url(value: str) -> str:
    href = html.unescape(value).strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = urljoin("https://duckduckgo.com", href)

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg_values = parse_qs(parsed.query).get("uddg", [])
        if uddg_values:
            decoded = unquote(uddg_values[0]).strip()
            return decoded
    return href


def _has_web_results(payload: dict[str, Any]) -> bool:
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url and title and "no direct result returned" not in title.lower():
            return True
    return False


def _http_get_json(url: str, *, headers: dict[str, str]) -> dict[str, Any]:
    body, _content_type, _status_code = _http_get(url, headers=headers)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON response from {url}.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected non-object JSON response from {url}.")
    return payload


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[str, str, int]:
    merged_headers = {"Accept": "*/*", "User-Agent": _HTTP_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    request = Request(url, headers=merged_headers, method="GET")
    try:
        with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            content_bytes = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
            status_code = int(response.getcode() or 200)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc.reason}") from exc

    return content_bytes.decode("utf-8", errors="replace"), content_type, status_code


def _validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("web_fetch only supports http/https URLs.")
    if not parsed.netloc:
        raise ValueError("web_fetch requires a fully-qualified URL.")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("web_fetch requires a hostname.")
    if _is_private_fetch_target(hostname) and not _allow_private_fetch():
        raise ValueError(
            "web_fetch blocks localhost/private-network targets by default. "
            f"Set {_ALLOW_PRIVATE_FETCH_ENV}=true to override."
        )


def _allow_private_fetch() -> bool:
    value = os.getenv(_ALLOW_PRIVATE_FETCH_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_private_fetch_target(hostname: str) -> bool:
    if hostname in {"localhost", "localhost.localdomain"}:
        return True
    if hostname.endswith(".local"):
        return True

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
    )


def _html_to_text(value: str) -> str:
    stripped = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", value)
    stripped = re.sub(r"(?is)<br\s*/?>", "\n", stripped)
    stripped = re.sub(
        r"(?is)</(p|div|h[1-6]|li|tr|section|article|main|header|footer)>",
        "\n",
        stripped,
    )
    stripped = re.sub(r"(?is)<[^>]+>", " ", stripped)
    unescaped = html.unescape(stripped)
    lines = [re.sub(r"\s+", " ", line).strip() for line in unescaped.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact.strip()


def _normalize_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("web_search 'count' must be an integer.") from exc
    if count < 1:
        return 1
    if count > _MAX_SEARCH_COUNT:
        return _MAX_SEARCH_COUNT
    return count


def _normalize_max_chars(value: Any) -> int:
    if value is None:
        return _DEFAULT_MAX_FETCH_CHARS
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("web_fetch 'max_chars' must be an integer.") from exc
    if parsed < 1:
        raise ValueError("web_fetch 'max_chars' must be >= 1.")
    if parsed > _MAX_FETCH_CHARS:
        return _MAX_FETCH_CHARS
    return parsed


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
