from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from refua_campaign.refua_mcp_adapter import ToolExecutionResult

_DEFAULT_AUTO_FETCH_MAX_URLS = 6
_DEFAULT_AUTO_FETCH_MAX_CHARS = 20_000


def expand_results_with_web_fetch(
    *,
    results: list[ToolExecutionResult],
    execute_tool: Callable[[str, dict[str, Any]], ToolExecutionResult],
    max_urls: int = _DEFAULT_AUTO_FETCH_MAX_URLS,
    max_chars: int = _DEFAULT_AUTO_FETCH_MAX_CHARS,
) -> tuple[list[ToolExecutionResult], int]:
    bounded_max_urls = max(0, int(max_urls))
    if bounded_max_urls == 0:
        return list(results), 0

    fetch_args_list = derive_auto_web_fetch_calls(
        results=results,
        max_urls=bounded_max_urls,
        max_chars=max(1, int(max_chars)),
    )
    if not fetch_args_list:
        return list(results), 0

    expanded = list(results)
    generated = 0
    for args in fetch_args_list:
        try:
            fetch_result = execute_tool("web_fetch", args)
        except Exception as exc:  # noqa: BLE001
            fetch_result = ToolExecutionResult(
                tool="web_fetch",
                args=dict(args),
                output={"error": str(exc), "url": str(args.get("url") or "")},
            )
        expanded.append(fetch_result)
        generated += 1
    return expanded, generated


def derive_auto_web_fetch_calls(
    *,
    results: list[ToolExecutionResult],
    max_urls: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    existing_fetch_urls = _existing_web_fetch_urls(results)
    queued: list[dict[str, Any]] = []
    seen: set[str] = set(existing_fetch_urls)

    for item in results:
        if item.tool != "web_search":
            continue
        output = item.output if isinstance(item.output, Mapping) else {}
        raw_results = output.get("results")
        if not isinstance(raw_results, list):
            continue
        for entry in raw_results:
            if not isinstance(entry, Mapping):
                continue
            url = str(entry.get("url") or "").strip()
            if not _is_public_http_url(url):
                continue
            if url in seen:
                continue
            seen.add(url)
            queued.append(
                {
                    "url": url,
                    "extract_mode": "text",
                    "max_chars": max_chars,
                }
            )
            if len(queued) >= max_urls:
                return queued
    return queued


def _existing_web_fetch_urls(results: list[ToolExecutionResult]) -> set[str]:
    urls: set[str] = set()
    for item in results:
        if item.tool != "web_fetch":
            continue
        args = item.args if isinstance(item.args, Mapping) else {}
        url = str(args.get("url") or "").strip()
        if url:
            urls.add(url)
            continue
        output = item.output if isinstance(item.output, Mapping) else {}
        out_url = str(output.get("url") or "").strip()
        if out_url:
            urls.add(out_url)
    return urls


def _is_public_http_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")
