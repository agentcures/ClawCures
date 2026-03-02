from __future__ import annotations

from refua_campaign.refua_mcp_adapter import ToolExecutionResult
from refua_campaign.web_evidence import (
    derive_auto_web_fetch_calls,
    expand_results_with_web_fetch,
)


def test_derive_auto_web_fetch_calls_skips_non_http_and_existing_urls() -> None:
    results = [
        ToolExecutionResult(
            tool="web_search",
            args={"query": "targets"},
            output={
                "results": [
                    {"url": "https://example.org/a"},
                    {"url": "ftp://example.org/b"},
                    {"url": "https://example.org/a"},
                ]
            },
        ),
        ToolExecutionResult(
            tool="web_fetch",
            args={"url": "https://example.org/already"},
            output={},
        ),
        ToolExecutionResult(
            tool="web_search",
            args={"query": "more targets"},
            output={
                "results": [
                    {"url": "https://example.org/already"},
                    {"url": "https://example.org/c"},
                ]
            },
        ),
    ]

    calls = derive_auto_web_fetch_calls(results=results, max_urls=5, max_chars=1234)
    assert calls == [
        {
            "url": "https://example.org/a",
            "extract_mode": "text",
            "max_chars": 1234,
        },
        {
            "url": "https://example.org/c",
            "extract_mode": "text",
            "max_chars": 1234,
        },
    ]


def test_expand_results_with_web_fetch_appends_generated_results() -> None:
    base_results = [
        ToolExecutionResult(
            tool="web_search",
            args={"query": "EGFR target"},
            output={"results": [{"url": "https://example.org/egfr"}]},
        )
    ]

    def _fake_execute(tool: str, args: dict[str, object]) -> ToolExecutionResult:
        assert tool == "web_fetch"
        return ToolExecutionResult(tool=tool, args=args, output={"url": args["url"]})

    expanded, generated = expand_results_with_web_fetch(
        results=base_results,
        execute_tool=_fake_execute,
        max_urls=3,
        max_chars=2000,
    )
    assert generated == 1
    assert len(expanded) == 2
    assert expanded[1].tool == "web_fetch"
    assert expanded[1].args["max_chars"] == 2000
