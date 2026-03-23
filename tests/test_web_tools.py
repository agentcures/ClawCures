from __future__ import annotations

import pytest

import refua_campaign.refua_mcp_adapter as adapter
from refua_campaign.refua_mcp_adapter import _validate_fetch_url


def test_validate_fetch_url_accepts_public_https() -> None:
    _validate_fetch_url("https://example.org/path")


def test_validate_fetch_url_rejects_localhost_by_default() -> None:
    with pytest.raises(ValueError, match="blocks localhost/private-network targets"):
        _validate_fetch_url("http://localhost:8000/health")


def test_validate_fetch_url_rejects_private_ip_by_default() -> None:
    with pytest.raises(ValueError, match="blocks localhost/private-network targets"):
        _validate_fetch_url("http://127.0.0.1:8080/status")


def test_validate_fetch_url_allows_private_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAWCURES_ALLOW_PRIVATE_WEB_FETCH", "true")
    _validate_fetch_url("http://127.0.0.1:8080/status")


def test_web_search_falls_back_to_duckduckgo_html_when_instant_answer_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TOOLS_WEB_SEARCH_API_KEY", raising=False)

    monkeypatch.setattr(
        adapter,
        "_web_search_duckduckgo",
        lambda *, query, count: {
            "provider": "duckduckgo_instant_answer",
            "query": query,
            "requested_count": count,
            "count": 1,
            "results": [
                {
                    "title": "No direct result returned",
                    "url": "",
                    "snippet": "DuckDuckGo Instant Answer returned no structured results.",
                }
            ],
        },
    )
    monkeypatch.setattr(
        adapter,
        "_web_search_duckduckgo_html",
        lambda *, query, count: {
            "provider": "duckduckgo_html",
            "query": query,
            "requested_count": count,
            "count": 1,
            "results": [
                {
                    "title": "EGFR target overview",
                    "url": "https://example.org/egfr",
                    "snippet": "EGFR is a validated therapeutic target in lung cancer.",
                }
            ],
        },
    )

    payload = adapter._web_search(query="lung cancer targets", count=3)
    assert payload["provider"] == "duckduckgo_html"
    assert payload["results"][0]["url"] == "https://example.org/egfr"


def test_web_search_falls_back_to_duckduckgo_html_when_instant_answer_is_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TOOLS_WEB_SEARCH_API_KEY", raising=False)

    def _raise_invalid_json(*, query: str, count: int) -> dict[str, object]:
        raise RuntimeError(
            "Expected JSON response from https://api.duckduckgo.com/?q=test."
        )

    monkeypatch.setattr(adapter, "_web_search_duckduckgo", _raise_invalid_json)
    monkeypatch.setattr(
        adapter,
        "_web_search_duckduckgo_html",
        lambda *, query, count: {
            "provider": "duckduckgo_html",
            "query": query,
            "requested_count": count,
            "count": 1,
            "results": [
                {
                    "title": "PCSK9 review",
                    "url": "https://example.org/pcsk9",
                    "snippet": "PCSK9 is a validated lipid-lowering target.",
                }
            ],
            "warning": "BRAVE_API_KEY not configured; using DuckDuckGo HTML fallback.",
        },
    )

    payload = adapter._web_search(query="ischemic heart disease targets", count=3)
    assert payload["provider"] == "duckduckgo_html"
    assert payload["results"][0]["url"] == "https://example.org/pcsk9"
    assert "Instant Answer failed" in payload["warning"]


def test_parse_duckduckgo_html_results_decodes_redirect_urls() -> None:
    html_payload = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpaper&amp;rut=abc">
      EGFR and KRAS in NSCLC
    </a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpaper&amp;rut=abc">
      EGFR and KRAS are actionable mutation targets in NSCLC.
    </a>
    """
    parsed = adapter._parse_duckduckgo_html_results(html_payload, count=5)
    assert len(parsed) == 1
    assert parsed[0]["url"] == "https://example.org/paper"
    assert "EGFR" in parsed[0]["title"]
