from __future__ import annotations

from refua_campaign.target_discovery import (
    extract_interesting_targets,
    summarize_interesting_targets,
)


def test_extract_interesting_targets_from_web_search_results() -> None:
    results = [
        {
            "tool": "web_search",
            "args": {
                "query": "NSCLC actionable therapeutic targets EGFR ALK KRAS MET review",
                "count": 5,
            },
            "output": {
                "provider": "duckduckgo_html",
                "query": "NSCLC actionable therapeutic targets EGFR ALK KRAS MET review",
                "results": [
                    {
                        "title": "New Actions on Actionable Mutations in Lung Cancers",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10252025/",
                        "snippet": (
                            "In NSCLC, EGFR, ALK, MET, BRAF, and KRAS carry targetable "
                            "mutations with targeted inhibitor therapies."
                        ),
                    }
                ],
            },
        }
    ]

    targets = extract_interesting_targets(results, min_score=10.0)
    names = {item["target"] for item in targets}
    assert "EGFR" in names
    assert "KRAS" in names
    assert any((item.get("disease") or "") == "lung cancer" for item in targets)


def test_extract_interesting_targets_from_web_fetch_text() -> None:
    results = [
        {
            "tool": "web_fetch",
            "args": {"url": "https://example.org/alzheimer-review"},
            "output": {
                "url": "https://example.org/alzheimer-review",
                "text": (
                    "Alzheimer disease therapeutic targets include APP, MAPT, and TREM2. "
                    "TREM2 agonist programs and MAPT-directed therapies are under study."
                ),
            },
        }
    ]

    targets = extract_interesting_targets(results, min_score=10.0)
    names = {item["target"] for item in targets}
    assert {"APP", "MAPT", "TREM2"}.issubset(names)

    summary = summarize_interesting_targets(targets)
    assert summary["total_targets"] >= 3
    assert "alzheimer disease" in summary["disease_counts"]
