from __future__ import annotations

from refua_campaign.evidence_quality import summarize_evidence_quality


def test_summarize_evidence_quality_counts_sources_and_domains() -> None:
    summary = summarize_evidence_quality(
        results=[
            {
                "tool": "web_search",
                "args": {"query": "EGFR"},
                "output": {
                    "results": [
                        {
                            "title": "EGFR review",
                            "url": "https://example.org/review",
                        },
                        {
                            "title": "KRAS review",
                            "url": "https://example.net/kras",
                        },
                    ]
                },
            },
            {
                "tool": "web_fetch",
                "args": {"url": "https://example.org/review"},
                "output": {"url": "https://example.org/review", "text": "EGFR text"},
            },
        ],
        interesting_targets=[
            {
                "target": "EGFR",
                "disease": "lung cancer",
                "score": 70.0,
                "source_count": 1,
                "source_urls": ["https://example.org/review"],
            }
        ],
        promising_cures=[
            {
                "cure_id": "c1",
                "target": "EGFR",
                "promising": True,
            }
        ],
    )
    assert summary["citation_count"] >= 2
    assert summary["unique_source_urls"] >= 2
    assert summary["unique_domains"] >= 2
    assert summary["quality_score"] > 0
    assert summary["promising_candidate_target_coverage"] == 1.0
