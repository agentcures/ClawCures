from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


def summarize_evidence_quality(
    *,
    results: list[Mapping[str, Any]],
    interesting_targets: list[Mapping[str, Any]],
    promising_cures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    citations = _collect_citations(results, interesting_targets)
    citation_urls = sorted({item["url"] for item in citations if item.get("url")})
    citation_domains = sorted({_domain_from_url(url) for url in citation_urls if _domain_from_url(url)})

    source_counter = Counter(item.get("tool", "unknown") for item in citations)
    unsupported_targets = [
        {
            "target": item.get("target"),
            "disease": item.get("disease"),
            "score": item.get("score"),
        }
        for item in interesting_targets
        if int(item.get("source_count") or 0) == 0
    ]

    promising_total = len(promising_cures)
    promising_with_target = sum(1 for item in promising_cures if item.get("target"))
    target_coverage = 0.0
    if promising_total > 0:
        target_coverage = round(promising_with_target / float(promising_total), 4)

    quality_score = _quality_score(
        unique_urls=len(citation_urls),
        unique_domains=len(citation_domains),
        fetch_count=int(source_counter.get("web_fetch", 0)),
        target_count=len(interesting_targets),
    )

    return {
        "quality_score": quality_score,
        "quality_band": _quality_band(quality_score),
        "citation_count": len(citations),
        "unique_source_urls": len(citation_urls),
        "unique_domains": len(citation_domains),
        "source_tool_counts": dict(source_counter),
        "top_domains": citation_domains[:10],
        "target_count": len(interesting_targets),
        "unsupported_target_count": len(unsupported_targets),
        "unsupported_targets": unsupported_targets[:20],
        "promising_candidate_count": promising_total,
        "promising_candidate_target_coverage": target_coverage,
    }


def _collect_citations(
    results: list[Mapping[str, Any]],
    interesting_targets: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in results:
        tool = str(item.get("tool") or "")
        args = item.get("args")
        output = item.get("output")
        args_map = args if isinstance(args, Mapping) else {}
        output_map = output if isinstance(output, Mapping) else {}

        if tool == "web_fetch":
            url = str(output_map.get("url") or args_map.get("url") or "").strip()
            _append_citation(
                citations,
                seen,
                tool=tool,
                url=url,
                title=str(output_map.get("title") or "").strip(),
            )
            continue

        if tool == "web_search":
            raw_results = output_map.get("results")
            if not isinstance(raw_results, list):
                continue
            for row in raw_results:
                if not isinstance(row, Mapping):
                    continue
                _append_citation(
                    citations,
                    seen,
                    tool=tool,
                    url=str(row.get("url") or "").strip(),
                    title=str(row.get("title") or "").strip(),
                )

    for target in interesting_targets:
        urls = target.get("source_urls")
        if not isinstance(urls, list):
            continue
        for url in urls:
            _append_citation(
                citations,
                seen,
                tool="interesting_targets",
                url=str(url).strip(),
                title=str(target.get("target") or "").strip(),
            )

    return citations


def _append_citation(
    citations: list[dict[str, str]],
    seen: set[tuple[str, str]],
    *,
    tool: str,
    url: str,
    title: str,
) -> None:
    if not url:
        return
    key = (tool, url)
    if key in seen:
        return
    seen.add(key)
    citations.append(
        {
            "tool": tool,
            "url": url,
            "title": title,
        }
    )


def _domain_from_url(url: str) -> str:
    value = url.strip()
    if not value:
        return ""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(value)
    except Exception:
        return ""
    return str(parsed.netloc or "").strip().lower()


def _quality_score(
    *,
    unique_urls: int,
    unique_domains: int,
    fetch_count: int,
    target_count: int,
) -> float:
    score = 15.0
    score += min(unique_urls, 30) * 1.8
    score += min(unique_domains, 15) * 2.5
    score += min(fetch_count, 20) * 1.5
    if target_count > 0 and unique_urls == 0:
        score *= 0.4
    return round(min(score, 100.0), 2)


def _quality_band(score: float) -> str:
    if score >= 75.0:
        return "high"
    if score >= 45.0:
        return "medium"
    return "low"
