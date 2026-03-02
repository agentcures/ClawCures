from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any

_TARGET_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9\-]{2,11}\b")
_TARGET_CONTEXT_HINTS: tuple[str, ...] = (
    "target",
    "targeted",
    "mutation",
    "inhibitor",
    "agonist",
    "antagonist",
    "pathway",
    "receptor",
    "kinase",
    "enzyme",
    "druggable",
    "therapeutic",
    "therapy",
)
_TARGET_STOPWORDS: set[str] = {
    "AND",
    "THE",
    "WITH",
    "THIS",
    "FROM",
    "FOR",
    "THAT",
    "WERE",
    "WILL",
    "HAVE",
    "HAS",
    "CAN",
    "MAY",
    "DNA",
    "RNA",
    "ATP",
    "WHO",
    "NIH",
    "FDA",
    "USA",
    "PMID",
    "PMC",
    "COVID",
    "HIV",
    "TB",
    "COPD",
    "CKD",
    "NASH",
    "MASH",
    "NSCLC",
    "ALS",
    "CVD",
}
_DISEASE_HINTS: tuple[tuple[str, str], ...] = (
    ("alzheimer", "alzheimer disease"),
    ("parkinson", "parkinson disease"),
    ("amyotrophic lateral sclerosis", "amyotrophic lateral sclerosis"),
    ("als", "amyotrophic lateral sclerosis"),
    ("ischemic heart", "ischemic heart disease"),
    ("heart failure", "heart failure"),
    ("stroke", "stroke"),
    ("lung cancer", "lung cancer"),
    ("nsclc", "lung cancer"),
    ("colorectal cancer", "colorectal cancer"),
    ("breast cancer", "breast cancer"),
    ("pancreatic cancer", "pancreatic cancer"),
    ("liver cancer", "liver cancer"),
    ("tuberculosis", "tuberculosis"),
    ("malaria", "malaria"),
    ("hiv", "hiv"),
    ("type 2 diabetes", "type 2 diabetes"),
    ("diabetes", "type 2 diabetes"),
    ("obesity", "obesity"),
    ("chronic kidney disease", "chronic kidney disease"),
    ("ckd", "chronic kidney disease"),
    ("copd", "copd"),
    ("asthma", "asthma"),
    ("pulmonary fibrosis", "pulmonary fibrosis"),
)


@dataclass
class _TargetEvidence:
    target: str
    disease: str | None
    mentions: int = 0
    context_hits: int = 0
    source_paths: set[str] = field(default_factory=set)
    source_urls: set[str] = field(default_factory=set)
    query_hints: set[str] = field(default_factory=set)
    tool_sources: set[str] = field(default_factory=set)


def extract_interesting_targets(
    results: list[Any],
    *,
    min_score: float = 30.0,
    max_targets: int = 20,
) -> list[dict[str, Any]]:
    evidence: dict[tuple[str | None, str], _TargetEvidence] = {}
    for item in results:
        tool, args, output = _result_parts(item)
        if tool == "web_search":
            _ingest_web_search(evidence, args=args, output=output)
        elif tool == "web_fetch":
            _ingest_web_fetch(evidence, args=args, output=output)

    discovered: list[dict[str, Any]] = []
    for key in sorted(evidence, key=lambda value: (value[0] or "", value[1])):
        item = evidence[key]
        score = _score_target(item)
        if score < float(min_score):
            continue
        source_count = len(item.source_paths)
        rationale = (
            f"Observed across {source_count} source block(s) with "
            f"{item.mentions} mention(s) and {item.context_hits} therapeutic context hit(s)."
        )
        discovered.append(
            {
                "target": item.target,
                "disease": item.disease,
                "score": score,
                "mentions": item.mentions,
                "context_hits": item.context_hits,
                "source_count": source_count,
                "source_urls": sorted(item.source_urls),
                "evidence_paths": sorted(item.source_paths),
                "query_hints": sorted(item.query_hints),
                "tool_sources": sorted(item.tool_sources),
                "rationale": rationale,
            }
        )

    discovered.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["mentions"]),
            int(item["context_hits"]),
        ),
        reverse=True,
    )
    return discovered[: max(1, int(max_targets))]


def summarize_interesting_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    disease_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        disease = target.get("disease")
        if isinstance(disease, str) and disease.strip():
            disease_counts[disease.strip()] += 1
    return {
        "total_targets": len(targets),
        "disease_counts": dict(sorted(disease_counts.items())),
        "top_targets": [item.get("target") for item in targets[:5] if item.get("target")],
    }


def _ingest_web_search(
    evidence: dict[tuple[str | None, str], _TargetEvidence],
    *,
    args: dict[str, Any],
    output: Any,
) -> None:
    output_map = output if isinstance(output, Mapping) else {}
    query = str(
        output_map.get("query") or args.get("query") or args.get("q") or ""
    ).strip()
    disease_hint = _infer_disease_hint(query)

    if query:
        query_scan = _scan_target_mentions(query)
        _record_scan(
            evidence,
            scan=query_scan,
            disease_hint=disease_hint,
            source_path="output.query",
            source_url="",
            query_hint=query,
            tool_source="web_search",
        )

    raw_results = output_map.get("results")
    if not isinstance(raw_results, list):
        return

    for idx, entry in enumerate(raw_results):
        if not isinstance(entry, Mapping):
            continue
        title = str(entry.get("title") or "").strip()
        snippet = str(entry.get("snippet") or "").strip()
        url = str(entry.get("url") or "").strip()
        block = " ".join(item for item in (title, snippet) if item).strip()
        if not block:
            continue
        scan = _scan_target_mentions(block)
        if not scan:
            continue
        _record_scan(
            evidence,
            scan=scan,
            disease_hint=disease_hint,
            source_path=f"output.results[{idx}]",
            source_url=url,
            query_hint=query,
            tool_source="web_search",
        )


def _ingest_web_fetch(
    evidence: dict[tuple[str | None, str], _TargetEvidence],
    *,
    args: dict[str, Any],
    output: Any,
) -> None:
    output_map = output if isinstance(output, Mapping) else {}
    source_url = str(output_map.get("url") or args.get("url") or "").strip()
    text = str(output_map.get("text") or "").strip()
    if not text:
        return

    # Bound parsing cost for long pages.
    parse_text = text[:120_000]
    disease_hint = _infer_disease_hint(f"{source_url} {parse_text[:1500]}")
    scan = _scan_target_mentions(parse_text)
    if not scan:
        return

    _record_scan(
        evidence,
        scan=scan,
        disease_hint=disease_hint,
        source_path="output.text",
        source_url=source_url,
        query_hint="",
        tool_source="web_fetch",
    )


def _record_scan(
    evidence: dict[tuple[str | None, str], _TargetEvidence],
    *,
    scan: dict[str, tuple[int, int]],
    disease_hint: str | None,
    source_path: str,
    source_url: str,
    query_hint: str,
    tool_source: str,
) -> None:
    for target, (mentions, context_hits) in scan.items():
        key = (disease_hint, target)
        item = evidence.get(key)
        if item is None:
            item = _TargetEvidence(target=target, disease=disease_hint)
            evidence[key] = item
        item.mentions += mentions
        item.context_hits += context_hits
        item.source_paths.add(source_path)
        if source_url:
            item.source_urls.add(source_url)
        if query_hint:
            item.query_hints.add(query_hint)
        item.tool_sources.add(tool_source)


def _scan_target_mentions(value: str) -> dict[str, tuple[int, int]]:
    mentions: dict[str, int] = defaultdict(int)
    context_hits: dict[str, int] = defaultdict(int)
    lowered = value.lower()
    for match in _TARGET_TOKEN_RE.finditer(value):
        token = match.group(0).upper()
        if not _looks_like_target(token):
            continue
        mentions[token] += 1
        start = max(0, match.start() - 70)
        end = min(len(value), match.end() + 70)
        window = lowered[start:end]
        if any(hint in window for hint in _TARGET_CONTEXT_HINTS):
            context_hits[token] += 1

    return {token: (count, context_hits.get(token, 0)) for token, count in mentions.items()}


def _looks_like_target(token: str) -> bool:
    if token in _TARGET_STOPWORDS:
        return False
    if token.isdigit():
        return False
    if len(token) <= 2:
        return False
    if token.startswith("HTTP"):
        return False
    return True


def _score_target(item: _TargetEvidence) -> float:
    score = 10.0
    score += min(item.mentions, 10) * 3.5
    score += min(len(item.source_paths), 6) * 5.0
    score += min(item.context_hits, 8) * 4.0
    if len(item.tool_sources) > 1:
        score += 6.0
    if item.context_hits == 0:
        score *= 0.6
    return round(min(score, 99.0), 2)


def _infer_disease_hint(value: str) -> str | None:
    lowered = value.lower()
    for pattern, label in _DISEASE_HINTS:
        if pattern in lowered:
            return label
    return None


def _result_parts(item: Any) -> tuple[str, dict[str, Any], Any]:
    if isinstance(item, Mapping):
        tool = str(item.get("tool") or "unknown_tool")
        args = item.get("args")
        output = item.get("output")
        return tool, dict(args) if isinstance(args, Mapping) else {}, output
    tool = str(getattr(item, "tool", "unknown_tool"))
    args_raw = getattr(item, "args", {})
    output = getattr(item, "output", None)
    return tool, dict(args_raw) if isinstance(args_raw, Mapping) else {}, output
