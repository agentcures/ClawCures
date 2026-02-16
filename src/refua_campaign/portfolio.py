from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PortfolioWeights:
    burden: float = 0.35
    tractability: float = 0.25
    unmet_need: float = 0.20
    translational_readiness: float = 0.10
    novelty: float = 0.10


@dataclass(frozen=True)
class RankedDisease:
    name: str
    score: float
    rationale: tuple[str, ...]
    raw: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "rationale": list(self.rationale),
            "raw": self.raw,
        }


def rank_disease_programs(
    diseases: list[dict[str, Any]],
    *,
    weights: PortfolioWeights,
) -> list[RankedDisease]:
    ranked: list[RankedDisease] = []
    for item in diseases:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("disease") or "unknown")
        burden = _bounded_score(item.get("burden"))
        tractability = _bounded_score(item.get("tractability"))
        unmet_need = _bounded_score(item.get("unmet_need"))
        translational = _bounded_score(item.get("translational_readiness"))
        novelty = _bounded_score(item.get("novelty"))

        score = (
            weights.burden * burden
            + weights.tractability * tractability
            + weights.unmet_need * unmet_need
            + weights.translational_readiness * translational
            + weights.novelty * novelty
        )

        rationale = (
            f"burden={burden:.3f}",
            f"tractability={tractability:.3f}",
            f"unmet_need={unmet_need:.3f}",
            f"translational_readiness={translational:.3f}",
            f"novelty={novelty:.3f}",
        )
        ranked.append(
            RankedDisease(
                name=name,
                score=round(score, 6),
                rationale=rationale,
                raw=item,
            )
        )
    ranked.sort(key=lambda entry: entry.score, reverse=True)
    return ranked


def _bounded_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric
