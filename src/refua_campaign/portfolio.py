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
    expected_value: float | None = None
    allocation_fraction: float | None = None
    recommended_budget: float | None = None
    decision: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "score": self.score,
            "rationale": list(self.rationale),
            "raw": self.raw,
        }
        if self.expected_value is not None:
            payload["expected_value"] = self.expected_value
        if self.allocation_fraction is not None:
            payload["allocation_fraction"] = self.allocation_fraction
        if self.recommended_budget is not None:
            payload["recommended_budget"] = self.recommended_budget
        if self.decision is not None:
            payload["decision"] = self.decision
        return payload


def rank_disease_programs(
    diseases: list[dict[str, Any]],
    *,
    weights: PortfolioWeights,
    total_budget: float | None = None,
    voi_weight: float = 0.15,
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
        voi = _bounded_score(item.get("voi", item.get("value_of_information")))

        score = (
            weights.burden * burden
            + weights.tractability * tractability
            + weights.unmet_need * unmet_need
            + weights.translational_readiness * translational
            + weights.novelty * novelty
        )
        expected_value = score * (1.0 + max(0.0, float(voi_weight)) * voi)

        rationale = (
            f"burden={burden:.3f}",
            f"tractability={tractability:.3f}",
            f"unmet_need={unmet_need:.3f}",
            f"translational_readiness={translational:.3f}",
            f"novelty={novelty:.3f}",
            f"voi={voi:.3f}",
        )
        ranked.append(
            RankedDisease(
                name=name,
                score=round(score, 6),
                rationale=rationale,
                raw=item,
                expected_value=round(expected_value, 6),
            )
        )
    ranked.sort(
        key=lambda entry: (
            float(entry.expected_value or 0.0),
            entry.score,
        ),
        reverse=True,
    )

    if total_budget is None:
        return [
            _with_decision(item, recommended_budget=None, allocation_fraction=None)
            for item in ranked
        ]

    budget_value = max(0.0, float(total_budget))
    ev_sum = sum(max(float(item.expected_value or 0.0), 0.0) for item in ranked)
    if ev_sum <= 0.0:
        return [
            _with_decision(item, recommended_budget=0.0, allocation_fraction=0.0)
            for item in ranked
        ]

    allocated: list[RankedDisease] = []
    for item in ranked:
        expected = max(float(item.expected_value or 0.0), 0.0)
        fraction = expected / ev_sum
        allocated.append(
            _with_decision(
                item,
                recommended_budget=round(budget_value * fraction, 4),
                allocation_fraction=round(fraction, 6),
            )
        )
    return allocated


def _with_decision(
    item: RankedDisease,
    *,
    recommended_budget: float | None,
    allocation_fraction: float | None,
) -> RankedDisease:
    decision = "advance" if float(item.score) >= 0.45 else "watch"
    return RankedDisease(
        name=item.name,
        score=item.score,
        rationale=item.rationale,
        raw=item.raw,
        expected_value=item.expected_value,
        allocation_fraction=allocation_fraction,
        recommended_budget=recommended_budget,
        decision=decision,
    )


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
