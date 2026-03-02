from __future__ import annotations

from refua_campaign.portfolio import PortfolioWeights, rank_disease_programs


def test_rank_disease_programs_orders_highest_score_first() -> None:
    ranked = rank_disease_programs(
        [
            {"name": "A", "burden": 0.9, "tractability": 0.2, "unmet_need": 0.9},
            {"name": "B", "burden": 0.7, "tractability": 0.9, "unmet_need": 0.7},
        ],
        weights=PortfolioWeights(),
    )
    assert ranked[0].name in {"A", "B"}
    assert ranked[0].score >= ranked[1].score


def test_rank_disease_programs_bounds_values() -> None:
    ranked = rank_disease_programs(
        [{"name": "bounded", "burden": 5, "tractability": -2, "unmet_need": 0.5}],
        weights=PortfolioWeights(),
    )
    assert len(ranked) == 1
    assert ranked[0].score >= 0.0


def test_rank_disease_programs_allocates_budget_when_requested() -> None:
    ranked = rank_disease_programs(
        [
            {"name": "A", "burden": 0.9, "tractability": 0.5, "unmet_need": 0.8, "voi": 0.8},
            {"name": "B", "burden": 0.6, "tractability": 0.6, "unmet_need": 0.6, "voi": 0.2},
        ],
        weights=PortfolioWeights(),
        total_budget=100.0,
        voi_weight=0.25,
    )
    assert len(ranked) == 2
    assert ranked[0].recommended_budget is not None
    assert ranked[1].recommended_budget is not None
    total = float(ranked[0].recommended_budget or 0.0) + float(
        ranked[1].recommended_budget or 0.0
    )
    assert round(total, 2) == 100.0
