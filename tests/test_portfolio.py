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
