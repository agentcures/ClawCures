from __future__ import annotations

from refua_campaign.agent_routing import infer_domain_from_objective, pick_model_for_phase


def test_infer_domain_from_objective_detects_oncology() -> None:
    domain = infer_domain_from_objective(
        "Prioritize cancer programs and identify actionable tumor targets."
    )
    assert domain == "oncology"


def test_pick_model_for_phase_prefers_phase_domain_key() -> None:
    model = pick_model_for_phase(
        phase="plan",
        objective="Build a lung cancer target plan.",
        model_map={
            "planner:oncology": "openclaw:oncology-planner",
            "planner": "openclaw:planner",
            "default": "openclaw:main",
        },
    )
    assert model == "openclaw:oncology-planner"


def test_pick_model_for_phase_falls_back_to_default() -> None:
    model = pick_model_for_phase(
        phase="critic-loop",
        objective="Find cures for all diseases.",
        model_map={"default": "openclaw:main"},
    )
    assert model == "openclaw:main"
