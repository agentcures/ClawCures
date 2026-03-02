from __future__ import annotations

from refua_campaign.translational_handoff import build_translational_handoff


def test_build_translational_handoff_includes_major_domains() -> None:
    payload = build_translational_handoff(
        objective="Find cures for all diseases.",
        interesting_targets=[
            {"disease": "lung cancer", "target": "EGFR", "score": 92.0},
            {"disease": "type 2 diabetes", "target": "GLP1R", "score": 85.0},
        ],
        promising_cures=[
            {
                "cure_id": "c1",
                "name": "candidate-egfr",
                "target": "EGFR",
                "score": 66.0,
                "promising": True,
            }
        ],
        evidence_quality={"quality_band": "medium"},
    )
    assert payload["evidence_quality_band"] == "medium"
    assert payload["priority_targets"]
    assert payload["preclinical_tasks"]
    assert payload["wetlab_tasks"]
    assert payload["clinical_tasks"]
    assert payload["regulatory_tasks"]
