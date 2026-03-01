from __future__ import annotations

from refua_campaign.promising_cures import (
    extract_promising_cures,
    summarize_promising_cures,
)


def test_extract_promising_cure_with_full_admet_properties() -> None:
    results = [
        {
            "tool": "refua_fold",
            "args": {
                "name": "kras_candidate_alpha",
                "entities": [{"type": "ligand", "smiles": "CCN"}],
            },
            "output": {
                "target": "KRAS",
                "affinity": {
                    "binding_probability": 0.88,
                    "ic50": 0.12,
                },
                "admet": {
                    "status": "success",
                    "results": [
                        {
                            "ligand_id": "lig",
                            "smiles": "CCN",
                            "admet_score": 0.83,
                            "safety_score": 0.91,
                            "assessment": "promising safety profile",
                            "predictions": {
                                "hERG": 0.12,
                                "AMES": 0.08,
                                "Bioavailability_Ma": 0.76,
                            },
                        }
                    ],
                },
            },
        }
    ]

    cures = extract_promising_cures(results)
    assert len(cures) == 1

    cure = cures[0]
    assert cure["promising"] is True
    assert cure["target"] == "KRAS"
    assert cure["smiles"] == "CCN"
    assert cure["metrics"]["admet_score"] is not None

    admet = cure["admet"]
    assert admet["status"] == "success"
    assert "admet_score" in admet["key_metrics"]
    assert admet["key_metrics"]["admet_score"] is not None
    assert len(admet["properties"]) >= 4


def test_extract_promising_cures_respects_negative_assessment() -> None:
    results = [
        {
            "tool": "refua_affinity",
            "args": {"name": "risky_candidate", "smiles": "CCO"},
            "output": {
                "target": "EGFR",
                "binding_probability": 0.9,
                "admet_score": 0.82,
                "assessment": "high risk toxicity liability",
            },
        }
    ]

    cures = extract_promising_cures(results)
    assert len(cures) == 1
    assert cures[0]["promising"] is False


def test_summarize_promising_cures_counts() -> None:
    cures = [
        {
            "promising": True,
            "admet": {"properties": {"admet_score": 0.9}},
        },
        {
            "promising": False,
            "admet": {"properties": {}},
        },
    ]

    summary = summarize_promising_cures(cures)
    assert summary["total_candidates"] == 2
    assert summary["promising_count"] == 1
    assert summary["with_admet_properties"] == 1
