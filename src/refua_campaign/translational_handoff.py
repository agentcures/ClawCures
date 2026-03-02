from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_translational_handoff(
    *,
    objective: str,
    interesting_targets: list[Mapping[str, Any]],
    promising_cures: list[Mapping[str, Any]],
    evidence_quality: Mapping[str, Any],
) -> dict[str, Any]:
    top_targets = interesting_targets[:5]
    top_candidates = promising_cures[:5]

    preclinical_tasks = [
        {
            "task": "Define assay panels for top disease-target hypotheses.",
            "inputs": [
                {
                    "disease": item.get("disease"),
                    "target": item.get("target"),
                    "evidence_score": item.get("score"),
                }
                for item in top_targets
            ],
            "success_criteria": [
                "Reproducible signal across independent assays",
                "Target engagement evidence with controls",
                "Predefined go/no-go metrics before scale-up",
            ],
        }
    ]

    wetlab_tasks = [
        {
            "task": "Run confirmatory wet-lab experiments for top candidates.",
            "candidates": [
                {
                    "cure_id": item.get("cure_id"),
                    "name": item.get("name"),
                    "target": item.get("target"),
                    "score": item.get("score"),
                }
                for item in top_candidates
            ],
            "required_controls": [
                "Positive control compound",
                "Negative control compound",
                "Batch and operator replication",
            ],
        }
    ]

    clinical_tasks = [
        {
            "task": "Prepare adaptive biomarker-enriched simulation packages.",
            "commands": [
                "refua_clinical_simulator(trial_id=..., include_workup=true)",
                "ClawCures trials-add --trial-id ... --phase \"Phase II\"",
            ],
            "gates": [
                "Translational biomarker alignment",
                "Safety margin acceptable before first-in-human",
                "Clear failure criteria to avoid sunk-cost bias",
            ],
        }
    ]

    regulatory_tasks = [
        {
            "task": "Produce regulatory evidence bundle and checklist.",
            "commands": [
                "refua-regulatory bundle-build --campaign-run <path>",
                "refua-regulatory checklist --template drug_discovery_comprehensive",
            ],
            "required_artifacts": [
                "Decision lineage",
                "Model/data provenance",
                "Checksum-verified evidence package",
            ],
        }
    ]

    return {
        "objective": objective,
        "evidence_quality_band": str(evidence_quality.get("quality_band") or "unknown"),
        "priority_targets": [
            {
                "disease": item.get("disease"),
                "target": item.get("target"),
                "score": item.get("score"),
            }
            for item in top_targets
        ],
        "priority_candidates": [
            {
                "cure_id": item.get("cure_id"),
                "name": item.get("name"),
                "target": item.get("target"),
                "score": item.get("score"),
                "promising": item.get("promising"),
            }
            for item in top_candidates
        ],
        "preclinical_tasks": preclinical_tasks,
        "wetlab_tasks": wetlab_tasks,
        "clinical_tasks": clinical_tasks,
        "regulatory_tasks": regulatory_tasks,
    }
