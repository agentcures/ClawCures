from __future__ import annotations

from collections.abc import Mapping

_DOMAIN_HINTS: tuple[tuple[str, str], ...] = (
    ("cancer", "oncology"),
    ("oncolog", "oncology"),
    ("tumor", "oncology"),
    ("neoplasm", "oncology"),
    ("heart", "cardiometabolic"),
    ("stroke", "cardiometabolic"),
    ("cardio", "cardiometabolic"),
    ("diabetes", "cardiometabolic"),
    ("metabolic", "cardiometabolic"),
    ("obesity", "cardiometabolic"),
    ("hiv", "infectious"),
    ("tuberculosis", "infectious"),
    ("malaria", "infectious"),
    ("infect", "infectious"),
    ("amr", "infectious"),
    ("alzheimer", "neuro"),
    ("parkinson", "neuro"),
    ("neuro", "neuro"),
    ("als", "neuro"),
    ("copd", "respiratory"),
    ("asthma", "respiratory"),
    ("pulmonary", "respiratory"),
    ("lung", "respiratory"),
)


def infer_domain_from_objective(objective: str) -> str:
    lowered = objective.lower()
    for token, domain in _DOMAIN_HINTS:
        if token in lowered:
            return domain
    return "general"


def pick_model_for_phase(
    *,
    phase: str,
    objective: str,
    model_map: Mapping[str, str] | None,
) -> str | None:
    if not model_map:
        return None

    domain = infer_domain_from_objective(objective)
    phase_name = _normalize_phase_bucket(phase)
    keys = (
        f"{phase_name}:{domain}",
        f"{phase_name}",
        domain,
        "default",
    )
    for key in keys:
        value = model_map.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_phase_bucket(value: str) -> str:
    lowered = value.strip().lower()
    if "critic" in lowered:
        return "critic"
    return "planner"
