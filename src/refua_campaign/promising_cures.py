from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_ADMET_PATH_HINTS: tuple[str, ...] = (
    "admet",
    "tox",
    "herg",
    "ames",
    "dili",
    "carcinogen",
    "clintox",
    "clearance",
    "half_life",
    "bioavailability",
    "solubility",
    "permeability",
    "caco2",
    "pampa",
    "cyp",
    "metabolic",
)

_NEGATIVE_ASSESSMENT_HINTS: tuple[str, ...] = (
    "high risk",
    "unsafe",
    "toxic",
    "toxicity",
    "poor",
    "liability",
)

_POSITIVE_ASSESSMENT_HINTS: tuple[str, ...] = (
    "promising",
    "favorable",
    "favourable",
    "good",
    "strong",
    "safe",
    "high confidence",
)


@dataclass(frozen=True)
class PromisingCure:
    cure_id: str
    name: str | None
    smiles: str | None
    target: str | None
    tool: str
    score: float
    promising: bool
    assessment: str | None
    metrics: dict[str, float | None]
    admet: dict[str, Any]
    evidence_paths: dict[str, str]
    tool_args: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "cure_id": self.cure_id,
            "name": self.name,
            "smiles": self.smiles,
            "target": self.target,
            "tool": self.tool,
            "score": self.score,
            "promising": self.promising,
            "assessment": self.assessment,
            "metrics": self.metrics,
            "admet": self.admet,
            "evidence_paths": self.evidence_paths,
            "tool_args": self.tool_args,
        }


def extract_promising_cures(
    results: list[Any],
    *,
    min_score: float = 55.0,
) -> list[dict[str, Any]]:
    extracted: list[PromisingCure] = []
    for index, item in enumerate(results):
        cure = _extract_cure_from_result(item=item, index=index, min_score=min_score)
        if cure is not None:
            extracted.append(cure)
    extracted.sort(key=lambda item: item.score, reverse=True)
    return [item.to_json() for item in extracted]


def summarize_promising_cures(cures: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cures)
    promising = 0
    with_admet = 0
    for cure in cures:
        if bool(cure.get("promising")):
            promising += 1
        admet = cure.get("admet")
        if isinstance(admet, Mapping):
            properties = admet.get("properties")
            if isinstance(properties, Mapping) and properties:
                with_admet += 1
    return {
        "total_candidates": total,
        "promising_count": promising,
        "with_admet_properties": with_admet,
    }


def _extract_cure_from_result(
    *,
    item: Any,
    index: int,
    min_score: float,
) -> PromisingCure | None:
    tool, args, output = _result_parts(item)

    flat: dict[str, Any] = {}
    _flatten(args, "args", flat)
    _flatten(output, "output", flat)

    evidence_paths: dict[str, str] = {}

    name, name_path = _pick_string(flat, aliases=[
        "name",
        "candidate_name",
        "compound_name",
        "ligand_name",
        "binder",
    ])
    if name_path:
        evidence_paths["name"] = name_path

    smiles, smiles_path = _pick_string(flat, aliases=[
        "smiles",
        "ligand_smiles",
        "compound_smiles",
    ])
    if smiles_path:
        evidence_paths["smiles"] = smiles_path

    target, target_path = _pick_string(flat, aliases=[
        "target",
        "target_name",
        "protein",
        "antigen",
    ])
    if target_path:
        evidence_paths["target"] = target_path

    binding_probability, binding_path = _pick_float(flat, aliases=[
        "binding_probability",
        "predicted_probability",
        "p_bind",
        "probability",
    ])
    if binding_path:
        evidence_paths["binding_probability"] = binding_path

    affinity, affinity_path = _pick_float(flat, aliases=[
        "affinity",
        "predicted_affinity",
        "delta_g",
    ])
    if affinity_path:
        evidence_paths["affinity"] = affinity_path

    ic50, ic50_path = _pick_float(flat, aliases=["ic50", "predicted_ic50"])
    if ic50_path:
        evidence_paths["ic50"] = ic50_path

    kd, kd_path = _pick_float(flat, aliases=["kd", "predicted_kd"])
    if kd_path:
        evidence_paths["kd"] = kd_path

    admet_properties = _collect_admet_properties(flat)
    admet_key_metrics = _collect_admet_key_metrics(admet_properties)
    admet_score = admet_key_metrics.get("admet_score")
    if admet_score is not None:
        evidence_paths.setdefault("admet_score", "admet.properties.admet_score")

    assessment, assessment_path = _pick_string(flat, aliases=[
        "assessment",
        "assessment_text",
        "admet_assessment",
        "safety_assessment",
        "summary",
    ])
    if assessment_path:
        evidence_paths["assessment"] = assessment_path

    metrics = {
        "binding_probability": binding_probability,
        "admet_score": admet_score,
        "affinity": affinity,
        "ic50": ic50,
        "kd": kd,
    }

    has_signal = any(value is not None for value in metrics.values()) or bool(smiles)
    if not has_signal:
        return None

    explicit_score, _ = _pick_float(flat, aliases=[
        "promising_score",
        "priority_score",
        "composite_score",
    ])
    if explicit_score is None:
        score = _score_candidate(metrics=metrics, assessment=assessment)
    else:
        score = round(max(0.0, min(float(explicit_score), 100.0)), 2)

    if not assessment:
        assessment = _assessment_from_score(score, admet_score)

    explicit_promising, _ = _pick_bool(flat, aliases=[
        "promising",
        "is_promising",
        "recommended",
        "is_recommended",
    ])
    if explicit_promising is None:
        lowered_assessment = (assessment or "").lower()
        has_negative_hint = any(token in lowered_assessment for token in _NEGATIVE_ASSESSMENT_HINTS)
        promising = score >= float(min_score) and not has_negative_hint
    else:
        promising = bool(explicit_promising)

    cure_id = _resolve_cure_id(
        name=name,
        smiles=smiles,
        tool=tool,
        index=index,
    )
    admet_payload: dict[str, Any] = {
        "properties": admet_properties,
        "key_metrics": admet_key_metrics,
        "status": _infer_admet_status(flat),
    }

    return PromisingCure(
        cure_id=cure_id,
        name=name,
        smiles=smiles,
        target=target,
        tool=tool,
        score=score,
        promising=promising,
        assessment=assessment,
        metrics=metrics,
        admet=admet_payload,
        evidence_paths=evidence_paths,
        tool_args=args,
    )


def _result_parts(item: Any) -> tuple[str, dict[str, Any], Any]:
    if isinstance(item, Mapping):
        tool = str(item.get("tool") or "unknown_tool")
        args = item.get("args")
        output = item.get("output")
        return tool, args if isinstance(args, Mapping) else {}, output

    tool = str(getattr(item, "tool", "unknown_tool"))
    args_raw = getattr(item, "args", {})
    output = getattr(item, "output", None)
    return tool, dict(args_raw) if isinstance(args_raw, Mapping) else {}, output


def _collect_admet_properties(flat: dict[str, Any]) -> dict[str, float | str | bool | None]:
    properties: dict[str, float | str | bool | None] = {}
    for path, value in flat.items():
        if not _is_scalar(value):
            continue
        lowered = path.lower()
        if "raw_output" in lowered or "raw_outputs" in lowered:
            continue
        if not any(token in lowered for token in _ADMET_PATH_HINTS):
            continue
        normalized_key = _normalize_admet_key(path)
        if normalized_key in properties:
            continue
        properties[normalized_key] = value
    return properties


def _collect_admet_key_metrics(
    admet_properties: dict[str, float | str | bool | None]
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "admet_score": None,
        "safety_score": None,
        "adme_score": None,
        "rdkit_score": None,
    }
    for metric_name in tuple(metrics):
        resolved = _find_admet_metric(admet_properties, metric_name)
        if resolved is not None:
            metrics[metric_name] = resolved
    return metrics


def _find_admet_metric(
    admet_properties: dict[str, float | str | bool | None],
    metric_name: str,
) -> float | None:
    direct = admet_properties.get(metric_name)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)

    for key, value in admet_properties.items():
        if metric_name not in key.lower():
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _resolve_cure_id(*, name: str | None, smiles: str | None, tool: str, index: int) -> str:
    if name:
        return f"{tool}:{_slugify(name)}"
    if smiles:
        return f"{tool}:{_slugify(smiles[:20])}"
    return f"{tool}:{index}"


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "-":
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "candidate"


def _score_candidate(*, metrics: dict[str, float | None], assessment: str | None) -> float:
    score = 0.0

    binding_probability = metrics.get("binding_probability")
    if binding_probability is not None:
        bp = binding_probability
        if bp > 1:
            bp = bp / 100.0
        score += 55.0 * _clamp01(bp)

    admet_score = metrics.get("admet_score")
    if admet_score is not None:
        ad = admet_score
        if ad > 1:
            ad = ad / 100.0
        score += 25.0 * _clamp01(ad)

    affinity = metrics.get("affinity")
    if affinity is not None:
        if affinity < 0:
            score += 12.0 * _clamp01((-affinity) / 15.0)
        else:
            score += 8.0 * _clamp01(affinity / 15.0)

    ic50 = metrics.get("ic50")
    if ic50 is not None and ic50 > 0:
        score += 8.0 * _potency_score(ic50)

    kd = metrics.get("kd")
    if kd is not None and kd > 0:
        score += 6.0 * _potency_score(kd)

    metric_count = sum(1 for value in metrics.values() if value is not None)
    score += min(metric_count, 5) * 1.5

    if assessment:
        lowered = assessment.lower()
        if any(token in lowered for token in _NEGATIVE_ASSESSMENT_HINTS):
            score -= 12.0
        elif any(token in lowered for token in _POSITIVE_ASSESSMENT_HINTS):
            score += 6.0

    return round(max(0.0, min(score, 100.0)), 2)


def _assessment_from_score(score: float, admet_score: float | None) -> str:
    if admet_score is not None:
        if admet_score >= 0.8:
            return "Promising ADMET profile with strong translational potential."
        if admet_score >= 0.65:
            return "Balanced ADMET profile with moderate optimization risk."
        return "ADMET profile indicates notable optimization risk."

    if score >= 80:
        return "High-confidence promising therapeutic candidate."
    if score >= 60:
        return "Promising candidate with meaningful follow-up signal."
    if score >= 45:
        return "Early signal candidate requiring optimization."
    return "Low-confidence candidate; substantial optimization required."


def _potency_score(value: float) -> float:
    transformed = 1.0 / (1.0 + math.log10(value + 1.0))
    return _clamp01(transformed)


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _infer_admet_status(flat: dict[str, Any]) -> str | None:
    value, _ = _pick_string(flat, aliases=[
        "admet_status",
        "status",
    ])
    if value is None:
        return None
    lowered = value.lower()
    if "success" in lowered:
        return "success"
    if "unavailable" in lowered:
        return "unavailable"
    if "failed" in lowered:
        return "failed"
    return value


def _normalize_admet_key(path: str) -> str:
    key = path
    if key.startswith("output."):
        key = key.removeprefix("output.")
    if "admet." in key:
        key = key.split("admet.", 1)[1]
    return key


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(nested, next_prefix, out)
        return
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            _flatten(nested, next_prefix, out)
        return
    if _is_scalar(value):
        out[prefix] = value


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _pick_string(flat: dict[str, Any], aliases: list[str]) -> tuple[str | None, str | None]:
    value, path = _pick_value(flat, aliases)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped, path
    return None, None


def _pick_float(flat: dict[str, Any], aliases: list[str]) -> tuple[float | None, str | None]:
    value, path = _pick_value(flat, aliases)
    if isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), path
    if isinstance(value, str):
        try:
            return float(value.strip()), path
        except ValueError:
            return None, None
    return None, None


def _pick_bool(flat: dict[str, Any], aliases: list[str]) -> tuple[bool | None, str | None]:
    value, path = _pick_value(flat, aliases)
    if isinstance(value, bool):
        return value, path
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True, path
        if lowered in {"false", "no", "0", "off"}:
            return False, path
    return None, None


def _pick_value(flat: dict[str, Any], aliases: list[str]) -> tuple[Any | None, str | None]:
    exact_hits: list[tuple[str, Any]] = []
    loose_hits: list[tuple[str, Any]] = []

    for path, value in flat.items():
        leaf = _leaf_token(path)
        lowered_leaf = leaf.lower()
        lowered_path = path.lower()

        for alias in aliases:
            alias_lower = alias.lower()
            if lowered_leaf == alias_lower:
                exact_hits.append((path, value))
                break
            if alias_lower in lowered_path:
                loose_hits.append((path, value))
                break

    if exact_hits:
        exact_hits.sort(key=lambda item: len(item[0]))
        return exact_hits[0][1], exact_hits[0][0]
    if loose_hits:
        loose_hits.sort(key=lambda item: len(item[0]))
        return loose_hits[0][1], loose_hits[0][0]
    return None, None


def _leaf_token(path: str) -> str:
    token = path
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    if "[" in token:
        token = token.split("[", 1)[0]
    return token

