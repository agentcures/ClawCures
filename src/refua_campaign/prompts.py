from __future__ import annotations

from pathlib import Path

from refua_campaign.config import default_prompt_path


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or default_prompt_path()
    return prompt_path.read_text(encoding="utf-8").strip()


def planner_suffix(allowed_tools: list[str]) -> str:
    allowed = set(allowed_tools)
    tools = ", ".join(sorted(allowed_tools))
    has_web_tools = "web_search" in allowed and "web_fetch" in allowed
    web_guidance = ""
    if has_web_tools:
        web_guidance = (
            " For disease target discovery objectives, gather evidence with "
            "web_search and web_fetch before proposing target-specific intervention calls."
        )
    stage_guidance = ""
    if {"refua_validate_spec", "refua_fold", "refua_affinity"} & allowed:
        stage_guidance = (
            " Prefer a staged progression: evidence tools -> refua_validate_spec -> "
            "design/affinity tools -> ADMET/clinical simulation where available."
        )
    entity_guidance = ""
    if {"refua_validate_spec", "refua_fold", "refua_affinity"} & allowed:
        entity_guidance = (
            " For refua_validate_spec, refua_fold, and refua_affinity, always include "
            '"entities" as typed objects, for example '
            '{"entities":[{"type":"protein","id":"A","sequence":"MKTAYI"},'
            '{"type":"ligand","id":"lig","smiles":"CCO"}]}.'
        )
    citation_guidance = (
        " Include evidence-bearing calls (web_search/web_fetch or refua_data_*) before "
        "high-cost hypothesis calls, and preserve source URLs in arguments/results."
    )
    return (
        "Output only valid JSON with this shape: "
        '{"calls":[{"tool":"<name>","args":{...}}]}. '
        f"Allowed tools: {tools}. "
        + web_guidance
        + stage_guidance
        + entity_guidance
        + citation_guidance
        + " "
        "Never emit markdown, prose, or comments."
    )
