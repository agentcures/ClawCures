from __future__ import annotations

from pathlib import Path

from refua_campaign.config import default_prompt_path


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or default_prompt_path()
    return prompt_path.read_text(encoding="utf-8").strip()


def planner_suffix(allowed_tools: list[str]) -> str:
    tools = ", ".join(sorted(allowed_tools))
    has_web_tools = "web_search" in set(allowed_tools) and "web_fetch" in set(
        allowed_tools
    )
    web_guidance = ""
    if has_web_tools:
        web_guidance = (
            " For disease target discovery objectives, gather evidence with "
            "web_search and web_fetch before proposing target-specific intervention calls."
        )
    return (
        "Output only valid JSON with this shape: "
        '{"calls":[{"tool":"<name>","args":{...}}]}. '
        f"Allowed tools: {tools}. "
        + web_guidance
        + " "
        "Never emit markdown, prose, or comments."
    )
