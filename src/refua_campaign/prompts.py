from __future__ import annotations

from pathlib import Path

from refua_campaign.config import default_prompt_path


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or default_prompt_path()
    return prompt_path.read_text(encoding="utf-8").strip()


def planner_suffix(allowed_tools: list[str]) -> str:
    tools = ", ".join(sorted(allowed_tools))
    return (
        "Output only valid JSON with this shape: "
        '{"calls":[{"tool":"<name>","args":{...}}]}. '
        f"Allowed tools: {tools}. "
        "Never emit markdown, prose, or comments."
    )
