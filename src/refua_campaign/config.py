from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_BASE_URL = "http://127.0.0.1:18789"
_DEFAULT_MODEL = "openclaw:main"
_DEFAULT_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class OpenClawConfig:
    base_url: str
    model: str
    timeout_seconds: float
    bearer_token: str | None

    @classmethod
    def from_env(cls) -> "OpenClawConfig":
        timeout_raw = os.getenv("REFUA_CAMPAIGN_TIMEOUT_SECONDS", "").strip()
        timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
        if timeout_raw:
            timeout_seconds = float(timeout_raw)

        token = (
            os.getenv("REFUA_CAMPAIGN_OPENCLAW_TOKEN", "").strip()
            or os.getenv("OPENCLAW_GATEWAY_TOKEN", "").strip()
            or os.getenv("OPENCLAW_GATEWAY_PASSWORD", "").strip()
            or None
        )
        return cls(
            base_url=os.getenv(
                "REFUA_CAMPAIGN_OPENCLAW_BASE_URL",
                _DEFAULT_BASE_URL,
            ).strip()
            or _DEFAULT_BASE_URL,
            model=os.getenv("REFUA_CAMPAIGN_OPENCLAW_MODEL", _DEFAULT_MODEL).strip()
            or _DEFAULT_MODEL,
            timeout_seconds=max(timeout_seconds, 1.0),
            bearer_token=token,
        )


@dataclass(frozen=True)
class CampaignRunConfig:
    objective: str
    output_path: Path | None = None
    dry_run: bool = False


def default_prompt_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "default_system_prompt.txt"
