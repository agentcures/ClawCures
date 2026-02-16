from __future__ import annotations

from refua_campaign.config import OpenClawConfig


def test_openclaw_config_uses_gateway_password_fallback(monkeypatch) -> None:
    monkeypatch.delenv("REFUA_CAMPAIGN_OPENCLAW_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv("OPENCLAW_GATEWAY_PASSWORD", "pw-token")

    cfg = OpenClawConfig.from_env()
    assert cfg.bearer_token == "pw-token"


def test_openclaw_config_token_precedence(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_GATEWAY_PASSWORD", "pw-token")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
    monkeypatch.setenv("REFUA_CAMPAIGN_OPENCLAW_TOKEN", "campaign-token")

    cfg = OpenClawConfig.from_env()
    assert cfg.bearer_token == "campaign-token"
