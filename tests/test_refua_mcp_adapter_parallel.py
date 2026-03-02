from __future__ import annotations

from typing import Any

import refua_campaign.refua_mcp_adapter as adapter_mod
from refua_campaign.refua_mcp_adapter import RefuaMcpAdapter


def test_execute_tools_parallel_handles_recoverable_errors(
    monkeypatch,
) -> None:
    def _ok_tool(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "args": kwargs}

    def _failing_tool(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        adapter_mod,
        "_load_tool_map",
        lambda: {"ok_tool": _ok_tool, "failing_tool": _failing_tool},
    )
    mcp = RefuaMcpAdapter()

    results = mcp.execute_tools_parallel(
        [
            ("ok_tool", {"value": 1}),
            ("failing_tool", {"value": 2}),
        ],
        max_workers=2,
        fail_fast=False,
    )
    assert len(results) == 2
    assert results[0].tool == "ok_tool"
    assert results[0].output["ok"] is True
    assert results[1].tool == "failing_tool"
    assert results[1].output["recoverable"] is True


def test_parallel_safe_tools_subset(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter_mod,
        "_load_tool_map",
        lambda: {
            "web_search": lambda **_kwargs: {},
            "refua_validate_spec": lambda **_kwargs: {},
            "refua_fold": lambda **_kwargs: {},
        },
    )
    mcp = RefuaMcpAdapter()
    safe = mcp.parallel_safe_tools()
    assert "web_search" in safe
    assert "refua_validate_spec" in safe
    assert "refua_fold" not in safe
