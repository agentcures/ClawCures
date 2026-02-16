from __future__ import annotations

from refua_campaign.autonomy import PlanPolicy, evaluate_plan_policy


def test_policy_rejects_unsupported_tool() -> None:
    check = evaluate_plan_policy(
        {
            "calls": [
                {
                    "tool": "unknown_tool",
                    "args": {},
                }
            ]
        },
        allowed_tools=["refua_validate_spec"],
        policy=PlanPolicy(max_calls=5, require_validate_first=True),
    )
    assert check.approved is False
    assert any("unsupported tool" in err for err in check.errors)


def test_policy_warns_if_validate_not_first() -> None:
    check = evaluate_plan_policy(
        {
            "calls": [
                {
                    "tool": "refua_fold",
                    "args": {},
                }
            ]
        },
        allowed_tools=["refua_validate_spec", "refua_fold"],
        policy=PlanPolicy(max_calls=5, require_validate_first=True),
    )
    assert check.approved is True
    assert any("First call is not refua_validate_spec" in msg for msg in check.warnings)
