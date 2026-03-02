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


def test_policy_enforces_stage_progression_when_enabled() -> None:
    check = evaluate_plan_policy(
        {
            "calls": [
                {
                    "tool": "refua_affinity",
                    "args": {"entities": []},
                }
            ]
        },
        allowed_tools=["refua_validate_spec", "refua_affinity"],
        policy=PlanPolicy(
            max_calls=5,
            require_validate_first=False,
            enforce_stage_progression=True,
        ),
    )
    assert check.approved is False
    assert any("requires prior refua_validate_spec" in err for err in check.errors)


def test_policy_requires_evidence_before_hypothesis_when_enabled() -> None:
    check = evaluate_plan_policy(
        {
            "calls": [
                {
                    "tool": "refua_validate_spec",
                    "args": {"entities": []},
                },
                {
                    "tool": "refua_fold",
                    "args": {"entities": []},
                },
            ]
        },
        allowed_tools=["refua_validate_spec", "refua_fold", "web_search"],
        policy=PlanPolicy(
            max_calls=5,
            require_validate_first=False,
            require_evidence_before_hypothesis=True,
        ),
    )
    assert check.approved is False
    assert any("requires evidence collection" in err for err in check.errors)
