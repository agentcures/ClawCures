from __future__ import annotations

from refua_campaign.openclaw_client import _extract_response_text
from refua_campaign.orchestrator import _extract_json_plan


def test_extract_json_plan_from_plain_json() -> None:
    plan = _extract_json_plan('{"calls":[{"tool":"refua_validate_spec","args":{}}]}')
    assert plan["calls"][0]["tool"] == "refua_validate_spec"


def test_extract_json_plan_from_wrapped_text() -> None:
    text = 'Plan follows:\n```json\n{"calls":[{"tool":"refua_job","args":{"job_id":"abc"}}]}\n```'
    plan = _extract_json_plan(text)
    assert plan["calls"][0]["args"]["job_id"] == "abc"


def test_extract_json_plan_normalizes_arguments_key() -> None:
    text = '{"calls":[{"tool":"refua_validate_spec","arguments":{"deep_validate":false}}]}'
    plan = _extract_json_plan(text)
    assert plan["calls"][0]["args"]["deep_validate"] is False


def test_extract_json_plan_supports_openai_function_shape() -> None:
    text = (
        '{"calls":[{"function":{"name":"refua_job","arguments":"{\\"job_id\\":\\"abc\\"}"}}]}'
    )
    plan = _extract_json_plan(text)
    assert plan["calls"][0]["tool"] == "refua_job"
    assert plan["calls"][0]["args"]["job_id"] == "abc"


def test_extract_json_plan_reads_nested_plan_key() -> None:
    text = '{"plan":{"calls":[{"name":"refua_validate_spec","args":{}}]}}'
    plan = _extract_json_plan(text)
    assert plan["calls"][0]["tool"] == "refua_validate_spec"


def test_extract_json_plan_canonicalizes_tool_alias_when_allowed() -> None:
    text = '{"calls":[{"tool":"validate_spec","args":{}}]}'
    plan = _extract_json_plan(text, allowed_tools=["refua_validate_spec"])
    assert plan["calls"][0]["tool"] == "refua_validate_spec"


def test_extract_response_text_prefers_output_text() -> None:
    payload = {"output_text": "hello world"}
    assert _extract_response_text(payload) == "hello world"


def test_extract_response_text_reads_nested_content() -> None:
    payload = {
        "output": [
            {
                "content": [
                    {"text": '{"calls":[]}'},
                ]
            }
        ]
    }
    assert _extract_response_text(payload) == '{"calls":[]}'
