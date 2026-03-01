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
