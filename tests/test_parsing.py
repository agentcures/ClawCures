from __future__ import annotations

from types import MethodType

import pytest

from refua_campaign.config import OpenClawConfig
from refua_campaign.openclaw_client import (
    OpenClawClient,
    _extract_function_calls,
    _extract_response_text,
    _parse_streaming_response,
)
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
    text = (
        '{"calls":[{"tool":"validate_spec","args":{"entities":[{"type":"protein","id":"target","sequence":"MKTAYI"}]}}]}'
    )
    plan = _extract_json_plan(text, allowed_tools=["refua_validate_spec"])
    assert plan["calls"][0]["tool"] == "refua_validate_spec"


def test_extract_json_plan_requires_entities_for_validate_spec_when_allowed() -> None:
    text = '{"calls":[{"tool":"refua_validate_spec","args":{"deep_validate":false}}]}'
    with pytest.raises(ValueError, match="must include 'entities'"):
        _extract_json_plan(text, allowed_tools=["refua_validate_spec"])


def test_extract_json_plan_requires_job_id_for_refua_job_when_allowed() -> None:
    text = '{"calls":[{"tool":"refua_job","args":{"action":"create_program"}}]}'
    with pytest.raises(ValueError, match="expects a 'job_id'"):
        _extract_json_plan(text, allowed_tools=["refua_job"])


def test_extract_json_plan_requires_query_for_web_search_when_allowed() -> None:
    text = '{"calls":[{"tool":"web_search","args":{"count":3}}]}'
    with pytest.raises(ValueError, match="must include a non-empty 'query'"):
        _extract_json_plan(text, allowed_tools=["web_search"])


def test_extract_json_plan_requires_url_for_web_fetch_when_allowed() -> None:
    text = '{"calls":[{"tool":"web_fetch","args":{"extract_mode":"markdown"}}]}'
    with pytest.raises(ValueError, match="must include a non-empty 'url'"):
        _extract_json_plan(text, allowed_tools=["web_fetch"])


def test_extract_json_plan_canonicalizes_websearch_alias_when_allowed() -> None:
    text = '{"calls":[{"tool":"websearch","args":{"query":"EGFR target biology"}}]}'
    plan = _extract_json_plan(text, allowed_tools=["web_search"])
    assert plan["calls"][0]["tool"] == "web_search"


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


def test_extract_function_calls_reads_openresponses_function_call_items() -> None:
    payload = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "web_search",
                "arguments": '{"query":"EGFR target biology","count":3}',
            }
        ]
    }

    calls = _extract_function_calls(payload)
    assert len(calls) == 1
    assert calls[0].call_id == "call_1"
    assert calls[0].name == "web_search"
    assert calls[0].arguments["query"] == "EGFR target biology"
    assert calls[0].arguments["count"] == 3


def test_create_response_forwards_optional_openclaw_fields() -> None:
    config = OpenClawConfig(
        base_url="http://localhost:12345",
        model="openclaw:test",
        timeout_seconds=10.0,
        bearer_token=None,
    )
    client = OpenClawClient(config)
    captured: dict[str, object] = {}

    def fake_post_json(
        self: OpenClawClient,
        path: str,
        payload: dict[str, object],
        *,
        stream: bool = False,
        on_stream_text=None,
    ) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        captured["stream"] = stream
        captured["on_stream_text"] = on_stream_text
        return {
            "id": "resp_2",
            "output_text": "done",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "web_search",
                    "arguments": '{"query":"KRAS pathway"}',
                }
            ],
        }

    client._post_json = MethodType(fake_post_json, client)  # type: ignore[method-assign]
    response = client.create_response(
        user_input="Find lung cancer targets",
        input_items=[{"type": "function_call_output", "call_id": "call_1", "output": "{}"}],
        instructions="Use tools.",
        user="campaign-main",
        store=True,
        tools=[
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        previous_response_id="resp_1",
    )

    assert captured["path"] == "/v1/responses"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
    ]
    assert payload["user"] == "campaign-main"
    assert payload["store"] is True
    assert payload["previous_response_id"] == "resp_1"
    assert captured["stream"] is False
    assert response.response_id == "resp_2"
    assert len(response.function_calls) == 1
    assert response.function_calls[0].name == "web_search"


def test_parse_streaming_response_builds_terminal_payload_and_text_chunks() -> None:
    class _FakeResponse:
        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"Hello "}\n'
            yield b'data: {"type":"response.output_text.delta","delta":"World"}\n'
            yield (
                b'data: {"type":"response.completed","response":{"id":"resp_9","output":[]}}\n'
            )
            yield b"data: [DONE]\n"

    chunks: list[str] = []
    payload = _parse_streaming_response(_FakeResponse(), on_stream_text=chunks.append)
    assert payload["id"] == "resp_9"
    assert payload["output_text"] == "Hello World"
    assert "".join(chunks) == "Hello World"


def test_create_response_forwards_model_override_and_stream_flag() -> None:
    config = OpenClawConfig(
        base_url="http://localhost:12345",
        model="openclaw:default",
        timeout_seconds=10.0,
        bearer_token=None,
    )
    client = OpenClawClient(config)
    captured: dict[str, object] = {}

    def fake_post_json(
        self: OpenClawClient,
        path: str,
        payload: dict[str, object],
        *,
        stream: bool = False,
        on_stream_text=None,
    ) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        captured["stream"] = stream
        captured["on_stream_text"] = on_stream_text
        return {"id": "resp_3", "output_text": "ok"}

    client._post_json = MethodType(fake_post_json, client)  # type: ignore[method-assign]
    response = client.create_response(
        user_input="test",
        instructions="instr",
        model="openclaw:oncology-planner",
        stream=True,
    )
    assert response.response_id == "resp_3"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "openclaw:oncology-planner"
    assert payload["stream"] is True
    assert captured["stream"] is True
