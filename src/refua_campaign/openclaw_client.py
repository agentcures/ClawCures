from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from refua_campaign.config import OpenClawConfig


@dataclass(frozen=True)
class OpenClawFunctionCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None


@dataclass
class OpenClawResponse:
    raw: dict[str, Any]
    text: str
    response_id: str | None = None
    function_calls: list[OpenClawFunctionCall] = field(default_factory=list)


class OpenClawClient:
    def __init__(self, config: OpenClawConfig) -> None:
        self._config = config

    def create_response(
        self,
        *,
        user_input: str,
        instructions: str,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
        store: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        previous_response_id: str | None = None,
        input_items: list[dict[str, Any]] | None = None,
        model: str | None = None,
        stream: bool | None = None,
        on_stream_text: Callable[[str], None] | None = None,
    ) -> OpenClawResponse:
        input_payload: str | list[dict[str, Any]]
        if input_items is not None:
            input_payload = list(input_items)
        else:
            input_payload = user_input

        payload: dict[str, Any] = {
            "model": (model or self._config.model),
            "input": input_payload,
            "instructions": instructions,
        }
        if metadata:
            payload["metadata"] = metadata
        if user:
            payload["user"] = user
        if store is not None:
            payload["store"] = bool(store)
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        stream_enabled = bool(stream)
        if stream is not None:
            payload["stream"] = stream_enabled

        response_json = self._post_json(
            "/v1/responses",
            payload,
            stream=stream_enabled,
            on_stream_text=on_stream_text,
        )
        function_calls = _extract_function_calls(response_json)
        return OpenClawResponse(
            raw=response_json,
            text=_extract_response_text(response_json),
            response_id=_extract_response_id(response_json),
            function_calls=function_calls,
        )

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        stream: bool = False,
        on_stream_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        url = urljoin(self._config.base_url.rstrip("/") + "/", path.lstrip("/"))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"

        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._config.timeout_seconds) as response:
                if stream:
                    return _parse_streaming_response(
                        response,
                        on_stream_text=on_stream_text,
                    )
                content = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenClaw API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Failed to reach OpenClaw API at {url}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenClaw API returned non-JSON content.") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(
                "OpenClaw API returned an unexpected response envelope type."
            )
        return parsed


def _parse_streaming_response(
    response: Any,
    *,
    on_stream_text: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    final_payload: dict[str, Any] | None = None
    output_text_chunks: list[str] = []

    for raw_line in response:
        try:
            line = raw_line.decode("utf-8", errors="replace")
        except Exception:
            continue
        stripped = line.strip()
        if not stripped or not stripped.startswith("data:"):
            continue
        event_payload = stripped[5:].strip()
        if not event_payload or event_payload == "[DONE]":
            continue
        try:
            parsed_event = json.loads(event_payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_event, dict):
            continue

        response_block = parsed_event.get("response")
        if isinstance(response_block, dict):
            final_payload = response_block

        event_type = str(parsed_event.get("type") or "").strip().lower()
        if event_type in {"response.output_text.delta", "response.output_text"}:
            delta = parsed_event.get("delta")
            if not isinstance(delta, str):
                delta = parsed_event.get("text")
            if isinstance(delta, str) and delta:
                output_text_chunks.append(delta)
                if on_stream_text is not None:
                    on_stream_text(delta)

    if final_payload is not None:
        if output_text_chunks and not final_payload.get("output_text"):
            final_payload = dict(final_payload)
            final_payload["output_text"] = "".join(output_text_chunks)
        return final_payload

    if output_text_chunks:
        return {"output_text": "".join(output_text_chunks), "output": []}

    raise RuntimeError("OpenClaw streaming response did not include a terminal payload.")


def _extract_response_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_function_calls(payload: dict[str, Any]) -> list[OpenClawFunctionCall]:
    parsed_calls: list[OpenClawFunctionCall] = []
    seen: set[tuple[str, str]] = set()

    output = payload.get("output")
    if isinstance(output, list):
        for index, item in enumerate(output):
            call = _parse_function_call_item(item, fallback_index=index)
            if call is None:
                continue
            key = (call.call_id, call.name)
            if key in seen:
                continue
            seen.add(key)
            parsed_calls.append(call)

    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        base_index = len(parsed_calls)
        for offset, item in enumerate(tool_calls):
            call = _parse_function_call_item(item, fallback_index=base_index + offset)
            if call is None:
                continue
            key = (call.call_id, call.name)
            if key in seen:
                continue
            seen.add(key)
            parsed_calls.append(call)

    return parsed_calls


def _parse_function_call_item(
    item: Any,
    *,
    fallback_index: int,
) -> OpenClawFunctionCall | None:
    if not isinstance(item, dict):
        return None

    call_type = str(item.get("type") or "").strip().lower()
    function_block = item.get("function")

    if call_type not in {"function_call", "tool_call", "function"}:
        if not isinstance(function_block, dict):
            return None

    name: str | None = None
    raw_arguments: str | None = None
    arguments_payload: Any = None

    if isinstance(function_block, dict):
        name_value = function_block.get("name")
        if isinstance(name_value, str) and name_value.strip():
            name = name_value.strip()
        arguments_payload = function_block.get("arguments")

    if name is None:
        name_value = item.get("name")
        if isinstance(name_value, str) and name_value.strip():
            name = name_value.strip()
    if name is None:
        return None

    if arguments_payload is None:
        arguments_payload = item.get("arguments")

    arguments: dict[str, Any] = {}
    if isinstance(arguments_payload, dict):
        arguments = arguments_payload
    elif isinstance(arguments_payload, str):
        raw_arguments = arguments_payload
        stripped = arguments_payload.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                arguments = parsed

    call_id_value = item.get("call_id") or item.get("id")
    if isinstance(call_id_value, str) and call_id_value.strip():
        call_id = call_id_value.strip()
    else:
        call_id = f"call_{fallback_index + 1}"

    return OpenClawFunctionCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = payload.get("output")
    chunks: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
            content = item.get("content")
            if isinstance(content, list):
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    piece_text = piece.get("text")
                    if isinstance(piece_text, str) and piece_text.strip():
                        chunks.append(piece_text.strip())

    if chunks:
        return "\n".join(chunks)

    fallback = json.dumps(payload, ensure_ascii=True)
    return fallback
