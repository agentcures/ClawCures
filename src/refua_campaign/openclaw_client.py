from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
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
    ) -> OpenClawResponse:
        input_payload: str | list[dict[str, Any]]
        if input_items is not None:
            input_payload = list(input_items)
        else:
            input_payload = user_input

        payload: dict[str, Any] = {
            "model": self._config.model,
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

        response_json = self._post_json("/v1/responses", payload)
        function_calls = _extract_function_calls(response_json)
        return OpenClawResponse(
            raw=response_json,
            text=_extract_response_text(response_json),
            response_id=_extract_response_id(response_json),
            function_calls=function_calls,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
