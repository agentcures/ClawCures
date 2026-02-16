from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from refua_campaign.config import OpenClawConfig


@dataclass
class OpenClawResponse:
    raw: dict[str, Any]
    text: str


class OpenClawClient:
    def __init__(self, config: OpenClawConfig) -> None:
        self._config = config

    def create_response(
        self,
        *,
        user_input: str,
        instructions: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "input": user_input,
            "instructions": instructions,
        }
        if metadata:
            payload["metadata"] = metadata

        response_json = self._post_json("/v1/responses", payload)
        return OpenClawResponse(
            raw=response_json,
            text=_extract_response_text(response_json),
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
