from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple
from urllib import error as url_error
from urllib import request as url_request


@dataclass(frozen=True)
class WMResponse:
    content_type: str
    body: bytes
    headers: Mapping[str, str]


class WMClient:
    """Minimal transport client for the EgoDex remote WM service."""

    def __init__(self, hostname: str, port: int, timeout_s: float = 3.0, scheme: str = "http"):
        self.hostname = hostname
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.scheme = scheme
        self._base_url = f"{self.scheme}://{self.hostname}:{self.port}"

    def _url(self, path: str) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{clean_path}"

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> WMResponse:
        request_bytes = json.dumps(payload).encode("utf-8")
        request_obj = url_request.Request(
            self._url(path),
            data=request_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "image/jpeg, application/json, application/octet-stream",
            },
        )
        try:
            with url_request.urlopen(request_obj, timeout=self.timeout_s) as response:
                return WMResponse(
                    content_type=(response.headers.get("Content-Type") or "")
                    .split(";", 1)[0]
                    .strip()
                    .lower(),
                    body=response.read(),
                    headers={key.lower(): value for key, value in response.headers.items()},
                )
        except url_error.URLError as exc:
            raise ConnectionError(f"WM request failed for {self._url(path)}: {exc}") from exc

    @staticmethod
    def _decode_base64_image(value: str) -> bytes:
        clean_value = value.strip()
        if clean_value.startswith("data:image"):
            clean_value = clean_value.split(",", 1)[-1]
        return base64.b64decode(clean_value)

    def _extract_observation_bytes(self, payload: Mapping[str, Any]) -> bytes:
        for key in ("obs_jpg", "observation_jpg", "observation", "obs", "jpg", "image"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return self._decode_base64_image(value)
            if isinstance(value, list):
                return bytes(value)
        raise ValueError("WM response payload does not include an observation image.")

    def wm_step(self, payload: Mapping[str, Any]) -> bytes:
        response = self._post_json("/step", payload)
        if response.content_type in {"image/jpeg", "image/jpg", "application/octet-stream"}:
            return response.body
        if response.content_type == "application/json":
            decoded = json.loads(response.body.decode("utf-8"))
            if isinstance(decoded, Mapping):
                return self._extract_observation_bytes(decoded)
            if isinstance(decoded, str):
                return self._decode_base64_image(decoded)
            raise ValueError("WM /step JSON response must be an object or base64 string.")
        if response.content_type in {"text/plain", "text/html"}:
            return self._decode_base64_image(response.body.decode("utf-8"))
        raise ValueError(f"Unsupported WM /step response type: {response.content_type}")

    def reset(self, new: bool) -> Tuple[bytes, Mapping[str, Any]]:
        response = self._post_json("/reset", {"new": bool(new)})

        if response.content_type in {"image/jpeg", "image/jpg", "application/octet-stream"}:
            raw_state = response.headers.get("x-wm-state")
            if raw_state:
                try:
                    parsed_state = json.loads(raw_state)
                    if isinstance(parsed_state, Mapping):
                        return response.body, parsed_state
                except json.JSONDecodeError:
                    pass
            return response.body, {}

        if response.content_type == "application/json":
            decoded = json.loads(response.body.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise ValueError("WM /reset JSON response must be an object.")
            obs_bytes = self._extract_observation_bytes(decoded)
            state = decoded.get("state", decoded.get("wm_state", {}))
            if not isinstance(state, Mapping):
                state = {}
            return obs_bytes, state

        raise ValueError(f"Unsupported WM /reset response type: {response.content_type}")
