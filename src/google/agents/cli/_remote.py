# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URL classification + auth headers for talking to remote agents.

Transport-neutral so all callers (ADK, A2A, Agent Runtime) agree on
how a ``--url`` is classified and what auth token it gets.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from typing import NamedTuple
from urllib.parse import urlparse, urlsplit, urlunsplit

import click

from google.agents.cli.auth import get_access_token, get_id_token

_AGENT_ENGINE_URL_FRAGMENT = "aiplatform.googleapis.com"
_REASONING_ENGINE_PATH = "reasoningEngines"
# A valid Agent Runtime host carries a location prefix: <location>-aiplatform.googleapis.com
_AGENT_RUNTIME_HOST_RE = re.compile(
    rf"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-{re.escape(_AGENT_ENGINE_URL_FRAGMENT)}"
)

# WebSocket traffic has its own ingress; HTTP goes through reasoningEngines/v1.
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/tutorial_bidi_stream_v2.ipynb
_AGENT_RUNTIME_WS_INGRESS = "reasoningEngines/ws/internal"

# urllib.parse treats a backslash before "@" as userinfo, while requests /
# urllib3 percent-encode it and connect to the host before it. Reject the
# ambiguous characters rather than trying to emulate both parsers.
_UNSAFE_URL_CHARS = ("\\", "\x00", "\r", "\n")
_UNSAFE_NETLOC_CHARS = ("@", "\\", "%")


def _parse_remote_url(url: str):
    """Parse an HTTP(S) URL and reject ambiguous or credential-bearing forms."""
    if not url:
        raise click.UsageError("A remote URL is required.")
    if any(char in url for char in _UNSAFE_URL_CHARS) or any(
        char.isspace() for char in url
    ):
        raise click.UsageError(
            "Invalid remote URL: backslashes, whitespace, and control "
            "characters are not allowed."
        )
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError as exc:
        raise click.UsageError("Invalid remote URL: malformed URL.") from exc
    if parts.scheme not in ("http", "https"):
        raise click.UsageError(
            "Invalid remote URL: only http:// and https:// URLs are supported."
        )
    if not hostname:
        raise click.UsageError("Invalid remote URL: missing host.")
    if any(char in parts.netloc for char in _UNSAFE_NETLOC_CHARS):
        raise click.UsageError(
            "Invalid remote URL: credentials and percent-encoded hosts are "
            "not supported."
        )
    try:
        parts.port
    except ValueError as exc:
        raise click.UsageError("Invalid remote URL: invalid port.") from exc
    return parts


def _has_path_segment(parts, segment: str) -> bool:
    return segment in parts.path.split("/")


def is_agent_runtime_url(url: str) -> bool:
    """Return ``True`` if *url* points to an Agent Runtime endpoint."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    if any(char in url for char in _UNSAFE_URL_CHARS) or any(
        char in parts.netloc for char in _UNSAFE_NETLOC_CHARS
    ):
        return False
    return bool(_AGENT_RUNTIME_HOST_RE.fullmatch(host)) and _has_path_segment(
        parts, _REASONING_ENGINE_PATH
    )


def is_legacy_agent_runtime_url(url: str) -> bool:
    """``True`` for a bare, non-passthrough Agent Runtime resource URL.

    This URL can only be used for legacy ``:streamQuery`` endpoints, not the
    ``/api/`` passthrough API.

    A URL already containing ``/api`` (the deployed APP_URL form served by the
    container) or any non-Agent-Runtime URL returns ``False``.
    """
    try:
        path = urlsplit(url).path
    except ValueError:
        return False
    return is_agent_runtime_url(url) and "/api" not in path


def validate_agent_runtime_url(url: str) -> None:
    """Validate a remote URL before credentials are attached.

    This rejects URL forms that ``urllib.parse`` and ``requests`` interpret
    differently (notably ``\\@``), then verifies that Agent Runtime URLs use
    the required ``<location>-aiplatform.googleapis.com`` host over HTTPS.
    """
    parts = _parse_remote_url(url)
    if not _has_path_segment(parts, _REASONING_ENGINE_PATH):
        return
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and _AGENT_RUNTIME_HOST_RE.fullmatch(host):
        return  # Host already carries a <location>- prefix.
    raise click.UsageError(
        "Detected an Agent Runtime URL with a missing location.\n"
        "  The location must appear in the host. The correct format is:\n"
        "    https://<LOCATION>-aiplatform.googleapis.com/v1/projects/<PROJECT>"
        "/locations/<LOCATION>/reasoningEngines/<ID>"
    )


def build_agent_runtime_passthrough_url(location: str, runtime_resource: str) -> str:
    """Base all clients can target. ``runtime_resource`` is the full
    ``projects/<num>/locations/<loc>/reasoningEngines/<id>`` resource name."""
    return (
        f"https://{location}-aiplatform.googleapis.com/reasoningEngines/v1/"
        f"{runtime_resource}/api"
    )


class AgentEndpoints(NamedTuple):
    """HTTP and WebSocket base URLs for one remote agent."""

    http_base: str
    ws_base: str


def http_to_ws_url(base_url: str) -> str:
    """Normalize a URL's scheme to ws/wss"""
    parts = urlsplit(base_url)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    return urlunsplit(parts._replace(scheme=scheme))


def build_agent_runtime_endpoints(location: str, runtime_resource: str) -> AgentEndpoints:
    """Build Agent Runtime ingress base URLs for a deployed engine."""
    host = f"{location}-{_AGENT_ENGINE_URL_FRAGMENT}"
    resource = runtime_resource.strip("/")
    return AgentEndpoints(
        http_base=build_agent_runtime_passthrough_url(location, resource),
        ws_base=f"wss://{host}/{_AGENT_RUNTIME_WS_INGRESS}/{resource}/api",
    )


def resolve_agent_endpoints(url: str) -> AgentEndpoints:
    """Resolve the HTTP and WebSocket base URLs for a remote agent."""
    http_url = url.rstrip("/")
    if not is_agent_runtime_url(http_url):
        return AgentEndpoints(http_base=http_url, ws_base=http_to_ws_url(http_url))

    validate_agent_runtime_url(http_url)
    location, resource = parse_agent_runtime_service_url(http_url)
    return build_agent_runtime_endpoints(location, resource.removesuffix("/api"))


def parse_agent_runtime_service_url(service_url: str) -> tuple[str, str]:
    """Split an Agent Runtime service URL into (location, runtime_resource).

    Handles both the ``/v1/`` and ``/v1beta1/`` API path variants.
    ``https://europe-west1-aiplatform.googleapis.com/v1/projects/123/locations/
    europe-west1/reasoningEngines/456`` →
    ``("europe-west1", "projects/123/locations/europe-west1/reasoningEngines/456")``.
    """
    host = urlparse(service_url).hostname or ""
    location = host.split(f"-{_AGENT_ENGINE_URL_FRAGMENT}", 1)[0]
    # Resource path follows the API version segment (v1, v1beta1, ...).
    match = re.search(r"/v1[^/]*/(.+)", service_url)
    runtime_resource = match.group(1) if match else service_url
    return location, runtime_resource


def _parse_header(value: str) -> tuple[str, str]:
    """Parse a ``Key: Value`` header string."""
    if ":" not in value:
        raise click.BadParameter(
            f"Invalid header format (expected 'Key: Value'): {value}"
        )
    key, _, val = value.partition(":")
    return key.strip(), val.strip()


def build_remote_headers(
    custom_headers: tuple[str, ...], url: str = ""
) -> dict[str, str]:
    """Build headers for remote requests.

    Auto-detects Google Cloud credentials unless the caller supplies
    an ``Authorization`` header via ``--header``.

    Uses an **access token** for Gemini platform / Agent Runtime URLs and an
    **identity token** (with the service URL as audience) for everything
    else (Cloud Run, GKE, etc.). Loopback URLs (the local dev server) get
    no token.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    parsed = dict(_parse_header(h) for h in custom_headers)
    parts = _parse_remote_url(url) if url else None

    if "Authorization" not in parsed and not _is_loopback_url(url):
        if parts is not None and parts.scheme != "https":
            logging.warning(
                "Refusing to send Google credentials over insecure URL: %s", url
            )
        else:
            try:
                if parts is not None and is_agent_runtime_url(url):
                    token = get_access_token()
                else:
                    audience = (
                        f"{parts.scheme}://{parts.netloc}" if parts is not None else ""
                    )
                    token = get_id_token(audience)
                headers["Authorization"] = f"Bearer {token}"
            except Exception as exc:
                logging.warning("Could not obtain credentials: %s", exc)

    headers.update(parsed)
    return headers


def _is_loopback_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    # *.localhost always resolves to loopback (RFC 6761).
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # 0.0.0.0 / :: is what local servers print as their bind address.
    return ip.is_loopback or ip.is_unspecified
