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

"""Regression tests for URL classification and credential attachment."""

import unittest
from unittest import mock

import click

from google.agents.cli import _remote

LEGIT_AGENT_RUNTIME_URL = (
    "https://europe-west1-aiplatform.googleapis.com/v1/projects/p/"
    "locations/europe-west1/reasoningEngines/1"
)
# urllib.parse parses this as a Google host; requests connects to 127.0.0.1.
BACKSLASH_TRAVERSAL_URL = (
    "http://127.0.0.1:8080\\@europe-west1-aiplatform.googleapis.com/"
    "reasoningEngines/1"
)


class RemoteUrlSecurityTest(unittest.TestCase):
    def test_legitimate_agent_runtime_url_is_classified(self):
        self.assertTrue(_remote.is_agent_runtime_url(LEGIT_AGENT_RUNTIME_URL))
        self.assertTrue(_remote.is_legacy_agent_runtime_url(LEGIT_AGENT_RUNTIME_URL))
        self.assertFalse(
            _remote.is_legacy_agent_runtime_url(
                f"{LEGIT_AGENT_RUNTIME_URL}/api"
            )
        )

    def test_backslash_url_is_not_classified_as_agent_runtime(self):
        self.assertFalse(_remote.is_agent_runtime_url(BACKSLASH_TRAVERSAL_URL))
        self.assertFalse(_remote.is_legacy_agent_runtime_url(BACKSLASH_TRAVERSAL_URL))

    def test_validate_rejects_backslash_url(self):
        with self.assertRaises(click.UsageError):
            _remote.validate_agent_runtime_url(BACKSLASH_TRAVERSAL_URL)

    def test_validate_rejects_userinfo(self):
        with self.assertRaises(click.UsageError):
            _remote.validate_agent_runtime_url(
                "https://attacker.example@europe-west1-aiplatform.googleapis.com/"
                "reasoningEngines/1"
            )

    def test_validate_rejects_http_agent_runtime_url(self):
        with self.assertRaises(click.UsageError):
            _remote.validate_agent_runtime_url(
                "http://europe-west1-aiplatform.googleapis.com/"
                "reasoningEngines/1"
            )

    def test_validate_accepts_legitimate_url(self):
        _remote.validate_agent_runtime_url(LEGIT_AGENT_RUNTIME_URL)

    def test_build_headers_rejects_backslash_url(self):
        with self.assertRaises(click.UsageError):
            _remote.build_remote_headers((), BACKSLASH_TRAVERSAL_URL)

    def test_build_headers_uses_access_token_for_agent_runtime(self):
        with mock.patch.object(
            _remote, "get_access_token", return_value="ACCESS"
        ) as access:
            headers = _remote.build_remote_headers((), LEGIT_AGENT_RUNTIME_URL)
        self.assertEqual(headers["Authorization"], "Bearer ACCESS")
        access.assert_called_once_with()

    def test_build_headers_uses_id_token_for_https_service(self):
        with mock.patch.object(
            _remote, "get_id_token", return_value="ID"
        ) as identity:
            headers = _remote.build_remote_headers((), "https://example.com/path")
        self.assertEqual(headers["Authorization"], "Bearer ID")
        identity.assert_called_once_with("https://example.com")

    def test_build_headers_skips_token_for_plain_http_service(self):
        with mock.patch.object(_remote, "get_id_token") as identity:
            with self.assertLogs(level="WARNING"):
                headers = _remote.build_remote_headers((), "http://example.com")
        self.assertNotIn("Authorization", headers)
        identity.assert_not_called()

    def test_build_headers_skips_token_for_loopback(self):
        with mock.patch.object(_remote, "get_id_token") as identity:
            headers = _remote.build_remote_headers((), "http://127.0.0.1:8080")
        self.assertNotIn("Authorization", headers)
        identity.assert_not_called()

    def test_custom_authorization_header_is_preserved(self):
        headers = _remote.build_remote_headers(
            ("Authorization: Bearer custom",), LEGIT_AGENT_RUNTIME_URL
        )
        self.assertEqual(headers["Authorization"], "Bearer custom")


if __name__ == "__main__":
    unittest.main()
