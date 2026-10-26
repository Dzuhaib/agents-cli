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

"""Regression tests for security hardening outside the remote URL path."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import click

from google.agents.cli.extension import _overrides
from google.agents.cli.extension._refs import RefParseError, parse_ref
from google.agents.cli.run import _local_server


class RemoteTemplatePathTest(unittest.TestCase):
    def test_relative_path_is_accepted(self):
        from google.agents.cli.scaffold.utils.remote_template import (
            _safe_template_relpath,
        )

        self.assertEqual(_safe_template_relpath("templates/python"), "templates/python")
        self.assertEqual(_safe_template_relpath(""), "")

    def test_absolute_and_parent_paths_are_rejected(self):
        from google.agents.cli.scaffold.utils.remote_template import (
            _safe_template_relpath,
        )

        for value in (
            "../../outside",
            "a/../../outside",
            "C:/Windows/System32",
            "C:\\Windows\\System32",
            "/etc/passwd",
            "..\\outside",
        ):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    _safe_template_relpath(value)


class PidFileTest(unittest.TestCase):
    def test_read_pid_file_rejects_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".google-agents-cli").mkdir()
            path = root / ".google-agents-cli" / "run_server.json"
            for payload in (
                [],
                {"pid": "123", "port": 18080},
                {"pid": 0, "port": 18080},
                {"pid": 123, "port": 70000},
                {"pid": True, "port": 18080},
            ):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    self.assertIsNone(_local_server._read_pid_file(root))

    def test_current_process_is_not_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(
                _local_server._is_owned_server_process(
                    os.getpid(), 18080, Path(tmp)
                )
            )

    def test_arbitrary_process_is_not_owned(self):
        process = mock.Mock()
        process.cwd.return_value = tempfile.gettempdir()
        process.cmdline.return_value = [sys.executable, "-c", "print('x')"]
        with mock.patch.object(_local_server.psutil, "Process", return_value=process):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertFalse(
                    _local_server._is_owned_server_process(123, 18080, Path(tmp))
                )

    def test_process_must_have_server_marker_and_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = mock.Mock()
            process.cwd.return_value = str(root)
            process.cmdline.return_value = [
                "uv",
                "run",
                "adk",
                "api_server",
                "--port",
                "18080",
            ]
            with mock.patch.object(
                _local_server.psutil, "Process", return_value=process
            ):
                self.assertTrue(
                    _local_server._is_owned_server_process(123, 18080, root)
                )
                self.assertFalse(
                    _local_server._is_owned_server_process(123, 18081, root)
                )


class ExtensionTrustTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ, {_overrides.PROJECT_EXTENSION_TRUST_ENV: ""}, clear=False
        )
        self.env.start()
        os.environ.pop(_overrides.PROJECT_EXTENSION_TRUST_ENV, None)
        self.addCleanup(self.env.stop)

    def test_user_scope_needs_no_prompt(self):
        _overrides.require_project_extension_trust(
            scope="user", extension_name="x", display_path="run"
        )

    def test_project_scope_fails_closed_without_a_tty(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(click.ClickException):
                _overrides.require_project_extension_trust(
                    scope="project", extension_name="x", display_path="run"
                )

    def test_project_scope_can_be_declined(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            with mock.patch.object(click, "confirm", return_value=False):
                with self.assertRaises(click.Abort):
                    _overrides.require_project_extension_trust(
                        scope="project", extension_name="x", display_path="run"
                    )

    def test_environment_override_allows_project_scope(self):
        os.environ[_overrides.PROJECT_EXTENSION_TRUST_ENV] = "1"
        _overrides.require_project_extension_trust(
            scope="project", extension_name="x", display_path="run"
        )


class ExtensionSchemeTest(unittest.TestCase):
    def test_plain_http_is_rejected(self):
        with self.assertRaises(RefParseError):
            parse_ref("http://example.com/org/repo")

    def test_https_is_allowed(self):
        self.assertEqual(parse_ref("https://example.com/org/repo").kind, "git")


if __name__ == "__main__":
    unittest.main()
