#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.config tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from pathlib import Path

import importlib.util

from wiki_rag.config import load_config_to_env


_HAS_YAML = importlib.util.find_spec("yaml") is not None
_HAS_DOTENV = importlib.util.find_spec("dotenv") is not None


@unittest.skipUnless(_HAS_YAML and _HAS_DOTENV, "pyyaml/python-dotenv dependencies not installed")
class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_precedence_env_over_yaml_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # YAML sets foo=from_yaml.
            (root / "config.yaml").write_text("storage:\n  collection_name: from_yaml\n", encoding="utf-8")

            # .env sets foo=from_dotenv.
            (root / ".env").write_text("COLLECTION_NAME=from_dotenv\n", encoding="utf-8")

            # OS env should win.
            os.environ["COLLECTION_NAME"] = "from_env"

            load_config_to_env(root_dir=root)

            self.assertEqual(os.environ.get("COLLECTION_NAME"), "from_env")

    def test_yaml_beats_dotenv_when_env_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "config.yaml").write_text("storage:\n  collection_name: from_yaml\n", encoding="utf-8")
            (root / ".env").write_text("COLLECTION_NAME=from_dotenv\n", encoding="utf-8")

            os.environ.pop("COLLECTION_NAME", None)

            load_config_to_env(root_dir=root)

            self.assertEqual(os.environ.get("COLLECTION_NAME"), "from_yaml")


if __name__ == "__main__":
    unittest.main()
