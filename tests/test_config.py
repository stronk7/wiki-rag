#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for configuration management."""

import os
import unittest

import yaml

from wiki_rag.config import Config
from wiki_rag.config_update import main as migrate_main


class TestConfig(unittest.TestCase):
    def test_config_fallback(self):
        """Test that config falls back to environment variables."""
        os.environ["MILVUS_URL"] = "http://env-url:19530"
        # Ensure no config.yaml exists for this test
        if os.path.exists("config_test.yaml"):
            os.remove("config_test.yaml")
        
        cfg = Config("config_test.yaml")
        self.assertEqual(cfg.get("database.milvus_url"), "http://env-url:19530")
        
    def test_config_yaml_priority(self):
        """Test that config.yaml has priority over environment variables."""
        os.environ["MILVUS_URL"] = "http://env-url:19530"
        with open("config_test.yaml", "w") as f:
            yaml.dump({"database": {"milvus_url": "http://yaml-url:19530"}}, f)
        
        cfg = Config("config_test.yaml")
        self.assertEqual(cfg.get("database.milvus_url"), "http://yaml-url:19530")
        os.remove("config_test.yaml")

    def test_migration(self):
        """Test the migration from .env to config.yaml."""
        if os.path.exists("config.yaml"):
            os.rename("config.yaml", "config.yaml.bak")
        
        with open(".env", "w") as f:
            f.write('MILVUS_URL="http://migrated-url:19530"\n')
            
        migrate_main()
        
        cfg = Config("config.yaml")
        self.assertEqual(cfg.get("database.milvus_url"), "http://migrated-url:19530")
        
        os.remove(".env")
        os.remove("config.yaml")
        if os.path.exists("config.yaml.bak"):
            os.rename("config.yaml.bak", "config.yaml")
