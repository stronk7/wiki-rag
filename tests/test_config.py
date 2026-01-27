#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for configuration management."""

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from wiki_rag.config import Config, ConfigError, get_config, reload_config


class TestConfig(unittest.TestCase):
    """Test cases for configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
        
        # Clear environment variables
        for key in list(os.environ.keys()):
            if key.startswith(('OPENAI_', 'MEDIAWIKI_', 'COLLECTION_', 'MILVUS_', 
                              'EMBEDDING_', 'LLM_', 'WRAPPER_', 'MCP_', 'AUTH_', 
                              'LANGSMITH_', 'LANGFUSE_', 'USER_AGENT', 'ENABLE_RATE_LIMITING')):
                del os.environ[key]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = {
            'openai': {
                'api_base': 'https://test.openai.com',
                'api_key': 'test_key'
            },
            'mediawiki': {
                'url': 'https://test.wiki.com',
                'namespaces': [0, 1, 2]
            }
        }
        
        yaml_path = Path(self.temp_dir) / 'config.yml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        config = Config(yaml_path)
        
        self.assertTrue(config.is_using_yaml())
        self.assertEqual(config.get('openai.api_base'), 'https://test.openai.com')
        self.assertEqual(config.get('openai.api_key'), 'test_key')
        self.assertEqual(config.get('mediawiki.url'), 'https://test.wiki.com')
        self.assertEqual(config.get('mediawiki.namespaces'), [0, 1, 2])
    
    def test_dotenv_fallback(self):
        """Test fallback to dotenv when YAML not found."""
        # Set environment variables
        os.environ['OPENAI_API_KEY'] = 'env_key'
        os.environ['MEDIAWIKI_URL'] = 'https://env.wiki.com'
        os.environ['EMBEDDING_DIMENSIONS'] = '512'
        
        config = Config()  # No YAML file exists
        
        self.assertFalse(config.is_using_yaml())
        self.assertEqual(config.get('openai.api_key'), 'env_key')
        self.assertEqual(config.get('mediawiki.url'), 'https://env.wiki.com')
        self.assertEqual(config.get('models.embedding_dimensions'), 512)
    
    def test_get_with_default(self):
        """Test getting configuration values with defaults."""
        yaml_content = {'test': {'value': 'present'}}
        yaml_path = Path(self.temp_dir) / 'config.yml'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        config = Config(yaml_path)
        
        # Existing value
        self.assertEqual(config.get('test.value'), 'present')
        
        # Non-existing value with default
        self.assertEqual(config.get('test.missing', 'default'), 'default')
        
        # Non-existing nested path with default
        self.assertEqual(config.get('missing.path', 'nested_default'), 'nested_default')
    
    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test invalid YAML file
        yaml_path = Path(self.temp_dir) / 'invalid.yml'
        with open(yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with self.assertRaises(ConfigError):
            Config(yaml_path)
    
    def test_global_config_instance(self):
        """Test global configuration instance management."""
        # Clear any existing global config
        global _config
        _config = None
        
        # Set up environment
        os.environ['OPENAI_API_KEY'] = 'global_test_key'
        
        # Get config instance
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same instance
        self.assertIs(config1, config2)
        self.assertEqual(config1.get('openai.api_key'), 'global_test_key')
        
        # Reload should create new instance
        config3 = reload_config()
        self.assertIsNot(config1, config3)
        self.assertEqual(config3.get('openai.api_key'), 'global_test_key')


class TestConfigUpdate(unittest.TestCase):
    """Test cases for configuration update utility."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_parse_comma_list(self):
        """Test parsing comma-separated lists."""
        from wiki_rag.config_update import _parse_comma_list
        
        # Test various formats
        self.assertEqual(_parse_comma_list('a,b,c'), ['a', 'b', 'c'])
        self.assertEqual(_parse_comma_list('a, b, c'), ['a', 'b', 'c'])
        self.assertEqual(_parse_comma_list(''), [])
        self.assertEqual(_parse_comma_list('single'), ['single'])
    
    def test_parse_bool(self):
        """Test parsing boolean values."""
        from wiki_rag.config_update import _parse_bool
        
        # Test various boolean formats
        self.assertTrue(_parse_bool('true'))
        self.assertTrue(_parse_bool('True'))
        self.assertTrue(_parse_bool('1'))
        self.assertTrue(_parse_bool('yes'))
        self.assertTrue(_parse_bool('on'))
        
        self.assertFalse(_parse_bool('false'))
        self.assertFalse(_parse_bool('False'))
        self.assertFalse(_parse_bool('0'))
        self.assertFalse(_parse_bool('no'))
        self.assertFalse(_parse_bool('off'))
    
    def test_cleanup_config(self):
        """Test configuration cleanup."""
        from wiki_rag.config_update import _cleanup_config
        
        # Test with empty values
        dirty_config = {
            'section1': {
                'key1': 'value1',
                'key2': '',
                'key3': 0,
                'key4': []
            },
            'section2': {
                'key5': None,
                'key6': 'value6'
            },
            'section3': {}
        }
        
        cleaned = _cleanup_config(dirty_config)
        
        # Should keep value1, value6, and 0, but remove empty strings and None
        self.assertIn('section1', cleaned)
        self.assertIn('key1', cleaned['section1'])
        self.assertIn('key3', cleaned['section1'])
        self.assertNotIn('key2', cleaned['section1'])
        self.assertIn('section2', cleaned)
        self.assertIn('key6', cleaned['section2'])
        self.assertNotIn('key5', cleaned['section2'])


if __name__ == '__main__':
    unittest.main()