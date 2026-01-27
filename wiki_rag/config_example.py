#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Example showing how to migrate existing code to use the new configuration system."""

from wiki_rag.config import get_config

# Old way (using os.getenv):
# import os
# openai_api_key = os.getenv('OPENAI_API_KEY')
# mediawiki_url = os.getenv('MEDIAWIKI_URL')

# New way (using config system with BC support):
config = get_config()

# Access configuration values using dot notation
openai_api_key = config.get('openai.api_key')
mediawiki_url = config.get('mediawiki.url')
embedding_model = config.get('models.embedding')

# You can also provide default values
max_turns = config.get('wrapper.chat_max_turns', 10)

# Check if using YAML or dotenv
if config.is_using_yaml():
    print("Using YAML configuration")
else:
    print("Using dotenv configuration (backward compatibility mode)")

# Get entire configuration as dictionary
config_dict = config.to_dict()

# Example of how to update existing modules:
def example_function():
    """Example function showing configuration usage."""
    config = get_config()
    
    # Access nested configuration
    openai_config = config.get('openai', {})
    api_base = openai_config.get('api_base', 'https://openai.com/v1')
    api_key = openai_config.get('api_key', '')
    
    # Access mediawiki configuration
    mediawiki_config = config.get('mediawiki', {})
    namespaces = mediawiki_config.get('namespaces', [0, 4, 12])
    excluded = mediawiki_config.get('excluded', '')
    
    # Access model configuration
    models_config = config.get('models', {})
    embedding_model = models_config.get('embedding', '')
    embedding_dims = models_config.get('embedding_dimensions', 768)
    llm_model = models_config.get('llm', '')
    
    # Use the configuration values...
    print(f"Using OpenAI API base: {api_base}")
    print(f"Processing namespaces: {namespaces}")
    print(f"Using embedding model: {embedding_model}")


if __name__ == '__main__':
    example_function()