#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Utility to convert .env to config.yaml."""

import os
from pathlib import Path

import yaml
from dotenv import dotenv_values


def main():
    """Convert .env file to config.yaml."""
    env_path = Path(".env")
    yaml_path = Path("config.yaml")

    if not env_path.exists():
        print(f"Error: {env_path} not found.")
        return

    if yaml_path.exists():
        print(f"Warning: {yaml_path} already exists. It will be overwritten.")
        confirm = input("Do you want to continue? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    # Load values from .env (without system environment variables)
    config = dotenv_values(env_path)
    
    # Convert OrderedDict to dict for clean YAML output
    config_dict = dict(config)
    
    # Process specific types if needed, but for a direct dump, keeping strings is safest
    # to preserve exact behavior unless we want to infer types.
    # Users can manually clean up the YAML types later.
    
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully converted {env_path} to {yaml_path}")
    except Exception as e:
        print(f"Error writing to {yaml_path}: {e}")

if __name__ == "__main__":
    main()
