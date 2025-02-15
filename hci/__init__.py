#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""hci package."""

import os

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Ugly constant, but nice to have for easy testing.
ROOT_DIR = Path(__file__).resolve().parent.parent

# To configure logging level globally. Note this cannot be set in config file / .env. Only via env variable.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

__version__ = "unknown"
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
