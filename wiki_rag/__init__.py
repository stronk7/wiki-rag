#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Convenient constant for the project root directory.
ROOT_DIR = Path(__file__).resolve().parent.parent

__version__ = "unknown"
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
