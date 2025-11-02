#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.vector package."""

# Make various classes and functions available at the package level.
from .base import BaseVector, load_vector_store

# Globally available vector store to be used by various applications.
store: BaseVector
