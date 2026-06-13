#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.vector.milvus tests.

These exercise the schema-tolerant behaviour (collections created before a
schema field was introduced keep working) without needing a real Milvus
instance: ``MilvusClient`` is patched out entirely.
"""

import unittest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from wiki_rag.vector.milvus import MilvusVector


def _make_cfg() -> SimpleNamespace:
    """Return a minimal fake Config exposing only what MilvusVector reads."""
    return SimpleNamespace(
        milvus=SimpleNamespace(url="http://localhost:19530", timeout=10.0),
        milvus_token="",
    )


def _describe(*field_names: str) -> dict:
    """Return a fake describe_collection payload listing the given fields."""
    return {"fields": [{"name": name} for name in field_names]}


# A collection holding the full current schema: every requested output field
# plus the (non-output) vector fields.
_CURRENT_FIELDS = (*MilvusVector.OUTPUT_FIELDS, "dense_vector", "sparse_vector")


class CollectionFieldsCacheTest(unittest.TestCase):
    """Tests for the per-instance schema cache."""

    def test_collection_fields_caches_describe_call(self):
        """describe_collection is issued once per collection, then cached."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        client = MagicMock()
        client.describe_collection.return_value = _describe("id", "text", "dense_vector")

        first = vector._collection_fields(client, "coll")
        second = vector._collection_fields(client, "coll")

        self.assertEqual(["id", "text", "dense_vector"], first)
        self.assertEqual(first, second)
        self.assertEqual(1, client.describe_collection.call_count)

    def test_drop_collection_invalidates_cache(self):
        """Dropping a collection clears its cached fields and warning state."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        vector._fields_cache["coll"] = list(_CURRENT_FIELDS)
        vector._warned_dropped_fields.add("coll")

        with patch("wiki_rag.vector.milvus.MilvusClient"):
            vector.drop_collection("coll")

        self.assertNotIn("coll", vector._fields_cache)
        self.assertNotIn("coll", vector._warned_dropped_fields)

    def test_rename_collection_moves_cache(self):
        """Renaming a collection moves its cached state to the new name."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        vector._fields_cache["old"] = list(_CURRENT_FIELDS)
        vector._warned_dropped_fields.add("old")

        with patch("wiki_rag.vector.milvus.MilvusClient"):
            vector.rename_collection("old", "new")

        self.assertNotIn("old", vector._fields_cache)
        self.assertEqual(list(_CURRENT_FIELDS), vector._fields_cache["new"])
        self.assertNotIn("old", vector._warned_dropped_fields)
        self.assertIn("new", vector._warned_dropped_fields)


class RetrieveOutputFieldsTest(unittest.TestCase):
    """Tests that retrieve() only requests output fields the collection has."""

    def _run_retrieve(self, schema_fields: tuple[str, ...]) -> list[str]:
        """Run retrieve() against a mocked collection, return the output_fields used."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        # Avoid any real embedding call.
        vector._embed_and_average_queries = MagicMock(return_value=[0.1, 0.2])  # type: ignore[method-assign]

        client = MagicMock()
        client.describe_collection.return_value = _describe(*schema_fields)
        # hybrid_search returns a list with one result group (here empty).
        client.hybrid_search.return_value = [[]]

        with patch("wiki_rag.vector.milvus.MilvusClient", return_value=client):
            vector.retrieve(
                collection_name="coll",
                embedding_model="model",
                embedding_dimensions=2,
                queries=["a question"],
            )

        _, kwargs = client.hybrid_search.call_args
        return kwargs["output_fields"]

    def test_legacy_collection_omits_missing_field(self):
        """A collection lacking one of the output fields must not request it."""
        # Simulate a collection created before "relations" existed.
        legacy_fields = tuple(f for f in _CURRENT_FIELDS if f != "relations")
        output_fields = self._run_retrieve(legacy_fields)

        self.assertNotIn("relations", output_fields)
        self.assertIn("id", output_fields)
        self.assertIn("page_id", output_fields)

    def test_full_collection_requests_all_output_fields(self):
        """A collection with the full schema requests every output field, in order."""
        output_fields = self._run_retrieve(_CURRENT_FIELDS)

        self.assertEqual(list(MilvusVector.OUTPUT_FIELDS), output_fields)


class InsertBatchToleranceTest(unittest.TestCase):
    """Tests that insert_batch() drops unknown keys and warns once."""

    def _record(self) -> dict:
        """Return a record carrying a key the legacy schema does not know about."""
        return {
            "id": "sec-1",
            "title": "T",
            "text": "body",
            "future_field": ["some value"],
        }

    def test_unknown_keys_dropped_on_legacy_collection(self):
        """Keys absent from the schema are stripped before insert."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        client = MagicMock()
        client.describe_collection.return_value = _describe("id", "title", "text")

        with patch("wiki_rag.vector.milvus.MilvusClient", return_value=client), \
                self.assertLogs("wiki_rag.vector.milvus", level="WARNING") as logs:
            vector.insert_batch("coll", [self._record()])

        (_, inserted), _ = client.insert.call_args
        self.assertEqual(1, len(inserted))
        self.assertNotIn("future_field", inserted[0])
        self.assertEqual({"id", "title", "text"}, set(inserted[0].keys()))
        self.assertIn("wr-index --full", logs.output[0])

    def test_known_keys_preserved(self):
        """When the schema has every key, nothing is stripped."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        client = MagicMock()
        client.describe_collection.return_value = _describe("id", "title", "text", "future_field")

        with patch("wiki_rag.vector.milvus.MilvusClient", return_value=client):
            vector.insert_batch("coll", [self._record()])

        (_, inserted), _ = client.insert.call_args
        self.assertIn("future_field", inserted[0])

    def test_dropped_fields_warns_once_per_collection(self):
        """The 'run wr-index --full' warning is logged once, not per batch."""
        vector = MilvusVector(_make_cfg())  # type: ignore[arg-type]
        client = MagicMock()
        client.describe_collection.return_value = _describe("id", "title", "text")

        with patch("wiki_rag.vector.milvus.MilvusClient", return_value=client), \
                patch("wiki_rag.vector.milvus.logger") as mock_logger:
            vector.insert_batch("coll", [self._record()])
            vector.insert_batch("coll", [self._record()])

        self.assertEqual(1, mock_logger.warning.call_count)
        self.assertIn("coll", vector._warned_dropped_fields)


if __name__ == "__main__":
    unittest.main()
