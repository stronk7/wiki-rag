#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.vector.milvus tests."""

import unittest

from unittest.mock import MagicMock, patch

from wiki_rag.vector.milvus import MilvusVector

NEW_SCHEMA_FIELDS = [
    {"name": "id"}, {"name": "title"}, {"name": "text"}, {"name": "source"},
    {"name": "dense_vector"}, {"name": "sparse_vector"}, {"name": "parent"},
    {"name": "children"}, {"name": "previous"}, {"name": "next"},
    {"name": "relations"}, {"name": "page_id"}, {"name": "doc_id"},
    {"name": "doc_title"}, {"name": "doc_hash"},
    {"name": "section_id"}, {"name": "chunk_index"}, {"name": "chunk_separator"},
]

# Collections that have chunking fields but predate chunk_separator (created
# between the chunking feature and this change).  Used for backward-compat tests.
NO_SEPARATOR_SCHEMA_FIELDS = [
    field for field in NEW_SCHEMA_FIELDS
    if field["name"] != "chunk_separator"
]

LEGACY_SCHEMA_FIELDS = [
    field for field in NEW_SCHEMA_FIELDS
    if field["name"] not in {"section_id", "chunk_index", "chunk_separator"}
]


def _make_vector() -> MilvusVector:
    """Return a MilvusVector without running the Config-based constructor."""
    vector = MilvusVector.__new__(MilvusVector)
    vector.uri = "http://localhost:19530"
    vector.token = ""
    vector.timeout = 30.0
    vector._fields_cache = {}
    vector._warned_dropped_fields = set()
    return vector


class TestInsertBatchFieldFiltering(unittest.TestCase):
    """insert_batch() must drop record fields unknown to the collection."""

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_unknown_fields_are_dropped_for_legacy_collections(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": LEGACY_SCHEMA_FIELDS}

        record = {"id": "abc", "title": "t", "section_id": "abc", "chunk_index": 0}
        _make_vector().insert_batch("legacy_col", [record])

        inserted = client.insert.call_args.args[1]
        self.assertEqual([{"id": "abc", "title": "t"}], inserted)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_known_fields_are_preserved_for_new_collections(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NEW_SCHEMA_FIELDS}

        record = {"id": "abc", "title": "t", "section_id": "abc", "chunk_index": 0}
        _make_vector().insert_batch("new_col", [record])

        inserted = client.insert.call_args.args[1]
        self.assertEqual([record], inserted)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_field_cache_describes_each_collection_only_once(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NEW_SCHEMA_FIELDS}

        vector = _make_vector()
        vector.insert_batch("col", [{"id": "1"}])
        vector.insert_batch("col", [{"id": "2"}])

        self.assertEqual(1, client.describe_collection.call_count)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_dropped_fields_warn_only_once_per_collection(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": LEGACY_SCHEMA_FIELDS}

        vector = _make_vector()
        with self.assertLogs("wiki_rag.vector.milvus", level="WARNING") as logs:
            vector.insert_batch("legacy_col", [{"id": "1", "section_id": "1"}])
            vector.insert_batch("legacy_col", [{"id": "2", "section_id": "2"}])
        self.assertEqual(1, len(logs.records))


class TestGetDocumentsContentsBySectionIds(unittest.TestCase):
    """get_documents_contents_by_section_ids() must reassemble chunks in order."""

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_chunks_are_reassembled_using_recorded_separators(self, mock_client_cls):
        """chunk_separator is used when the field is present in the schema."""
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NEW_SCHEMA_FIELDS}
        client.query.return_value = [
            {"section_id": "s1", "chunk_index": 2, "title": "Title", "text": "third", "chunk_separator": ""},
            {"section_id": "s1", "chunk_index": 0, "title": "Title", "text": "first", "chunk_separator": " "},
            {"section_id": "s2", "chunk_index": 0, "title": "Other", "text": "alone", "chunk_separator": ""},
            {"section_id": "s1", "chunk_index": 1, "title": "Title", "text": "second", "chunk_separator": "\n\n"},
        ]

        result = _make_vector().get_documents_contents_by_section_ids("col", ["s1", "s2"])

        self.assertEqual(
            {
                "s1": "Title\n\nfirst second\n\nthird",
                "s2": "Other\n\nalone",
            },
            result,
        )

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_chunks_without_separator_field_fall_back_to_double_newline(self, mock_client_cls):
        r"""Collections without chunk_separator join chunks with \n\n (pre-separator behaviour)."""
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NO_SEPARATOR_SCHEMA_FIELDS}
        client.query.return_value = [
            {"section_id": "s1", "chunk_index": 2, "title": "Title", "text": "third"},
            {"section_id": "s1", "chunk_index": 0, "title": "Title", "text": "first"},
            {"section_id": "s1", "chunk_index": 1, "title": "Title", "text": "second"},
        ]

        result = _make_vector().get_documents_contents_by_section_ids("col", ["s1"])

        self.assertEqual({"s1": "Title\n\nfirst\n\nsecond\n\nthird"}, result)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_chunk_separator_included_in_query_output_fields_when_available(self, mock_client_cls):
        """chunk_separator is requested only when the schema has the field."""
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NEW_SCHEMA_FIELDS}
        client.query.return_value = []

        _make_vector().get_documents_contents_by_section_ids("col", ["s1"])

        output_fields = client.query.call_args.kwargs["output_fields"]
        self.assertIn("chunk_separator", output_fields)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_chunk_separator_not_requested_for_old_chunking_schema(self, mock_client_cls):
        """No chunk_separator in output_fields for collections without the field."""
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": NO_SEPARATOR_SCHEMA_FIELDS}
        client.query.return_value = []

        _make_vector().get_documents_contents_by_section_ids("col", ["s1"])

        output_fields = client.query.call_args.kwargs["output_fields"]
        self.assertNotIn("chunk_separator", output_fields)

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_legacy_collection_falls_back_to_by_id_lookup(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.describe_collection.return_value = {"fields": LEGACY_SCHEMA_FIELDS}
        client.query.return_value = [
            {"id": "s1", "title": "Title", "text": "whole section"},
        ]

        result = _make_vector().get_documents_contents_by_section_ids("col", ["s1"])

        self.assertEqual({"s1": "Title\n\nwhole section"}, result)
        # The legacy path queries by primary key ids, not by a filter.
        self.assertEqual(["s1"], client.query.call_args.kwargs["ids"])

    @patch("wiki_rag.vector.milvus.MilvusClient")
    def test_empty_input_returns_empty_dict_without_querying(self, mock_client_cls):
        result = _make_vector().get_documents_contents_by_section_ids("col", [])
        self.assertEqual({}, result)
        mock_client_cls.assert_not_called()


class TestRetrieveOutputFields(unittest.TestCase):
    """retrieve() must only request chunking fields when the schema has them."""

    def _run_retrieve(self, schema_fields: list[dict]) -> list[str]:
        with patch("wiki_rag.vector.milvus.MilvusClient") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value = client
            client.describe_collection.return_value = {"fields": schema_fields}
            client.hybrid_search.return_value = [[]]
            vector = _make_vector()
            with patch.object(vector, "_embed_and_average_queries", return_value=[0.1] * 4):
                vector.retrieve(
                    collection_name="col",
                    embedding_model="model",
                    embedding_dimensions=4,
                    queries=["question"],
                )
            return client.hybrid_search.call_args.kwargs["output_fields"]

    def test_legacy_collection_does_not_request_chunking_fields(self):
        output_fields = self._run_retrieve(LEGACY_SCHEMA_FIELDS)
        self.assertNotIn("section_id", output_fields)
        self.assertNotIn("chunk_index", output_fields)

    def test_new_collection_requests_chunking_fields(self):
        output_fields = self._run_retrieve(NEW_SCHEMA_FIELDS)
        self.assertIn("section_id", output_fields)
        self.assertIn("chunk_index", output_fields)
        self.assertIn("chunk_separator", output_fields)
