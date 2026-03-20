#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.load.util tests."""

import json
import tempfile
import unittest

from datetime import UTC, datetime
from pathlib import Path

from wiki_rag.load.util import merge_incremental_pages, save_parsed_pages


def _make_page(page_id: int, title: str = "Test Page") -> dict:
    """Create a minimal page dict for testing."""
    return {
        "id": page_id,
        "title": title,
        "sections": [],
        "categories": [],
        "templates": [],
        "internal_links": [],
        "external_links": [],
        "language_links": [],
    }


class TestMergeIncrementalPages(unittest.TestCase):

    def test_merge_incremental_pages_created(self):
        """Page in parsed but not in base gets change_type='created'."""
        base_pages: dict = {}
        parsed_pages = [_make_page(1, "New Page")]
        result = merge_incremental_pages(base_pages, parsed_pages, {}, {})
        self.assertEqual(1, len(result))
        self.assertEqual("created", result[0]["change_type"])

    def test_merge_incremental_pages_updated(self):
        """Page in parsed and in base gets change_type='updated'."""
        base_pages = {1: _make_page(1, "Old Title")}
        parsed_pages = [_make_page(1, "New Title")]
        result = merge_incremental_pages(base_pages, parsed_pages, {}, {1: {"rev_id": 2}})
        self.assertEqual(1, len(result))
        self.assertEqual("updated", result[0]["change_type"])
        self.assertEqual("New Title", result[0]["title"])

    def test_merge_incremental_pages_deleted(self):
        """Base page with final log state 'delete' gets change_type='deleted'."""
        base_pages = {1: _make_page(1, "Deleted Page")}
        parsed_pages: list = []
        final_log_states = {1: "delete"}
        result = merge_incremental_pages(base_pages, parsed_pages, final_log_states, {})
        self.assertEqual(1, len(result))
        self.assertEqual("deleted", result[0]["change_type"])

    def test_merge_incremental_pages_already_deleted_in_base_is_dropped(self):
        """Base page already carrying change_type='deleted' is excluded from the merged result."""
        already_deleted = {**_make_page(1, "Old Deleted Page"), "change_type": "deleted"}
        base_pages = {1: already_deleted}
        parsed_pages: list = []
        result = merge_incremental_pages(base_pages, parsed_pages, {}, {})
        self.assertEqual(0, len(result))

    def test_merge_incremental_pages_already_deleted_in_base_not_restored(self):
        """Base page with change_type='deleted' is excluded even when not in final_log_states."""
        already_deleted = {**_make_page(2, "Stale Deleted"), "change_type": "deleted"}
        unchanged = _make_page(3, "Unchanged")
        base_pages = {2: already_deleted, 3: unchanged}
        result = merge_incremental_pages(base_pages, [], {}, {})
        self.assertEqual(1, len(result))
        self.assertEqual(3, result[0]["id"])

    def test_merge_incremental_pages_unchanged(self):
        """Base page not revised and not deleted gets change_type=None."""
        base_pages = {1: _make_page(1, "Unchanged Page")}
        parsed_pages: list = []
        result = merge_incremental_pages(base_pages, parsed_pages, {}, {})
        self.assertEqual(1, len(result))
        self.assertIsNone(result[0]["change_type"])

    def test_merge_incremental_pages_prior_created_resets_to_none(self):
        """Base page with change_type='created' (from prior incremental) resets to None when untouched."""
        prior_created = {**_make_page(1, "New Page"), "change_type": "created"}
        base_pages = {1: prior_created}
        result = merge_incremental_pages(base_pages, [], {}, {})
        self.assertEqual(1, len(result))
        self.assertIsNone(result[0]["change_type"])

    def test_merge_incremental_pages_prior_updated_resets_to_none(self):
        """Base page with change_type='updated' (from prior incremental) resets to None when untouched."""
        prior_updated = {**_make_page(1, "Updated Page"), "change_type": "updated"}
        base_pages = {1: prior_updated}
        result = merge_incremental_pages(base_pages, [], {}, {})
        self.assertEqual(1, len(result))
        self.assertIsNone(result[0]["change_type"])

    def test_merge_incremental_pages_mixed(self):
        """Combination of created, updated, deleted, unchanged, and previously-deleted pages."""
        already_deleted = {**_make_page(5, "Prev Deleted Page"), "change_type": "deleted"}
        base_pages = {
            2: _make_page(2, "Updated Page"),
            3: _make_page(3, "Deleted Page"),
            4: _make_page(4, "Unchanged Page"),
            5: already_deleted,
        }
        parsed_pages = [
            _make_page(1, "Created Page"),
            _make_page(2, "Updated Page v2"),
        ]
        final_log_states = {3: "delete"}
        revised_page_ids = {2: {"rev_id": 5, "timestamp": "2026-01-01T00:00:00Z", "title": "Updated Page v2", "ns": 0}}
        result = merge_incremental_pages(base_pages, parsed_pages, final_log_states, revised_page_ids)
        self.assertEqual(4, len(result))
        by_id = {p["id"]: p for p in result}
        self.assertEqual("created", by_id[1]["change_type"])
        self.assertEqual("updated", by_id[2]["change_type"])
        self.assertEqual("deleted", by_id[3]["change_type"])
        self.assertIsNone(by_id[4]["change_type"])
        self.assertNotIn(5, by_id)


class TestSaveParsedPages(unittest.TestCase):

    def _load_output(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def test_save_parsed_pages_full_dump_type(self):
        """Full dump saves dump_type='full' and base_dump=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test-dump.json"
            ts = datetime.now(UTC).replace(microsecond=0)
            save_parsed_pages([], output_file, ts, "https://example.com")
            data = self._load_output(output_file)
            self.assertEqual("full", data["sites"][0]["dump_type"])
            self.assertIsNone(data["sites"][0]["base_dump"])

    def test_save_parsed_pages_incremental_dump_type(self):
        """Incremental dump saves dump_type='incremental' and base_dump filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test-dump.json"
            ts = datetime.now(UTC).replace(microsecond=0)
            save_parsed_pages(
                [], output_file, ts, "https://example.com",
                dump_type="incremental", base_dump="collection-2026-01-01-00-00.json",
            )
            data = self._load_output(output_file)
            self.assertEqual("incremental", data["sites"][0]["dump_type"])
            self.assertEqual("collection-2026-01-01-00-00.json", data["sites"][0]["base_dump"])

    def test_save_parsed_pages_adds_change_type(self):
        """Pages without change_type get change_type=None after saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test-dump.json"
            ts = datetime.now(UTC).replace(microsecond=0)
            pages = [_make_page(1, "Page One")]
            save_parsed_pages(pages, output_file, ts, "https://example.com")
            data = self._load_output(output_file)
            self.assertIsNone(data["sites"][0]["pages"][0]["change_type"])


if __name__ == "__main__":
    unittest.main()
