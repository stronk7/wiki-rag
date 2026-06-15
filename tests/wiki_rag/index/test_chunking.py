#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for the index-time section chunking strategies."""

import unittest

from wiki_rag.index.chunking import Chunk, _normalise_separator, split_section


class TestNormaliseSeparator(unittest.TestCase):
    """_normalise_separator() maps raw whitespace to canonical values."""

    def test_empty_string_returns_empty(self):
        self.assertEqual("", _normalise_separator(""))

    def test_single_space_returns_space(self):
        self.assertEqual(" ", _normalise_separator(" "))

    def test_multiple_spaces_return_space(self):
        self.assertEqual(" ", _normalise_separator("   "))

    def test_tab_returns_space(self):
        self.assertEqual(" ", _normalise_separator("\t"))

    def test_single_newline_returns_single_newline(self):
        self.assertEqual("\n", _normalise_separator("\n"))

    def test_newline_with_spaces_returns_single_newline(self):
        self.assertEqual("\n", _normalise_separator(" \n "))

    def test_two_newlines_returns_double_newline(self):
        self.assertEqual("\n\n", _normalise_separator("\n\n"))

    def test_three_newlines_returns_double_newline(self):
        self.assertEqual("\n\n", _normalise_separator("\n\n\n"))

    def test_double_newline_with_spaces_returns_double_newline(self):
        self.assertEqual("\n\n", _normalise_separator(" \n \n "))


class TestSplitSectionNone(unittest.TestCase):
    """Strategy "none" preserves the historical trim-at-limit behaviour."""

    def test_short_text_is_returned_unchanged(self):
        [chunk] = split_section("short text", strategy="none", max_bytes=100)
        self.assertEqual("short text", chunk.text)
        self.assertEqual("short text", chunk.embed_text)
        self.assertEqual("", chunk.separator)

    def test_oversized_text_is_trimmed_to_byte_limit(self):
        text = "á" * 60  # 120 UTF-8 bytes.
        [chunk] = split_section(text, strategy="none", max_bytes=99)
        # 99 bytes cuts the 50th "á" mid-codepoint, so 49 chars survive.
        self.assertEqual("á" * 49, chunk.text)
        self.assertLessEqual(len(chunk.text.encode("utf-8")), 99)

    def test_single_chunk_has_empty_separator(self):
        [chunk] = split_section("hello world", strategy="none", max_bytes=100)
        self.assertEqual("", chunk.separator)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual([], split_section("   ", strategy="none", max_bytes=100))


class TestSplitSectionFixed(unittest.TestCase):
    """Strategy "fixed" splits at max_bytes on whitespace boundaries with embed-only overlap."""

    def test_short_text_yields_single_chunk(self):
        [chunk] = split_section("short text", strategy="fixed", max_bytes=100)
        self.assertEqual("short text", chunk.text)
        self.assertEqual("short text", chunk.embed_text)
        self.assertEqual("", chunk.separator)

    def test_every_chunk_respects_the_byte_limit(self):
        text = ("word " * 1000).strip()
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.encode("utf-8")), 200)

    def test_round_trip_text_plus_separator_reproduces_original(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=0)
        self.assertEqual(text, "".join(c.text + c.separator for c in chunks))

    def test_stored_text_is_overlap_free(self):
        """chunk.text must NOT contain words from the previous chunk."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=40)
        self.assertGreater(len(chunks), 1)
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            prev_words = set(prev.text.split())
            next_stored_words = set(nxt.text.split())
            # No word from the previous chunk should appear in the next stored body.
            self.assertFalse(prev_words & next_stored_words)

    def test_embed_text_carries_overlap_from_previous_chunk(self):
        """chunk.embed_text must share words with the preceding chunk.text."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=40)
        self.assertGreater(len(chunks), 1)
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            prev_words = set(prev.text.split())
            next_lead_words = set(nxt.embed_text.split()[:8])
            # Unique words, so any intersection proves overlapping content in embed_text.
            self.assertTrue(prev_words & next_lead_words)

    def test_first_chunk_embed_text_equals_text(self):
        """The first chunk has no predecessor, so embed_text == text."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=40)
        self.assertEqual(chunks[0].text, chunks[0].embed_text)

    def test_last_chunk_has_empty_separator(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("", chunks[-1].separator)

    def test_space_boundary_separator_is_space(self):
        """Word-boundary cuts produce a space separator."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=0)
        # All non-last separators should be " " for space-separated word text.
        for chunk in chunks[:-1]:
            self.assertEqual(" ", chunk.separator)

    def test_multibyte_characters_never_split_mid_codepoint(self):
        text = "ñ" * 500  # 1000 bytes, no whitespace at all.
        chunks = split_section(text, strategy="fixed", max_bytes=99, overlap_bytes=0)
        self.assertEqual(text, "".join(c.text + c.separator for c in chunks))
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.encode("utf-8")), 99)

    def test_mid_word_cut_separator_is_empty(self):
        """No-whitespace text produces mid-word cuts with empty separators."""
        text = "ñ" * 500
        chunks = split_section(text, strategy="fixed", max_bytes=99, overlap_bytes=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertEqual("", chunk.separator)


class TestSplitSectionParagraph(unittest.TestCase):
    """Strategy "paragraph" packs whole paragraphs greedily up to max_bytes."""

    def test_paragraphs_are_packed_without_splitting_them(self):
        paragraphs = [("alpha " * 10).strip(), ("beta " * 10).strip(), ("gamma " * 10).strip()]
        text = "\n\n".join(paragraphs)
        chunks = split_section(text, strategy="paragraph", max_bytes=130)
        self.assertEqual(2, len(chunks))
        self.assertEqual("\n\n".join(paragraphs[:2]), chunks[0].text)
        self.assertEqual(paragraphs[2], chunks[1].text)

    def test_paragraph_chunk_separators_are_double_newline(self):
        r"""Paragraph-strategy chunks join with \n\n."""
        paragraphs = [("alpha " * 10).strip(), ("beta " * 10).strip(), ("gamma " * 10).strip()]
        text = "\n\n".join(paragraphs)
        chunks = split_section(text, strategy="paragraph", max_bytes=130)
        self.assertEqual(2, len(chunks))
        self.assertEqual("\n\n", chunks[0].separator)
        self.assertEqual("", chunks[1].separator)

    def test_single_oversized_paragraph_falls_back_to_fixed_split(self):
        text = ("word " * 200).strip()  # One paragraph, ~1000 bytes.
        chunks = split_section(text, strategy="paragraph", max_bytes=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.encode("utf-8")), 300)

    def test_text_within_limit_is_kept_whole(self):
        text = "first paragraph\n\nsecond paragraph"
        [chunk] = split_section(text, strategy="paragraph", max_bytes=100)
        self.assertEqual(text, chunk.text)
        self.assertEqual("", chunk.separator)

    def test_every_chunk_respects_the_byte_limit(self):
        text = "\n\n".join(("püra " * 30).strip() for _ in range(20))
        chunks = split_section(text, strategy="paragraph", max_bytes=400)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.encode("utf-8")), 400)

    def test_last_chunk_always_has_empty_separator(self):
        text = "\n\n".join(("word " * 30).strip() for _ in range(10))
        chunks = split_section(text, strategy="paragraph", max_bytes=200)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("", chunks[-1].separator)

    def test_embed_text_equals_text_for_paragraph_strategy(self):
        """Paragraph strategy applies no overlap, so embed_text == text always."""
        text = "\n\n".join(("word " * 40).strip() for _ in range(5))
        chunks = split_section(text, strategy="paragraph", max_bytes=200)
        for chunk in chunks:
            self.assertEqual(chunk.text, chunk.embed_text)

    def test_round_trip_paragraph_chunks(self):
        """Joining text+separator across paragraph chunks reproduces the original."""
        paragraphs = [("word " * 20).strip() for _ in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = split_section(text, strategy="paragraph", max_bytes=150)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(text, "".join(c.text + c.separator for c in chunks))


class TestSplitSectionValidation(unittest.TestCase):
    """Invalid arguments are rejected."""

    def test_unknown_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            split_section("text", strategy="bogus", max_bytes=100)

    def test_chunk_is_a_dataclass(self):
        [chunk] = split_section("hello", strategy="none", max_bytes=100)
        self.assertIsInstance(chunk, Chunk)
