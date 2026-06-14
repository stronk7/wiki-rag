#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for the index-time section chunking strategies."""

import unittest

from wiki_rag.index.chunking import split_section


class TestSplitSectionNone(unittest.TestCase):
    """Strategy "none" preserves the historical trim-at-limit behaviour."""

    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(["short text"], split_section("short text", strategy="none", max_bytes=100))

    def test_oversized_text_is_trimmed_to_byte_limit(self):
        text = "á" * 60  # 120 UTF-8 bytes.
        [result] = split_section(text, strategy="none", max_bytes=99)
        # 99 bytes cuts the 50th "á" mid-codepoint, so 49 chars survive.
        self.assertEqual("á" * 49, result)
        self.assertLessEqual(len(result.encode("utf-8")), 99)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual([], split_section("   ", strategy="none", max_bytes=100))


class TestSplitSectionFixed(unittest.TestCase):
    """Strategy "fixed" splits at max_bytes on whitespace boundaries with overlap."""

    def test_short_text_yields_single_chunk(self):
        self.assertEqual(["short text"], split_section("short text", strategy="fixed", max_bytes=100))

    def test_every_chunk_respects_the_byte_limit(self):
        text = ("word " * 1000).strip()
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 200)

    def test_no_content_is_lost_without_overlap(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=0)
        self.assertEqual(text, " ".join(chunks))

    def test_overlap_repeats_content_between_consecutive_chunks(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_section(text, strategy="fixed", max_bytes=200, overlap_bytes=40)
        self.assertGreater(len(chunks), 1)
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            prev_words = set(prev.split())
            next_lead_words = set(nxt.split()[:8])
            # Unique words, so any intersection proves repeated (overlapping) content.
            self.assertTrue(prev_words & next_lead_words)

    def test_multibyte_characters_never_split_mid_codepoint(self):
        text = "ñ" * 500  # 1000 bytes, no whitespace at all.
        chunks = split_section(text, strategy="fixed", max_bytes=99, overlap_bytes=0)
        self.assertEqual(text, "".join(chunks))
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 99)


class TestSplitSectionParagraph(unittest.TestCase):
    """Strategy "paragraph" packs whole paragraphs greedily up to max_bytes."""

    def test_paragraphs_are_packed_without_splitting_them(self):
        paragraphs = [("alpha " * 10).strip(), ("beta " * 10).strip(), ("gamma " * 10).strip()]
        text = "\n\n".join(paragraphs)
        chunks = split_section(text, strategy="paragraph", max_bytes=130)
        self.assertEqual(2, len(chunks))
        self.assertEqual("\n\n".join(paragraphs[:2]), chunks[0])
        self.assertEqual(paragraphs[2], chunks[1])

    def test_single_oversized_paragraph_falls_back_to_fixed_split(self):
        text = ("word " * 200).strip()  # One paragraph, ~1000 bytes.
        chunks = split_section(text, strategy="paragraph", max_bytes=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 300)

    def test_text_within_limit_is_kept_whole(self):
        text = "first paragraph\n\nsecond paragraph"
        self.assertEqual([text], split_section(text, strategy="paragraph", max_bytes=100))

    def test_every_chunk_respects_the_byte_limit(self):
        text = "\n\n".join(("püra " * 30).strip() for _ in range(20))
        chunks = split_section(text, strategy="paragraph", max_bytes=400)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 400)


class TestSplitSectionValidation(unittest.TestCase):
    """Invalid arguments are rejected."""

    def test_unknown_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            split_section("text", strategy="bogus", max_bytes=100)
