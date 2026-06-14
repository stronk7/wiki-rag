#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Section chunking strategies applied at index time.

All limits are measured in UTF-8 **bytes**, not characters, because the
vector-store varchar limits are byte-based (see the Milvus ``text`` field).
Splits never break a multi-byte codepoint and prefer whitespace boundaries.
"""

import logging

logger = logging.getLogger(__name__)

STRATEGIES = ("none", "fixed", "paragraph")


def split_section(
        text: str,
        *,
        strategy: str,
        max_bytes: int,
        overlap_bytes: int = 0,
) -> list[str]:
    """Split a section body into chunks according to the given strategy.

    Args:
        text: Tidied section body text.
        strategy: One of ``"none"`` (trim at limit, historical behaviour),
            ``"fixed"`` (hard split on whitespace boundaries with overlap)
            or ``"paragraph"`` (greedy paragraph packing, fixed fallback
            for oversized paragraphs, no overlap).
        max_bytes: Maximum UTF-8 byte length of every returned chunk.
        overlap_bytes: Approximate number of trailing bytes from the previous
            chunk repeated at the start of the next one (``"fixed"`` only).

    Returns:
        List of non-empty chunk texts, in document order. Empty input
        yields an empty list.

    Raises:
        ValueError: If *strategy* is not one of :data:`STRATEGIES`.

    """
    if strategy not in STRATEGIES:
        msg = f"Unknown chunking strategy: {strategy!r}. Valid: {STRATEGIES}"
        raise ValueError(msg)

    text = text.strip()
    if not text:
        return []

    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    if strategy == "none":
        return [_trim_to_bytes(text, max_bytes)]
    if strategy == "fixed":
        return _split_fixed(text, max_bytes, overlap_bytes)
    return _split_paragraph(text, max_bytes)


def _trim_to_bytes(text: str, max_bytes: int) -> str:
    """Trim *text* to at most *max_bytes* UTF-8 bytes on a codepoint boundary."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").strip()


def _split_fixed(text: str, max_bytes: int, overlap_bytes: int) -> list[str]:
    """Split on the nearest whitespace at or before *max_bytes*, with overlap."""
    # Clamp the overlap so every iteration always consumes more new text than
    # it repeats, guaranteeing termination even with extreme settings.
    overlap_bytes = min(overlap_bytes, max_bytes // 4)

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining.encode("utf-8")) <= max_bytes:
            chunks.append(remaining)
            break
        head = _trim_to_bytes(remaining, max_bytes)
        # Prefer a whitespace boundary so words are not cut in half, but only
        # when one exists in the tail half of the head (avoids degenerate
        # tiny chunks on whitespace-poor text).
        cut = head.rfind(" ", len(head) // 2)
        if cut > 0:
            head = head[:cut]
        chunks.append(head.strip())
        consumed = remaining[:len(head)]
        remaining = remaining[len(head):].strip()
        if overlap_bytes > 0 and remaining:
            overlap = _trim_to_bytes(consumed[-(overlap_bytes * 2):], overlap_bytes)
            # Start the overlap on a word boundary.
            overlap = overlap.split(" ", 1)[-1] if " " in overlap else overlap
            remaining = f"{overlap} {remaining}"
    return [chunk for chunk in chunks if chunk]


def _split_paragraph(text: str, max_bytes: int) -> list[str]:
    """Pack whole paragraphs greedily; oversized paragraphs use fixed split."""
    separator = "\n\n"
    separator_bytes = len(separator.encode("utf-8"))

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for paragraph in (p.strip() for p in text.split(separator)):
        if not paragraph:
            continue
        paragraph_bytes = len(paragraph.encode("utf-8"))
        if paragraph_bytes > max_bytes:
            # Flush whatever is packed, then fixed-split the big paragraph.
            if current:
                chunks.append(separator.join(current))
                current, current_bytes = [], 0
            chunks.extend(_split_fixed(paragraph, max_bytes, overlap_bytes=0))
            continue
        if current and current_bytes + separator_bytes + paragraph_bytes > max_bytes:
            chunks.append(separator.join(current))
            current, current_bytes = [], 0
        current_bytes += (separator_bytes if current else 0) + paragraph_bytes
        current.append(paragraph)
    if current:
        chunks.append(separator.join(current))
    return chunks
