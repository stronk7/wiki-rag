#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to proceed to index to some collection is a vector store / index."""

import json
import logging
import uuid

from datetime import UTC, datetime, timedelta
from pathlib import Path

from jsonschema import ValidationError, validate
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from tqdm import tqdm

import wiki_rag.vector as vector

from wiki_rag import ROOT_DIR
from wiki_rag.index.chunking import split_section

logger = logging.getLogger(__name__)


def load_parsed_information(input_file: Path) -> dict:
    """Load the parsed information from the file."""
    information = []
    try:
        with open(input_file) as f:
            information = json.load(f)
    except Exception as e:
        logger.error(f"Error loading the parsed information from {input_file}: {e}")

    # If the old format (array of pages) is detected, let's convert it to the new format,
    # (basic information in "meta" and pages in "sites").
    if isinstance(information, list):
        logger.warning(f"Old format detected in {input_file}, converting to new format.")
        file_mod_time = datetime.fromtimestamp(input_file.stat().st_mtime, UTC)
        two_days_ago = file_mod_time - timedelta(days=2)  # ftime -48h so we don't miss anything on incremental index.
        information = {
            "meta": {
                "timestamp": two_days_ago.isoformat(),
                "num_sites": 1,
            },
            "sites": [
                {
                    "site_url": "unknown",
                    "dump_type": "full",
                    "base_dump": None,
                    "num_pages": len(information),
                    "pages": information,
                }
            ]
        }

    # Let's validate the schema as much as we can.
    schema = json.load(open(ROOT_DIR / "wiki_rag/schema.json"))
    try:
        validate(information, schema)
        logger.debug("Successfully parsed the JSON information")
    except ValidationError as e:
        msg = f"Error validating the JSON information from {input_file}: {e}"
        logger.error(msg)
        exit(1)

    return information


def chunking_signature(strategy: str, max_bytes: int, overlap_bytes: int) -> str:
    """Return the canonical chunking signature recorded in the marker file.

    With strategy ``"none"`` the only effective limit is the storage one
    (5000 bytes), so the signature is normalised to a constant: tweaking
    max_bytes/overlap_bytes while keeping "none" must not invalidate the
    marker (and legacy single-line markers map to this same value).

    Args:
        strategy: Chunking strategy name.
        max_bytes: Maximum chunk size in UTF-8 bytes.
        overlap_bytes: Overlap between consecutive chunks in UTF-8 bytes.

    Returns:
        The signature string, e.g. ``"paragraph:3000:300"``.

    """
    if strategy == "none":
        return "none:5000:0"
    return f"{strategy}:{max_bytes}:{overlap_bytes}"


def marker_content(dump_name: str, signature: str) -> str:
    """Return the idempotency marker file content for a just-indexed dump.

    Args:
        dump_name: Basename of the dump file that was indexed.
        signature: Chunking signature (see :func:`chunking_signature`).

    Returns:
        The two-line marker content (dump name, then chunking signature).

    """
    return f"{dump_name}\n{signature}\n"


def marker_signature(marker_text: str) -> str:
    """Return the chunking signature recorded in a marker file.

    Legacy single-line markers (written before chunking existed) carry an
    implicit ``"none:5000:0"`` signature.

    Args:
        marker_text: Raw content of the marker file.

    Returns:
        The recorded chunking signature (see :func:`chunking_signature`).

    """
    lines = marker_text.strip().splitlines()
    if len(lines) > 1:
        return lines[1].strip()
    return "none:5000:0"


def marker_matches(marker_text: str, dump_name: str, signature: str) -> bool:
    """Return True when the marker proves the dump is already indexed as configured.

    Legacy single-line markers (written before chunking existed) carry an
    implicit ``"none:5000:0"`` signature, so enabling or changing chunking
    forces the next run to actually reindex.

    Args:
        marker_text: Raw content of the marker file.
        dump_name: Basename of the dump file about to be indexed.
        signature: Current chunking signature (see :func:`chunking_signature`).

    Returns:
        True when both the dump name and the chunking signature match.

    """
    lines = marker_text.strip().splitlines()
    if not lines or lines[0].strip() != dump_name:
        return False
    return marker_signature(marker_text) == signature


def create_temp_collection_schema(collection_name: str, embedding_dimension: int) -> None:
    """Create a temporary schema for the collection."""
    vector.store.create_collection(collection_name, embedding_dimension)


def _flush_embedding_batch(
        embeddings: OpenAIEmbeddings,
        collection_name: str,
        records: list[dict],
        texts: list[str],
) -> int:
    """Embed a batch of texts, attach the vectors and insert the records.

    Embedding all *texts* in a single call (rather than one per section) is the
    throughput win: the OpenAI client sends them as one request, amortising the
    per-call latency over the whole batch.

    Args:
        embeddings: Configured embedding client.
        collection_name: Target collection.
        records: Records to insert, each still missing its ``dense_vector``.
        texts: Texts to embed, aligned 1:1 with *records*.

    Returns:
        The number of records successfully inserted (0 when the batch is empty
        or the embedding/insertion step fails).

    """
    if not records:
        return 0
    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as e:
        logger.error(f"Failed to embed a batch of {len(texts)} sections: {e}")
        return 0
    for record, dense_vector in zip(records, vectors, strict=True):
        record["dense_vector"] = dense_vector
    try:
        vector.store.insert_batch(collection_name, records)
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")
        return 0
    return len(records)


def index_pages(
        pages: list[dict],
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_api_base: str = "",
        embedding_api_key: str = "",
        embedding_max_retries: int = 3,
        embedding_batch_size: int = 8,
        chunk_strategy: str = "none",
        chunk_max_bytes: int = 5000,
        chunk_overlap_bytes: int = 0,
) -> list[int]:
    """Index the pages to the collection.

    Section bodies are split into chunks according to *chunk_strategy* (see
    :func:`wiki_rag.index.chunking.split_section`). Each chunk becomes one
    vector-store record carrying the owning ``section_id`` and its 0-based
    ``chunk_index``; the relationship fields keep referencing section ids.
    The first chunk reuses the section's own id, so single-chunk sections
    (the vast majority) produce records identical to the pre-chunking format.

    Chunk records are accumulated and embedded in batches of
    ``embedding_batch_size`` (one embedding request per batch instead of one
    per chunk), then inserted together.

    Embedding API calls are retried on rate limits and transient errors:
    ``embedding_max_retries`` is passed to the OpenAI client, which backs off
    exponentially and honours any ``Retry-After`` response header.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Don't log (INFO) all http requests.

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        dimensions=embedding_dimension,
        check_embedding_ctx_length=False,
        base_url=embedding_api_base or None,
        api_key=SecretStr(embedding_api_key) if embedding_api_key else None,
        max_retries=embedding_max_retries,
    )

    num_pages = 0
    num_sections = 0
    # Records and their texts accumulate here until a full batch is ready; each
    # record gets its dense_vector filled in at flush time (see _flush_embedding_batch).
    pending_records: list[dict] = []
    pending_texts: list[str] = []

    for page in tqdm(pages, desc="Processing pages", unit="pages"):
        if page.get("change_type") == "deleted":
            continue
        for section in page["sections"]:
            # Skip sections whose body is empty after tidying — typically a
            # heading whose content lives entirely in its subsections, or an
            # image/template-only (or absent) page lead. Indexing them would
            # embed the title alone, yielding a low-information vector that
            # surfaces as a contentless hit for unrelated queries.
            if not (section["text"] or "").strip():
                logger.debug(f"Skipping empty section: {section['doc_title']} / {section['title']}")
                continue

            # Calculate the preamble text (doc + section title).
            text_preamble = section["doc_title"]
            if section["title"] != section["doc_title"]:
                text_preamble = text_preamble + f" / {section['title']}"
            text_preamble = text_preamble.strip()

            # Split the section body into chunks. All limits are UTF-8 bytes
            # (the vector-store varchar limit is byte-based); min() is a
            # belt-and-braces guard so no strategy can ever exceed the
            # 5000-byte storage limit of the "text" field.
            text_content = section["text"] if section["text"] else ""
            chunks = split_section(
                text_content,
                strategy=chunk_strategy,
                max_bytes=min(chunk_max_bytes, 5000),
                overlap_bytes=chunk_overlap_bytes,
            )
            if chunk_strategy == "none" and len(text_content.encode("utf-8")) > 5000:
                logger.warning(f'Text too long for section "{text_preamble}", trimmed to 5000 bytes.')
            elif len(chunks) > 1:
                logger.info(f'Section "{text_preamble}" split into {len(chunks)} chunks.')

            section_id = str(section["id"])
            for chunk_index, chunk_text in enumerate(chunks):
                # Chunk 0 keeps the section's own id so single-chunk sections
                # remain identical to the pre-chunking format and wiki-link
                # relations keep resolving. Further chunks derive deterministic
                # ids from the section id, mirroring how section ids derive
                # from doc ids at load time.
                if chunk_index == 0:
                    chunk_id = section_id
                else:
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{section_id}-{chunk_index}".encode()))

                complete_text = text_preamble + "\n\n" + chunk_text
                logger.debug(f"Queuing {text_preamble} (chunk {chunk_index}) for embedding, text len {len(chunk_text)}")

                # Build the record now, but leave dense_vector to be filled in at flush time.
                pending_records.append({
                    "id": chunk_id,
                    "section_id": section_id,
                    "chunk_index": chunk_index,
                    "title": section["title"],
                    "text": chunk_text,
                    "source": section["source"],
                    "parent": str(section["parent"]) if section["parent"] else None,
                    "children": [str(child) for child in section["children"]],
                    "previous": [str(prv) for prv in section["previous"]],
                    "next": [str(nxt) for nxt in section["next"]],
                    "relations": [str(rel) for rel in section["relations"]],
                    "categories": [str(cat) for cat in page.get("categories", [])],
                    "page_id": int(section["page_id"]),
                    "doc_id": str(section["doc_id"]),
                    "doc_title": section["doc_title"],
                    "doc_hash": str(section["doc_hash"]),
                })
                pending_texts.append(complete_text)

                if len(pending_records) >= embedding_batch_size:
                    num_sections += _flush_embedding_batch(
                        embeddings, collection_name, pending_records, pending_texts)
                    pending_records = []
                    pending_texts = []
        num_pages += 1

    # Flush any trailing partial batch.
    num_sections += _flush_embedding_batch(embeddings, collection_name, pending_records, pending_texts)

    return [num_pages, num_sections]


def index_pages_incremental(
        pages: list[dict],
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_api_base: str = "",
        embedding_api_key: str = "",
        embedding_max_retries: int = 3,
        embedding_batch_size: int = 8,
        chunk_strategy: str = "none",
        chunk_max_bytes: int = 5000,
        chunk_overlap_bytes: int = 0,
) -> dict[str, int]:
    """Incrementally update the live collection based on each page's change_type.

    Pages with `change_type` of ``"deleted"`` or ``"updated"`` have their
    existing sections removed from the collection first. Pages with
    ``"created"`` or ``"updated"`` are then re-embedded and inserted.
    Pages with no `change_type` (``None``) are unchanged and skipped.

    Args:
        pages: List of page dicts from the incremental dump.
        collection_name: Name of the live collection to update in-place.
        embedding_model: Embedding model to use for new/updated pages.
        embedding_dimension: Embedding vector dimensions.
        embedding_api_base: Base URL for the OpenAI-compatible embedding endpoint.
        embedding_api_key: API key for the embedding endpoint.
        embedding_max_retries: Max retries for embedding API calls (rate limits
            and transient errors).
        embedding_batch_size: Number of sections embedded per API call.
        chunk_strategy: Section chunking strategy (see
            :func:`wiki_rag.index.chunking.split_section`).
        chunk_max_bytes: Maximum chunk size in UTF-8 bytes.
        chunk_overlap_bytes: Overlap between consecutive chunks in UTF-8 bytes.

    Returns:
        Summary dict with keys ``"deleted"``, ``"updated"``, ``"created"``,
        ``"skipped"``, and ``"sections_indexed"``.

    """
    delete_ids: list[int] = []
    pages_to_insert: list[dict] = []
    counts = {"deleted": 0, "updated": 0, "created": 0, "skipped": 0}

    for page in pages:
        change_type = page.get("change_type")
        if change_type == "deleted":
            delete_ids.append(page["id"])
            counts["deleted"] += 1
        elif change_type == "updated":
            delete_ids.append(page["id"])
            pages_to_insert.append(page)
            counts["updated"] += 1
        elif change_type == "created":
            pages_to_insert.append(page)
            counts["created"] += 1
        else:
            counts["skipped"] += 1

    vector.store.delete_by_page_ids(collection_name, delete_ids)

    [_, sections_indexed] = index_pages(
        pages_to_insert, collection_name, embedding_model, embedding_dimension,
        embedding_api_base, embedding_api_key, embedding_max_retries, embedding_batch_size,
        chunk_strategy=chunk_strategy,
        chunk_max_bytes=chunk_max_bytes,
        chunk_overlap_bytes=chunk_overlap_bytes,
    )
    counts["sections_indexed"] = sections_indexed

    return counts


def replace_previous_collection(collection_name: str, temp_collection_name: str) -> None:
    """Replace the previous collection with the new one.

    Compacts the temporary collection, drops any existing live collection,
    renames the temporary collection to the live name, and then loads the
    renamed collection into memory so it is immediately ready for querying.

    Args:
        collection_name: Name of the live collection to replace.
        temp_collection_name: Name of the temporary collection that holds the
            freshly indexed data.

    """
    if not vector.store.collection_exists(temp_collection_name):
        msg = f"Collection {temp_collection_name} does not exist."
        raise ValueError(msg)

    logger.info(f"Flushing collection {temp_collection_name!r} to seal segments before compaction")
    vector.store.flush_collection(temp_collection_name)
    logger.info(f"Compacting collection {temp_collection_name!r}")
    vector.store.compact_collection(temp_collection_name)

    if vector.store.collection_exists(collection_name):
        vector.store.drop_collection(collection_name)
    vector.store.rename_collection(temp_collection_name, collection_name)

    logger.info(f"Loading collection {collection_name!r} into memory for querying")
    vector.store.load_collection(collection_name)
