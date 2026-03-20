#!/usr/bin/env bash
#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FULL_DUMPS="${MAX_FULL_DUMPS:-3}"
MAX_INCREMENTAL_DUMPS="${MAX_INCREMENTAL_DUMPS:-5}"
MIN_RETENTION_DAYS="${MIN_RETENTION_DAYS:-7}"
LOADER_DUMP_PATH="${LOADER_DUMP_PATH:-./data}"

DRY_RUN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            cat <<EOF
Usage: $(basename "$0") [--dry-run] [--help]

Delete old dump files from LOADER_DUMP_PATH based on configurable retention
rules. Intended to run periodically alongside wr-load --incremental.

Options:
  --dry-run   List candidates for deletion without removing any files.
  --help, -h  Show this help and exit.

Required environment variables:
  COLLECTION_NAME         Filename prefix filter (e.g. "mywiki")

Optional environment variables (with defaults):
  LOADER_DUMP_PATH        Dump directory              (default: ./data)
  MAX_FULL_DUMPS          Max full dumps to retain    (default: 3)
  MAX_INCREMENTAL_DUMPS   Max incremental dumps       (default: 5)
  MIN_RETENTION_DAYS      Min age (days) before a file is eligible for
                          deletion (grace period)     (default: 7)

Algorithm:
  1. Lists all {COLLECTION_NAME}-*.json files in LOADER_DUMP_PATH, sorted
     chronologically.
  2. Always protects the most recent file — wr-index and wr-load
     --incremental both depend on it.
  3. For every other file (newest-to-oldest): skips files younger than
     MIN_RETENTION_DAYS (grace period), reads dump_type from
     sites[0].dump_type in the JSON, and keeps up to MAX_FULL_DUMPS full
     dumps and MAX_INCREMENTAL_DUMPS incremental dumps. Deletes the rest.
  4. Prints a summary: scanned / protected / kept / deleted.

Exit codes:
  0  Success
  1  Missing COLLECTION_NAME
  2  Dump directory does not exist
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $arg" >&2
            echo "Usage: $(basename "$0") [--dry-run] [--help]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${COLLECTION_NAME:-}" ]]; then
    echo "Error: COLLECTION_NAME environment variable is required." >&2
    exit 1
fi

if [[ ! -d "$LOADER_DUMP_PATH" ]]; then
    echo "Error: Dump directory '$LOADER_DUMP_PATH' does not exist." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Collect dump files, sorted chronologically (alphabetical = chronological
# given the YYYY-MM-DD-HH-MM date-based naming convention)
# ---------------------------------------------------------------------------
files=()
while IFS= read -r file; do
    files+=("$file")
done < <(find "$LOADER_DUMP_PATH" -maxdepth 1 -name "${COLLECTION_NAME}-*.json" -type f | sort)

total_files="${#files[@]}"

if [[ "$total_files" -eq 0 ]]; then
    echo "No dump files found matching '${COLLECTION_NAME}-*.json' in '$LOADER_DUMP_PATH'. Nothing to do."
    exit 0
fi

echo "Found $total_files dump file(s) in '$LOADER_DUMP_PATH' matching '${COLLECTION_NAME}-*.json'."
if [[ "$DRY_RUN" == "true" ]]; then
    echo "(Dry run — no files will be deleted.)"
fi
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Return the age of a file in whole days (floor division).
get_file_age_days() {
    python3 -c "
import os, sys, time
print(int((time.time() - os.path.getmtime(sys.argv[1])) / 86400))
" "$1"
}

# Return the dump_type from sites[0] of a dump JSON, defaulting to "full"
# for old-format dumps that pre-date the dump_type field.
get_dump_type() {
    python3 -c "
import json, sys
try:
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
    sites = data.get('sites', [])
    print(sites[0].get('dump_type', 'full') if sites else 'full')
except Exception:
    print('full')
" "$1" 2>/dev/null || echo "full"
}

# ---------------------------------------------------------------------------
# Processing — iterate from newest to oldest, skipping the most recent file
# ---------------------------------------------------------------------------

# The most recent file is always protected: wr-index and wr-load --incremental
# both depend on finding the latest dump.
most_recent="${files[$((total_files - 1))]}"

count_scanned=0
count_protected=0
count_kept=0
count_deleted=0

kept_full=0
kept_incremental=0

delete_list=()

echo "Processing files (newest to oldest, excluding most recent):"

idx=$((total_files - 2))
while [[ $idx -ge 0 ]]; do
    file="${files[$idx]}"
    idx=$((idx - 1))
    count_scanned=$((count_scanned + 1))

    age=$(get_file_age_days "$file")

    # Grace period: file is too young to be eligible for deletion.
    if [[ "$age" -lt "$MIN_RETENTION_DAYS" ]]; then
        echo "  [PROTECTED] (grace period, ${age}d old): $(basename "$file")"
        count_protected=$((count_protected + 1))
        continue
    fi

    dump_type=$(get_dump_type "$file")

    if [[ "$dump_type" == "full" ]]; then
        if [[ "$kept_full" -lt "$MAX_FULL_DUMPS" ]]; then
            kept_full=$((kept_full + 1))
            echo "  [KEPT]      (full #${kept_full}/${MAX_FULL_DUMPS}, ${age}d old): $(basename "$file")"
            count_kept=$((count_kept + 1))
        else
            echo "  [DELETE]    (full quota exceeded, ${age}d old): $(basename "$file")"
            delete_list+=("$file")
            count_deleted=$((count_deleted + 1))
        fi
    else
        if [[ "$kept_incremental" -lt "$MAX_INCREMENTAL_DUMPS" ]]; then
            kept_incremental=$((kept_incremental + 1))
            echo "  [KEPT]      (incremental #${kept_incremental}/${MAX_INCREMENTAL_DUMPS}, ${age}d old): $(basename "$file")"
            count_kept=$((count_kept + 1))
        else
            echo "  [DELETE]    (incremental quota exceeded, ${age}d old): $(basename "$file")"
            delete_list+=("$file")
            count_deleted=$((count_deleted + 1))
        fi
    fi
done

echo "  [PROTECTED] (most recent, always kept): $(basename "$most_recent")"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "Summary:"
echo "  Scanned:   $count_scanned file(s) (excluding most recent)"
echo "  Protected: $count_protected file(s) (within ${MIN_RETENTION_DAYS}-day grace period)"
echo "  Kept:      $count_kept file(s) (within retention quota)"
echo "  Deleted:   $count_deleted file(s)"

# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
if [[ "${#delete_list[@]}" -gt 0 ]]; then
    echo ""
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Dry run — skipping deletion of $count_deleted file(s)."
    else
        echo "Deleting $count_deleted file(s)..."
        for file in "${delete_list[@]}"; do
            rm -f "$file"
        done
        echo "Done."
    fi
fi
