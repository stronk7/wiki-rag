#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Some utilities to be used by the whole package."""

import contextlib
import fcntl
import logging
import time

from collections.abc import Generator
from pathlib import Path

import colorlog


@contextlib.contextmanager
def instance_lock(collection_name: str, dump_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive per-instance lock to prevent concurrent wr-load / wr-index runs.

    Uses an advisory file lock (fcntl.flock) so the lock is released automatically
    if the process is killed. The lock file is stored as
    ``{dump_path}/{collection_name}.lock``.

    Args:
        collection_name: The collection name, used as the lock file prefix.
        dump_path: Directory where the lock file is created.

    Raises:
        SystemExit: If the lock cannot be acquired (another process holds it),
            or if the lock file cannot be opened.

    """
    lock_file = dump_path / f"{collection_name}.lock"
    logger = logging.getLogger(__name__)
    try:
        fd = lock_file.open("w")
    except OSError as exc:
        logger.error("Failed to open lock file '%s': %s", lock_file, exc)
        raise SystemExit(1) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        logger.error(
            "Another wr-load or wr-index is already running for collection '%s'. "
            "Lock file: %s",
            collection_name,
            lock_file,
        )
        raise SystemExit(1)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up the logging configuration."""
    logging.Formatter.converter = time.gmtime

    log = logging.getLogger()

    # Set the log level explicitly
    log.setLevel(level)

    # Let's add the color handler.
    if not log.hasHandlers():
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s.%(msecs)03dZ %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            reset=True,
            style="%",
        ))
        log.addHandler(handler)

    return log
