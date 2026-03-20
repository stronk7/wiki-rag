#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.util tests."""

import logging
import tempfile
import unittest

from pathlib import Path

import colorlog  # Import necessary to check handler type

from wiki_rag.util import instance_lock, setup_logging


class TestUtil(unittest.TestCase):

    def setUp(self):
        """Clear all handlers before each test."""
        self.logger = logging.getLogger("test_logger")
        self.logger.handlers.clear()

    def test_logger_initialization(self):
        logger = setup_logging("DEBUG")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)  # Expect DEBUG level

    def test_handler_setup(self):
        # Call setup_logging
        logger = setup_logging("INFO")
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], colorlog.StreamHandler)

    def test_no_duplicate_handlers(self):
        logger = setup_logging("INFO")
        original_handler_count = len(logger.handlers)
        setup_logging("INFO")  # Call again
        self.assertEqual(len(logger.handlers), original_handler_count)


class TestInstanceLock(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dump_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lock_acquired_and_released(self):
        """Lock is acquired inside the context and the lock file is created."""
        with instance_lock("mycollection", self.dump_path):
            lock_file = self.dump_path / "mycollection.lock"
            self.assertTrue(lock_file.exists())

    def test_lock_released_after_context(self):
        """A second acquisition succeeds after the first context exits."""
        with instance_lock("mycollection", self.dump_path):
            pass
        # Should not raise — lock is released.
        with instance_lock("mycollection", self.dump_path):
            pass

    def test_concurrent_lock_raises_system_exit(self):
        """A second concurrent acquisition for the same collection raises SystemExit."""
        with instance_lock("mycollection", self.dump_path):
            with self.assertRaises(SystemExit):
                with instance_lock("mycollection", self.dump_path):
                    pass  # pragma: no cover

    def test_different_collections_do_not_conflict(self):
        """Locks for different collection names are independent."""
        with instance_lock("collection_a", self.dump_path):
            # Should not raise — different lock file.
            with instance_lock("collection_b", self.dump_path):
                pass

    def test_missing_directory_raises_system_exit(self):
        """SystemExit is raised when the dump directory does not exist."""
        missing = self.dump_path / "nonexistent"
        with self.assertRaises(SystemExit):
            with instance_lock("mycollection", missing):
                pass  # pragma: no cover


if __name__ == "__main__":
    unittest.main()
