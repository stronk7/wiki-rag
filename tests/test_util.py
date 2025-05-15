#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.util tests."""

import logging
import unittest

import colorlog  # Import necessary to check handler type

from wiki_rag.util import setup_logging


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


if __name__ == "__main__":
    unittest.main()
