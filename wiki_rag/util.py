#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Some utilities to be used by the whole package."""

import logging
import time

import colorlog


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
