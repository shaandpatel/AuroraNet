"""
Centralized logging configuration for the application.
"""
import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configures the root logger for consistent, clean logging across the project.

    - Sets a standard format: [TIMESTAMP] [LEVEL] - MODULE - MESSAGE
    - Clears any existing handlers to prevent duplicate logs.
    - Adds a new handler that streams to standard output.
    """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(name)s - %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers and add a new one
    root_logger.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
