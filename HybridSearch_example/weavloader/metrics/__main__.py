#!/usr/bin/env python3
"""
Entry point for running the metrics package as a module.
"""

import logging
from .server import start_metrics_server

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_metrics_server()
