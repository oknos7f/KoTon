#!/usr/bin/env python3
"""
Launcher for KoTon 3-Window Live Security Detection Demo.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from koton.config import FLASK_HOST, FLASK_PORT
from koton.web.server import start_server


def main():
    start_server(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    main()
