#!/usr/bin/env python3
"""Script to collect and prepare training data."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import collect_data


if __name__ == '__main__':
    collect_data()
