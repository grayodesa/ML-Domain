#!/usr/bin/env python3
"""Script to collect and prepare training data."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import collect_data


def main():
    parser = argparse.ArgumentParser(
        description='Collect gambling and benign domains for training'
    )
    parser.add_argument(
        '--gambling-file',
        type=str,
        help='Path to local txt file with additional gambling domains (one per line)'
    )
    parser.add_argument(
        '--benign-file',
        type=str,
        help='Path to local CSV/txt file with benign domains (supports Tranco format: rank,domain)'
    )

    args = parser.parse_args()

    # Convert to Path if provided
    gambling_file = Path(args.gambling_file) if args.gambling_file else None
    benign_file = Path(args.benign_file) if args.benign_file else None

    # Validate files exist if provided
    if gambling_file and not gambling_file.exists():
        print(f"Error: File not found: {gambling_file}")
        sys.exit(1)

    if benign_file and not benign_file.exists():
        print(f"Error: File not found: {benign_file}")
        sys.exit(1)

    # Run data collection
    collect_data(gambling_file=gambling_file, benign_file=benign_file)


if __name__ == '__main__':
    main()
