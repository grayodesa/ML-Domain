#!/usr/bin/env python3
"""DNS validation script for gambling domain predictions."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dns_validation import DNSValidator, DNSValidationPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Validate gambling domain predictions using DNS queries'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/results/predictions.csv',
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Minimum confidence threshold for validation (default: 0.8)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='DNS query timeout in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=10.0,
        help='Maximum DNS queries per second (default: 10.0)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=100,
        help='Maximum concurrent DNS queries (default: 100)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retry attempts for failed queries (default: 2)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Do not filter for gambling predictions (use when input already contains only gambling domains)'
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create validator
    validator = DNSValidator(
        timeout=args.timeout,
        max_retries=args.max_retries,
        rate_limit=args.rate_limit,
        concurrency=args.concurrency
    )

    # Create and run pipeline
    pipeline = DNSValidationPipeline(validator)

    try:
        stats = pipeline.run(
            predictions_file=input_file,
            output_dir=output_dir,
            confidence_threshold=args.confidence_threshold,
            filter_gambling=not args.no_filter
        )

        if stats:
            print("\n✓ DNS validation completed successfully!")
            print(f"\nOutput files in {output_dir}:")
            print("  - dns_validation.csv (full results)")
            print("  - active_gambling_domains.txt (active domains)")
            print("  - parked_gambling_domains.txt (parked domains)")
            print("  - dns_validation_stats.json (statistics)")

    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
