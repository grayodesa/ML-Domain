#!/usr/bin/env python3
"""DNS validation script for gambling domain predictions - OPTIMIZED for Unbound."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dns_validation import DNSValidator, DNSValidationPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Validate gambling domain predictions using DNS queries (optimized for local Unbound resolver)'
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
        default=10.0,  # CHANGED: 5.0 → 10.0
        help='DNS query timeout in seconds (default: 10.0)'
    )
    # REMOVED: --rate-limit parameter (not needed for local DNS)
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1500,  # CHANGED: 100 → 1500
        help='Maximum concurrent DNS queries (default: 1500, optimized for local Unbound)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retry attempts for failed queries (default: 2)'
    )
    parser.add_argument(
        '--resolver-pool-size',
        type=int,
        default=50,
        help='Number of DNS resolver instances in pool (default: 50)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Do not filter for gambling predictions (use when input already contains only gambling domains)'
    )
    # NEW: Performance presets
    parser.add_argument(
        '--preset',
        type=str,
        choices=['balanced', 'fast', 'reliable'],
        default='balanced',
        help='Performance preset: balanced (default), fast (max speed), reliable (max accuracy)'
    )
    # NEW: Chunked processing parameters
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Chunk size for batch processing (default: 10000, auto-detect based on dataset size)'
    )
    parser.add_argument(
        '--use-chunked',
        action='store_true',
        help='Force chunked processing even for small datasets'
    )
    parser.add_argument(
        '--no-chunked',
        action='store_true',
        help='Disable chunked processing even for large datasets'
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

    # Apply preset configurations
    if args.preset == 'fast':
        print("⚡ Using FAST preset (maximum speed, higher error tolerance)")
        timeout = 8.0
        max_retries = 1
        concurrency = 2500
        resolver_pool_size = 100
    elif args.preset == 'reliable':
        print("🛡️  Using RELIABLE preset (maximum accuracy, slower)")
        timeout = 15.0
        max_retries = 3
        concurrency = 1000
        resolver_pool_size = 30
    else:  # balanced
        print("⚖️  Using BALANCED preset (optimal speed/accuracy)")
        timeout = args.timeout
        max_retries = args.max_retries
        concurrency = args.concurrency
        resolver_pool_size = args.resolver_pool_size

    # Create validator
    validator = DNSValidator(
        timeout=timeout,
        max_retries=max_retries,
        concurrency=concurrency,
        resolver_pool_size=resolver_pool_size
    )

    # Create and run pipeline
    pipeline = DNSValidationPipeline(validator)

    # Determine chunked processing setting
    use_chunked = None
    if args.use_chunked:
        use_chunked = True
    elif args.no_chunked:
        use_chunked = False
    # Otherwise None = auto-detect based on size

    try:
        print(f"\n📊 Input: {input_file}")
        print(f"📁 Output: {output_dir}")
        print(f"🎯 Confidence threshold: {args.confidence_threshold}")

        stats = pipeline.run(
            predictions_file=input_file,
            output_dir=output_dir,
            confidence_threshold=args.confidence_threshold,
            filter_gambling=not args.no_filter,
            chunk_size=args.chunk_size,
            use_chunked=use_chunked
        )

        if stats:
            print("\n" + "="*60)
            print("✅ DNS VALIDATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\n📂 Output files in {output_dir}:")
            print("  • dns_validation.csv - Full validation results")
            print("  • active_gambling_domains.txt - Active domains list")
            print("  • parked_gambling_domains.txt - Parked domains list")
            print("  • dns_validation_stats.json - Detailed statistics")
            
            # Quick summary
            print(f"\n📈 Quick Summary:")
            print(f"  • Total validated: {stats['total_domains']}")
            print(f"  • Active domains: {stats['active_domains']} ({stats['active_rate_pct']:.1f}%)")
            print(f"  • Parked domains: {stats['parked_domains']} ({stats['parking_rate_pct']:.1f}%)")
            print(f"  • Success rate: {stats['success_rate_pct']:.1f}%")

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()