#!/usr/bin/env python3
"""Script to run inference on unlabeled domains."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description='Run inference on unlabeled domains')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file with domains (one per line)')
    parser.add_argument('--model-path', type=str, default='models',
                       help='Path to trained model directory')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['auto', 'logistic', 'random_forest'],
                       help='Model type to use (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default='data/results',
                       help='Output directory for predictions')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold for filtering')
    args = parser.parse_args()

    run_inference(
        input_file=Path(args.input),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence,
        model_type=args.model_type
    )


if __name__ == '__main__':
    main()
