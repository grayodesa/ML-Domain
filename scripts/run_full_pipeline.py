#!/usr/bin/env python3
"""Run the complete training pipeline."""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import collect_data
from src.feature_engineering import prepare_features
from src.model_training import GamblingDomainClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Run complete training pipeline for gambling domain classifier'
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

    print("=" * 60)
    print("GAMBLING DOMAIN CLASSIFIER - FULL PIPELINE")
    print("=" * 60)

    # Step 1: Data Collection
    print("\n### STEP 1: DATA COLLECTION ###")
    train_df, test_df = collect_data(gambling_file=gambling_file, benign_file=benign_file)

    # Step 2: Feature Engineering
    print("\n### STEP 2: FEATURE ENGINEERING ###")
    X_train, y_train, X_test, y_test, extractor = prepare_features(train_df, test_df)

    # Step 3: Model Training
    print("\n### STEP 3: MODEL TRAINING ###")
    classifier = GamblingDomainClassifier()
    classifier.train(X_train, y_train, tune_hyperparams=True)

    # Step 4: Evaluation
    print("\n### STEP 4: MODEL EVALUATION ###")
    output_dir = Path('models')
    metrics = classifier.evaluate(X_test, y_test, output_dir)

    # Feature importance
    classifier.plot_feature_importance(extractor.feature_names, output_dir, top_n=20)

    # Save model
    classifier.save(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nModel artifacts saved to {output_dir}")
    print(f"\nFinal metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (gambling): {metrics['precision_gambling']:.4f}")
    print(f"  Recall (gambling): {metrics['recall_gambling']:.4f}")
    print(f"  F1 (gambling): {metrics['f1_gambling']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # Check if metrics meet requirements
    print("\nRequirements check:")
    print(f"  Accuracy ≥ 0.92: {'✓' if metrics['accuracy'] >= 0.92 else '✗'}")
    print(f"  Precision ≥ 0.90: {'✓' if metrics['precision_gambling'] >= 0.90 else '✗'}")
    print(f"  Recall ≥ 0.85: {'✓' if metrics['recall_gambling'] >= 0.85 else '✗'}")


if __name__ == '__main__':
    main()
