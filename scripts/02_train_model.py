#!/usr/bin/env python3
"""Script to train the gambling domain classifier."""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import prepare_features
from src.model_training import GamblingDomainClassifier, RandomForestDomainClassifier


def main():
    parser = argparse.ArgumentParser(description='Train gambling domain classifier')
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic', 'random_forest'],
        default='logistic',
        help='Model type to train (default: logistic)'
    )
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Enable hyperparameter tuning (slower but better results)')
    parser.add_argument('--train-data', type=str, default='data/processed/train.csv',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str, default='data/processed/test.csv',
                       help='Path to test data')
    args = parser.parse_args()

    model_name = "Logistic Regression" if args.model == 'logistic' else "Random Forest"
    print("=" * 60)
    print(f"GAMBLING DOMAIN CLASSIFIER - TRAINING ({model_name.upper()})")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"Train set: {len(train_df)} domains")
    print(f"Test set: {len(test_df)} domains")

    # Prepare features
    X_train, y_train, X_test, y_test, extractor = prepare_features(train_df, test_df)

    # Train model
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 60)

    # Instantiate appropriate classifier
    if args.model == 'logistic':
        classifier = GamblingDomainClassifier()
    else:  # random_forest
        classifier = RandomForestDomainClassifier()

    classifier.train(X_train, y_train, tune_hyperparams=args.tune_hyperparams)

    # Evaluate
    output_dir = Path('models')
    metrics = classifier.evaluate(X_test, y_test, output_dir)

    # Feature importance
    classifier.plot_feature_importance(extractor.feature_names, output_dir, top_n=20)

    # Save model
    classifier.save(output_dir)

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel and artifacts saved to {output_dir}")
    print(f"\nKey metrics:")
    print(f"  Model type: {args.model}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (gambling): {metrics['precision_gambling']:.4f}")
    print(f"  Recall (gambling): {metrics['recall_gambling']:.4f}")
    print(f"  F1 (gambling): {metrics['f1_gambling']:.4f}")

    if 'avg_inference_ms' in metrics:
        print(f"  Avg inference time: {metrics['avg_inference_ms']:.3f} ms/domain")


if __name__ == '__main__':
    main()
