"""Inference module for classifying unlabeled domains."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import time
from .feature_engineering import DomainFeatureExtractor
from .model_training import GamblingDomainClassifier
from .utils import normalize_domain


class DomainClassifierInference:
    """Inference engine for gambling domain classification."""

    def __init__(self, model_path: Path):
        """Initialize inference engine."""
        self.model_path = model_path
        self.classifier = GamblingDomainClassifier()
        self.feature_extractor = DomainFeatureExtractor()

        # Load model and feature extractor
        self.classifier.load(model_path)
        self.feature_extractor.load(model_path)

    def predict_batch(self, domains: List[str]) -> pd.DataFrame:
        """Predict classifications for a batch of domains."""
        print(f"Running inference on {len(domains)} domains...")

        start_time = time.time()

        # Normalize domains to registered domain (handles subdomains, co.uk, etc.)
        original_domains = domains
        normalized_domains = [normalize_domain(d) for d in domains]

        # Extract features from normalized domains
        X = self.feature_extractor.create_feature_matrix(normalized_domains)

        # Predict
        predictions = self.classifier.model.predict(X)
        probabilities = self.classifier.model.predict_proba(X)

        # Create results DataFrame with both original and normalized domains
        results = pd.DataFrame({
            'original_domain': original_domains,
            'domain': normalized_domains,
            'prediction': predictions,
            'probability_benign': probabilities[:, 0],
            'probability_gambling': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1)
        })

        # Add classification label
        results['label'] = results['prediction'].map({0: 'benign', 1: 'gambling'})

        elapsed_time = time.time() - start_time
        throughput = len(domains) / elapsed_time

        print(f"\nInference completed in {elapsed_time:.2f} seconds")
        print(f"Throughput: {throughput:,.0f} domains/second")

        # Statistics
        gambling_count = (predictions == 1).sum()
        benign_count = (predictions == 0).sum()

        print(f"\nPredictions:")
        print(f"  Gambling: {gambling_count} ({gambling_count/len(domains)*100:.1f}%)")
        print(f"  Benign: {benign_count} ({benign_count/len(domains)*100:.1f}%)")

        return results

    def filter_by_confidence(self, results: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Filter predictions by confidence threshold."""
        filtered = results[results['confidence'] >= threshold].copy()

        print(f"\nFiltering by confidence >= {threshold}:")
        print(f"  Kept: {len(filtered)} domains ({len(filtered)/len(results)*100:.1f}%)")
        print(f"  Removed: {len(results) - len(filtered)} domains")

        return filtered

    def analyze_predictions(self, results: pd.DataFrame, output_dir: Path):
        """Analyze and visualize predictions."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confidence distribution
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Confidence histogram
        axes[0, 0].hist(results['confidence'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Confidence Distribution')
        axes[0, 0].axvline(x=0.8, color='red', linestyle='--', label='Threshold=0.8')
        axes[0, 0].legend()

        # Probability distributions by class
        gambling_domains = results[results['prediction'] == 1]
        benign_domains = results[results['prediction'] == 0]

        axes[0, 1].hist(gambling_domains['probability_gambling'], bins=30,
                       alpha=0.7, label='Predicted Gambling', color='red', edgecolor='black')
        axes[0, 1].hist(benign_domains['probability_benign'], bins=30,
                       alpha=0.7, label='Predicted Benign', color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Probability Distribution by Predicted Class')
        axes[0, 1].legend()

        # Prediction counts
        prediction_counts = results['label'].value_counts()
        axes[1, 0].bar(prediction_counts.index, prediction_counts.values,
                      color=['green', 'red'], edgecolor='black')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Prediction Counts')

        # Confidence by prediction
        results.boxplot(column='confidence', by='label', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Prediction')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Confidence by Prediction')
        plt.suptitle('')

        plt.tight_layout()
        output_path = output_dir / 'prediction_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Prediction analysis saved to {output_path}")


def load_domains_from_file(file_path: Path) -> List[str]:
    """Load domains from a text file (one per line)."""
    with open(file_path, 'r') as f:
        domains = [line.strip() for line in f if line.strip()]
    return domains


def run_inference(input_file: Path, model_path: Path, output_dir: Path,
                 confidence_threshold: float = 0.8):
    """Run inference on domains from input file."""
    print("=" * 60)
    print("GAMBLING DOMAIN CLASSIFIER - INFERENCE")
    print("=" * 60)

    # Load domains
    domains = load_domains_from_file(input_file)
    print(f"Loaded {len(domains)} domains from {input_file}")

    # Initialize inference engine
    inference = DomainClassifierInference(model_path)

    # Predict
    results = inference.predict_batch(domains)

    # Save all predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_path = output_dir / 'predictions.csv'
    results.to_csv(all_predictions_path, index=False)
    print(f"\nAll predictions saved to {all_predictions_path}")

    # Filter by confidence
    filtered_results = inference.filter_by_confidence(results, confidence_threshold)
    filtered_path = output_dir / f'predictions_confident_{int(confidence_threshold*100)}.csv'
    filtered_results.to_csv(filtered_path, index=False)
    print(f"Confident predictions saved to {filtered_path}")

    # Analyze
    inference.analyze_predictions(results, output_dir)

    # Save gambling domains separately for manual review
    gambling_domains = results[results['prediction'] == 1].sort_values('probability_gambling', ascending=False)
    gambling_path = output_dir / 'predicted_gambling.csv'
    gambling_domains.to_csv(gambling_path, index=False)
    print(f"\nPredicted gambling domains saved to {gambling_path}")

    return results


if __name__ == '__main__':
    # Example usage
    pass
