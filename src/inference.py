"""Inference module for classifying unlabeled domains."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import time
import warnings
from .feature_engineering import DomainFeatureExtractor
from .model_training import GamblingDomainClassifier, RandomForestDomainClassifier
from .utils import normalize_domain
from tqdm import tqdm

# Suppress sklearn feature name warnings when using sparse matrices
warnings.filterwarnings('ignore', message='X does not have valid feature names')


class DomainClassifierInference:
    """Inference engine for gambling domain classification."""

    def __init__(self, model_path: Path, model_type: str = 'auto'):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model directory
            model_type: Type of model ('logistic', 'random_forest', or 'auto' to detect)
        """
        self.model_path = model_path
        self.feature_extractor = DomainFeatureExtractor()

        # Auto-detect model type if not specified
        if model_type == 'auto':
            logistic_exists = (model_path / 'logistic_regression.joblib').exists()
            rf_exists = (model_path / 'random_forest.joblib').exists()

            if rf_exists and not logistic_exists:
                model_type = 'random_forest'
            elif logistic_exists and not rf_exists:
                model_type = 'logistic'
            elif rf_exists and logistic_exists:
                # Both exist, prefer RF (newer/better)
                model_type = 'random_forest'
                print("Note: Both models found, using Random Forest")
            else:
                raise FileNotFoundError(f"No model files found in {model_path}")

        self.model_type = model_type

        # Instantiate appropriate classifier
        if model_type == 'logistic':
            self.classifier = GamblingDomainClassifier()
            print("Loading Logistic Regression model...")
        else:  # random_forest
            self.classifier = RandomForestDomainClassifier()
            print("Loading Random Forest model...")

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

    def predict_batch_chunked(self, domains: List[str], batch_size: int = 50000,
                               output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Predict classifications for a batch of domains using chunked processing.

        Memory-efficient for large datasets (5M+ domains) by processing in chunks
        and optionally writing results incrementally to disk.

        Args:
            domains: List of domains to classify
            batch_size: Number of domains to process per chunk (default 50000)
            output_path: If provided, results are written incrementally to this CSV file

        Returns:
            DataFrame with predictions (may be partial if output_path is used)
        """
        print(f"Running chunked inference on {len(domains):,} domains...")
        print(f"Batch size: {batch_size:,} domains per chunk")

        start_time = time.time()
        total_domains = len(domains)
        num_chunks = (total_domains + batch_size - 1) // batch_size

        # Statistics accumulators
        total_gambling = 0
        total_benign = 0

        # Process in chunks
        all_results = []
        first_chunk = True

        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks", unit="chunk"):
            chunk_start = chunk_idx * batch_size
            chunk_end = min(chunk_start + batch_size, total_domains)
            chunk_domains = domains[chunk_start:chunk_end]

            # Normalize domains
            original_chunk = chunk_domains
            normalized_chunk = [normalize_domain(d) for d in chunk_domains]

            # Extract features - this now uses sparse matrices internally
            X_chunk = self.feature_extractor.create_feature_matrix_sparse(normalized_chunk)

            # Predict
            predictions = self.classifier.model.predict(X_chunk)
            probabilities = self.classifier.model.predict_proba(X_chunk)

            # Create chunk results
            chunk_results = pd.DataFrame({
                'original_domain': original_chunk,
                'domain': normalized_chunk,
                'prediction': predictions,
                'probability_benign': probabilities[:, 0],
                'probability_gambling': probabilities[:, 1],
                'confidence': np.max(probabilities, axis=1)
            })

            # Add classification label
            chunk_results['label'] = chunk_results['prediction'].map({0: 'benign', 1: 'gambling'})

            # Update statistics
            total_gambling += (predictions == 1).sum()
            total_benign += (predictions == 0).sum()

            # Write incrementally to disk if output path provided
            if output_path:
                chunk_results.to_csv(
                    output_path,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                # Don't accumulate results in memory
            else:
                all_results.append(chunk_results)

        elapsed_time = time.time() - start_time
        throughput = total_domains / elapsed_time

        print(f"\nInference completed in {elapsed_time:.2f} seconds")
        print(f"Throughput: {throughput:,.0f} domains/second")
        print(f"\nPredictions:")
        print(f"  Gambling: {total_gambling:,} ({total_gambling/total_domains*100:.1f}%)")
        print(f"  Benign: {total_benign:,} ({total_benign/total_domains*100:.1f}%)")

        # Return combined results or empty DataFrame if written to disk
        if output_path:
            print(f"\nResults written incrementally to {output_path}")
            return pd.DataFrame()  # Empty since results are on disk
        else:
            return pd.concat(all_results, ignore_index=True)

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
                 confidence_threshold: float = 0.8, model_type: str = 'auto',
                 batch_size: int = None, use_chunked: bool = None):
    """
    Run inference on domains from input file.

    Args:
        input_file: Path to file with domains (one per line)
        model_path: Path to trained model directory
        output_dir: Output directory for predictions
        confidence_threshold: Confidence threshold for filtering
        model_type: Model type ('auto', 'logistic', 'random_forest')
        batch_size: Batch size for chunked processing (default: 50000)
        use_chunked: Force chunked processing. If None, auto-detect based on dataset size
    """
    print("=" * 60)
    print("GAMBLING DOMAIN CLASSIFIER - INFERENCE")
    print("=" * 60)

    # Load domains
    domains = load_domains_from_file(input_file)
    print(f"Loaded {len(domains):,} domains from {input_file}")

    # Initialize inference engine
    inference = DomainClassifierInference(model_path, model_type=model_type)

    # Auto-detect whether to use chunked processing
    if use_chunked is None:
        # Use chunked processing for datasets > 100k domains
        use_chunked = len(domains) > 100000

    # Set default batch size
    if batch_size is None:
        batch_size = 50000

    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_path = output_dir / 'predictions.csv'

    # Predict
    if use_chunked:
        print(f"\nUsing chunked processing (batch_size={batch_size:,})")
        print("Results will be written incrementally to disk to save memory.\n")

        # Write results incrementally during prediction
        results = inference.predict_batch_chunked(
            domains,
            batch_size=batch_size,
            output_path=all_predictions_path
        )

        print(f"\nAll predictions saved to {all_predictions_path}")

        # For large datasets, process filtering and analysis in chunks too
        print("\nProcessing results for filtering and analysis...")
        chunk_size = 100000
        filtered_chunks = []
        gambling_chunks = []

        for chunk in pd.read_csv(all_predictions_path, chunksize=chunk_size):
            # Filter by confidence
            filtered = chunk[chunk['confidence'] >= confidence_threshold]
            filtered_chunks.append(filtered)

            # Extract gambling domains
            gambling = chunk[chunk['prediction'] == 1]
            gambling_chunks.append(gambling)

        # Combine and save filtered results
        if filtered_chunks:
            filtered_results = pd.concat(filtered_chunks, ignore_index=True)
            filtered_path = output_dir / f'predictions_confident_{int(confidence_threshold*100)}.csv'
            filtered_results.to_csv(filtered_path, index=False)
            print(f"Confident predictions saved to {filtered_path}")
            print(f"  Kept: {len(filtered_results):,} domains ({len(filtered_results)/len(domains)*100:.1f}%)")

        # Combine and save gambling domains
        if gambling_chunks:
            gambling_domains = pd.concat(gambling_chunks, ignore_index=True)
            gambling_domains = gambling_domains.sort_values('probability_gambling', ascending=False)
            gambling_path = output_dir / 'predicted_gambling.csv'
            gambling_domains.to_csv(gambling_path, index=False)
            print(f"\nPredicted gambling domains saved to {gambling_path}")

        # Skip visualization for very large datasets (would consume too much memory)
        if len(domains) <= 500000:
            print("\nGenerating analysis plots (this may take a moment for large datasets)...")
            results_sample = pd.read_csv(all_predictions_path, nrows=100000)  # Sample for viz
            inference.analyze_predictions(results_sample, output_dir)
        else:
            print("\nSkipping visualization for datasets > 500k domains (too memory intensive)")

        return pd.DataFrame()  # Return empty since results are on disk

    else:
        print("\nUsing standard batch processing (all in memory)\n")

        results = inference.predict_batch(domains)

        # Save all predictions
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
