"""Model training module for gambling domain classification."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple


class GamblingDomainClassifier:
    """Logistic Regression classifier for gambling domains."""

    def __init__(self):
        self.model = None
        self.best_params = None

    def train_baseline(self, X_train, y_train) -> LogisticRegression:
        """Train baseline Logistic Regression model."""
        print("Training baseline model...")

        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"Baseline CV F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return model

    def tune_hyperparameters(self, X_train, y_train) -> LogisticRegression:
        """Tune hyperparameters using GridSearchCV."""
        print("Tuning hyperparameters...")

        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'class_weight': ['balanced'],
            'max_iter': [2000],  # Increased for better convergence
        }

        # Add l1 penalty for solvers that support it
        param_grid_l1 = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced'],
            'max_iter': [2000],  # Increased for better convergence
        }

        base_model = LogisticRegression(random_state=42)

        # Grid search with F1 scoring
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_

    def train(self, X_train, y_train, tune_hyperparams: bool = True):
        """Train the model."""
        if tune_hyperparams:
            self.model = self.tune_hyperparameters(X_train, y_train)
        else:
            self.model = self.train_baseline(X_train, y_train)

        return self.model

    def evaluate(self, X_test, y_test, output_dir: Path) -> Dict:
        """Evaluate model and generate metrics."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Gambling']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Calculate specific metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'precision_gambling': float(precision),
            'recall_gambling': float(recall),
            'f1_gambling': float(f1),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        }

        # Save metrics
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics saved to {metrics_path}")

        # Generate visualizations
        self.plot_confusion_matrix(cm, output_dir)
        self.plot_roc_curve(y_test, y_pred_proba, output_dir)
        self.plot_precision_recall_curve(y_test, y_pred_proba, output_dir)

        return metrics

    def plot_confusion_matrix(self, cm, output_dir: Path):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Gambling'],
                    yticklabels=['Benign', 'Gambling'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        output_path = output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {output_path}")

    def plot_roc_curve(self, y_test, y_pred_proba, output_dir: Path):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {output_path}")

    def plot_precision_recall_curve(self, y_test, y_pred_proba, output_dir: Path):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / 'precision_recall_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Precision-recall curve saved to {output_path}")

    def plot_feature_importance(self, feature_names: list, output_dir: Path, top_n: int = 20):
        """Plot top feature importances."""
        if self.model is None:
            print("Model not trained yet")
            return

        # Get coefficients
        coefficients = self.model.coef_[0]

        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # Sort by absolute value
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

        # Top features
        top_features = feature_importance.head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Most Important Features')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        output_path = output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {output_path}")

        # Save top features to CSV
        csv_path = output_dir / 'top_features.csv'
        top_features.to_csv(csv_path, index=False)
        print(f"Top features saved to {csv_path}")

    def save(self, path: Path):
        """Save the trained model."""
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / 'logistic_regression.joblib'
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load(self, path: Path):
        """Load a trained model."""
        model_path = path / 'logistic_regression.joblib'
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")


if __name__ == '__main__':
    # This will be called from the training script
    pass
