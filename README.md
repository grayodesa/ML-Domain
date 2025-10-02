# Gambling Domain Classifier

ML classifier for detecting gambling-themed domains using logistic regression with DNS validation capabilities.

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run complete training pipeline
python scripts/run_full_pipeline.py
```

Or run steps individually:

```bash
# Step 1: Collect data
python scripts/01_collect_data.py

# Step 2: Train model (with hyperparameter tuning)
python scripts/02_train_model.py --tune-hyperparams

# Step 3: Run inference on unlabeled domains
python scripts/03_run_inference.py --input data/raw/test_domains.txt
```

## Project Structure

```
├── data/
│   ├── raw/              # Source data
│   ├── processed/        # Train/test splits
│   └── results/          # Inference results
├── models/               # Trained models and artifacts
├── src/                  # Source modules
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── inference.py
└── scripts/              # Executable scripts
```

## Features

- **Keyword-based**: Multilingual gambling terms (casino, bet, poker, etc.)
- **TLD features**: Gambling-specific TLDs (.bet, .casino, .poker)
- **Structural**: Domain length, numbers, hyphens, character patterns
- **Character n-grams**: TF-IDF on bigrams and trigrams

## Performance Targets

- Accuracy: ≥92%
- Precision (gambling): ≥90%
- Recall (gambling): ≥85%
- Inference speed: ≥50,000 domains/second

## Model Artifacts

After training, the following artifacts are saved to `models/`:

- `logistic_regression.joblib` - Trained model
- `tfidf_vectorizer.joblib` - Feature vectorizer
- `model_metrics.json` - Performance metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve
- `feature_importance.png` - Top feature coefficients
