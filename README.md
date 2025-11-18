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

# Step 1 (with additional local gambling domains):
python scripts/01_collect_data.py --gambling-file data/raw/verified_casinos.txt

# Step 2: Train model (with hyperparameter tuning)
python scripts/02_train_model.py --tune-hyperparams

# Step 3: Run inference on unlabeled domains
python scripts/03_run_inference.py --input data/raw/test_domains.txt
```

### Adding Custom Data Sources

You can supplement the default data sources with your own domain lists:

#### Custom Gambling Domains

```bash
# Create a text file with one domain per line
# Example: data/raw/verified_casinos.txt
# casino-online.com
# best-poker-site.net
# sports-betting.io

# Run data collection with your custom file
python scripts/01_collect_data.py --gambling-file data/raw/verified_casinos.txt
```

**Format requirements for gambling domain files:**
- One domain per line
- Plain text file (UTF-8 encoding)
- Comments starting with `#` are ignored
- Empty lines are ignored
- Domains will be automatically cleaned and validated

#### Custom Benign Domains

```bash
# CSV format (Tranco-style: rank,domain)
# Example: data/raw/benign_domains.csv
# 1,google.com
# 2,youtube.com
# 3,facebook.com

# Or simple text format (one domain per line)
# Example: data/raw/benign_domains.txt
# google.com
# youtube.com
# facebook.com

# Run data collection with your custom benign domains
python scripts/01_collect_data.py --benign-file data/raw/benign_domains.csv
```

**Format requirements for benign domain files:**
- Supports both CSV and TXT formats
- CSV format: `rank,domain` or just `domain`
- TXT format: one domain per line
- Comments starting with `#` are ignored
- Gambling keywords are automatically filtered out
- Domains will be automatically cleaned and validated

#### Using Both Custom Sources

```bash
# Run with both custom gambling and benign domains
python scripts/01_collect_data.py \
  --gambling-file data/raw/verified_casinos.txt \
  --benign-file data/raw/benign_domains.csv

# Or run the full pipeline with custom domains
python scripts/run_full_pipeline.py \
  --gambling-file data/raw/verified_casinos.txt \
  --benign-file data/raw/benign_domains.csv
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
