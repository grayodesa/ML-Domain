# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML classifier for gambling-themed domains with DNS validation. The system processes domains from Certificate Transparency logs, classifying them as gambling/non-gambling using machine learning (Logistic Regression or Random Forest) and validating results through DNS queries.

**Primary Language:** Python 3.9+
**Project Type:** Machine learning classification pipeline
**Supported Models:** Logistic Regression, Random Forest

## Project Architecture

### Pipeline Flow
```
Data Collection → Feature Engineering → Model Training →
Model Evaluation → Inference → DNS Validation → Results Analysis
```

### Core Components

1. **Data Collection Module** (`src/data_collection.py`)
   - Fetches gambling domains from BlockList Project and Hagezi
   - Supports loading additional gambling domains from local txt files
   - Collects benign domains from Cloudflare Radar and Tranco
   - Supports loading benign domains from local CSV/txt files (Tranco format supported)
   - Handles deduplication and domain validation
   - Creates 80/20 train/test split

2. **Feature Engineering Module** (`src/feature_engineering.py`)
   - Keyword-based features (gambling terms: casino, bet, poker, slots, etc.)
   - Multilingual support (kasino, apuesta, казино, wetten, etc.)
   - TLD features (.bet, .casino, .poker)
   - Structural features (domain length, numbers, hyphens)
   - Character n-grams with TF-IDF vectorization

3. **Model Training Module** (`src/model_training.py`)
   - **Logistic Regression** (fast, interpretable, ~5 MB model)
     - L2 regularization, liblinear solver
     - Hyperparameter tuning via GridSearchCV
     - Training time: 2-5 minutes (optimized)
   - **Random Forest** (higher accuracy, ~50-100 MB model)
     - 200 estimators, max_depth=30
     - Apple Silicon optimized (n_jobs=8)
     - Training time: 3-8 minutes
   - 3-5 fold stratified cross-validation
   - Feature importance analysis (coefficients/Gini)

4. **Inference Module** (`src/inference.py`)
   - Batch prediction on unlabeled domains
   - Probability scores and confidence thresholds
   - High throughput: ≥50,000 domains/second

5. **DNS Validation Module** (`src/dns_validation.py`)
   - Async DNS queries for NS records
   - Parking detection using MISP warninglists
   - Nameservers include: sedoparking.com, bodis.com, parkingcrew.net
   - Concurrency: 100-500 simultaneous queries

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pipeline Execution
```bash
# Step 1: Collect training data
python scripts/01_collect_data.py

# Step 1 (with custom data sources):
# - Gambling domains from local txt file
python scripts/01_collect_data.py --gambling-file data/raw/verified_casinos.txt

# - Benign domains from local CSV file (Tranco format: rank,domain)
python scripts/01_collect_data.py --benign-file data/raw/benign_domains.csv

# - Both custom gambling and benign domains
python scripts/01_collect_data.py \
  --gambling-file data/raw/verified_casinos.txt \
  --benign-file data/raw/benign_domains.csv

# Step 2: Train model with hyperparameter tuning

#  Option A: Logistic Regression (default, faster)
python scripts/02_train_model.py --tune-hyperparams

# Option B: Random Forest (higher accuracy, M1 optimized)
python scripts/02_train_model.py --model random_forest --tune-hyperparams

# Step 3: Run inference on unlabeled domains
python scripts/03_run_inference.py --input data/raw/unlabeled_test.txt

# Step 4: Validate predictions via DNS
python scripts/04_validate_dns.py --input data/results/predictions.csv
```

### Analysis
```bash
# Launch Jupyter for exploratory analysis
jupyter notebook

# Notebooks in order:
# 1. notebooks/01_eda.ipynb - Exploratory Data Analysis
# 2. notebooks/02_feature_engineering.ipynb - Feature analysis
# 3. notebooks/03_model_evaluation.ipynb - Model evaluation
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_model.py
pytest tests/test_dns.py
```

## Key Technical Requirements

### Performance Targets
- **Model Accuracy:** ≥92% on holdout set
- **Precision (gambling):** ≥90%
- **Recall (gambling):** ≥85%
- **Inference Speed:** ≥50,000 domains/second
- **DNS Queries Reduction:** ≥60% filtered before DNS lookups
- **Parking Detection:** ≥90% accuracy

### Model Configurations

**Logistic Regression** (default):
```python
LogisticRegression(
    C=10.0,           # Optimal from hyperparameter tuning
    penalty='l2',
    solver='liblinear',
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
```

**Random Forest** (Apple Silicon optimized):
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=8,          # M1 Max: 8 performance cores
    class_weight='balanced',
    random_state=42
)
```

### Model Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 92.29% | ~93-95% (estimated) |
| Precision | 95.36% | ~94-96% |
| Recall | 88.86% | ~90-93% |
| Training Time (159k samples) | 2-5 min | 3-8 min |
| Inference Speed | <1 ms/domain | 2-5 ms/domain |
| Model Size | ~5 MB | ~50-100 MB |
| Interpretability | High (coefficients) | Medium (Gini importance) |
| Use Case | Production, fast inference | Higher accuracy needed |

### DNS Validation
- Async resolver (aiodns or dnspython)
- Timeout: 3-5 seconds per query
- Retry logic: 2 attempts on failure
- Success rate target: ≥95%

## Data Structure

```
data/
├── raw/                    # Source data
│   ├── gambling_domains.txt
│   ├── benign_domains.txt
│   └── unlabeled_test.txt
├── processed/              # Prepared datasets
│   ├── train.csv
│   ├── test.csv
│   └── features.csv
└── results/                # Inference outputs
    ├── predictions.csv
    └── dns_validation.csv

models/
├── logistic_regression.joblib    # Logistic Regression model
├── random_forest.joblib           # Random Forest model (optional)
├── tfidf_vectorizer.joblib        # Feature vectorizer
├── feature_metadata.joblib        # Feature names metadata
├── model_metrics.json             # LogReg metrics
├── model_metrics_rf.json          # RF metrics (if trained)
└── rf_metadata.json               # RF config (if trained)
```

## Dependencies

### Core ML Stack
- scikit-learn==1.5.0 - Machine learning
- pandas==2.2.0 - Data manipulation
- numpy==1.26.0 - Numerical operations
- matplotlib==3.8.0 - Visualization
- joblib==1.3.0 - Model serialization

### DNS Operations
- aiodns==3.2.0 - Async DNS resolver
- dnspython==2.6.0 - DNS queries alternative
- aiohttp==3.9.0 - Async HTTP

### Utilities
- requests==2.31.0 - HTTP requests
- tqdm==4.66.0 - Progress bars

## Important Implementation Notes

### Feature Engineering
- Character n-grams capture l33t speak variations (cas1no, p0ker, b3t)
- TF-IDF vectorization for top 100-500 n-grams
- Multilingual keyword matching for international domains
- Vectorizer must be saved for consistent inference

### Model Training
- Use `class_weight='balanced'` to handle class imbalance
- Hyperparameter grid: C=[0.01, 0.1, 1.0, 10.0], penalty=['l1', 'l2']
- Always report feature importance (top 20 coefficients)
- Generate confusion matrix and ROC curve

### DNS Validation
- Reference MISP warninglists for parking nameservers
- Implement proper timeout and retry logic
- Track query success/failure rates
- Consider alternative resolvers for fallback

## Parking Nameservers (partial list)
```
sedoparking.com
bodis.com
parkingcrew.net
domaincontrol.com (GoDaddy)
registrar-servers.com (Namecheap)
afternic.com
namebrightdns.com
parklogic.com
above.com
```

## Output Formats

### Predictions CSV
- `domain` - Domain name
- `prediction` - 0=benign, 1=gambling
- `probability_benign` - Benign class probability
- `probability_gambling` - Gambling class probability
- `confidence` - Max probability score

### DNS Validation CSV
- `domain` - Domain name
- `ns_records` - JSON list of nameservers
- `is_parked` - Boolean flag
- `parking_provider` - Provider name if detected
- `query_status` - success/timeout/nxdomain

## Success Criteria

- Training dataset: 2,000-5,000 domains per class
- Test dataset: 10,000-20,000 unlabeled domains
- Model size: <50MB
- Memory footprint: <500MB during inference
- DNS query reduction: ≥60% of domains filtered
- Code coverage: ≥70%
