# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML classifier for gambling-themed domains with DNS validation. The system processes domains from Certificate Transparency logs, classifying them as gambling/non-gambling using logistic regression and validating results through DNS queries.

**Primary Language:** Python 3.9+
**Project Type:** Machine learning classification pipeline

## Project Architecture

### Pipeline Flow
```
Data Collection → Feature Engineering → Model Training →
Model Evaluation → Inference → DNS Validation → Results Analysis
```

### Core Components

1. **Data Collection Module** (`src/data_collection.py`)
   - Fetches gambling domains from BlockList Project and Hagezi
   - Collects benign domains from Cloudflare Radar and Tranco
   - Handles deduplication and domain validation
   - Creates 80/20 train/test split

2. **Feature Engineering Module** (`src/feature_engineering.py`)
   - Keyword-based features (gambling terms: casino, bet, poker, slots, etc.)
   - Multilingual support (kasino, apuesta, казино, wetten, etc.)
   - TLD features (.bet, .casino, .poker)
   - Structural features (domain length, numbers, hyphens)
   - Character n-grams with TF-IDF vectorization

3. **Model Training Module** (`src/model_training.py`)
   - Logistic Regression with L2 regularization
   - Hyperparameter tuning via GridSearchCV
   - 5-fold stratified cross-validation
   - Feature importance analysis

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

# Step 2: Train model with hyperparameter tuning
python scripts/02_train_model.py --tune-hyperparams

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

### Model Configuration
```python
LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

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
├── logistic_regression.joblib    # Trained model
├── tfidf_vectorizer.joblib       # Feature vectorizer
└── model_metrics.json            # Performance metrics
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
