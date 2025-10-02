# Training Guide - Gambling Domain Classifier

## Quick Start

### 1. Add Benign Training Data

**Option A: Use the provided starter list (100 domains)**
```bash
# Already created at data/raw/benign_domains.txt
# Contains popular sites: google.com, github.com, netflix.com, etc.
# Note: Domains are automatically normalized to registered domain
# e.g., www.example.co.uk -> example.co.uk
```

**Option B: Download larger dataset (recommended for production)**
```bash
# Download Cisco Umbrella Top 1M domains
cd data/raw
curl -O http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip
unzip top-1m.csv.zip
cd ../..

# The script will automatically use top-1m.csv if it exists
```

**Option C: Add your own domains**
```bash
# Edit data/raw/benign_domains.txt
# Add one domain per line (no http://, no www.)
# Example:
#   example.com
#   mysite.org
#   another-domain.net
```

### 2. Train the Model

**Full pipeline (recommended):**
```bash
source venv/bin/activate
python scripts/run_full_pipeline.py
```

**Or step-by-step:**
```bash
source venv/bin/activate

# Step 1: Collect and prepare data
python scripts/01_collect_data.py

# Step 2: Train model with hyperparameter tuning
python scripts/02_train_model.py --tune-hyperparams

# Check the results in models/model_metrics.json
```

### 3. Run Inference

**On your own domains:**
```bash
# Create a file with domains to classify (one per line)
cat > data/raw/my_domains.txt <<EOF
example-casino.com
my-business.com
suspicious-domain.net
EOF

# Run inference
python scripts/03_run_inference.py --input data/raw/my_domains.txt

# Results saved to:
#   data/results/predictions.csv - all predictions
#   data/results/predicted_gambling.csv - gambling domains only
#   data/results/prediction_analysis.png - visualization
```

## File Locations

### Input Files
- `data/raw/benign_domains.txt` - Your benign domains (one per line)
- `data/raw/top-1m.csv` - Optional: Cisco/Tranco top domains CSV
- Gambling domains are auto-fetched from GitHub blocklists

### Output Files
- `data/processed/train.csv` - Training dataset (80%)
- `data/processed/test.csv` - Test dataset (20%)
- `models/logistic_regression.joblib` - Trained model
- `models/tfidf_vectorizer.joblib` - Feature vectorizer
- `models/model_metrics.json` - Performance metrics
- `models/*.png` - Visualizations (ROC curve, confusion matrix, etc.)

## Training Parameters

### Default Settings (in src/data_collection.py)
- **Gambling sources**:
  - BlockList Project (~2,500 domains)
  - Hagezi Medium List (~40,000 domains) - auto-fetched
- **Benign limit**: 10,000 domains max
- **Train/test split**: 80/20
- **Random seed**: 42 (for reproducibility)
- **Domain normalization**: All domains normalized to registered domain
  - Handles subdomains: `www.example.com` → `example.com`
  - Handles country codes: `shop.example.co.uk` → `example.co.uk`

### Hyperparameters (tuned automatically)
- **C** (regularization): [0.01, 0.1, 1.0, 10.0]
- **Penalty**: l1, l2
- **Solver**: lbfgs, liblinear, saga
- **Class weight**: balanced (handles imbalance)
- **Max iterations**: 1000

## Expected Results

With balanced data (2000+ domains per class), you should see:
- **Accuracy**: >95%
- **Precision (gambling)**: >90%
- **Recall (gambling)**: >85%
- **Training time**: 1-3 minutes
- **Model size**: <50MB

## Troubleshooting

### "Only 16 benign domains collected"
→ Add data/raw/benign_domains.txt or download top-1m.csv

### "Features don't match"
→ Retrain from scratch: delete models/ folder and run pipeline again

### "Low accuracy on benign class"
→ Add more diverse benign domains (aim for 2000+ to match gambling domains)

### "Convergence warnings"
→ Normal during hyperparameter search, final model should converge

## Advanced: Custom Configuration

Edit `src/data_collection.py` to customize:
- Add more gambling keyword sources
- Change train/test split ratio
- Adjust benign domain filtering rules
- Add custom domain validation logic

Edit `src/feature_engineering.py` to customize:
- Add/remove gambling keywords
- Change TLD lists
- Adjust n-gram range (currently 2-3)
- Change max_features (currently 500)

Edit `src/model_training.py` to customize:
- Hyperparameter search grid
- Cross-validation folds (currently 5)
- Scoring metric (currently F1)
