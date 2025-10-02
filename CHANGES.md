# Changes Summary

## Improvements Made

### 1. Domain Normalization (CT Logs Support)

**Problem**: CT logs contain full subdomains (e.g., `www.subdomain.example.com`), but we should classify the registered domain (e.g., `example.com`). Special handling needed for country-code TLDs like `.co.uk`, `.co.nz`.

**Solution**:
- Created `src/utils.py` with `normalize_domain()` function
- Uses `tldextract` library to properly extract registered domain
- Handles all cases:
  - `www.example.com` → `example.com`
  - `sub.example.co.uk` → `example.co.uk`
  - `a.b.c.example.org` → `example.org`

**Changes**:
- `src/utils.py` - New utility module
- `src/data_collection.py` - Normalizes all domains during collection
- `src/inference.py` - Normalizes domains before classification
- Added `test_normalization.py` to verify correctness

**Testing**: All 12 test cases pass ✓

### 2. Improved Gambling Training Data

**Problem**: Only 2,500 gambling domains from BlockList Project, leading to imbalanced dataset.

**Solution**:
- Added Hagezi Medium List (~40,000 gambling domains)
- URL: `https://cdn.jsdelivr.net/gh/hagezi/dns-blocklists@latest/wildcard/gambling.medium-onlydomains.txt`

**Changes**:
- `src/data_collection.py` - Updated `GAMBLING_SOURCES` list
- `TRAINING_GUIDE.md` - Updated documentation

**Expected Impact**:
- Much larger gambling dataset (2.5k → 40k)
- Better balance with benign domains
- Improved model accuracy and recall

## Files Modified

1. **New Files**:
   - `src/utils.py` - Domain normalization utilities
   - `test_normalization.py` - Test suite for normalization

2. **Modified Files**:
   - `src/data_collection.py` - Added normalization and Hagezi source
   - `src/inference.py` - Added normalization before inference
   - `TRAINING_GUIDE.md` - Updated documentation

## How to Use

### Test Domain Normalization
```bash
source venv/bin/activate
python test_normalization.py
```

### Retrain with New Data
```bash
source venv/bin/activate

# Option 1: Full pipeline
python scripts/run_full_pipeline.py

# Option 2: Step by step
python scripts/01_collect_data.py
python scripts/02_train_model.py --tune-hyperparams
```

### Inference Output
The inference results now include both:
- `original_domain` - Input domain as provided
- `domain` - Normalized registered domain used for classification

Example output:
```
original_domain,domain,prediction,probability_gambling,label
www.bet365.com,bet365.com,1,0.9999,gambling
subdomain.example.co.uk,example.co.uk,0,0.02,benign
```

## Next Steps

1. **Retrain the model** with the improved dataset:
   - 40k+ gambling domains
   - Balanced benign domains
   - All normalized to registered domains

2. **Verify performance** on CT log data:
   - Test with subdomain variations
   - Check country-code TLD handling
   - Measure accuracy improvements

3. **Production deployment**:
   - Normalize CT log domains before inference
   - Use `original_domain` for reporting
   - Use `domain` for classification
