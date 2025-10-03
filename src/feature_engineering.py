"""Feature engineering module for domain classification."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
from pathlib import Path
from typing import Dict, List, Tuple


# Gambling keywords (multilingual and variations)
GAMBLING_KEYWORDS = [
    # English
    'casino', 'bet', 'betting', 'poker', 'slots', 'roulette', 'jackpot',
    'bingo', 'wager', 'gamble', 'gambling', 'lottery', 'spin', 'sportsbook',
    'odds', 'blackjack', 'craps', 'dice', 'vegas', 'stake', 'win', 'bonus',
    # Multilingual
    'kasino', 'apuesta', 'apostas', 'wetten', 'juego', 'jeu',
    # L33t speak variations
    'cas1no', 'c4sino', 'b3t', 'p0ker', 'sl0ts', 'gambl3',
]

# Gambling-specific TLDs
GAMBLING_TLDS = ['.bet', '.casino', '.poker', '.game', '.win', '.games']

# Common legitimate TLDs
COMMON_TLDS = ['.com', '.net', '.org', '.io', '.co', '.uk', '.de', '.fr', '.es']


class DomainFeatureExtractor:
    """Extract features from domain names."""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.manual_feature_names = None  # Store expected manual feature columns

    def extract_keyword_features(self, domain: str) -> Dict[str, float]:
        """Extract gambling keyword features."""
        features = {}
        domain_lower = domain.lower()

        # Count keyword occurrences
        total_keyword_count = 0
        for keyword in GAMBLING_KEYWORDS:
            count = domain_lower.count(keyword)
            if count > 0:
                features[f'keyword_{keyword}'] = count
                total_keyword_count += count

        features['total_keyword_count'] = total_keyword_count
        features['has_gambling_keyword'] = float(total_keyword_count > 0)

        return features

    def extract_tld_features(self, domain: str) -> Dict[str, float]:
        """Extract TLD-based features."""
        features = {}

        # Check for gambling TLDs
        for tld in GAMBLING_TLDS:
            features[f'tld_{tld.replace(".", "")}'] = float(domain.endswith(tld))

        # Check for common TLDs
        for tld in COMMON_TLDS:
            features[f'tld_{tld.replace(".", "")}'] = float(domain.endswith(tld))

        return features

    def extract_structural_features(self, domain: str) -> Dict[str, float]:
        """Extract structural features from domain."""
        features = {}

        # Length
        features['domain_length'] = len(domain)

        # Number of subdomains (count dots)
        features['dot_count'] = domain.count('.')

        # Number presence
        digits = sum(c.isdigit() for c in domain)
        features['digit_count'] = digits
        features['has_digits'] = float(digits > 0)

        # Hyphen presence
        hyphens = domain.count('-')
        features['hyphen_count'] = hyphens
        features['has_hyphen'] = float(hyphens > 0)

        # Ratio of digits to letters
        letters = sum(c.isalpha() for c in domain)
        if letters > 0:
            features['digit_letter_ratio'] = digits / letters
        else:
            features['digit_letter_ratio'] = 0

        # Consecutive repeated characters
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(domain)):
            if domain[i] == domain[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        features['max_consecutive_chars'] = max_consecutive

        # Vowel/consonant ratio
        vowels = sum(1 for c in domain.lower() if c in 'aeiou')
        if letters > 0:
            features['vowel_ratio'] = vowels / letters
        else:
            features['vowel_ratio'] = 0

        return features

    def extract_features_single(self, domain: str) -> Dict[str, float]:
        """Extract all features for a single domain (except n-grams)."""
        features = {}

        features.update(self.extract_keyword_features(domain))
        features.update(self.extract_tld_features(domain))
        features.update(self.extract_structural_features(domain))

        return features

    def fit_tfidf(self, domains: List[str], max_features: int = 500):
        """Fit TF-IDF vectorizer on character n-grams."""
        print(f"Fitting TF-IDF vectorizer on {len(domains)} domains...")

        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 3),  # Bigrams and trigrams
            max_features=max_features,
            lowercase=True,
            min_df=2,  # Ignore very rare n-grams
        )

        self.tfidf_vectorizer.fit(domains)
        print(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")

    def create_feature_matrix(self, domains: List[str], is_training: bool = False) -> pd.DataFrame:
        """Create feature matrix for a list of domains."""
        print(f"Extracting features for {len(domains)} domains...")

        # Extract manual features with progress bar
        from tqdm import tqdm
        manual_features = []
        for domain in tqdm(domains, desc="Extracting manual features", unit="domain"):
            features = self.extract_features_single(domain)
            manual_features.append(features)

        manual_df = pd.DataFrame(manual_features)

        # Fill missing columns with 0 (for keywords that don't appear)
        manual_df = manual_df.fillna(0)

        # If training, store the column names; if not, align to training columns
        if is_training or self.manual_feature_names is None:
            self.manual_feature_names = manual_df.columns.tolist()
        else:
            # Ensure test data has same columns as training data
            for col in self.manual_feature_names:
                if col not in manual_df.columns:
                    manual_df[col] = 0
            # Keep only the columns from training (drop any extra)
            manual_df = manual_df[self.manual_feature_names]

        # Extract TF-IDF features
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

        tfidf_matrix = self.tfidf_vectorizer.transform(domains)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'ngram_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        # Combine all features
        feature_df = pd.concat([manual_df, tfidf_df], axis=1)

        # Store feature names (only once during training)
        if is_training or not self.feature_names:
            self.feature_names = feature_df.columns.tolist()

        print(f"Feature matrix shape: {feature_df.shape}")
        return feature_df

    def save(self, path: Path):
        """Save the feature extractor."""
        path.mkdir(parents=True, exist_ok=True)

        vectorizer_path = path / 'tfidf_vectorizer.joblib'
        joblib.dump(self.tfidf_vectorizer, vectorizer_path)

        metadata_path = path / 'feature_metadata.joblib'
        joblib.dump({
            'feature_names': self.feature_names,
            'manual_feature_names': self.manual_feature_names,
        }, metadata_path)

        print(f"Saved feature extractor to {path}")

    def load(self, path: Path):
        """Load the feature extractor."""
        vectorizer_path = path / 'tfidf_vectorizer.joblib'
        self.tfidf_vectorizer = joblib.load(vectorizer_path)

        metadata_path = path / 'feature_metadata.joblib'
        metadata = joblib.load(metadata_path)
        self.feature_names = metadata['feature_names']
        self.manual_feature_names = metadata.get('manual_feature_names', None)

        print(f"Loaded feature extractor from {path}")


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
    """Prepare features for training and testing."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    extractor = DomainFeatureExtractor()

    # Fit TF-IDF on training data
    extractor.fit_tfidf(train_df['domain'].tolist())

    # Create feature matrices
    X_train = extractor.create_feature_matrix(train_df['domain'].tolist(), is_training=True)
    y_train = train_df['label'].values

    X_test = extractor.create_feature_matrix(test_df['domain'].tolist(), is_training=False)
    y_test = test_df['label'].values

    # Save extractor
    extractor.save(Path('models'))

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_test, y_test, extractor


if __name__ == '__main__':
    # Test with sample domains
    test_domains = [
        'bet365.com',
        'pokerstars.net',
        'google.com',
        'wikipedia.org',
        'casino-royal.bet',
        'github.com',
        'sl0ts-jackpot.com',
    ]

    extractor = DomainFeatureExtractor()
    extractor.fit_tfidf(test_domains)

    features = extractor.create_feature_matrix(test_domains)
    print(features)
