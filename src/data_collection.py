"""Data collection module for fetching gambling and benign domains."""

import requests
import tldextract
import pandas as pd
from typing import List, Set
from pathlib import Path
import re
from .utils import normalize_domain


# Data source URLs
GAMBLING_SOURCES = [
    #    "https://raw.githubusercontent.com/blocklistproject/Lists/master/gambling.txt",
    "https://cdn.jsdelivr.net/gh/hagezi/dns-blocklists@latest/wildcard/gambling.medium-onlydomains.txt",
]

BENIGN_SOURCES = [
    "https://tranco-list.eu/top-1m.csv.zip",  # Will need special handling
]

# Gambling keywords for filtering benign domains
GAMBLING_KEYWORDS = [
    "casino",
    "bet",
    "betting",
    "poker",
    "slots",
    "roulette",
    "jackpot",
    "bingo",
    "wager",
    "gamble",
    "gambling",
    "lottery",
    "spin",
    "sportsbook",
    "odds",
    "kasino",
    "apuesta",
    "apostas",
    "wetten",
]


def fetch_from_url(url: str) -> List[str]:
    """Fetch domain list from URL."""
    print(f"Fetching from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text.splitlines()


def clean_domain(line: str) -> str:
    """Extract clean domain from various formats."""
    # Remove comments
    if "#" in line:
        line = line.split("#")[0]

    line = line.strip()
    if not line:
        return ""

    # Remove common prefixes
    line = re.sub(r"^https?://", "", line)
    line = re.sub(r"^www\.", "", line)
    line = re.sub(r"^[0-9.]+\s+", "", line)  # Remove IP addresses (hosts file format)

    # Remove port numbers
    line = re.sub(r":[0-9]+$", "", line)

    # Remove paths
    if "/" in line:
        line = line.split("/")[0]

    return line.strip().lower()


def validate_domain(domain: str) -> bool:
    """Validate domain format."""
    if not domain or len(domain) < 4:
        return False

    # Must have at least one dot
    if "." not in domain:
        return False

    # Basic character check
    if not re.match(r"^[a-z0-9.-]+$", domain):
        return False

    # Use tldextract to validate
    try:
        extracted = tldextract.extract(domain)
        return bool(extracted.domain and extracted.suffix)
    except:
        return False


def has_gambling_keyword(domain: str) -> bool:
    """Check if domain contains gambling keywords."""
    domain_lower = domain.lower()
    return any(keyword in domain_lower for keyword in GAMBLING_KEYWORDS)


def fetch_gambling_domains() -> Set[str]:
    """Fetch gambling domains from multiple sources."""
    all_domains = set()

    for url in GAMBLING_SOURCES:
        try:
            lines = fetch_from_url(url)
            for line in lines:
                domain = clean_domain(line)
                if domain and validate_domain(domain):
                    # Normalize to registered domain (handles subdomains, co.uk, etc.)
                    normalized = normalize_domain(domain)
                    all_domains.add(normalized)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    print(f"Collected {len(all_domains)} unique gambling domains")
    return all_domains


def fetch_benign_domains(limit: int = 100000) -> Set[str]:
    """Fetch benign domains from local file or Tranco CSV."""
    all_domains = set()

    # Try to load from local benign_domains.txt
    benign_file = Path("data/raw/benign_domains.txt")
    if benign_file.exists():
        print(f"Loading benign domains from {benign_file}")
        with open(benign_file, "r") as f:
            for line in f:
                domain = clean_domain(line)
                if (
                    domain
                    and validate_domain(domain)
                    and not has_gambling_keyword(domain)
                ):
                    # Normalize to registered domain
                    normalized = normalize_domain(domain)
                    all_domains.add(normalized)
                    if len(all_domains) >= limit:
                        break
        print(f"Collected {len(all_domains)} unique benign domains from file")
        return all_domains

    # Try to load from Tranco CSV (format: rank,domain)
    tranco_file = Path("data/raw/top-1m.csv")
    if tranco_file.exists():
        print(f"Loading benign domains from {tranco_file}")
        with open(tranco_file, "r") as f:
            for line in f:
                if "," in line:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        domain = clean_domain(parts[1])
                    else:
                        domain = clean_domain(parts[0])
                else:
                    domain = clean_domain(line)

                if (
                    domain
                    and validate_domain(domain)
                    and not has_gambling_keyword(domain)
                ):
                    # Normalize to registered domain
                    normalized = normalize_domain(domain)
                    all_domains.add(normalized)
                    if len(all_domains) >= limit:
                        break
        print(f"Collected {len(all_domains)} unique benign domains from Tranco list")
        return all_domains

    # Fallback: use hardcoded examples
    print("Warning: No benign domain file found. Using hardcoded examples.")
    print("Please create data/raw/benign_domains.txt or data/raw/top-1m.csv")
    example_benign = [
        "google.com",
        "youtube.com",
        "facebook.com",
        "twitter.com",
        "instagram.com",
        "linkedin.com",
        "reddit.com",
        "amazon.com",
        "wikipedia.org",
        "apple.com",
        "microsoft.com",
        "github.com",
        "stackoverflow.com",
        "medium.com",
        "netflix.com",
        "spotify.com",
    ]

    for domain in example_benign:
        if validate_domain(domain) and not has_gambling_keyword(domain):
            # Normalize to registered domain (though these are already normalized)
            normalized = normalize_domain(domain)
            all_domains.add(normalized)

    print(
        f"Collected {len(all_domains)} unique benign domains (limited set for testing)"
    )
    return all_domains


def create_dataset(
    gambling_domains: Set[str],
    benign_domains: Set[str],
    output_path: Path,
    train_ratio: float = 0.8,
):
    """Create train/test datasets with labels."""
    # Create labeled dataset
    gambling_df = pd.DataFrame(
        {
            "domain": list(gambling_domains),
            "label": 1,  # gambling = 1
        }
    )

    benign_df = pd.DataFrame(
        {
            "domain": list(benign_domains),
            "label": 0,  # benign = 0
        }
    )

    # Combine and shuffle
    full_df = pd.concat([gambling_df, benign_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split train/test
    split_idx = int(len(full_df) * train_ratio)
    train_df = full_df[:split_idx]
    test_df = full_df[split_idx:]

    # Save
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nDataset created:")
    print(
        f"  Train: {len(train_df)} domains ({train_df['label'].sum()} gambling, {(~train_df['label'].astype(bool)).sum()} benign)"
    )
    print(
        f"  Test: {len(test_df)} domains ({test_df['label'].sum()} gambling, {(~test_df['label'].astype(bool)).sum()} benign)"
    )
    print(f"  Saved to {output_path}")

    return train_df, test_df


def collect_data():
    """Main data collection pipeline."""
    print("=" * 60)
    print("GAMBLING DOMAIN CLASSIFIER - DATA COLLECTION")
    print("=" * 60)

    # Fetch data
    gambling_domains = fetch_gambling_domains()
    benign_domains = fetch_benign_domains()

    # Create datasets
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = create_dataset(gambling_domains, benign_domains, output_path)

    return train_df, test_df


if __name__ == "__main__":
    collect_data()
