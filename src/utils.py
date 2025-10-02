"""Utility functions for domain processing."""

import tldextract


def normalize_domain(domain: str) -> str:
    """
    Normalize domain to registered domain (second-level + TLD).

    Handles special cases like co.uk, co.nz, etc.

    Examples:
        www.example.com -> example.com
        subdomain.example.co.uk -> example.co.uk
        a.b.c.example.org -> example.org

    Args:
        domain: Domain name (may include subdomains, protocol, etc.)

    Returns:
        Normalized registered domain
    """
    # Extract components using tldextract
    extracted = tldextract.extract(domain)

    # Registered domain = domain + suffix
    # This handles co.uk, co.nz, etc. automatically
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"

    # Fallback: return as-is if extraction failed
    return domain.lower()


def normalize_domains_batch(domains: list) -> list:
    """Normalize a batch of domains."""
    return [normalize_domain(d) for d in domains]
