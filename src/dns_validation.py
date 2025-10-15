"""DNS validation module for gambling domain classification - OPTIMIZED for Unbound."""

import asyncio
import time
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass, asdict
import aiodns
import socket
import ipaddress


# Known parking nameservers (from specs and MISP warninglists)
PARKING_NAMESERVERS_2025 = {
    'sedoparking.com',
    'bodis.com',
    'parkingcrew.net',
    'parklogic.com',
    'above.com',
    'afternic.com',
    'namebrightdns.com',
    'dns-parking.com',
    'ztomy.com',
}

# Known parking IPs (updated 2025)
PARKING_IPS_2025 = {
    '3.33.130.190',      # AWS, 28,700 domains
    '15.197.148.33',     # AWS, 28,852 domains  
    '76.223.26.96',      # AWS, 65,965 domains
    '75.2.115.196',      # AWS, 55,237 domains
    '75.2.18.233',       # AWS, 40,858 domains
    '13.248.148.254',    # AWS, 65,617 domains
    '13.248.213.45',     # AWS, 17,051 domains
    '76.223.67.189',     # AWS, 17,392 domains
    '15.197.225.128',    # AWS, 12,464 domains
    '15.197.130.221',    # AWS, 13,017 domains
    '13.248.243.5',      # AWS, 11,148 domains
    '76.223.105.230',    # AWS, 10,939 domains
    '3.33.251.168',      # AWS, 12,337 domains
    
    # Bodis (known parking service)
    '199.59.243.228',    # Bodis, 37,512 domains
    
    # Sedo (parking service)
    '91.195.240.12',     # Sedo, 21,231 domains
    '91.195.240.94',     # Sedo, 4,111 domains
    '64.190.63.222',     # Sedo, 7,805 domains
    '91.195.240.19',     # Sedo, 5,772 domains
    
    # Confluence Networks
    '208.91.197.27',     # 24,637 domains
    
    # Google Cloud (high domain count)
    '34.149.87.45',      # Google Cloud, 22,064 domains
    
    # DigitalOcean (high domain count)
    '209.38.154.253',    # 16,752 domains
    
    # Shopify edge case (already in your list)
    '23.227.38.65',      # Shopify, 11,954 domains
}

# Parking IP ranges (CIDR notation)
PARKING_IP_RANGES_2025 = [
    '185.53.176.0/22',   # Covers 185.53.176.0 - 185.53.179.255
]

# Parse CIDR ranges into ip_network objects (done once at module load)
PARKING_NETWORKS = [ipaddress.ip_network(cidr) for cidr in PARKING_IP_RANGES_2025]

# Local Unbound DNS server
DNS_SERVERS = [
    '127.0.0.1',         # Local Unbound resolver
]


@dataclass
class DNSResult:
    """DNS validation result for a single domain."""
    domain: str
    ns_records: List[str]
    a_records: List[str]
    cname_records: List[str]
    is_parked: bool
    parking_provider: Optional[str]
    query_status: str  # success, timeout, nxdomain, error
    query_time: float


class DNSValidator:
    """Validates domains using DNS queries to detect parking - OPTIMIZED for local Unbound."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 2,
        concurrency: int = 1500,  # INCREASED from 100
        resolver_pool_size: int = 50  # NEW: Pool of resolvers
    ):
        """
        Initialize DNS validator.

        Args:
            timeout: Timeout in seconds for DNS queries (increased for reliability)
            max_retries: Maximum retry attempts for failed queries
            concurrency: Maximum concurrent queries (optimized for Unbound)
            resolver_pool_size: Number of DNS resolver instances in pool
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.resolver_pool_size = resolver_pool_size
        self.semaphore = asyncio.Semaphore(concurrency)

        # Lazy initialization - create resolvers inside event loop
        self.resolver_pool = None
        self.resolver_index = 0
        self.resolver_lock = asyncio.Lock()

        # Statistics for monitoring
        self.stats = {
            'queries': 0,
            'success': 0,
            'errors': 0,
            'timeouts': 0,
            'nxdomain': 0
        }
        self.stats_lock = asyncio.Lock()

    async def _get_resolver(self) -> aiodns.DNSResolver:
        """Get next resolver from pool in round-robin fashion. Creates pool on first use."""
        async with self.resolver_lock:
            # Lazy initialization - create pool inside event loop
            if self.resolver_pool is None:
                self.resolver_pool = [
                    aiodns.DNSResolver(
                        timeout=self.timeout,
                        nameservers=DNS_SERVERS,
                        tries=1
                    )
                    for _ in range(self.resolver_pool_size)
                ]

            resolver = self.resolver_pool[self.resolver_index]
            self.resolver_index = (self.resolver_index + 1) % len(self.resolver_pool)
            return resolver

    def _check_parking_ns(self, ns_records: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if NS records indicate parking.

        Args:
            ns_records: List of nameserver records

        Returns:
            Tuple of (is_parked, parking_provider)
        """
        for ns in ns_records:
            ns_lower = ns.lower().rstrip('.')
            for parking_ns in PARKING_NAMESERVERS_2025:
                if parking_ns in ns_lower:
                    return True, parking_ns
        return False, None

    def _check_parking_ip(self, a_records: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if A records indicate parking (supports individual IPs and CIDR ranges).

        Args:
            a_records: List of A records (IP addresses)

        Returns:
            Tuple of (is_parked, parking_provider)
        """
        for ip_str in a_records:
            # Check individual IPs first (fast set lookup)
            if ip_str in PARKING_IPS_2025:
                return True, f"IP:{ip_str}"

            # Check CIDR ranges
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                for network in PARKING_NETWORKS:
                    if ip_obj in network:
                        return True, f"CIDR:{network}"
            except ValueError:
                # Invalid IP address, skip
                pass

        return False, None

    async def _query_ns_records(self, domain: str, resolver: aiodns.DNSResolver) -> List[str]:
        """
        Query NS records for a domain.

        Args:
            domain: Domain name to query
            resolver: aiodns resolver instance

        Returns:
            List of nameserver records
        """
        try:
            result = await resolver.query(domain, 'NS')
            return [ns.host for ns in result]
        except aiodns.error.DNSError:
            return []

    async def _query_a_records(self, domain: str, resolver: aiodns.DNSResolver) -> List[str]:
        """
        Query A records for a domain.

        Args:
            domain: Domain name to query
            resolver: aiodns resolver instance

        Returns:
            List of IP addresses
        """
        try:
            result = await resolver.query(domain, 'A')
            return [r.host for r in result]
        except aiodns.error.DNSError:
            return []

    async def _query_cname_records(self, domain: str, resolver: aiodns.DNSResolver) -> List[str]:
        """
        Query CNAME records for a domain.

        Args:
            domain: Domain name to query
            resolver: aiodns resolver instance

        Returns:
            List of CNAME targets
        """
        try:
            result = await resolver.query(domain, 'CNAME')
            return [r.cname for r in result]
        except aiodns.error.DNSError:
            return []

    async def _update_stats(self, status: str):
        """Update statistics thread-safely."""
        async with self.stats_lock:
            self.stats['queries'] += 1
            if status == 'success':
                self.stats['success'] += 1
            elif status == 'timeout':
                self.stats['timeouts'] += 1
            elif status == 'nxdomain':
                self.stats['nxdomain'] += 1
            else:
                self.stats['errors'] += 1

    async def _validate_domain(self, domain: str) -> DNSResult:
        """
        Validate a single domain with retries.

        Args:
            domain: Domain to validate

        Returns:
            DNSResult object
        """
        async with self.semaphore:
            start_time = time.time()
            ns_records = []
            a_records = []
            cname_records = []
            query_status = "error"

            for attempt in range(self.max_retries + 1):
                try:
                    # Get resolver from pool
                    resolver = await self._get_resolver()

                    # Query NS, A, and CNAME records concurrently
                    ns_task = self._query_ns_records(domain, resolver)
                    a_task = self._query_a_records(domain, resolver)
                    cname_task = self._query_cname_records(domain, resolver)

                    ns_records, a_records, cname_records = await asyncio.gather(
                        ns_task, a_task, cname_task,
                        return_exceptions=True
                    )

                    # Handle exceptions from gather
                    if isinstance(ns_records, Exception):
                        ns_records = []
                    if isinstance(a_records, Exception):
                        a_records = []
                    if isinstance(cname_records, Exception):
                        cname_records = []

                    # Determine query status
                    if not ns_records and not a_records and not cname_records:
                        query_status = "nxdomain"
                    else:
                        query_status = "success"
                    break

                except asyncio.TimeoutError:
                    query_status = "timeout"
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1)  # Shorter retry delay

                except Exception as e:
                    query_status = f"error: {str(e)}"
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1)

            query_time = time.time() - start_time

            # Update statistics
            await self._update_stats(query_status)

            # Check for parking
            is_parked = False
            parking_provider = None

            if query_status == "success":
                # Check NS records first
                is_parked_ns, provider_ns = self._check_parking_ns(ns_records)
                if is_parked_ns:
                    is_parked = True
                    parking_provider = provider_ns
                else:
                    # Check A records if NS check didn't find parking
                    is_parked_ip, provider_ip = self._check_parking_ip(a_records)
                    if is_parked_ip:
                        is_parked = True
                        parking_provider = provider_ip

            return DNSResult(
                domain=domain,
                ns_records=ns_records,
                a_records=a_records,
                cname_records=cname_records,
                is_parked=is_parked,
                parking_provider=parking_provider,
                query_status=query_status,
                query_time=query_time
            )

    async def validate_batch(self, domains: List[str]) -> List[DNSResult]:
        """
        Validate a batch of domains with progress reporting.

        Args:
            domains: List of domains to validate

        Returns:
            List of DNSResult objects
        """
        total = len(domains)
        print(f"\nStarting validation of {total} domains...")
        print(f"Concurrency: {self.semaphore._value}")
        print(f"Timeout: {self.timeout}s")
        print(f"Max retries: {self.max_retries}")
        
        start_time = time.time()
        
        # Create tasks
        tasks = [self._validate_domain(domain) for domain in domains]
        
        # Progress tracking
        completed = 0
        last_report = start_time
        
        # Process with progress updates
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            # Report progress every 5 seconds
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - start_time
                qps = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / qps if qps > 0 else 0
                
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                      f"QPS: {qps:.0f} | "
                      f"ETA: {eta/60:.1f}m | "
                      f"Success: {self.stats['success']} | "
                      f"Errors: {self.stats['errors']} | "
                      f"Timeouts: {self.stats['timeouts']}")
                last_report = now
        
        elapsed = time.time() - start_time
        final_qps = total / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ Validation complete!")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average QPS: {final_qps:.0f}")
        print(f"Success rate: {self.stats['success']/total*100:.1f}%")
        
        return results

    async def validate_batch_chunked(
        self,
        domains: List[str],
        chunk_size: int = 10000,
        output_path: Optional[Path] = None
    ) -> List[DNSResult]:
        """
        Validate domains in chunks to manage memory efficiently.

        Memory-efficient for large datasets (100k+ domains) by processing in chunks
        and optionally writing results incrementally to disk.

        Args:
            domains: List of domains to validate
            chunk_size: Number of domains to process per chunk (default 10000)
            output_path: If provided, results are written incrementally to this CSV file

        Returns:
            List of DNSResult objects (may be empty if output_path is used)
        """
        from tqdm import tqdm

        total = len(domains)
        num_chunks = (total + chunk_size - 1) // chunk_size

        print(f"\nStarting chunked validation of {total:,} domains...")
        print(f"Chunk size: {chunk_size:,} domains per chunk")
        print(f"Total chunks: {num_chunks}")
        print(f"Concurrency: {self.semaphore._value}")
        print(f"Timeout: {self.timeout}s")

        start_time = time.time()
        all_results = []
        first_chunk = True

        # Process each chunk
        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks", unit="chunk"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_domains = domains[chunk_start:chunk_end]

            # Validate chunk
            chunk_results = await self.validate_batch(chunk_domains)

            # Write incrementally to disk if output path provided
            if output_path:
                # Convert to DataFrame for CSV writing
                chunk_data = [asdict(r) for r in chunk_results]
                chunk_df = pd.DataFrame(chunk_data)

                # Convert lists to JSON strings for CSV
                chunk_df['ns_records'] = chunk_df['ns_records'].apply(json.dumps)
                chunk_df['a_records'] = chunk_df['a_records'].apply(json.dumps)
                chunk_df['cname_records'] = chunk_df['cname_records'].apply(json.dumps)

                # Write to CSV incrementally
                chunk_df.to_csv(
                    output_path,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                # Don't accumulate results in memory
            else:
                all_results.extend(chunk_results)

        elapsed = time.time() - start_time
        final_qps = total / elapsed if elapsed > 0 else 0

        print(f"\n✅ Chunked validation complete!")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"Average QPS: {final_qps:.0f}")
        print(f"Success rate: {self.stats['success']/total*100:.1f}%")

        # Return combined results or empty list if written to disk
        if output_path:
            print(f"Results written incrementally to {output_path}")
            return []
        else:
            return all_results

    def validate_batch_sync(self, domains: List[str]) -> List[DNSResult]:
        """
        Synchronous wrapper for validate_batch.

        Args:
            domains: List of domains to validate

        Returns:
            List of DNSResult objects
        """
        return asyncio.run(self.validate_batch(domains))

    def validate_batch_chunked_sync(
        self,
        domains: List[str],
        chunk_size: int = 10000,
        output_path: Optional[Path] = None
    ) -> List[DNSResult]:
        """
        Synchronous wrapper for validate_batch_chunked.

        Args:
            domains: List of domains to validate
            chunk_size: Number of domains to process per chunk
            output_path: If provided, results are written incrementally to this CSV file

        Returns:
            List of DNSResult objects (may be empty if output_path is used)
        """
        return asyncio.run(self.validate_batch_chunked(domains, chunk_size, output_path))


class DNSValidationPipeline:
    """Complete DNS validation pipeline with reporting."""

    def __init__(self, validator: Optional[DNSValidator] = None):
        """
        Initialize DNS validation pipeline.

        Args:
            validator: DNSValidator instance (creates default if None)
        """
        self.validator = validator or DNSValidator()

    def run(
        self,
        predictions_file: Path,
        output_dir: Path,
        confidence_threshold: float = 0.8,
        filter_gambling: bool = True,
        chunk_size: int = None,
        use_chunked: bool = None
    ) -> Dict:
        """
        Run DNS validation pipeline.

        Args:
            predictions_file: Path to predictions CSV
            output_dir: Directory for output files
            confidence_threshold: Minimum confidence to validate
            filter_gambling: If True, filter for gambling predictions; if False, validate all
            chunk_size: Chunk size for chunked processing (default: 10000)
            use_chunked: Force chunked processing. If None, auto-detect based on dataset size

        Returns:
            Statistics dictionary
        """
        print("=" * 60)
        print("DNS VALIDATION PIPELINE - OPTIMIZED FOR UNBOUND")
        print("=" * 60)

        # Load predictions
        print(f"\nLoading predictions from {predictions_file}...")
        df = pd.read_csv(predictions_file)

        # Filter for gambling predictions above threshold
        if filter_gambling:
            gambling_mask = (df['prediction'] == 1) & (df['confidence'] >= confidence_threshold)
            gambling_domains = df[gambling_mask]['domain'].tolist()
            print(f"Total predictions: {len(df)}")
            print(f"Gambling predictions (≥{confidence_threshold} confidence): {len(gambling_domains)}")
        else:
            # Apply only confidence threshold (assumes all are gambling already)
            if 'confidence' in df.columns:
                gambling_domains = df[df['confidence'] >= confidence_threshold]['domain'].tolist()
                print(f"Total domains: {len(df)}")
                print(f"Domains with ≥{confidence_threshold} confidence: {len(gambling_domains)}")
            else:
                gambling_domains = df['domain'].tolist()
                print(f"Total domains to validate: {len(gambling_domains)}")

        if not gambling_domains:
            print("\nNo gambling domains to validate!")
            return {}

        # Auto-detect whether to use chunked processing
        if use_chunked is None:
            # Use chunked processing for datasets > 50k domains
            use_chunked = len(gambling_domains) > 50000

        # Set default chunk size
        if chunk_size is None:
            chunk_size = 10000

        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / 'dns_validation.csv'

        # Run DNS validation
        print(f"\nValidating {len(gambling_domains):,} domains...")
        print(f"Validator settings:")
        print(f"  - Timeout: {self.validator.timeout}s")
        print(f"  - Concurrency: {self.validator.semaphore._value}")
        print(f"  - Max retries: {self.validator.max_retries}")
        print(f"  - Rate limiting: DISABLED (not needed for local DNS)")

        start_time = time.time()

        if use_chunked:
            print(f"\nUsing chunked processing (chunk_size={chunk_size:,})")
            print("Results will be written incrementally to disk to save memory.\n")

            # Use chunked validation with incremental CSV writing
            results = self.validator.validate_batch_chunked_sync(
                gambling_domains,
                chunk_size=chunk_size,
                output_path=results_path
            )

            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"Completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
            print(f"Average: {len(gambling_domains)/elapsed:.1f} domains/second")
            print(f"{'='*60}")

            print(f"\nFull results saved to {results_path}")

            # Calculate statistics from file in streaming fashion
            print("\nCalculating statistics from saved results...")
            stats = self._calculate_stats_from_file(results_path, len(gambling_domains))

            # Extract and save domain lists from CSV in streaming fashion
            self._save_domain_lists_from_file(results_path, output_dir)

        else:
            print("\nUsing standard batch processing (all in memory)\n")

            results = self.validator.validate_batch_sync(gambling_domains)
            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"Completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
            print(f"Average: {len(gambling_domains)/elapsed:.1f} domains/second")
            print(f"{'='*60}")

            # Convert results to DataFrame
            results_data = [asdict(r) for r in results]
            results_df = pd.DataFrame(results_data)

            # Convert lists to JSON strings for CSV
            results_df['ns_records'] = results_df['ns_records'].apply(json.dumps)
            results_df['a_records'] = results_df['a_records'].apply(json.dumps)
            results_df['cname_records'] = results_df['cname_records'].apply(json.dumps)

            # Calculate statistics
            stats = self._calculate_stats(results)

            # Save outputs
            self._save_outputs(results, results_df, stats, output_dir)

        # Save statistics
        stats_path = output_dir / 'dns_validation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")

        # Print summary
        self._print_summary(stats)

        return stats

    def _calculate_stats(self, results: List[DNSResult]) -> Dict:
        """Calculate validation statistics."""
        total = len(results)
        active = sum(1 for r in results if not r.is_parked and r.query_status == "success")
        parked = sum(1 for r in results if r.is_parked)
        success = sum(1 for r in results if r.query_status == "success")
        timeout = sum(1 for r in results if r.query_status == "timeout")
        nxdomain = sum(1 for r in results if r.query_status == "nxdomain")
        error = sum(1 for r in results if r.query_status.startswith("error"))

        # Parking provider breakdown
        parking_providers = {}
        for r in results:
            if r.is_parked and r.parking_provider:
                parking_providers[r.parking_provider] = parking_providers.get(r.parking_provider, 0) + 1

        # Average query time
        avg_query_time = sum(r.query_time for r in results) / total if total > 0 else 0

        return {
            'total_domains': total,
            'active_domains': active,
            'parked_domains': parked,
            'successful_queries': success,
            'timeout_queries': timeout,
            'nxdomain_queries': nxdomain,
            'error_queries': error,
            'success_rate_pct': (success / total * 100) if total > 0 else 0,
            'parking_rate_pct': (parked / success * 100) if success > 0 else 0,
            'active_rate_pct': (active / success * 100) if success > 0 else 0,
            'avg_query_time': avg_query_time,
            'parking_providers': parking_providers
        }

    def _calculate_stats_from_file(self, csv_path: Path, total_domains: int) -> Dict:
        """
        Calculate statistics from CSV file in streaming fashion.

        Memory-efficient for large result files by processing in chunks.

        Args:
            csv_path: Path to DNS validation results CSV
            total_domains: Total number of domains validated

        Returns:
            Statistics dictionary
        """
        # Initialize counters
        active = 0
        parked = 0
        success = 0
        timeout = 0
        nxdomain = 0
        error = 0
        total_query_time = 0.0
        parking_providers = {}

        # Process CSV in chunks
        chunk_size = 10000
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Count statuses
            success += (chunk['query_status'] == 'success').sum()
            timeout += (chunk['query_status'] == 'timeout').sum()
            nxdomain += (chunk['query_status'] == 'nxdomain').sum()
            error += chunk['query_status'].str.startswith('error').sum()

            # Count active and parked
            active += ((chunk['is_parked'] == False) & (chunk['query_status'] == 'success')).sum()
            parked += (chunk['is_parked'] == True).sum()

            # Accumulate query times
            total_query_time += chunk['query_time'].sum()

            # Count parking providers
            parked_chunk = chunk[chunk['is_parked'] == True]
            for provider in parked_chunk['parking_provider'].dropna():
                parking_providers[provider] = parking_providers.get(provider, 0) + 1

        avg_query_time = total_query_time / total_domains if total_domains > 0 else 0

        return {
            'total_domains': total_domains,
            'active_domains': int(active),
            'parked_domains': int(parked),
            'successful_queries': int(success),
            'timeout_queries': int(timeout),
            'nxdomain_queries': int(nxdomain),
            'error_queries': int(error),
            'success_rate_pct': (success / total_domains * 100) if total_domains > 0 else 0,
            'parking_rate_pct': (parked / success * 100) if success > 0 else 0,
            'active_rate_pct': (active / success * 100) if success > 0 else 0,
            'avg_query_time': avg_query_time,
            'parking_providers': parking_providers
        }

    def _save_domain_lists_from_file(self, csv_path: Path, output_dir: Path):
        """
        Extract and save active/parked domain lists from CSV in streaming fashion.

        Memory-efficient for large result files by processing in chunks.

        Args:
            csv_path: Path to DNS validation results CSV
            output_dir: Directory for output files
        """
        active_path = output_dir / 'active_gambling_domains.txt'
        parked_path = output_dir / 'parked_gambling_domains.txt'

        # Open output files
        with open(active_path, 'w') as active_file, open(parked_path, 'w') as parked_file:
            active_count = 0
            parked_count = 0

            # Process CSV in chunks
            chunk_size = 10000
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                # Extract active domains
                active_domains = chunk[
                    (chunk['is_parked'] == False) & (chunk['query_status'] == 'success')
                ]['domain'].tolist()

                # Extract parked domains
                parked_domains = chunk[chunk['is_parked'] == True]['domain'].tolist()

                # Write to files
                if active_domains:
                    active_file.write('\n'.join(active_domains) + '\n')
                    active_count += len(active_domains)

                if parked_domains:
                    parked_file.write('\n'.join(parked_domains) + '\n')
                    parked_count += len(parked_domains)

        print(f"Active domains saved to {active_path} ({active_count:,} domains)")
        print(f"Parked domains saved to {parked_path} ({parked_count:,} domains)")

    def _save_outputs(
        self,
        results: List[DNSResult],
        results_df: pd.DataFrame,
        stats: Dict,
        output_dir: Path
    ):
        """Save all output files."""
        # Save full results CSV
        results_path = output_dir / 'dns_validation.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nFull results saved to {results_path}")

        # Save active domains list
        active_domains = [r.domain for r in results if not r.is_parked and r.query_status == "success"]
        active_path = output_dir / 'active_gambling_domains.txt'
        with open(active_path, 'w') as f:
            f.write('\n'.join(active_domains))
        print(f"Active domains saved to {active_path} ({len(active_domains)} domains)")

        # Save parked domains list
        parked_domains = [r.domain for r in results if r.is_parked]
        parked_path = output_dir / 'parked_gambling_domains.txt'
        with open(parked_path, 'w') as f:
            f.write('\n'.join(parked_domains))
        print(f"Parked domains saved to {parked_path} ({len(parked_domains)} domains)")

        # Save statistics
        stats_path = output_dir / 'dns_validation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")

    def _print_summary(self, stats: Dict):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"\nTotal domains validated: {stats['total_domains']}")
        print(f"Successful queries: {stats['successful_queries']} ({stats['success_rate_pct']:.1f}%)")
        print(f"Failed queries: timeout={stats['timeout_queries']}, "
              f"nxdomain={stats['nxdomain_queries']}, "
              f"error={stats['error_queries']}")

        print(f"\n--- Results (of successful queries) ---")
        print(f"Active gambling domains: {stats['active_domains']} ({stats['active_rate_pct']:.1f}%)")
        print(f"Parked domains: {stats['parked_domains']} ({stats['parking_rate_pct']:.1f}%)")

        if stats['parking_providers']:
            print(f"\n--- Parking Providers ---")
            for provider, count in sorted(stats['parking_providers'].items(), key=lambda x: -x[1]):
                print(f"  {provider}: {count}")

        print(f"\nAverage query time: {stats['avg_query_time']:.3f} seconds")


if __name__ == '__main__':
    # Example usage for testing
    validator = DNSValidator(
        timeout=10.0,
        max_retries=2,
        concurrency=1500
    )
    
    # Test with a few domains
    test_domains = ['google.com', 'github.com', 'example.com']
    results = validator.validate_batch_sync(test_domains)
    
    for r in results:
        print(f"{r.domain}: {r.query_status} ({r.query_time:.3f}s)")