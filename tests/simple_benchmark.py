import asyncio
import json
import logging
import os
import sys
import time
import timeit
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qtinfra.infra.logging import (
    EnterpriseLogger, AsyncLogHandler, SecurityConfig,
    setup_enterprise_logging, LogMetrics
)

class SimpleBenchmark:
    def __init__(self):
        self.results = {}
    
    def run_quick_benchmark(self) -> Dict[str, Any]:
        print("Running Quick Enterprise Logging Benchmark")
        print("=" * 45)
        
        # Test sync throughput
        print("\n1. Synchronous Logging Throughput")
        print("-" * 30)
        
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': False
        }
        logger = EnterpriseLogger('benchmark_sync', 'benchmark-service', config)
        
        def log_messages(count: int):
            for i in range(count):
                logger.info(f'Benchmark message {i}', iteration=i)
        
        # Quick benchmark
        message_count = 1000
        start_time = time.time()
        log_messages(message_count)
        duration = time.time() - start_time
        throughput = message_count / duration
        
        print(f"{message_count:,} messages: {throughput:.0f} logs/sec ({duration:.3f}s)")
        
        self.results['sync_throughput'] = {
            'messages': message_count,
            'duration': duration,
            'throughput': throughput
        }
        
        # Test async throughput
        print("\n2. Asynchronous Logging Test")
        print("-" * 30)
        
        async_config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': True
        }
        async_logger = EnterpriseLogger('benchmark_async', 'benchmark-service', async_config)
        
        start_time = time.time()
        for i in range(500):  # Smaller count for async test
            async_logger.info(f'Async message {i}', iteration=i)
        time.sleep(0.1)  # Allow async processing
        duration = time.time() - start_time
        async_throughput = 500 / duration
        
        print(f"500 async messages: {async_throughput:.0f} logs/sec ({duration:.3f}s)")
        
        self.results['async_throughput'] = {
            'messages': 500,
            'duration': duration,
            'throughput': async_throughput
        }
        
        return self.results
    
    def print_summary(self):
        print("\n" + "=" * 45)
        print("BENCHMARK SUMMARY")
        print("=" * 45)
        
        if 'sync_throughput' in self.results:
            sync_throughput = self.results['sync_throughput']['throughput']
            print(f"Sync Throughput: {sync_throughput:.0f} logs/sec")
        
        if 'async_throughput' in self.results:
            async_throughput = self.results['async_throughput']['throughput']
            print(f"Async Throughput: {async_throughput:.0f} logs/sec")
        
        print("\nRecommendations:")
        print("- Use async logging for high-throughput scenarios")
        print("- Monitor queue sizes to prevent message loss")
        print("- Consider log sampling for very high-volume applications")

def main():
    try:
        benchmark = SimpleBenchmark()
        results = benchmark.run_quick_benchmark()
        benchmark.print_summary()
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {}

if __name__ == '__main__':
    main()