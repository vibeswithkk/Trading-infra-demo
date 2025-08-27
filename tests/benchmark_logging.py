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

class LoggingBenchmark:
    def __init__(self):
        self.results = {}
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        print("Running Enterprise Logging Performance Benchmarks")
        print("=" * 50)
        
        self.results['sync_throughput'] = self.benchmark_sync_throughput()
        self.results['async_throughput'] = self.benchmark_async_throughput()
        self.results['queue_overflow'] = self.benchmark_queue_overflow()
        self.results['encryption_overhead'] = self.benchmark_encryption_overhead()
        self.results['concurrent_logging'] = self.benchmark_concurrent_logging()
        self.results['memory_usage'] = self.benchmark_memory_usage()
        
        self.print_summary()
        return self.results
    
    def benchmark_sync_throughput(self) -> Dict[str, float]:
        print("\n1. Synchronous Logging Throughput")
        print("-" * 30)
        
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': False
        }
        logger = EnterpriseLogger('benchmark_sync', 'benchmark-service', config)
        
        def log_messages(count: int):
            for i in range(count):
                logger.info(f'Benchmark message {i}', 
                           iteration=i, 
                           data={'key': 'value', 'number': i})
        
        # Warmup
        log_messages(100)
        
        # Benchmark
        message_counts = [1000, 5000, 10000]
        results = {}
        
        for count in message_counts:
            duration = timeit.timeit(lambda: log_messages(count), number=1)
            throughput = count / duration
            results[f'{count}_messages'] = {
                'duration': duration,
                'throughput': throughput
            }
            print(f"{count:,} messages: {throughput:.0f} logs/sec ({duration:.3f}s)")
        
        return results
    
    async def benchmark_async_throughput(self) -> Dict[str, float]:
        print("\n2. Asynchronous Logging Throughput")
        print("-" * 30)
        
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': True
        }
        logger = EnterpriseLogger('benchmark_async', 'benchmark-service', config)
        
        async def log_messages_async(count: int):
            tasks = []
            for i in range(count):
                task = asyncio.create_task(
                    self._async_log(logger, f'Async message {i}', iteration=i)
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
        
        # Warmup
        await log_messages_async(100)
        
        # Benchmark
        message_counts = [1000, 5000, 10000, 20000]
        results = {}
        
        for count in message_counts:
            start_time = time.time()
            await log_messages_async(count)
            duration = time.time() - start_time
            throughput = count / duration
            results[f'{count}_messages'] = {
                'duration': duration,
                'throughput': throughput
            }
            print(f"{count:,} messages: {throughput:.0f} logs/sec ({duration:.3f}s)")
        
        return results
    
    async def _async_log(self, logger, message, **kwargs):
        await asyncio.sleep(0)  # Yield control
        logger.info(message, **kwargs)
    
    def benchmark_queue_overflow(self) -> Dict[str, Any]:
        print("\n3. Queue Overflow Behavior")
        print("-" * 30)
        
        # Create handler with small queue
        mock_handler = logging.StreamHandler()
        async_handler = AsyncLogHandler(mock_handler, queue_size=100)
        
        logger = logging.getLogger('overflow_test')
        logger.addHandler(async_handler)
        logger.setLevel(logging.INFO)
        
        # Flood the queue
        dropped_count = 0
        total_messages = 1000
        
        start_time = time.time()
        for i in range(total_messages):
            record = logging.LogRecord(
                'overflow_test', logging.INFO, 'test.py', 1, 
                f'Overflow test {i}', (), None
            )
            
            initial_size = async_handler.queue.qsize()
            async_handler.emit(record)
            
            if async_handler.queue.qsize() == initial_size:
                dropped_count += 1
        
        duration = time.time() - start_time
        
        # Wait for queue to process
        time.sleep(0.5)
        
        results = {
            'total_messages': total_messages,
            'dropped_messages': dropped_count,
            'processed_messages': total_messages - dropped_count,
            'drop_rate': dropped_count / total_messages,
            'duration': duration
        }
        
        print(f"Total messages: {total_messages:,}")
        print(f"Dropped messages: {dropped_count:,} ({results['drop_rate']:.1%})")
        print(f"Processed messages: {results['processed_messages']:,}")
        
        async_handler.close()
        return results
    
    def benchmark_encryption_overhead(self) -> Dict[str, float]:
        print("\n4. Encryption Overhead")
        print("-" * 30)
        
        # Without encryption
        config_plain = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': False,
            'security': {'enable_encryption': False}
        }
        logger_plain = EnterpriseLogger('benchmark_plain', 'benchmark-service', config_plain)
        
        # With encryption
        config_encrypted = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': False,
            'security': {'enable_encryption': True, 'enable_signing': True}
        }
        logger_encrypted = EnterpriseLogger('benchmark_encrypted', 'benchmark-service', config_encrypted)
        
        def log_batch(logger, count):
            for i in range(count):
                logger.info(f'Encryption test {i}', 
                           data={'sensitive': 'password123', 'public': 'data'})
        
        message_count = 1000
        
        # Benchmark plain logging
        plain_duration = timeit.timeit(lambda: log_batch(logger_plain, message_count), number=1)
        plain_throughput = message_count / plain_duration
        
        # Benchmark encrypted logging
        encrypted_duration = timeit.timeit(lambda: log_batch(logger_encrypted, message_count), number=1)
        encrypted_throughput = message_count / encrypted_duration
        
        overhead = ((encrypted_duration - plain_duration) / plain_duration) * 100
        
        results = {
            'plain_throughput': plain_throughput,
            'encrypted_throughput': encrypted_throughput,
            'overhead_percentage': overhead,
            'plain_duration': plain_duration,
            'encrypted_duration': encrypted_duration
        }
        
        print(f"Plain logging: {plain_throughput:.0f} logs/sec")
        print(f"Encrypted logging: {encrypted_throughput:.0f} logs/sec")
        print(f"Encryption overhead: {overhead:.1f}%")
        
        return results
    
    def benchmark_concurrent_logging(self) -> Dict[str, Any]:
        print("\n5. Concurrent Logging Performance")
        print("-" * 30)
        
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': True
        }
        logger = EnterpriseLogger('benchmark_concurrent', 'benchmark-service', config)
        
        def worker_task(worker_id: int, message_count: int):
            start_time = time.time()
            for i in range(message_count):
                logger.info(f'Worker {worker_id} message {i}', 
                           worker_id=worker_id, message_id=i)
            return time.time() - start_time
        
        thread_counts = [1, 2, 4, 8, 16]
        messages_per_thread = 1000
        results = {}
        
        for thread_count in thread_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                for worker_id in range(thread_count):
                    future = executor.submit(worker_task, worker_id, messages_per_thread)
                    futures.append(future)
                
                worker_durations = []
                for future in as_completed(futures):
                    worker_durations.append(future.result())
            
            total_duration = time.time() - start_time
            total_messages = thread_count * messages_per_thread
            overall_throughput = total_messages / total_duration
            
            results[f'{thread_count}_threads'] = {
                'total_messages': total_messages,
                'total_duration': total_duration,
                'overall_throughput': overall_throughput,
                'avg_worker_duration': mean(worker_durations),
                'worker_durations': worker_durations
            }
            
            print(f\"{thread_count} threads: {overall_throughput:.0f} logs/sec \"
                  f\"({total_messages:,} messages in {total_duration:.3f}s)\")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        print(\"\n6. Memory Usage Analysis\")
        print(\"-\" * 30)
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            print(\"psutil not available, skipping memory benchmark\")
            return {'error': 'psutil not available'}
        
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': True
        }
        logger = EnterpriseLogger('benchmark_memory', 'benchmark-service', config)
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Log a large number of messages
        message_count = 50000
        large_data = {'data': 'x' * 1000}  # 1KB per message
        
        start_time = time.time()
        for i in range(message_count):
            logger.info(f\"Memory test {i}\", iteration=i, **large_data)
        
        duration = time.time() - start_time
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        results = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'message_count': message_count,
            'duration': duration,
            'throughput': message_count / duration,
            'memory_per_message_kb': (memory_growth * 1024) / message_count if memory_growth > 0 else 0
        }
        
        print(f\"Initial memory: {initial_memory:.1f} MB\")
        print(f\"Final memory: {final_memory:.1f} MB\")
        print(f\"Memory growth: {memory_growth:.1f} MB\")
        print(f\"Memory per message: {results['memory_per_message_kb']:.2f} KB\")
        print(f\"Throughput: {results['throughput']:.0f} logs/sec\")
        
        return results
    
    def print_summary(self):
        print(\"\n\" + \"=\" * 50)
        print(\"BENCHMARK SUMMARY\")
        print(\"=\" * 50)
        
        if 'sync_throughput' in self.results:
            sync_best = max([r['throughput'] for r in self.results['sync_throughput'].values()])
            print(f\"Best Sync Throughput: {sync_best:.0f} logs/sec\")
        
        if 'async_throughput' in self.results:
            async_best = max([r['throughput'] for r in self.results['async_throughput'].values()])
            print(f\"Best Async Throughput: {async_best:.0f} logs/sec\")
        
        if 'queue_overflow' in self.results:
            drop_rate = self.results['queue_overflow']['drop_rate']
            print(f\"Queue Drop Rate: {drop_rate:.1%}\")
        
        if 'encryption_overhead' in self.results:
            overhead = self.results['encryption_overhead']['overhead_percentage']
            print(f\"Encryption Overhead: {overhead:.1f}%\")
        
        if 'memory_usage' in self.results and 'error' not in self.results['memory_usage']:
            memory_per_msg = self.results['memory_usage']['memory_per_message_kb']
            print(f\"Memory per Message: {memory_per_msg:.2f} KB\")
        
        print(\"\nRecommendations:\")
        print(\"- Use async logging for high-throughput scenarios\")
        print(\"- Monitor queue sizes to prevent message loss\")
        print(\"- Consider encryption overhead in performance planning\")
        print(\"- Implement log sampling for very high-volume applications\")

    def save_results(self, filename: str = 'benchmark_results.json'):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f\"\nResults saved to {filename}\")

async def main():
    benchmark = LoggingBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.save_results()
    
    return results

if __name__ == '__main__':
    asyncio.run(main())