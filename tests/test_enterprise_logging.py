import asyncio
import json
import logging
import os
import pytest
import sys
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qtinfra.infra.logging import (
    PIIScrubber, CircuitBreaker, RateLimiter, LogSampler, AsyncLogHandler,
    EnterpriseLogger, SecurityConfig, SecurityClassification,
    setup_enterprise_logging, health_check, LogMetrics
)

@pytest.fixture
def security_config():
    return SecurityConfig(
        enable_scrubbing=True,
        enable_encryption=True,
        enable_signing=True,
        pii_fields={'password', 'email', 'phone'}
    )

@pytest.fixture
def logger_config():
    return {
        'service_name': 'test-service',
        'version': '1.0.0',
        'environment': 'test',
        'destinations': [{'type': 'console', 'level': 'DEBUG'}],
        'async_logging': False
    }

class TestPIIScrubber:
    def test_scrub_email(self):
        data = {'message': 'User john.doe@example.com logged in'}
        result = PIIScrubber.scrub_data(data, {'email'})
        assert 'john.doe@example.com' not in result['message']
        assert 'jo***le.com' in result['message']
    
    def test_scrub_phone(self):
        data = {'message': 'Call me at +1-555-123-4567'}
        result = PIIScrubber.scrub_data(data, {'phone'})
        assert '+1-555-123-4567' not in result['message']
        assert '+1***567' in result['message']
    
    def test_scrub_nested_dict(self):
        data = {
            'user': {
                'email': 'test@example.com',
                'name': 'John Doe'
            }
        }
        result = PIIScrubber.scrub_data(data, {'email'})
        assert result['user']['email'] == 'te***om'
        assert result['user']['name'] == 'John Doe'
    
    def test_scrub_list(self):
        data = ['user@example.com', 'normal text']
        result = PIIScrubber.scrub_data(data, {'email'})
        assert 'us***om' in result[0]
        assert result[1] == 'normal text'
    
    def test_mask_value(self):
        assert PIIScrubber._mask_value('password123') == 'pa***23'
        assert PIIScrubber._mask_value('abc') == '***'

class TestCircuitBreaker:
    def test_closed_state_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        mock_func = Mock(return_value='success')
        
        result = cb.call(mock_func)
        assert result == 'success'
        assert cb.state == 'closed'
    
    def test_open_state_after_failures(self):
        cb = CircuitBreaker(failure_threshold=2)
        failing_func = Mock(side_effect=Exception('error'))
        
        with pytest.raises(Exception):
            cb.call(failing_func)
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == 'open'
        
        with pytest.raises(Exception, match="Circuit breaker is open"):
            cb.call(Mock())
    
    def test_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        failing_func = Mock(side_effect=Exception('error'))
        
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == 'open'
        
        time.sleep(0.2)
        success_func = Mock(return_value='success')
        result = cb.call(success_func)
        
        assert result == 'success'
        assert cb.state == 'closed'

class TestRateLimiter:
    def test_rate_limiting(self):
        limiter = RateLimiter(max_rate=2, window_size=1.0)
        
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
    
    def test_rate_limiting_window_reset(self):
        limiter = RateLimiter(max_rate=1, window_size=0.1)
        
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
        
        time.sleep(0.2)
        assert limiter.is_allowed() is True
    
    def test_concurrent_rate_limiting(self):
        limiter = RateLimiter(max_rate=5, window_size=1.0)
        
        def check_rate():
            return limiter.is_allowed()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_rate) for _ in range(10)]
            results = [f.result() for f in futures]
        
        allowed_count = sum(results)
        assert allowed_count <= 5

class TestLogSampler:
    def test_error_always_logged(self):
        sampler = LogSampler(sample_rate=0.1)
        
        for _ in range(10):
            assert sampler.should_log('error') is True
            assert sampler.should_log('critical') is True
    
    def test_sampling_rate(self):
        sampler = LogSampler(sample_rate=0.5)
        
        results = [sampler.should_log('info') for _ in range(100)]
        logged_count = sum(results)
        
        assert 40 <= logged_count <= 60
    
    def test_no_sampling(self):
        sampler = LogSampler(sample_rate=1.0)
        
        for _ in range(10):
            assert sampler.should_log('info') is True

class TestAsyncLogHandler:
    @pytest.fixture
    def mock_handler(self):
        return Mock(spec=logging.Handler)
    
    def test_async_emit(self, mock_handler):
        async_handler = AsyncLogHandler(mock_handler, queue_size=10)
        record = logging.LogRecord(
            'test', logging.INFO, 'test.py', 1, 'test message', (), None
        )
        
        async_handler.emit(record)
        time.sleep(0.1)
        
        mock_handler.emit.assert_called_once_with(record)
    
    def test_queue_overflow(self, mock_handler):
        async_handler = AsyncLogHandler(mock_handler, queue_size=2)
        
        for i in range(5):
            record = logging.LogRecord(
                'test', logging.INFO, 'test.py', 1, f'message {i}', (), None
            )
            async_handler.emit(record)
        
        time.sleep(0.1)
        assert mock_handler.emit.call_count <= 2
    
    def test_shutdown(self, mock_handler):
        async_handler = AsyncLogHandler(mock_handler)
        assert async_handler.worker.is_alive()
        
        async_handler.close()
        time.sleep(0.1)
        
        assert not async_handler.worker.is_alive()
        mock_handler.close.assert_called_once()

class TestEnterpriseLogger:
    def test_logger_creation(self, logger_config):
        logger = EnterpriseLogger('test.logger', 'test-service', logger_config)
        assert logger.service_name == 'test-service'
        assert logger.name == 'test.logger'
    
    def test_logging_methods(self, logger_config):
        with patch('qtinfra.infra.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.handlers = []
            
            logger = EnterpriseLogger('test', 'test-service', logger_config)
            
            logger.info('test message', key='value')
            logger.error('error message', error_code=500)
            logger.debug('debug message')
            
            assert mock_logger.log.call_count >= 3
    
    def test_audit_logging(self, logger_config):
        with patch('qtinfra.infra.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.handlers = []
            
            logger = EnterpriseLogger('test', 'test-service', logger_config)
            logger.audit('CREATE', 'user', 'admin123', ip='192.168.1.1')
            
            mock_logger.log.assert_called()
    
    def test_span_context(self, logger_config):
        logger = EnterpriseLogger('test', 'test-service', logger_config)
        
        with logger.span('test_operation', tag1='value1') as span:
            assert 'trace_id' in span
            assert 'span_id' in span
            assert span['operation_name'] == 'test_operation'
            assert span['tags']['tag1'] == 'value1'
    
    def test_span_error_handling(self, logger_config):
        logger = EnterpriseLogger('test', 'test-service', logger_config)
        
        with pytest.raises(ValueError):
            with logger.span('failing_operation') as span:
                raise ValueError('test error')
    
    def test_metrics_collection(self):
        metrics = EnterpriseLogger.get_metrics()
        assert 'total_logs' in metrics
        assert 'error_count' in metrics
        assert 'circuit_breaker_state' in metrics
    
    def test_sampling_configuration(self):
        EnterpriseLogger.configure_sampling(0.5)
        assert EnterpriseLogger._sampler.sample_rate == 0.5
    
    def test_rate_limiting_configuration(self):
        EnterpriseLogger.configure_rate_limiting(100, 2.0)
        assert EnterpriseLogger._rate_limiter.max_rate == 100
        assert EnterpriseLogger._rate_limiter.window_size == 2.0

class TestHealthCheck:
    def test_health_check_healthy(self):
        result = health_check()
        assert 'status' in result
        assert 'metrics' in result
        assert 'timestamp' in result
    
    def test_health_check_structure(self):
        result = health_check()
        assert result['status'] in ['healthy', 'degraded']
        assert isinstance(result['metrics'], dict)
        assert 'total_logs' in result['metrics']

class TestSecurityConfig:
    def test_auto_key_generation(self):
        config = SecurityConfig(enable_encryption=True, enable_signing=True)
        assert config.encryption_key is not None
        assert config.signing_key is not None
    
    def test_key_bytes_conversion(self):
        config = SecurityConfig(enable_encryption=True)
        key_bytes = config.get_encryption_key_bytes()
        assert isinstance(key_bytes, bytes)
        assert len(key_bytes) == 32

@pytest.mark.asyncio
class TestPerformance:
    async def test_high_throughput_logging(self, logger_config):
        logger_config['async_logging'] = True
        logger = EnterpriseLogger('perf_test', 'test-service', logger_config)
        
        start_time = time.time()
        
        tasks = []
        for i in range(1000):
            task = asyncio.create_task(
                self._log_message(logger, f'message {i}')
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        throughput = 1000 / duration
        
        assert throughput > 100
    
    async def _log_message(self, logger, message):
        logger.info(message, index=1, data={'key': 'value'})
    
    def test_memory_usage_under_load(self, logger_config):
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        logger = EnterpriseLogger('memory_test', 'test-service', logger_config)
        
        for i in range(10000):
            logger.info(f'Memory test message {i}', data={'large': 'x' * 100})
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 100 * 1024 * 1024

if __name__ == '__main__':
    pytest.main([__file__, '-v'])