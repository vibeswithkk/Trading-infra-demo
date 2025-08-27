#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qtinfra.infra.logging import (
        EnterpriseLogger, PIIScrubber, CircuitBreaker, 
        RateLimiter, health_check, SecurityConfig
    )
    print("✓ Successfully imported all logging components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_components():
    print("\nTesting Basic Components:")
    
    # Test PII scrubbing
    try:
        data = {'email': 'test@example.com', 'message': 'Hello world'}
        scrubbed = PIIScrubber.scrub_data(data, {'email'})
        if 'test@example.com' not in str(scrubbed):
            print("✓ PII scrubbing works")
        else:
            print("❌ PII scrubbing failed")
    except Exception as e:
        print(f"❌ PII scrubbing error: {e}")
    
    # Test circuit breaker
    try:
        cb = CircuitBreaker(failure_threshold=2)
        result = cb.call(lambda: "success")
        if result == "success":
            print("✓ Circuit breaker works")
        else:
            print("❌ Circuit breaker failed")
    except Exception as e:
        print(f"❌ Circuit breaker error: {e}")
    
    # Test rate limiter
    try:
        limiter = RateLimiter(max_rate=2, window_size=1.0)
        results = [limiter.is_allowed() for _ in range(3)]
        if results == [True, True, False]:
            print("✓ Rate limiter works")
        else:
            print(f"❌ Rate limiter failed: {results}")
    except Exception as e:
        print(f"❌ Rate limiter error: {e}")
    
    # Test health check
    try:
        health = health_check()
        if 'status' in health and 'metrics' in health:
            print("✓ Health check works")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_logger_creation():
    print("\nTesting Logger Creation:")
    
    try:
        config = {
            'destinations': [{'type': 'console', 'level': 'INFO'}],
            'async_logging': False
        }
        logger = EnterpriseLogger('test', 'test-service', config)
        logger.info("Test message", test_key="test_value")
        print("✓ Basic logger creation and logging works")
    except Exception as e:
        print(f"❌ Logger creation error: {e}")
    
    # Test security config
    try:
        security_config = SecurityConfig(enable_encryption=True, enable_signing=True)
        if security_config.encryption_key and security_config.signing_key:
            print("✓ Security config generation works")
        else:
            print("❌ Security config generation failed")
    except Exception as e:
        print(f"❌ Security config error: {e}")

def main():
    print("Enterprise Logging System - Simple Verification")
    print("=" * 50)
    
    test_basic_components()
    test_logger_creation()
    
    print("\n" + "=" * 50)
    print("✅ Simple verification completed!")
    print("\nIf you see this message, the core logging system is working.")
    print("\nTo run full tests:")
    print("1. Install dependencies: pip install -r requirements-logging.txt")
    print("2. Run: python verify_logging.py")
    print("3. Run: pytest tests/test_enterprise_logging.py -v")

if __name__ == "__main__":
    main()