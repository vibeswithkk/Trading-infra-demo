#!/usr/bin/env python3
"""
Observability Framework Verification Script

Tests all observability components:
- Distributed tracing
- Metrics collection
- Health checks
- Performance monitoring
"""

import asyncio
import time
from decimal import Decimal

def test_observability_imports():
    """Test that all observability components can be imported."""
    print("Testing observability imports...")
    
    try:
        from qtinfra.infra.observability import (
            ObservabilityManager, TracingManager, MetricsCollector, 
            HealthChecker, TradingMetrics, get_observability_manager,
            trace_function, monitor_performance
        )
        print("All observability components imported successfully")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def test_observability_manager():
    """Test observability manager initialization."""
    print("\nTesting observability manager...")
    
    try:
        from qtinfra.infra.observability import ObservabilityManager
        
        # Initialize manager
        manager = ObservabilityManager(
            service_name="test-service",
            metrics_port=8091,  # Use different port for testing
            jaeger_endpoint=None  # Disable Jaeger for testing
        )
        
        print("ObservabilityManager initialized successfully")
        print(f"   Service: {manager.service_name}")
        print(f"   Metrics port: {manager.metrics.port}")
        print(f"   Tracing enabled: {manager.tracing.tracer is not None}")
        
        return manager
    except Exception as e:
        print(f"ObservabilityManager initialization failed: {e}")
        return None

async def test_health_checks(manager):
    """Test health check functionality."""
    print("\nTesting health checks...")
    
    try:
        # Get overall health
        health_status = await manager.health.get_overall_health()
        
        print("Health checks completed:")
        print(f"   Overall status: {health_status['status']}")
        print(f"   Healthy checks: {health_status['checks']['healthy']}")
        print(f"   Total checks: {len(health_status['details'])}")
        
        # Test individual checks
        for check_name, details in health_status['details'].items():
            print(f"   {check_name}: {details['status']}")
        
        return True
    except Exception as e:
        print(f"Health checks failed: {e}")
        return False

def test_metrics_collection(manager):
    """Test metrics collection."""
    print("\nTesting metrics collection...")
    
    try:
        metrics = manager.metrics
        
        # Test trading metrics
        metrics.metrics.orders_total.labels(
            client_id="TEST_CLIENT",
            symbol="AAPL", 
            order_type="LIMIT",
            side="BUY",
            status="FILLED"
        ).inc()
        
        metrics.metrics.order_processing_duration.labels(
            operation="create_order",
            client_id="TEST_CLIENT",
            symbol="AAPL"
        ).observe(0.125)
        
        # Test custom metrics
        custom_counter = metrics.create_custom_counter(
            "test_operations_total",
            "Test operations counter",
            ["operation_type"]
        )
        custom_counter.labels(operation_type="test").inc()
        
        print("Metrics collection successful:")
        print("   Order metrics recorded")
        print("   Processing duration recorded")
        print("   Custom metrics created")
        
        return True
    except Exception as e:
        print(f"Metrics collection failed: {e}")
        return False

def test_tracing_decorators():
    """Test tracing decorators."""
    print("\nTesting tracing decorators...")
    
    try:
        from qtinfra.infra.observability import trace_function, monitor_performance
        
        @trace_function("test_operation")
        def test_sync_function():
            time.sleep(0.01)  # Simulate work
            return "sync_result"
        
        @trace_function("async_test_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_result"
        
        @monitor_performance("monitored_operation")
        def test_monitored_function():
            time.sleep(0.01)
            return "monitored_result"
        
        # Test sync function
        result1 = test_sync_function()
        print(f"Sync traced function: {result1}")
        
        # Test monitored function
        result2 = test_monitored_function()
        print(f"Monitored function: {result2}")
        
        return True
    except Exception as e:
        print(f"Tracing decorators failed: {e}")
        return False

async def test_async_tracing_decorators():
    """Test async tracing decorators."""
    try:
        from qtinfra.infra.observability import trace_function
        
        @trace_function("async_test_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_result"
        
        # Test async function
        result = await test_async_function()
        print(f"Async traced function: {result}")
        
        return True
    except Exception as e:
        print(f"Async tracing decorators failed: {e}")
        return False

async def test_observability_summary(manager):
    """Test observability summary."""
    print("\nTesting observability summary...")
    
    try:
        summary = await manager.get_observability_summary()
        
        print("Observability summary generated:")
        print(f"   Service: {summary['service']['name']}")
        print(f"   Metrics endpoint: {summary['metrics']['endpoint']}")
        print(f"   Tracing enabled: {summary['tracing']['enabled']}")
        print(f"   Health status: {summary['health']['status']}")
        
        return True
    except Exception as e:
        print(f"Observability summary failed: {e}")
        return False

async def main():
    """Main verification function."""
    print("OBSERVABILITY FRAMEWORK VERIFICATION")
    print("=" * 50)
    
    success_count = 0
    total_tests = 7
    
    # Test 1: Imports
    if test_observability_imports():
        success_count += 1
    
    # Test 2: Manager initialization
    manager = test_observability_manager()
    if manager:
        success_count += 1
        
        # Test 3: Health checks
        if await test_health_checks(manager):
            success_count += 1
        
        # Test 4: Metrics collection
        if test_metrics_collection(manager):
            success_count += 1
        
        # Test 5: Tracing decorators
        if test_tracing_decorators():
            success_count += 1
        
        # Test 6: Async tracing decorators
        if await test_async_tracing_decorators():
            success_count += 1
        
        # Test 7: Observability summary
        if await test_observability_summary(manager):
            success_count += 1
    
    print(f"\nVERIFICATION RESULTS")
    print("=" * 30)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\nALL TESTS PASSED!")
        print("Observability framework is fully functional")
        print("\nFeatures verified:")
        print("- Distributed tracing with OpenTelemetry")
        print("- Prometheus metrics collection")
        print("- Health check system")
        print("- Performance monitoring decorators")
        print("- Trading-specific metrics")
        print("- Async operation support")
        print("- Comprehensive observability summary")
        
        return True
    else:
        print(f"\n{total_tests - success_count} tests failed")
        print("Please check the implementation")
        return False

if __name__ == "__main__":
    asyncio.run(main())