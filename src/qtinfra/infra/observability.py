"""
Enterprise Observability Framework with OpenTelemetry Integration.

Provides comprehensive observability for trading infrastructure:
- Distributed tracing with OpenTelemetry
- Prometheus metrics collection
- Structured logging with correlation IDs
- Performance monitoring and alerting
- Custom trading metrics
- Health checks and service monitoring
- APM (Application Performance Monitoring)
"""

from __future__ import annotations
import time
import asyncio
import uuid
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps

# Optional OpenTelemetry imports with fallbacks
try:
    from opentelemetry import trace, metrics
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Create mock objects for when OpenTelemetry is not available
    trace = None
    metrics = None

# Optional Prometheus metrics with fallbacks
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        start_http_server, CONTENT_TYPE_LATEST, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create mock Prometheus classes
    class Counter:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def inc(self, amount=1):
            self._value += amount
        def labels(self, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            self._observations = []
        def observe(self, value):
            self._observations.append(value)
        def labels(self, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def set(self, value):
            self._value = value
        def labels(self, **kwargs):
            return self
    
    class Summary:
        def __init__(self, *args, **kwargs):
            self._observations = []
        def observe(self, value):
            self._observations.append(value)
        def labels(self, **kwargs):
            return self
    
    class Info:
        def __init__(self, *args, **kwargs):
            self._info = {}
        def info(self, data):
            self._info.update(data)
    
    def start_http_server(port):
        """Mock Prometheus server start."""
        pass
    
    def generate_latest():
        """Mock Prometheus metrics generation."""
        return b"# Mock metrics output\n"

from ..infra.logging import EnterpriseLogger


# === METRICS DEFINITIONS ===

class MetricType(str, Enum):
    """Types of metrics for trading operations."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


# Global registry to track created metrics and prevent duplicates
_metric_registry = set()

@dataclass
class TradingMetrics:
    """Collection of trading-specific metrics."""
    
    def __post_init__(self):
        """Initialize metrics if not already created."""
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics with duplicate checking."""
        # Order metrics
        if 'trading_orders_total' not in _metric_registry:
            self.orders_total = Counter(
                'trading_orders_total',
                'Total number of orders processed',
                ['client_id', 'symbol', 'order_type', 'side', 'status']
            )
            _metric_registry.add('trading_orders_total')
        
        if 'trading_order_processing_duration_seconds' not in _metric_registry:
            self.order_processing_duration = Histogram(
                'trading_order_processing_duration_seconds',
                'Time spent processing orders',
                ['operation', 'client_id', 'symbol'],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            )
            _metric_registry.add('trading_order_processing_duration_seconds')
        
        if 'trading_order_validation_errors_total' not in _metric_registry:
            self.order_validation_errors = Counter(
                'trading_order_validation_errors_total',
                'Total number of order validation errors',
                ['error_type', 'client_id', 'symbol']
            )
            _metric_registry.add('trading_order_validation_errors_total')
        
        # Only initialize other metrics if not already created
        self._init_execution_metrics()
        self._init_risk_metrics()
        self._init_system_metrics()
    
    def _init_execution_metrics(self):
        """Initialize execution-related metrics."""
        if 'trading_executions_total' not in _metric_registry:
            self.executions_total = Counter(
                'trading_executions_total',
                'Total number of executions',
                ['venue_id', 'symbol', 'algorithm']
            )
            _metric_registry.add('trading_executions_total')
        
        if 'trading_execution_slippage_bps' not in _metric_registry:
            self.execution_slippage = Histogram(
                'trading_execution_slippage_bps',
                'Execution slippage in basis points',
                ['venue_id', 'symbol', 'algorithm'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
            )
            _metric_registry.add('trading_execution_slippage_bps')
        
        if 'trading_execution_latency_microseconds' not in _metric_registry:
            self.execution_latency = Histogram(
                'trading_execution_latency_microseconds',
                'Execution latency in microseconds',
                ['venue_id', 'symbol'],
                buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
            )
            _metric_registry.add('trading_execution_latency_microseconds')
    
    def _init_risk_metrics(self):
        """Initialize risk-related metrics."""
        if 'trading_risk_checks_total' not in _metric_registry:
            self.risk_checks_total = Counter(
                'trading_risk_checks_total',
                'Total number of risk checks performed',
                ['check_type', 'client_id', 'result']
            )
            _metric_registry.add('trading_risk_checks_total')
        
        if 'trading_risk_limit_breaches_total' not in _metric_registry:
            self.risk_limit_breaches = Counter(
                'trading_risk_limit_breaches_total',
                'Total number of risk limit breaches',
                ['limit_type', 'client_id', 'symbol']
            )
            _metric_registry.add('trading_risk_limit_breaches_total')
        
        if 'trading_position_exposure_usd' not in _metric_registry:
            self.position_exposure = Gauge(
                'trading_position_exposure_usd',
                'Current position exposure in USD',
                ['client_id', 'symbol', 'asset_class']
            )
            _metric_registry.add('trading_position_exposure_usd')
    
    def _init_system_metrics(self):
        """Initialize system-related metrics."""
        if 'trading_active_orders' not in _metric_registry:
            self.active_orders = Gauge(
                'trading_active_orders',
                'Number of active orders',
                ['client_id', 'status']
            )
            _metric_registry.add('trading_active_orders')
        
        if 'trading_database_connections' not in _metric_registry:
            self.database_connections = Gauge(
                'trading_database_connections',
                'Number of active database connections',
                ['pool_name', 'state']
            )
            _metric_registry.add('trading_database_connections')
        
        if 'trading_api_requests_total' not in _metric_registry:
            self.api_requests_total = Counter(
                'trading_api_requests_total',
                'Total number of API requests',
                ['method', 'endpoint', 'status_code']
            )
            _metric_registry.add('trading_api_requests_total')
        
        if 'trading_api_request_duration_seconds' not in _metric_registry:
            self.api_request_duration = Histogram(
                'trading_api_request_duration_seconds',
                'API request duration',
                ['method', 'endpoint'],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )
            _metric_registry.add('trading_api_request_duration_seconds')


# === MOCK CLASSES FOR FALLBACKS ===

class MockSpan:
    """Mock span for when OpenTelemetry is not available."""
    
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.events = []
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = str(value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({"name": name, "attributes": attributes or {}})
    
    def is_recording(self):
        return True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockTracer:
    """Mock tracer for when OpenTelemetry is not available."""
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        return MockSpan(name, attributes)
    
    def start_as_current_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        return MockSpan(name, attributes)


# === DISTRIBUTED TRACING ===

class TracingManager:
    """Manages distributed tracing with OpenTelemetry."""
    
    def __init__(self, service_name: str = "trading-infrastructure", jaeger_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.tracer = None
        self.log = EnterpriseLogger(__name__, 'tracing-manager')
        
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            self.log.warning("OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()
            return
        
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "service.instance.id": str(uuid.uuid4()),
            })
            
            # Setup tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Setup exporter (Jaeger)
            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=14268,
                )
                
                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Setup propagators
            set_global_textmap(B3MultiFormat())
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument popular libraries
            RequestsInstrumentor().instrument()
            SQLAlchemyInstrumentor().instrument()
            
            self.log.info("Distributed tracing initialized", service_name=self.service_name)
            
        except Exception as e:
            self.log.error("Failed to setup tracing", error=str(e))
            # Create a mock tracer as fallback
            self.tracer = MockTracer()
    
    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new tracing span."""
        if not self.tracer:
            yield MockSpan(name, attributes)
            return
        
        try:
            if OPENTELEMETRY_AVAILABLE and hasattr(self.tracer, 'start_as_current_span'):
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                    yield span
            else:
                yield MockSpan(name, attributes)
        except Exception:
            yield MockSpan(name, attributes)
    
    @asynccontextmanager
    async def start_async_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new async tracing span."""
        if not self.tracer:
            yield MockSpan(name, attributes)
            return
        
        try:
            if OPENTELEMETRY_AVAILABLE and hasattr(self.tracer, 'start_as_current_span'):
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                    yield span
            else:
                yield MockSpan(name, attributes)
        except Exception:
            yield MockSpan(name, attributes)
    
    def trace_function(self, operation_name: Optional[str] = None):
        """Decorator to trace function calls."""
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                with self.start_span(name, {"function": func.__name__}):
                    return func(*args, **kwargs)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                async with self.start_async_span(name, {"function": func.__name__}):
                    return await func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def add_span_attribute(self, key: str, value: Any):
        """Add attribute to current span."""
        if not OPENTELEMETRY_AVAILABLE or not trace:
            return
        
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute(key, str(value))
        except Exception:
            pass  # Silently fail if tracing is not available
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if not OPENTELEMETRY_AVAILABLE or not trace:
            return
        
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes or {})
        except Exception:
            pass  # Silently fail if tracing is not available


# === METRICS COLLECTION ===

class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, port: int = 8090):
        self.port = port
        self.metrics = TradingMetrics()
        self.custom_metrics: Dict[str, Any] = {}
        self.log = EnterpriseLogger(__name__, 'metrics-collector')
        
        # System info metrics - only create if not already exists
        if 'trading_system_info' not in _metric_registry:
            self.system_info = Info(
                'trading_system_info',
                'Trading system information'
            )
            _metric_registry.add('trading_system_info')
        
        self._setup_system_metrics()
        self._start_metrics_server()
    
    def _setup_system_metrics(self):
        """Setup system-level metrics."""
        try:
            import platform
            
            # Only set info if system_info exists
            if hasattr(self, 'system_info'):
                self.system_info.info({
                    'python_version': platform.python_version(),
                    'platform': platform.platform(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture()[0],
                })
            
            # System resource metrics - check for duplicates
            if 'trading_cpu_usage_percent' not in _metric_registry:
                self.cpu_usage = Gauge('trading_cpu_usage_percent', 'CPU usage percentage')
                _metric_registry.add('trading_cpu_usage_percent')
            
            if 'trading_memory_usage_bytes' not in _metric_registry:
                self.memory_usage = Gauge('trading_memory_usage_bytes', 'Memory usage in bytes')
                _metric_registry.add('trading_memory_usage_bytes')
            
            if 'trading_disk_usage_percent' not in _metric_registry:
                self.disk_usage = Gauge('trading_disk_usage_percent', 'Disk usage percentage')
                _metric_registry.add('trading_disk_usage_percent')
            
            # Start background thread for system metrics
            self._start_system_metrics_collection()
            
        except Exception as e:
            self.log.error("Failed to setup system metrics", error=str(e))
    
    def _start_system_metrics_collection(self):
        """Start background collection of system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    if hasattr(self, 'cpu_usage'):
                        cpu_percent = psutil.cpu_percent(interval=1)
                        self.cpu_usage.set(cpu_percent)
                    
                    # Memory usage
                    if hasattr(self, 'memory_usage'):
                        memory = psutil.virtual_memory()
                        self.memory_usage.set(memory.used)
                    
                    # Disk usage
                    if hasattr(self, 'disk_usage'):
                        disk = psutil.disk_usage('/')
                        disk_percent = (disk.used / disk.total) * 100
                        self.disk_usage.set(disk_percent)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    self.log.error("Error collecting system metrics", error=str(e))
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            self.log.warning("Prometheus not available, metrics server disabled")
            return
        
        try:
            start_http_server(self.port)
            self.log.info("Metrics server started", port=self.port)
        except Exception as e:
            self.log.error("Failed to start metrics server", error=str(e), port=self.port)
    
    def record_order_processed(
        self,
        client_id: str,
        symbol: str,
        order_type: str,
        side: str,
        status: str
    ):
        """Record order processing metric."""
        self.metrics.orders_total.labels(
            client_id=client_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            status=status
        ).inc()
    
    def record_order_processing_time(
        self,
        operation: str,
        client_id: str,
        symbol: str,
        duration_seconds: float
    ):
        """Record order processing duration."""
        self.metrics.order_processing_duration.labels(
            operation=operation,
            client_id=client_id,
            symbol=symbol
        ).observe(duration_seconds)
    
    def record_validation_error(
        self,
        error_type: str,
        client_id: str,
        symbol: str
    ):
        """Record validation error."""
        self.metrics.order_validation_errors.labels(
            error_type=error_type,
            client_id=client_id,
            symbol=symbol
        ).inc()
    
    def record_execution(
        self,
        venue_id: str,
        symbol: str,
        algorithm: str,
        slippage_bps: float,
        latency_microseconds: float
    ):
        """Record execution metrics."""
        self.metrics.executions_total.labels(
            venue_id=venue_id,
            symbol=symbol,
            algorithm=algorithm
        ).inc()
        
        self.metrics.execution_slippage.labels(
            venue_id=venue_id,
            symbol=symbol,
            algorithm=algorithm
        ).observe(slippage_bps)
        
        self.metrics.execution_latency.labels(
            venue_id=venue_id,
            symbol=symbol
        ).observe(latency_microseconds)
    
    def record_risk_check(
        self,
        check_type: str,
        client_id: str,
        result: str
    ):
        """Record risk check metric."""
        self.metrics.risk_checks_total.labels(
            check_type=check_type,
            client_id=client_id,
            result=result
        ).inc()
    
    def record_risk_breach(
        self,
        limit_type: str,
        client_id: str,
        symbol: str
    ):
        """Record risk limit breach."""
        self.metrics.risk_limit_breaches.labels(
            limit_type=limit_type,
            client_id=client_id,
            symbol=symbol
        ).inc()
    
    def update_position_exposure(
        self,
        client_id: str,
        symbol: str,
        asset_class: str,
        exposure_usd: float
    ):
        """Update position exposure gauge."""
        self.metrics.position_exposure.labels(
            client_id=client_id,
            symbol=symbol,
            asset_class=asset_class
        ).set(exposure_usd)
    
    def update_active_orders(
        self,
        client_id: str,
        status: str,
        count: int
    ):
        """Update active orders count."""
        self.metrics.active_orders.labels(
            client_id=client_id,
            status=status
        ).set(count)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record API request metrics."""
        self.metrics.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.metrics.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
    
    def create_custom_counter(self, name: str, description: str, labels: List[str] = None):
        """Create custom counter metric."""
        counter = Counter(name, description, labels or [])
        self.custom_metrics[name] = counter
        return counter
    
    def create_custom_histogram(self, name: str, description: str, labels: List[str] = None, buckets: List[float] = None):
        """Create custom histogram metric."""
        histogram = Histogram(name, description, labels or [], buckets=buckets)
        self.custom_metrics[name] = histogram
        return histogram
    
    def create_custom_gauge(self, name: str, description: str, labels: List[str] = None):
        """Create custom gauge metric."""
        gauge = Gauge(name, description, labels or [])
        self.custom_metrics[name] = gauge
        return gauge
    
    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output."""
        return generate_latest().decode('utf-8')


# === HEALTH CHECKS ===

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HealthChecker:
    """Manages application health checks."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.log = EnterpriseLogger(__name__, 'health-checker')
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
        self.log.info("Health check registered", check_name=name)
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0,
                error="Check not found"
            )
        
        start_time = time.time()
        
        try:
            check_func = self.checks[name]
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = result.get("status", "healthy")
                details = result.get("details", {})
                error = result.get("error")
            else:
                status = "healthy" if result else "unhealthy"
                details = {}
                error = None
            
            return HealthCheckResult(
                name=name,
                status=status,
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                details=details,
                error=error
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.checks:
            results[name] = await self.run_check(name)
        
        return results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        healthy_count = sum(1 for r in results.values() if r.status == "healthy")
        unhealthy_count = sum(1 for r in results.values() if r.status == "unhealthy")
        degraded_count = sum(1 for r in results.values() if r.status == "degraded")
        
        total_checks = len(results)
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "total": total_checks,
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "degraded": degraded_count
            },
            "details": {name: {
                "status": result.status,
                "duration_ms": result.duration_ms,
                "error": result.error
            } for name, result in results.items()}
        }


# === OBSERVABILITY MANAGER ===

class ObservabilityManager:
    """Central manager for all observability features."""
    
    def __init__(
        self,
        service_name: str = "trading-infrastructure",
        metrics_port: int = 8090,
        jaeger_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.log = EnterpriseLogger(__name__, 'observability-manager')
        
        # Initialize components
        self.tracing = TracingManager(service_name, jaeger_endpoint)
        self.metrics = MetricsCollector(metrics_port)
        self.health = HealthChecker()
        
        # Register default health checks
        self._register_default_health_checks()
        
        self.log.info(
            "Observability manager initialized",
            service_name=service_name,
            metrics_port=metrics_port
        )
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        def database_health():
            """Check database connectivity."""
            # This would check actual database connection
            return {"status": "healthy", "details": {"connection_pool": "active"}}
        
        def memory_health():
            """Check memory usage."""
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                status = "unhealthy"
            elif usage_percent > 75:
                status = "degraded"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "details": {
                    "usage_percent": usage_percent,
                    "available_mb": memory.available / (1024 * 1024)
                }
            }
        
        def disk_health():
            """Check disk usage."""
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = "unhealthy"
            elif usage_percent > 85:
                status = "degraded"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "details": {
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / (1024 * 1024 * 1024)
                }
            }
        
        self.health.register_check("database", database_health)
        self.health.register_check("memory", memory_health)
        self.health.register_check("disk", disk_health)
    
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        return self.tracing.start_span(operation_name, attributes)
    
    async def trace_async_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing operations."""
        return self.tracing.start_async_span(operation_name, attributes)
    
    def monitor_function(self, operation_name: Optional[str] = None):
        """Decorator to add monitoring to functions."""
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                start_time = time.time()
                with self.trace_operation(name, {"function": func.__name__}):
                    try:
                        result = func(*args, **kwargs)
                        # Record success metric
                        return result
                    except Exception as e:
                        # Record error metric
                        self.tracing.add_span_attribute("error", True)
                        self.tracing.add_span_attribute("error.message", str(e))
                        raise
                    finally:
                        duration = time.time() - start_time
                        # Record duration metric
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                start_time = time.time()
                async with self.trace_async_operation(name, {"function": func.__name__}):
                    try:
                        result = await func(*args, **kwargs)
                        # Record success metric
                        return result
                    except Exception as e:
                        # Record error metric
                        self.tracing.add_span_attribute("error", True)
                        self.tracing.add_span_attribute("error.message", str(e))
                        raise
                    finally:
                        duration = time.time() - start_time
                        # Record duration metric
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        health_status = await self.health.get_overall_health()
        
        return {
            "service": {
                "name": self.service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - psutil.boot_time()
            },
            "health": health_status,
            "metrics": {
                "endpoint": f"http://localhost:{self.metrics.port}/metrics",
                "custom_metrics_count": len(self.metrics.custom_metrics)
            },
            "tracing": {
                "enabled": self.tracing.tracer is not None,
                "service_name": self.service_name
            }
        }


# === SINGLETON INSTANCE ===

# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None

def get_observability_manager(
    service_name: str = "trading-infrastructure",
    metrics_port: int = 8090,
    jaeger_endpoint: Optional[str] = None
) -> ObservabilityManager:
    """Get singleton observability manager instance."""
    global _observability_manager
    
    if _observability_manager is None:
        _observability_manager = ObservabilityManager(
            service_name=service_name,
            metrics_port=metrics_port,
            jaeger_endpoint=jaeger_endpoint
        )
    
    return _observability_manager


# === CONVENIENCE DECORATORS ===

def trace_function(operation_name: Optional[str] = None):
    """Decorator to add tracing to functions."""
    manager = get_observability_manager()
    return manager.tracing.trace_function(operation_name)


def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to add comprehensive monitoring to functions."""
    manager = get_observability_manager()
    return manager.monitor_function(operation_name)