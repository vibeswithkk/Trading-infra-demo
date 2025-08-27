from __future__ import annotations
import asyncio
import base64
import concurrent.futures
import functools
import gzip
import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import socket
import ssl
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
from weakref import WeakSet

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    AESGCM = None

try:
    from pydantic import BaseModel, Field, ValidationError, validator
except ImportError:
    BaseModel = object
    ValidationError = Exception
    Field = lambda **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = lambda *args, **kwargs: None

try:
    import kafka
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

trace_context: ContextVar[Dict[str, Any]] = ContextVar('trace_context', default={})
user_context: ContextVar[Dict[str, Any]] = ContextVar('user_context', default={})
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})

if PROMETHEUS_AVAILABLE:
    log_counter = Counter("enterprise_logs_total", "Total logs emitted", ["service", "level", "environment"])
    error_gauge = Gauge("enterprise_logs_errors_total", "Total error logs")
    log_latency = Histogram("enterprise_log_processing_seconds", "Log processing latency")
    queue_size_gauge = Gauge("enterprise_log_queue_size", "Current log queue size")
    dropped_logs_counter = Counter("enterprise_logs_dropped_total", "Total dropped logs", ["reason"])
else:
    log_counter = error_gauge = log_latency = queue_size_gauge = dropped_logs_counter = None

class LogSchema(BaseModel if BaseModel != object else object):
    timestamp: str = Field(alias="@timestamp")
    version: str = Field(default="1", alias="@version")
    level: str
    message: str
    logger: str
    service: Dict[str, Any]
    host: Dict[str, Any]
    process: Dict[str, Any]
    labels: Dict[str, Any]
    
    class Config:
        allow_population_by_field_name = True
        
    @validator('level')
    def validate_level(cls, v):
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if v.lower() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}')
        return v.lower()
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f'Invalid timestamp format: {v}')
        return v

class LogLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

class SecurityClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class OutputDestination(Enum):
    CONSOLE = "console"
    FILE = "file"
    SYSLOG = "syslog"
    HTTP = "http"
    KAFKA = "kafka"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"

@dataclass
class LogMetrics:
    total_logs: int = 0
    error_count: int = 0
    warning_count: int = 0
    bytes_written: int = 0
    dropped_logs: int = 0
    last_error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    
    def increment(self, level: str, size: int):
        self.total_logs += 1
        self.bytes_written += size
        if level == 'error':
            self.error_count += 1
        elif level == 'warning':
            self.warning_count += 1

@dataclass
class SecurityConfig:
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    enable_signing: bool = False
    signing_key: Optional[str] = None
    pii_fields: Set[str] = field(default_factory=lambda: {'password', 'ssn', 'credit_card', 'email', 'phone'})
    enable_scrubbing: bool = True
    classification: SecurityClassification = SecurityClassification.INTERNAL
    
    def __post_init__(self):
        if self.enable_encryption and not self.encryption_key:
            self.encryption_key = base64.b64encode(os.urandom(32)).decode('utf-8')
        if self.enable_signing and not self.signing_key:
            self.signing_key = base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def get_encryption_key_bytes(self) -> bytes:
        if not self.encryption_key:
            raise ValueError("Encryption key not set")
        return base64.b64decode(self.encryption_key.encode('utf-8'))
    
    def get_signing_key_bytes(self) -> bytes:
        if not self.signing_key:
            raise ValueError("Signing key not set")
        return base64.b64decode(self.signing_key.encode('utf-8'))

class PIIScrubber:
    PII_PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b'),
        'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        'uuid': re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'),
    }
    
    @classmethod
    def scrub_data(cls, data: Any, fields: Set[str]) -> Any:
        if isinstance(data, dict):
            return {k: cls._mask_value(v) if k.lower() in fields else cls.scrub_data(v, fields) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.scrub_data(item, fields) for item in data]
        elif isinstance(data, str):
            return cls._scrub_text(data)
        return data
    
    @classmethod
    def _mask_value(cls, value: Any) -> str:
        if isinstance(value, str) and len(value) > 4:
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
        return '***'
    
    @classmethod
    def _scrub_text(cls, text: str) -> str:
        for pattern_name, pattern in cls.PII_PATTERNS.items():
            text = pattern.sub(lambda m: cls._mask_value(m.group()), text)
        return text

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                raise e

class RateLimiter:
    def __init__(self, max_rate: int = 1000, window_size: float = 1.0):
        self.max_rate = max_rate
        self.window_size = window_size
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        with self._lock:
            now = time.time()
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            if len(self.requests) < self.max_rate:
                self.requests.append(now)
                return True
            return False

class LogSampler:
    def __init__(self, sample_rate: float = 1.0):
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self._counter = 0
        self._lock = threading.Lock()
    
    def should_log(self, level: str) -> bool:
        if level in ('error', 'critical'):
            return True
        
        with self._lock:
            self._counter += 1
            return (self._counter % int(1 / self.sample_rate)) == 0 if self.sample_rate < 1.0 else True

class KafkaHandler(logging.Handler):
    def __init__(self, topic: str, bootstrap_servers: List[str], **kafka_config):
        super().__init__()
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required for KafkaHandler")
        self.topic = topic
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.encode('utf-8'),
            **kafka_config
        )
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.producer.send(self.topic, value=log_entry)
        except Exception:
            self.handleError(record)
    
    def close(self):
        if hasattr(self, 'producer'):
            self.producer.flush()
            self.producer.close()
        super().close()

class RedisHandler(logging.Handler):
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 channel: str = 'logs', **redis_config):
        super().__init__()
        if not REDIS_AVAILABLE:
            raise ImportError("redis is required for RedisHandler")
        self.channel = channel
        self.redis_client = redis.Redis(host=host, port=port, db=db, **redis_config)
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.redis_client.publish(self.channel, log_entry)
        except Exception:
            self.handleError(record)
    
    def close(self):
        if hasattr(self, 'redis_client'):
            self.redis_client.close()
        super().close()

class ElasticsearchHandler(logging.Handler):
    def __init__(self, hosts: List[str], index_name: str = 'logs', **es_config):
        super().__init__()
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch is required for ElasticsearchHandler")
        self.index_name = index_name
        self.es_client = elasticsearch.Elasticsearch(hosts, **es_config)
    
    def emit(self, record):
        try:
            log_entry = json.loads(self.format(record))
            doc_id = f"{log_entry.get('service', {}).get('name', 'unknown')}_{int(time.time() * 1000000)}"
            self.es_client.index(
                index=f"{self.index_name}-{datetime.now().strftime('%Y.%m.%d')}",
                id=doc_id,
                body=log_entry
            )
        except Exception:
            self.handleError(record)
    
    def close(self):
        if hasattr(self, 'es_client'):
            self.es_client.transport.close()
        super().close()

class AsyncLogHandler(logging.Handler):
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self._shutdown = False
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.queue_size = queue_size
    
    def emit(self, record):
        try:
            self.queue.put_nowait(record)
            if queue_size_gauge:
                queue_size_gauge.set(self.queue.qsize())
        except queue.Full:
            if dropped_logs_counter:
                dropped_logs_counter.labels(reason="queue_full").inc()
    
    def _worker(self):
        while not self._shutdown:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:
                    break
                start_time = time.time()
                self.target_handler.emit(record)
                if log_latency:
                    log_latency.observe(time.time() - start_time)
                self.queue.task_done()
                if queue_size_gauge:
                    queue_size_gauge.set(self.queue.qsize())
            except queue.Empty:
                continue
            except Exception:
                if dropped_logs_counter:
                    dropped_logs_counter.labels(reason="processing_error").inc()
    
    def close(self):
        self._shutdown = True
        self.queue.put(None)
        self.worker.join(timeout=5.0)
        self.target_handler.close()
        super().close()
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self._shutdown = False
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
    
    def emit(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass
    
    def _worker(self):
        while not self._shutdown:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def close(self):
        self._shutdown = True
        self.queue.put(None)
        self.worker.join(timeout=5.0)
        self.target_handler.close()
        super().close()

class EnterpriseJsonFormatter(logging.Formatter):
    def __init__(self, service_name: str = "unknown", version: str = "1.0.0", 
                 environment: str = "development", security_config: Optional[SecurityConfig] = None):
        super().__init__()
        self.service_name = service_name
        self.version = version
        self.environment = environment
        self.security_config = security_config or SecurityConfig()
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        self.instance_id = str(uuid.uuid4())[:8]
        self.boot_time = datetime.now(timezone.utc).isoformat()
        
    def format(self, record):
        trace_ctx = trace_context.get({})
        user_ctx = user_context.get({})
        request_ctx = request_context.get({})
        
        base_entry = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "@version": "1",
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": record.name,
            "thread": record.thread,
            "service": {
                "name": self.service_name,
                "version": self.version,
                "environment": self.environment,
                "instance_id": self.instance_id
            },
            "host": {
                "name": self.hostname,
                "os": platform.system(),
                "architecture": platform.machine(),
                "ip": self._get_local_ip()
            },
            "process": {
                "pid": self.process_id,
                "name": sys.argv[0] if sys.argv else "unknown"
            },
            "labels": {
                "classification": self.security_config.classification.value
            }
        }
        
        if trace_ctx:
            base_entry["trace"] = trace_ctx
        
        if user_ctx:
            base_entry["user"] = user_ctx
            
        if request_ctx:
            base_entry["request"] = request_ctx
        
        if hasattr(record, 'extra_data'):
            extra_data = record.extra_data
            if self.security_config.enable_scrubbing:
                extra_data = PIIScrubber.scrub_data(extra_data, self.security_config.pii_fields)
            base_entry.update(extra_data)
        
        if record.exc_info:
            base_entry["exception"] = {
                "class": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": self.formatException(record.exc_info)
            }
        
        log_json = json.dumps(base_entry, separators=(',', ':'), default=str)
        
        if self.security_config.enable_signing:
            signature = self._sign_log(log_json)
            signed_entry = {"log": base_entry, "signature": signature}
            log_json = json.dumps(signed_entry, separators=(',', ':'))
        
        if self.security_config.enable_encryption:
            log_json = self._encrypt_log(log_json)
        
        return log_json
    
    def _get_local_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def _sign_log(self, log_data: str) -> str:
        if not self.security_config.signing_key:
            return ""
        key = self.security_config.signing_key.encode()
        return hmac.new(key, log_data.encode(), hashlib.sha256).hexdigest()
    
    def _encrypt_log(self, log_data: str) -> str:
        if not AESGCM or not self.security_config.encryption_key:
            return log_data
        
        try:
            aesgcm = AESGCM(self.security_config.get_encryption_key_bytes())
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, log_data.encode('utf-8'), None)
            
            encrypted_payload = {
                "encrypted": True,
                "nonce": base64.b64encode(nonce).decode('utf-8'),
                "data": base64.b64encode(ciphertext).decode('utf-8'),
                "algorithm": "AES-GCM"
            }
            return json.dumps(encrypted_payload, separators=(',', ':'))
        except Exception as e:
            if dropped_logs_counter:
                dropped_logs_counter.labels(reason="encryption_error").inc()
            return log_data

class MultiDestinationHandler(logging.Handler):
    def __init__(self, destinations: List[Dict[str, Any]], validate_schema: bool = True):
        super().__init__()
        self.handlers = []
        self.validate_schema = validate_schema
        self._setup_handlers(destinations)
    
    def _setup_handlers(self, destinations: List[Dict[str, Any]]):
        for dest_config in destinations:
            dest_type = dest_config.get('type')
            handler = None
            
            if dest_type == 'console':
                handler = logging.StreamHandler()
            elif dest_type == 'file':
                handler = logging.handlers.RotatingFileHandler(
                    filename=dest_config.get('filename', 'app.log'),
                    maxBytes=dest_config.get('max_bytes', 10*1024*1024),
                    backupCount=dest_config.get('backup_count', 5)
                )
            elif dest_type == 'syslog':
                handler = logging.handlers.SysLogHandler(
                    address=dest_config.get('address', ('localhost', 514))
                )
            elif dest_type == 'kafka' and KAFKA_AVAILABLE:
                handler = KafkaHandler(
                    topic=dest_config.get('topic', 'logs'),
                    bootstrap_servers=dest_config.get('bootstrap_servers', ['localhost:9092'])
                )
            elif dest_type == 'redis' and REDIS_AVAILABLE:
                handler = RedisHandler(
                    host=dest_config.get('host', 'localhost'),
                    port=dest_config.get('port', 6379),
                    channel=dest_config.get('channel', 'logs')
                )
            elif dest_type == 'elasticsearch' and ELASTICSEARCH_AVAILABLE:
                handler = ElasticsearchHandler(
                    hosts=dest_config.get('hosts', ['localhost:9200']),
                    index_name=dest_config.get('index_name', 'logs')
                )
            
            if handler:
                level = dest_config.get('level', 'INFO')
                handler.setLevel(getattr(logging, level.upper()))
                self.handlers.append(handler)
    
    def emit(self, record):
        if self.validate_schema and BaseModel != object:
            try:
                log_data = json.loads(self.format(record))
                LogSchema(**log_data)
            except (ValidationError, json.JSONDecodeError) as e:
                if dropped_logs_counter:
                    dropped_logs_counter.labels(reason="schema_validation").inc()
                return
        
        for handler in self.handlers:
            if record.levelno >= handler.level:
                try:
                    handler.emit(record)
                    if log_counter:
                        service_name = getattr(record, 'service_name', 'unknown')
                        environment = getattr(record, 'environment', 'unknown')
                        log_counter.labels(
                            service=service_name, 
                            level=record.levelname.lower(), 
                            environment=environment
                        ).inc()
                    if record.levelname.lower() == 'error' and error_gauge:
                        error_gauge.inc()
                except Exception:
                    if dropped_logs_counter:
                        dropped_logs_counter.labels(reason="handler_error").inc()
    
    def close(self):
        for handler in self.handlers:
            handler.close()
        super().close()

class EnterpriseLogger:
    _instances = WeakSet()
    _metrics = LogMetrics()
    _circuit_breaker = CircuitBreaker()
    _rate_limiter = RateLimiter()
    _sampler = LogSampler()
    
    def __init__(self, name: str = __name__, service_name: str = "unknown", 
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.service_name = service_name
        self.config = config or {}
        self._logger = logging.getLogger(name)
        self._setup_logger()
        EnterpriseLogger._instances.add(self)
    
    def _setup_logger(self):
        if self._logger.handlers:
            return
        
        security_config = SecurityConfig(
            enable_encryption=self.config.get('security', {}).get('enable_encryption', False),
            enable_signing=self.config.get('security', {}).get('enable_signing', False),
            enable_scrubbing=self.config.get('security', {}).get('enable_scrubbing', True)
        )
        
        formatter = EnterpriseJsonFormatter(
            service_name=self.service_name,
            version=self.config.get('version', '1.0.0'),
            environment=self.config.get('environment', 'development'),
            security_config=security_config
        )
        
        destinations = self.config.get('destinations', [{'type': 'console'}])
        handler = MultiDestinationHandler(destinations)
        
        if self.config.get('async_logging', True):
            handler = AsyncLogHandler(handler)
        
        handler.setFormatter(formatter)
        
        level = os.getenv('LOG_LEVEL', self.config.get('level', 'INFO')).upper()
        self._logger.setLevel(getattr(logging, level))
        self._logger.addHandler(handler)
        self._logger.propagate = False
    
    def _should_log(self, level: str) -> bool:
        return (self._rate_limiter.is_allowed() and 
                self._sampler.should_log(level))
    
    def _log_with_context(self, level: str, msg: str, **kwargs):
        if not self._should_log(level):
            return
        
        try:
            extra_data = dict(kwargs)
            log_size = len(msg) + len(str(extra_data))
            
            def _emit_log():
                log_level = getattr(logging, level.upper())
                self._logger.log(log_level, msg, extra={'extra_data': extra_data})
            
            self._circuit_breaker.call(_emit_log)
            self._metrics.increment(level, log_size)
            
        except Exception as e:
            self._metrics.dropped_logs += 1
            self._metrics.last_error = str(e)
    
    def debug(self, msg: str, **kwargs):
        self._log_with_context('debug', msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log_with_context('info', msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log_with_context('warning', msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log_with_context('error', msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log_with_context('critical', msg, **kwargs)
    
    def audit(self, action: str, resource: str, user_id: str = None, **kwargs):
        audit_data = {
            'audit': True,
            'action': action,
            'resource': resource,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.info(f"Audit: {action} on {resource}", **audit_data)
    
    @contextmanager
    def span(self, operation_name: str, **tags):
        span_id = str(uuid.uuid4())
        parent_trace = trace_context.get({})
        
        span_context = {
            'trace_id': parent_trace.get('trace_id', str(uuid.uuid4())),
            'span_id': span_id,
            'parent_span_id': parent_trace.get('span_id'),
            'operation_name': operation_name,
            'start_time': time.time(),
            'tags': tags
        }
        
        token = trace_context.set(span_context)
        
        try:
            self.debug(f"Span started: {operation_name}", span=span_context)
            yield span_context
        except Exception as e:
            span_context['error'] = True
            span_context['error_message'] = str(e)
            self.error(f"Span error: {operation_name}", span=span_context, exception=str(e))
            raise
        finally:
            span_context['duration'] = time.time() - span_context['start_time']
            self.debug(f"Span completed: {operation_name}", span=span_context)
            trace_context.reset(token)
    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        return {
            'total_logs': cls._metrics.total_logs,
            'error_count': cls._metrics.error_count,
            'warning_count': cls._metrics.warning_count,
            'bytes_written': cls._metrics.bytes_written,
            'dropped_logs': cls._metrics.dropped_logs,
            'last_error': cls._metrics.last_error,
            'uptime': time.time() - cls._metrics.start_time,
            'circuit_breaker_state': cls._circuit_breaker.state,
            'active_loggers': len(cls._instances)
        }
    
    @classmethod
    def configure_sampling(cls, sample_rate: float):
        cls._sampler = LogSampler(sample_rate)
    
    @classmethod
    def configure_rate_limiting(cls, max_rate: int, window_size: float = 1.0):
        cls._rate_limiter = RateLimiter(max_rate, window_size)

def setup_enterprise_logging(service_name: str = "unknown", config: Optional[Dict[str, Any]] = None):
    config = config or {}
    
    root_logger = logging.getLogger()
    
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    enterprise_logger = EnterpriseLogger("root", service_name, config)
    
    integrations = config.get('integrations', {
        'uvicorn': True,
        'fastapi': True,
        'sqlalchemy': True,
        'requests': True,
        'urllib3': True
    })
    
    for logger_name, enabled in integrations.items():
        if enabled:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            for handler in enterprise_logger._logger.handlers:
                logger.addHandler(handler)
            logger.setLevel(enterprise_logger._logger.level)
            logger.propagate = False

def set_trace_context(trace_id: str = None, span_id: str = None, **additional):
    context = {
        'trace_id': trace_id or str(uuid.uuid4()),
        'span_id': span_id or str(uuid.uuid4()),
        **additional
    }
    return trace_context.set(context)

def set_user_context(user_id: str = None, session_id: str = None, **additional):
    context = {
        'user_id': user_id,
        'session_id': session_id,
        **additional
    }
    return user_context.set({k: v for k, v in context.items() if v is not None})

def set_request_context(request_id: str = None, method: str = None, path: str = None, **additional):
    context = {
        'request_id': request_id or str(uuid.uuid4()),
        'method': method,
        'path': path,
        **additional
    }
    return request_context.set({k: v for k, v in context.items() if v is not None})

def get_trace_id() -> Optional[str]:
    return trace_context.get({}).get('trace_id')

def get_span_id() -> Optional[str]:
    return trace_context.get({}).get('span_id')

def clear_all_context():
    trace_context.set({})
    user_context.set({})
    request_context.set({})

def health_check() -> Dict[str, Any]:
    metrics = EnterpriseLogger.get_metrics()
    return {
        'status': 'healthy' if metrics['circuit_breaker_state'] == 'closed' else 'degraded',
        'metrics': metrics,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

JsonLogger = EnterpriseLogger
setup_logging = setup_enterprise_logging
set_trace_id = lambda trace_id=None: set_trace_context(trace_id=trace_id).get('trace_id')
clear_trace_id = clear_all_context