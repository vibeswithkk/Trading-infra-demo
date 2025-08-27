from __future__ import annotations
import asyncio
import logging
import os
import re
import ssl
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Type
from urllib.parse import urlparse

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = lambda *args, **kwargs: None

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    SQLAlchemyInstrumentor = None

import sqlalchemy
from sqlalchemy import Column, Integer, DateTime, String, event, pool, text
from sqlalchemy.exc import (
    DisconnectionError, SQLAlchemyError, OperationalError,
    DatabaseError, IntegrityError, TimeoutError as SQLTimeoutError
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.sql import func

from .config import settings
from .logging import EnterpriseLogger

logger = EnterpriseLogger(__name__, 'database-manager')

if PROMETHEUS_AVAILABLE:
    db_connections_active = Gauge('db_connections_active', 'Active database connections')
    db_connections_total = Counter('db_connections_total', 'Total database connections created')
    db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    db_errors_total = Counter('db_errors_total', 'Total database errors', ['error_type'])
    db_slow_queries_total = Counter('db_slow_queries_total', 'Total slow queries')
    db_pool_size = Gauge('db_pool_size', 'Database connection pool size')
    db_pool_checked_out = Gauge('db_pool_checked_out', 'Checked out connections from pool')
    db_transactions_total = Counter('db_transactions_total', 'Total database transactions', ['status'])
else:
    db_connections_active = db_connections_total = db_query_duration = None
    db_errors_total = db_slow_queries_total = db_pool_size = db_pool_checked_out = db_transactions_total = None

if OPENTELEMETRY_AVAILABLE:
    tracer = trace.get_tracer(__name__)
else:
    tracer = None

class DatabaseState(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"

class ConnectionStrategy(Enum):
    SINGLE = "single"
    MASTER_SLAVE = "master_slave"
    CLUSTER = "cluster"
    SHARDED = "sharded"

@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    query_timeout: int = 30
    slow_query_threshold: float = 1.0
    echo: bool = False
    echo_pool: bool = False
    isolation_level: Optional[str] = None
    connect_args: Dict[str, Any] = field(default_factory=dict)
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    enable_ssl: bool = True
    ssl_context: Optional[ssl.SSLContext] = None
    connection_strategy: ConnectionStrategy = ConnectionStrategy.SINGLE
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_audit_logging: bool = True
    enable_pii_scrubbing: bool = True
    
    def __post_init__(self):
        if self.enable_ssl and not self.ssl_context:
            self.ssl_context = ssl.create_default_context()
            if 'postgresql' in self.url:
                self.ssl_context.check_hostname = False
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED
            
        if self.enable_ssl and 'postgresql' in self.url:
            if 'sslmode' not in self.url:
                self.connect_args['sslmode'] = 'require'
            if 'sslcontext' not in self.connect_args:
                self.connect_args['ssl_context'] = self.ssl_context

class PIIScrubber:
    PII_PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3,4}[-.]?[0-9]{4}\b|\b[0-9]{3}-[0-9]{4}\b'),
        'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    }
    
    @classmethod
    def scrub_query(cls, query_text: str) -> str:
        scrubbed = query_text
        for pattern_name, pattern in cls.PII_PATTERNS.items():
            scrubbed = pattern.sub(lambda m: f'[{pattern_name.upper()}_REDACTED]', scrubbed)
        return scrubbed

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                    logger.info("Circuit breaker transitioning to half-open state")
                else:
                    raise Exception("Circuit breaker is open - database unavailable")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - database recovered")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                raise e

class BaseMixin:
    @declared_attr
    def id(cls):
        return Column(Integer, primary_key=True, autoincrement=True)
    
    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    @declared_attr
    def updated_at(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    @declared_attr
    def version(cls):
        return Column(Integer, default=1, nullable=False)

class EnterpriseBase(DeclarativeBase):
    __abstract__ = True
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, '__tablename__'):
            logger.debug(f"Registered table: {cls.__tablename__}", table=cls.__tablename__)

class DatabaseManager:
    _instance = None
    _instances = weakref.WeakSet()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or self._create_default_config()
        self.engines: Dict[str, AsyncEngine] = {}
        self.session_makers: Dict[str, async_sessionmaker] = {}
        self.circuit_breaker = CircuitBreaker()
        self.start_time = time.time()
        self._health_check_task = None
        self._initialized = True
        
        DatabaseManager._instances.add(self)
        
        if OPENTELEMETRY_AVAILABLE and SQLAlchemyInstrumentor:
            SQLAlchemyInstrumentor().instrument()
        
        logger.info("Database manager initialized", 
                   strategy=self.config.connection_strategy.value,
                   pool_size=self.config.pool_size,
                   ssl_enabled=self.config.enable_ssl)
    
    def _create_default_config(self) -> DatabaseConfig:
        db_url = os.getenv('QTINFRA_DB_URL') or settings.db_url
        
        if not db_url:
            raise ValueError("Database URL must be provided via QTINFRA_DB_URL environment variable or settings")
        
        parsed = urlparse(db_url)
        config_overrides = {}
        
        if 'postgresql' in db_url:
            config_overrides.update({
                'pool_size': 20,
                'max_overflow': 30,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'enable_ssl': True,
                'connect_args': {
                    'server_settings': {
                        'application_name': 'qtinfra-trading-system',
                        'jit': 'off'
                    }
                }
            })
        elif 'mysql' in db_url:
            config_overrides.update({
                'pool_size': 15,
                'max_overflow': 25,
                'pool_recycle': 7200,
                'enable_ssl': True,
                'connect_args': {
                    'charset': 'utf8mb4',
                    'sql_mode': 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'
                }
            })
        elif 'sqlite' in db_url:
            config_overrides.update({
                'pool_size': 1,
                'max_overflow': 0,
                'enable_ssl': False,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            })
        
        return DatabaseConfig(url=db_url, **config_overrides)
    
    async def initialize(self) -> None:
        try:
            await self._setup_engines()
            await self._setup_event_listeners()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            if db_errors_total:
                db_errors_total.labels(error_type='initialization_error').inc()
            raise
    
    async def _setup_engines(self) -> None:
        engine_kwargs = {
            'echo': self.config.echo,
            'echo_pool': self.config.echo_pool,
            'connect_args': self.config.connect_args,
            **self.config.engine_kwargs
        }
        
        # Only add pool parameters for databases that support them
        if 'sqlite' not in self.config.url:
            engine_kwargs.update({
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout,
                'pool_recycle': self.config.pool_recycle,
                'pool_pre_ping': self.config.pool_pre_ping,
            })
        
        if self.config.isolation_level:
            engine_kwargs['isolation_level'] = self.config.isolation_level
        
        self.engines['primary'] = create_async_engine(self.config.url, **engine_kwargs)
        
        self.session_makers['primary'] = async_sessionmaker(
            self.engines['primary'],
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database engines configured", 
                   engine_count=len(self.engines),
                   pool_size=self.config.pool_size)
    
    async def _setup_event_listeners(self) -> None:
        for engine_name, engine in self.engines.items():
            @event.listens_for(engine.sync_engine, "connect")
            def on_connect(dbapi_connection, connection_record):
                if db_connections_total:
                    db_connections_total.inc()
                logger.debug("Database connection established", engine=engine_name)
            
            @event.listens_for(engine.sync_engine, "checkout")
            def on_checkout(dbapi_connection, connection_record, connection_proxy):
                if db_connections_active:
                    db_connections_active.inc()
            
            @event.listens_for(engine.sync_engine, "checkin")
            def on_checkin(dbapi_connection, connection_record):
                if db_connections_active:
                    db_connections_active.dec()
            
            @event.listens_for(engine.sync_engine, "invalidate")
            def on_invalidate(dbapi_connection, connection_record, exception):
                if db_errors_total:
                    db_errors_total.labels(error_type='connection_invalidated').inc()
                logger.warning("Database connection invalidated", 
                             engine=engine_name, error=str(exception))
    
    @asynccontextmanager
    async def get_session(self, engine_name: str = 'primary') -> AsyncGenerator[AsyncSession, None]:
        if engine_name not in self.session_makers:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        session = self.session_makers[engine_name]()
        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        span_context = None
        if tracer:
            span_context = tracer.start_span(f"database_session_{engine_name}")
            span_context.set_attribute("db.session_id", session_id)
            span_context.set_attribute("db.engine", engine_name)
        
        try:
            logger.debug("Database session started", session_id=session_id, engine=engine_name)
            
            async with session:
                yield session
                
            if db_transactions_total:
                db_transactions_total.labels(status='committed').inc()
            
        except Exception as e:
            if db_transactions_total:
                db_transactions_total.labels(status='rollback').inc()
            if db_errors_total:
                db_errors_total.labels(error_type='session_error').inc()
            
            if span_context:
                span_context.set_attribute("error", True)
                span_context.set_attribute("error.message", str(e))
            
            logger.error("Database session error", 
                        session_id=session_id, 
                        error=str(e), 
                        error_type=type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            if db_query_duration:
                db_query_duration.observe(duration)
            
            if span_context:
                span_context.set_attribute("duration", duration)
                span_context.end()
            
            logger.debug("Database session completed", 
                        session_id=session_id, 
                        duration=duration)
    
    async def execute_with_retry(self, query, params=None, engine_name: str = 'primary', max_retries: Optional[int] = None):
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                return await self.circuit_breaker.call(self._execute_query, query, params, engine_name)
            except (DisconnectionError, OperationalError, SQLTimeoutError) as e:
                if attempt == max_retries:
                    logger.error("Query failed after all retries", 
                               query=self._scrub_query_for_logging(str(query)[:100]), 
                               attempts=attempt + 1, 
                               error=str(e))
                    if db_errors_total:
                        db_errors_total.labels(error_type='query_max_retries_exceeded').inc()
                    raise
                
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning("Query failed, retrying", 
                             attempt=attempt + 1, 
                             wait_time=wait_time, 
                             error=str(e))
                await asyncio.sleep(wait_time)
    
    async def _execute_query(self, query, params=None, engine_name: str = 'primary'):
        async with self.get_session(engine_name) as session:
            if isinstance(query, str):
                query_obj = text(query)
            else:
                query_obj = query
            
            start_time = time.time()
            
            span_context = None
            if tracer:
                span_context = tracer.start_span("database_query")
                span_context.set_attribute("db.statement", self._scrub_query_for_logging(str(query_obj)))
                span_context.set_attribute("db.engine", engine_name)
            
            try:
                result = await asyncio.wait_for(
                    session.execute(query_obj, params or {}),
                    timeout=self.config.query_timeout
                )
                
                duration = time.time() - start_time
                
                if duration > self.config.slow_query_threshold:
                    if db_slow_queries_total:
                        db_slow_queries_total.inc()
                    logger.warning("Slow query detected", 
                                 query=self._scrub_query_for_logging(str(query_obj)[:200]),
                                 duration=duration,
                                 threshold=self.config.slow_query_threshold)
                
                if self.config.enable_audit_logging:
                    logger.info("Query executed", 
                               query_type=query_obj.__class__.__name__,
                               duration=duration,
                               row_count=result.rowcount if hasattr(result, 'rowcount') else None)
                
                if span_context:
                    span_context.set_attribute("db.rows_affected", result.rowcount if hasattr(result, 'rowcount') else 0)
                    span_context.set_attribute("duration", duration)
                
                return result
                
            except asyncio.TimeoutError:
                logger.error("Query timeout exceeded", 
                           query=self._scrub_query_for_logging(str(query_obj)[:100]),
                           timeout=self.config.query_timeout)
                if db_errors_total:
                    db_errors_total.labels(error_type='query_timeout').inc()
                if span_context:
                    span_context.set_attribute("error", True)
                    span_context.set_attribute("error.type", "timeout")
                raise SQLTimeoutError("Query execution timed out")
            except Exception as e:
                if span_context:
                    span_context.set_attribute("error", True)
                    span_context.set_attribute("error.message", str(e))
                raise
            finally:
                if span_context:
                    span_context.end()
    
    def _scrub_query_for_logging(self, query_text: str) -> str:
        if self.config.enable_pii_scrubbing:
            return PIIScrubber.scrub_query(query_text)
        return query_text
    
    async def health_check(self, engine_name: str = 'primary') -> Dict[str, Any]:
        """Perform comprehensive health check on database connection"""
        health_status = {
            'engine': engine_name,
            'status': 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'response_time': None,
            'pool_status': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            async with self.get_session(engine_name) as session:
                # Simple connectivity test
                await session.execute(text('SELECT 1'))
                
                # Pool status check
                engine = self.engines[engine_name]
                pool = engine.pool
                
                # Handle different pool types
                try:
                    health_status['pool_status'] = {
                        'size': getattr(pool, 'size', lambda: 'N/A')(),
                        'checked_in': getattr(pool, 'checkedin', lambda: 'N/A')(),
                        'checked_out': getattr(pool, 'checkedout', lambda: 'N/A')(),
                        'overflow': getattr(pool, 'overflow', lambda: 'N/A')(),
                        'invalid': getattr(pool, 'invalid', lambda: 'N/A')()
                    }
                except Exception:
                    # For SQLite StaticPool and other simple pools
                    health_status['pool_status'] = {
                        'pool_type': type(pool).__name__,
                        'status': 'active'
                    }
                
                response_time = time.time() - start_time
                health_status['response_time'] = response_time
                
                if response_time < 0.1:
                    health_status['status'] = 'healthy'
                elif response_time < 1.0:
                    health_status['status'] = 'degraded'
                else:
                    health_status['status'] = 'slow'
                    health_status['errors'].append(f'Slow response time: {response_time:.3f}s')
                
                logger.debug("Health check completed", 
                           engine=engine_name, 
                           status=health_status['status'],
                           response_time=response_time)
                
        except Exception as e:
            health_status['status'] = 'unavailable'
            health_status['errors'].append(str(e))
            health_status['response_time'] = time.time() - start_time
            
            logger.error("Health check failed", 
                        engine=engine_name, 
                        error=str(e))
            
            if db_errors_total:
                db_errors_total.labels(error_type='health_check_failed').inc()
        
        return health_status
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        info = {
            'uptime': time.time() - self.start_time,
            'config': {
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout,
                'query_timeout': self.config.query_timeout,
                'ssl_enabled': self.config.enable_ssl,
                'connection_strategy': self.config.connection_strategy.value
            },
            'engines': {},
            'circuit_breaker': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'last_failure_time': self.circuit_breaker.last_failure_time
            }
        }
        
        for engine_name in self.engines.keys():
            info['engines'][engine_name] = await self.health_check(engine_name)
        
        return info
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Database health monitoring started", 
                   interval=self.config.health_check_interval)
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                for engine_name in self.engines.keys():
                    health_status = await self.health_check(engine_name)
                    
                    if health_status['status'] == 'unavailable':
                        logger.error("Database health check failed", 
                                   engine=engine_name,
                                   errors=health_status['errors'])
                    elif health_status['status'] == 'degraded':
                        logger.warning("Database performance degraded", 
                                     engine=engine_name,
                                     response_time=health_status['response_time'])
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("Health monitoring stopped")
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Database health monitoring stopped")
    
    async def shutdown(self) -> None:
        logger.info("Starting database shutdown")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        for engine_name, engine in self.engines.items():
            try:
                await engine.dispose()
                logger.debug("Engine disposed", engine=engine_name)
            except Exception as e:
                logger.error("Error disposing engine", engine=engine_name, error=str(e))
        
        logger.info("Database shutdown completed")

class BaseRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.db_manager.get_session() as session:
            yield session
    
    async def execute_query(self, query, params=None):
        return await self.db_manager.execute_with_retry(query, params)
    
    async def create(self, model_class: Type, **kwargs) -> Any:
        async with self.get_session() as session:
            instance = model_class(**kwargs)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance
    
    async def get_by_id(self, model_class: Type, id: int) -> Optional[Any]:
        async with self.get_session() as session:
            result = await session.get(model_class, id)
            return result
    
    async def update(self, instance: Any, **kwargs) -> Any:
        async with self.get_session() as session:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance
    
    async def delete(self, instance: Any) -> None:
        async with self.get_session() as session:
            await session.delete(instance)
            await session.commit()

Base = EnterpriseBase
db_manager = DatabaseManager()

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with db_manager.get_session() as session:
        yield session

async def init_db() -> None:
    await db_manager.initialize()

async def close_db() -> None:
    await db_manager.shutdown()

engine = None
SessionLocal = None

async def _setup_legacy_compatibility():
    global engine, SessionLocal
    if 'primary' in db_manager.engines:
        engine = db_manager.engines['primary']
        SessionLocal = db_manager.session_makers['primary']

if hasattr(asyncio, 'current_task'):
    try:
        asyncio.current_task()
        asyncio.create_task(_setup_legacy_compatibility())
    except RuntimeError:
        pass