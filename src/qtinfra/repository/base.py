from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, 
    Sequence, Set, Type, TypeVar, Union
)
from weakref import WeakValueDictionary

from prometheus_client import Counter, Histogram
from sqlalchemy import (
    and_, delete, desc, func, or_, select, text, update
)
from sqlalchemy.exc import (
    IntegrityError, OperationalError, SQLAlchemyError, 
    DisconnectionError
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.sql import Select

from ..infra.config import get_settings
from ..infra.logging import EnterpriseLogger

settings = get_settings()
logger = EnterpriseLogger(__name__, 'repository-layer')

T = TypeVar("T")

class SortDirection(str, Enum):
    """Sort direction enumeration for query ordering."""
    ASC = "ASC"
    DESC = "DESC"

class CacheStrategy(str, Enum):
    """Cache strategy enumeration for repository caching."""
    NONE = "NONE"
    READ_THROUGH = "READ_THROUGH"
    WRITE_THROUGH = "WRITE_THROUGH"
    WRITE_BEHIND = "WRITE_BEHIND"

@dataclass
class QueryFilter:
    """Query filter for dynamic filtering."""
    field: str
    operator: str
    value: Any
    case_sensitive: bool = True

@dataclass
class SortOrder:
    """Sort order configuration."""
    field: str
    direction: SortDirection = SortDirection.ASC

@dataclass
class PaginationRequest:
    """Pagination request configuration."""
    offset: int = 0
    limit: int = 50
    max_limit: int = 1000
    
    def __post_init__(self):
        if self.limit > self.max_limit:
            self.limit = self.max_limit
        if self.offset < 0:
            self.offset = 0

@dataclass
class PaginationResponse(Generic[T]):
    """Pagination response with results and metadata."""
    items: List[T]
    total_count: int
    offset: int
    limit: int
    has_next: bool
    has_previous: bool

class CircuitBreaker:
    """Circuit breaker for database fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise RuntimeError("Circuit breaker is open - database unavailable")

            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                raise e

class CacheManager:
    """Cache manager for repository caching."""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.READ_THROUGH):
        self.strategy = strategy
        self.cache = WeakValueDictionary()
        self.ttl_cache = {}
        self.default_ttl = 3600
        
    def get_cache_key(self, model_type: Type, identifier: Any) -> str:
        return f"{model_type.__name__}:{identifier}"
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if key in self.ttl_cache:
                if time.time() > self.ttl_cache[key]:
                    del self.cache[key]
                    del self.ttl_cache[key]
                    return None
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.cache[key] = value
        expire_time = time.time() + (ttl or self.default_ttl)
        self.ttl_cache[key] = expire_time
    
    async def delete(self, key: str) -> None:
        self.cache.pop(key, None)
        self.ttl_cache.pop(key, None)

class RepositoryMetrics:
    """Repository metrics for monitoring."""
    
    def __init__(self):
        self.query_counter = Counter('repository_queries_total', 'Total repository queries', ['operation', 'model'])
        self.query_duration = Histogram('repository_query_duration_seconds', 'Repository query duration', ['operation', 'model'])
        self.error_counter = Counter('repository_errors_total', 'Repository errors', ['operation', 'error_type'])

class AsyncRepository(Generic[T]):
    """Enterprise-grade async repository with comprehensive features.
    
    Provides full CRUD operations, filtering, pagination, error handling,
    logging, caching, bulk operations, soft delete, and advanced querying.
    
    Note: Commit handling is managed at the service/unit-of-work layer.
    This repository focuses on data access operations without transaction boundaries.
    """
    
    def __init__(
        self, 
        session: AsyncSession, 
        model: Type[T],
        cache_strategy: CacheStrategy = CacheStrategy.READ_THROUGH,
        enable_soft_delete: bool = True,
        enable_audit: bool = True
    ):
        """Initialize enterprise repository.
        
        Args:
            session: SQLAlchemy async session
            model: SQLAlchemy model class
            cache_strategy: Caching strategy to use
            enable_soft_delete: Enable soft delete functionality
            enable_audit: Enable audit logging
        """
        self.session = session
        self.model = model
        self.model_name = model.__name__
        self.cache_manager = CacheManager(cache_strategy)
        self.circuit_breaker = CircuitBreaker()
        self.metrics = RepositoryMetrics()
        self.enable_soft_delete = enable_soft_delete
        self.enable_audit = enable_audit
        self.log = EnterpriseLogger(__name__, f'repository-{self.model_name.lower()}')
        
    async def add(self, entity: T, **kwargs) -> T:
        """Add a new entity to the repository.
        
        Args:
            entity: Entity to add
            **kwargs: Additional context (e.g., user_id for audit)
            
        Returns:
            Added entity with generated ID
            
        Raises:
            IntegrityError: If entity violates database constraints
            OperationalError: If database operation fails
        """
        operation_start = time.time()
        
        try:
            if self.enable_audit:
                entity_dict = self._entity_to_dict(entity)
                self.log.info("Creating entity", 
                           model=self.model_name, 
                           entity_id=getattr(entity, 'id', None),
                           audit_data=entity_dict)
            
            result = await self.circuit_breaker.call(self._add_internal, entity, **kwargs)
            
            # Cache the result
            if hasattr(entity, 'id') and entity.id:
                cache_key = self.cache_manager.get_cache_key(self.model, entity.id)
                await self.cache_manager.set(cache_key, result)
            
            # Metrics
            self.metrics.query_counter.labels(operation='add', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='add', model=self.model_name).observe(duration)
            
            self.log.info("Entity created successfully", 
                        model=self.model_name,
                        entity_id=getattr(result, 'id', None),
                        duration_ms=duration * 1000)
            
            return result
            
        except IntegrityError as e:
            self.metrics.error_counter.labels(operation='add', error_type='integrity_error').inc()
            self.log.error("Integrity error creating entity", 
                        model=self.model_name, 
                        error=str(e))
            raise
        except OperationalError as e:
            self.metrics.error_counter.labels(operation='add', error_type='operational_error').inc()
            self.log.error("Operational error creating entity", 
                        model=self.model_name, 
                        error=str(e))
            raise
        except Exception as e:
            self.metrics.error_counter.labels(operation='add', error_type='unknown_error').inc()
            self.log.error("Unknown error creating entity", 
                        model=self.model_name, 
                        error=str(e),
                        error_type=type(e).__name__)
            raise

    async def _add_internal(self, entity: T, **kwargs) -> T:
        """Internal add method with audit field handling."""
        # Set audit fields
        if self.enable_audit and hasattr(entity, 'created_at'):
            entity.created_at = datetime.now(timezone.utc)
        if hasattr(entity, 'created_by'):
            entity.created_by = kwargs.get('user_id')
        
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def get(self, id_: Any, include_deleted: bool = False) -> Optional[T]:
        """Get entity by ID with caching and soft delete support.
        
        Args:
            id_: Entity ID
            include_deleted: Include soft-deleted entities
            
        Returns:
            Entity if found, None otherwise
            
        Raises:
            OperationalError: If database operation fails
        """
        operation_start = time.time()
        
        try:
            # Check cache first
            cache_key = self.cache_manager.get_cache_key(self.model, id_)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                self.log.debug("Cache hit", model=self.model_name, entity_id=id_)
                return cached_result
            
            # Database query
            result = await self.circuit_breaker.call(self._get_internal, id_, include_deleted)
            
            # Cache the result
            if result:
                await self.cache_manager.set(cache_key, result)
            
            # Metrics
            self.metrics.query_counter.labels(operation='get', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='get', model=self.model_name).observe(duration)
            
            if result:
                self.log.debug("Entity retrieved", model=self.model_name, entity_id=id_)
            
            return result
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='get', error_type=type(e).__name__).inc()
            self.log.error("Error retrieving entity", 
                        model=self.model_name, 
                        entity_id=id_,
                        error=str(e))
            raise

    async def _get_internal(self, id_: Any, include_deleted: bool = False) -> Optional[T]:
        """Internal get method with soft delete handling."""
        query = select(self.model).where(self.model.id == id_)
        
        # Soft delete handling
        if self.enable_soft_delete and hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.where(self.model.deleted_at.is_(None))
        elif self.enable_soft_delete and hasattr(self.model, 'is_active') and not include_deleted:
            query = query.where(self.model.is_active == True)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update(self, entity: T = None, id_: Any = None, **fields) -> Optional[T]:
        """Update entity with partial field updates.
        
        Args:
            entity: Entity instance to update (if provided)
            id_: Entity ID to update (if entity not provided)
            **fields: Fields to update
            
        Returns:
            Updated entity or None if not found
            
        Raises:
            ValueError: If neither entity nor id_ provided
            IntegrityError: If update violates constraints
            OperationalError: If database operation fails
        """
        if entity is None and id_ is None:
            raise ValueError("Either entity or id_ must be provided")
        
        operation_start = time.time()
        
        try:
            if entity is None:
                entity = await self.get(id_)
                if entity is None:
                    return None
            
            if self.enable_audit:
                self.log.info("Updating entity", 
                           model=self.model_name, 
                           entity_id=getattr(entity, 'id', None),
                           fields=list(fields.keys()))
            
            result = await self.circuit_breaker.call(self._update_internal, entity, **fields)
            
            # Cache invalidation
            if hasattr(entity, 'id') and entity.id:
                cache_key = self.cache_manager.get_cache_key(self.model, entity.id)
                await self.cache_manager.set(cache_key, result)
            
            # Metrics
            self.metrics.query_counter.labels(operation='update', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='update', model=self.model_name).observe(duration)
            
            self.log.info("Entity updated successfully", 
                        model=self.model_name,
                        entity_id=getattr(result, 'id', None),
                        duration_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='update', error_type=type(e).__name__).inc()
            self.log.error("Error updating entity", 
                        model=self.model_name, 
                        entity_id=getattr(entity, 'id', None) if entity else id_,
                        error=str(e))
            raise

    async def _update_internal(self, entity: T, **fields) -> T:
        """Internal update method with audit field handling."""
        # Apply field updates
        for field, value in fields.items():
            if hasattr(entity, field):
                setattr(entity, field, value)
        
        # Set audit fields
        if self.enable_audit and hasattr(entity, 'updated_at'):
            entity.updated_at = datetime.now(timezone.utc)
        if hasattr(entity, 'updated_by'):
            entity.updated_by = fields.get('user_id')
        if hasattr(entity, 'version'):
            entity.version += 1
        
        await self.session.merge(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_or_id: Union[T, Any], hard_delete: bool = False, **kwargs) -> bool:
        """Delete entity by instance or ID with soft delete support.
        
        Args:
            entity_or_id: Entity instance or ID to delete
            hard_delete: Force hard delete even if soft delete is enabled
            **kwargs: Additional context (e.g., user_id for audit)
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            OperationalError: If database operation fails
        """
        operation_start = time.time()
        
        try:
            # Get entity if ID provided
            if not hasattr(entity_or_id, '__dict__'):
                entity = await self.get(entity_or_id)
                if entity is None:
                    return False
            else:
                entity = entity_or_id
            
            if self.enable_audit:
                self.log.info("Deleting entity", 
                           model=self.model_name, 
                           entity_id=getattr(entity, 'id', None),
                           hard_delete=hard_delete)
            
            result = await self.circuit_breaker.call(self._delete_internal, entity, hard_delete, **kwargs)
            
            # Cache invalidation
            if hasattr(entity, 'id') and entity.id:
                cache_key = self.cache_manager.get_cache_key(self.model, entity.id)
                await self.cache_manager.delete(cache_key)
            
            # Metrics
            self.metrics.query_counter.labels(operation='delete', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='delete', model=self.model_name).observe(duration)
            
            self.log.info("Entity deleted successfully", 
                        model=self.model_name,
                        entity_id=getattr(entity, 'id', None),
                        hard_delete=hard_delete,
                        duration_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='delete', error_type=type(e).__name__).inc()
            self.log.error("Error deleting entity", 
                        model=self.model_name, 
                        error=str(e))
            raise

    async def _delete_internal(self, entity: T, hard_delete: bool = False, **kwargs) -> bool:
        """Internal delete method with soft delete handling."""
        if not hard_delete and self.enable_soft_delete:
            if hasattr(entity, 'deleted_at'):
                # Soft delete with timestamp
                entity.deleted_at = datetime.now(timezone.utc)
                if hasattr(entity, 'deleted_by'):
                    entity.deleted_by = kwargs.get('user_id')
                await self.session.merge(entity)
            elif hasattr(entity, 'is_active'):
                # Soft delete with boolean flag
                entity.is_active = False
                if hasattr(entity, 'deleted_by'):
                    entity.deleted_by = kwargs.get('user_id')
                await self.session.merge(entity)
            else:
                # No soft delete fields, perform hard delete
                await self.session.delete(entity)
        else:
            # Hard delete
            await self.session.delete(entity)
        
        await self.session.flush()
        return True

    async def list(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[PaginationRequest] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        include_relationships: Optional[List[str]] = None,
        include_deleted: bool = False
    ) -> Union[Sequence[T], PaginationResponse[T]]:
        """List entities with filtering, pagination, and ordering.
        
        Args:
            filters: Dictionary of filters {field: value} or {field: {operator: value}}
            pagination: Pagination configuration
            order_by: Field(s) to order by, can include direction like 'field:desc'
            include_relationships: Relationships to eagerly load
            include_deleted: Include soft-deleted entities
            
        Returns:
            Sequence of entities or PaginationResponse if pagination provided
            
        Raises:
            ValueError: If invalid filter or order_by provided
            OperationalError: If database operation fails
        """
        operation_start = time.time()
        
        try:
            result = await self.circuit_breaker.call(
                self._list_internal, 
                filters, pagination, order_by, include_relationships, include_deleted
            )
            
            # Metrics
            self.metrics.query_counter.labels(operation='list', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='list', model=self.model_name).observe(duration)
            
            item_count = len(result.items) if isinstance(result, PaginationResponse) else len(result)
            self.log.debug("Entities listed", 
                         model=self.model_name,
                         count=item_count,
                         duration_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='list', error_type=type(e).__name__).inc()
            self.log.error("Error listing entities", 
                        model=self.model_name,
                        error=str(e))
            raise

    async def _list_internal(
        self,
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[PaginationRequest] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        include_relationships: Optional[List[str]] = None,
        include_deleted: bool = False
    ) -> Union[Sequence[T], PaginationResponse[T]]:
        """Internal list method with query building."""
        query = select(self.model)
        
        # Apply filters
        if filters:
            conditions = []
            for field, value in filters.items():
                if not hasattr(self.model, field):
                    raise ValueError(f"Field '{field}' does not exist on model {self.model_name}")
                
                if isinstance(value, dict):
                    # Complex filter with operator
                    for operator, op_value in value.items():
                        condition = self._build_filter_condition(field, operator, op_value)
                        conditions.append(condition)
                else:
                    # Simple equality filter
                    conditions.append(getattr(self.model, field) == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Soft delete handling
        if self.enable_soft_delete and not include_deleted:
            if hasattr(self.model, 'deleted_at'):
                query = query.where(self.model.deleted_at.is_(None))
            elif hasattr(self.model, 'is_active'):
                query = query.where(self.model.is_active == True)
        
        # Apply ordering
        if order_by:
            order_fields = order_by if isinstance(order_by, list) else [order_by]
            for order_field in order_fields:
                if ':' in order_field:
                    field_name, direction = order_field.split(':', 1)
                    direction = direction.upper()
                else:
                    field_name, direction = order_field, 'ASC'
                
                if not hasattr(self.model, field_name):
                    raise ValueError(f"Field '{field_name}' does not exist on model {self.model_name}")
                
                field = getattr(self.model, field_name)
                if direction == 'DESC':
                    field = desc(field)
                query = query.order_by(field)
        
        # Apply relationship loading
        if include_relationships:
            for relationship in include_relationships:
                if hasattr(self.model, relationship):
                    query = query.options(selectinload(getattr(self.model, relationship)))
        
        # Handle pagination
        if pagination:
            # Get total count first
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar()
            
            # Apply pagination
            query = query.offset(pagination.offset).limit(pagination.limit)
            
            # Execute query
            result = await self.session.execute(query)
            items = list(result.scalars().all())
            
            return PaginationResponse(
                items=items,
                total_count=total_count,
                offset=pagination.offset,
                limit=pagination.limit,
                has_next=pagination.offset + pagination.limit < total_count,
                has_previous=pagination.offset > 0
            )
        else:
            result = await self.session.execute(query)
            return result.scalars().all()

    def _build_filter_condition(self, field: str, operator: str, value: Any):
        """Build SQLAlchemy filter condition based on operator."""
        field_attr = getattr(self.model, field)
        
        if operator == 'eq':
            return field_attr == value
        elif operator == 'ne':
            return field_attr != value
        elif operator == 'gt':
            return field_attr > value
        elif operator == 'gte':
            return field_attr >= value
        elif operator == 'lt':
            return field_attr < value
        elif operator == 'lte':
            return field_attr <= value
        elif operator == 'like':
            return field_attr.like(value)
        elif operator == 'ilike':
            return field_attr.ilike(value)
        elif operator == 'in':
            return field_attr.in_(value)
        elif operator == 'not_in':
            return ~field_attr.in_(value)
        elif operator == 'is_null':
            return field_attr.is_(None) if value else field_attr.is_not(None)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    async def add_all(self, entities: List[T], batch_size: int = 1000, **kwargs) -> List[T]:
        """Bulk insert entities with batching.
        
        Args:
            entities: List of entities to insert
            batch_size: Number of entities per batch
            **kwargs: Additional context (e.g., user_id for audit)
            
        Returns:
            List of inserted entities with generated IDs
            
        Raises:
            IntegrityError: If entities violate constraints
            OperationalError: If database operation fails
        """
        operation_start = time.time()
        
        try:
            if self.enable_audit:
                self.log.info("Bulk creating entities", 
                           model=self.model_name, 
                           count=len(entities))
            
            results = []
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                batch_results = await self.circuit_breaker.call(self._add_all_internal, batch, **kwargs)
                results.extend(batch_results)
            
            # Metrics
            self.metrics.query_counter.labels(operation='add_all', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='add_all', model=self.model_name).observe(duration)
            
            self.log.info("Bulk entities created successfully", 
                        model=self.model_name,
                        count=len(entities),
                        duration_ms=duration * 1000)
            
            return results
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='add_all', error_type=type(e).__name__).inc()
            self.log.error("Error bulk creating entities", 
                        model=self.model_name,
                        count=len(entities),
                        error=str(e))
            raise

    async def _add_all_internal(self, entities: List[T], **kwargs) -> List[T]:
        """Internal bulk add method with audit field handling."""
        current_time = datetime.now(timezone.utc)
        user_id = kwargs.get('user_id')
        
        for entity in entities:
            if self.enable_audit and hasattr(entity, 'created_at'):
                entity.created_at = current_time
            if hasattr(entity, 'created_by'):
                entity.created_by = user_id
        
        self.session.add_all(entities)
        await self.session.flush()
        
        # Refresh all entities to get generated IDs
        for entity in entities:
            await self.session.refresh(entity)
        
        return entities

    async def delete_all(self, filters: Dict[str, Any], hard_delete: bool = False, **kwargs) -> int:
        """Bulk delete entities matching filters.
        
        Args:
            filters: Dictionary of filters to match entities for deletion
            hard_delete: Force hard delete even if soft delete is enabled
            **kwargs: Additional context (e.g., user_id for audit)
            
        Returns:
            Number of entities deleted
            
        Raises:
            ValueError: If no filters provided (safety measure)
            OperationalError: If database operation fails
        """
        if not filters:
            raise ValueError("Filters are required for bulk delete (safety measure)")
        
        operation_start = time.time()
        
        try:
            if self.enable_audit:
                self.log.info("Bulk deleting entities", 
                           model=self.model_name, 
                           filters=filters,
                           hard_delete=hard_delete)
            
            result = await self.circuit_breaker.call(self._delete_all_internal, filters, hard_delete, **kwargs)
            
            # Metrics
            self.metrics.query_counter.labels(operation='delete_all', model=self.model_name).inc()
            duration = time.time() - operation_start
            self.metrics.query_duration.labels(operation='delete_all', model=self.model_name).observe(duration)
            
            self.log.info("Bulk entities deleted successfully", 
                        model=self.model_name,
                        count=result,
                        duration_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            self.metrics.error_counter.labels(operation='delete_all', error_type=type(e).__name__).inc()
            self.log.error("Error bulk deleting entities", 
                        model=self.model_name,
                        filters=filters,
                        error=str(e))
            raise

    async def _delete_all_internal(self, filters: Dict[str, Any], hard_delete: bool = False, **kwargs) -> int:
        """Internal bulk delete method."""
        if not hard_delete and self.enable_soft_delete:
            # Soft delete with update
            update_values = {}
            if hasattr(self.model, 'deleted_at'):
                update_values['deleted_at'] = datetime.now(timezone.utc)
            elif hasattr(self.model, 'is_active'):
                update_values['is_active'] = False
            
            if hasattr(self.model, 'deleted_by'):
                update_values['deleted_by'] = kwargs.get('user_id')
            
            if update_values:
                # Build where conditions
                conditions = []
                for field, value in filters.items():
                    conditions.append(getattr(self.model, field) == value)
                
                result = await self.session.execute(
                    update(self.model).where(and_(*conditions)).values(update_values)
                )
                await self.session.flush()
                return result.rowcount
        
        # Hard delete
        conditions = []
        for field, value in filters.items():
            conditions.append(getattr(self.model, field) == value)
        
        result = await self.session.execute(
            delete(self.model).where(and_(*conditions))
        )
        await self.session.flush()
        return result.rowcount

    async def get_by(self, **filters) -> Optional[T]:
        """Get single entity by field values.
        
        Args:
            **filters: Field filters as keyword arguments
            
        Returns:
            First matching entity or None
            
        Raises:
            OperationalError: If database operation fails
        """
        try:
            query = select(self.model)
            conditions = []
            
            for field, value in filters.items():
                if not hasattr(self.model, field):
                    raise ValueError(f"Field '{field}' does not exist on model {self.model_name}")
                conditions.append(getattr(self.model, field) == value)
            
            # Soft delete handling
            if self.enable_soft_delete:
                if hasattr(self.model, 'deleted_at'):
                    conditions.append(self.model.deleted_at.is_(None))
                elif hasattr(self.model, 'is_active'):
                    conditions.append(self.model.is_active == True)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            entity = result.scalar_one_or_none()
            
            self.log.debug("Entity retrieved by filters", 
                         model=self.model_name,
                         filters=filters,
                         found=entity is not None)
            
            return entity
            
        except Exception as e:
            self.log.error("Error getting entity by filters", 
                        model=self.model_name,
                        filters=filters,
                        error=str(e))
            raise

    async def exists(self, **filters) -> bool:
        """Check if entity exists with given filters.
        
        Args:
            **filters: Field filters as keyword arguments
            
        Returns:
            True if entity exists, False otherwise
            
        Raises:
            OperationalError: If database operation fails
        """
        try:
            query = select(func.count(self.model.id))
            conditions = []
            
            for field, value in filters.items():
                if not hasattr(self.model, field):
                    raise ValueError(f"Field '{field}' does not exist on model {self.model_name}")
                conditions.append(getattr(self.model, field) == value)
            
            # Soft delete handling
            if self.enable_soft_delete:
                if hasattr(self.model, 'deleted_at'):
                    conditions.append(self.model.deleted_at.is_(None))
                elif hasattr(self.model, 'is_active'):
                    conditions.append(self.model.is_active == True)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            exists = count > 0
            self.log.debug("Entity existence check", 
                         model=self.model_name,
                         filters=filters,
                         exists=exists)
            
            return exists
            
        except Exception as e:
            self.log.error("Error checking entity existence", 
                        model=self.model_name,
                        filters=filters,
                        error=str(e))
            raise

    async def count(self, filters: Optional[Dict[str, Any]] = None, include_deleted: bool = False) -> int:
        """Count entities with optional filters.
        
        Args:
            filters: Optional dictionary of filters
            include_deleted: Include soft-deleted entities in count
            
        Returns:
            Number of matching entities
            
        Raises:
            OperationalError: If database operation fails
        """
        try:
            query = select(func.count(self.model.id))
            conditions = []
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if not hasattr(self.model, field):
                        raise ValueError(f"Field '{field}' does not exist on model {self.model_name}")
                    conditions.append(getattr(self.model, field) == value)
            
            # Soft delete handling
            if self.enable_soft_delete and not include_deleted:
                if hasattr(self.model, 'deleted_at'):
                    conditions.append(self.model.deleted_at.is_(None))
                elif hasattr(self.model, 'is_active'):
                    conditions.append(self.model.is_active == True)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            self.log.debug("Entity count", 
                         model=self.model_name,
                         filters=filters,
                         count=count)
            
            return count
            
        except Exception as e:
            self.log.error("Error counting entities", 
                        model=self.model_name,
                        filters=filters,
                        error=str(e))
            raise

    async def commit_and_refresh(self, entity: T) -> T:
        """Helper method to commit transaction and refresh entity.
        
        Note: Use this only when transaction management is delegated to repository.
        For most cases, commit should be handled at service/unit-of-work layer.
        
        Args:
            entity: Entity to commit and refresh
            
        Returns:
            Refreshed entity
            
        Raises:
            OperationalError: If commit fails
        """
        try:
            await self.session.commit()
            await self.session.refresh(entity)
            
            self.log.debug("Entity committed and refreshed", 
                         model=self.model_name,
                         entity_id=getattr(entity, 'id', None))
            
            return entity
            
        except Exception as e:
            await self.session.rollback()
            self.log.error("Error committing entity", 
                        model=self.model_name,
                        entity_id=getattr(entity, 'id', None),
                        error=str(e))
            raise

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for audit logging.
        
        Args:
            entity: Entity to convert
            
        Returns:
            Dictionary representation of entity
        """
        try:
            if hasattr(entity, '__dict__'):
                return {k: str(v) for k, v in entity.__dict__.items() if not k.startswith('_')}
            return {"entity": str(entity)}
        except Exception:
            return {"entity": "serialization_failed"}

    async def health_check(self) -> Dict[str, Any]:
        """Repository health check.
        
        Returns:
            Health status dictionary
        """
        try:
            # Test basic connectivity
            test_query = select(func.count(self.model.id))
            await self.session.execute(test_query)
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "circuit_breaker_state": self.circuit_breaker.state,
                "cache_strategy": self.cache_manager.strategy.value,
                "soft_delete_enabled": self.enable_soft_delete,
                "audit_enabled": self.enable_audit,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_test_factory(self):
        """Get test factory for creating test entities.
        
        Returns:
            Test factory function
        """
        def create_test_entity(**kwargs):
            """Create test entity with default values.
            
            Args:
                **kwargs: Field overrides
                
            Returns:
                Test entity instance
            """
            defaults = {
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'is_active': True
            }
            defaults.update(kwargs)
            
            # Remove fields that don't exist on the model
            valid_fields = {k: v for k, v in defaults.items() if hasattr(self.model, k)}
            
            return self.model(**valid_fields)
        
        return create_test_entity

    async def clear_cache(self) -> None:
        """Clear repository cache."""
        await self.cache_manager.clear()
        self.log.info("Repository cache cleared", model=self.model_name)

    def get_metrics(self) -> Dict[str, Any]:
        """Get repository metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "model": self.model_name,
            "cache_strategy": self.cache_manager.strategy.value,
            "circuit_breaker_state": self.circuit_breaker.state,
            "soft_delete_enabled": self.enable_soft_delete,
            "audit_enabled": self.enable_audit,
            "query_counts": dict(self.metrics.query_counter._value._value),
            "error_counts": dict(self.metrics.error_counter._value._value)
        }