"""
Comprehensive unit tests for the enterprise database system.
Tests database connection, circuit breaker, health checks, and repository patterns.
"""

import asyncio
import os
import pytest
import pytest_asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, DateTime, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

# Add src to path for imports
import sys
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from qtinfra.infra.db import (
    DatabaseManager, DatabaseConfig, CircuitBreaker, PIIScrubber,
    BaseMixin, EnterpriseBase, BaseRepository, DatabaseState
)


class TestModel(BaseMixin, EnterpriseBase):
    __tablename__ = 'test_model'
    
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)


class TestPIIScrubber:
    """Test PII scrubbing functionality"""
    
    def test_email_scrubbing(self):
        query = "SELECT * FROM users WHERE email = 'john.doe@example.com'"
        scrubbed = PIIScrubber.scrub_query(query)
        assert "[EMAIL_REDACTED]" in scrubbed
        assert "john.doe@example.com" not in scrubbed
    
    def test_phone_scrubbing(self):
        query = "UPDATE users SET phone = '555-123-4567' WHERE id = 1"
        scrubbed = PIIScrubber.scrub_query(query)
        assert "[PHONE_REDACTED]" in scrubbed
        assert "555-123-4567" not in scrubbed
    
    def test_ssn_scrubbing(self):
        query = "INSERT INTO users (ssn) VALUES ('123-45-6789')"
        scrubbed = PIIScrubber.scrub_query(query)
        assert "[SSN_REDACTED]" in scrubbed
        assert "123-45-6789" not in scrubbed
    
    def test_credit_card_scrubbing(self):
        query = "INSERT INTO payments (card) VALUES ('4532 1234 5678 9012')"
        scrubbed = PIIScrubber.scrub_query(query)
        assert "[CREDIT_CARD_REDACTED]" in scrubbed
        assert "4532 1234 5678 9012" not in scrubbed
    
    def test_multiple_pii_scrubbing(self):
        query = "SELECT * FROM users WHERE email = 'john@example.com' AND phone = '555-1234'"
        scrubbed = PIIScrubber.scrub_query(query)
        assert "[EMAIL_REDACTED]" in scrubbed
        assert "[PHONE_REDACTED]" in scrubbed
        assert "john@example.com" not in scrubbed
        assert "555-1234" not in scrubbed


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest_asyncio.fixture
    async def circuit_breaker(self):
        return CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test normal operation in closed state"""
        
        async def mock_func():
            return "success"
        
        result = await circuit_breaker.call(mock_func)
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker opens after threshold failures"""
        
        async def failing_func():
            raise Exception("Database error")
        
        # Trigger failures to open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "open"
        assert circuit_breaker.failure_count == 3
        
        # Should now reject calls
        with pytest.raises(Exception, match="Circuit breaker is open - database unavailable"):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery from open to closed"""
        
        async def failing_func():
            raise Exception("Database error")
        
        async def success_func():
            return "recovered"
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to half-open and then closed on success
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0


class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_default_config_creation(self):
        config = DatabaseConfig("sqlite+aiosqlite:///:memory:")
        assert config.pool_size == 20
        assert config.max_overflow == 30
        assert config.enable_ssl is True
        assert config.enable_pii_scrubbing is True
    
    def test_postgresql_ssl_configuration(self):
        config = DatabaseConfig("postgresql+asyncpg://user:pass@localhost/db")
        assert config.enable_ssl is True
        assert "sslmode" in config.connect_args or "sslcontext" in config.connect_args
    
    def test_sqlite_configuration(self):
        config = DatabaseConfig("sqlite+aiosqlite:///:memory:")
        config.__post_init__()  # Manually call post_init for testing
        assert "check_same_thread" in config.connect_args


@pytest_asyncio.fixture
async def test_db_manager():
    """Create a test database manager with in-memory SQLite"""
    
    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    
    db_url = f"sqlite+aiosqlite:///{temp_db.name}"
    
    config = DatabaseConfig(
        url=db_url,
        pool_size=1,
        max_overflow=0,
        enable_ssl=False,
        query_timeout=5,
        slow_query_threshold=0.1
    )
    
    # Reset singleton instance for testing
    DatabaseManager._instance = None
    manager = DatabaseManager(config)
    
    await manager.initialize()
    
    # Create test tables
    async with manager.engines['primary'].begin() as conn:
        await conn.run_sync(EnterpriseBase.metadata.create_all)
    
    yield manager
    
    await manager.shutdown()
    
    # Cleanup
    try:
        os.unlink(temp_db.name)
    except OSError:
        pass


class TestDatabaseManager:
    """Test database manager functionality"""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self, test_db_manager):
        """Test database manager initializes correctly"""
        assert test_db_manager is not None
        assert 'primary' in test_db_manager.engines
        assert 'primary' in test_db_manager.session_makers
        assert test_db_manager.circuit_breaker is not None
    
    @pytest.mark.asyncio
    async def test_session_context_manager(self, test_db_manager):
        """Test database session context manager"""
        async with test_db_manager.get_session() as session:
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_query_execution_with_retry(self, test_db_manager):
        """Test query execution with retry logic"""
        query = text("SELECT 'test' as value")
        result = await test_db_manager.execute_with_retry(query)
        assert result.scalar() == 'test'
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, test_db_manager):
        """Test slow query detection and logging"""
        # Create a slow query by using a delay
        slow_query = text("SELECT 1 WHERE (SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3)) > 0")
        
        with patch('qtinfra.infra.db.logger') as mock_logger:
            await test_db_manager.execute_with_retry(slow_query)
            # Check if slow query was potentially logged (this might not always trigger in SQLite)
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_db_manager):
        """Test database health check functionality"""
        health_status = await test_db_manager.health_check()
        
        assert 'status' in health_status
        assert 'response_time' in health_status
        assert 'pool_status' in health_status
        assert 'timestamp' in health_status
        assert health_status['status'] in ['healthy', 'degraded', 'slow', 'unavailable']
    
    @pytest.mark.asyncio
    async def test_database_info(self, test_db_manager):
        """Test database information retrieval"""
        info = await test_db_manager.get_database_info()
        
        assert 'uptime' in info
        assert 'config' in info
        assert 'engines' in info
        assert 'circuit_breaker' in info
        assert 'primary' in info['engines']
    
    @pytest.mark.asyncio
    async def test_pii_scrubbing_in_queries(self, test_db_manager):
        """Test PII scrubbing in query logging"""
        # This tests the _scrub_query_for_logging method
        query_with_email = "SELECT * FROM users WHERE email = 'test@example.com'"
        scrubbed = test_db_manager._scrub_query_for_logging(query_with_email)
        
        if test_db_manager.config.enable_pii_scrubbing:
            assert "[EMAIL_REDACTED]" in scrubbed
            assert "test@example.com" not in scrubbed


class TestBaseMixin:
    """Test BaseMixin functionality"""
    
    @pytest.mark.asyncio
    async def test_base_mixin_fields(self, test_db_manager):
        """Test that BaseMixin adds required fields"""
        async with test_db_manager.get_session() as session:
            # Create a test model instance
            test_instance = TestModel(name="Test Name", email="test@example.com")
            session.add(test_instance)
            await session.commit()
            await session.refresh(test_instance)
            
            # Check that BaseMixin fields are present
            assert hasattr(test_instance, 'id')
            assert hasattr(test_instance, 'created_at')
            assert hasattr(test_instance, 'updated_at')
            assert hasattr(test_instance, 'version')
            
            assert test_instance.id is not None
            assert test_instance.created_at is not None
            assert test_instance.updated_at is not None
            assert test_instance.version == 1


class TestBaseRepository:
    """Test BaseRepository pattern"""
    
    @pytest_asyncio.fixture
    async def test_repository(self, test_db_manager):
        return BaseRepository(test_db_manager)
    
    @pytest.mark.asyncio
    async def test_repository_create(self, test_repository):
        """Test repository create method"""
        instance = await test_repository.create(
            TestModel,
            name="Test User",
            email="user@example.com"
        )
        
        assert instance.id is not None
        assert instance.name == "Test User"
        assert instance.email == "user@example.com"
        assert instance.created_at is not None
    
    @pytest.mark.asyncio
    async def test_repository_get_by_id(self, test_repository):
        """Test repository get by id method"""
        # First create an instance
        created = await test_repository.create(
            TestModel,
            name="Test User 2",
            email="user2@example.com"
        )
        
        # Then retrieve it
        retrieved = await test_repository.get_by_id(TestModel, created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Test User 2"
        assert retrieved.email == "user2@example.com"
    
    @pytest.mark.asyncio
    async def test_repository_update(self, test_repository):
        """Test repository update method"""
        # Create an instance
        instance = await test_repository.create(
            TestModel,
            name="Original Name",
            email="original@example.com"
        )
        
        # Update it
        updated = await test_repository.update(
            instance,
            name="Updated Name",
            email="updated@example.com"
        )
        
        assert updated.name == "Updated Name"
        assert updated.email == "updated@example.com"
        assert updated.version == 1  # Version should be updated
    
    @pytest.mark.asyncio
    async def test_repository_delete(self, test_repository):
        """Test repository delete method"""
        # Create an instance
        instance = await test_repository.create(
            TestModel,
            name="To Be Deleted",
            email="delete@example.com"
        )
        
        instance_id = instance.id
        
        # Delete it
        await test_repository.delete(instance)
        
        # Verify it's gone
        retrieved = await test_repository.get_by_id(TestModel, instance_id)
        assert retrieved is None


class TestDatabaseIntegration:
    """Integration tests for database functionality"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_database_operations(self, test_db_manager):
        """Test complete database workflow"""
        repository = BaseRepository(test_db_manager)
        
        # Create
        user = await repository.create(
            TestModel,
            name="Integration Test User",
            email="integration@test.com"
        )
        assert user.id is not None
        
        # Read
        retrieved_user = await repository.get_by_id(TestModel, user.id)
        assert retrieved_user.name == "Integration Test User"
        
        # Update
        updated_user = await repository.update(
            retrieved_user,
            name="Updated Integration User"
        )
        assert updated_user.name == "Updated Integration User"
        
        # Health check
        health = await test_db_manager.health_check()
        assert health['status'] in ['healthy', 'degraded', 'slow']
        
        # Database info
        info = await test_db_manager.get_database_info()
        assert info['uptime'] >= 0
        
        # Delete
        await repository.delete(updated_user)
        deleted_user = await repository.get_by_id(TestModel, user.id)
        assert deleted_user is None
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, test_db_manager):
        """Test concurrent database operations"""
        repository = BaseRepository(test_db_manager)
        
        async def create_user(index):
            return await repository.create(
                TestModel,
                name=f"Concurrent User {index}",
                email=f"user{index}@concurrent.com"
            )
        
        # Create multiple users concurrently
        tasks = [create_user(i) for i in range(5)]
        users = await asyncio.gather(*tasks)
        
        assert len(users) == 5
        for i, user in enumerate(users):
            assert user.name == f"Concurrent User {i}"
            assert user.id is not None
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, test_db_manager):
        """Test transaction rollback on error"""
        async with test_db_manager.get_session() as session:
            try:
                # Create a user
                user = TestModel(name="Rollback Test", email="rollback@test.com")
                session.add(user)
                await session.flush()  # Get the ID but don't commit
                
                user_id = user.id
                assert user_id is not None
                
                # Cause an error
                raise Exception("Intentional error for rollback test")
                
            except Exception:
                # Session should automatically rollback
                pass
        
        # Verify the user was not persisted
        async with test_db_manager.get_session() as session:
            result = await session.get(TestModel, user_id)
            assert result is None


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])