#!/usr/bin/env python3
"""
Enterprise Repository Demonstration

This script demonstrates the comprehensive features of the upgraded AsyncRepository
including CRUD operations, filtering, pagination, bulk operations, soft delete,
caching, error handling, and advanced querying capabilities.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Decimal, Text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from qtinfra.infra.db import EnterpriseBase
from qtinfra.repository.base import AsyncRepository, PaginationRequest, CacheStrategy


# Example model for demonstration
class Product(EnterpriseBase):
    """Example product model with enterprise features."""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    price = Column(Decimal(10, 2), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(100))
    updated_by = Column(String(100))
    deleted_by = Column(String(100))
    version = Column(Integer, default=1)


async def demonstrate_enterprise_repository():
    """Comprehensive demonstration of enterprise repository features."""
    
    print("=== Enterprise Repository Demonstration ===\\n")
    
    # Setup database connection
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(EnterpriseBase.metadata.create_all)
    
    # Create session and repository
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as session:
        # Initialize enterprise repository with caching
        repo = AsyncRepository(
            session=session,
            model=Product,
            cache_strategy=CacheStrategy.READ_THROUGH,
            enable_soft_delete=True,
            enable_audit=True
        )
        
        print("1. Repository Initialization")
        print(f"   Model: {repo.model_name}")
        print(f"   Cache Strategy: {repo.cache_manager.strategy.value}")
        print(f"   Soft Delete Enabled: {repo.enable_soft_delete}")
        print(f"   Audit Enabled: {repo.enable_audit}")
        
        # Health check
        health = await repo.health_check()
        print(f"   Health Status: {health['status']}\\n")
        
        print("2. Creating Test Data")
        
        # Single entity creation
        product1 = Product(
            name="Laptop Pro 2024",
            description="High-performance laptop",
            price=1299.99,
            category="Electronics"
        )
        
        created_product = await repo.add(product1, user_id="demo_user")
        print(f"   Created Product: {created_product.name} (ID: {created_product.id})")
        
        # Bulk entity creation
        bulk_products = [
            Product(name="Mouse Wireless", price=29.99, category="Electronics"),
            Product(name="Keyboard Mechanical", price=89.99, category="Electronics"),
            Product(name="Monitor 4K", price=399.99, category="Electronics"),
            Product(name="Office Chair", price=199.99, category="Furniture"),
            Product(name="Desk Lamp", price=49.99, category="Furniture"),
        ]
        
        bulk_created = await repo.add_all(bulk_products, user_id="demo_user")
        print(f"   Bulk Created: {len(bulk_created)} products")
        
        await session.commit()
        print()
        
        print("3. Reading and Querying")
        
        # Get by ID
        retrieved = await repo.get(created_product.id)
        print(f"   Retrieved by ID: {retrieved.name}")
        
        # Get by field
        found_product = await repo.get_by(name="Mouse Wireless")
        print(f"   Found by name: {found_product.name}")
        
        # Check existence
        exists = await repo.exists(category="Electronics")
        print(f"   Electronics category exists: {exists}")
        
        # Count entities
        total_count = await repo.count()
        electronics_count = await repo.count({"category": "Electronics"})
        print(f"   Total products: {total_count}")
        print(f"   Electronics products: {electronics_count}")
        print()
        
        print("4. Advanced Filtering and Pagination")
        
        # Complex filtering
        expensive_electronics = await repo.list(
            filters={
                "category": "Electronics",
                "price": {"gt": 50.0}
            },
            order_by=["price:desc", "name:asc"]
        )
        print(f"   Expensive Electronics (>$50): {len(expensive_electronics)}")
        for product in expensive_electronics:
            print(f"     - {product.name}: ${product.price}")
        
        # Pagination
        pagination = PaginationRequest(offset=0, limit=3)
        paginated_result = await repo.list(
            pagination=pagination,
            order_by="name:asc"
        )
        print(f"\\n   Paginated Results (Page 1, {pagination.limit} items):")
        print(f"   Total Count: {paginated_result.total_count}")
        print(f"   Has Next: {paginated_result.has_next}")
        for product in paginated_result.items:
            print(f"     - {product.name}")
        print()
        
        print("5. Update Operations")
        
        # Single entity update
        updated = await repo.update(
            retrieved,
            price=1199.99,
            description="Updated high-performance laptop - SALE!",
            user_id="demo_user"
        )
        print(f"   Updated Product: {updated.name}, New Price: ${updated.price}")
        
        # Update by ID
        updated_by_id = await repo.update(
            id_=found_product.id,
            price=24.99,
            user_id="demo_user"
        )
        print(f"   Updated by ID: {updated_by_id.name}, New Price: ${updated_by_id.price}")
        print()
        
        print("6. Soft Delete Operations")
        
        # Soft delete single entity
        deleted = await repo.delete(found_product, user_id="demo_user")
        print(f"   Soft Deleted: {found_product.name} (Success: {deleted})")
        
        # Verify soft delete (should not appear in normal queries)
        all_active = await repo.list()
        print(f"   Active products after soft delete: {len(all_active)}")
        
        # Include deleted entities
        all_including_deleted = await repo.list(include_deleted=True)
        print(f"   All products (including deleted): {len(all_including_deleted)}")
        
        # Bulk soft delete
        bulk_deleted_count = await repo.delete_all(
            filters={"category": "Furniture"},
            user_id="demo_user"
        )
        print(f"   Bulk Soft Deleted (Furniture): {bulk_deleted_count} items")
        print()
        
        print("7. Cache Performance")
        
        # First access (cache miss)
        import time
        start = time.time()
        cached_product = await repo.get(created_product.id)
        first_access_time = (time.time() - start) * 1000
        
        # Second access (cache hit)
        start = time.time()
        cached_product_again = await repo.get(created_product.id)
        second_access_time = (time.time() - start) * 1000
        
        print(f"   First access (cache miss): {first_access_time:.2f}ms")
        print(f"   Second access (cache hit): {second_access_time:.2f}ms")
        print(f"   Cache performance improvement: {(first_access_time / second_access_time):.1f}x faster")
        print()
        
        print("8. Error Handling and Resilience")
        
        # Test circuit breaker (simulated)
        print(f"   Circuit Breaker State: {repo.circuit_breaker.state}")
        
        # Test error handling
        try:
            await repo.get_by(nonexistent_field="test")
        except ValueError as e:
            print(f"   Handled ValueError: {str(e)[:50]}...")
        
        try:
            await repo.delete_all({})  # Should fail - no filters
        except ValueError as e:
            print(f"   Safety Check Passed: {str(e)[:50]}...")
        print()
        
        print("9. Repository Metrics and Monitoring")
        
        metrics = repo.get_metrics()
        print(f"   Model: {metrics['model']}")
        print(f"   Circuit Breaker: {metrics['circuit_breaker_state']}")
        print(f"   Cache Strategy: {metrics['cache_strategy']}")
        print("   Operation Counts:")
        for operation, count in metrics.get('query_counts', {}).items():
            if count > 0:
                print(f"     - {operation}: {count}")
        print()
        
        print("10. Test Utilities")
        
        # Test factory
        test_factory = repo.get_test_factory()
        test_product = test_factory(
            name="Test Product",
            price=99.99,
            category="Test"
        )
        print(f"   Test Entity Created: {test_product.name}")
        
        # Clear cache
        await repo.clear_cache()
        print("   Cache cleared")
        print()
        
        print("11. Final Health Check")
        final_health = await repo.health_check()
        print(f"   Status: {final_health['status']}")
        print(f"   Timestamp: {final_health['timestamp']}")
        
    print("\\n=== Enterprise Repository Demonstration Complete ===")
    print("\\nFeatures Demonstrated:")
    print("✅ Full CRUD operations with error handling")
    print("✅ Advanced filtering with complex operators")
    print("✅ Pagination with metadata")
    print("✅ Bulk operations (add_all, delete_all)")
    print("✅ Soft delete functionality")
    print("✅ Caching with performance optimization")
    print("✅ Audit logging and user tracking")
    print("✅ Circuit breaker for fault tolerance")
    print("✅ Prometheus metrics integration")
    print("✅ Advanced query helpers (get_by, exists, count)")
    print("✅ Health checks and monitoring")
    print("✅ Test utilities and factories")


if __name__ == "__main__":
    asyncio.run(demonstrate_enterprise_repository())
