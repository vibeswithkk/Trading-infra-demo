#!/usr/bin/env python3
"""
Comprehensive Enterprise Database Example

Demonstrates all enterprise database features including:
- Model definition with BaseMixin
- Repository pattern implementation
- Circuit breaker and retry logic
- Health monitoring
- PII scrubbing
- OpenTelemetry tracing
- Prometheus metrics
- Connection pooling
- SSL security
- Migration support
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean, Numeric, text
from sqlalchemy.orm import relationship

from qtinfra.infra.db import (
    DatabaseManager, DatabaseConfig, BaseMixin, EnterpriseBase, 
    BaseRepository, get_session, init_db, close_db
)
from qtinfra.infra.logging import EnterpriseLogger

# Set up logging
logger = EnterpriseLogger(__name__, 'enterprise-db-example')


# ============================================================================
# Model Definitions with BaseMixin
# ============================================================================

class User(BaseMixin, EnterpriseBase):
    """User model with enterprise features"""
    __tablename__ = 'users'
    
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='[REDACTED]')>"


class TradingAccount(BaseMixin, EnterpriseBase):
    """Trading account model"""
    __tablename__ = 'trading_accounts'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    account_number = Column(String(50), unique=True, nullable=False, index=True)
    account_type = Column(String(20), nullable=False)  # 'demo', 'live', 'paper'
    balance = Column(Numeric(precision=15, scale=2), nullable=False, default=0)
    currency = Column(String(3), nullable=False, default='USD')
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User")
    trades = relationship("Trade", back_populates="account")
    
    def __repr__(self):
        return f"<TradingAccount(id={self.id}, account_number='[REDACTED]', balance={self.balance})>"


class Trade(BaseMixin, EnterpriseBase):
    """Trade model with full audit trail"""
    __tablename__ = 'trades'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('trading_accounts.id'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)  # 'BUY', 'SELL'
    quantity = Column(Numeric(precision=15, scale=8), nullable=False)
    price = Column(Numeric(precision=15, scale=8), nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), nullable=False, default='PENDING')  # 'PENDING', 'FILLED', 'CANCELLED'
    order_type = Column(String(20), nullable=False)  # 'MARKET', 'LIMIT', 'STOP'
    notes = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    account = relationship("TradingAccount", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', side='{self.side}', quantity={self.quantity})>"


# ============================================================================
# Repository Implementations
# ============================================================================

class UserRepository(BaseRepository):
    """Repository for User operations with enterprise features"""
    
    async def create_user(self, username: str, email: str, full_name: str) -> User:
        """Create a new user with validation"""
        logger.info("Creating new user", username=username, email="[REDACTED]")
        
        # Check for existing user
        existing = await self.get_user_by_username(username)
        if existing:
            raise ValueError(f"User with username '{username}' already exists")
        
        existing_email = await self.get_user_by_email(email)
        if existing_email:
            raise ValueError(f"User with email already exists")
        
        user = await self.create(
            User,
            username=username,
            email=email,
            full_name=full_name
        )
        
        logger.info("User created successfully", user_id=user.id, username=username)
        return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": username}
            )
            row = result.first()
            if row:
                return User(**dict(row._mapping))
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email with PII scrubbing in logs"""
        logger.debug("Looking up user by email", email="[REDACTED]")
        
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
    
    async def update_last_login(self, user_id: int) -> User:
        """Update user's last login timestamp"""
        user = await self.get_by_id(User, user_id)
        if not user:
            raise ValueError(f"User with id {user_id} not found")
        
        return await self.update(user, last_login=datetime.now(timezone.utc))
    
    async def get_active_users(self) -> List[User]:
        """Get all active users"""
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(User).where(User.is_active == True)
            )
            return list(result.scalars())


class TradingAccountRepository(BaseRepository):
    """Repository for trading account operations"""
    
    async def create_account(self, user_id: int, account_type: str, 
                           initial_balance: float = 0.0, currency: str = 'USD') -> TradingAccount:
        """Create a new trading account"""
        # Generate unique account number
        import uuid
        account_number = f"ACC-{uuid.uuid4().hex[:8].upper()}"
        
        logger.info("Creating trading account", 
                   user_id=user_id, 
                   account_type=account_type,
                   account_number="[REDACTED]")
        
        account = await self.create(
            TradingAccount,
            user_id=user_id,
            account_number=account_number,
            account_type=account_type,
            balance=initial_balance,
            currency=currency
        )
        
        logger.info("Trading account created", account_id=account.id)
        return account
    
    async def update_balance(self, account_id: int, new_balance: float) -> TradingAccount:
        """Update account balance with audit logging"""
        account = await self.get_by_id(TradingAccount, account_id)
        if not account:
            raise ValueError(f"Account with id {account_id} not found")
        
        old_balance = account.balance
        updated_account = await self.update(account, balance=new_balance)
        
        logger.info("Account balance updated", 
                   account_id=account_id,
                   old_balance=float(old_balance),
                   new_balance=new_balance)
        
        return updated_account
    
    async def get_accounts_by_user(self, user_id: int) -> List[TradingAccount]:
        """Get all accounts for a user"""
        async with self.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TradingAccount).where(TradingAccount.user_id == user_id)
            )
            return list(result.scalars())


class TradeRepository(BaseRepository):
    """Repository for trade operations with comprehensive auditing"""
    
    async def create_trade(self, user_id: int, account_id: int, symbol: str,
                          side: str, quantity: float, price: float, 
                          order_type: str = 'MARKET', notes: str = None) -> Trade:
        """Create a new trade with full validation"""
        logger.info("Creating trade", 
                   user_id=user_id,
                   account_id=account_id,
                   symbol=symbol,
                   side=side,
                   quantity=quantity,
                   price=price)
        
        # Validate inputs
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side: {side}")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if price <= 0:
            raise ValueError("Price must be positive")
        
        trade = await self.create(
            Trade,
            user_id=user_id,
            account_id=account_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            notes=notes
        )
        
        logger.info("Trade created successfully", trade_id=trade.id)
        return trade
    
    async def execute_trade(self, trade_id: int) -> Trade:
        """Execute a pending trade"""
        trade = await self.get_by_id(Trade, trade_id)
        if not trade:
            raise ValueError(f"Trade with id {trade_id} not found")
        
        if trade.status != 'PENDING':
            raise ValueError(f"Trade {trade_id} is not in PENDING status")
        
        updated_trade = await self.update(
            trade, 
            status='FILLED',
            executed_at=datetime.now(timezone.utc)
        )
        
        logger.info("Trade executed", 
                   trade_id=trade_id,
                   symbol=trade.symbol,
                   side=trade.side,
                   quantity=float(trade.quantity))
        
        return updated_trade
    
    async def get_trades_by_user(self, user_id: int, limit: int = 100) -> List[Trade]:
        """Get trades for a user with limit"""
        async with self.get_session() as session:
            from sqlalchemy import select, desc
            result = await session.execute(
                select(Trade)
                .where(Trade.user_id == user_id)
                .order_by(desc(Trade.created_at))
                .limit(limit)
            )
            return list(result.scalars())


# ============================================================================
# Service Layer with Enterprise Features
# ============================================================================

class TradingService:
    """High-level trading service demonstrating enterprise patterns"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.user_repo = UserRepository(db_manager)
        self.account_repo = TradingAccountRepository(db_manager)
        self.trade_repo = TradeRepository(db_manager)
    
    async def onboard_user(self, username: str, email: str, full_name: str) -> dict:
        """Complete user onboarding with account creation"""
        logger.info("Starting user onboarding", username=username)
        
        try:
            # Create user
            user = await self.user_repo.create_user(username, email, full_name)
            
            # Create demo account
            demo_account = await self.account_repo.create_account(
                user.id, 'demo', initial_balance=10000.0
            )
            
            # Create live account
            live_account = await self.account_repo.create_account(
                user.id, 'live', initial_balance=0.0
            )
            
            logger.info("User onboarding completed", 
                       user_id=user.id,
                       demo_account_id=demo_account.id,
                       live_account_id=live_account.id)
            
            return {
                'user': user,
                'demo_account': demo_account,
                'live_account': live_account
            }
            
        except Exception as e:
            logger.error("User onboarding failed", error=str(e))
            raise
    
    async def place_trade(self, username: str, account_type: str, symbol: str,
                         side: str, quantity: float, price: float) -> Trade:
        """Place a trade with full validation and processing"""
        logger.info("Placing trade", 
                   username=username,
                   symbol=symbol,
                   side=side,
                   quantity=quantity)
        
        # Get user
        user = await self.user_repo.get_user_by_username(username)
        if not user:
            raise ValueError(f"User {username} not found")
        
        # Get account
        accounts = await self.account_repo.get_accounts_by_user(user.id)
        account = next((acc for acc in accounts if acc.account_type == account_type), None)
        if not account:
            raise ValueError(f"No {account_type} account found for user")
        
        # Create trade
        trade = await self.trade_repo.create_trade(
            user.id, account.id, symbol, side, quantity, price
        )
        
        # Execute trade (in real system, this would go through execution engine)
        executed_trade = await self.trade_repo.execute_trade(trade.id)
        
        logger.info("Trade completed", trade_id=executed_trade.id)
        return executed_trade
    
    async def get_user_portfolio(self, username: str) -> dict:
        """Get complete user portfolio information"""
        user = await self.user_repo.get_user_by_username(username)
        if not user:
            raise ValueError(f"User {username} not found")
        
        accounts = await self.account_repo.get_accounts_by_user(user.id)
        trades = await self.trade_repo.get_trades_by_user(user.id, limit=50)
        
        return {
            'user': user,
            'accounts': accounts,
            'recent_trades': trades,
            'total_accounts': len(accounts),
            'total_trades': len(trades)
        }


# ============================================================================
# Example Usage and Demonstration
# ============================================================================

async def demonstrate_enterprise_features():
    """Comprehensive demonstration of all enterprise database features"""
    
    logger.info("Starting enterprise database demonstration")
    
    try:
        # Initialize database
        await init_db()
        
        # Get database manager instance
        db_manager = DatabaseManager()
        
        # Create tables (in real app, use Alembic migrations)
        async with db_manager.engines['primary'].begin() as conn:
            await conn.run_sync(EnterpriseBase.metadata.create_all)
        
        # Demonstrate health monitoring
        logger.info("=== Health Check Demo ===")
        health_status = await db_manager.health_check()
        logger.info("Database health check", status=health_status['status'])
        
        # Get comprehensive database info
        db_info = await db_manager.get_database_info()
        logger.info("Database info retrieved", 
                   uptime=db_info['uptime'],
                   pool_size=db_info['config']['pool_size'])
        
        # Initialize service
        trading_service = TradingService(db_manager)
        
        # Demonstrate user onboarding
        logger.info("=== User Onboarding Demo ===")
        onboarding_result = await trading_service.onboard_user(
            username="trader_john",
            email="john.trader@example.com",
            full_name="John Trader"
        )
        
        user = onboarding_result['user']
        demo_account = onboarding_result['demo_account']
        
        # Demonstrate trade placement
        logger.info("=== Trade Placement Demo ===")
        trade1 = await trading_service.place_trade(
            username="trader_john",
            account_type="demo",
            symbol="EURUSD",
            side="BUY",
            quantity=1000.0,
            price=1.0850
        )
        
        trade2 = await trading_service.place_trade(
            username="trader_john",
            account_type="demo", 
            symbol="GBPUSD",
            side="SELL",
            quantity=500.0,
            price=1.2650
        )
        
        # Demonstrate portfolio retrieval
        logger.info("=== Portfolio Demo ===")
        portfolio = await trading_service.get_user_portfolio("trader_john")
        logger.info("Portfolio retrieved", 
                   accounts=len(portfolio['accounts']),
                   trades=len(portfolio['recent_trades']))
        
        # Demonstrate concurrent operations
        logger.info("=== Concurrent Operations Demo ===")
        
        async def create_user_and_trade(index):
            username = f"trader_{index}"
            email = f"trader{index}@example.com"
            
            result = await trading_service.onboard_user(
                username=username,
                email=email,
                full_name=f"Trader {index}"
            )
            
            trade = await trading_service.place_trade(
                username=username,
                account_type="demo",
                symbol="USDJPY",
                side="BUY" if index % 2 == 0 else "SELL",
                quantity=1000.0,
                price=150.25
            )
            
            return result, trade
        
        # Create multiple users and trades concurrently
        tasks = [create_user_and_trade(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)
        
        logger.info("Concurrent operations completed", users_created=len(results))
        
        # Demonstrate error handling and retry
        logger.info("=== Error Handling Demo ===")
        try:
            # Try to create duplicate user (should fail)
            await trading_service.onboard_user(
                username="trader_john",  # Duplicate
                email="different@example.com",
                full_name="Different User"
            )
        except ValueError as e:
            logger.warning("Expected error caught", error=str(e))
        
        # Demonstrate circuit breaker (simulated)
        logger.info("=== Circuit Breaker Demo ===")
        circuit_breaker_info = {
            'state': db_manager.circuit_breaker.state,
            'failure_count': db_manager.circuit_breaker.failure_count
        }
        logger.info("Circuit breaker status", **circuit_breaker_info)
        
        # Final health check
        final_health = await db_manager.health_check()
        logger.info("Final health check", status=final_health['status'])
        
        logger.info("Enterprise database demonstration completed successfully")
        
    except Exception as e:
        logger.error("Demonstration failed", error=str(e))
        raise
    finally:
        # Clean up
        await close_db()


async def main():
    """Main entry point"""
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set database URL for demo (use SQLite for simplicity)
    os.environ['QTINFRA_DB_URL'] = 'sqlite+aiosqlite:///demo_trading.db'
    
    try:
        await demonstrate_enterprise_features()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())