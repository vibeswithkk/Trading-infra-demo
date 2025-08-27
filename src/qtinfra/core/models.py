"""
Enterprise-grade domain models for Trading Infrastructure.

Implements Domain-Driven Design (DDD) patterns with:
- Rich domain entities with business logic
- Value objects for type safety
- Comprehensive validation
- Audit trails and compliance features
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, DateTime, Numeric, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.hybrid import hybrid_property

from ..infra.db import Base


# === ENUMS ===

class OrderStatus(str, Enum):
    """Order status enumeration with state machine validation."""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    ROUTED = "ROUTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IOC = "IOC"
    FOK = "FOK"
    GTD = "GTD"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT_SELL = "SHORT_SELL"
    BUY_TO_COVER = "BUY_TO_COVER"


class AssetClass(str, Enum):
    """Asset class enumeration."""
    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    FX = "FX"
    COMMODITY = "COMMODITY"
    DERIVATIVE = "DERIVATIVE"
    CRYPTO = "CRYPTO"


class Currency(str, Enum):
    """Currency enumeration."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"


# === VALUE OBJECTS ===

@dataclass(frozen=True)
class Money:
    """Value object for monetary amounts with currency."""
    amount: Decimal
    currency: Currency
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        # Ensure proper decimal precision
        object.__setattr__(self, 'amount', self.amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency.value}"


# === AUDIT MIXIN ===

class AuditMixin:
    """Mixin for audit fields."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Soft delete support
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    deleted_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)



# === DOMAIN ENTITIES ===

class Order(Base, AuditMixin):
    """Enterprise-grade Order entity with rich domain logic."""
    __tablename__ = "orders"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Core order fields
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    parent_order_id: Mapped[Optional[str]] = mapped_column(
        String(36), 
        ForeignKey("orders.id"), 
        nullable=True
    )
    
    # Symbol and asset information
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    exchange: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    
    # Order characteristics
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=OrderStatus.PENDING.value, nullable=False, index=True)
    
    # Quantities and prices
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    remaining_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    stop_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    average_fill_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Currency and notional
    currency: Mapped[str] = mapped_column(String(3), default=Currency.USD.value, nullable=False)
    notional_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    
    # Time-based fields
    time_in_force: Mapped[str] = mapped_column(String(16), default="GTC", nullable=False)
    expiration_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Routing and execution
    routing_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    execution_algorithm: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    target_venue: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Execution tracking
    routed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    first_fill_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_fill_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Compliance and risk
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Fees and costs
    commission: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    fees: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    slippage: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Metadata
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    external_order_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # JSON fields for complex data
    routing_decision: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    execution_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    compliance_checks: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    risk_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('quantity > 0', name='check_positive_quantity'),
        CheckConstraint('filled_quantity >= 0', name='check_non_negative_filled_quantity'),
        CheckConstraint('filled_quantity <= quantity', name='check_filled_not_exceeds_quantity'),
        CheckConstraint('price IS NULL OR price > 0', name='check_positive_price'),
        CheckConstraint('stop_price IS NULL OR stop_price > 0', name='check_positive_stop_price'),
        Index('idx_orders_client_status', 'client_id', 'status'),
        Index('idx_orders_symbol_created', 'symbol', 'created_at'),
        Index('idx_orders_status_created', 'status', 'created_at'),
        UniqueConstraint('external_order_id', name='uq_external_order_id'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.remaining_quantity:
            self.remaining_quantity = self.quantity
    
    # === DOMAIN LOGIC ===
    
    @hybrid_property
    def is_terminal_state(self) -> bool:
        """Check if order is in a terminal state."""
        terminal_states = {
            OrderStatus.FILLED.value,
            OrderStatus.SETTLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value
        }
        return self.status in terminal_states
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if order is active (can be traded)."""
        active_states = {
            OrderStatus.PENDING.value,
            OrderStatus.VALIDATED.value,
            OrderStatus.ROUTED.value,
            OrderStatus.PARTIALLY_FILLED.value
        }
        return self.status in active_states
    
    @hybrid_property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return Decimal('0')
        return (self.filled_quantity / self.quantity) * Decimal('100')
    
    def validate_business_rules(self) -> List[str]:
        """Validate business rules and return list of violations."""
        violations = []
        
        # Quantity validations
        if self.quantity <= 0:
            violations.append("Quantity must be positive")
        
        if self.filled_quantity < 0:
            violations.append("Filled quantity cannot be negative")
        
        if self.filled_quantity > self.quantity:
            violations.append("Filled quantity cannot exceed order quantity")
        
        # Price validations
        if self.order_type in [OrderType.LIMIT.value, OrderType.STOP_LIMIT.value] and not self.price:
            violations.append(f"Price required for {self.order_type} orders")
        
        if self.price and self.price <= 0:
            violations.append("Price must be positive")
        
        # Time validations
        if self.time_in_force == "GTD" and not self.expiration_date:
            violations.append("Expiration date required for GTD orders")
        
        if self.expiration_date and self.expiration_date <= datetime.now(timezone.utc):
            violations.append("Expiration date must be in the future")
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for API responses."""
        return {
            "id": self.id,
            "client_id": self.client_id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "order_type": self.order_type,
            "side": self.side,
            "status": self.status,
            "quantity": float(self.quantity),
            "filled_quantity": float(self.filled_quantity),
            "remaining_quantity": float(self.remaining_quantity),
            "price": float(self.price) if self.price else None,
            "average_fill_price": float(self.average_fill_price) if self.average_fill_price else None,
            "currency": self.currency,
            "time_in_force": self.time_in_force,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fill_percentage": float(self.fill_percentage),
            "is_active": self.is_active,
            "is_terminal_state": self.is_terminal_state
        }


class Execution(Base, AuditMixin):
    """Trade execution entity."""
    __tablename__ = "executions"
    
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    order_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("orders.id"), 
        nullable=False,
        index=True
    )
    
    # Execution details
    venue_id: Mapped[str] = mapped_column(String(64), nullable=False)
    venue_order_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    execution_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    executed_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    executed_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    execution_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    
    # Costs and fees
    commission: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    slippage: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Market data at execution
    market_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    bid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    ask_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Execution metadata
    liquidity_flag: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # MAKER/TAKER
    execution_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    contra_party: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('executed_quantity > 0', name='check_positive_executed_quantity'),
        CheckConstraint('executed_price > 0', name='check_positive_executed_price'),
        CheckConstraint('commission >= 0', name='check_non_negative_commission'),
        CheckConstraint('fees >= 0', name='check_non_negative_fees'),
        Index('idx_executions_order_time', 'order_id', 'execution_time'),
        Index('idx_executions_venue_time', 'venue_id', 'execution_time'),
    )


# Keep the original Trade model for backward compatibility
class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20,8))
    price: Mapped[Decimal] = mapped_column(Numeric(20,8))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)