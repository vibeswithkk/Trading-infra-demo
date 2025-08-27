"""
Enterprise-grade validation layer using Pydantic.

Provides comprehensive input validation, business rule enforcement,
and API contract definitions with:
- Strong type safety
- Business rule validation
- Automatic error message generation
- Integration with OpenAPI/FastAPI
- Performance optimizations
"""

from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum

from pydantic import (
    BaseModel, Field, field_validator, model_validator, constr, 
    condecimal, conint, EmailStr, HttpUrl
)
from pydantic.types import PositiveInt, NonNegativeInt, PositiveFloat

from .models import OrderType, OrderSide, AssetClass, Currency, OrderStatus
from .exceptions import OrderValidationError, RiskLimitExceededError


# === BASE VALIDATION MODELS ===

class BaseValidationModel(BaseModel):
    """Base model with common validation configurations."""
    
    class Config:
        # Validation settings
        validate_assignment = True
        populate_by_name = True
        use_enum_values = True
        
        # JSON settings
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }
        
        # Schema settings
        json_schema_extra = {
            "examples": []
        }


# === CORE VALIDATION SCHEMAS ===

class OrderCreateRequest(BaseValidationModel):
    """Request schema for creating orders with comprehensive validation."""
    
    # Core order fields
    client_id: constr(min_length=1, max_length=255, strip_whitespace=True) = Field(
        ...,
        description="Unique client identifier",
        example="CLIENT_001"
    )
    
    symbol: constr(min_length=1, max_length=32, strip_whitespace=True, to_upper=True) = Field(
        ...,
        description="Trading symbol (will be normalized to uppercase)",
        example="AAPL"
    )
    
    asset_class: AssetClass = Field(
        ...,
        description="Asset class for the instrument"
    )
    
    order_type: OrderType = Field(
        ...,
        description="Type of order to place"
    )
    
    side: OrderSide = Field(
        ...,
        description="Order side (BUY/SELL)"
    )
    
    quantity: condecimal(gt=0, max_digits=20, decimal_places=8) = Field(
        ...,
        description="Order quantity (must be positive)",
        example=Decimal("100.0")
    )
    
    # Optional pricing fields
    price: Optional[condecimal(gt=0, max_digits=20, decimal_places=8)] = Field(
        None,
        description="Limit price (required for LIMIT and STOP_LIMIT orders)",
        example=Decimal("150.25")
    )
    
    stop_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=8)] = Field(
        None,
        description="Stop price (required for STOP and STOP_LIMIT orders)",
        example=Decimal("148.00")
    )
    
    # Time and expiration
    time_in_force: Literal["GTC", "GTD", "IOC", "FOK", "DAY"] = Field(
        "GTC",
        description="Time in force for the order"
    )
    
    expiration_date: Optional[datetime] = Field(
        None,
        description="Expiration date (required for GTD orders)"
    )
    
    # Currency and exchange
    currency: Currency = Field(
        Currency.USD,
        description="Order currency"
    )
    
    exchange: Optional[constr(min_length=1, max_length=32, strip_whitespace=True)] = Field(
        None,
        description="Target exchange (optional)",
        example="NASDAQ"
    )
    
    # Execution parameters
    execution_algorithm: Optional[constr(max_length=64)] = Field(
        None,
        description="Execution algorithm preference",
        example="TWAP"
    )
    
    routing_strategy: Optional[constr(max_length=64)] = Field(
        None,
        description="Routing strategy preference",
        example="BEST_EXECUTION"
    )
    
    # Risk and compliance
    bypass_risk_checks: bool = Field(
        False,
        description="Bypass pre-trade risk checks (requires special permission)"
    )
    
    # Metadata
    tags: Optional[List[constr(max_length=50)]] = Field(
        None,
        description="Order tags for classification and tracking",
        max_items=10
    )
    
    notes: Optional[constr(max_length=1000)] = Field(
        None,
        description="Order notes or comments"
    )
    
    external_order_id: Optional[constr(min_length=1, max_length=255)] = Field(
        None,
        description="External system order ID for correlation"
    )
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, v):
        """Validate symbol format."""
        if not re.match(r'^[A-Z0-9._-]+$', v):
            raise ValueError('Symbol must contain only alphanumeric characters, periods, underscores, and hyphens')
        return v
    
    @field_validator('expiration_date')
    @classmethod
    def validate_expiration_date(cls, v, info):
        """Validate expiration date logic."""
        if v and v <= datetime.now(timezone.utc):
            raise ValueError('Expiration date must be in the future')
        
        # Check if GTD requires expiration
        data = info.data if hasattr(info, 'data') else {}
        if data.get('time_in_force') == 'GTD' and not v:
            raise ValueError('GTD orders require an expiration date')
        
        return v
    
    @model_validator(mode='after')
    def validate_price_requirements(self):
        """Validate price requirements based on order type."""
        order_type = self.order_type
        price = self.price
        stop_price = self.stop_price
        
        # Price required for LIMIT and STOP_LIMIT orders
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not price:
            raise ValueError(f'{order_type} orders require a price')
        
        # Stop price required for STOP and STOP_LIMIT orders
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not stop_price:
            raise ValueError(f'{order_type} orders require a stop price')
        
        # Validate price relationship for STOP_LIMIT orders
        if order_type == OrderType.STOP_LIMIT and price and stop_price:
            side = self.side
            if side == OrderSide.BUY and price < stop_price:
                raise ValueError('For BUY STOP_LIMIT orders, price must be >= stop price')
            elif side == OrderSide.SELL and price > stop_price:
                raise ValueError('For SELL STOP_LIMIT orders, price must be <= stop price')
        
        return self
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity_precision(cls, v):
        """Validate quantity precision for different asset classes."""
        # For most asset classes, allow up to 8 decimal places
        if v.as_tuple().exponent < -8:
            raise ValueError('Quantity precision cannot exceed 8 decimal places')
        return v
    
    class Config(BaseValidationModel.Config):
        json_schema_extra = {
            "examples": [
                {
                    "client_id": "CLIENT_001",
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_type": "LIMIT",
                    "side": "BUY",
                    "quantity": "100.0",
                    "price": "150.25",
                    "time_in_force": "GTC",
                    "currency": "USD"
                }
            ]
        }


class OrderUpdateRequest(BaseValidationModel):
    """Request schema for updating orders."""
    
    # Updatable fields
    quantity: Optional[condecimal(gt=0, max_digits=20, decimal_places=8)] = Field(
        None,
        description="New order quantity"
    )
    
    price: Optional[condecimal(gt=0, max_digits=20, decimal_places=8)] = Field(
        None,
        description="New limit price"
    )
    
    stop_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=8)] = Field(
        None,
        description="New stop price"
    )
    
    time_in_force: Optional[Literal["GTC", "GTD", "IOC", "FOK", "DAY"]] = Field(
        None,
        description="New time in force"
    )
    
    expiration_date: Optional[datetime] = Field(
        None,
        description="New expiration date"
    )
    
    tags: Optional[List[constr(max_length=50)]] = Field(
        None,
        description="Updated order tags",
        max_items=10
    )
    
    notes: Optional[constr(max_length=1000)] = Field(
        None,
        description="Updated order notes"
    )


class ExecutionCreateRequest(BaseValidationModel):
    """Request schema for creating executions."""
    
    order_id: constr(min_length=1, max_length=36) = Field(
        ...,
        description="Order ID for the execution"
    )
    
    venue_id: constr(min_length=1, max_length=64) = Field(
        ...,
        description="Execution venue identifier"
    )
    
    executed_quantity: condecimal(gt=0, max_digits=20, decimal_places=8) = Field(
        ...,
        description="Executed quantity"
    )
    
    executed_price: condecimal(gt=0, max_digits=20, decimal_places=8) = Field(
        ...,
        description="Execution price"
    )
    
    execution_time: Optional[datetime] = Field(
        None,
        description="Execution timestamp (defaults to current time)"
    )
    
    # Optional fields
    venue_order_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Venue-specific order ID"
    )
    
    execution_id: Optional[constr(max_length=255)] = Field(
        None,
        description="Venue-specific execution ID"
    )
    
    commission: Optional[condecimal(ge=0, max_digits=20, decimal_places=8)] = Field(
        Decimal('0'),
        description="Commission charged"
    )
    
    fees: Optional[condecimal(ge=0, max_digits=20, decimal_places=8)] = Field(
        Decimal('0'),
        description="Additional fees"
    )
    
    liquidity_flag: Optional[Literal["MAKER", "TAKER", "UNKNOWN"]] = Field(
        None,
        description="Liquidity provision flag"
    )
    
    @field_validator('execution_time')
    @classmethod
    def validate_execution_time(cls, v):
        """Validate execution time is not in the future."""
        if v and v > datetime.now(timezone.utc):
            raise ValueError('Execution time cannot be in the future')
        return v or datetime.now(timezone.utc)


# === QUERY AND FILTER SCHEMAS ===

class OrderQueryRequest(BaseValidationModel):
    """Request schema for querying orders."""
    
    # Filter parameters
    client_id: Optional[str] = Field(None, description="Filter by client ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    asset_class: Optional[AssetClass] = Field(None, description="Filter by asset class")
    status: Optional[OrderStatus] = Field(None, description="Filter by order status")
    
    # Date range filters
    start_date: Optional[datetime] = Field(
        None,
        description="Filter orders created after this date"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Filter orders created before this date"
    )
    
    # Pagination
    page: conint(ge=1) = Field(1, description="Page number (1-based)")
    page_size: conint(ge=1, le=1000) = Field(50, description="Number of items per page")
    
    # Sorting
    sort_by: Optional[Literal["created_at", "updated_at", "symbol", "quantity", "status"]] = Field(
        "created_at",
        description="Field to sort by"
    )
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        "desc",
        description="Sort order"
    )
    
    # Additional filters
    include_inactive: bool = Field(False, description="Include soft-deleted orders")
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """Validate date range logic."""
        data = info.data if hasattr(info, 'data') else {}
        start_date = data.get('start_date')
        if start_date and v and v <= start_date:
            raise ValueError('End date must be after start date')
        return v


class RiskCheckRequest(BaseValidationModel):
    """Request schema for risk checks."""
    
    client_id: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Client ID to check"
    )
    
    symbol: Optional[str] = Field(None, description="Symbol to check")
    asset_class: Optional[AssetClass] = Field(None, description="Asset class to check")
    
    # Position check
    proposed_quantity: Optional[condecimal(max_digits=20, decimal_places=8)] = Field(
        None,
        description="Proposed position change"
    )
    
    # Notional check
    proposed_notional: Optional[condecimal(max_digits=20, decimal_places=2)] = Field(
        None,
        description="Proposed notional value"
    )
    
    currency: Currency = Field(Currency.USD, description="Currency for calculations")
    
    # Check types
    check_position_limits: bool = Field(True, description="Check position limits")
    check_concentration_limits: bool = Field(True, description="Check concentration limits")
    check_var_limits: bool = Field(True, description="Check VaR limits")
    check_regulatory_rules: bool = Field(True, description="Check regulatory compliance")


# === RESPONSE SCHEMAS ===

class OrderResponse(BaseValidationModel):
    """Response schema for order data."""
    
    id: str = Field(..., description="Order ID")
    client_id: str = Field(..., description="Client ID")
    symbol: str = Field(..., description="Trading symbol")
    asset_class: str = Field(..., description="Asset class")
    order_type: str = Field(..., description="Order type")
    side: str = Field(..., description="Order side")
    status: str = Field(..., description="Order status")
    
    # Quantities and prices
    quantity: Decimal = Field(..., description="Order quantity")
    filled_quantity: Decimal = Field(..., description="Filled quantity")
    remaining_quantity: Decimal = Field(..., description="Remaining quantity")
    price: Optional[Decimal] = Field(None, description="Order price")
    average_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    
    # Metadata
    currency: str = Field(..., description="Order currency")
    time_in_force: str = Field(..., description="Time in force")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Calculated fields
    fill_percentage: float = Field(..., description="Fill percentage")
    is_active: bool = Field(..., description="Whether order is active")
    is_terminal_state: bool = Field(..., description="Whether order is in terminal state")


class ExecutionResponse(BaseValidationModel):
    """Response schema for execution data."""
    
    id: str = Field(..., description="Execution ID")
    order_id: str = Field(..., description="Order ID")
    venue_id: str = Field(..., description="Venue ID")
    
    executed_quantity: Decimal = Field(..., description="Executed quantity")
    executed_price: Decimal = Field(..., description="Execution price")
    execution_time: datetime = Field(..., description="Execution timestamp")
    
    commission: Decimal = Field(..., description="Commission")
    fees: Decimal = Field(..., description="Fees")
    
    created_at: datetime = Field(..., description="Record creation timestamp")


class ValidationErrorResponse(BaseValidationModel):
    """Standardized validation error response."""
    
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    field_errors: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Field-specific validation errors"
    )
    code: Optional[str] = Field(None, description="Error code")


class PaginatedResponse(BaseValidationModel):
    """Generic paginated response wrapper."""
    
    items: List[Any] = Field(..., description="Response items")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


# === BUSINESS RULE VALIDATORS ===

class BusinessRuleValidator:
    """Centralized business rule validation."""
    
    @staticmethod
    def validate_order_create(request: OrderCreateRequest, context: Dict[str, Any] = None) -> List[str]:
        """Validate order creation business rules."""
        violations = []
        context = context or {}
        
        # Market hours validation
        if context.get('check_market_hours', True):
            # Simplified market hours check (would be more complex in production)
            current_time = datetime.now(timezone.utc)
            if current_time.weekday() >= 5:  # Weekend
                violations.append("Orders cannot be placed during weekends")
        
        # Minimum order size validation
        min_order_size = context.get('min_order_size', {}).get(request.symbol, Decimal('1'))
        if request.quantity < min_order_size:
            violations.append(f"Order quantity {request.quantity} is below minimum size {min_order_size}")
        
        # Maximum order size validation
        max_order_size = context.get('max_order_size', {}).get(request.symbol, Decimal('1000000'))
        if request.quantity > max_order_size:
            violations.append(f"Order quantity {request.quantity} exceeds maximum size {max_order_size}")
        
        # Asset-specific validations
        if request.asset_class == AssetClass.CRYPTO:
            # Crypto-specific rules
            if request.quantity < Decimal('0.001'):
                violations.append("Crypto orders must be at least 0.001")
        
        return violations
    
    @staticmethod
    def validate_position_limits(
        client_id: str, 
        symbol: str, 
        proposed_quantity: Decimal,
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Validate position limits."""
        violations = []
        context = context or {}
        
        # Get current position (would query database in real implementation)
        current_position = context.get('current_position', Decimal('0'))
        new_position = current_position + proposed_quantity
        
        # Get position limit (would query database in real implementation)
        position_limit = context.get('position_limit', Decimal('10000'))
        
        if abs(new_position) > position_limit:
            violations.append(
                f"Position limit exceeded: new position {new_position} would exceed limit {position_limit}"
            )
        
        return violations
    
    @staticmethod
    def validate_concentration_limits(
        client_id: str,
        asset_class: AssetClass,
        proposed_notional: Decimal,
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Validate concentration limits."""
        violations = []
        context = context or {}
        
        # Get portfolio value (would query database in real implementation)
        portfolio_value = context.get('portfolio_value', Decimal('1000000'))
        current_concentration = context.get('current_concentration', Decimal('0'))
        
        # Calculate new concentration
        new_concentration = ((current_concentration * portfolio_value) + proposed_notional) / portfolio_value
        
        # Get concentration limit (would query database in real implementation)
        concentration_limit = context.get('concentration_limit', {}).get(asset_class.value, Decimal('0.3'))  # 30%
        
        if new_concentration > concentration_limit:
            violations.append(
                f"Concentration limit exceeded: {asset_class} concentration would be "
                f"{new_concentration:.2%}, limit is {concentration_limit:.2%}"
            )
        
        return violations


# === VALIDATION UTILITIES ===

def create_validation_context(
    client_id: str,
    symbol: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create validation context with client and market data."""
    context = {
        'client_id': client_id,
        'symbol': symbol,
        'timestamp': datetime.now(timezone.utc),
        'check_market_hours': True
    }
    
    if additional_context:
        context.update(additional_context)
    
    return context


def validate_and_raise(
    request: BaseValidationModel,
    business_rules: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Validate request and raise OrderValidationError if violations found."""
    violations = business_rules or []
    
    if violations:
        raise OrderValidationError(
            message=f"Validation failed: {'; '.join(violations)}",
            validation_errors=violations,
            context=context
        )