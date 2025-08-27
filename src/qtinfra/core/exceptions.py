"""
Enterprise-grade custom exception hierarchy for Trading Infrastructure.

This module provides domain-specific exceptions that offer better error handling,
debugging capabilities, and integration with enterprise logging systems.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(str, Enum):
    """Error categories for classification and reporting."""
    VALIDATION = "VALIDATION"
    BUSINESS_RULE = "BUSINESS_RULE"
    TECHNICAL = "TECHNICAL"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"
    INTEGRATION = "INTEGRATION"
    PERFORMANCE = "PERFORMANCE"


@dataclass
class ErrorContext:
    """Enhanced error context for enterprise debugging and monitoring."""
    timestamp: datetime
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: Optional[str] = None
    operation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TradingInfraError(Exception):
    """
    Base exception for all trading infrastructure errors.
    
    Provides enterprise-grade error handling with:
    - Structured error information
    - Contextual metadata
    - Severity classification
    - Category classification
    - Correlation tracking
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.TECHNICAL,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code or self.__class__.__name__
        self.context = context or ErrorContext(timestamp=datetime.utcnow())
        self.cause = cause
        self.user_message = user_message or "An error occurred processing your request"
        self.retry_after = retry_after
        self.metadata = kwargs
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_code": self.error_code,
            "retry_after": self.retry_after,
            "timestamp": self.context.timestamp.isoformat(),
            "correlation_id": self.context.correlation_id,
            "metadata": self.metadata,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# === ORDER-RELATED EXCEPTIONS ===

class OrderError(TradingInfraError):
    """Base exception for order-related errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.BUSINESS_RULE,
            **kwargs
        )
        self.order_id = order_id
        if order_id:
            self.metadata["order_id"] = order_id


class OrderValidationError(OrderError):
    """Exception raised when order validation fails."""
    
    def __init__(
        self, 
        message: str, 
        order_id: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            order_id=order_id,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            user_message="Order validation failed. Please check your order details.",
            **kwargs
        )
        self.validation_errors = validation_errors or []
        self.metadata["validation_errors"] = self.validation_errors


class OrderNotFoundError(OrderError):
    """Exception raised when an order cannot be found."""
    
    def __init__(self, order_id: str, **kwargs):
        super().__init__(
            f"Order not found: {order_id}",
            order_id=order_id,
            severity=ErrorSeverity.LOW,
            user_message=f"Order {order_id} was not found.",
            **kwargs
        )


class OrderStateError(OrderError):
    """Exception raised when order operation is invalid for current state."""
    
    def __init__(
        self, 
        message: str, 
        order_id: str, 
        current_state: str, 
        expected_states: List[str],
        **kwargs
    ):
        super().__init__(
            message,
            order_id=order_id,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Order {order_id} cannot be processed in its current state.",
            **kwargs
        )
        self.current_state = current_state
        self.expected_states = expected_states
        self.metadata.update({
            "current_state": current_state,
            "expected_states": expected_states
        })


class DuplicateOrderError(OrderError):
    """Exception raised when attempting to create a duplicate order."""
    
    def __init__(self, order_id: str, existing_order_id: str, **kwargs):
        super().__init__(
            f"Duplicate order detected: {order_id} (existing: {existing_order_id})",
            order_id=order_id,
            severity=ErrorSeverity.MEDIUM,
            user_message="A similar order already exists.",
            **kwargs
        )
        self.existing_order_id = existing_order_id
        self.metadata["existing_order_id"] = existing_order_id


# === RISK MANAGEMENT EXCEPTIONS ===

class RiskError(TradingInfraError):
    """Base exception for risk management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_RULE,
            **kwargs
        )


class RiskLimitExceededError(RiskError):
    """Exception raised when risk limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        limit_type: str, 
        current_value: float, 
        limit_value: float,
        client_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            user_message="Risk limit exceeded. Order cannot be processed.",
            **kwargs
        )
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.client_id = client_id
        self.metadata.update({
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "client_id": client_id
        })


class PositionLimitExceededError(RiskLimitExceededError):
    """Exception raised when position limits are exceeded."""
    
    def __init__(
        self, 
        symbol: str, 
        current_position: float, 
        limit: float,
        client_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"Position limit exceeded for {symbol}: {current_position} > {limit}",
            limit_type="position",
            current_value=current_position,
            limit_value=limit,
            client_id=client_id,
            **kwargs
        )
        self.symbol = symbol
        self.metadata["symbol"] = symbol


class ConcentrationLimitExceededError(RiskLimitExceededError):
    """Exception raised when concentration limits are exceeded."""
    
    def __init__(
        self, 
        asset_class: str, 
        current_concentration: float, 
        limit: float,
        client_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"Concentration limit exceeded for {asset_class}: {current_concentration}% > {limit}%",
            limit_type="concentration",
            current_value=current_concentration,
            limit_value=limit,
            client_id=client_id,
            **kwargs
        )
        self.asset_class = asset_class
        self.metadata["asset_class"] = asset_class


class VaRLimitExceededError(RiskLimitExceededError):
    """Exception raised when VaR limits are exceeded."""
    
    def __init__(
        self, 
        current_var: float, 
        var_limit: float,
        confidence_level: float = 0.95,
        client_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"VaR limit exceeded: {current_var} > {var_limit} (confidence: {confidence_level})",
            limit_type="var",
            current_value=current_var,
            limit_value=var_limit,
            client_id=client_id,
            **kwargs
        )
        self.confidence_level = confidence_level
        self.metadata["confidence_level"] = confidence_level


# === ROUTING EXCEPTIONS ===

class RoutingError(TradingInfraError):
    """Base exception for order routing errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_RULE,
            **kwargs
        )
        self.order_id = order_id
        if order_id:
            self.metadata["order_id"] = order_id


class NoRoutingAvailableError(RoutingError):
    """Exception raised when no routing destination is available."""
    
    def __init__(
        self, 
        order_id: str, 
        symbol: str, 
        attempted_venues: List[str],
        **kwargs
    ):
        super().__init__(
            f"No routing available for order {order_id} ({symbol})",
            order_id=order_id,
            severity=ErrorSeverity.HIGH,
            user_message="Order cannot be routed at this time. Please try again later.",
            **kwargs
        )
        self.symbol = symbol
        self.attempted_venues = attempted_venues
        self.metadata.update({
            "symbol": symbol,
            "attempted_venues": attempted_venues
        })


class VenueUnavailableError(RoutingError):
    """Exception raised when a trading venue is unavailable."""
    
    def __init__(self, venue_id: str, reason: str, **kwargs):
        super().__init__(
            f"Venue unavailable: {venue_id} - {reason}",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.venue_id = venue_id
        self.reason = reason
        self.metadata.update({
            "venue_id": venue_id,
            "reason": reason
        })


class RoutingConfigurationError(RoutingError):
    """Exception raised when routing configuration is invalid."""
    
    def __init__(self, message: str, config_issue: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TECHNICAL,
            **kwargs
        )
        self.config_issue = config_issue
        self.metadata["config_issue"] = config_issue


# === EXECUTION EXCEPTIONS ===

class ExecutionError(TradingInfraError):
    """Base exception for trade execution errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_RULE,
            **kwargs
        )
        self.order_id = order_id
        if order_id:
            self.metadata["order_id"] = order_id


class PartialFillError(ExecutionError):
    """Exception raised when partial fill handling fails."""
    
    def __init__(
        self, 
        order_id: str, 
        requested_quantity: float, 
        filled_quantity: float,
        **kwargs
    ):
        super().__init__(
            f"Partial fill for order {order_id}: {filled_quantity}/{requested_quantity}",
            order_id=order_id,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.requested_quantity = requested_quantity
        self.filled_quantity = filled_quantity
        self.metadata.update({
            "requested_quantity": requested_quantity,
            "filled_quantity": filled_quantity
        })


class ExecutionTimeoutError(ExecutionError):
    """Exception raised when order execution times out."""
    
    def __init__(self, order_id: str, timeout_seconds: int, **kwargs):
        super().__init__(
            f"Execution timeout for order {order_id} after {timeout_seconds}s",
            order_id=order_id,
            severity=ErrorSeverity.HIGH,
            retry_after=timeout_seconds,
            **kwargs
        )
        self.timeout_seconds = timeout_seconds
        self.metadata["timeout_seconds"] = timeout_seconds


class InsufficientLiquidityError(ExecutionError):
    """Exception raised when insufficient liquidity is available."""
    
    def __init__(
        self, 
        symbol: str, 
        requested_quantity: float, 
        available_quantity: float,
        **kwargs
    ):
        super().__init__(
            f"Insufficient liquidity for {symbol}: requested {requested_quantity}, available {available_quantity}",
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Insufficient liquidity available for {symbol}",
            **kwargs
        )
        self.symbol = symbol
        self.requested_quantity = requested_quantity
        self.available_quantity = available_quantity
        self.metadata.update({
            "symbol": symbol,
            "requested_quantity": requested_quantity,
            "available_quantity": available_quantity
        })


# === COMPLIANCE EXCEPTIONS ===

class ComplianceError(TradingInfraError):
    """Base exception for compliance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.COMPLIANCE,
            **kwargs
        )


class RegulatoryViolationError(ComplianceError):
    """Exception raised when regulatory rules are violated."""
    
    def __init__(
        self, 
        regulation: str, 
        violation_details: str,
        client_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"Regulatory violation ({regulation}): {violation_details}",
            user_message="Order violates regulatory requirements and cannot be processed.",
            **kwargs
        )
        self.regulation = regulation
        self.violation_details = violation_details
        self.client_id = client_id
        self.metadata.update({
            "regulation": regulation,
            "violation_details": violation_details,
            "client_id": client_id
        })


class AMLViolationError(ComplianceError):
    """Exception raised when AML (Anti-Money Laundering) rules are violated."""
    
    def __init__(self, reason: str, client_id: str, **kwargs):
        super().__init__(
            f"AML violation for client {client_id}: {reason}",
            user_message="Order blocked due to compliance requirements.",
            **kwargs
        )
        self.reason = reason
        self.client_id = client_id
        self.metadata.update({
            "reason": reason,
            "client_id": client_id
        })


class KYCViolationError(ComplianceError):
    """Exception raised when KYC (Know Your Customer) requirements are not met."""
    
    def __init__(self, client_id: str, missing_documents: List[str], **kwargs):
        super().__init__(
            f"KYC requirements not met for client {client_id}",
            user_message="Additional documentation required to process order.",
            **kwargs
        )
        self.client_id = client_id
        self.missing_documents = missing_documents
        self.metadata.update({
            "client_id": client_id,
            "missing_documents": missing_documents
        })


# === SECURITY EXCEPTIONS ===

class SecurityError(TradingInfraError):
    """Base exception for security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            **kwargs
        )


class UnauthorizedAccessError(SecurityError):
    """Exception raised when unauthorized access is attempted."""
    
    def __init__(
        self, 
        resource: str, 
        action: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            f"Unauthorized access to {resource} for action {action}",
            user_message="You are not authorized to perform this action.",
            **kwargs
        )
        self.resource = resource
        self.action = action
        self.user_id = user_id
        self.metadata.update({
            "resource": resource,
            "action": action,
            "user_id": user_id
        })


class AuthenticationError(SecurityError):
    """Exception raised when authentication fails."""
    
    def __init__(self, reason: str, **kwargs):
        super().__init__(
            f"Authentication failed: {reason}",
            user_message="Authentication failed. Please check your credentials.",
            **kwargs
        )
        self.reason = reason
        self.metadata["reason"] = reason


class SessionExpiredError(SecurityError):
    """Exception raised when user session has expired."""
    
    def __init__(self, session_id: str, **kwargs):
        super().__init__(
            f"Session expired: {session_id}",
            user_message="Your session has expired. Please log in again.",
            **kwargs
        )
        self.session_id = session_id
        self.metadata["session_id"] = session_id


# === TECHNICAL EXCEPTIONS ===

class DatabaseError(TradingInfraError):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str, operation: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TECHNICAL,
            user_message="A technical error occurred. Please try again later.",
            **kwargs
        )
        self.operation = operation
        self.metadata["operation"] = operation


class ExternalServiceError(TradingInfraError):
    """Exception raised when external service calls fail."""
    
    def __init__(
        self, 
        service_name: str, 
        error_message: str,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            f"External service error ({service_name}): {error_message}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTEGRATION,
            user_message="External service temporarily unavailable. Please try again later.",
            **kwargs
        )
        self.service_name = service_name
        self.status_code = status_code
        self.metadata.update({
            "service_name": service_name,
            "status_code": status_code
        })


class ConfigurationError(TradingInfraError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TECHNICAL,
            **kwargs
        )
        self.config_key = config_key
        self.metadata["config_key"] = config_key


class CircuitBreakerOpenError(TradingInfraError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, retry_after: int = 60, **kwargs):
        super().__init__(
            f"Circuit breaker open for {service_name}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TECHNICAL,
            retry_after=retry_after,
            user_message="Service temporarily unavailable. Please try again later.",
            **kwargs
        )
        self.service_name = service_name
        self.metadata["service_name"] = service_name


# === PERFORMANCE EXCEPTIONS ===

class PerformanceError(TradingInfraError):
    """Base exception for performance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            **kwargs
        )


class TimeoutError(PerformanceError):
    """Exception raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Timeout in {operation} after {timeout_seconds}s",
            severity=ErrorSeverity.MEDIUM,
            retry_after=int(timeout_seconds),
            **kwargs
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.metadata.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds
        })


class RateLimitExceededError(PerformanceError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        resource: str, 
        current_rate: float, 
        limit: float,
        window_seconds: int = 60,
        **kwargs
    ):
        super().__init__(
            f"Rate limit exceeded for {resource}: {current_rate}/{limit} per {window_seconds}s",
            severity=ErrorSeverity.MEDIUM,
            retry_after=window_seconds,
            user_message="Too many requests. Please wait before trying again.",
            **kwargs
        )
        self.resource = resource
        self.current_rate = current_rate
        self.limit = limit
        self.window_seconds = window_seconds
        self.metadata.update({
            "resource": resource,
            "current_rate": current_rate,
            "limit": limit,
            "window_seconds": window_seconds
        })


# === UTILITY FUNCTIONS ===

def create_error_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    service_name: Optional[str] = None,
    operation: Optional[str] = None,
    **metadata
) -> ErrorContext:
    """Create error context with provided information."""
    return ErrorContext(
        timestamp=datetime.utcnow(),
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        service_name=service_name,
        operation=operation,
        metadata=metadata
    )


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is retryable, False otherwise
    """
    if isinstance(error, TradingInfraError):
        # Check if error has retry_after set
        if error.retry_after is not None:
            return True
        
        # Specific retryable error types
        retryable_types = (
            TimeoutError,
            ExternalServiceError,
            CircuitBreakerOpenError,
            RateLimitExceededError,
            DatabaseError
        )
        
        return isinstance(error, retryable_types)
    
    return False


def get_error_response(error: Exception) -> Dict[str, Any]:
    """
    Convert any exception to standardized error response format.
    
    Args:
        error: Exception to convert
        
    Returns:
        Standardized error response dictionary
    """
    if isinstance(error, TradingInfraError):
        return error.to_dict()
    
    # Handle standard exceptions
    return {
        "error_type": type(error).__name__,
        "message": str(error),
        "user_message": "An unexpected error occurred",
        "severity": ErrorSeverity.HIGH.value,
        "category": ErrorCategory.TECHNICAL.value,
        "error_code": type(error).__name__,
        "timestamp": datetime.utcnow().isoformat(),
        "retryable": is_retryable_error(error)
    }