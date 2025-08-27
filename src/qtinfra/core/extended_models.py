"""
Extended enterprise models for advanced trading infrastructure features.

Includes models for:
- Risk management and limits
- Compliance and regulatory reporting
- Advanced analytics and ML
- Settlement and clearing
- Market data and feeds
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, DateTime, Numeric, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)

from ..infra.db import Base
from .models import AuditMixin, AssetClass, Currency


# === RISK MANAGEMENT MODELS ===

class RiskLimit(Base, AuditMixin):
    """Risk limits for clients and positions."""
    __tablename__ = "risk_limits"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Limit details
    limit_type: Mapped[str] = mapped_column(String(32), nullable=False)  # POSITION, VAR, NOTIONAL, etc.
    asset_class: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default=Currency.USD.value, nullable=False)
    
    # Limit values
    limit_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    warning_threshold: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    current_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    
    # Limit configuration
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiration_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        CheckConstraint('limit_value > 0', name='check_positive_limit_value'),
        CheckConstraint('current_value >= 0', name='check_non_negative_current_value'),
        Index('idx_risk_limits_client_type', 'client_id', 'limit_type'),
    )


class Position(Base, AuditMixin):
    """Client positions across assets."""
    __tablename__ = "positions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Position details
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    
    # Position quantities
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    available_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    locked_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    
    # Cost and valuation
    average_cost: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    market_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    unrealized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    
    # Currency and dates
    currency: Mapped[str] = mapped_column(String(3), default=Currency.USD.value, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    __table_args__ = (
        CheckConstraint('available_quantity >= 0', name='check_non_negative_available_quantity'),
        CheckConstraint('locked_quantity >= 0', name='check_non_negative_locked_quantity'),
        UniqueConstraint('client_id', 'symbol', 'asset_class', name='uq_client_symbol_position'),
        Index('idx_positions_client_symbol', 'client_id', 'symbol'),
    )


class VaRCalculation(Base, AuditMixin):
    """Value at Risk calculations."""
    __tablename__ = "var_calculations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # VaR parameters
    confidence_level: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # e.g., 0.95 for 95%
    holding_period_days: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    calculation_method: Mapped[str] = mapped_column(String(32), nullable=False)  # HISTORICAL, MONTE_CARLO, PARAMETRIC
    
    # VaR results
    var_amount: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    expected_shortfall: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default=Currency.USD.value, nullable=False)
    
    # Calculation metadata
    calculation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    portfolio_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    number_of_scenarios: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    __table_args__ = (
        CheckConstraint('confidence_level > 0 AND confidence_level < 1', name='check_valid_confidence_level'),
        CheckConstraint('holding_period_days > 0', name='check_positive_holding_period'),
        CheckConstraint('var_amount >= 0', name='check_non_negative_var'),
        Index('idx_var_client_date', 'client_id', 'calculation_date'),
    )


# === COMPLIANCE MODELS ===

class ComplianceReport(Base, AuditMixin):
    """Compliance reports for regulatory requirements."""
    __tablename__ = "compliance_reports"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Report details
    report_type: Mapped[str] = mapped_column(String(64), nullable=False)  # MIFID_II, DODD_FRANK, etc.
    regulation: Mapped[str] = mapped_column(String(64), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(32), nullable=False)
    
    # Report period
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Report content
    report_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="GENERATED", nullable=False)
    submission_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        CheckConstraint('end_date >= start_date', name='check_valid_report_period'),
        Index('idx_compliance_reports_type_date', 'report_type', 'generated_at'),
    )


class AuditLog(Base):
    """Audit log for all system activities."""
    __tablename__ = "audit_logs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_category: Mapped[str] = mapped_column(String(32), nullable=False)  # ORDER, RISK, COMPLIANCE, SECURITY
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Context information
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    
    # Event data
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Technical context
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # IPv6 support
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    __table_args__ = (
        Index('idx_audit_logs_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_logs_client_timestamp', 'client_id', 'timestamp'),
        Index('idx_audit_logs_event_timestamp', 'event_type', 'timestamp'),
    )


class RegulatoryRule(Base, AuditMixin):
    """Regulatory rules and compliance checks."""
    __tablename__ = "regulatory_rules"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Rule identification
    rule_name: Mapped[str] = mapped_column(String(255), nullable=False)
    rule_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    regulation: Mapped[str] = mapped_column(String(64), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(32), nullable=False)
    
    # Rule definition
    description: Mapped[str] = mapped_column(Text, nullable=False)
    rule_logic: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    applicable_asset_classes: Mapped[List[str]] = mapped_column(JSON, nullable=True)
    
    # Rule configuration
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiration_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Enforcement
    enforcement_level: Mapped[str] = mapped_column(String(16), nullable=False)  # WARNING, BLOCK, REJECT
    exemptions: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_regulatory_rules_regulation', 'regulation', 'is_active'),
    )


# === ADVANCED ANALYTICS MODELS ===

class TransactionCostAnalysis(Base, AuditMixin):
    """Transaction Cost Analysis (TCA) results."""
    __tablename__ = "transaction_cost_analysis"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String(36), ForeignKey("orders.id"), nullable=False, index=True)
    execution_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    
    # Cost components
    market_impact: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    timing_cost: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    explicit_costs: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    implicit_costs: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal('0'), nullable=False)
    total_cost: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # Benchmarks and metrics
    arrival_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    vwap_benchmark: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    twap_benchmark: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    implementation_shortfall: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Analysis metadata
    analysis_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    analysis_method: Mapped[str] = mapped_column(String(32), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default=Currency.USD.value, nullable=False)
    
    __table_args__ = (
        Index('idx_tca_order_date', 'order_id', 'analysis_date'),
    )


class MarketDataFeed(Base, AuditMixin):
    """Market data feed for pricing and analytics."""
    __tablename__ = "market_data_feeds"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol information
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    
    # Price data
    bid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    ask_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    last_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    mid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Volume data
    bid_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    ask_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    market_session: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # PRE, REGULAR, POST
    
    # Data quality
    data_source: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2), nullable=True)  # 0.00-1.00
    
    __table_args__ = (
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_exchange_timestamp', 'exchange', 'timestamp'),
    )


class MLModelOutput(Base, AuditMixin):
    """Machine Learning model predictions and outputs."""
    __tablename__ = "ml_model_outputs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model information
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)  # PRICE_PREDICTION, RISK_SCORING, etc.
    
    # Input context
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    
    # Model output
    prediction: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    feature_importance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timing and performance
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    __table_args__ = (
        CheckConstraint('confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)', 
                       name='check_valid_confidence_score'),
        Index('idx_ml_outputs_model_timestamp', 'model_name', 'prediction_timestamp'),
    )


# === SETTLEMENT MODELS ===

class Settlement(Base, AuditMixin):
    """Trade settlement tracking."""
    __tablename__ = "settlements"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String(36), ForeignKey("orders.id"), nullable=False, index=True)
    execution_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    
    # Settlement details
    settlement_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # Settlement status
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False, index=True)
    settlement_method: Mapped[str] = mapped_column(String(32), nullable=False)  # DVP, FOP, etc.
    
    # Counterparty information
    counterparty_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    custodian: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Settlement instructions
    delivery_instructions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    payment_instructions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        CheckConstraint('amount != 0', name='check_non_zero_amount'),
        CheckConstraint('value_date >= settlement_date', name='check_valid_value_date'),
        Index('idx_settlements_order_status', 'order_id', 'status'),
        Index('idx_settlements_date_status', 'settlement_date', 'status'),
    )


# === EXECUTION VENUE MODELS ===

class ExecutionVenue(Base, AuditMixin):
    """Trading venues and their characteristics."""
    __tablename__ = "execution_venues"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Venue identification
    venue_code: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    venue_name: Mapped[str] = mapped_column(String(255), nullable=False)
    venue_type: Mapped[str] = mapped_column(String(32), nullable=False)  # EXCHANGE, ATS, DARK_POOL, etc.
    
    # Geographic and regulatory
    country: Mapped[str] = mapped_column(String(2), nullable=False)  # ISO country code
    jurisdiction: Mapped[str] = mapped_column(String(32), nullable=False)
    regulator: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Venue characteristics
    supported_asset_classes: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    trading_hours: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    min_order_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    max_order_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Performance metrics
    average_fill_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    average_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    connectivity_status: Mapped[str] = mapped_column(String(16), default="CONNECTED", nullable=False)
    
    __table_args__ = (
        Index('idx_execution_venues_type_active', 'venue_type', 'is_active'),
    )


# === SMART ROUTING MODELS ===

class SmartRoutingDecision(Base, AuditMixin):
    """Smart routing decisions and rationale."""
    __tablename__ = "smart_routing_decisions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String(36), ForeignKey("orders.id"), nullable=False, index=True)
    
    # Routing decision
    selected_venue: Mapped[str] = mapped_column(String(64), nullable=False)
    routing_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Decision rationale
    rationale: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    expected_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    expected_fill_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    
    # Alternative venues considered
    alternative_venues: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    
    # Decision metadata
    decision_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision_engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    market_conditions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_valid_confidence_score'),
        CheckConstraint('expected_fill_rate IS NULL OR (expected_fill_rate >= 0 AND expected_fill_rate <= 1)', 
                       name='check_valid_fill_rate'),
        Index('idx_smart_routing_order_timestamp', 'order_id', 'decision_timestamp'),
    )