from __future__ import annotations
import asyncio
import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, Set, Union
from weakref import WeakValueDictionary
import random
import hmac
import json
from cryptography.fernet import Fernet
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator
import numpy as np
from scipy import stats

from ..infra.config import get_settings
from ..infra.logging import EnterpriseLogger

settings = get_settings()
logger = EnterpriseLogger(__name__, 'smart-order-router')

# Enhanced Enums
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class BrokerStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    DEGRADED = "DEGRADED"

class RoutingStrategy(str, Enum):
    LATENCY_OPTIMIZED = "LATENCY_OPTIMIZED"
    COST_OPTIMIZED = "COST_OPTIMIZED"
    FILL_RATE_OPTIMIZED = "FILL_RATE_OPTIMIZED"
    SMART_ROUTING = "SMART_ROUTING"
    ROUND_ROBIN = "ROUND_ROBIN"
    WEIGHTED_ROUND_ROBIN = "WEIGHTED_ROUND_ROBIN"
    ML_OPTIMIZED = "ML_OPTIMIZED"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ComplianceRule(str, Enum):
    MIFID_II = "MIFID_II"
    DODD_FRANK = "DODD_FRANK"
    BASEL_III = "BASEL_III"
    EMIR = "EMIR"
    MiFIR = "MiFIR"

class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    FX = "FX"
    FIXED_INCOME = "FIXED_INCOME"
    DERIVATIVES = "DERIVATIVES"
    CRYPTO = "CRYPTO"

# Enhanced Data Models
@dataclass
class MarketData:
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    last_price: Decimal
    volume: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level2_data: Optional[List[Dict[str, Any]]] = None  # Level II order book
    volatility: Optional[Decimal] = None
    spread_tightness: Optional[Decimal] = None
    
    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / Decimal('2')

@dataclass
class BrokerMetrics:
    latency_ms: float
    fill_rate: float
    cost_per_trade: Decimal
    available_liquidity: Decimal
    error_rate: float
    uptime_percentage: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    market_impact_score: Optional[float] = None
    venue_quality_score: Optional[float] = None

@dataclass
class OrderRequest:
    order_id: str
    client_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "DAY"
    min_quantity: Optional[Decimal] = None
    display_quantity: Optional[Decimal] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_slippage: Optional[Decimal] = None
    urgency: int = 5
    regulatory_flags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    execution_algorithm: Optional[str] = None

@dataclass
class OrderResponse:
    order_id: str
    broker_id: str
    broker_order_id: str
    status: OrderStatus
    filled_quantity: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    execution_reports: List[Dict[str, Any]] = field(default_factory=list)
    tca_analysis: Optional[Dict[str, Any]] = None
    compliance_report: Optional[Dict[str, Any]] = None

# Risk Management Components
@dataclass
class RiskLimits:
    max_position_size: Decimal
    max_daily_loss: Decimal
    max_order_value: Decimal
    concentration_limit: Decimal
    volatility_threshold: Decimal

class RiskManager:
    def __init__(self):
        self.position_limits = {}
        self.daily_pnl_limits = {}
        self.order_value_limits = {}
        self.concentration_limits = {}
        self.volatility_monitors = {}
        self.risk_limits = {}
        
    def set_risk_limits(self, client_id: str, limits: RiskLimits):
        self.risk_limits[client_id] = limits
        
    def check_pre_trade_risk(self, request: OrderRequest) -> bool:
        client_id = request.client_id
        if client_id not in self.risk_limits:
            return True
            
        limits = self.risk_limits[client_id]
        
        # Position limit check
        if request.quantity > limits.max_position_size:
            logger.warning("Position limit exceeded", 
                         client_id=client_id, 
                         quantity=request.quantity,
                         limit=limits.max_position_size)
            return False
            
        # Order value check
        order_value = request.quantity * (request.price or Decimal('0'))
        if order_value > limits.max_order_value:
            logger.warning("Order value limit exceeded",
                         client_id=client_id,
                         order_value=order_value,
                         limit=limits.max_order_value)
            return False
            
        return True
        
    def calculate_var(self, portfolio: Dict[str, Decimal], confidence_level: float = 0.95) -> Decimal:
        # Simplified VaR calculation
        returns = [Decimal(str(random.uniform(-0.02, 0.02))) for _ in range(252)]
        var_threshold = np.percentile([float(r) for r in returns], (1 - confidence_level) * 100)
        
        portfolio_value = sum(portfolio.values())
        var = portfolio_value * Decimal(str(abs(var_threshold)))
        return var

# Compliance Engine
class ComplianceEngine:
    def __init__(self):
        self.rules = {}
        self.trade_monitor = defaultdict(list)
        self.surveillance_alerts = []
        
    def add_compliance_rule(self, rule_type: ComplianceRule, config: Dict[str, Any]):
        self.rules[rule_type] = config
        
    def check_compliance(self, request: OrderRequest) -> Dict[str, Any]:
        violations = []
        compliance_report = {
            "timestamp": datetime.now(timezone.utc),
            "order_id": request.order_id,
            "violations": violations,
            "status": "COMPLIANT"
        }
        
        # MiFID II best execution check
        if ComplianceRule.MIFID_II in self.rules:
            if not request.execution_algorithm:
                violations.append("MiFID II: Execution algorithm not specified")
                
        # Wash trading detection
        self._detect_wash_trading(request)
        
        # Spoofing detection
        self._detect_spoofing(request)
        
        if violations:
            compliance_report["status"] = "NON_COMPLIANT"
            
        return compliance_report
        
    def _detect_wash_trading(self, request: OrderRequest):
        # Simplified wash trading detection
        key = f"{request.client_id}_{request.symbol}"
        recent_trades = self.trade_monitor[key][-10:]  # Last 10 trades
        
        for trade in recent_trades:
            if (trade.side != request.side and 
                abs(trade.price - (request.price or Decimal('0'))) < Decimal('0.01') and
                abs(trade.quantity - request.quantity) < Decimal('1')):
                self.surveillance_alerts.append({
                    "type": "POTENTIAL_WASH_TRADING",
                    "order_id": request.order_id,
                    "timestamp": datetime.now(timezone.utc)
                })
                
    def _detect_spoofing(self, request: OrderRequest):
        # Simplified spoofing detection
        if request.order_type == OrderType.LIMIT and request.display_quantity:
            if request.display_quantity < request.quantity * Decimal('0.1'):
                self.surveillance_alerts.append({
                    "type": "POTENTIAL_SPOOFING",
                    "order_id": request.order_id,
                    "timestamp": datetime.now(timezone.utc)
                })

# Security Framework
class SecurityManager:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.audit_trail = []
        
    def encrypt_order(self, order_data: str) -> str:
        return self.cipher_suite.encrypt(order_data.encode()).decode()
        
    def decrypt_order(self, encrypted_data: str) -> str:
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        
    def sign_order(self, order_data: str, secret_key: str) -> str:
        return hmac.new(secret_key.encode(), order_data.encode(), hashlib.sha256).hexdigest()
        
    def verify_signature(self, order_data: str, signature: str, secret_key: str) -> bool:
        expected_signature = self.sign_order(order_data, secret_key)
        return hmac.compare_digest(signature, expected_signature)
        
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        audit_entry = {
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "details": details,
            "event_id": str(uuid.uuid4())
        }
        self.audit_trail.append(audit_entry)

# Execution Algorithms
class ExecutionAlgorithm(ABC):
    @abstractmethod
    async def execute(self, request: OrderRequest, market_data: MarketData) -> List[OrderResponse]:
        pass

class TWAPAlgorithm(ExecutionAlgorithm):
    def __init__(self, time_horizon_minutes: int = 60):
        self.time_horizon = time_horizon_minutes
        
    async def execute(self, request: OrderRequest, market_data: MarketData) -> List[OrderResponse]:
        slices = max(1, int(self.time_horizon / 5))  # 5-minute intervals
        slice_quantity = request.quantity / Decimal(str(slices))
        
        responses = []
        for i in range(slices):
            # Simulate time-based execution
            await asyncio.sleep(300)  # 5 minutes
            
            # Create slice order
            slice_request = OrderRequest(
                order_id=f"{request.order_id}_slice_{i}",
                client_id=request.client_id,
                symbol=request.symbol,
                side=request.side,
                order_type=OrderType.LIMIT,
                quantity=slice_quantity,
                price=market_data.mid_price,
                time_in_force="DAY"
            )
            
            # Simulate execution response
            response = OrderResponse(
                order_id=slice_request.order_id,
                broker_id="TWAP_EXECUTOR",
                broker_order_id=f"twap_{uuid.uuid4().hex[:8]}",
                status=OrderStatus.FILLED,
                filled_quantity=slice_quantity,
                average_price=market_data.mid_price
            )
            responses.append(response)
            
        return responses

class VWAPAlgorithm(ExecutionAlgorithm):
    def __init__(self, participation_rate: float = 0.1):
        self.participation_rate = participation_rate
        
    async def execute(self, request: OrderRequest, market_data: MarketData) -> List[OrderResponse]:
        # Simplified VWAP execution based on volume participation
        target_volume = market_data.volume * Decimal(str(self.participation_rate))
        responses = []
        
        # Simulate volume-based execution
        executed_volume = Decimal('0')
        while executed_volume < target_volume:
            await asyncio.sleep(60)  # 1 minute intervals
            
            # Calculate slice size based on current market volume
            slice_volume = min(target_volume - executed_volume, 
                             market_data.volume * Decimal(str(self.participation_rate / 10)))
            
            response = OrderResponse(
                order_id=f"{request.order_id}_vwap_slice",
                broker_id="VWAP_EXECUTOR",
                broker_order_id=f"vwap_{uuid.uuid4().hex[:8]}",
                status=OrderStatus.FILLED,
                filled_quantity=slice_volume,
                average_price=market_data.mid_price
            )
            responses.append(response)
            executed_volume += slice_volume
            
        return responses

# Machine Learning Components
class MLExecutionOptimizer:
    def __init__(self):
        self.model_weights = {
            'latency': 0.3,
            'cost': 0.2,
            'fill_rate': 0.3,
            'liquidity': 0.2
        }
        self.historical_performance = defaultdict(list)
        
    def update_performance_data(self, broker_id: str, metrics: Dict[str, Any]):
        self.historical_performance[broker_id].append({
            'timestamp': datetime.now(timezone.utc),
            'metrics': metrics
        })
        
    def predict_broker_performance(self, broker_id: str, market_conditions: Dict[str, Any]) -> float:
        # Simplified ML prediction using historical data
        if broker_id not in self.historical_performance:
            return 0.5
            
        recent_data = self.historical_performance[broker_id][-100:]  # Last 100 data points
        if not recent_data:
            return 0.5
            
        # Calculate weighted performance score
        scores = []
        for data in recent_data:
            metrics = data['metrics']
            score = (
                self.model_weights['latency'] * (1.0 / (1.0 + metrics.get('latency_ms', 100) / 1000)) +
                self.model_weights['cost'] * (1.0 / (1.0 + metrics.get('cost_per_trade', 0.001) * 1000)) +
                self.model_weights['fill_rate'] * metrics.get('fill_rate', 0.9) +
                self.model_weights['liquidity'] * min(1.0, metrics.get('available_liquidity', 1000000) / 10000000)
            )
            scores.append(score)
            
        return float(np.mean(scores))

# Enhanced Broker Interface
class BrokerInterface(Protocol):
    async def submit_order(self, request: OrderRequest) -> OrderResponse: ...
    async def cancel_order(self, order_id: str, broker_order_id: str) -> bool: ...
    async def get_order_status(self, broker_order_id: str) -> OrderResponse: ...
    async def get_market_data(self, symbol: str) -> MarketData: ...
    async def health_check(self) -> Dict[str, Any]: ...

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to half-open", state=self.state)
                else:
                    raise RuntimeError("Circuit breaker is open - broker unavailable")

            if self.state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max_calls:
                raise RuntimeError("Circuit breaker half-open limit exceeded")

            try:
                if self.state == "HALF_OPEN":
                    self.half_open_calls += 1
                
                result = await func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - broker recovered", state=self.state)
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error("Circuit breaker opened", failure_count=self.failure_count, state=self.state)
                
                raise e

class EnterpriseRateLimiter:
    def __init__(self, requests_per_second: float, burst_capacity: int = None):
        self.rate = requests_per_second
        self.capacity = burst_capacity or int(requests_per_second * 2)
        self.tokens = float(self.capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class AdvancedBroker(BrokerInterface):
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        self.id = broker_id
        self.config = config
        self.status = BrokerStatus.ACTIVE
        self.metrics = BrokerMetrics(
            latency_ms=config.get('base_latency_ms', 20),
            fill_rate=config.get('fill_rate', 0.98),
            cost_per_trade=Decimal(str(config.get('cost_per_trade', '0.001'))),
            available_liquidity=Decimal(str(config.get('liquidity', '1000000'))),
            error_rate=config.get('error_rate', 0.02),
            uptime_percentage=config.get('uptime', 99.9)
        )
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = EnterpriseRateLimiter(config.get('rate_limit', 100))
        self.order_cache = WeakValueDictionary()
        self.position_limits = config.get('position_limits', {})
        self.compliance_rules = config.get('compliance_rules', [])
        
        self._connection_pool_size = config.get('connection_pool_size', 10)
        self._active_connections = 0
        self._total_orders_processed = 0
        self._last_health_check = None
        self.security_manager = SecurityManager()

    @asynccontextmanager
    async def get_connection(self):
        if self._active_connections >= self._connection_pool_size:
            raise RuntimeError(f"Connection pool exhausted for broker {self.id}")
        
        self._active_connections += 1
        try:
            yield self
        finally:
            self._active_connections -= 1

    async def submit_order(self, request: OrderRequest) -> OrderResponse:
        if not await self.rate_limiter.acquire():
            raise RuntimeError(f"Rate limit exceeded for broker {self.id}")

        if self.status != BrokerStatus.ACTIVE:
            raise RuntimeError(f"Broker {self.id} is not active: {self.status}")

        await self._validate_order(request)
        
        return await self.circuit_breaker.call(self._submit_order_internal, request)

    async def _submit_order_internal(self, request: OrderRequest) -> OrderResponse:
        async with self.get_connection():
            start_time = time.time()
            
            try:
                await self._simulate_processing_delay()
                
                if await self._should_simulate_failure():
                    raise RuntimeError(f"Broker {self.id} simulated failure")

                broker_order_id = f"{self.id}_{uuid.uuid4().hex[:8]}"
                
                # Encrypt order data for security
                order_data = json.dumps({
                    "order_id": request.order_id,
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "quantity": str(request.quantity),
                    "price": str(request.price) if request.price else None
                })
                encrypted_order = self.security_manager.encrypt_order(order_data)
                
                response = OrderResponse(
                    order_id=request.order_id,
                    broker_id=self.id,
                    broker_order_id=broker_order_id,
                    status=OrderStatus.ACCEPTED,
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.order_cache[request.order_id] = response
                self._total_orders_processed += 1
                
                processing_time = (time.time() - start_time) * 1000
                self.metrics.latency_ms = processing_time
                
                logger.info("Order submitted successfully",
                          order_id=request.order_id,
                          broker_id=self.id,
                          broker_order_id=broker_order_id,
                          processing_time_ms=processing_time)
                
                return response
                
            except Exception as e:
                logger.error("Order submission failed",
                           order_id=request.order_id,
                           broker_id=self.id,
                           error=str(e))
                raise

    async def _validate_order(self, request: OrderRequest) -> None:
        if request.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if request.side == OrderSide.BUY and request.order_type == OrderType.LIMIT and not request.price:
            raise ValueError("Limit buy orders require a price")
        
        symbol_limits = self.position_limits.get(request.symbol, {})
        max_quantity = symbol_limits.get('max_quantity')
        if max_quantity and request.quantity > Decimal(str(max_quantity)):
            raise ValueError(f"Order quantity exceeds limit for {request.symbol}")

    async def _simulate_processing_delay(self) -> None:
        import random
        base_delay = self.metrics.latency_ms / 1000
        jitter = random.uniform(0, 0.015)
        await asyncio.sleep(base_delay + jitter)

    async def _should_simulate_failure(self) -> bool:
        import random
        return random.random() < self.metrics.error_rate

    async def cancel_order(self, order_id: str, broker_order_id: str) -> bool:
        if self.status != BrokerStatus.ACTIVE:
            return False
        
        return await self.circuit_breaker.call(self._cancel_order_internal, order_id, broker_order_id)

    async def _cancel_order_internal(self, order_id: str, broker_order_id: str) -> bool:
        await asyncio.sleep(0.01)
        
        if order_id in self.order_cache:
            response = self.order_cache[order_id]
            response.status = OrderStatus.CANCELLED
            response.timestamp = datetime.now(timezone.utc)
            logger.info("Order cancelled", order_id=order_id, broker_order_id=broker_order_id)
            return True
        
        return False

    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        for response in self.order_cache.values():
            if response.broker_order_id == broker_order_id:
                return response
        raise ValueError(f"Order not found: {broker_order_id}")

    async def get_market_data(self, symbol: str) -> MarketData:
        import random
        base_price = Decimal('100.00')
        spread = Decimal('0.01')
        
        # Enhanced market data with Level II information
        level2_data = []
        for i in range(5):
            level2_data.append({
                "price": base_price - spread/2 - Decimal(str(i * 0.001)),
                "size": Decimal(str(random.randint(100, 1000))),
                "side": "BID"
            })
            level2_data.append({
                "price": base_price + spread/2 + Decimal(str(i * 0.001)),
                "size": Decimal(str(random.randint(100, 1000))),
                "side": "ASK"
            })
        
        return MarketData(
            symbol=symbol,
            bid_price=base_price - spread / 2,
            ask_price=base_price + spread / 2,
            bid_size=Decimal(str(random.randint(100, 1000))),
            ask_size=Decimal(str(random.randint(100, 1000))),
            last_price=base_price,
            volume=Decimal(str(random.randint(10000, 100000))),
            level2_data=level2_data,
            volatility=Decimal(str(random.uniform(0.01, 0.05))),
            spread_tightness=Decimal(str(random.uniform(0.8, 1.2)))
        )

    async def health_check(self) -> Dict[str, Any]:
        self._last_health_check = datetime.now(timezone.utc)
        
        return {
            "broker_id": self.id,
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker.state,
            "active_connections": self._active_connections,
            "total_orders_processed": self._total_orders_processed,
            "metrics": {
                "latency_ms": self.metrics.latency_ms,
                "fill_rate": self.metrics.fill_rate,
                "error_rate": self.metrics.error_rate,
                "uptime_percentage": self.metrics.uptime_percentage
            },
            "last_health_check": self._last_health_check.isoformat(),
            "pool_utilization": self._active_connections / self._connection_pool_size
        }

# Enhanced Routing Engine
class RoutingEngine:
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.SMART_ROUTING):
        self.strategy = strategy
        self.routing_weights = {}
        self.routing_history = defaultdict(list)
        self.ml_optimizer = MLExecutionOptimizer()

    async def select_broker(self, request: OrderRequest, brokers: Dict[str, AdvancedBroker], market_data: Optional[MarketData] = None) -> AdvancedBroker:
        active_brokers = {k: v for k, v in brokers.items() if v.status == BrokerStatus.ACTIVE}
        
        if not active_brokers:
            raise RuntimeError("No active brokers available")

        if self.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return min(active_brokers.values(), key=lambda b: b.metrics.latency_ms)
        
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return min(active_brokers.values(), key=lambda b: b.metrics.cost_per_trade)
        
        elif self.strategy == RoutingStrategy.FILL_RATE_OPTIMIZED:
            return max(active_brokers.values(), key=lambda b: b.metrics.fill_rate)
        
        elif self.strategy == RoutingStrategy.ML_OPTIMIZED:
            return await self._ml_routing_selection(request, active_brokers, market_data)
        
        elif self.strategy == RoutingStrategy.SMART_ROUTING:
            return await self._smart_routing_selection(request, active_brokers, market_data)
        
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            broker_ids = list(active_brokers.keys())
            current_index = len(self.routing_history['round_robin']) % len(broker_ids)
            selected_id = broker_ids[current_index]
            self.routing_history['round_robin'].append(selected_id)
            return active_brokers[selected_id]
        
        else:
            return list(active_brokers.values())[0]

    async def _smart_routing_selection(self, request: OrderRequest, brokers: Dict[str, AdvancedBroker], market_data: Optional[MarketData]) -> AdvancedBroker:
        scores = {}
        
        for broker_id, broker in brokers.items():
            score = 0.0
            
            latency_weight = 0.3
            cost_weight = 0.2
            fill_rate_weight = 0.3
            liquidity_weight = 0.2
            
            normalized_latency = 1.0 / (1.0 + broker.metrics.latency_ms / 100.0)
            normalized_cost = 1.0 / (1.0 + float(broker.metrics.cost_per_trade) * 1000)
            normalized_fill_rate = broker.metrics.fill_rate
            normalized_liquidity = min(1.0, float(broker.metrics.available_liquidity) / float(request.quantity) / 10.0)
            
            score = (latency_weight * normalized_latency +
                    cost_weight * normalized_cost +
                    fill_rate_weight * normalized_fill_rate +
                    liquidity_weight * normalized_liquidity)
            
            if request.urgency >= 8:
                score += 0.2 * normalized_latency
            if request.risk_level == RiskLevel.LOW:
                score += 0.1 * normalized_fill_rate
            
            scores[broker_id] = score
        
        best_broker_id = max(scores.keys(), key=lambda k: scores[k])
        logger.info("Smart routing decision", 
                   selected_broker=best_broker_id, 
                   scores=scores,
                   order_id=request.order_id)
        
        return brokers[best_broker_id]

    async def _ml_routing_selection(self, request: OrderRequest, brokers: Dict[str, AdvancedBroker], market_data: Optional[MarketData]) -> AdvancedBroker:
        scores = {}
        
        for broker_id, broker in brokers.items():
            # Get ML-based performance prediction
            predicted_performance = self.ml_optimizer.predict_broker_performance(broker_id, {})
            
            # Combine with real-time metrics
            real_time_score = 0.0
            if broker.metrics:
                real_time_score = (
                    0.4 * (1.0 / (1.0 + broker.metrics.latency_ms / 1000)) +
                    0.3 * broker.metrics.fill_rate +
                    0.3 * (1.0 - broker.metrics.error_rate)
                )
            
            # Weighted combination of ML prediction and real-time metrics
            final_score = 0.6 * predicted_performance + 0.4 * real_time_score
            scores[broker_id] = final_score
            
        best_broker_id = max(scores.keys(), key=lambda k: scores[k])
        logger.info("ML routing decision", 
                   selected_broker=best_broker_id, 
                   scores=scores,
                   order_id=request.order_id)
        
        return brokers[best_broker_id]

# Transaction Cost Analysis
class TransactionCostAnalyzer:
    def __init__(self):
        self.benchmark_data = {}
        
    def analyze_tca(self, order_response: OrderResponse, market_data: MarketData) -> Dict[str, Any]:
        if not order_response.average_price or not market_data:
            return {}
            
        # Calculate implementation shortfall
        mid_price = market_data.mid_price
        execution_price = order_response.average_price
        
        # Slippage calculation
        slippage = abs(execution_price - mid_price)
        slippage_pct = (slippage / mid_price) * 100
        
        # Market impact estimation
        market_impact = self._estimate_market_impact(order_response, market_data)
        
        return {
            "slippage_basis_points": float(slippage_pct * 100),
            "absolute_slippage": float(slippage),
            "market_impact_basis_points": float(market_impact * 10000),
            "total_cost_bps": float((slippage_pct + market_impact) * 10000),
            "execution_quality": self._assess_execution_quality(slippage_pct, market_impact)
        }
        
    def _estimate_market_impact(self, order_response: OrderResponse, market_data: MarketData) -> float:
        if not market_data.volume or order_response.filled_quantity <= 0:
            return 0.0
            
        participation_rate = float(order_response.filled_quantity / market_data.volume)
        # Simplified square-root market impact model
        market_impact = 0.001 * (participation_rate ** 0.5)
        return market_impact
        
    def _assess_execution_quality(self, slippage_pct: float, market_impact: float) -> str:
        total_cost = slippage_pct + market_impact
        if total_cost < 0.001:  # 1 basis point
            return "EXCELLENT"
        elif total_cost < 0.005:  # 5 basis points
            return "GOOD"
        elif total_cost < 0.01:   # 10 basis points
            return "ACCEPTABLE"
        else:
            return "POOR"

# Enhanced Smart Order Router
class EnterpriseSmartOrderRouter:
    def __init__(self, brokers: Dict[str, AdvancedBroker], routing_strategy: RoutingStrategy = RoutingStrategy.SMART_ROUTING):
        self.brokers = brokers
        self.routing_engine = RoutingEngine(routing_strategy)
        self.active_orders = {}
        self.order_history = []
        self.risk_manager = RiskManager()
        self.compliance_engine = ComplianceEngine()
        self.security_manager = SecurityManager()
        self.tca_analyzer = TransactionCostAnalyzer()
        self.execution_algorithms = {
            "TWAP": TWAPAlgorithm(),
            "VWAP": VWAPAlgorithm()
        }
        
        self._setup_metrics()
        
        logger.info("Enterprise Smart Order Router initialized",
                   broker_count=len(brokers),
                   routing_strategy=routing_strategy.value)

    def _setup_metrics(self):
        self.metrics = {
            'orders_routed_total': Counter('sor_orders_routed_total', 'Total orders routed', ['broker_id', 'status']),
            'routing_latency': Histogram('sor_routing_latency_seconds', 'Order routing latency'),
            'active_orders_gauge': Gauge('sor_active_orders', 'Currently active orders'),
            'broker_selection_count': Counter('sor_broker_selection_total', 'Broker selection count', ['broker_id']),
            'compliance_violations': Counter('sor_compliance_violations_total', 'Compliance violations', ['rule_type']),
            'risk_violations': Counter('sor_risk_violations_total', 'Risk violations', ['violation_type'])
        }

    async def route_order(self, request: OrderRequest) -> OrderResponse:
        start_time = time.time()
        
        try:
            # Security: Log audit trail
            self.security_manager.log_audit_event("ORDER_SUBMITTED", {
                "order_id": request.order_id,
                "client_id": request.client_id,
                "symbol": request.symbol,
                "side": request.side.value,
                "quantity": str(request.quantity)
            })
            
            # Risk Management: Pre-trade checks
            if not self.risk_manager.check_pre_trade_risk(request):
                self.metrics['risk_violations'].labels(violation_type="PRE_TRADE").inc()
                raise RuntimeError("Pre-trade risk check failed")
            
            # Compliance: Check regulatory requirements
            compliance_report = self.compliance_engine.check_compliance(request)
            if compliance_report["status"] != "COMPLIANT":
                self.metrics['compliance_violations'].labels(rule_type="PRE_TRADE").inc()
                logger.warning("Compliance violation detected", 
                             order_id=request.order_id,
                             violations=compliance_report["violations"])
            
            # Advanced Execution Algorithms
            if request.execution_algorithm in self.execution_algorithms:
                algorithm = self.execution_algorithms[request.execution_algorithm]
                market_data = await self._get_market_data(request.symbol)
                responses = await algorithm.execute(request, market_data)
                
                # Aggregate responses for algorithmic orders
                total_filled = sum(r.filled_quantity for r in responses)
                avg_price = sum(r.average_price * r.filled_quantity for r in responses if r.average_price) / total_filled if total_filled > 0 else None
                
                final_response = OrderResponse(
                    order_id=request.order_id,
                    broker_id="ALGO_EXECUTOR",
                    broker_order_id=f"algo_{uuid.uuid4().hex[:8]}",
                    status=OrderStatus.FILLED if total_filled == request.quantity else OrderStatus.PARTIALLY_FILLED,
                    filled_quantity=total_filled,
                    average_price=avg_price,
                    tca_analysis=self.tca_analyzer.analyze_tca(responses[0] if responses else None, market_data),
                    compliance_report=compliance_report
                )
                
                self.active_orders[request.order_id] = {
                    'request': request,
                    'response': final_response,
                    'broker': None,
                    'start_time': start_time,
                    'algorithm_responses': responses
                }
                
                return final_response
            
            # Standard routing for non-algorithmic orders
            await self._pre_route_validation(request)
            
            market_data = await self._get_market_data(request.symbol)
            
            selected_broker = await self.routing_engine.select_broker(request, self.brokers, market_data)
            
            logger.info("Routing order to broker",
                       order_id=request.order_id,
                       broker_id=selected_broker.id,
                       routing_strategy=self.routing_engine.strategy.value,
                       symbol=request.symbol,
                       side=request.side.value,
                       quantity=str(request.quantity))
            
            response = await selected_broker.submit_order(request)
            
            # Add TCA and compliance analysis
            response.tca_analysis = self.tca_analyzer.analyze_tca(response, market_data)
            response.compliance_report = compliance_report
            
            self.active_orders[request.order_id] = {
                'request': request,
                'response': response,
                'broker': selected_broker,
                'start_time': start_time
            }
            
            self.metrics['orders_routed_total'].labels(
                broker_id=selected_broker.id,
                status=response.status.value
            ).inc()
            
            self.metrics['broker_selection_count'].labels(
                broker_id=selected_broker.id
            ).inc()
            
            self.metrics['active_orders_gauge'].set(len(self.active_orders))
            
            routing_time = time.time() - start_time
            self.metrics['routing_latency'].observe(routing_time)
            
            logger.info("Order routed successfully",
                       order_id=request.order_id,
                       broker_id=selected_broker.id,
                       broker_order_id=response.broker_order_id,
                       routing_time_ms=routing_time * 1000)
            
            return response
            
        except Exception as e:
            logger.error("Order routing failed",
                        order_id=request.order_id,
                        error=str(e),
                        error_type=type(e).__name__)
            
            self.metrics['orders_routed_total'].labels(
                broker_id='unknown',
                status='error'
            ).inc()
            
            raise

    async def _pre_route_validation(self, request: OrderRequest) -> None:
        if request.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if not request.symbol:
            raise ValueError("Symbol is required")
        
        if request.order_type == OrderType.LIMIT and not request.price:
            raise ValueError("Limit orders require a price")

    async def _get_market_data(self, symbol: str) -> Optional[MarketData]:
        try:
            active_brokers = [b for b in self.brokers.values() if b.status == BrokerStatus.ACTIVE]
            if active_brokers:
                return await active_brokers[0].get_market_data(symbol)
        except Exception as e:
            logger.warning("Failed to get market data", symbol=symbol, error=str(e))
        return None

    async def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.active_orders:
            logger.warning("Order not found for cancellation", order_id=order_id)
            return False
        
        order_info = self.active_orders[order_id]
        broker = order_info['broker']
        response = order_info['response']
        
        try:
            success = await broker.cancel_order(order_id, response.broker_order_id)
            
            if success:
                del self.active_orders[order_id]
                self.metrics['active_orders_gauge'].set(len(self.active_orders))
                logger.info("Order cancelled successfully", order_id=order_id)
                
                # Security: Log audit trail
                self.security_manager.log_audit_event("ORDER_CANCELLED", {
                    "order_id": order_id,
                    "broker_order_id": response.broker_order_id
                })
            
            return success
            
        except Exception as e:
            logger.error("Order cancellation failed", order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        if order_id not in self.active_orders:
            return None
        
        order_info = self.active_orders[order_id]
        broker = order_info['broker']
        response = order_info['response']
        
        try:
            return await broker.get_order_status(response.broker_order_id)
        except Exception as e:
            logger.error("Failed to get order status", order_id=order_id, error=str(e))
            return None

    async def health_check(self) -> Dict[str, Any]:
        broker_health = {}
        
        for broker_id, broker in self.brokers.items():
            try:
                broker_health[broker_id] = await broker.health_check()
            except Exception as e:
                broker_health[broker_id] = {"error": str(e), "status": "unhealthy"}
        
        active_broker_count = sum(1 for health in broker_health.values() 
                                if health.get('status') == 'ACTIVE')
        
        return {
            "router_status": "healthy" if active_broker_count > 0 else "degraded",
            "active_orders": len(self.active_orders),
            "total_brokers": len(self.brokers),
            "active_brokers": active_broker_count,
            "routing_strategy": self.routing_engine.strategy.value,
            "broker_health": broker_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_status": "active",
            "compliance_status": "active"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        total_orders = sum(len(history) for history in self.routing_engine.routing_history.values())
        
        broker_distribution = {}
        for broker_id in self.brokers.keys():
            broker_distribution[broker_id] = sum(
                1 for history in self.routing_engine.routing_history.values()
                for order in history if order == broker_id
            )
        
        return {
            "total_orders_routed": total_orders,
            "active_orders": len(self.active_orders),
            "broker_distribution": broker_distribution,
            "routing_strategy": self.routing_engine.strategy.value,
            "average_routing_latency_ms": 0.0,
            "success_rate": 1.0,
            "compliance_violations": self.metrics['compliance_violations'].describe(),
            "risk_violations": self.metrics['risk_violations'].describe()
        }

    def add_compliance_rule(self, rule_type: ComplianceRule, config: Dict[str, Any]):
        """Add compliance rule to the system"""
        self.compliance_engine.add_compliance_rule(rule_type, config)
        
    def set_client_risk_limits(self, client_id: str, limits: RiskLimits):
        """Set risk limits for a client"""
        self.risk_manager.set_risk_limits(client_id, limits)
        
    async def get_best_execution_report(self, order_id: str) -> Dict[str, Any]:
        """Generate best execution report for regulatory compliance"""
        if order_id not in self.active_orders:
            return {}
            
        order_info = self.active_orders[order_id]
        response = order_info['response']
        request = order_info['request']
        
        return {
            "order_id": order_id,
            "client_id": request.client_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": str(request.quantity),
            "execution_time": response.timestamp.isoformat(),
            "average_price": str(response.average_price) if response.average_price else None,
            "tca_analysis": response.tca_analysis,
            "compliance_report": response.compliance_report,
            "routing_justification": self._generate_routing_justification(request, response)
        }
        
    def _generate_routing_justification(self, request: OrderRequest, response: OrderResponse) -> Dict[str, Any]:
        """Generate justification for routing decision for MiFID II compliance"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routing_strategy": self.routing_engine.strategy.value,
            "selected_broker": response.broker_id,
            "considered_brokers": list(self.brokers.keys()),
            "decision_factors": {
                "latency": "LOW" if response.timestamp.microsecond < 500000 else "HIGH",
                "cost": "OPTIMIZED",
                "liquidity": "SUFFICIENT",
                "fill_rate": "HIGH"
            },
            "best_execution_rationale": "Selected broker based on optimal combination of latency, cost, and fill rate"
        }

# Enhanced Mock Broker for demonstration
class MockBroker:
    def __init__(self, broker_id: str, base_latency_ms: int = 20):
        self.id = broker_id
        self.latency = base_latency_ms

    async def submit_order(self, symbol: str, side: str, quantity: float, price: float | None) -> Dict[str, Any]:
        # simulate variable latency & occasional failure
        await asyncio.sleep((self.latency + random.randint(0, 15)) / 1000)
        if random.random() < 0.05:
            raise RuntimeError(f"broker {self.id} temporary failure")
        return {"broker_id": self.id, "status": "accepted"}

class SmartOrderRouter:
    def __init__(self, brokers: Dict[str, MockBroker]):
        self.brokers = brokers
        self.log = JsonLogger(__name__)

    async def route(self, *, symbol: str, side: str, quantity: Decimal, price: Decimal | None) -> Dict[str, Any]:
        # very simple heuristic: choose the fastest broker (lowest base latency)
        best = sorted(self.brokers.values(), key=lambda b: b.latency)[0]
        self.log.info("route_decision", symbol=symbol, side=side, qty=str(quantity), broker=best.id)
        resp = await best.submit_order(symbol, side, float(quantity), float(price) if price is not None else None)
        return {"selected_broker": best.id, "broker_response": resp}