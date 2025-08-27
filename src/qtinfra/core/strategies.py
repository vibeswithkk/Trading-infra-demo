"""
Enterprise-grade Strategy and Plugin Framework for Trading Infrastructure.

Implements extensibility patterns for:
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Risk management strategies
- Routing algorithms
- Compliance engines
- Event handling plugins
- ML model integration

Features:
- Strategy pattern for algorithm selection
- Plugin system for dynamic loading
- Configuration-driven behavior
- Performance optimization
- Enterprise observability
"""

from __future__ import annotations
import abc
import asyncio
import importlib
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Protocol, Type, Union, Callable
from enum import Enum
from dataclasses import dataclass, field

from ..core.models import Order, Execution, OrderType, OrderSide
from ..core.exceptions import (
    TradingInfraError, OrderValidationError, RoutingError, 
    ExternalServiceError, ConfigurationError
)
from ..infra.logging import EnterpriseLogger


# === EXECUTION ALGORITHM STRATEGIES ===

class ExecutionAlgorithmType(str, Enum):
    """Execution algorithm types."""
    TWAP = "TWAP"  # Time Weighted Average Price
    VWAP = "VWAP"  # Volume Weighted Average Price
    POV = "POV"    # Percentage of Volume
    IS = "IS"      # Implementation Shortfall
    AGGRESSIVE = "AGGRESSIVE"
    PASSIVE = "PASSIVE"
    ICEBERG = "ICEBERG"
    HIDDEN = "HIDDEN"
    ML_OPTIMIZED = "ML_OPTIMIZED"


@dataclass
class ExecutionParameters:
    """Parameters for execution algorithms."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    participation_rate: Optional[float] = None  # For POV
    max_volume_rate: Optional[float] = None     # Max % of market volume
    price_improvement: Optional[Decimal] = None
    hidden_size: Optional[Decimal] = None       # For iceberg orders
    urgency: Optional[float] = None             # 0.0 = passive, 1.0 = aggressive
    risk_aversion: Optional[float] = None       # For Implementation Shortfall
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of execution algorithm."""
    algorithm_used: str
    total_executed: Decimal
    average_price: Decimal
    slippage: Decimal
    market_impact: Decimal
    timing_cost: Decimal
    total_cost: Decimal
    confidence_score: float
    execution_details: Dict[str, Any] = field(default_factory=dict)


class ExecutionStrategy(abc.ABC):
    """Abstract base class for execution strategies."""
    
    def __init__(self, name: str, logger: Optional[EnterpriseLogger] = None):
        self.name = name
        self.log = logger or EnterpriseLogger(__name__, f'strategy-{name.lower()}')
    
    @abc.abstractmethod
    async def execute(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute the order using this strategy."""
        pass
    
    @abc.abstractmethod
    def validate_parameters(self, parameters: ExecutionParameters) -> List[str]:
        """Validate strategy-specific parameters."""
        pass
    
    @abc.abstractmethod
    def estimate_cost(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate execution cost for this strategy."""
        pass


class TWAPStrategy(ExecutionStrategy):
    """Time Weighted Average Price execution strategy."""
    
    def __init__(self):
        super().__init__("TWAP")
    
    async def execute(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute order using TWAP algorithm."""
        try:
            self.log.info(
                "Executing TWAP strategy",
                order_id=order.id,
                quantity=float(order.quantity),
                symbol=order.symbol
            )
            
            # Calculate time slices
            start_time = parameters.start_time or datetime.now(timezone.utc)
            end_time = parameters.end_time or (start_time + timedelta(hours=1))
            duration = (end_time - start_time).total_seconds()
            
            # Default to 10-minute slices
            slice_duration = min(600, duration / 10)  # seconds
            num_slices = int(duration / slice_duration)
            
            slice_quantity = order.quantity / num_slices
            total_executed = Decimal('0')
            total_value = Decimal('0')
            
            # Simulate execution over time slices
            current_time = start_time
            market_price = Decimal(str(market_data.get('current_price', order.price or '100.0')))
            
            for i in range(num_slices):
                # Add some price variation (±0.1%)
                price_variation = market_price * Decimal('0.001') * (i % 3 - 1)
                execution_price = market_price + price_variation
                
                # Execute slice
                executed_qty = min(slice_quantity, order.quantity - total_executed)
                slice_value = executed_qty * execution_price
                
                total_executed += executed_qty
                total_value += slice_value
                
                self.log.debug(
                    "TWAP slice executed",
                    slice=i+1,
                    quantity=float(executed_qty),
                    price=float(execution_price)
                )
                
                current_time += timedelta(seconds=slice_duration)
                
                if total_executed >= order.quantity:
                    break
            
            average_price = total_value / total_executed if total_executed > 0 else Decimal('0')
            benchmark_price = market_price
            slippage = abs(average_price - benchmark_price) / benchmark_price * 100
            
            return ExecutionResult(
                algorithm_used="TWAP",
                total_executed=total_executed,
                average_price=average_price,
                slippage=slippage,
                market_impact=slippage * Decimal('0.5'),  # Estimate 50% of slippage as impact
                timing_cost=slippage * Decimal('0.3'),    # Estimate 30% as timing cost
                total_cost=slippage,
                confidence_score=0.85,
                execution_details={
                    "num_slices": num_slices,
                    "slice_duration_seconds": slice_duration,
                    "benchmark_price": float(benchmark_price)
                }
            )
            
        except Exception as e:
            self.log.error("TWAP execution failed", error=str(e))
            raise ExternalServiceError("TWAP", str(e))
    
    def validate_parameters(self, parameters: ExecutionParameters) -> List[str]:
        """Validate TWAP parameters."""
        violations = []
        
        if parameters.start_time and parameters.end_time:
            if parameters.end_time <= parameters.start_time:
                violations.append("End time must be after start time")
        
        if parameters.participation_rate and parameters.participation_rate > 1.0:
            violations.append("Participation rate cannot exceed 100%")
        
        return violations
    
    def estimate_cost(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate TWAP execution cost."""
        # Simple cost model: lower cost for longer execution periods
        duration_hours = 1.0  # Default
        if parameters.start_time and parameters.end_time:
            duration_hours = (parameters.end_time - parameters.start_time).total_seconds() / 3600
        
        base_cost = Decimal('0.05')  # 5 basis points
        duration_discount = max(Decimal('0.5'), Decimal('1.0') / Decimal(str(duration_hours)))
        
        return base_cost * duration_discount


class VWAPStrategy(ExecutionStrategy):
    """Volume Weighted Average Price execution strategy."""
    
    def __init__(self):
        super().__init__("VWAP")
    
    async def execute(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute order using VWAP algorithm."""
        try:
            self.log.info(
                "Executing VWAP strategy",
                order_id=order.id,
                quantity=float(order.quantity),
                symbol=order.symbol
            )
            
            # Get historical volume profile
            volume_profile = market_data.get('volume_profile', [
                {'time': '09:30', 'volume_pct': 0.15},
                {'time': '10:00', 'volume_pct': 0.12},
                {'time': '11:00', 'volume_pct': 0.08},
                {'time': '12:00', 'volume_pct': 0.05},
                {'time': '13:00', 'volume_pct': 0.06},
                {'time': '14:00', 'volume_pct': 0.09},
                {'time': '15:00', 'volume_pct': 0.18},
                {'time': '16:00', 'volume_pct': 0.27}
            ])
            
            participation_rate = parameters.participation_rate or 0.1  # 10% default
            market_price = Decimal(str(market_data.get('current_price', order.price or '100.0')))
            
            total_executed = Decimal('0')
            total_value = Decimal('0')
            
            # Execute according to volume profile
            for i, volume_period in enumerate(volume_profile):
                if total_executed >= order.quantity:
                    break
                
                # Calculate quantity for this period based on volume
                period_quantity = order.quantity * Decimal(str(volume_period['volume_pct'])) * Decimal(str(participation_rate))
                period_quantity = min(period_quantity, order.quantity - total_executed)
                
                # Price variation based on volume (higher volume = less impact)
                impact_factor = Decimal('0.02') * (Decimal('1') - Decimal(str(volume_period['volume_pct'])))
                execution_price = market_price * (Decimal('1') + impact_factor)
                
                period_value = period_quantity * execution_price
                total_executed += period_quantity
                total_value += period_value
                
                self.log.debug(
                    "VWAP period executed",
                    period=volume_period['time'],
                    quantity=float(period_quantity),
                    price=float(execution_price),
                    volume_pct=volume_period['volume_pct']
                )
            
            average_price = total_value / total_executed if total_executed > 0 else Decimal('0')
            benchmark_price = market_price
            slippage = abs(average_price - benchmark_price) / benchmark_price * 100
            
            return ExecutionResult(
                algorithm_used="VWAP",
                total_executed=total_executed,
                average_price=average_price,
                slippage=slippage,
                market_impact=slippage * Decimal('0.7'),  # Higher market impact for VWAP
                timing_cost=slippage * Decimal('0.2'),
                total_cost=slippage,
                confidence_score=0.90,
                execution_details={
                    "participation_rate": participation_rate,
                    "volume_periods": len(volume_profile),
                    "benchmark_price": float(benchmark_price)
                }
            )
            
        except Exception as e:
            self.log.error("VWAP execution failed", error=str(e))
            raise ExternalServiceError("VWAP", str(e))
    
    def validate_parameters(self, parameters: ExecutionParameters) -> List[str]:
        """Validate VWAP parameters."""
        violations = []
        
        if parameters.participation_rate:
            if parameters.participation_rate <= 0 or parameters.participation_rate > 0.5:
                violations.append("Participation rate must be between 0 and 50%")
        
        return violations
    
    def estimate_cost(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate VWAP execution cost."""
        participation_rate = parameters.participation_rate or 0.1
        
        # Higher participation = higher cost
        base_cost = Decimal('0.04')  # 4 basis points
        participation_penalty = Decimal(str(participation_rate)) * Decimal('0.1')
        
        return base_cost + participation_penalty


class ImplementationShortfallStrategy(ExecutionStrategy):
    """Implementation Shortfall execution strategy."""
    
    def __init__(self):
        super().__init__("Implementation Shortfall")
    
    async def execute(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute order using Implementation Shortfall algorithm."""
        try:
            self.log.info(
                "Executing Implementation Shortfall strategy",
                order_id=order.id,
                quantity=float(order.quantity),
                symbol=order.symbol
            )
            
            urgency = parameters.urgency or 0.5  # Moderate urgency
            risk_aversion = parameters.risk_aversion or 0.5
            
            market_price = Decimal(str(market_data.get('current_price', order.price or '100.0')))
            volatility = Decimal(str(market_data.get('volatility', '0.02')))  # 2% daily vol
            
            # Calculate optimal execution trajectory
            # Higher urgency = faster execution, higher market impact
            # Higher risk aversion = slower execution, lower timing risk
            
            execution_rate = urgency * (2 - risk_aversion)  # 0.5 to 2.0
            execution_periods = max(1, int(10 / execution_rate))  # 1 to 20 periods
            
            total_executed = Decimal('0')
            total_value = Decimal('0')
            arrival_price = market_price
            
            for period in range(execution_periods):
                if total_executed >= order.quantity:
                    break
                
                # Calculate period quantity (more aggressive early on if urgent)
                remaining_qty = order.quantity - total_executed
                if period == execution_periods - 1:
                    period_qty = remaining_qty
                else:
                    urgency_factor = 1 + (urgency - 0.5) * (1 - period / execution_periods)
                    period_qty = remaining_qty / (execution_periods - period) * Decimal(str(urgency_factor))
                    period_qty = min(period_qty, remaining_qty)
                
                # Calculate market impact
                impact_rate = Decimal('0.01') * (period_qty / order.quantity) ** Decimal('0.5')
                impact_price = market_price * (Decimal('1') + impact_rate)
                
                # Add timing cost (random walk)
                timing_factor = volatility * Decimal(str((period + 1) ** 0.5)) * Decimal('0.1')
                timing_price = impact_price + (market_price * timing_factor * (Decimal('1') - Decimal(str(risk_aversion))))
                
                period_value = period_qty * timing_price
                total_executed += period_qty
                total_value += period_value
                
                self.log.debug(
                    "IS period executed",
                    period=period+1,
                    quantity=float(period_qty),
                    price=float(timing_price),
                    impact=float(impact_rate * 100)
                )
            
            average_price = total_value / total_executed if total_executed > 0 else Decimal('0')
            implementation_shortfall = (average_price - arrival_price) / arrival_price * 100
            
            # Decompose into market impact and timing cost
            market_impact = implementation_shortfall * Decimal('0.6')
            timing_cost = implementation_shortfall * Decimal('0.4')
            
            return ExecutionResult(
                algorithm_used="Implementation Shortfall",
                total_executed=total_executed,
                average_price=average_price,
                slippage=abs(implementation_shortfall),
                market_impact=abs(market_impact),
                timing_cost=abs(timing_cost),
                total_cost=abs(implementation_shortfall),
                confidence_score=0.88,
                execution_details={
                    "urgency": urgency,
                    "risk_aversion": risk_aversion,
                    "execution_periods": execution_periods,
                    "arrival_price": float(arrival_price),
                    "implementation_shortfall_bps": float(implementation_shortfall)
                }
            )
            
        except Exception as e:
            self.log.error("Implementation Shortfall execution failed", error=str(e))
            raise ExternalServiceError("Implementation Shortfall", str(e))
    
    def validate_parameters(self, parameters: ExecutionParameters) -> List[str]:
        """Validate Implementation Shortfall parameters."""
        violations = []
        
        if parameters.urgency and (parameters.urgency < 0 or parameters.urgency > 1):
            violations.append("Urgency must be between 0 and 1")
        
        if parameters.risk_aversion and (parameters.risk_aversion < 0 or parameters.risk_aversion > 1):
            violations.append("Risk aversion must be between 0 and 1")
        
        return violations
    
    def estimate_cost(
        self,
        order: Order,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Estimate Implementation Shortfall execution cost."""
        urgency = parameters.urgency or 0.5
        risk_aversion = parameters.risk_aversion or 0.5
        
        # Cost increases with urgency, decreases with risk aversion
        base_cost = Decimal('0.06')  # 6 basis points
        urgency_cost = Decimal(str(urgency)) * Decimal('0.03')
        risk_discount = Decimal(str(risk_aversion)) * Decimal('0.02')
        
        return base_cost + urgency_cost - risk_discount


# === STRATEGY FACTORY ===

class ExecutionStrategyFactory:
    """Factory for creating execution strategies."""
    
    _strategies: Dict[str, Type[ExecutionStrategy]] = {
        ExecutionAlgorithmType.TWAP: TWAPStrategy,
        ExecutionAlgorithmType.VWAP: VWAPStrategy,
        ExecutionAlgorithmType.IS: ImplementationShortfallStrategy,
    }
    
    @classmethod
    def create_strategy(cls, algorithm_type: ExecutionAlgorithmType) -> ExecutionStrategy:
        """Create execution strategy instance."""
        if algorithm_type not in cls._strategies:
            raise ConfigurationError(
                f"Unknown execution algorithm: {algorithm_type}",
                config_key="execution_algorithm"
            )
        
        strategy_class = cls._strategies[algorithm_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, algorithm_type: str, strategy_class: Type[ExecutionStrategy]):
        """Register custom execution strategy."""
        cls._strategies[algorithm_type] = strategy_class
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies."""
        return list(cls._strategies.keys())


# === PLUGIN SYSTEM ===

class Plugin(abc.ABC):
    """Base class for plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.log = EnterpriseLogger(__name__, f'plugin-{name.lower()}')
    
    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class EventPlugin(Plugin):
    """Plugin for handling order events."""
    
    @abc.abstractmethod
    async def on_order_created(self, order: Order, context: Dict[str, Any]) -> None:
        """Handle order created event."""
        pass
    
    @abc.abstractmethod
    async def on_order_executed(self, order: Order, execution: Execution, context: Dict[str, Any]) -> None:
        """Handle order executed event."""
        pass
    
    @abc.abstractmethod
    async def on_order_cancelled(self, order: Order, reason: str, context: Dict[str, Any]) -> None:
        """Handle order cancelled event."""
        pass


class NotificationPlugin(EventPlugin):
    """Example notification plugin."""
    
    def __init__(self):
        super().__init__("NotificationPlugin", "1.0.0")
        self.webhook_url: Optional[str] = None
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize notification plugin."""
        self.webhook_url = config.get('webhook_url')
        self.log.info("Notification plugin initialized", webhook_url=self.webhook_url)
    
    async def cleanup(self) -> None:
        """Cleanup notification plugin."""
        self.log.info("Notification plugin cleaned up")
    
    async def on_order_created(self, order: Order, context: Dict[str, Any]) -> None:
        """Send notification when order is created."""
        if self.webhook_url:
            message = {
                "event": "order_created",
                "order_id": order.id,
                "client_id": order.client_id,
                "symbol": order.symbol,
                "quantity": float(order.quantity),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            # In production, would send HTTP request to webhook_url
            self.log.info("Order created notification", message=message)
    
    async def on_order_executed(self, order: Order, execution: Execution, context: Dict[str, Any]) -> None:
        """Send notification when order is executed."""
        if self.webhook_url:
            message = {
                "event": "order_executed",
                "order_id": order.id,
                "execution_id": execution.id,
                "executed_quantity": float(execution.executed_quantity),
                "executed_price": float(execution.executed_price),
                "timestamp": execution.execution_time.isoformat()
            }
            self.log.info("Order executed notification", message=message)
    
    async def on_order_cancelled(self, order: Order, reason: str, context: Dict[str, Any]) -> None:
        """Send notification when order is cancelled."""
        if self.webhook_url:
            message = {
                "event": "order_cancelled",
                "order_id": order.id,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.log.info("Order cancelled notification", message=message)


class PluginManager:
    """Manager for loading and executing plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.event_plugins: List[EventPlugin] = []
        self.log = EnterpriseLogger(__name__, 'plugin-manager')
    
    async def load_plugin(self, plugin_class: Type[Plugin], config: Dict[str, Any]) -> None:
        """Load and initialize a plugin."""
        try:
            plugin = plugin_class()
            await plugin.initialize(config)
            
            self.plugins[plugin.name] = plugin
            
            if isinstance(plugin, EventPlugin):
                self.event_plugins.append(plugin)
            
            self.log.info("Plugin loaded", plugin_name=plugin.name, version=plugin.version)
            
        except Exception as e:
            self.log.error("Failed to load plugin", plugin_class=plugin_class.__name__, error=str(e))
            raise
    
    async def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            
            try:
                await plugin.cleanup()
                
                if isinstance(plugin, EventPlugin):
                    self.event_plugins.remove(plugin)
                
                del self.plugins[plugin_name]
                
                self.log.info("Plugin unloaded", plugin_name=plugin_name)
                
            except Exception as e:
                self.log.error("Failed to unload plugin", plugin_name=plugin_name, error=str(e))
    
    async def trigger_order_created(self, order: Order, context: Dict[str, Any] = None) -> None:
        """Trigger order created event for all plugins."""
        context = context or {}
        
        for plugin in self.event_plugins:
            if plugin.enabled:
                try:
                    await plugin.on_order_created(order, context)
                except Exception as e:
                    self.log.error(
                        "Plugin error on order created",
                        plugin_name=plugin.name,
                        order_id=order.id,
                        error=str(e)
                    )
    
    async def trigger_order_executed(
        self,
        order: Order,
        execution: Execution,
        context: Dict[str, Any] = None
    ) -> None:
        """Trigger order executed event for all plugins."""
        context = context or {}
        
        for plugin in self.event_plugins:
            if plugin.enabled:
                try:
                    await plugin.on_order_executed(order, execution, context)
                except Exception as e:
                    self.log.error(
                        "Plugin error on order executed",
                        plugin_name=plugin.name,
                        order_id=order.id,
                        error=str(e)
                    )
    
    async def trigger_order_cancelled(
        self,
        order: Order,
        reason: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Trigger order cancelled event for all plugins."""
        context = context or {}
        
        for plugin in self.event_plugins:
            if plugin.enabled:
                try:
                    await plugin.on_order_cancelled(order, reason, context)
                except Exception as e:
                    self.log.error(
                        "Plugin error on order cancelled",
                        plugin_name=plugin.name,
                        order_id=order.id,
                        error=str(e)
                    )
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins."""
        return {
            name: {
                "version": plugin.version,
                "enabled": plugin.enabled,
                "type": type(plugin).__name__
            }
            for name, plugin in self.plugins.items()
        }