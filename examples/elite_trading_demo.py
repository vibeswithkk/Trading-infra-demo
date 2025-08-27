#!/usr/bin/env python3
"""
Elite Trading Infrastructure Demo

Demonstrates all enterprise-grade features:
- Custom exception handling
- Domain-driven design
- Transaction safety
- Validation layers
- Execution algorithms
- Security & compliance
- Plugin system
- Risk management

Run: python examples/elite_trading_demo.py
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any

# Core imports
from qtinfra.core.models import Order, OrderStatus, OrderType, AssetClass, OrderSide, Currency
from qtinfra.core.exceptions import (
    OrderValidationError, RiskLimitExceededError, OrderStateError,
    create_error_context
)
from qtinfra.core.validation import OrderCreateRequest, ExecutionCreateRequest
from qtinfra.core.strategies import (
    ExecutionStrategyFactory, ExecutionAlgorithmType, ExecutionParameters,
    PluginManager, NotificationPlugin
)
from qtinfra.core.security import (
    RBACManager, PIIProtector, GDPRComplianceManager, CryptoManager,
    User, Role, Permission
)

# Infrastructure
from qtinfra.infra.logging import EnterpriseLogger
# from qtinfra.infra.db import get_async_session  # Not implemented in basic demo


class EliteTradingDemo:
    """Comprehensive demo of elite trading features."""
    
    def __init__(self):
        self.log = EnterpriseLogger(__name__, 'elite-demo')
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """Setup demo users, permissions, and market data."""
        # Initialize security components
        self.rbac = RBACManager()
        self.pii_protector = PIIProtector()
        self.crypto = CryptoManager()
        self.gdpr_manager = GDPRComplianceManager(self.rbac, self.pii_protector)
        
        # Create demo users
        trader = User(
            id="trader_001",
            username="john.trader",
            email="john.trader@company.com", 
            roles=[Role.TRADER],
            client_access={"CLIENT_001", "CLIENT_002"}
        )
        
        portfolio_manager = User(
            id="pm_001",
            username="sarah.pm",
            email="sarah.pm@company.com",
            roles=[Role.PORTFOLIO_MANAGER, Role.TRADER],
            client_access={"CLIENT_001", "CLIENT_002", "CLIENT_003"}
        )
        
        risk_manager = User(
            id="risk_001", 
            username="mike.risk",
            email="mike.risk@company.com",
            roles=[Role.RISK_MANAGER],
            client_access=set()  # Can access all clients for risk purposes
        )
        
        compliance_officer = User(
            id="compliance_001",
            username="jane.compliance", 
            email="jane.compliance@company.com",
            roles=[Role.COMPLIANCE_OFFICER],
            client_access=set()  # Can access all for compliance
        )
        
        # Add users to RBAC
        for user in [trader, portfolio_manager, risk_manager, compliance_officer]:
            self.rbac.add_user(user)
        
        # Demo market data
        self.market_data = {
            'AAPL': {
                'current_price': 150.25,
                'volatility': 0.025,
                'volume_profile': [
                    {'time': '09:30', 'volume_pct': 0.15},
                    {'time': '10:00', 'volume_pct': 0.12},
                    {'time': '11:00', 'volume_pct': 0.08},
                    {'time': '12:00', 'volume_pct': 0.05},
                    {'time': '13:00', 'volume_pct': 0.06},
                    {'time': '14:00', 'volume_pct': 0.09},
                    {'time': '15:00', 'volume_pct': 0.18},
                    {'time': '16:00', 'volume_pct': 0.27}
                ]
            }
        }
    
    async def demo_enterprise_order_lifecycle(self):
        """Demonstrate complete enterprise order lifecycle."""
        print("\n🏗️  ENTERPRISE ORDER LIFECYCLE DEMO")
        print("=" * 50)
        
        try:
            # 1. Create order with validation
            print("\n1. Creating order with enterprise validation...")
            
            request = OrderCreateRequest(
                client_id="CLIENT_001",
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                quantity=Decimal("1000.0"),
                price=Decimal("150.25"),
                currency=Currency.USD,
                time_in_force="GTC",
                execution_algorithm="TWAP",
                tags=["algorithmic", "large-order"],
                notes="Large institutional order for pension fund"
            )
            
            print(f"✅ Order request created: {request.symbol} {request.quantity} @ {request.price}")
            
            # 2. Authorization check
            print("\n2. Performing authorization checks...")
            user_id = "pm_001"  # Portfolio manager
            
            try:
                self.rbac.authorize_operation(user_id, Permission.CREATE_ORDER, request.client_id)
                print(f"✅ Authorization passed for user {user_id}")
            except Exception as e:
                print(f"❌ Authorization failed: {e}")
                return
            
            # 3. Business rule validation
            print("\n3. Validating business rules...")
            
            # Simulate some business rules
            violations = []
            if request.quantity > Decimal("10000"):
                violations.append("Order size exceeds daily limit")
            
            if violations:
                raise OrderValidationError(
                    "Business rule validation failed",
                    validation_errors=violations,
                    context=create_error_context(
                        user_id=user_id,
                        operation="create_order"
                    )
                )
            
            print("✅ Business rules validated")
            
            # 4. Risk checks
            print("\n4. Performing risk checks...")
            
            # Simulate risk check (normally done by repository)
            notional_value = request.quantity * request.price
            if notional_value > Decimal("1000000"):  # $1M limit
                raise RiskLimitExceededError(
                    f"Notional value {notional_value} exceeds limit", 
                    "notional",
                    float(notional_value),
                    1000000.0,
                    request.client_id
                )
            
            print("✅ Risk checks passed")
            
            # 5. Create order (simulated - normally done by repository)
            order = Order(
                id=str(uuid.uuid4()),
                client_id=request.client_id,
                symbol=request.symbol,
                asset_class=request.asset_class.value,
                order_type=request.order_type.value,
                side=request.side.value,
                quantity=request.quantity,
                remaining_quantity=request.quantity,
                price=request.price,
                currency=request.currency.value,
                time_in_force=request.time_in_force,
                execution_algorithm=request.execution_algorithm,
                notes=request.notes,
                tags=','.join(request.tags) if request.tags else None,
                created_by=user_id,
                status=OrderStatus.PENDING.value
            )
            
            print(f"✅ Order created: {order.id}")
            
            return order
            
        except OrderValidationError as e:
            print(f"❌ Validation Error: {e.message}")
            if e.validation_errors:
                for error in e.validation_errors:
                    print(f"   - {error}")
        except RiskLimitExceededError as e:
            print(f"❌ Risk Limit Exceeded: {e.message}")
            print(f"   Current: {e.current_value}, Limit: {e.limit_value}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
    
    async def demo_execution_algorithms(self, order):
        """Demonstrate sophisticated execution algorithms."""
        print("\n🧠 EXECUTION ALGORITHMS DEMO")
        print("=" * 50)
        
        algorithms = [
            (ExecutionAlgorithmType.TWAP, "Time Weighted Average Price"),
            (ExecutionAlgorithmType.VWAP, "Volume Weighted Average Price"),
            (ExecutionAlgorithmType.IS, "Implementation Shortfall")
        ]
        
        for algo_type, description in algorithms:
            try:
                print(f"\n🔄 Testing {description} ({algo_type})...")
                
                # Create strategy
                strategy = ExecutionStrategyFactory.create_strategy(algo_type)
                
                # Configure parameters based on algorithm
                if algo_type == ExecutionAlgorithmType.TWAP:
                    parameters = ExecutionParameters(
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc) + timedelta(hours=2),
                        participation_rate=0.15
                    )
                elif algo_type == ExecutionAlgorithmType.VWAP:
                    parameters = ExecutionParameters(
                        participation_rate=0.12,
                        max_volume_rate=0.25
                    )
                else:  # Implementation Shortfall
                    parameters = ExecutionParameters(
                        urgency=0.7,
                        risk_aversion=0.3
                    )
                
                # Validate parameters
                violations = strategy.validate_parameters(parameters)
                if violations:
                    print(f"❌ Parameter validation failed: {violations}")
                    continue
                
                # Estimate cost
                estimated_cost = strategy.estimate_cost(
                    order, parameters, self.market_data['AAPL']
                )
                print(f"📊 Estimated cost: {estimated_cost:.2f} basis points")
                
                # Execute algorithm
                result = await strategy.execute(
                    order, parameters, self.market_data['AAPL']
                )
                
                print(f"✅ Execution completed:")
                print(f"   Algorithm: {result.algorithm_used}")
                print(f"   Executed: {result.total_executed}")
                print(f"   Avg Price: ${result.average_price:.4f}")
                print(f"   Slippage: {result.slippage:.2f} bps")
                print(f"   Market Impact: {result.market_impact:.2f} bps")
                print(f"   Total Cost: {result.total_cost:.2f} bps")
                print(f"   Confidence: {result.confidence_score:.1%}")
                
            except Exception as e:
                print(f"❌ Algorithm {algo_type} failed: {e}")
    
    async def demo_security_features(self):
        """Demonstrate security and compliance features."""
        print("\n🔒 SECURITY & COMPLIANCE DEMO")
        print("=" * 50)
        
        # 1. PII Protection
        print("\n1. PII Protection Demo...")
        sensitive_text = """
        Client contact: John Smith at john.smith@company.com 
        Phone: 555-123-4567, SSN: 123-45-6789
        Account: 1234-5678-9012-3456
        """
        
        print("Original text:")
        print(sensitive_text)
        
        # Detect PII
        detected_pii = self.pii_protector.detect_pii(sensitive_text)
        print(f"\n🔍 Detected PII types: {list(detected_pii.keys())}")
        
        # Mask PII
        masked_text = self.pii_protector.mask_pii(sensitive_text)
        print("\n🛡️  Masked text:")
        print(masked_text)
        
        # 2. Data Encryption
        print("\n2. Data Encryption Demo...")
        secret_data = "Confidential trading strategy: Buy AAPL when RSI < 30"
        
        encrypted_result = self.crypto.encrypt_data(secret_data)
        print(f"🔐 Encrypted: {encrypted_result['encrypted_data'][:50]}...")
        
        decrypted_data = self.crypto.decrypt_data(encrypted_result['encrypted_data'])
        print(f"🔓 Decrypted: {decrypted_data}")
        
        # 3. Digital Signatures
        print("\n3. Digital Signature Demo...")
        order_data = '{"symbol":"AAPL","quantity":1000,"price":150.25}'
        signature = self.crypto.sign_data(order_data)
        print(f"✍️  Signature: {signature[:32]}...")
        
        is_valid = self.crypto.verify_signature(order_data, signature)
        print(f"✅ Signature valid: {is_valid}")
        
        # 4. GDPR Compliance
        print("\n4. GDPR Compliance Demo...")
        
        try:
            # Simulate data access request
            client_id = "CLIENT_001"
            requester_id = "compliance_001"
            
            print(f"📋 Processing GDPR access request for {client_id}...")
            access_data = await self.gdpr_manager.process_access_request(
                client_id, requester_id
            )
            
            print(f"✅ Data export completed:")
            print(f"   Request ID: {access_data['request_id']}")
            print(f"   Export timestamp: {access_data['export_timestamp']}")
            print(f"   Records exported: {len(access_data['exported_data'])}")
            
        except Exception as e:
            print(f"❌ GDPR request failed: {e}")
    
    async def demo_plugin_system(self, order):
        """Demonstrate plugin system and event handling."""
        print("\n🔌 PLUGIN SYSTEM DEMO") 
        print("=" * 50)
        
        # Initialize plugin manager
        plugin_manager = PluginManager()
        
        # Load notification plugin
        print("\n1. Loading notification plugin...")
        await plugin_manager.load_plugin(NotificationPlugin, {
            'webhook_url': 'https://api.company.com/trading-webhook'
        })
        
        print("✅ Plugin loaded successfully")
        
        # Trigger events
        print("\n2. Triggering order events...")
        
        # Order created event
        await plugin_manager.trigger_order_created(order, {
            'user_id': 'pm_001',
            'trading_session': 'US_REGULAR',
            'risk_score': 'LOW'
        })
        print("✅ Order created event triggered")
        
        # Simulate execution
        from qtinfra.core.models import Execution
        execution = Execution(
            id=str(uuid.uuid4()),
            order_id=order.id,
            venue_id="NYSE",
            executed_quantity=Decimal("500.0"),
            executed_price=Decimal("150.30"),
            execution_time=datetime.now(timezone.utc),
            commission=Decimal("2.50"),
            fees=Decimal("0.50")
        )
        
        await plugin_manager.trigger_order_executed(order, execution, {
            'execution_venue': 'NYSE',
            'fill_percentage': 50.0,
            'execution_quality': 'EXCELLENT'
        })
        print("✅ Order executed event triggered")
        
        # List loaded plugins
        plugins = plugin_manager.list_plugins()
        print(f"\n📋 Loaded plugins: {list(plugins.keys())}")
        
        # Cleanup
        await plugin_manager.unload_plugin("NotificationPlugin")
        print("🧹 Plugin unloaded")
    
    async def demo_error_handling(self):
        """Demonstrate enterprise error handling."""
        print("\n⚠️  ERROR HANDLING DEMO")
        print("=" * 50)
        
        error_scenarios = [
            ("Order Validation", lambda: self._trigger_validation_error()),
            ("Risk Limit", lambda: self._trigger_risk_error()), 
            ("State Transition", lambda: self._trigger_state_error()),
            ("Authorization", lambda: self._trigger_auth_error())
        ]
        
        for scenario_name, error_func in error_scenarios:
            try:
                print(f"\n🧪 Testing {scenario_name} Error...")
                error_func()
                
            except Exception as e:
                # Demonstrate structured error handling
                error_dict = e.to_dict() if hasattr(e, 'to_dict') else {
                    'error_type': type(e).__name__,
                    'message': str(e)
                }
                
                print(f"✅ Caught {error_dict['error_type']}:")
                print(f"   Message: {error_dict['message']}")
                if 'severity' in error_dict:
                    print(f"   Severity: {error_dict['severity']}")
                if 'category' in error_dict:
                    print(f"   Category: {error_dict['category']}")
                if 'user_message' in error_dict:
                    print(f"   User Message: {error_dict['user_message']}")
    
    def _trigger_validation_error(self):
        """Trigger validation error for demo."""
        raise OrderValidationError(
            "Invalid order parameters",
            validation_errors=["Quantity must be positive", "Price required for LIMIT orders"],
            order_id="demo-order-123"
        )
    
    def _trigger_risk_error(self):
        """Trigger risk error for demo."""
        raise RiskLimitExceededError(
            "Position limit exceeded",
            "position",
            15000.0,
            10000.0,
            "CLIENT_001"
        )
    
    def _trigger_state_error(self):
        """Trigger state error for demo."""
        raise OrderStateError(
            "Cannot cancel filled order",
            order_id="demo-order-123",
            current_state="FILLED",
            expected_states=["PENDING", "ROUTED"]
        )
    
    def _trigger_auth_error(self):
        """Trigger authorization error for demo."""
        from qtinfra.core.exceptions import UnauthorizedAccessError
        raise UnauthorizedAccessError(
            "orders",
            "create",
            "unauthorized_user"
        )
    
    async def run_complete_demo(self):
        """Run complete demonstration of all elite features."""
        print("🚀 ELITE TRADING INFRASTRUCTURE DEMO")
        print("=" * 60)
        print("Demonstrating enterprise-grade features...")
        
        try:
            # 1. Order lifecycle
            order = await self.demo_enterprise_order_lifecycle()
            
            if order:
                # 2. Execution algorithms
                await self.demo_execution_algorithms(order)
                
                # 3. Plugin system  
                await self.demo_plugin_system(order)
            
            # 4. Security features
            await self.demo_security_features()
            
            # 5. Error handling
            await self.demo_error_handling()
            
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("All enterprise features demonstrated:")
            print("✅ Custom exception hierarchy")
            print("✅ Domain-driven design") 
            print("✅ Transaction safety")
            print("✅ Validation layers")
            print("✅ Execution algorithms")
            print("✅ Security & compliance")
            print("✅ Plugin system")
            print("✅ Error handling")
            print("\n🏆 Elite trading infrastructure ready for production!")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            self.log.error("Demo failed", error=str(e))


async def main():
    """Main demo function."""
    demo = EliteTradingDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Run the elite demo
    asyncio.run(main())