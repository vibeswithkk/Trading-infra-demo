import pytest
from decimal import Decimal
from qtinfa.router.sor import SmartOrderRouter, MockBroker

@pytest.mark.asyncio
async def test_router_selects_fastest():
    sor = SmartOrderRouter(brokers={
        "B1": MockBroker("B1", 15),
        "B2": MockBroker("B2", 45),
    })
    res = await sor.route(symbol="AAPL", side="BUY", quantity=Decimal("10"), price=None)
    assert res["selected_broker"] == "B1"