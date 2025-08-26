import asyncio
import pytest
from qtinfa.infra.db import SessionLocal, engine, Base
from qtinfa.core.models import Order
from qtinfa.repository.orders import OrderRepository

@pytest.mark.asyncio
async def test_repository_add_and_get():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with SessionLocal() as session:
        repo = OrderRepository(session)
        o = Order(symbol="AAPL", side="BUY", quantity=1, price=None)
        await repo.add(o)
        await session.commit()
        assert o.id is not None
        got = await repo.get(o.id)
        assert got is not None
        assert got.symbol == "AAPL"