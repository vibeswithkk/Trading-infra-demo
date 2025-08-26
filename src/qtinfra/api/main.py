from __future__ import annotations
from decimal import Decimal
from fastapi import FastAPI
from pydantic import BaseModel
from ..infra.config import settings
from ..infra.db import engine, SessionLocal, Base
from ..core.models import Order
from ..repository.orders import OrderRepository
from ..router.sor import SmartOrderRouter, MockBroker

app = FastAPI(title=settings.app_name)

class OrderIn(BaseModel):
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal | None = None

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/orders")
async def create_order(data: OrderIn):
    async with SessionLocal() as session:
        repo = OrderRepository(session)
        order = Order(symbol=data.symbol, side=data.side, quantity=data.quantity, price=data.price)
        await repo.add(order)
        await session.commit()
        return {"id": order.id, "status": "NEW"}

@app.post("/route")
async def route_order(data: OrderIn):
    sor = SmartOrderRouter(brokers={
        "BRK1": MockBroker("BRK1", 15),
        "BRK2": MockBroker("BRK2", 25),
        "BRK3": MockBroker("BRK3", 35),
    })
    result = await sor.route(symbol=data.symbol, side=data.side, quantity=data.quantity, price=data.price)
    return result