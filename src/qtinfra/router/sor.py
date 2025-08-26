from __future__ import annotations
import asyncio, random
from typing import Dict, Any, List
from decimal import Decimal
from ..infra.logging import JsonLogger

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