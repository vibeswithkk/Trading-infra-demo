from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from .base import AsyncRepository
from ..core.models import Order

class OrderRepository(AsyncRepository[Order]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Order)