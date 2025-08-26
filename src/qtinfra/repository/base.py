from __future__ import annotations
from typing import Generic, TypeVar, Type, Sequence, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..infra.logging import JsonLogger

T = TypeVar("T")

class AsyncRepository(Generic[T]):
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
        self.log = JsonLogger(__name__)

    async def add(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def get(self, id_: Any) -> Optional[T]:
        res = await self.session.execute(select(self.model).where(self.model.id == id_))
        return res.scalar_one_or_none()

    async def list(self, limit: int = 100) -> Sequence[T]:
        res = await self.session.execute(select(self.model).limit(limit))
        return res.scalars().all()