from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Numeric
from ..infra.db import Base

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY/SELL
    quantity: Mapped[Decimal] = mapped_column(Numeric(20,8))
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20,8), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="NEW", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20,8))
    price: Mapped[Decimal] = mapped_column(Numeric(20,8))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)