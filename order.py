"""Order model for the limit order book.

An :class:`Order` is the fundamental unit that flows through the simulator.
It carries everything the matching engine needs to place, match and track it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    """Which side of the book an order sits on."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        """Return the opposite side (BUY <-> SELL)."""
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    """Whether an order rests in the book (LIMIT) or crosses it (MARKET)."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    """A single order.

    Attributes:
        order_id: Unique identifier used for the order-id lookup and cancels.
        timestamp: Logical time the order entered the book (used for FIFO).
        side: BUY or SELL.
        order_type: LIMIT or MARKET.
        quantity: Original size of the order.
        price: Limit price. ``None`` for market orders (which have no price).
        remaining: Quantity still open; defaults to ``quantity`` on creation.
    """

    order_id: int
    timestamp: int
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    remaining: int = field(default=-1)

    def __post_init__(self) -> None:
        if self.remaining < 0:
            self.remaining = self.quantity
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

    @property
    def is_filled(self) -> bool:
        """True once the order has no remaining quantity."""
        return self.remaining == 0

    @property
    def filled_quantity(self) -> int:
        """How much of the order has been executed so far."""
        return self.quantity - self.remaining
