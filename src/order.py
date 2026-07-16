"""Order primitives: the `Side` / `OrderType` enums and the `Order` dataclass.

An `Order` is the atomic unit that flows through the whole system. Both resting
limit orders and aggressive market orders are represented by the same object;
they differ only in `order_type` (and whether a price is attached).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(Enum):
    """Which side of the book an order sits on / aggresses against."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    """A resting priced order (LIMIT) or an immediate liquidity taker (MARKET)."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


# `eq=False` gives identity-based equality, which is what we want: two distinct
# orders are never "equal" even if their fields momentarily coincide. This keeps
# `deque.remove(order)` / `list.remove(order)` O(n)-but-correct on object identity.
@dataclass(eq=False)
class Order:
    """A single order.

    Attributes
    ----------
    order_id:   Unique identifier used for the book's O(1) lookup / cancellation.
    side:       BUY or SELL.
    quantity:   Remaining, unfilled size. Mutated in place as the order fills.
    price:      Limit price. `None` for market orders.
    order_type: LIMIT or MARKET.
    timestamp:  Event time the order entered the book (used as the trade time).
    owner:      Tag identifying the submitter, e.g. "background" or "MM". Lets the
                simulator attribute fills to the market-making agent.
    """

    order_id: int
    side: Side
    quantity: int
    price: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    timestamp: float = 0.0
    owner: str = "background"

    def __post_init__(self) -> None:
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price.")
        if self.quantity <= 0:
            raise ValueError("Order quantity must be strictly positive.")

    @property
    def is_buy(self) -> bool:
        return self.side is Side.BUY
