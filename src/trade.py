"""The `Trade` record produced whenever two orders match.

A trade always occurs at the *resting* (maker) order's price — the taker crosses
the spread and receives price improvement, exactly as on a real exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

from .order import Side


@dataclass(frozen=True)
class Trade:
    """An executed match between a resting (maker) and aggressing (taker) order."""

    timestamp: float
    price: float
    quantity: int
    maker_order_id: int
    taker_order_id: int
    aggressor: Side  # the side that crossed the spread / took liquidity

    @property
    def notional(self) -> float:
        return self.price * self.quantity
