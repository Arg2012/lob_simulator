"""Trade model.

A :class:`Trade` is emitted every time an incoming (aggressing) order matches
against a resting (passive) order in the book.
"""

from __future__ import annotations

from dataclasses import dataclass

from order import Side


@dataclass(frozen=True)
class Trade:
    """An execution between a taker (aggressor) and a maker (resting order).

    The trade always prints at the *resting* order's price, which is the
    standard convention: the passive side sets the price, the aggressor pays it.

    Attributes:
        timestamp: Logical time of the execution.
        price: Execution price (the resting/maker order's limit price).
        quantity: Number of units exchanged.
        aggressor_side: Side of the incoming order that triggered the match.
        taker_order_id: Order id of the aggressor.
        maker_order_id: Order id of the resting order that was hit.
    """

    timestamp: int
    price: float
    quantity: int
    aggressor_side: Side
    taker_order_id: int
    maker_order_id: int
