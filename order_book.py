"""The central limit order book.

The book stores resting orders on two sides. Each *price level* is a FIFO queue
(``deque``) so that time priority within a price is preserved automatically:
new orders append to the back, matches consume from the front.

* ``bids`` - BUY orders; best (highest) price has priority.
* ``asks`` - SELL orders; best (lowest) price has priority.
* ``orders`` - order-id -> Order lookup, used for O(1) cancels.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from order import Order, Side


class OrderBook:
    """A price-time priority limit order book."""

    def __init__(self) -> None:
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}
        self.orders: Dict[int, Order] = {}

    # --- internal helpers ----------------------------------------------------

    def _side_book(self, side: Side) -> Dict[float, Deque[Order]]:
        return self.bids if side is Side.BUY else self.asks

    # --- mutations -----------------------------------------------------------

    def add_resting(self, order: Order) -> None:
        """Append a (non-marketable) order to the back of its price level."""
        book = self._side_book(order.side)
        book.setdefault(order.price, deque()).append(order)
        self.orders[order.order_id] = order

    def remove(self, order_id: int) -> Optional[Order]:
        """Remove an order by id. Returns the order, or ``None`` if unknown."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return None
        book = self._side_book(order.side)
        level = book.get(order.price)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:
                pass
            if not level:
                del book[order.price]
        return order

    # --- top of book ---------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        """Highest resting bid price, or ``None`` if the bid side is empty."""
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        """Lowest resting ask price, or ``None`` if the ask side is empty."""
        return min(self.asks) if self.asks else None

    def spread(self) -> Optional[float]:
        """Best ask minus best bid, or ``None`` if either side is empty."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return round(ask - bid, 10)

    def mid(self) -> Optional[float]:
        """Midpoint between best bid and best ask, or ``None`` if one is empty."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    # --- depth ---------------------------------------------------------------

    @staticmethod
    def _depth(book: Dict[float, Deque[Order]]) -> int:
        return sum(o.remaining for level in book.values() for o in level)

    def total_bid_depth(self) -> int:
        """Total remaining quantity resting on the bid side."""
        return self._depth(self.bids)

    def total_ask_depth(self) -> int:
        """Total remaining quantity resting on the ask side."""
        return self._depth(self.asks)

    def depth_at(self, side: Side, price: float) -> int:
        """Remaining quantity resting at a specific price on ``side``."""
        level = self._side_book(side).get(round(price, 2))
        return sum(o.remaining for o in level) if level else 0

    # --- inspection ----------------------------------------------------------

    def levels(self, side: Side, depth: int = 5) -> List[Tuple[float, int]]:
        """Return the top ``depth`` (price, size) levels of ``side``.

        Bids are returned high-to-low, asks low-to-high (best first).
        """
        book = self._side_book(side)
        prices = sorted(book, reverse=(side is Side.BUY))
        out: List[Tuple[float, int]] = []
        for p in prices[:depth]:
            out.append((p, sum(o.remaining for o in book[p])))
        return out

    def snapshot(self, depth: int = 5) -> str:
        """A human-readable ladder view of the top of the book."""
        asks = self.levels(Side.SELL, depth)[::-1]
        bids = self.levels(Side.BUY, depth)
        lines = ["price      size   side"]
        for p, s in asks:
            lines.append(f"{p:>8.2f}  {s:>6}   ASK")
        for p, s in bids:
            lines.append(f"{p:>8.2f}  {s:>6}   BID")
        return "\n".join(lines)
