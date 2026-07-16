"""The limit order book.

Design
------
Each side of the book is a dict mapping ``price -> PriceLevel``. A `PriceLevel`
holds a FIFO queue (``collections.deque``) of resting orders, giving us
*price-time priority*: better prices match first, and within a price level the
earliest-arriving order matches first.

To find the best bid / ask quickly we keep, alongside each dict, a *sorted* list
of the live prices and maintain it with `bisect`. That gives:

* best bid / ask ....... O(1)   (peek the end of the sorted list)
* add a new price level  O(k)   (bisect insort)
* cancel / drop a level  O(k)   (bisect locate + list pop)

where k is the number of *distinct* price levels — small and roughly constant for
a realistic book. Order lookup / cancellation by id is O(1) via `self.orders`.

The book itself only *stores* liquidity; crossing/matching lives in
`matching_engine.py` so the two concerns stay cleanly separated.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .order import Order, Side


@dataclass
class PriceLevel:
    """A FIFO queue of resting orders that all share the same price."""

    price: float
    orders: Deque[Order] = field(default_factory=deque)
    total_quantity: int = 0

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.total_quantity += order.quantity

    def remove(self, order: Order) -> None:
        # Identity-based removal (Order uses eq=False). O(n) in the queue length,
        # which is fine at simulation scale.
        self.orders.remove(order)
        self.total_quantity -= order.quantity

    def __len__(self) -> int:
        return len(self.orders)


class OrderBook:
    """A two-sided limit order book with O(1) top-of-book and id lookup."""

    def __init__(self) -> None:
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        # Live prices per side, kept sorted ascending so best bid = last,
        # best ask = first.
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []
        # id -> Order, for O(1) cancellation and inspection.
        self.orders: Dict[int, Order] = {}

    # ------------------------------------------------------------------ helpers
    def _side_maps(self, side: Side) -> Tuple[Dict[float, PriceLevel], List[float]]:
        """Return the (levels dict, sorted-prices list) for a given side."""
        if side is Side.BUY:
            return self.bids, self._bid_prices
        return self.asks, self._ask_prices

    # ------------------------------------------------------------- mutating ops
    def add_limit_order(self, order: Order) -> None:
        """Rest a (non-marketable remainder of a) limit order in the book."""
        levels, prices = self._side_maps(order.side)
        level = levels.get(order.price)
        if level is None:
            level = PriceLevel(order.price)
            levels[order.price] = level
            bisect.insort(prices, order.price)
        level.add(order)
        self.orders[order.order_id] = order

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Remove a resting order by id. Returns the order, or None if unknown."""
        order = self.orders.get(order_id)
        if order is None:
            return None
        levels, prices = self._side_maps(order.side)
        level = levels[order.price]
        level.remove(order)
        if len(level) == 0:
            self._drop_level(order.side, order.price)
        del self.orders[order_id]
        return order

    def _drop_level(self, side: Side, price: float) -> None:
        """Delete an emptied price level and its entry in the sorted-price list."""
        levels, prices = self._side_maps(side)
        if price in levels and len(levels[price]) == 0:
            del levels[price]
            idx = bisect.bisect_left(prices, price)
            if idx < len(prices) and prices[idx] == price:
                prices.pop(idx)

    # ----------------------------------------------------------------- queries
    def best_bid(self) -> Optional[float]:
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def midpoint(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def bid_depth(self, levels: Optional[int] = None) -> int:
        """Total resting bid quantity across the top `levels` prices (all if None)."""
        prices = list(reversed(self._bid_prices))  # best (highest) first
        if levels is not None:
            prices = prices[:levels]
        return sum(self.bids[p].total_quantity for p in prices)

    def ask_depth(self, levels: Optional[int] = None) -> int:
        """Total resting ask quantity across the top `levels` prices (all if None)."""
        prices = self._ask_prices  # best (lowest) first
        if levels is not None:
            prices = prices[:levels]
        return sum(self.asks[p].total_quantity for p in prices)

    def depth_snapshot(self, levels: int = 5) -> Tuple[List[Tuple[float, int]],
                                                       List[Tuple[float, int]]]:
        """Return (bids, asks) as [(price, qty), ...], best price first per side."""
        bid_prices = list(reversed(self._bid_prices))[:levels]
        ask_prices = self._ask_prices[:levels]
        bids = [(p, self.bids[p].total_quantity) for p in bid_prices]
        asks = [(p, self.asks[p].total_quantity) for p in ask_prices]
        return bids, asks

    # ---------------------------------------------------------------- invariant
    def check_invariants(self) -> None:
        """Assert internal consistency. Used by tests and as a runtime guard.

        * The book is never crossed (best bid < best ask).
        * Every `PriceLevel.total_quantity` equals the sum of its orders.
        * The sorted-price lists mirror the level dicts exactly.
        * `self.orders` contains precisely the resting orders.
        """
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None:
            assert bid < ask, f"Crossed book: bid {bid} >= ask {ask}"

        counted = 0
        for side in (Side.BUY, Side.SELL):
            levels, prices = self._side_maps(side)
            assert sorted(levels.keys()) == prices, "Sorted-price list out of sync"
            for price, level in levels.items():
                assert len(level) > 0, "Empty level was not dropped"
                assert level.total_quantity == sum(o.quantity for o in level.orders)
                for o in level.orders:
                    assert o.price == price and o.side is side
                    assert self.orders.get(o.order_id) is o
                    counted += 1
        assert counted == len(self.orders), "orders lookup out of sync with book"
