"""A minimal, symmetric market maker.

The strategy is intentionally trivial: on every step it cancels its previous
quotes and re-posts a fresh bid and ask a fixed number of ticks either side of
the current mid. There is no inventory skew, no alpha - the point is only to
have an agent that continuously provides two-sided liquidity so we can measure
inventory and PnL, not to trade well.

Inventory & PnL accounting uses the average-cost method:

* **Inventory** - signed position (positive = long).
* **Realised PnL** - locked in whenever a fill *reduces* the position, valued
  against the running average entry price.
* **Unrealised PnL** - open position marked to the current mid.
* **Total PnL** - realised + unrealised.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from events import Event, add_order, cancel_order
from order import Side
from trade import Trade


class MarketMaker:
    """Continuously quotes both sides and tracks inventory / PnL."""

    def __init__(self, id_gen: Iterator[int], half_spread_ticks: int = 2,
                 quote_size: int = 2, tick: float = 0.01,
                 fallback_price: float = 100.0) -> None:
        self.id_gen = id_gen
        self.half_spread_ticks = half_spread_ticks
        self.quote_size = quote_size
        self.tick = tick

        # Position / PnL state.
        self.inventory: int = 0
        self.avg_price: float = 0.0
        self.cash: float = 0.0
        self.realised: float = 0.0
        self.unrealised: float = 0.0
        self.total: float = 0.0

        # Quote bookkeeping.
        self.my_orders: set[int] = set()
        self.bid_id: Optional[int] = None
        self.ask_id: Optional[int] = None
        self.submitted_qty: int = 0
        self.filled_qty: int = 0
        self.last_mid: float = fallback_price

    # --- quoting -------------------------------------------------------------

    def requote(self, book, timestamp: int) -> List[Event]:
        """Cancel old quotes and post fresh two-sided quotes around the mid.

        Returns the events to feed into the engine (two cancels + two adds).
        """
        events: List[Event] = []
        if self.bid_id is not None:
            events.append(cancel_order(timestamp, self.bid_id))
        if self.ask_id is not None:
            events.append(cancel_order(timestamp, self.ask_id))

        mid = book.mid()
        if mid is None:
            mid = self.last_mid
        self.last_mid = mid

        bid_price = round(mid - self.half_spread_ticks * self.tick, 2)
        ask_price = round(mid + self.half_spread_ticks * self.tick, 2)

        self.bid_id = next(self.id_gen)
        self.ask_id = next(self.id_gen)
        events.append(add_order(timestamp, self.bid_id, Side.BUY, bid_price,
                                self.quote_size))
        events.append(add_order(timestamp, self.ask_id, Side.SELL, ask_price,
                                self.quote_size))
        self.my_orders.add(self.bid_id)
        self.my_orders.add(self.ask_id)
        self.submitted_qty += 2 * self.quote_size
        return events

    # --- fills / accounting --------------------------------------------------

    def on_trades(self, trades: List[Trade]) -> None:
        """Update inventory and PnL from any trades that touch our orders."""
        for tr in trades:
            if tr.taker_order_id in self.my_orders:
                my_side = tr.aggressor_side
            elif tr.maker_order_id in self.my_orders:
                my_side = tr.aggressor_side.opposite
            else:
                continue
            self._apply_fill(my_side, tr.price, tr.quantity)

    def _apply_fill(self, side: Side, price: float, qty: int) -> None:
        signed = qty if side is Side.BUY else -qty
        self.cash -= signed * price
        old = self.inventory

        if old == 0 or (old > 0) == (signed > 0):
            # Opening or adding to the position: update the average entry price.
            self.inventory = old + signed
            self.avg_price = (self.avg_price * old + price * signed) / self.inventory
        else:
            # Reducing / closing: realise PnL on the closed portion.
            closing = min(abs(signed), abs(old))
            direction = 1 if old > 0 else -1
            self.realised += closing * (price - self.avg_price) * direction
            self.inventory = old + signed
            if self.inventory != 0 and (self.inventory > 0) != (old > 0):
                # Flipped through zero: the remainder opens a new position.
                self.avg_price = price

        self.filled_qty += qty

    def mark(self, mid: Optional[float]) -> None:
        """Mark the open position to ``mid`` and refresh unrealised/total PnL."""
        if mid is None:
            mid = self.last_mid
        self.unrealised = self.inventory * (mid - self.avg_price)
        self.total = self.realised + self.unrealised

    @property
    def fill_rate(self) -> float:
        """Fraction of quoted volume that actually executed."""
        return self.filled_qty / self.submitted_qty if self.submitted_qty else 0.0
