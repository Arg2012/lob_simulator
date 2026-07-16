"""A simple market-making agent.

The agent continuously posts a two-sided quote (one bid, one ask) around the
current mid-price and earns the spread when both sides fill. It tracks:

* **inventory**       — net signed position (+long / -short).
* **realised PnL**    — locked-in profit from round-trips, via average-cost.
* **unrealised PnL**  — mark-to-market of the open inventory at the current mid.

Inventory risk is managed with a linear **skew**: as the agent accumulates a long
position it lowers both quotes (making it keener to sell and less keen to buy),
which mean-reverts inventory back toward zero. It also stops quoting the side that
would breach `max_inventory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .matching_engine import MatchingEngine
from .order import Order, OrderType, Side


@dataclass
class MarketMaker:
    """A single-instrument, symmetric-quote market maker with inventory skew."""

    half_spread: float = 0.02   # distance of each quote from the mid (price units)
    order_size: int = 5         # size posted on each side
    max_inventory: int = 100    # hard cap on |inventory|
    skew_ticks: float = 0.005   # quote shift per unit of inventory
    tick: float = 0.01
    owner: str = "MM"

    # --- mutable state ---------------------------------------------------------
    inventory: int = 0
    cash: float = 0.0            # running cash flow (+ from sells, - from buys)
    realised_pnl: float = 0.0
    avg_price: float = 0.0       # average cost of the current open position
    active_bid_id: Optional[int] = None
    active_ask_id: Optional[int] = None
    _next_id: int = field(default=10_000_000)  # MM ids live in a reserved range

    # ------------------------------------------------------------------ fills
    def on_fill(self, side: Side, price: float, quantity: int) -> None:
        """Update inventory, cash and realised PnL when one of our quotes fills.

        `side` is the side *we* traded: BUY means we bought (inventory up).
        Realised PnL is booked on the portion of the trade that reduces or closes
        the existing position (average-cost accounting).
        """
        direction = 1 if side is Side.BUY else -1
        signed = direction * quantity
        prev = self.inventory
        new = prev + signed

        # Cash: buying spends cash, selling receives it.
        self.cash += -direction * price * quantity

        increasing = prev == 0 or (prev > 0) == (signed > 0)
        if increasing:
            # Blend into the average cost of the (growing) position.
            total_cost = self.avg_price * abs(prev) + price * quantity
            self.avg_price = total_cost / abs(new) if new != 0 else 0.0
        else:
            # Reducing / flipping: realise PnL on the closed portion.
            closed = min(quantity, abs(prev))
            if prev > 0:              # we were long and are now selling
                self.realised_pnl += (price - self.avg_price) * closed
            else:                     # we were short and are now buying
                self.realised_pnl += (self.avg_price - price) * closed
            if abs(signed) > abs(prev):
                self.avg_price = price  # position flipped; new basis is this price
            elif new == 0:
                self.avg_price = 0.0

        self.inventory = new

    def unrealised_pnl(self, mid: Optional[float]) -> float:
        """Mark-to-market PnL of the open position at the current mid."""
        if self.inventory == 0 or mid is None:
            return 0.0
        return (mid - self.avg_price) * self.inventory

    def total_pnl(self, mid: Optional[float]) -> float:
        return self.realised_pnl + self.unrealised_pnl(mid)

    # ------------------------------------------------------------------ quoting
    def requote(self, engine: MatchingEngine, mid: float, timestamp: float,
                owner_registry: dict) -> None:
        """Cancel any live quotes and repost a fresh two-sided quote around `mid`.

        `owner_registry` maps order_id -> owner so the simulator can attribute
        fills back to us even after the order leaves the book.
        """
        # Pull existing quotes so we always reprice from scratch.
        if self.active_bid_id is not None:
            engine.cancel_order(self.active_bid_id)
            self.active_bid_id = None
        if self.active_ask_id is not None:
            engine.cancel_order(self.active_ask_id)
            self.active_ask_id = None

        # Inventory skew: long inventory pushes both quotes down (keener to sell).
        skew = self.skew_ticks * self.inventory
        bid_price = round(mid - self.half_spread - skew, 2)
        ask_price = round(mid + self.half_spread - skew, 2)

        # Respect the inventory cap: stop adding to a side that would breach it.
        if self.inventory + self.order_size <= self.max_inventory:
            self.active_bid_id = self._post(engine, Side.BUY, bid_price,
                                            timestamp, owner_registry)
        if self.inventory - self.order_size >= -self.max_inventory:
            self.active_ask_id = self._post(engine, Side.SELL, ask_price,
                                            timestamp, owner_registry)

    def _post(self, engine: MatchingEngine, side: Side, price: float,
              timestamp: float, owner_registry: dict) -> int:
        order_id = self._next_id
        self._next_id += 1
        owner_registry[order_id] = self.owner
        order = Order(
            order_id=order_id,
            side=side,
            quantity=self.order_size,
            price=price,
            order_type=OrderType.LIMIT,
            timestamp=timestamp,
            owner=self.owner,
        )
        # A quote could occasionally be marketable (e.g. after a jump); the engine
        # will fill it immediately and on_fill will be invoked by the simulator.
        engine.submit_limit_order(order)
        return order_id
