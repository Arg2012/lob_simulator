"""Price-time priority matching engine.

This is the single source of truth for how orders match. Both the live
simulator and the historical replay push :class:`Event` objects through
:meth:`MatchingEngine.process_event`, so they exercise identical logic.

Matching rules
--------------
1. **Price priority** - a buy always matches the lowest ask first; a sell
   always matches the highest bid first.
2. **Time priority (FIFO)** - within one price level, the order that arrived
   first is filled first (it sits at the front of the level's queue).
3. A **market order** sweeps across as many price levels as needed until it is
   filled or the opposite side is exhausted; any unfilled remainder is dropped.
4. A **limit order** executes immediately while it is marketable, then rests the
   remainder in the book.
5. Trades print at the **resting order's price** (price improvement accrues to
   the aggressor).
"""

from __future__ import annotations

from typing import List

from events import Event, EventType
from order import Order, OrderType, Side
from order_book import OrderBook
from trade import Trade


class MatchingEngine:
    """Processes events against an :class:`OrderBook`, emitting trades."""

    def __init__(self, book: OrderBook | None = None) -> None:
        self.book = book or OrderBook()
        self.trades: List[Trade] = []

    # --- event dispatch ------------------------------------------------------

    def process_event(self, event: Event) -> List[Trade]:
        """Apply one event to the book and return the trades it generated."""
        if event.event_type is EventType.ADD_ORDER:
            order = Order(event.order_id, event.timestamp, event.side,
                          OrderType.LIMIT, event.quantity, event.price)
            return self.add_limit_order(order)
        if event.event_type is EventType.MARKET_ORDER:
            order = Order(event.order_id, event.timestamp, event.side,
                          OrderType.MARKET, event.quantity, None)
            return self.add_market_order(order)
        if event.event_type is EventType.CANCEL_ORDER:
            self.cancel(event.order_id)
            return []
        raise ValueError(f"Unknown event type: {event.event_type}")

    # --- order entry points --------------------------------------------------

    def add_limit_order(self, order: Order) -> List[Trade]:
        """Match a limit order against the book, then rest any remainder."""
        trades = self._match(order, limit_price=order.price)
        if order.remaining > 0:
            self.book.add_resting(order)
        self.trades.extend(trades)
        return trades

    def add_market_order(self, order: Order) -> List[Trade]:
        """Match a market order across price levels; discard any remainder."""
        trades = self._match(order, limit_price=None)
        self.trades.extend(trades)
        return trades

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. Returns True if an order was removed."""
        return self.book.remove(order_id) is not None

    # --- core matching loop --------------------------------------------------

    def _match(self, incoming: Order, limit_price: float | None) -> List[Trade]:
        """Match ``incoming`` against the opposite side.

        ``limit_price`` bounds how far the order may walk the book:
        ``None`` (market) means no bound. The loop consumes the best price
        level first, and within a level consumes the front (oldest) order
        first, giving strict price-then-time priority.
        """
        trades: List[Trade] = []
        opposite = self.book.asks if incoming.side is Side.BUY else self.book.bids

        while incoming.remaining > 0 and opposite:
            best_price = min(opposite) if incoming.side is Side.BUY else max(opposite)

            if limit_price is not None:
                # Stop once the best available price is worse than our limit.
                if incoming.side is Side.BUY and best_price > limit_price:
                    break
                if incoming.side is Side.SELL and best_price < limit_price:
                    break

            level = opposite[best_price]
            while level and incoming.remaining > 0:
                resting = level[0]
                fill = min(incoming.remaining, resting.remaining)

                incoming.remaining -= fill
                resting.remaining -= fill
                trades.append(Trade(
                    timestamp=incoming.timestamp,
                    price=best_price,
                    quantity=fill,
                    aggressor_side=incoming.side,
                    taker_order_id=incoming.order_id,
                    maker_order_id=resting.order_id,
                ))

                if resting.remaining == 0:
                    level.popleft()
                    self.book.orders.pop(resting.order_id, None)

            if not level:
                del opposite[best_price]

        return trades
