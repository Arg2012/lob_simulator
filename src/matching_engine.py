"""The matching engine: price-time-priority crossing logic.

The engine owns an `OrderBook` and turns incoming orders into `Trade`s. It is the
single place where liquidity is *consumed*; the book itself only stores resting
orders.

Matching rules
--------------
* A **market order** walks the opposite side of the book from the best price
  outward, consuming as much liquidity as needed across multiple price levels,
  until it is filled or the book is empty. Any unfilled remainder is discarded
  (a market order has no price at which to rest).
* A **limit order** first crosses against any *marketable* liquidity (opposite
  orders at or better than its limit price), again in price-time priority. Any
  remaining quantity then rests in the book.
* Every fill executes at the **resting (maker) order's price** — the aggressor
  receives price improvement.
"""

from __future__ import annotations

from typing import List, Optional

from .order import Order, OrderType, Side
from .order_book import OrderBook
from .trade import Trade


class MatchingEngine:
    """Matches incoming orders against an order book, producing trades."""

    def __init__(self, book: Optional[OrderBook] = None) -> None:
        self.book: OrderBook = book if book is not None else OrderBook()
        self.trades: List[Trade] = []  # full execution history

    # --------------------------------------------------------------- public API
    def submit_limit_order(self, order: Order) -> List[Trade]:
        """Cross a limit order against the book, then rest any remainder."""
        if order.order_type is not OrderType.LIMIT:
            raise ValueError("submit_limit_order expects a LIMIT order.")
        trades = self._match(order, marketable_check=True)
        if order.quantity > 0:
            self.book.add_limit_order(order)
        return trades

    def submit_market_order(self, order: Order) -> List[Trade]:
        """Cross a market order against the book; discard any unfilled remainder."""
        if order.order_type is not OrderType.MARKET:
            raise ValueError("submit_market_order expects a MARKET order.")
        return self._match(order, marketable_check=False)

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancel a resting order by id (delegates to the book)."""
        return self.book.cancel_order(order_id)

    # ------------------------------------------------------------------ matching
    def _match(self, taker: Order, marketable_check: bool) -> List[Trade]:
        """Walk the opposite side, filling `taker` in price-time priority.

        Parameters
        ----------
        taker:            the aggressing order (mutated in place as it fills).
        marketable_check: True for limit orders — stop once the best opposite
                          price is no longer at/through the taker's limit price.
                          False for market orders — take whatever is available.
        """
        trades: List[Trade] = []
        book = self.book
        opp_side = taker.side.opposite

        # Choose the opposite side's structures and its best-price accessor.
        if taker.is_buy:
            opp_levels = book.asks
            best_fn = book.best_ask
        else:
            opp_levels = book.bids
            best_fn = book.best_bid

        while taker.quantity > 0:
            best_price = best_fn()
            if best_price is None:
                break  # opposite side is empty

            if marketable_check and not self._crosses(taker, best_price):
                break  # limit price no longer marketable — rest the remainder

            level = opp_levels[best_price]

            # FIFO within the level: always match the front-of-queue order.
            while taker.quantity > 0 and len(level) > 0:
                maker = level.orders[0]
                fill_qty = min(taker.quantity, maker.quantity)

                trades.append(Trade(
                    timestamp=taker.timestamp,
                    price=maker.price,        # trade at the maker's price
                    quantity=fill_qty,
                    maker_order_id=maker.order_id,
                    taker_order_id=taker.order_id,
                    aggressor=taker.side,
                ))

                taker.quantity -= fill_qty
                maker.quantity -= fill_qty
                level.total_quantity -= fill_qty

                if maker.quantity == 0:
                    level.orders.popleft()
                    del book.orders[maker.order_id]

            if len(level) == 0:
                book._drop_level(opp_side, best_price)

        self.trades.extend(trades)
        return trades

    @staticmethod
    def _crosses(taker: Order, best_opposite_price: float) -> bool:
        """Is `taker`'s limit price marketable against `best_opposite_price`?"""
        if taker.is_buy:
            return taker.price >= best_opposite_price
        return taker.price <= best_opposite_price
