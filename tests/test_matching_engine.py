"""Tests for the matching engine: price priority, FIFO, partial fills,
market-order sweeping, cancellations and book invariants.

These are the core correctness tests that demonstrate the engine behaves like a
genuine price-time priority matcher.
"""

import random

from events import add_order, cancel_order, market_order
from matching_engine import MatchingEngine
from order import Side


def add(eng, ts, oid, side, price, qty):
    return eng.process_event(add_order(ts, oid, side, price, qty))


def test_price_priority_best_price_fills_first():
    """A buy must hit the lowest ask, regardless of arrival order."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 101.0, 5)   # worse (higher) ask, arrived first
    add(eng, 1, 2, Side.SELL, 100.0, 5)   # better (lower) ask, arrived second

    trades = add(eng, 2, 3, Side.BUY, 101.0, 5)  # marketable buy

    assert len(trades) == 1
    assert trades[0].price == 100.0            # best price filled first
    assert trades[0].maker_order_id == 2
    # The 101.0 ask is untouched.
    assert eng.book.depth_at(Side.SELL, 101.0) == 5


def test_time_priority_fifo_within_level():
    """Within one price level, the earliest order fills first (FIFO)."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 5)   # arrived first
    add(eng, 1, 2, Side.SELL, 100.0, 5)   # arrived second

    trades = eng.process_event(market_order(2, 3, Side.BUY, 5))

    assert len(trades) == 1
    assert trades[0].maker_order_id == 1   # oldest at the level filled first
    assert eng.book.depth_at(Side.SELL, 100.0) == 5  # order 2 still resting


def test_partial_fill_rests_remainder():
    """A marketable limit order fills what it can and rests the rest."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 3)          # only 3 available
    trades = add(eng, 1, 2, Side.BUY, 100.0, 10)  # wants 10

    assert sum(t.quantity for t in trades) == 3
    assert eng.book.best_ask() is None           # ask consumed
    # Remaining 7 rests on the bid at 100.0.
    assert eng.book.best_bid() == 100.0
    assert eng.book.depth_at(Side.BUY, 100.0) == 7


def test_resting_order_partial_fill_keeps_remainder():
    """A resting order that is partially hit keeps its remaining quantity."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 10)
    trades = eng.process_event(market_order(1, 2, Side.BUY, 4))

    assert sum(t.quantity for t in trades) == 4
    assert eng.book.depth_at(Side.SELL, 100.0) == 6
    assert eng.book.orders[1].remaining == 6


def test_market_order_sweeps_multiple_levels():
    """A market order walks across price levels until filled."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 2)
    add(eng, 1, 2, Side.SELL, 100.5, 2)
    add(eng, 2, 3, Side.SELL, 101.0, 2)

    trades = eng.process_event(market_order(3, 4, Side.BUY, 5))

    assert sum(t.quantity for t in trades) == 5
    prices = [t.price for t in trades]
    assert prices == [100.0, 100.5, 101.0]    # in price order
    assert [t.quantity for t in trades] == [2, 2, 1]
    # 1 unit left resting at the top level.
    assert eng.book.depth_at(Side.SELL, 101.0) == 1


def test_limit_order_rests_when_not_marketable():
    """A non-crossing limit order simply rests; no trades."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 101.0, 5)
    trades = add(eng, 1, 2, Side.BUY, 100.0, 5)  # below best ask

    assert trades == []
    assert eng.book.best_bid() == 100.0
    assert eng.book.best_ask() == 101.0


def test_market_order_stops_when_book_empty():
    """A market order with no liquidity left simply stops (remainder dropped)."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 2)
    trades = eng.process_event(market_order(1, 2, Side.BUY, 10))

    assert sum(t.quantity for t in trades) == 2
    assert eng.book.best_ask() is None
    assert eng.book.best_bid() is None  # market orders never rest


def test_cancel_removes_resting_order():
    eng = MatchingEngine()
    add(eng, 0, 1, Side.BUY, 99.0, 5)
    assert eng.book.best_bid() == 99.0

    removed = eng.cancel(1)
    assert removed is True
    assert eng.book.best_bid() is None
    assert 1 not in eng.book.orders


def test_cancel_unknown_order_is_noop():
    eng = MatchingEngine()
    add(eng, 0, 1, Side.BUY, 99.0, 5)
    assert eng.cancel(999) is False       # unknown id
    assert eng.book.best_bid() == 99.0     # book unchanged


def test_cancelled_order_no_longer_matches():
    """A cancelled resting order must not participate in later matches."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 5)
    eng.cancel(1)
    trades = eng.process_event(market_order(1, 2, Side.BUY, 5))
    assert trades == []                    # nothing to match against


def test_trade_prints_at_resting_price():
    """Aggressor gets price improvement; trade prints at the maker's price."""
    eng = MatchingEngine()
    add(eng, 0, 1, Side.SELL, 100.0, 5)
    trades = add(eng, 1, 2, Side.BUY, 105.0, 5)  # willing to pay up to 105
    assert trades[0].price == 100.0              # but fills at the resting 100


def test_book_invariants_under_random_flow():
    """After a stream of random events the book stays internally consistent."""
    eng = MatchingEngine()
    rng = random.Random(123)
    oid = 0
    for ts in range(2000):
        oid += 1
        roll = rng.random()
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = rng.randint(1, 5)
        if roll < 0.6:
            price = round(100 + rng.randint(-5, 5) * 0.5, 2)
            eng.process_event(add_order(ts, oid, side, price, qty))
        elif roll < 0.8:
            eng.process_event(market_order(ts, oid, side, qty))
        else:
            # Cancel a random known order id (mostly no-ops, which is fine).
            eng.cancel(rng.randint(1, oid))

    book = eng.book
    bid, ask = book.best_bid(), book.best_ask()

    # 1. The book is never crossed.
    if bid is not None and ask is not None:
        assert bid < ask
    # 2. Depths are non-negative and match the order lookup.
    assert book.total_bid_depth() >= 0
    assert book.total_ask_depth() >= 0
    lookup_total = sum(o.remaining for o in book.orders.values())
    assert lookup_total == book.total_bid_depth() + book.total_ask_depth()
    # 3. Every resting order has positive remaining quantity.
    assert all(o.remaining > 0 for o in book.orders.values())
    # 4. Every id in a level is present in the lookup and vice versa.
    level_ids = {o.order_id for lvl in list(book.bids.values()) + list(book.asks.values())
                 for o in lvl}
    assert level_ids == set(book.orders.keys())
