"""Unit tests for the matching engine: FIFO, market/limit execution,
partial fills, cancellations and multi-level matching."""

import itertools

import pytest

from src.matching_engine import MatchingEngine
from src.order import Order, OrderType, Side

_ids = itertools.count(1)


def limit(side, price, qty, ts=0.0, owner="background"):
    return Order(next(_ids), side, qty, price, OrderType.LIMIT, ts, owner)


def market(side, qty, ts=0.0):
    return Order(next(_ids), side, qty, None, OrderType.MARKET, ts)


# --------------------------------------------------------------------- FIFO
def test_fifo_time_priority_within_price_level():
    """At one price, the earliest-arriving order must fill first."""
    eng = MatchingEngine()
    a = limit(Side.SELL, 100.0, 5, ts=1.0)
    b = limit(Side.SELL, 100.0, 5, ts=2.0)
    eng.submit_limit_order(a)
    eng.submit_limit_order(b)

    trades = eng.submit_market_order(market(Side.BUY, 5))
    assert len(trades) == 1
    assert trades[0].maker_order_id == a.order_id  # 'a' arrived first
    # 'b' should be untouched and still resting at full size.
    assert eng.book.orders[b.order_id].quantity == 5


def test_fifo_across_two_makers_partial_second():
    eng = MatchingEngine()
    a = limit(Side.SELL, 100.0, 4, ts=1.0)
    b = limit(Side.SELL, 100.0, 4, ts=2.0)
    eng.submit_limit_order(a)
    eng.submit_limit_order(b)

    trades = eng.submit_market_order(market(Side.BUY, 6))
    # 4 from a (fully), 2 from b (partial) — in that order.
    assert [t.maker_order_id for t in trades] == [a.order_id, b.order_id]
    assert [t.quantity for t in trades] == [4, 2]
    assert a.order_id not in eng.book.orders
    assert eng.book.orders[b.order_id].quantity == 2


# ------------------------------------------------------------- market orders
def test_market_order_consumes_multiple_levels():
    eng = MatchingEngine()
    eng.submit_limit_order(limit(Side.SELL, 100.0, 3))
    eng.submit_limit_order(limit(Side.SELL, 101.0, 3))
    eng.submit_limit_order(limit(Side.SELL, 102.0, 3))

    trades = eng.submit_market_order(market(Side.BUY, 7))
    assert sum(t.quantity for t in trades) == 7
    # Prices consumed from best (100) outward.
    assert [t.price for t in trades] == [100.0, 101.0, 102.0]
    assert [t.quantity for t in trades] == [3, 3, 1]
    # 102 level should retain the leftover 2.
    assert eng.book.ask_depth() == 2


def test_market_order_exhausts_book_and_discards_remainder():
    eng = MatchingEngine()
    eng.submit_limit_order(limit(Side.SELL, 100.0, 2))
    trades = eng.submit_market_order(market(Side.BUY, 10))
    assert sum(t.quantity for t in trades) == 2
    assert eng.book.best_ask() is None  # book emptied
    # No resting remainder: market orders never rest.
    assert len(eng.book.orders) == 0


# ------------------------------------------------------------ limit crossing
def test_marketable_limit_executes_then_rests_remainder():
    eng = MatchingEngine()
    eng.submit_limit_order(limit(Side.SELL, 100.0, 4))
    # Buy 10 @ 100: 4 execute, 6 rest as the new best bid.
    taker = limit(Side.BUY, 100.0, 10)
    trades = eng.submit_limit_order(taker)
    assert sum(t.quantity for t in trades) == 4
    assert eng.book.best_bid() == 100.0
    assert eng.book.orders[taker.order_id].quantity == 6


def test_non_marketable_limit_just_rests():
    eng = MatchingEngine()
    eng.submit_limit_order(limit(Side.SELL, 101.0, 5))
    taker = limit(Side.BUY, 100.0, 5)  # below best ask -> not marketable
    trades = eng.submit_limit_order(taker)
    assert trades == []
    assert eng.book.best_bid() == 100.0
    assert eng.book.spread() == pytest.approx(1.0)


def test_limit_respects_price_limit_across_levels():
    """A limit buy must not fill against asks above its limit price."""
    eng = MatchingEngine()
    eng.submit_limit_order(limit(Side.SELL, 100.0, 3))
    eng.submit_limit_order(limit(Side.SELL, 101.0, 3))
    taker = limit(Side.BUY, 100.0, 6)  # willing to pay only up to 100
    trades = eng.submit_limit_order(taker)
    assert sum(t.quantity for t in trades) == 3        # only the 100 level
    assert eng.book.orders[taker.order_id].quantity == 3  # remainder rests @100
    assert eng.book.asks[101.0].total_quantity == 3       # 101 untouched


# --------------------------------------------------------------- partial fill
def test_partial_fill_updates_resting_quantity():
    eng = MatchingEngine()
    resting = limit(Side.BUY, 100.0, 10)
    eng.submit_limit_order(resting)
    eng.submit_market_order(market(Side.SELL, 3))
    assert eng.book.orders[resting.order_id].quantity == 7
    assert eng.book.bids[100.0].total_quantity == 7


# --------------------------------------------------------------- cancellations
def test_cancel_removes_order_and_level():
    eng = MatchingEngine()
    o = limit(Side.BUY, 100.0, 5)
    eng.submit_limit_order(o)
    assert eng.book.best_bid() == 100.0
    cancelled = eng.cancel_order(o.order_id)
    assert cancelled is o
    assert eng.book.best_bid() is None
    assert o.order_id not in eng.book.orders


def test_cancel_one_of_two_keeps_level():
    eng = MatchingEngine()
    a = limit(Side.BUY, 100.0, 5)
    b = limit(Side.BUY, 100.0, 5)
    eng.submit_limit_order(a)
    eng.submit_limit_order(b)
    eng.cancel_order(a.order_id)
    assert eng.book.bids[100.0].total_quantity == 5
    assert list(eng.book.bids[100.0].orders) == [b]


def test_cancel_unknown_id_is_noop():
    eng = MatchingEngine()
    assert eng.cancel_order(999) is None


def test_cancelled_order_is_skipped_in_matching():
    eng = MatchingEngine()
    a = limit(Side.SELL, 100.0, 5, ts=1.0)
    b = limit(Side.SELL, 100.0, 5, ts=2.0)
    eng.submit_limit_order(a)
    eng.submit_limit_order(b)
    eng.cancel_order(a.order_id)          # remove the front-of-queue order
    trades = eng.submit_market_order(market(Side.BUY, 5))
    assert trades[0].maker_order_id == b.order_id  # now b is first
