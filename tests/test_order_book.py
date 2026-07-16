"""Unit tests for the order book: queries, depth and invariants."""

import itertools

import pytest

from src.order import Order, OrderType, Side
from src.order_book import OrderBook

_ids = itertools.count(1)


def limit(side, price, qty):
    return Order(next(_ids), side, qty, price, OrderType.LIMIT)


def test_best_prices_spread_and_mid():
    book = OrderBook()
    book.add_limit_order(limit(Side.BUY, 99.0, 5))
    book.add_limit_order(limit(Side.BUY, 98.0, 5))
    book.add_limit_order(limit(Side.SELL, 101.0, 5))
    book.add_limit_order(limit(Side.SELL, 102.0, 5))

    assert book.best_bid() == 99.0
    assert book.best_ask() == 101.0
    assert book.spread() == pytest.approx(2.0)
    assert book.midpoint() == pytest.approx(100.0)


def test_empty_book_queries_return_none():
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None
    assert book.midpoint() is None


def test_depth_totals_and_level_limit():
    book = OrderBook()
    book.add_limit_order(limit(Side.BUY, 99.0, 5))
    book.add_limit_order(limit(Side.BUY, 99.0, 3))   # same level accumulates
    book.add_limit_order(limit(Side.BUY, 98.0, 4))
    assert book.bids[99.0].total_quantity == 8
    assert book.bid_depth() == 12
    assert book.bid_depth(levels=1) == 8  # only the best level


def test_depth_snapshot_ordering():
    book = OrderBook()
    for p in (97.0, 98.0, 99.0):
        book.add_limit_order(limit(Side.BUY, p, 1))
    for p in (101.0, 102.0, 103.0):
        book.add_limit_order(limit(Side.SELL, p, 1))
    bids, asks = book.depth_snapshot(levels=2)
    assert [p for p, _ in bids] == [99.0, 98.0]     # best (highest) bid first
    assert [p for p, _ in asks] == [101.0, 102.0]   # best (lowest) ask first


def test_sorted_price_lists_stay_consistent():
    book = OrderBook()
    a = limit(Side.BUY, 99.0, 5)
    b = limit(Side.BUY, 97.0, 5)
    c = limit(Side.BUY, 98.0, 5)
    for o in (a, b, c):
        book.add_limit_order(o)
    assert book._bid_prices == [97.0, 98.0, 99.0]
    book.cancel_order(c.order_id)  # drop the middle level
    assert book._bid_prices == [97.0, 99.0]
    book.check_invariants()


def test_invariants_hold_after_mixed_operations():
    book = OrderBook()
    orders = [limit(Side.BUY, 99.0, 5), limit(Side.BUY, 98.0, 5),
              limit(Side.SELL, 101.0, 5), limit(Side.SELL, 102.0, 5)]
    for o in orders:
        book.add_limit_order(o)
    book.cancel_order(orders[0].order_id)
    book.check_invariants()
    # Book must never be crossed.
    assert book.best_bid() < book.best_ask()
