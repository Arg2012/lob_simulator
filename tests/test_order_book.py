"""Tests for the OrderBook data structure (top of book, depth, cancels)."""

from collections import deque

from events import add_order
from matching_engine import MatchingEngine
from order import Order, OrderType, Side
from order_book import OrderBook


def make_book_with(engine, *orders):
    for ts, oid, side, price, qty in orders:
        engine.process_event(add_order(ts, oid, side, price, qty))


def test_best_bid_ask_and_spread():
    eng = MatchingEngine()
    make_book_with(
        eng,
        (0, 1, Side.BUY, 99.0, 5),
        (1, 2, Side.BUY, 99.5, 3),   # better bid
        (2, 3, Side.SELL, 101.0, 4),
        (3, 4, Side.SELL, 100.5, 2),  # better ask
    )
    book = eng.book
    assert book.best_bid() == 99.5
    assert book.best_ask() == 100.5
    assert book.spread() == 1.0
    assert book.mid() == 100.0


def test_depth_accounting():
    eng = MatchingEngine()
    make_book_with(
        eng,
        (0, 1, Side.BUY, 99.0, 5),
        (1, 2, Side.BUY, 99.0, 3),
        (2, 3, Side.SELL, 101.0, 4),
    )
    book = eng.book
    assert book.total_bid_depth() == 8
    assert book.total_ask_depth() == 4
    assert book.depth_at(Side.BUY, 99.0) == 8


def test_empty_book_top_of_book_is_none():
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None
    assert book.mid() is None
    assert book.total_bid_depth() == 0


def test_levels_are_sorted_best_first():
    eng = MatchingEngine()
    make_book_with(
        eng,
        (0, 1, Side.BUY, 98.0, 1),
        (1, 2, Side.BUY, 99.0, 2),
        (2, 3, Side.SELL, 101.0, 3),
        (3, 4, Side.SELL, 100.0, 4),
    )
    bids = eng.book.levels(Side.BUY, depth=5)
    asks = eng.book.levels(Side.SELL, depth=5)
    assert [p for p, _ in bids] == [99.0, 98.0]     # high to low
    assert [p for p, _ in asks] == [100.0, 101.0]   # low to high


def test_remove_returns_order_and_empties_level():
    book = OrderBook()
    order = Order(1, 0, Side.BUY, OrderType.LIMIT, 5, 99.0)
    book.add_resting(order)
    removed = book.remove(1)
    assert removed is order
    assert book.best_bid() is None
    assert 99.0 not in book.bids  # empty level is pruned
