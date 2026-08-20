"""Tests for replay reconstruction, scenarios and market-maker accounting."""

import os
import tempfile

from events import add_order, market_order, write_events_csv
from market_maker import MarketMaker
from matching_engine import MatchingEngine
from order import Side
from replay import ReplayEngine, load_and_replay
from simulator import (SCENARIOS, ScenarioConfig, generate_event_stream,
                       run_scenario, run_simulation)


def test_replay_reconstructs_same_book_as_direct_processing():
    """Replaying a CSV must reproduce exactly the same book as processing the
    events directly through the engine (they share the matching engine)."""
    events = [
        add_order(0, 1, Side.SELL, 101.0, 5),
        add_order(1, 2, Side.SELL, 100.5, 3),
        add_order(2, 3, Side.BUY, 99.5, 4),
        market_order(3, 4, Side.BUY, 2),
        add_order(4, 5, Side.BUY, 100.0, 6),
    ]

    # Direct processing.
    direct = MatchingEngine()
    for e in events:
        direct.process_event(e)

    # Round-trip through CSV + replay.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "events.csv")
        write_events_csv(events, path)
        replayed = load_and_replay(path).book

    assert replayed.best_bid() == direct.book.best_bid()
    assert replayed.best_ask() == direct.book.best_ask()
    assert replayed.total_bid_depth() == direct.book.total_bid_depth()
    assert replayed.total_ask_depth() == direct.book.total_ask_depth()


def test_replay_up_to_allows_point_in_time_inspection():
    events = [
        add_order(0, 1, Side.SELL, 101.0, 5),
        add_order(1, 2, Side.BUY, 99.0, 5),
        add_order(2, 3, Side.BUY, 100.0, 5),
    ]
    eng = ReplayEngine()
    eng.replay(events, up_to=2)          # only first two events
    assert eng.book.best_bid() == 99.0
    eng.replay(events)                   # finish
    assert eng.book.best_bid() == 100.0


def test_generate_event_stream_is_several_thousand_events():
    cfg = ScenarioConfig(name="hist", n_steps=1200, seed=42)
    events = generate_event_stream(cfg)
    assert len(events) > 3000            # "several thousand"


def test_scenarios_produce_different_behaviour():
    """The three scenarios must yield visibly different market characteristics."""
    from metrics import compute_metrics

    results = {name: run_scenario(cfg) for name, cfg in SCENARIOS.items()}
    metrics = {n: compute_metrics(r) for n, r in results.items()}
    spreads = {n: m["avg_spread"] for n, m in metrics.items()}
    depths = {n: m["avg_bid_depth"] + m["avg_ask_depth"] for n, m in metrics.items()}

    # High volatility trades widest; low liquidity is the thinnest book.
    assert spreads["high_volatility"] > spreads["normal"]
    assert spreads["low_liquidity"] > spreads["normal"]
    assert depths["low_liquidity"] < depths["normal"]
    assert depths["high_volatility"] < depths["normal"]
    # All three spreads are distinct.
    assert len(set(round(v, 4) for v in spreads.values())) == 3


def test_market_maker_realises_pnl_on_round_trip():
    """Buy low then sell high must realise a positive PnL on the closed lot."""
    import itertools

    mm = MarketMaker(itertools.count(1))
    mm.my_orders.update({10, 11})

    # Simulate a fill where our resting BUY (maker) is hit at 100.
    from trade import Trade
    mm.on_trades([Trade(0, 100.0, 5, Side.SELL, 99, 10)])  # we bought 5 @ 100
    assert mm.inventory == 5
    assert mm.avg_price == 100.0

    # Now our resting SELL (maker) is hit at 102 -> close the 5 lot.
    mm.on_trades([Trade(1, 102.0, 5, Side.BUY, 98, 11)])
    assert mm.inventory == 0
    assert mm.realised == 5 * (102.0 - 100.0)

    mm.mark(102.0)
    assert mm.unrealised == 0.0
    assert mm.total == mm.realised


def test_market_maker_tracks_inventory_in_full_run():
    result = run_simulation(SCENARIOS["normal"])
    mm = result.market_maker
    # The maker actually traded and its accounting is self-consistent.
    assert mm.filled_qty > 0
    assert 0.0 <= mm.fill_rate <= 1.0
    # Final snapshot inventory equals the maker's inventory.
    assert result.snapshots[-1].inventory == mm.inventory
