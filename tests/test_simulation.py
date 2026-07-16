"""Integration tests: replay round-trip, scenario runs, market-maker accounting."""

import os

import numpy as np

from src.market_maker import MarketMaker
from src.matching_engine import MatchingEngine
from src.order import Side
from src.replay import load_events, replay, write_events
from src.simulator import SCENARIOS, generate_order_flow, run_scenario


def test_generate_order_flow_is_deterministic():
    a = generate_order_flow("normal", n_events=500, seed=42)
    b = generate_order_flow("normal", n_events=500, seed=42)
    assert len(a) == len(b)
    assert [(e.event_type, e.order_id) for e in a] == \
           [(e.event_type, e.order_id) for e in b]


def test_replay_csv_roundtrip(tmp_path):
    events = generate_order_flow("normal", n_events=1000, seed=1)
    path = os.path.join(tmp_path, "events.csv")
    write_events(path, events)
    reloaded = load_events(path)
    assert len(reloaded) == len(events)

    # Replaying the reconstructed events yields a consistent book.
    engine = replay(reloaded)
    engine.book.check_invariants()
    if engine.book.best_bid() is not None and engine.book.best_ask() is not None:
        assert engine.book.best_bid() < engine.book.best_ask()


def test_all_scenarios_run_and_are_consistent():
    for scenario in SCENARIOS:
        cfg = SCENARIOS[scenario]
        res = run_scenario(scenario, n_events=800, seed=3)
        # One snapshot per event = warm-up ladder (2 * ladder_levels) + flow.
        expected = 2 * cfg.ladder_levels + 800
        assert res.timestamps.size == expected
        assert res.mid.size == res.timestamps.size
        # PnL identity: total == realised + unrealised at every step.
        np.testing.assert_allclose(
            res.total_pnl, res.realised_pnl + res.unrealised_pnl, atol=1e-6)


def test_market_maker_realised_pnl_on_round_trip():
    """Buy 10 @ 100 then sell 10 @ 101 -> realised PnL = 10."""
    mm = MarketMaker()
    mm.on_fill(Side.BUY, 100.0, 10)
    assert mm.inventory == 10
    assert mm.avg_price == 100.0
    mm.on_fill(Side.SELL, 101.0, 10)
    assert mm.inventory == 0
    assert mm.realised_pnl == 10.0           # (101-100) * 10
    assert mm.unrealised_pnl(105.0) == 0.0   # flat -> no mark-to-market


def test_market_maker_unrealised_pnl_marks_to_mid():
    mm = MarketMaker()
    mm.on_fill(Side.BUY, 100.0, 5)
    assert mm.unrealised_pnl(102.0) == 10.0   # (102-100)*5
    assert mm.unrealised_pnl(99.0) == -5.0    # (99-100)*5


def test_market_maker_short_then_cover():
    mm = MarketMaker()
    mm.on_fill(Side.SELL, 100.0, 5)   # go short
    assert mm.inventory == -5
    mm.on_fill(Side.BUY, 98.0, 5)     # cover cheaper -> profit
    assert mm.inventory == 0
    assert mm.realised_pnl == 10.0    # (100-98)*5
