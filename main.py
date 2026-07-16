"""End-to-end demo driver for the limit order book simulator.

Running this script will:

1. Generate a synthetic sample event stream and write it to ``data/sample_events.csv``.
2. Demonstrate historical replay by reconstructing the book from that CSV.
3. Run the three market scenarios (normal / high volatility / low liquidity),
   each with the market-making agent quoting into the book.
4. Print an execution-quality metrics table and save all figures into ``outputs/``.

Usage
-----
    python main.py                 # run everything with defaults
    python main.py --events 8000   # more events per scenario
    python main.py --seed 7        # different random seed
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.matching_engine import MatchingEngine
from src.metrics import format_metrics, metrics_table
from src.plots import (plot_book_snapshot, plot_scenario_comparison,
                       save_all_plots)
from src.replay import load_events, replay
from src.simulator import SCENARIOS, generate_order_flow, run_scenario

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_events.csv")


def build_sample_data(n_events: int, seed: int) -> None:
    """Generate a several-thousand-event stream and persist it as CSV."""
    from src.replay import write_events

    os.makedirs(DATA_DIR, exist_ok=True)
    events = generate_order_flow("normal", n_events=n_events, seed=seed)
    write_events(SAMPLE_CSV, events)
    print(f"[data] wrote {len(events)} events -> {os.path.relpath(SAMPLE_CSV, HERE)}")


def demo_replay() -> MatchingEngine:
    """Reconstruct the book from the sample CSV and print a summary."""
    events = load_events(SAMPLE_CSV)
    engine = replay(events)
    book = engine.book
    print("\n[replay] reconstructed book from historical events")
    print(f"  events replayed : {len(events)}")
    print(f"  trades matched  : {len(engine.trades)}")
    print(f"  best bid / ask  : {book.best_bid()} / {book.best_ask()}")
    print(f"  spread / mid    : {book.spread()} / {book.midpoint()}")
    print(f"  resting orders  : {len(book.orders)}")
    book.check_invariants()
    plot_book_snapshot(engine, OUT_DIR, name="replay_book_snapshot")
    return engine


def run_all_scenarios(n_events: int, seed: int) -> None:
    """Run every scenario, print metrics, and save all plots."""
    results = {}
    for scenario in SCENARIOS:
        print(f"\n[sim] running scenario '{scenario}' ...")
        res = run_scenario(scenario, n_events=n_events, seed=seed)
        results[scenario] = res
        print(format_metrics(res))
        paths = save_all_plots(res, OUT_DIR)
        print(f"  saved {len(paths)} figures to outputs/")

    # Comparison table + figure.
    table = metrics_table(results)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n[metrics] scenario comparison")
    print(table.to_string())
    table.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
    plot_scenario_comparison(results, OUT_DIR)
    print(f"\n[done] all outputs written to {os.path.relpath(OUT_DIR, HERE)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="LOB simulator demo driver")
    parser.add_argument("--events", type=int, default=5000,
                        help="events per scenario / in the sample stream")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    build_sample_data(n_events=args.events, seed=args.seed)
    demo_replay()
    run_all_scenarios(n_events=args.events, seed=args.seed)


if __name__ == "__main__":
    main()
