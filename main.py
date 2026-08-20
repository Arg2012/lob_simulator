"""End-to-end driver for the limit order book simulator.

Running ``python main.py`` will:

1. Generate a synthetic historical event stream and save it to
   ``data/sample_events.csv``.
2. Replay that stream through the shared matching engine to reconstruct the
   order book, and print a reconstructed book snapshot.
3. Run the three market scenarios (normal / high volatility / low liquidity)
   with the market maker.
4. Print a metrics comparison table.
5. Save the comparison plots into ``outputs/``.

All data is synthetic and seeded for reproducibility.
"""

from __future__ import annotations

import os

import pandas as pd

from events import write_events_csv
from metrics import metrics_table
from plots import generate_all_plots
from replay import load_and_replay
from simulator import SCENARIOS, ScenarioConfig, generate_event_stream, run_scenario

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_events.csv")


def build_sample_data() -> int:
    """Generate the synthetic event stream and write it to CSV. Returns count."""
    os.makedirs(DATA_DIR, exist_ok=True)
    config = ScenarioConfig(name="historical", n_steps=1200, seed=42)
    events = generate_event_stream(config)
    write_events_csv(events, SAMPLE_CSV)
    return len(events)


def demo_replay(n_events: int) -> None:
    """Reconstruct the book from CSV and show it at a mid-stream point."""
    print("\n" + "=" * 62)
    print("ORDER-BOOK RECONSTRUCTION FROM HISTORICAL EVENT STREAM")
    print("=" * 62)
    print(f"Loaded {n_events} synthetic events from "
          f"{os.path.relpath(SAMPLE_CSV, HERE)}")

    # Reconstruct up to the halfway point and inspect the book there.
    halfway = n_events // 2
    mid_engine = load_and_replay(SAMPLE_CSV, up_to=halfway)
    print(f"\nReconstructed book after {halfway} events:")
    print(mid_engine.book.snapshot(depth=5))
    print(f"best bid={mid_engine.book.best_bid()}  "
          f"best ask={mid_engine.book.best_ask()}  "
          f"spread={mid_engine.book.spread()}")

    # Replay the remainder and report the final reconstructed state.
    full = load_and_replay(SAMPLE_CSV)
    print(f"\nFully reconstructed book after {n_events} events:")
    print(f"best bid={full.book.best_bid()}  best ask={full.book.best_ask()}  "
          f"spread={full.book.spread()}  matched trades={len(full.trades)}")


def run_scenarios() -> dict:
    """Run all three scenarios with the market maker."""
    print("\n" + "=" * 62)
    print("MARKET SCENARIOS (native conditions + market maker)")
    print("=" * 62)
    results = {}
    for name, config in SCENARIOS.items():
        results[name] = run_scenario(config)
        print(f"  ran scenario '{name}' ({config.n_steps} steps x2 passes)")
    return results


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    n_events = build_sample_data()
    demo_replay(n_events)

    results = run_scenarios()

    print("\n" + "=" * 62)
    print("METRICS COMPARISON")
    print("=" * 62)
    table = metrics_table(results)
    print(table.to_string())
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))

    paths = generate_all_plots(results, OUT_DIR)
    print("\n" + "=" * 62)
    print("PLOTS SAVED")
    print("=" * 62)
    for name, path in paths.items():
        print(f"  {name:10s} -> {os.path.relpath(path, HERE)}")

    print("\nDone. All outputs written to outputs/.")


if __name__ == "__main__":
    main()
