"""Execution-quality and market-quality metrics.

All metrics are derived from a `SimulationResult` (the per-event time series plus
a few aggregates recorded during the run). Keeping the computation here — separate
from the simulation loop — means the same functions can be pointed at *any*
`SimulationResult`, including one built from replayed real data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .simulator import SimulationResult


def compute_metrics(result: SimulationResult) -> Dict[str, float]:
    """Summarise one simulation run as a flat dict of scalar metrics."""
    spread = result.spread[~np.isnan(result.spread)]

    fill_rate = 0.0
    if result.n_limit_orders > 0:
        filled = len(result.limit_order_ids & result.filled_limit_ids)
        fill_rate = filled / result.n_limit_orders

    slippage = np.asarray(result.market_slippage, dtype=float)

    return {
        "n_events": int(result.timestamps.size),
        "n_trades": len(result.trades),
        "avg_spread": float(np.mean(spread)) if spread.size else float("nan"),
        "median_spread": float(np.median(spread)) if spread.size else float("nan"),
        "avg_bid_depth": float(np.mean(result.bid_depth)),
        "avg_ask_depth": float(np.mean(result.ask_depth)),
        "final_inventory": int(result.inventory[-1]) if result.inventory.size else 0,
        "max_abs_inventory": int(np.max(np.abs(result.inventory)))
        if result.inventory.size else 0,
        "realised_pnl": float(result.realised_pnl[-1]) if result.realised_pnl.size else 0.0,
        "unrealised_pnl": float(result.unrealised_pnl[-1]) if result.unrealised_pnl.size else 0.0,
        "total_pnl": float(result.total_pnl[-1]) if result.total_pnl.size else 0.0,
        "fill_rate": fill_rate,
        "avg_slippage": float(np.mean(slippage)) if slippage.size else 0.0,
        "n_market_orders": int(slippage.size),
    }


def metrics_table(results: Dict[str, SimulationResult]) -> pd.DataFrame:
    """Build a tidy comparison table (one column per scenario)."""
    rows = {name: compute_metrics(res) for name, res in results.items()}
    df = pd.DataFrame(rows)
    return df


def format_metrics(result: SimulationResult) -> str:
    """Human-readable one-block summary for console output."""
    m = compute_metrics(result)
    lines = [f"Scenario: {result.scenario}"]
    lines.append("-" * 40)
    lines.append(f"  events / trades      : {m['n_events']} / {m['n_trades']}")
    lines.append(f"  avg spread           : {m['avg_spread']:.4f}")
    lines.append(f"  avg bid / ask depth  : {m['avg_bid_depth']:.1f} / {m['avg_ask_depth']:.1f}")
    lines.append(f"  fill rate            : {m['fill_rate']:.1%}")
    lines.append(f"  market orders        : {m['n_market_orders']}")
    lines.append(f"  avg slippage         : {m['avg_slippage']:.4f}")
    lines.append(f"  final inventory      : {m['final_inventory']}")
    lines.append(f"  max |inventory|      : {m['max_abs_inventory']}")
    lines.append(f"  realised PnL         : {m['realised_pnl']:.2f}")
    lines.append(f"  unrealised PnL       : {m['unrealised_pnl']:.2f}")
    lines.append(f"  total PnL            : {m['total_pnl']:.2f}")
    return "\n".join(lines)
