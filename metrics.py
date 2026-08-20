"""Market-quality and execution-quality metrics.

Given a :class:`~simulator.ScenarioResult`, compute the summary statistics used
to compare scenarios. Metrics fall into two groups:

**Market conditions** (measured from the native order flow, no market maker):

* **average spread** - mean best-ask minus best-bid.
* **average bid / ask depth** - mean total resting quantity per side.
* **average slippage** - mean market-order execution price vs. the pre-trade
  mid (positive = adverse); a proxy for execution cost / market impact.

**Agent execution quality** (measured with the market maker participating):

* **inventory** - the market maker's position over time (final and peak).
* **fill rate** - fraction of the market maker's quoted volume that executed.
* **realised / total PnL** - locked-in and mark-to-market PnL at the end.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from simulator import ScenarioResult, SimulationResult


def compute_metrics(scenario: ScenarioResult) -> Dict[str, float]:
    """Return a flat dict of summary metrics for one scenario."""
    market_df = scenario.market.to_dataframe()
    agent_df = scenario.agent.to_dataframe()
    mm = scenario.agent.market_maker

    spread = market_df["spread"].dropna()
    slippages = scenario.market.slippages

    return {
        # --- market conditions (native flow) ---
        "avg_spread": float(spread.mean()) if not spread.empty else float("nan"),
        "avg_bid_depth": float(market_df["bid_depth"].mean()),
        "avg_ask_depth": float(market_df["ask_depth"].mean()),
        "avg_slippage": float(np.mean(slippages)) if slippages else 0.0,
        "num_market_orders": len(slippages),
        # --- agent execution quality (with market maker) ---
        "fill_rate": float(mm.fill_rate),
        "final_inventory": int(agent_df["inventory"].iloc[-1]),
        "max_abs_inventory": int(agent_df["inventory"].abs().max()),
        "realised_pnl": float(mm.realised),
        "total_pnl": float(mm.total),
    }


def metrics_table(results: Dict[str, ScenarioResult]) -> pd.DataFrame:
    """Build a scenario-by-metric comparison table."""
    rows = {name: compute_metrics(res) for name, res in results.items()}
    return pd.DataFrame(rows).T


def inventory_series(result: SimulationResult) -> List[int]:
    """The market maker's inventory over time (one value per step)."""
    return [s.inventory for s in result.snapshots]
