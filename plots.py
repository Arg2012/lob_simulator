"""Matplotlib visualisations comparing the three scenarios.

Every figure overlays the three scenarios so their differing characteristics
are directly comparable. Market-condition figures (midprice, spread, depth) are
drawn from the native order flow; agent figures (inventory, PnL) are drawn from
the market-maker run. Figures are written to ``outputs/`` as PNGs.
"""

from __future__ import annotations

import os
from typing import Callable, Dict

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save files without a display
import matplotlib.pyplot as plt  # noqa: E402

from simulator import ScenarioResult, SimulationResult  # noqa: E402


def _overlay(results: Dict[str, ScenarioResult], source: Callable[[ScenarioResult], SimulationResult],
             series: Callable, title: str, ylabel: str, filename: str, outdir: str) -> str:
    """Overlay one series across all scenarios and save the figure."""
    plt.figure(figsize=(10, 5))
    for name, res in results.items():
        df = source(res).to_dataframe()
        plt.plot(df["step"], series(df), label=name, linewidth=1.0)
    plt.title(title)
    plt.xlabel("event step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def generate_all_plots(results: Dict[str, ScenarioResult],
                       outdir: str = "outputs") -> Dict[str, str]:
    """Generate every required plot and return {plot_name: filepath}."""
    os.makedirs(outdir, exist_ok=True)

    market = lambda r: r.market   # noqa: E731
    agent = lambda r: r.agent     # noqa: E731
    paths: Dict[str, str] = {}

    # --- market conditions (native order flow) ---
    paths["midprice"] = _overlay(
        results, market, lambda df: df["mid"],
        "Midprice over time", "mid price", "midprice.png", outdir)
    paths["spread"] = _overlay(
        results, market, lambda df: df["spread"],
        "Spread over time", "spread", "spread.png", outdir)
    paths["depth"] = _overlay(
        results, market, lambda df: df["bid_depth"] + df["ask_depth"],
        "Order-book depth over time (bid + ask)", "total resting quantity",
        "depth.png", outdir)

    # --- agent execution (with market maker) ---
    paths["inventory"] = _overlay(
        results, agent, lambda df: df["inventory"],
        "Market-maker inventory over time", "inventory", "inventory.png", outdir)
    paths["pnl"] = _overlay(
        results, agent, lambda df: df["total_pnl"],
        "Market-maker total PnL over time", "PnL", "pnl.png", outdir)

    return paths
