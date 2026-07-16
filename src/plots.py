"""Matplotlib visualisations of a simulation run.

Every function saves a PNG into the target output directory and returns its path.
Plots use a non-interactive backend so the module works headless (CI, servers).
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

from .matching_engine import MatchingEngine
from .simulator import SimulationResult

_STYLE = {"linewidth": 1.0}


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_midprice(result: SimulationResult, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.timestamps, result.mid, color="#1f77b4", **_STYLE)
    ax.set_title(f"Mid-price through time — {result.scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("mid-price")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, f"{result.scenario}_midprice.png")


def plot_spread(result: SimulationResult, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.timestamps, result.spread, color="#d62728", **_STYLE)
    ax.set_title(f"Spread through time — {result.scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("best ask - best bid")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, f"{result.scenario}_spread.png")


def plot_depth(result: SimulationResult, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.timestamps, result.bid_depth, color="#2ca02c",
            label="bid depth (top 5)", **_STYLE)
    ax.plot(result.timestamps, result.ask_depth, color="#ff7f0e",
            label="ask depth (top 5)", **_STYLE)
    ax.set_title(f"Book depth through time — {result.scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("resting quantity")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, outdir, f"{result.scenario}_depth.png")


def plot_inventory(result: SimulationResult, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.timestamps, result.inventory, color="#9467bd", **_STYLE)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Market-maker inventory through time — {result.scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("net inventory")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, f"{result.scenario}_inventory.png")


def plot_pnl(result: SimulationResult, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.timestamps, result.realised_pnl, label="realised", **_STYLE)
    ax.plot(result.timestamps, result.unrealised_pnl, label="unrealised", **_STYLE)
    ax.plot(result.timestamps, result.total_pnl, label="total",
            color="black", linewidth=1.3)
    ax.set_title(f"Market-maker PnL through time — {result.scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("PnL")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, outdir, f"{result.scenario}_pnl.png")


def plot_book_snapshot(engine: MatchingEngine, outdir: str,
                       name: str = "book_snapshot", levels: int = 10) -> str:
    """Horizontal depth chart of the current book (a classic LOB visual)."""
    bids, asks = engine.book.depth_snapshot(levels=levels)
    fig, ax = plt.subplots(figsize=(8, 5))
    if bids:
        bp, bq = zip(*bids)
        ax.barh([str(p) for p in bp], bq, color="#2ca02c", label="bids")
    if asks:
        ap, aq = zip(*asks)
        ax.barh([str(p) for p in ap], aq, color="#d62728", label="asks")
    ax.set_title("Order-book depth snapshot")
    ax.set_xlabel("resting quantity")
    ax.set_ylabel("price")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, outdir, f"{name}.png")


def save_all_plots(result: SimulationResult, outdir: str) -> List[str]:
    """Generate the full set of per-scenario figures."""
    return [
        plot_midprice(result, outdir),
        plot_spread(result, outdir),
        plot_depth(result, outdir),
        plot_inventory(result, outdir),
        plot_pnl(result, outdir),
    ]


def plot_scenario_comparison(results: Dict[str, SimulationResult],
                             outdir: str) -> str:
    """Bar chart comparing average spread and total PnL across scenarios."""
    from .metrics import compute_metrics

    names = list(results.keys())
    spreads = [compute_metrics(r)["avg_spread"] for r in results.values()]
    pnls = [compute_metrics(r)["total_pnl"] for r in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, spreads, color="#d62728")
    ax1.set_title("Average spread by scenario")
    ax1.set_ylabel("avg spread")
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(names, pnls, color="#1f77b4")
    ax2.set_title("Market-maker total PnL by scenario")
    ax2.set_ylabel("total PnL")
    ax2.tick_params(axis="x", rotation=20)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "scenario_comparison.png")
