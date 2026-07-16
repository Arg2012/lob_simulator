"""Synthetic market generation and the end-to-end simulation loop.

Two responsibilities live here:

1. `generate_order_flow` — a configurable stochastic order-flow model that emits a
   stream of `Event`s. Three named scenarios (normal / high volatility / low
   liquidity) parameterise it differently.
2. `run_scenario` — drives the events through the matching engine while a
   `MarketMaker` quotes into the book, recording a per-event snapshot of the
   market state and the agent's PnL / inventory into a `SimulationResult`.

The order-flow model is deliberately simple and transparent: a latent mid-price
does a random walk, and each event is probabilistically a market order, a
cancellation, or a new limit order placed near the mid.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .events import Event, EventType
from .market_maker import MarketMaker
from .matching_engine import MatchingEngine
from .order import Order, OrderType, Side
from .replay import apply_event
from .trade import Trade


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters controlling the synthetic order-flow model."""

    interarrival: float      # mean time between events (arbitrary time units)
    vol: float               # std dev of each latent mid random-walk step
    p_market: float          # probability an event is a market order
    p_cancel: float          # probability an event is a cancellation
    ladder_levels: int       # price levels seeded on each side at warm-up
    quote_levels: int        # how far from mid new limit orders are placed
    base_size: int           # typical resting order size
    market_size_max: int     # max size of an aggressive market order
    tick: float = 0.01


# The three required market regimes.
SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        interarrival=1.0, vol=0.015, p_market=0.15, p_cancel=0.25,
        ladder_levels=6, quote_levels=5, base_size=6, market_size_max=8),
    "high_volatility": ScenarioConfig(
        interarrival=1.0, vol=0.070, p_market=0.28, p_cancel=0.20,
        ladder_levels=6, quote_levels=6, base_size=5, market_size_max=16),
    "low_liquidity": ScenarioConfig(
        interarrival=2.5, vol=0.025, p_market=0.12, p_cancel=0.35,
        ladder_levels=3, quote_levels=4, base_size=2, market_size_max=4),
}


# ----------------------------------------------------------------------- results
@dataclass
class SimulationResult:
    """Time series of market state and market-maker performance for one run."""

    scenario: str
    timestamps: np.ndarray
    mid: np.ndarray
    spread: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    inventory: np.ndarray
    realised_pnl: np.ndarray
    unrealised_pnl: np.ndarray
    total_pnl: np.ndarray
    trades: List[Trade] = field(default_factory=list)

    # Aggregates needed for execution-quality metrics.
    n_limit_orders: int = 0
    filled_limit_ids: set = field(default_factory=set)
    limit_order_ids: set = field(default_factory=set)
    market_slippage: List[float] = field(default_factory=list)


# ------------------------------------------------------------------- order flow
def generate_order_flow(scenario: str, n_events: int, seed: int = 0,
                        start_price: float = 100.0) -> List[Event]:
    """Generate a synthetic event stream for a named scenario.

    The stream begins with a warm-up ladder that seeds a two-sided book, followed
    by `n_events` stochastic events (market orders, cancels and new limit orders).
    """
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'. "
                       f"Choose from {list(SCENARIOS)}.")
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    tick = cfg.tick

    events: List[Event] = []
    ids = itertools.count(1)
    ts = 0.0
    # Track live resting orders we created so cancellations target real ids.
    live: Dict[int, Side] = {}

    def snap(price: float) -> float:
        return round(round(price / tick) * tick, 2)

    # --- warm-up: seed a symmetric ladder so the book opens two-sided ----------
    for i in range(1, cfg.ladder_levels + 1):
        for side in (Side.BUY, Side.SELL):
            price = start_price - i * tick if side is Side.BUY else start_price + i * tick
            oid = next(ids)
            events.append(Event(ts, EventType.ADD_ORDER, oid, side,
                                snap(price), cfg.base_size))
            live[oid] = side

    # --- stochastic flow -------------------------------------------------------
    mid = start_price
    for _ in range(n_events):
        ts += float(rng.exponential(cfg.interarrival))
        mid = max(tick * 5, mid + float(rng.normal(0.0, cfg.vol)))
        r = rng.random()

        if r < cfg.p_market:
            # Aggressive market order.
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = int(rng.integers(1, cfg.market_size_max + 1))
            events.append(Event(ts, EventType.MARKET_ORDER, next(ids),
                                side, None, qty))

        elif r < cfg.p_market + cfg.p_cancel and live:
            # Cancel a random live order.
            victim = int(rng.choice(list(live.keys())))
            side = live.pop(victim)
            events.append(Event(ts, EventType.CANCEL_ORDER, victim, side))

        else:
            # New passive limit order, placed a few ticks off the mid so it rests.
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            offset = (int(rng.integers(0, cfg.quote_levels)) + 1) * tick
            price = mid - offset if side is Side.BUY else mid + offset
            qty = int(rng.integers(1, cfg.base_size + 1))
            oid = next(ids)
            events.append(Event(ts, EventType.ADD_ORDER, oid, side,
                                snap(price), qty))
            live[oid] = side

    return events


# ----------------------------------------------------------------------- driver
def run_scenario(scenario: str, n_events: int = 5000, seed: int = 0,
                 with_market_maker: bool = True,
                 requote_interval: int = 5,
                 market_maker: Optional[MarketMaker] = None) -> SimulationResult:
    """Run one scenario end-to-end and return its recorded time series."""
    events = generate_order_flow(scenario, n_events, seed)
    engine = MatchingEngine()
    mm = market_maker if market_maker is not None else MarketMaker()

    # Owner registry: order_id -> owner ("background" / "MM"). Lets us attribute
    # fills to the agent even after its order has left the book.
    owner: Dict[int, str] = {}

    ts_hist, mid_hist, spread_hist = [], [], []
    bd_hist, ad_hist, inv_hist = [], [], []
    rpnl_hist, upnl_hist, tpnl_hist = [], [], []

    result = SimulationResult(
        scenario=scenario,
        timestamps=np.array([]), mid=np.array([]), spread=np.array([]),
        bid_depth=np.array([]), ask_depth=np.array([]), inventory=np.array([]),
        realised_pnl=np.array([]), unrealised_pnl=np.array([]),
        total_pnl=np.array([]),
    )

    last_mid = 100.0
    for i, event in enumerate(events):
        # 1) Market maker refreshes its two-sided quote periodically.
        current_mid = engine.book.midpoint()
        if with_market_maker and current_mid is not None and i % requote_interval == 0:
            mm.requote(engine, current_mid, event.timestamp, owner)

        # 2) Register ownership of background orders for fill attribution.
        if event.event_type in (EventType.ADD_ORDER, EventType.MARKET_ORDER):
            owner.setdefault(event.order_id, event.owner)
        if event.event_type is EventType.ADD_ORDER:
            result.n_limit_orders += 1
            result.limit_order_ids.add(event.order_id)

        # Pre-trade mid for slippage measurement on market orders.
        pre_mid = engine.book.midpoint()
        pre_mid = pre_mid if pre_mid is not None else last_mid

        # 3) Apply the event.
        trades = apply_event(engine, event)

        # 4) Post-processing: attribute fills, measure slippage, book invariants.
        if trades:
            filled_qty = sum(t.quantity for t in trades)
            vwap = sum(t.price * t.quantity for t in trades) / filled_qty
            for t in trades:
                result.filled_limit_ids.add(t.maker_order_id)
                result.filled_limit_ids.add(t.taker_order_id)
                # Market-maker fill attribution (the MM is always the maker here).
                if owner.get(t.maker_order_id) == mm.owner:
                    mm.on_fill(t.aggressor.opposite, t.price, t.quantity)
                if owner.get(t.taker_order_id) == mm.owner:
                    mm.on_fill(t.aggressor, t.price, t.quantity)

            if event.event_type is EventType.MARKET_ORDER:
                # Positive slippage = executed worse than the pre-trade mid.
                sign = 1.0 if event.side is Side.BUY else -1.0
                result.market_slippage.append(sign * (vwap - pre_mid))

        result.trades.extend(trades)

        # 5) Snapshot the market + agent state.
        mid = engine.book.midpoint()
        mid = mid if mid is not None else last_mid
        last_mid = mid
        spread = engine.book.spread()

        ts_hist.append(event.timestamp)
        mid_hist.append(mid)
        spread_hist.append(spread if spread is not None else np.nan)
        bd_hist.append(engine.book.bid_depth(levels=5))
        ad_hist.append(engine.book.ask_depth(levels=5))
        inv_hist.append(mm.inventory)
        rpnl_hist.append(mm.realised_pnl)
        upnl_hist.append(mm.unrealised_pnl(mid))
        tpnl_hist.append(mm.total_pnl(mid))

    # Guard: the reconstructed book must still be internally consistent.
    engine.book.check_invariants()

    result.timestamps = np.asarray(ts_hist)
    result.mid = np.asarray(mid_hist)
    result.spread = np.asarray(spread_hist)
    result.bid_depth = np.asarray(bd_hist)
    result.ask_depth = np.asarray(ad_hist)
    result.inventory = np.asarray(inv_hist)
    result.realised_pnl = np.asarray(rpnl_hist)
    result.unrealised_pnl = np.asarray(upnl_hist)
    result.total_pnl = np.asarray(tpnl_hist)
    return result
