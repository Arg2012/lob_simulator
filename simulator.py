"""Event-driven market simulator and synthetic order-flow generation.

The simulator is a sequential event loop. On each step it:

1. lets the market maker re-quote (cancel + repost around the mid),
2. generates a batch of synthetic background order flow,
3. feeds every resulting event through the shared :class:`MatchingEngine`,
4. records a snapshot of the book and market-maker state.

Three scenarios (``normal``, ``high_volatility``, ``low_liquidity``) share the
same generator but with different parameters, so each produces visibly
different spread / depth / inventory / execution behaviour.

All randomness is seeded, so runs are reproducible. The generated data is
**synthetic** - it is not, and does not claim to be, real exchange data.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from events import Event, EventType, add_order, cancel_order, market_order
from matching_engine import MatchingEngine
from market_maker import MarketMaker
from order import Side

TICK = 0.01
START_PRICE = 100.0


@dataclass
class ScenarioConfig:
    """Parameters that shape one synthetic market scenario."""

    name: str
    n_steps: int = 1500
    limits_per_step: int = 6          # background limit orders posted per step
    market_prob: float = 0.15         # chance a given order is a market order
    cancel_prob: float = 0.20         # chance to also cancel a resting order
    volatility: float = 0.02          # std of the fair-value random walk (price)
    mean_reversion: float = 0.02      # pull of the fair value back to its anchor
    half_spread_ticks: int = 5        # how far limits are posted from fair value
    max_size: int = 5                 # max order size
    seed: int = 1


# Three ready-made scenarios with deliberately different characteristics.
SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        name="normal", volatility=0.02, market_prob=0.15, cancel_prob=0.20,
        limits_per_step=6, half_spread_ticks=5, max_size=5, seed=1),
    "high_volatility": ScenarioConfig(
        name="high_volatility", volatility=0.10, market_prob=0.30,
        cancel_prob=0.20, limits_per_step=6, half_spread_ticks=9, max_size=5,
        seed=2),
    "low_liquidity": ScenarioConfig(
        name="low_liquidity", volatility=0.03, market_prob=0.30,
        cancel_prob=0.45, limits_per_step=2, half_spread_ticks=6, max_size=2,
        seed=3),
}


@dataclass
class Snapshot:
    """One row of recorded market / market-maker state."""

    step: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    bid_depth: int
    ask_depth: int
    inventory: int
    realised_pnl: float
    unrealised_pnl: float
    total_pnl: float


@dataclass
class SimulationResult:
    """Everything produced by one event-loop run."""

    config: ScenarioConfig
    snapshots: List[Snapshot]
    market_maker: MarketMaker
    trades: list = field(default_factory=list)
    slippages: List[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Snapshots as a tidy DataFrame for metrics and plotting."""
        return pd.DataFrame([asdict(s) for s in self.snapshots])


@dataclass
class ScenarioResult:
    """A scenario evaluated from two complementary angles.

    We separate *market conditions* from *agent performance* so that neither
    contaminates the other:

    * ``market`` - the same order flow run **without** the market maker. Its
      spread and depth characterise the market's intrinsic liquidity.
    * ``agent`` - the same seeded order flow run **with** the market maker
      participating. Its inventory, PnL, fill rate and slippage measure how the
      agent performs *in* that market.
    """

    config: ScenarioConfig
    market: SimulationResult
    agent: SimulationResult


class FlowGenerator:
    """Generates synthetic background order flow for one scenario.

    A latent ``fair`` value random-walks each step. Limit orders are posted
    around it (buys below, sells above) to build a two-sided book; market
    orders consume liquidity; cancels remove resting orders. Occasionally a
    limit is posted aggressively so that limit orders also cross the book.
    """

    def __init__(self, config: ScenarioConfig, id_gen: Iterator[int],
                 tick: float = TICK, start_price: float = START_PRICE) -> None:
        self.cfg = config
        self.id_gen = id_gen
        self.tick = tick
        self.fair = start_price
        self.anchor = start_price
        self.rng = np.random.RandomState(config.seed)
        self.live_ids: List[int] = []

    def _price(self, ticks_from_fair: int) -> float:
        return round(self.fair + ticks_from_fair * self.tick, 2)

    def step(self, timestamp: int) -> List[Event]:
        """Produce the background events for a single time step."""
        events: List[Event] = []
        # Latent fair value: a mean-reverting Gaussian random walk. The gentle
        # pull to the anchor keeps the book two-sided instead of drifting off.
        pull = self.cfg.mean_reversion * (self.anchor - self.fair)
        self.fair = round(self.fair + pull + self.rng.normal(0.0, self.cfg.volatility), 2)

        for _ in range(self.cfg.limits_per_step):
            # Occasionally cancel a resting order.
            if self.live_ids and self.rng.rand() < self.cfg.cancel_prob:
                idx = self.rng.randint(len(self.live_ids))
                oid = self.live_ids.pop(idx)
                events.append(cancel_order(timestamp, oid))

            side = Side.BUY if self.rng.rand() < 0.5 else Side.SELL
            qty = int(self.rng.randint(1, self.cfg.max_size + 1))

            if self.rng.rand() < self.cfg.market_prob:
                oid = next(self.id_gen)
                events.append(market_order(timestamp, oid, side, qty))
                continue

            # Limit order. Usually passive (rests), occasionally aggressive.
            aggressive = self.rng.rand() < 0.15
            offset = self.rng.randint(1, self.cfg.half_spread_ticks + 1)
            if side is Side.BUY:
                ticks = offset if not aggressive else -offset
            else:
                ticks = -offset if not aggressive else offset
            # For a buy, positive price offset means below fair (passive).
            price = self._price(-ticks if side is Side.BUY else ticks)

            oid = next(self.id_gen)
            events.append(add_order(timestamp, oid, side, price, qty))
            self.live_ids.append(oid)

        return events


def _snapshot(step: int, book, mm: MarketMaker) -> Snapshot:
    mid = book.mid()
    mm.mark(mid)
    return Snapshot(
        step=step,
        best_bid=book.best_bid(),
        best_ask=book.best_ask(),
        mid=mid,
        spread=book.spread(),
        bid_depth=book.total_bid_depth(),
        ask_depth=book.total_ask_depth(),
        inventory=mm.inventory,
        realised_pnl=mm.realised,
        unrealised_pnl=mm.unrealised,
        total_pnl=mm.total,
    )


def run_simulation(config: ScenarioConfig,
                   with_market_maker: bool = True) -> SimulationResult:
    """Run one scenario end-to-end and return the recorded result."""
    id_gen: Iterator[int] = itertools.count(1)
    engine = MatchingEngine()
    flow = FlowGenerator(config, id_gen)
    mm = MarketMaker(id_gen, fallback_price=START_PRICE)

    snapshots: List[Snapshot] = []
    slippages: List[float] = []

    for step in range(config.n_steps):
        if with_market_maker:
            for event in mm.requote(engine.book, step):
                mm.on_trades(engine.process_event(event))

        for event in flow.step(step):
            mid_before = engine.book.mid()
            trades = engine.process_event(event)
            mm.on_trades(trades)

            # Slippage: for market orders, how far the average fill price landed
            # from the mid just before the order arrived (positive = worse).
            if event.event_type is EventType.MARKET_ORDER and trades and mid_before is not None:
                filled = sum(t.quantity for t in trades)
                avg_fill = sum(t.price * t.quantity for t in trades) / filled
                if event.side is Side.BUY:
                    slippages.append(avg_fill - mid_before)
                else:
                    slippages.append(mid_before - avg_fill)

        snapshots.append(_snapshot(step, engine.book, mm))

    return SimulationResult(config, snapshots, mm, engine.trades, slippages)


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Run a scenario twice: native market flow, then with the market maker."""
    market = run_simulation(config, with_market_maker=False)
    agent = run_simulation(config, with_market_maker=True)
    return ScenarioResult(config, market, agent)


def generate_event_stream(config: ScenarioConfig) -> List[Event]:
    """Generate a pure background-flow event stream (no market maker).

    Used to produce the synthetic ``data/sample_events.csv`` that the replay
    engine reconstructs. Contains several thousand events for a default config.
    """
    id_gen: Iterator[int] = itertools.count(1)
    flow = FlowGenerator(config, id_gen)
    events: List[Event] = []
    for step in range(config.n_steps):
        events.extend(flow.step(step))
    return events
