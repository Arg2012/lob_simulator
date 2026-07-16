# High-Frequency Limit Order Book Simulator

An event-driven **limit order book (LOB)** and **matching engine**, written from
scratch in Python. It supports limit orders, market orders, cancellations and
FIFO queue dynamics; reconstructs order books from historical event streams; and
is used to study spread, depth, inventory and execution quality across different
market regimes.

The project is deliberately compact and readable. Every component — the book, the
matching engine, the event system, the replay layer and the market-making agent —
is small enough to reason about, yet the pieces compose into a realistic
microstructure sandbox.

---

## Highlights

- **Price-time-priority matching engine** — better prices match first; within a
  price level, earlier orders match first (strict FIFO).
- **Full order lifecycle** — limit buys/sells, market buys/sells, partial fills,
  multi-level sweeps and cancellations.
- **Event-driven core** — everything is a stream of `ADD_ORDER` / `MARKET_ORDER` /
  `CANCEL_ORDER` events, so synthetic and historical data share one code path.
- **Historical replay** — reconstruct a book from a CSV of events, exactly as you
  would from a real exchange feed.
- **Synthetic markets** — three regimes: *normal*, *high volatility*, *low
  liquidity*, each with distinct order flow.
- **Market-making agent** — posts two-sided quotes, skews on inventory, and tracks
  realised / unrealised PnL.
- **Metrics & plots** — spread, depth, inventory, PnL, fill rate and slippage,
  with publication-quality figures saved to `outputs/`.
- **Tested** — a `pytest` suite covering FIFO, partial fills, multi-level matching,
  cancellations and book invariants.

---

## Project structure

```
lob_simulator/
├── src/
│   ├── order.py            # Order, Side, OrderType
│   ├── trade.py            # Trade record
│   ├── events.py           # Event vocabulary (ADD / CANCEL / MARKET)
│   ├── order_book.py       # The book: price levels, FIFO queues, id lookup
│   ├── matching_engine.py  # Price-time-priority crossing logic
│   ├── replay.py           # CSV load/write + historical replay
│   ├── simulator.py        # Synthetic order flow + end-to-end run loop
│   ├── market_maker.py     # Inventory-aware market-making agent
│   ├── metrics.py          # Execution-quality metrics
│   └── plots.py            # Matplotlib figures
├── data/
│   └── sample_events.csv   # Generated synthetic event stream (several thousand)
├── tests/                  # pytest suite
├── outputs/                # Generated figures + metrics_summary.csv
├── main.py                 # Demo driver: generate → replay → simulate → plot
└── README.md
```

---

## How it works

### The order book

Each side of the book is a dictionary mapping `price -> PriceLevel`, and each
`PriceLevel` holds a `collections.deque` of resting orders. Alongside each side we
keep a **sorted list of live prices**, maintained with `bisect`. This gives:

| Operation                | Cost                                   |
|--------------------------|----------------------------------------|
| best bid / best ask      | `O(1)` (peek the sorted list)          |
| add a new price level    | `O(k)` (bisect insort)                 |
| cancel / drop a level    | `O(k)` + `O(1)` id lookup              |
| cancel by order id       | `O(1)` lookup + `O(q)` queue removal   |

where `k` is the number of *distinct* price levels and `q` the depth of a single
queue — both small in practice. Order-id lookup is a flat dict, so cancellations
are effectively constant time.

The book exposes exactly the queries a microstructure study needs: `best_bid`,
`best_ask`, `spread`, `midpoint`, `bid_depth`, `ask_depth` and a full
`depth_snapshot`.

### Price-time priority (the matching rule)

When an aggressive order arrives, the engine walks the **opposite** side of the
book from the best price outward:

1. **Price priority** — it consumes the best price level first, then the next,
   and so on.
2. **Time priority** — within a level it always fills the *front* of the FIFO
   queue first (the earliest-arriving order).
3. Every fill executes at the **resting (maker) order's price**, so the aggressor
   receives price improvement — just like a real exchange.

### Matching logic

- **Market order** — takes liquidity across as many levels as needed until it is
  filled or the book is empty. Any unfilled remainder is discarded (a market order
  has no price at which to rest).
- **Limit order** — first crosses against any *marketable* liquidity (opposite
  orders at or better than its limit price). It stops as soon as the best opposite
  price is beyond its limit, and any remaining quantity **rests** in the book.
- **Cancellation** — removes a resting order by id in `O(1)` lookup, tidying up the
  price level (and the sorted-price list) if it becomes empty.

The book maintains a `check_invariants()` guard used by the tests and after each
simulation run: the book is never crossed, every level's cached quantity equals
the sum of its orders, and the sorted-price lists mirror the level dictionaries.

### The event system

Everything runs through a single event vocabulary (`events.py`):

```
ADD_ORDER      submit a resting limit order
MARKET_ORDER   cross the spread and take liquidity
CANCEL_ORDER   remove a resting order by id
```

`replay.apply_event` is the one function translating events into engine calls, so
**synthetic simulation and historical replay share the same execution path**. That
is the seam that makes it easy to swap in real exchange data later — only the
producer of events changes.

### Historical replay

`replay.py` reconstructs a book by processing every event in a CSV sequentially.
The schema is intentionally exchange-agnostic:

```
timestamp,event_type,order_id,side,price,quantity
0.000000,ADD_ORDER,1,BUY,99.99,6
0.000000,ADD_ORDER,2,SELL,100.01,6
1.284000,MARKET_ORDER,842,SELL,,4
1.902000,CANCEL_ORDER,17,BUY,,
```

To use real data, map your feed into these columns and call `replay_file(path)`.

### Simulation

`simulator.py` contains a small, transparent order-flow model. A latent mid-price
follows a random walk; each event is probabilistically a market order, a
cancellation, or a new limit order placed a few ticks off the mid. Three scenarios
parameterise it:

| Scenario          | Character                                             |
|-------------------|-------------------------------------------------------|
| `normal`          | Moderate volatility, tight spread, deep book          |
| `high_volatility` | Large price steps, more aggressive flow, thinner book |
| `low_liquidity`   | Sparse arrivals, small sizes, shallow book            |

### The market-making agent

`market_maker.py` posts a two-sided quote around the mid every few events. It:

- places a **bid** and an **ask** at `mid ± half_spread`,
- **skews** both quotes against its inventory (a growing long position lowers both
  quotes, encouraging it to sell and mean-revert toward flat),
- stops quoting a side that would breach its inventory cap,
- tracks **inventory**, **realised PnL** (average-cost accounting on round-trips)
  and **unrealised PnL** (mark-to-market of the open position at the mid).

### Analysis & metrics

`metrics.py` computes, per run: average / median spread, average bid & ask depth,
inventory through time, realised / unrealised / total PnL, **fill rate** (fraction
of submitted limit orders that executed) and **slippage** (market-order execution
VWAP versus the pre-trade mid). `plots.py` renders mid-price, spread, depth,
inventory and PnL time series, a book-depth snapshot, and a cross-scenario
comparison — all saved to `outputs/`.

---

## Getting started

### Requirements

- Python 3.11+
- `numpy`, `pandas`, `matplotlib`, `pytest`

```bash
pip install numpy pandas matplotlib pytest
```

### Run the full demo

Generates the sample event stream, demonstrates replay, runs all three scenarios,
prints a metrics table and writes every figure to `outputs/`:

```bash
python main.py
```

Options:

```bash
python main.py --events 8000     # more events per scenario / larger sample
python main.py --seed 7          # different random seed (fully reproducible)
```

### Run the tests

```bash
python -m pytest -q
```

### Use it as a library

```python
from src.matching_engine import MatchingEngine
from src.order import Order, OrderType, Side

engine = MatchingEngine()
engine.submit_limit_order(Order(1, Side.SELL, 5, 100.0, OrderType.LIMIT))
engine.submit_limit_order(Order(2, Side.SELL, 5, 101.0, OrderType.LIMIT))

# A market buy sweeps the book in price-time priority.
trades = engine.submit_market_order(Order(3, Side.BUY, 7, None, OrderType.MARKET))
for t in trades:
    print(t.quantity, "@", t.price)      # 5 @ 100.0, then 2 @ 101.0

print("best ask:", engine.book.best_ask())   # 101.0 (3 left resting)
```

Replay a historical file:

```python
from src.replay import replay_file
engine = replay_file("data/sample_events.csv")
print(engine.book.best_bid(), engine.book.best_ask(), engine.book.spread())
```

Run a scenario and compute metrics:

```python
from src.simulator import run_scenario
from src.metrics import format_metrics

result = run_scenario("high_volatility", n_events=5000, seed=0)
print(format_metrics(result))
```

---

## Reading the results

Some findings worth discussing (produced by `python main.py`):

- **Spread widens with volatility.** The `high_volatility` scenario shows a
  materially larger average spread and higher market-order slippage than `normal`.
- **The naive market maker is adversely selected.** In trending / volatile regimes
  its inventory drifts toward the cap and it loses money — a textbook illustration
  of *inventory risk* and *adverse selection*. This is exactly what motivates the
  inventory-skew term (and, in a richer model, dynamic spread widening and hedging).
- **Depth and fill rate move together with liquidity.** The `low_liquidity`
  scenario has a shallow book and a lower limit-order fill rate.

These are honest outputs of a simple model — the point is that the simulator is
rich enough to *observe* real microstructure effects and to serve as a testbed for
smarter strategies.

---

## Design notes

- **Separation of concerns.** The book *stores* liquidity; the engine *consumes*
  it; the simulator *produces* events; the agent *reacts*. Each can be tested and
  swapped independently.
- **One execution path.** Synthetic and historical flow both go through
  `apply_event`, which keeps replay and simulation honest with respect to each
  other.
- **From scratch.** No third-party order-book library is used — the matching
  engine, queues and priority logic are all implemented here.

## Possible extensions

- Iceberg / hidden orders, order modification (price/size amends).
- Latency modelling and a proper discrete-event scheduler.
- Smarter agents (Avellaneda–Stoikov market making, execution algos like TWAP/VWAP).
- Ingest a real L3 message feed by mapping it onto the existing event schema.
