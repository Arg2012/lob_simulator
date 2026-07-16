"""Historical replay: reconstruct an order book from an event stream.

The replay engine reconstructs the state of the book by processing every event in
the file sequentially — exactly the procedure you would use against a real
exchange's message feed. The only exchange-specific piece is *reading* the events;
`apply_event` is completely agnostic about where they came from, which is what
makes swapping in real data straightforward.

CSV schema (one event per row)::

    timestamp,event_type,order_id,side,price,quantity

* ADD_ORDER    rows carry side, price and quantity.
* MARKET_ORDER rows carry side and quantity (price blank).
* CANCEL_ORDER rows carry only order_id (side/price kept for readability).
"""

from __future__ import annotations

import csv
from typing import List, Optional

from .events import Event, EventType
from .matching_engine import MatchingEngine
from .order import Order, OrderType, Side
from .trade import Trade


def _parse_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    return float(text) if text else None


def _parse_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    return int(text) if text else None


def load_events(path: str) -> List[Event]:
    """Read an event CSV into a list of `Event`s (order preserved)."""
    events: List[Event] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            side_text = (row.get("side") or "").strip()
            events.append(Event(
                timestamp=float(row["timestamp"]),
                event_type=EventType(row["event_type"].strip()),
                order_id=int(row["order_id"]),
                side=Side(side_text) if side_text else None,
                price=_parse_float(row.get("price", "")),
                quantity=_parse_int(row.get("quantity", "")),
            ))
    return events


def write_events(path: str, events: List[Event]) -> None:
    """Persist an event stream to CSV in the schema `load_events` expects."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "event_type", "order_id",
                         "side", "price", "quantity"])
        for e in events:
            writer.writerow([
                f"{e.timestamp:.6f}",
                e.event_type.value,
                e.order_id,
                e.side.value if e.side is not None else "",
                "" if e.price is None else f"{e.price:.2f}",
                "" if e.quantity is None else e.quantity,
            ])


def apply_event(engine: MatchingEngine, event: Event) -> List[Trade]:
    """Apply a single event to the engine, returning any resulting trades.

    This is the one function that translates the abstract event vocabulary into
    concrete engine calls; both replay and live simulation route through it.
    """
    if event.event_type is EventType.ADD_ORDER:
        order = Order(
            order_id=event.order_id,
            side=event.side,
            quantity=event.quantity,
            price=event.price,
            order_type=OrderType.LIMIT,
            timestamp=event.timestamp,
            owner=event.owner,
        )
        return engine.submit_limit_order(order)

    if event.event_type is EventType.MARKET_ORDER:
        order = Order(
            order_id=event.order_id,
            side=event.side,
            quantity=event.quantity,
            price=None,
            order_type=OrderType.MARKET,
            timestamp=event.timestamp,
            owner=event.owner,
        )
        return engine.submit_market_order(order)

    if event.event_type is EventType.CANCEL_ORDER:
        engine.cancel_order(event.order_id)
        return []

    raise ValueError(f"Unknown event type: {event.event_type}")


def replay(events: List[Event],
           engine: Optional[MatchingEngine] = None) -> MatchingEngine:
    """Reconstruct a book by applying every event in sequence."""
    engine = engine if engine is not None else MatchingEngine()
    for event in events:
        apply_event(engine, event)
    return engine


def replay_file(path: str,
                engine: Optional[MatchingEngine] = None) -> MatchingEngine:
    """Convenience wrapper: load a CSV and replay it."""
    return replay(load_events(path), engine)
