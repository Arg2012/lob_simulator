"""Event model and CSV (de)serialisation.

Everything that mutates the book is expressed as an :class:`Event`. The same
event objects drive the live simulator and the historical replay, so the two
paths share one code path through the matching engine.

CSV schema (one event per row)::

    timestamp,event_type,order_id,side,price,quantity

* ``ADD_ORDER``    - a limit order (``side``, ``price`` and ``quantity`` set).
* ``MARKET_ORDER`` - a market order (``price`` blank; ``side``/``quantity`` set).
* ``CANCEL_ORDER`` - cancel a resting order (only ``order_id`` is required).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from order import Side


class EventType(str, Enum):
    """The three event kinds the engine understands."""

    ADD_ORDER = "ADD_ORDER"
    MARKET_ORDER = "MARKET_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"


@dataclass
class Event:
    """A single instruction for the matching engine.

    Fields not relevant to a given event type are ``None`` (e.g. a cancel only
    needs ``order_id``; a market order has no ``price``).
    """

    timestamp: int
    event_type: EventType
    order_id: int
    side: Optional[Side] = None
    price: Optional[float] = None
    quantity: Optional[int] = None


# --- Convenience constructors -------------------------------------------------


def add_order(timestamp: int, order_id: int, side: Side, price: float,
              quantity: int) -> Event:
    """Build an ADD_ORDER (limit) event."""
    return Event(timestamp, EventType.ADD_ORDER, order_id, side, price, quantity)


def market_order(timestamp: int, order_id: int, side: Side,
                 quantity: int) -> Event:
    """Build a MARKET_ORDER event."""
    return Event(timestamp, EventType.MARKET_ORDER, order_id, side, None, quantity)


def cancel_order(timestamp: int, order_id: int) -> Event:
    """Build a CANCEL_ORDER event."""
    return Event(timestamp, EventType.CANCEL_ORDER, order_id)


# --- CSV (de)serialisation ----------------------------------------------------

CSV_HEADER = ["timestamp", "event_type", "order_id", "side", "price", "quantity"]


def write_events_csv(events: List[Event], path: str) -> None:
    """Write an event stream to ``path`` using the documented CSV schema."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for e in events:
            writer.writerow([
                e.timestamp,
                e.event_type.value,
                e.order_id,
                e.side.value if e.side is not None else "",
                "" if e.price is None else f"{e.price:.2f}",
                "" if e.quantity is None else e.quantity,
            ])


def read_events_csv(path: str) -> List[Event]:
    """Read an event stream previously written with :func:`write_events_csv`."""
    events: List[Event] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            side = Side(row["side"]) if row["side"] else None
            price = float(row["price"]) if row["price"] else None
            qty = int(row["quantity"]) if row["quantity"] else None
            events.append(Event(
                timestamp=int(row["timestamp"]),
                event_type=EventType(row["event_type"]),
                order_id=int(row["order_id"]),
                side=side,
                price=price,
                quantity=qty,
            ))
    return events
