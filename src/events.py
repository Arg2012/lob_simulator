"""The event vocabulary that drives the simulator.

Everything — synthetic simulation and historical replay alike — is expressed as a
stream of `Event`s. This is the seam that lets us swap synthetic data for real
exchange data later: only the *producer* of events changes, never the engine that
consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .order import Side


class EventType(Enum):
    ADD_ORDER = "ADD_ORDER"        # submit a resting limit order
    CANCEL_ORDER = "CANCEL_ORDER"  # remove a resting order by id
    MARKET_ORDER = "MARKET_ORDER"  # cross the spread and take liquidity


@dataclass
class Event:
    """A single instruction for the matching engine.

    Not every field is meaningful for every event type:

    * ADD_ORDER    uses side, price and quantity.
    * MARKET_ORDER uses side and quantity (price is `None`).
    * CANCEL_ORDER uses only order_id (side/price kept for a self-describing CSV).
    """

    timestamp: float
    event_type: EventType
    order_id: int
    side: Optional[Side] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    owner: str = "background"
