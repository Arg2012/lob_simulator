"""Historical event-stream replay / order-book reconstruction.

The replay engine reads a CSV event stream and re-applies every event, in
order, through the **same** :class:`MatchingEngine` used by the live simulator.
Reconstructing the book is therefore nothing more than replaying the events:
after event *k* the book is exactly what it would have been at that point in
history.

The engine is deliberately agnostic about *where* the events come from. The
bundled ``data/sample_events.csv`` is synthetic (see the README), but a real
exchange message feed could be converted into the same CSV schema and replayed
without touching a single line of matching logic.
"""

from __future__ import annotations

from typing import List, Optional

from events import Event, read_events_csv
from matching_engine import MatchingEngine
from order_book import OrderBook
from trade import Trade


class ReplayEngine:
    """Reconstructs an order book by replaying a stream of events."""

    def __init__(self) -> None:
        self.engine = MatchingEngine()
        self.processed = 0

    @property
    def book(self) -> OrderBook:
        """The order book as reconstructed so far."""
        return self.engine.book

    @property
    def trades(self) -> List[Trade]:
        """All trades generated during replay."""
        return self.engine.trades

    def step(self) -> None:  # pragma: no cover - trivial helper
        raise NotImplementedError("Use replay(); events are supplied externally")

    def replay(self, events: List[Event], up_to: Optional[int] = None) -> OrderBook:
        """Replay ``events[:up_to]`` and return the reconstructed book.

        Args:
            events: The ordered event stream.
            up_to: If given, stop after this many events (lets you inspect the
                book at any historical point). ``None`` replays everything.
        """
        end = len(events) if up_to is None else up_to
        for event in events[self.processed:end]:
            self.engine.process_event(event)
            self.processed += 1
        return self.book


def load_and_replay(path: str, up_to: Optional[int] = None) -> ReplayEngine:
    """Load a CSV event stream and replay it, returning the engine.

    The returned engine exposes ``.book`` and ``.trades`` for inspection.
    """
    events = read_events_csv(path)
    engine = ReplayEngine()
    engine.replay(events, up_to=up_to)
    return engine
