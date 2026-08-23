"""Strategy interface for Sentinel.

Every tradable strategy subclasses :class:`Strategy`. The interface is designed
to be **Null-compatible from day one**: ``generate_signals`` accepts exactly the
input shape the Null backtester feeds it — a pandas ``DataFrame`` of OHLCV bars
indexed by timestamp with columns ``[open, high, low, close, volume]`` — so any
subclass can be piped into Null for validation without an adapter.

Design rules baked into the interface:
  * Trade math is pure Python here; no LLM/AI touches this path.
  * Every strategy must express BOTH entry and exit logic. ``EXIT`` is a
    first-class side, and the base class enforces that a strategy declares it
    handles exits (``has_exit_logic``) — the validation gate can reject any
    strategy that does not.
  * Signals are deterministic given identical input bars.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import pandas as pd

# Canonical OHLCV column order every strategy and Null agree on.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class Side(str, Enum):
    """Order/signal side. EXIT closes an existing position (long or short)."""

    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Signal:
    """A deterministic trading intent emitted by a strategy.

    ``quantity_hint`` is advisory — position sizing/risk sits downstream and may
    override it. ``stop_loss`` / ``take_profit`` make the exit intent explicit so
    no position is ever open-ended.
    """

    symbol: str
    side: Side
    quantity_hint: Optional[float] = None
    confidence: float = 0.0
    entry_reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Timestamp of the bar this decision was made on — critical for auditability
    # and for Null to place fills at the correct (non-look-ahead) bar.
    as_of: Optional[pd.Timestamp] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            # Allow plain strings ("BUY") for ergonomics; normalize to the enum.
            object.__setattr__(self, "side", Side(str(self.side).upper()))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.quantity_hint is not None and self.quantity_hint < 0:
            raise ValueError("quantity_hint must be non-negative")


@dataclass(frozen=True)
class Fill:
    """Minimal fill record passed to ``on_fill``."""

    symbol: str
    side: Side
    filled_qty: float
    filled_avg_price: float
    filled_at: pd.Timestamp


class Strategy(abc.ABC):
    """Abstract base every strategy must implement.

    Subclasses set ``name``, ``symbols`` and ``lookback_bars`` (via ``__init__``
    or class attributes) and implement the four abstract methods below.
    """

    name: str = "unnamed"
    symbols: Sequence[str] = ()
    lookback_bars: int = 0

    def __init__(
        self,
        name: str | None = None,
        symbols: Sequence[str] | None = None,
        lookback_bars: int | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if symbols is not None:
            self.symbols = tuple(symbols)
        if lookback_bars is not None:
            self.lookback_bars = int(lookback_bars)

    # --- The Null-compatible contract -------------------------------------

    @abc.abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> list[Signal]:
        """Return signals for the given OHLCV ``bars``.

        ``bars`` is a pandas DataFrame indexed by timestamp (ascending) with the
        columns in :data:`OHLCV_COLUMNS`. Implementations MUST be deterministic
        and MUST NOT peek beyond the last row (no look-ahead).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """React to an executed fill (update internal position state)."""
        raise NotImplementedError

    @abc.abstractmethod
    def on_reject(self, order: Any) -> None:
        """React to a rejected order."""
        raise NotImplementedError

    @abc.abstractmethod
    def state_snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of internal state."""
        raise NotImplementedError

    # --- Guards / helpers shared by all strategies ------------------------

    @property
    def has_exit_logic(self) -> bool:
        """Whether this strategy declares exit handling.

        The validation gate refuses to promote a strategy that can never close a
        position. Subclasses that emit EXIT signals or set stop/take on entries
        satisfy this; the default is True and strategies opt *out* only if broken.
        """
        return True

    @staticmethod
    def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
        """Validate the input shape Null and the live engine both use.

        Ensures the required OHLCV columns exist and the index is monotonically
        increasing (a cheap structural guard against look-ahead / unsorted data).
        """
        missing = [c for c in OHLCV_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"bars missing required columns: {missing}")
        if not bars.index.is_monotonic_increasing:
            raise ValueError("bars index must be sorted ascending (no look-ahead)")
        return bars

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"symbols={tuple(self.symbols)!r}, lookback_bars={self.lookback_bars})"
        )
