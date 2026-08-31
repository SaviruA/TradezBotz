"""Symbols to pay special attention to.

A watchlist changes what gets fetched **first** and what gets **reported**. It
must never change what gets *measured*.

That distinction is the whole design, and it is not pedantry. A hand-picked list
of symbols is picked by a human who already knows how those symbols did --
that is the researcher's own memory acting as lookahead, and it is invisible to
every guardrail in this repository. `observed_at` cannot catch it, the trial
registry cannot see it, and the Deflated Sharpe has no way to know the universe
was chosen after the fact. Backtesting a watchlist would be the single most
effective way to fool ourselves available here, and it would produce beautiful
numbers while doing it.

So the watchlist is wired into exactly three places:

  backfill priority     watched symbols are fetched before the rest, so a name
                        you care about is never the one missing coverage
  intraday priority     same, for the session reduction
  reporting             `status` and `watch status` show them specifically

and it is deliberately absent from the labelling and backtest paths.
`guard_universe` exists to make an accidental connection fail loudly rather than
silently produce a flattering result.

The file lives in the repository rather than in the encrypted state blob so it
is versioned, diffable, and editable by hand. A reason is required for each
entry: "why is this here" is the first thing anyone asks six weeks later, and an
undocumented watchlist becomes superstition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

#: Default location, relative to the repository root.
WATCHLIST_FILE = "watchlist.yml"


class WatchlistError(RuntimeError):
    pass


@dataclass(frozen=True)
class WatchedSymbol:
    symbol: str
    reason: str
    added: date

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "reason": self.reason,
                "added": self.added.isoformat()}


class Watchlist:
    """A versioned list of symbols warranting extra attention."""

    def __init__(self, path: str | Path = WATCHLIST_FILE) -> None:
        self.path = Path(path)

    def load(self) -> list[WatchedSymbol]:
        if not self.path.exists():
            return []
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        out: list[WatchedSymbol] = []
        for entry in raw.get("symbols") or []:
            if not isinstance(entry, dict) or not entry.get("symbol"):
                continue
            added = entry.get("added")
            if isinstance(added, str):
                added = date.fromisoformat(added)
            elif not isinstance(added, date):
                added = date.today()
            out.append(WatchedSymbol(
                symbol=str(entry["symbol"]).strip().upper(),
                reason=str(entry.get("reason") or "").strip(),
                added=added,
            ))
        return out

    def symbols(self) -> list[str]:
        return [w.symbol for w in self.load()]

    #: Written above the data on every save. The warning belongs in the file
    #: rather than only in this module, because the file is what someone edits
    #: at 2am six months from now.
    HEADER = (
        "# Symbols to pay special attention to.\n"
        "#\n"
        "# Affects FETCH PRIORITY and REPORTING only -- never the backtest\n"
        "# universe. A hand-picked list is chosen by someone who already knows\n"
        "# how those names did, so measuring only them is lookahead laundered\n"
        "# through human memory, and no guardrail in this repo can detect it.\n"
        "# See src/tradezbotz/research/watchlist.py.\n"
        "#\n"
        "# Edit by hand or use: python -m tradezbotz watch add SYM --reason \"...\"\n"
    )

    def save(self, entries: list[WatchedSymbol]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "symbols": [e.to_dict() for e in sorted(entries, key=lambda e: e.symbol)]
        }
        self.path.write_text(
            self.HEADER + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def add(self, symbol: str, reason: str) -> WatchedSymbol:
        """Add a symbol. A reason is required, not optional.

        An undocumented watchlist entry becomes superstition: nobody remembers
        why it is there, so nobody is willing to remove it.
        """
        symbol = symbol.strip().upper()
        if not symbol:
            raise WatchlistError("symbol cannot be empty")
        if not reason.strip():
            raise WatchlistError(
                f"a reason is required for {symbol}. Six weeks from now the "
                "first question will be why it is on the list."
            )
        entries = [e for e in self.load() if e.symbol != symbol]
        entry = WatchedSymbol(symbol, reason.strip(), date.today())
        entries.append(entry)
        self.save(entries)
        return entry

    def remove(self, symbol: str) -> bool:
        symbol = symbol.strip().upper()
        entries = self.load()
        kept = [e for e in entries if e.symbol != symbol]
        if len(kept) == len(entries):
            return False
        self.save(kept)
        return True

    def contains(self, symbol: str) -> bool:
        return symbol.strip().upper() in set(self.symbols())


def prioritise(pending: list[str], watched: list[str]) -> list[str]:
    """Reorder a queue so watched symbols come first.

    Reordering only -- nothing is added and nothing is dropped. A watchlist that
    could *remove* symbols from the queue would quietly shrink the measured
    universe, which is the failure this module exists to prevent.
    """
    watch = {s.upper() for s in watched}
    first = [s for s in pending if s.upper() in watch]
    rest = [s for s in pending if s.upper() not in watch]
    return first + rest


def guard_universe(universe: list[str], watched: list[str]) -> None:
    """Refuse a research universe that looks like it came from the watchlist.

    Called from the paths that build a backtest population. A hand-picked list
    is chosen by someone who already knows the outcomes, so measuring only those
    names is lookahead laundered through human memory -- and unlike every other
    bias here, nothing in the data would reveal it.

    The check is deliberately crude: if the universe is small and consists
    mostly of watched names, something has gone wrong upstream.
    """
    if not watched or not universe:
        return
    watch = {s.upper() for s in watched}
    overlap = sum(1 for s in universe if s.upper() in watch)
    if len(universe) <= max(len(watch) * 2, 50) and overlap > len(universe) * 0.5:
        raise WatchlistError(
            f"refusing to measure a universe of {len(universe)} symbols of which "
            f"{overlap} are watchlisted. A hand-picked universe is selected by "
            "someone who already knows the outcomes; that is lookahead, and no "
            "guardrail here can detect it after the fact. The watchlist sets "
            "fetch priority and reporting, never the measured population."
        )
