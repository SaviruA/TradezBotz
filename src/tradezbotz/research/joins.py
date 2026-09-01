"""Join the other data families onto labelled events, point-in-time.

`features.FeatureBuilder` did this for daily-bar indicators and unblocked
fourteen hypotheses at once. Everything else stayed blocked for the same reason
and needed the same fix: a `Selector` sees `(payload, label)` and nothing else,
so any data family that cannot reach the payload cannot be tested, however
complete its own module is.

Three families, three joins here:

  ProfileJoin       intraday volume profile and the intraday liquidity sweep,
                    from ProfileStore
  HoldingsJoin      congressional purchases and 13F/13D stakes, from the event
                    store they already live in
  FundamentalsJoin  XBRL valuation multiples, from a local facts cache

**Every one of them is a lookahead risk, and they fail in the same direction.**
Each asks "what else was true about this symbol around this date", and the
tempting implementation -- take the nearest record -- silently reads records
filed *after* the entry. A 13F describing 30 June is not knowable until it is
filed 45 days later; a congressional trade dated 12 March may be disclosed in
May; an XBRL fact for Q2 appears in the Q2 10-Q, not on 30 June. So each join
filters on the *disclosure* timestamp and never on the event date, and each one
has a test that fails if that is reversed.

The asymmetry matters: a join that is too strict loses coverage, which shows up
honestly as fewer trades. A join that is too loose invents an edge and nothing
in the output says so.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Sequence

from .labeler import Label

#: How far back a related disclosure still counts as "recent" for the event
#: being labelled. 90 days spans a quarter, so a congressional purchase or a
#: stake filing lands inside the window of the next insider transaction on the
#: same name rather than falling between the cracks.
RELATED_WINDOW_DAYS = 90

#: Sessions of stored intraday history required before the profile features
#: are computed. Below this the merged profile's point of control sits on too
#: little volume to mean anything.
MIN_PROFILE_SESSIONS = 5

#: Prior daily sessions defining the level an intraday sweep runs. Matches
#: `indicators.SWEEP_PERIOD` deliberately, so the daily and intraday versions
#: are testing the same level and their results are comparable.
SWEEP_LEVEL_PERIOD = 20


def _as_datetime(day: date) -> datetime:
    """Midnight UTC on `day`, for comparison against `observed_at`.

    Deliberately the START of the entry day. An event disclosed during the
    entry session is not knowable when we buy that session's open, so the
    cutoff excludes the entry day itself rather than including it.
    """
    return datetime.combine(day, dtime(0, 0), tzinfo=timezone.utc)


class ProfileJoin:
    """Intraday volume profile and liquidity sweep, per labelled event.

    Reads sessions strictly before the entry day. The entry session has not
    happened when the order goes out, and its profile would encode the very
    move being measured.
    """

    #: Whether `features` takes the payload as well as the label. Declared so
    #: `enrich_all` can dispatch without probing for exceptions.
    needs_payload = False

    def __init__(self, store, price_cache=None, basis: str | None = None) -> None:
        self.store = store
        self.price_cache = price_cache
        self.basis = basis
        self.enriched = 0
        self.skipped_no_sessions = 0

    def features(self, label: Label) -> dict:
        if not label.symbol or label.entry_day is None:
            return {}
        from .microstructure import (
            above_poc,
            below_value_area,
            delta_ratio,
            in_low_volume_node,
            swept_low_intraday,
        )

        # `range` is inclusive of both ends, so step back one day to exclude
        # the entry session itself.
        last = label.entry_day - timedelta(days=1)
        sessions = self.store.range(
            label.symbol, last - timedelta(days=RELATED_WINDOW_DAYS), last)
        sessions = [s for s in sessions if s.day < label.entry_day]
        if len(sessions) < MIN_PROFILE_SESSIONS:
            self.skipped_no_sessions += 1
            return {}

        # The reference price is the last known close, not the entry price:
        # entry_price is the entry session's open, which is not knowable when
        # the decision is made.
        recent = sessions[-1]
        price = recent.session_close
        if price is None:
            price = recent.vwap

        out: dict = {"has_profile": True, "profile_sessions": len(sessions)}
        try:
            out["above_poc"] = above_poc(sessions, price)
            out["below_value_area"] = below_value_area(sessions, price)
            out["in_low_volume_node"] = in_low_volume_node(sessions, price)
        except Exception:  # noqa: BLE001 - a degenerate profile is not a bug
            return {}

        ratio = delta_ratio(sessions)
        if ratio is not None:
            out["delta_ratio"] = ratio
            out["positive_delta"] = ratio >= 0.10

        level = self._sweep_level(label)
        if level is not None and recent.has_timing:
            out["swept_low_intraday"] = swept_low_intraday(recent, level)

        self.enriched += 1
        return out

    def _sweep_level(self, label: Label) -> float | None:
        """The prior 20-session low, from daily bars.

        The intraday sweep needs a level to have been swept, and that level is
        a daily-bar quantity. Excludes the entry day for the same reason
        everything else here does.
        """
        if self.price_cache is None:
            return None
        kwargs = {} if self.basis is None else {"basis": self.basis}
        series = self.price_cache.get(
            label.symbol,
            label.entry_day - timedelta(days=SWEEP_LEVEL_PERIOD * 3),
            label.entry_day - timedelta(days=1),
            **kwargs,
        )
        bars = [b for b in series.bars if b.day < label.entry_day]
        if len(bars) < SWEEP_LEVEL_PERIOD:
            return None
        return min(b.low for b in bars[-SWEEP_LEVEL_PERIOD:])

    def summary(self) -> str:
        return (f"profiles: {self.enriched:,} events enriched, "
                f"{self.skipped_no_sessions:,} had under "
                f"{MIN_PROFILE_SESSIONS} prior sessions stored")


class HoldingsJoin:
    """Congressional purchases and 13F/13D stakes near a labelled event.

    **The disclosure lag is the whole difficulty.** A House PTR carries a
    transaction date and a filing date up to 45 days apart; a 13F describes a
    quarter-end and is filed 45 days after it; a 13D has 5 business days. The
    only date a backtest may use is the filing one, which is what
    `EventStore.as_of` already enforces -- so this join goes through `as_of`
    rather than querying the table directly, and inherits the guarantee instead
    of reimplementing it.
    """

    needs_payload = False

    def __init__(self, store, window_days: int = RELATED_WINDOW_DAYS) -> None:
        self.store = store
        self.window_days = window_days
        self._by_symbol: dict[str, list[dict]] | None = None
        self.enriched = 0
        self.congress_hits = 0
        self.stake_hits = 0

    def _index(self) -> dict[str, list[dict]]:
        """Load every congress/holding/stake disclosure once, keyed by symbol.

        One pass rather than a query per event: the alternative is 180,000
        SQLite round trips against a table these three kinds barely populate.
        """
        if self._by_symbol is not None:
            return self._by_symbol
        from .holdings import KIND_CONGRESS, KIND_HOLDING, KIND_STAKE

        index: dict[str, list[dict]] = {}
        far_future = datetime.now(timezone.utc) + timedelta(days=365 * 20)
        for kind in (KIND_CONGRESS, KIND_STAKE, KIND_HOLDING):
            for row in self.store.as_of(far_future, kind=kind):
                symbol = (row.get("symbol") or "").upper()
                if not symbol:
                    continue
                index.setdefault(symbol, []).append({
                    "kind": kind,
                    "observed_at": datetime.fromisoformat(row["observed_at"]),
                    "payload": row["payload"],
                })
        for rows in index.values():
            rows.sort(key=lambda r: r["observed_at"])
        self._by_symbol = index
        return index

    def features(self, label: Label) -> dict:
        if not label.symbol or label.entry_day is None:
            return {}
        rows = self._index().get(label.symbol.upper())
        if not rows:
            return {}

        cutoff = _as_datetime(label.entry_day)
        floor = cutoff - timedelta(days=self.window_days)
        # observed_at, never occurred_at. A trade dated within the window but
        # disclosed after the entry was not knowable, and including it is the
        # exact lookahead this module exists to avoid.
        recent = [r for r in rows if floor <= r["observed_at"] < cutoff]
        if not recent:
            return {}

        from .holdings import KIND_CONGRESS, KIND_HOLDING, KIND_STAKE

        congress_buys = [
            r for r in recent
            if r["kind"] == KIND_CONGRESS and r["payload"].get("is_purchase")
        ]
        stakes = [r for r in recent if r["kind"] == KIND_STAKE]
        holdings = [r for r in recent if r["kind"] == KIND_HOLDING]

        out = {
            "has_holdings": True,
            "congress_bought": bool(congress_buys),
            "congress_buy_count": len(congress_buys),
            "stake_filed": bool(stakes),
            "activist_stake": any(
                r["payload"].get("activist") for r in stakes),
            "institution_added": bool(holdings),
        }
        if congress_buys:
            self.congress_hits += 1
            out["congress_amount_high"] = max(
                float(r["payload"].get("amount_high") or 0.0)
                for r in congress_buys)
        if stakes:
            self.stake_hits += 1
        self.enriched += 1
        return out

    def summary(self) -> str:
        n = len(self._index()) if self._by_symbol is not None else 0
        return (f"holdings: {n:,} symbols carry a disclosure; "
                f"{self.enriched:,} events matched one inside "
                f"{self.window_days} days ({self.congress_hits:,} congress "
                f"purchases, {self.stake_hits:,} stakes)")


class FundamentalsJoin:
    """Valuation multiples at the entry date, from cached XBRL facts.

    Point-in-time comes free here and is worth stating anyway:
    `fundamentals.visible` filters every fact on its `filed` date, so a
    restatement of a 2019 quarter published in 2021 cannot reach a 2019
    decision. That is the entire reason this reads the SEC rather than a
    display site.

    Facts are read from a local cache. Fetching per event is not an option --
    the SEC allows 8 requests a second and the event store holds six figures of
    events -- so `ingest-fundamentals` populates the cache and this join is
    offline, like every other part of `measure`.
    """

    #: True: the issuer CIK lives in the Form 4 payload, not on the label.
    needs_payload = True

    def __init__(self, cache, price_cache=None, basis: str | None = None) -> None:
        self.cache = cache
        self.price_cache = price_cache
        self.basis = basis
        self._memo: dict[tuple[str, date], dict] = {}
        self.enriched = 0
        self.skipped_no_facts = 0
        self.skipped_no_price = 0

    def features(self, payload: dict, label: Label) -> dict:
        cik = str(payload.get("issuer_cik") or "").lstrip("0")
        if not cik or label.entry_day is None:
            return {}
        key = (cik, label.entry_day)
        hit = self._memo.get(key)
        if hit is not None:
            return hit

        from .fundamentals import size_band, snapshot

        raw = self.cache.get(cik)
        if not raw:
            self.skipped_no_facts += 1
            self._memo[key] = {}
            return {}

        # `as_of` is the day BEFORE entry: a filing published during the entry
        # session is not knowable when we buy that session's open.
        as_of = label.entry_day - timedelta(days=1)
        snap = snapshot(None, cik, as_of, facts=raw)
        price = self._price(label)
        if price is None:
            self.skipped_no_price += 1
            self._memo[key] = {}
            return {}

        out: dict = {"has_fundamentals": True}
        cap = snap.market_cap(price)
        if cap is not None:
            out["market_cap"] = cap
            band = size_band(cap)
            if band:
                out["size_band"] = band
        for name, value in (
            ("price_to_sales", snap.price_to_sales(price)),
            ("price_to_earnings", snap.price_to_earnings(price)),
            ("price_to_free_cash_flow", snap.price_to_free_cash_flow(price)),
            ("ev_to_ebitda", snap.ev_to_ebitda(price)),
            ("value_growth_score", snap.value_growth_score(price)),
            ("gross_margin", snap.gross_margin),
            ("revenue_growth", snap.revenue_growth),
        ):
            if value is not None:
                out[name] = value
        if snap.profitable is not None:
            out["profitable"] = snap.profitable

        self.enriched += 1
        self._memo[key] = out
        return out

    def _price(self, label: Label) -> float | None:
        """Last close before the entry session.

        Not `label.entry_price`: that is the entry session's open, which the
        decision cannot see. Using it would put a few hours of lookahead into
        every multiple.
        """
        if self.price_cache is None:
            return None
        kwargs = {} if self.basis is None else {"basis": self.basis}
        series = self.price_cache.get(
            label.symbol, label.entry_day - timedelta(days=30),
            label.entry_day - timedelta(days=1), **kwargs)
        bars = [b for b in series.bars if b.day < label.entry_day]
        return bars[-1].close if bars else None

    def summary(self) -> str:
        return (f"fundamentals: {self.enriched:,} (cik, day) pairs computed; "
                f"{self.skipped_no_facts:,} had no cached facts, "
                f"{self.skipped_no_price:,} had no prior close")


def enrich_all(payloads: Sequence[dict], labels: Sequence[Label],
               *joins) -> list[dict]:
    """Apply every supplied join, returning new payloads.

    New dicts rather than mutation, for the same reason `FeatureBuilder` does
    it: the payload is what was filed, and a derived column that looks like
    source data is a problem three months later.
    """
    out = []
    for payload, label in zip(payloads, labels):
        merged = dict(payload)
        for join in joins:
            if join is None:
                continue
            # Declared, not probed. Dispatching on whether `features(payload,
            # label)` raises TypeError would swallow a genuine TypeError raised
            # *inside* the join and silently retry it with the wrong arguments,
            # turning a bug into missing data.
            if join.needs_payload:
                extra = join.features(payload, label)
            else:
                extra = join.features(label)
            merged.update(extra)
        out.append(merged)
    return out


class MacroJoin:
    """Geopolitical regime at the entry date.

    The one join that needs no per-symbol coverage, which is exactly why it is
    usable where news sentiment is not: a single world-level series conditions
    every event, so a universe of microcaps that journalists ignore is no
    obstacle.

    Strictly a conditioner. One number per day cannot distinguish two symbols,
    so these fields only ever answer "does this signal behave differently when
    the world looks dangerous" -- which is a real question and the shape
    `all_of` was built for.
    """

    needs_payload = False

    def __init__(self, store, lookback_days: int | None = None) -> None:
        self.store = store
        self.lookback_days = lookback_days
        self.enriched = 0
        self.skipped_no_history = 0

    def features(self, label: Label) -> dict:
        if label.entry_day is None:
            return {}
        from .macro import REGIME_LOOKBACK_DAYS

        regime = self.store.regime_at(
            label.entry_day,
            lookback_days=self.lookback_days or REGIME_LOOKBACK_DAYS)
        if regime is None:
            self.skipped_no_history += 1
            return {}
        self.enriched += 1
        return {"has_macro": True, **regime}

    def summary(self) -> str:
        span = self.store.span()
        where = f"{span[0]} to {span[1]}" if span else "empty"
        return (f"macro: {self.enriched:,} events carry a geopolitical regime, "
                f"{self.skipped_no_history:,} predate enough trailing history "
                f"(series covers {where})")
