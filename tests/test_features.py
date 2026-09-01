"""Tests for point-in-time feature enrichment and the backtest cost table.

The central claim under test is the truncation rule: an indicator may only see
bars that closed before the session we buy the open of. Everything else here is
bookkeeping around that one property, because getting it wrong produces a strong
and entirely fictitious edge on every momentum feature simultaneously -- the
failure mode that looks most like success.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.candidates import FEATURE_HYPOTHESES, all_candidates
from tradezbotz.research.costs import FALLBACK_COST_BPS, CostTable
from tradezbotz.research.features import (
    BOOLEAN_FEATURES,
    FEATURE_BARS,
    MIN_FEATURE_BARS,
    NUMERIC_FEATURES,
    FeatureBuilder,
    features_at,
    prior_bars,
)
from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.prices import Bar, Series


def bars(n: int, start: date = date(2024, 1, 1), price: float = 10.0,
         step: float = 0.0, volume: float = 1_000.0) -> list[Bar]:
    """Flat-by-default daily bars. `step` adds a per-session drift."""
    out = []
    for i in range(n):
        p = price + step * i
        out.append(Bar(day=start + timedelta(days=i), open=p, high=p * 1.01,
                       low=p * 0.99, close=p, volume=volume))
    return out


def label_for(symbol: str, entry: date) -> Label:
    return Label(symbol=symbol, observed_at=datetime(2024, 6, 1, tzinfo=UTC),
                 entry_day=entry, entry_price=10.0, returns={1: 0.01},
                 coverage=Coverage.COMPLETE)


class FakeCache:
    """Minimal stand-in for PriceCache, counting reads."""

    def __init__(self, series: dict[str, list[Bar]]) -> None:
        self.series = series
        self.reads = 0

    def get(self, symbol, start, end, basis=None):
        self.reads += 1
        rows = [b for b in self.series.get(symbol, []) if start <= b.day <= end]
        return Series(symbol=symbol, bars=tuple(rows), requested_start=start,
                      requested_end=end)


# --- the truncation rule ----------------------------------------------------

def test_prior_bars_excludes_the_entry_session():
    """The entry bar exists in the series and must not be visible.

    We trade its OPEN. Its close, high, low and volume are all in the future at
    the moment the order goes out.
    """
    b = bars(10)
    entry = b[7].day

    out = prior_bars(b, entry)

    assert len(out) == 7
    assert all(bar.day < entry for bar in out)
    assert out[-1].day == b[6].day


def test_prior_bars_caps_the_window():
    b = bars(FEATURE_BARS + 200)
    out = prior_bars(b, b[-1].day)
    assert len(out) == FEATURE_BARS


def test_a_breakout_that_happens_on_the_entry_bar_is_not_visible():
    """The regression this module exists to prevent.

    Price is flat for a year, then the entry session gaps to a new high. Reading
    the indicator on the entry bar reports a breakout -- and that breakout is
    the same session whose return we are about to measure. Reading it on the
    prior bar reports nothing, which is what was actually knowable.
    """
    b = bars(200, price=10.0)
    spike_day = b[-1].day + timedelta(days=1)
    b.append(Bar(day=spike_day, open=10.0, high=20.0, low=10.0, close=19.0,
                 volume=50_000.0))

    truncated = features_at(prior_bars(b, spike_day))
    including = features_at(b)

    assert truncated["donchian_breakout"] is False
    assert including["donchian_breakout"] is True


def test_features_are_omitted_rather_than_defaulted_on_short_history():
    """A payload full of False from missing history is indistinguishable from
    one where every condition genuinely failed, and the two mean opposite
    things."""
    assert features_at(bars(MIN_FEATURE_BARS - 1)) == {}


def test_every_declared_feature_is_actually_produced():
    out = features_at(bars(FEATURE_BARS, step=0.02))
    for key in BOOLEAN_FEATURES:
        assert key in out, f"{key} declared but never written"
        assert isinstance(out[key], bool)
    # Numerics may legitimately be absent when their window is short; on a full
    # window they must all be there.
    for key in NUMERIC_FEATURES:
        assert key in out, f"{key} declared but never written"


# --- the builder -------------------------------------------------------------

def test_enrich_does_not_mutate_the_stored_payload():
    """Payloads come out of the event store and represent what was filed. A
    feature is our computation about that filing, not part of it."""
    cache = FakeCache({"AAA": bars(200)})
    builder = FeatureBuilder(cache)
    original = {"transaction_code": "P"}

    out = builder.enrich([original], [label_for("AAA", bars(200)[-1].day)])

    assert original == {"transaction_code": "P"}
    assert out[0]["transaction_code"] == "P"
    assert out[0] is not original


def test_each_symbol_is_read_from_the_cache_once():
    cache = FakeCache({"AAA": bars(200)})
    builder = FeatureBuilder(cache)
    day = bars(200)[-1].day
    labels = [label_for("AAA", day), label_for("AAA", day),
              label_for("AAA", day - timedelta(days=1))]

    builder.enrich([{}, {}, {}], labels)

    assert cache.reads == 1


def test_missing_bars_are_counted_not_defaulted():
    cache = FakeCache({})
    builder = FeatureBuilder(cache)

    out = builder.enrich([{"x": 1}], [label_for("ZZZ", date(2024, 6, 1))])

    assert "has_features" not in out[0]
    assert builder.skipped_no_bars == 1


def test_a_label_with_no_entry_day_yields_no_features():
    cache = FakeCache({"AAA": bars(200)})
    builder = FeatureBuilder(cache)
    unlabelled = Label(symbol="AAA", observed_at=datetime(2024, 6, 1, tzinfo=UTC),
                       entry_day=None, entry_price=None, returns={},
                       coverage=Coverage.NO_ENTRY_BAR)

    assert builder.features(unlabelled) == {}


# --- the cost table ----------------------------------------------------------

def test_cost_table_never_sees_bars_from_the_entry_month():
    """Point-in-time, for the same reason as the features.

    Letting the trailing window include the entry month lets the price path of
    the trade being measured set that trade's cost, which biases in whichever
    direction the trade happened to go.
    """
    seen = {}

    class SpyCache(FakeCache):
        def get(self, symbol, start, end, basis=None):
            seen["end"] = end
            return super().get(symbol, start, end, basis=basis)

    cache = SpyCache({"AAA": bars(400, start=date(2023, 1, 1))})
    table = CostTable(cache)

    table(label_for("AAA", date(2023, 9, 14)))

    assert seen["end"] == date(2023, 9, 1)


def test_cost_table_falls_back_pessimistically_and_counts_it():
    """A missing estimate cannot be skipped -- every trade must carry a cost or
    the whole result silently reverts to uncosted. So it falls back to the
    universe median, not to the model floor."""
    table = CostTable(FakeCache({}))

    charged = table(label_for("ZZZ", date(2024, 6, 3)))

    assert charged == pytest.approx(FALLBACK_COST_BPS / 10_000)
    assert table.fell_back == 1
    assert table.estimated == 0


def test_cost_table_memoises_by_month():
    cache = FakeCache({"AAA": bars(400, start=date(2023, 1, 1))})
    table = CostTable(cache)

    first = table(label_for("AAA", date(2023, 9, 5)))
    same_month = table(label_for("AAA", date(2023, 9, 26)))

    assert first == same_month
    assert table.estimated == 1


def test_cost_table_charges_something_for_every_label():
    """The property `BacktestResult.costed` depends on: one cost per trade, no
    gaps. A gap flips the whole result to uncosted, where `survives_costs` is
    False and the cost gate reports nothing."""
    cache = FakeCache({"AAA": bars(400, start=date(2023, 1, 1))})
    table = CostTable(cache)
    labels = [label_for("AAA", date(2023, 9, 5)),
              label_for("MISSING", date(2023, 9, 5))]

    charges = [table(lab) for lab in labels]

    assert len(charges) == len(labels)
    assert all(c > 0 for c in charges)


# --- the registry ------------------------------------------------------------

def test_every_feature_hypothesis_names_a_feature_that_exists():
    """A selector on a misspelled key returns False forever, which reads as a
    measurement -- 'this indicator never fired' -- rather than as the typo it
    is. Nothing else in the pipeline would catch it."""
    for key, _, _ in FEATURE_HYPOTHESES:
        assert key in BOOLEAN_FEATURES, f"{key} is not a feature this code emits"


def test_every_blocked_candidate_names_a_mechanical_reason():
    for cand in all_candidates():
        if not cand.runnable:
            assert cand.blocked_by.strip(), f"{cand.name} blocked with no reason"


def test_every_candidate_records_a_prior():
    """A prior recorded before the run is the only thing that lets an opinion be
    contradicted rather than quietly act."""
    for cand in all_candidates():
        assert cand.prior.strip(), f"{cand.name} has no recorded prior"


def test_dropping_features_drops_the_feature_candidates():
    """Not the same as leaving them in to report zero trades: a zero-trade row
    reads as 'measured, nothing there' and this was never measured."""
    names = {c.name for c in all_candidates(with_features=False)}
    assert "near_high" not in names
    assert "open-market buy" in names
