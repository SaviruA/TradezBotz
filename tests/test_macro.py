"""Tests for the geopolitical regime conditioner.

The load-bearing test is `test_the_regime_percentile_never_uses_the_future`.
Ranking a day against the full-sample distribution would use four decades of
subsequent history to decide what counted as "high risk" at the time, and every
regime label would carry a little of the future. It would also look completely
reasonable in the output, which is why it needs a test rather than a review.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.joins import MacroJoin
from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.macro import (
    HIGH_RISK_PERCENTILE,
    MIN_REGIME_HISTORY,
    GprDay,
    MacroStore,
)


def store_with(values, tmp_path, start=date(2015, 1, 1)):
    s = MacroStore(tmp_path / "m.db")
    s.put_many([GprDay(day=start + timedelta(days=i), gprd=float(v))
                for i, v in enumerate(values)])
    return s


def label_on(day: date) -> Label:
    return Label(symbol="AAA", observed_at=datetime(2020, 1, 1, tzinfo=UTC),
                 entry_day=day, entry_price=10.0, returns={5: 0.01},
                 coverage=Coverage.COMPLETE)


# --- the lookahead rule -----------------------------------------------------

def test_the_regime_percentile_never_uses_the_future(tmp_path):
    """A quiet stretch followed by an extreme one. Judged on its own trailing
    history the quiet day is unremarkable; judged against the whole series --
    including the crisis that had not happened yet -- it looks like a calm
    outlier. Only the first reading was knowable at the time."""
    quiet = [100.0] * 400 + [101.0]
    later_crisis = [900.0] * 400
    s = store_with(quiet + later_crisis, tmp_path)

    asked = date(2015, 1, 1) + timedelta(days=401)
    regime = s.regime_at(asked)
    s.close()

    # 101 against 400 days of 100 is the top of its own trailing distribution.
    assert regime["gpr_high"] is True
    # Against the full series, including the crisis, it would be near the bottom.
    assert regime["gpr_percentile"] >= HIGH_RISK_PERCENTILE


def test_the_reading_used_is_strictly_before_the_entry_day(tmp_path):
    """The day's own index counts that day's newspapers, so it is not available
    while the session is being traded."""
    s = store_with([100.0] * 400 + [999.0], tmp_path)
    entry = date(2015, 1, 1) + timedelta(days=400)

    regime = s.regime_at(entry)
    s.close()

    assert regime["gpr"] == 100.0, "the 999 printed on the entry day itself"


def test_value_at_is_also_strictly_before(tmp_path):
    s = store_with([1.0, 2.0, 3.0], tmp_path)

    assert s.value_at(date(2015, 1, 3)) == 2.0
    assert s.value_at(date(2015, 1, 1)) is None
    s.close()


def test_too_little_trailing_history_yields_nothing(tmp_path):
    """Ranking against a handful of days is not a percentile."""
    s = store_with([100.0] * (MIN_REGIME_HISTORY - 10), tmp_path)

    assert s.regime_at(date(2015, 1, 1) + timedelta(days=200)) is None
    s.close()


def test_the_trailing_window_is_bounded(tmp_path):
    """Old history drops out, so 'high risk' means high relative to the recent
    world rather than to the Cold War."""
    s = store_with([500.0] * 1000 + [100.0] * 1000, tmp_path)
    late = date(2015, 1, 1) + timedelta(days=1999)

    regime = s.regime_at(late, lookback_days=365)
    s.close()

    assert regime["gpr_observations"] <= 366
    assert regime["gpr_high"] is False, "judged against recent calm, not old crises"


# --- bands ------------------------------------------------------------------

def test_high_and_low_bands_are_mutually_exclusive(tmp_path):
    s = store_with([float(i % 200) for i in range(1200)], tmp_path)

    for offset in (400, 700, 1000, 1150):
        r = s.regime_at(date(2015, 1, 1) + timedelta(days=offset))
        if r:
            assert not (r["gpr_high"] and r["gpr_low"])
    s.close()


def test_an_extreme_reading_lands_in_the_high_band(tmp_path):
    s = store_with([50.0] * 400 + [900.0], tmp_path)

    r = s.regime_at(date(2015, 1, 1) + timedelta(days=401))
    s.close()

    assert r["gpr_high"] is True and r["gpr_low"] is False


# --- the join ---------------------------------------------------------------

def test_the_join_writes_regime_fields_into_the_payload(tmp_path):
    s = store_with([float(50 + i % 100) for i in range(800)], tmp_path)
    join = MacroJoin(s)

    out = join.features(label_on(date(2015, 1, 1) + timedelta(days=700)))
    s.close()

    assert out["has_macro"] is True
    assert "gpr" in out and "gpr_percentile" in out
    assert isinstance(out["gpr_high"], bool)
    assert join.enriched == 1


def test_the_join_counts_events_that_predate_the_history(tmp_path):
    s = store_with([100.0] * 50, tmp_path)
    join = MacroJoin(s)

    out = join.features(label_on(date(2015, 1, 20)))
    s.close()

    assert out == {}
    assert join.skipped_no_history == 1


def test_a_label_without_an_entry_day_is_skipped(tmp_path):
    s = store_with([100.0] * 400, tmp_path)
    join = MacroJoin(s)
    unlabelled = Label(symbol="AAA", observed_at=datetime(2020, 1, 1, tzinfo=UTC),
                       entry_day=None, entry_price=None, returns={},
                       coverage=Coverage.NO_DATA)

    assert join.features(unlabelled) == {}
    s.close()


def test_the_join_needs_no_per_symbol_coverage(tmp_path):
    """The property that makes this usable where news sentiment is not: one
    world-level series conditions every symbol, so a universe journalists ignore
    is no obstacle."""
    s = store_with([float(50 + i % 100) for i in range(800)], tmp_path)
    join = MacroJoin(s)
    day = date(2015, 1, 1) + timedelta(days=700)

    a = join.features(label_on(day))
    b = join.features(Label(
        symbol="ZZZZ-NEVER-COVERED", observed_at=datetime(2020, 1, 1, tzinfo=UTC),
        entry_day=day, entry_price=1.0, returns={5: 0.0},
        coverage=Coverage.COMPLETE))
    s.close()

    assert a["gpr"] == b["gpr"], "the regime does not depend on the symbol"


# --- the workbook parser ----------------------------------------------------

def test_the_parser_rejects_an_unexpected_workbook_shape(monkeypatch):
    """A silent change in the published file should fail here with a clear
    message, not produce an empty series that reads as 'no geopolitical risk'.

    The frame is injected rather than written to a real workbook: building one
    would pull in an Excel *writer* as a test-only dependency, and what is being
    tested is the column check, not pandas.
    """
    import pandas as pd

    from tradezbotz.research import macro
    from tradezbotz.research.macro import MacroError, parse_gpr_workbook

    monkeypatch.setattr(
        pd, "read_excel", lambda *a, **k: pd.DataFrame({"something_else": [1, 2]}))

    with pytest.raises(MacroError, match="shape changed"):
        parse_gpr_workbook(b"irrelevant")


def test_the_parser_explains_the_missing_xls_reader():
    """The published file is a legacy .xls. Without xlrd pandas raises something
    opaque, and the fix is a one-line install worth naming."""
    from tradezbotz.research.macro import MacroError, parse_gpr_workbook

    with pytest.raises(MacroError, match="xlrd"):
        parse_gpr_workbook(b"not a workbook at all")


def test_the_store_records_when_it_was_fetched(tmp_path):
    """GPR is recomputed when its methodology moves, so knowing when a value was
    pulled is the only way to notice the past changed underneath a result."""
    s = store_with([100.0] * 10, tmp_path)

    assert s.fetched_at() is not None
    s.close()
