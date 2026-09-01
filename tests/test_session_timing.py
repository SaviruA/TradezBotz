"""Tests for the session-sequence fields and the intraday liquidity sweep.

What is being protected: the stored reduction is otherwise entirely order-free
-- min, max, sum, a histogram -- so an ordering bug in it produces a perfectly
plausible profile and nothing fails. These fields are the first ones that carry
sequence, and they are unrecoverable after the fact because raw minute bars are
never kept.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.intraday import (
    PROFILE_TIMING_COLUMNS,
    MinuteBar,
    ProfileStore,
    SessionProfile,
)
from tradezbotz.research.microstructure import (
    MIN_SHARE_AFTER_SWEEP,
    TimingUnavailable,
    build_profile,
    require_timing,
    swept_high_intraday,
    swept_low_intraday,
)

DAY = date(2025, 3, 4)
OPEN = datetime(2025, 3, 4, 14, 30, tzinfo=UTC)   # 09:30 ET


def minutes(spec):
    """Build a session from (offset, low, high, close, volume) tuples."""
    return [
        MinuteBar(ts=OPEN + timedelta(minutes=off), open=close, high=high,
                  low=low, close=close, volume=vol)
        for off, low, high, close, vol in spec
    ]


def sweep_session(low_at: int, total_minutes: int = 100):
    """A session that dips to 9.0 at `low_at` and trades at 10.0 otherwise."""
    spec = []
    for i in range(total_minutes):
        if i == low_at:
            spec.append((i, 9.0, 10.0, 9.5, 500.0))
        else:
            spec.append((i, 9.9, 10.1, 10.0, 100.0))
    return minutes(spec)


def stale_profile() -> SessionProfile:
    """A session as it would have been reduced before the timing fields."""
    return SessionProfile(
        symbol="OLD", day=DAY, low=9.0, high=11.0, volume=1000.0, vwap=10.0,
        histogram=(0.0,) * 40, delta=0.0, unsigned_volume=0.0, minute_count=100,
    )


# --- the reduction ----------------------------------------------------------

def test_build_profile_records_when_the_extreme_printed():
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    assert profile.low_minute == 10
    assert profile.has_timing


def test_build_profile_records_the_volume_left_after_the_low():
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    # 89 minutes of 100 shares each follow the low bar.
    assert profile.volume_after_low == pytest.approx(8_900.0)
    assert profile.share_after_low == pytest.approx(8_900.0 / profile.volume)


def test_build_profile_records_the_session_open_and_close():
    """Not derivable from low/high, and without them the close's position in
    the session range is unknown."""
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    assert profile.session_open == pytest.approx(10.0)
    assert profile.session_close == pytest.approx(10.0)


def test_out_of_order_bars_do_not_corrupt_the_timing():
    """Everything the reduction computed before these fields was order-free, so
    an unsorted input was harmless and would never have been noticed. It is not
    harmless now."""
    session = sweep_session(low_at=10)
    shuffled = list(reversed(session))

    ordered = build_profile("AAA", DAY, session)
    scrambled = build_profile("AAA", DAY, shuffled)

    assert scrambled.low_minute == ordered.low_minute == 10
    assert scrambled.volume_after_low == ordered.volume_after_low


def test_the_earliest_bar_wins_a_tie_on_the_extreme():
    """A level touched twice was first reached at the first touch; dating it to
    the second would understate how much session was left to reclaim in."""
    session = sweep_session(low_at=10)
    session[60] = MinuteBar(ts=OPEN + timedelta(minutes=60), open=9.5, high=10.0,
                            low=9.0, close=9.5, volume=500.0)

    assert build_profile("AAA", DAY, session).low_minute == 10


# --- storage and migration --------------------------------------------------

def test_timing_survives_a_store_round_trip(tmp_path):
    store = ProfileStore(tmp_path / "p.db")
    store.put(build_profile("AAA", DAY, sweep_session(low_at=10)))

    back = store.get("AAA", DAY)
    store.close()

    assert back.low_minute == 10
    assert back.session_close == pytest.approx(10.0)
    assert back.has_timing


OLD_SCHEMA = """
CREATE TABLE session_profiles (
    symbol TEXT NOT NULL, day TEXT NOT NULL, low REAL NOT NULL,
    high REAL NOT NULL, volume REAL NOT NULL, vwap REAL NOT NULL,
    histogram BLOB NOT NULL, delta REAL NOT NULL,
    unsigned_volume REAL NOT NULL, minute_count INTEGER NOT NULL,
    rth_only INTEGER NOT NULL,
    flow_method TEXT NOT NULL DEFAULT 'tick_minute',
    PRIMARY KEY (symbol, day));
CREATE TABLE profile_fetches (
    symbol TEXT NOT NULL, day TEXT NOT NULL, fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, day));
"""


def test_a_store_written_before_the_columns_existed_is_migrated(tmp_path):
    """CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so without an
    explicit migration an older database keeps the old shape and every insert
    naming the new columns fails."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO session_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("OLD", DAY.isoformat(), 9.0, 11.0, 1000.0, 10.0, bytes(160),
         5.0, 0.0, 100, 1, "tick_minute"))
    conn.commit()
    conn.close()

    store = ProfileStore(path)
    held = store.get("OLD", DAY)
    # A new session still writes, alongside the migrated one.
    store.put(build_profile("AAA", DAY, sweep_session(low_at=10)))
    untimed = store.count_untimed()
    store.close()

    assert held is not None, "the pre-existing row survived the migration"
    assert held.has_timing is False
    assert untimed == 1


def test_an_untimed_session_reads_as_absent_not_as_minute_zero():
    """The failure this guards: a None handled as zero puts every old session's
    low at the opening bell, which is a specific and very wrong claim."""
    stale = stale_profile()

    assert stale.has_timing is False
    assert stale.low_minute is None
    assert stale.share_after_low is None


def test_every_declared_timing_column_reaches_the_table(tmp_path):
    store = ProfileStore(tmp_path / "p.db")
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(session_profiles)")}
    store.close()

    for name, _ in PROFILE_TIMING_COLUMNS:
        assert name in cols


# --- the sweep detector -----------------------------------------------------

def test_an_early_sweep_that_reclaims_fires():
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    assert swept_low_intraday(profile, level=9.5) is True


def test_a_reclaim_on_the_closing_print_does_not_fire():
    """The whole reason these fields exist. Both sessions pierce the level and
    close above it, so the daily bar reports them identically -- one is a stop
    run, the other is a breakdown that ticked up at the bell."""
    late = build_profile("AAA", DAY, sweep_session(low_at=97))

    assert late.low_minute == 97
    assert late.share_after_low < MIN_SHARE_AFTER_SWEEP
    assert swept_low_intraday(late, level=9.5) is False


def test_a_session_that_never_reached_the_level_does_not_fire():
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    assert swept_low_intraday(profile, level=8.0) is False


def test_a_session_that_closed_below_the_level_does_not_fire():
    """Pierced and stayed pierced is a breakdown, not a sweep."""
    session = sweep_session(low_at=10)
    session[-1] = MinuteBar(ts=OPEN + timedelta(minutes=99), open=9.2, high=9.3,
                            low=9.1, close=9.2, volume=100.0)

    assert swept_low_intraday(build_profile("AAA", DAY, session),
                              level=9.5) is False


def test_the_bearish_sweep_is_the_mirror():
    spec = [(i, 9.9, 10.1, 10.0, 100.0) for i in range(100)]
    spec[10] = (10, 10.0, 11.0, 10.5, 500.0)
    profile = build_profile("AAA", DAY, minutes(spec))

    assert swept_high_intraday(profile, level=10.5) is True
    assert swept_low_intraday(profile, level=10.5) is False


def test_a_session_without_timing_is_refused_rather_than_skipped():
    """Silently skipping would restrict a study to whatever happened to be
    reduced after the change, making the sample definition a fact about
    deployment history rather than about the market."""
    stale = stale_profile()

    with pytest.raises(TimingUnavailable) as exc:
        require_timing([stale])

    assert "refetched" in str(exc.value)
    assert "--refresh-untimed" in str(exc.value)

    with pytest.raises(TimingUnavailable):
        swept_low_intraday(stale, level=9.5)


# --- batched writes ---------------------------------------------------------

def test_put_many_stores_every_profile(tmp_path):
    store = ProfileStore(tmp_path / "p.db")
    profiles = [build_profile(f"S{i}", DAY, sweep_session(low_at=10))
                for i in range(25)]

    n = store.put_many(profiles)

    assert n == 25
    assert store.count() == 25
    assert store.get("S7", DAY).low_minute == 10
    store.close()


def test_put_many_also_marks_them_fetched(tmp_path):
    """Without this the backfill re-requests every stored session forever --
    `was_fetched` is what makes the step resumable."""
    store = ProfileStore(tmp_path / "p.db")

    store.put_many([build_profile("AAA", DAY, sweep_session(low_at=10))])

    assert store.was_fetched("AAA", DAY) is True
    store.close()


def test_put_many_on_an_empty_batch_is_a_no_op(tmp_path):
    store = ProfileStore(tmp_path / "p.db")
    assert store.put_many([]) == 0
    store.close()


def test_mark_many_fetched_records_sessions_that_had_no_prints(tmp_path):
    """A session with no prints is a real outcome, not a gap. Unmarked, it is
    re-requested on every run for ever."""
    store = ProfileStore(tmp_path / "p.db")

    store.mark_many_fetched([("AAA", DAY), ("BBB", DAY)])

    assert store.was_fetched("AAA", DAY) is True
    assert store.was_fetched("BBB", DAY) is True
    assert store.count() == 0, "marked as attempted, not stored as a profile"
    store.close()


def test_batched_and_per_session_writes_agree(tmp_path):
    """The batched path duplicates put()'s column list, so it can drift from it.
    A mismatch would write values into the wrong columns rather than fail."""
    one = ProfileStore(tmp_path / "one.db")
    many = ProfileStore(tmp_path / "many.db")
    profile = build_profile("AAA", DAY, sweep_session(low_at=10))

    one.put(profile)
    many.put_many([profile])

    assert one.get("AAA", DAY) == many.get("AAA", DAY)
    one.close()
    many.close()
