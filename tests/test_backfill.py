"""Tests for the resumable backfill runner.

The resume and failure-isolation tests matter most: this job runs for hours
unattended, and the failure modes that actually happen are interruption and one
bad ticker, not logic errors.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.backfill import (
    DONE,
    FAILED,
    BackfillRunner,
    symbols_from_events,
)
from tradezbotz.research.prices import Bar, PriceError, Series

START, END = date(2024, 9, 1), date(2026, 8, 1)


class FakeSource:
    """Returns bars for anything except symbols listed in `broken`."""

    def __init__(self, broken: set[str] | None = None, n_bars: int = 3):
        self.broken = broken or set()
        self.n_bars = n_bars
        self.calls: list[str] = []

    def daily_bars(self, symbol, start, end):
        self.calls.append(symbol)
        if symbol in self.broken:
            raise PriceError(f"boom for {symbol}")
        bars = tuple(
            Bar(date(2025, 3, 3 + i), 100.0, 101.0, 99.0, 100.5, 1e6)
            for i in range(self.n_bars)
        )
        return Series(symbol, bars, start, end, is_active=True)


@pytest.fixture
def runner_factory(tmp_path):
    made: list[BackfillRunner] = []

    def make(source, **kw):
        r = BackfillRunner(source, tmp_path / "ckpt.db", start=START, end=END, **kw)
        made.append(r)
        return r

    yield make
    for r in made:
        r.close()


def test_enqueue_is_idempotent(runner_factory):
    r = runner_factory(FakeSource())

    assert r.enqueue(["AAPL", "MSFT"]) == 2
    assert r.enqueue(["AAPL", "MSFT"]) == 0, "re-enqueueing must not duplicate"
    assert r.progress().total == 2


def test_enqueue_normalises_and_drops_blanks(runner_factory):
    r = runner_factory(FakeSource())

    r.enqueue([" aapl ", "", None, "msft"])

    assert sorted(r.pending()) == ["AAPL", "MSFT"]


def test_run_marks_symbols_done(runner_factory):
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAPL", "MSFT"])

    prog = r.run()

    assert prog.done == 2 and prog.failed == 0
    assert r.pending() == [], "nothing left to do"


def test_completed_symbols_are_not_refetched(runner_factory):
    """The whole point of checkpointing: a rerun must not repeat hours of work."""
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAPL", "MSFT"])
    r.run()

    r.run()

    assert sorted(src.calls) == ["AAPL", "MSFT"], "second run refetched"


def test_interrupted_run_resumes_where_it_stopped(runner_factory):
    """Simulates Ctrl-C / systemctl stop / VM reboot mid-backfill."""
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAA", "BBB", "CCC", "DDD"])

    # Stop after the first symbol completes.
    def stop_after_one(symbol, prog):
        r.request_stop()

    r.run(on_progress=stop_after_one)
    after_first = list(src.calls)

    assert len(after_first) == 1
    assert len(r.pending()) == 3

    # A fresh runner over the same checkpoint file picks up the remainder.
    src2 = FakeSource()
    r2 = runner_factory(src2)
    r2.run()

    assert sorted(src2.calls) == ["BBB", "CCC", "DDD"]
    assert r2.pending() == []


def test_one_bad_ticker_does_not_end_the_run(runner_factory):
    src = FakeSource(broken={"BAD"})
    r = runner_factory(src)
    r.enqueue(["AAA", "BAD", "ZZZ"])

    prog = r.run()

    assert prog.done == 2
    assert prog.failed == 1
    assert sorted(src.calls) == ["AAA", "BAD", "ZZZ"], "run continued past the failure"


def test_failures_are_retried_then_parked(runner_factory):
    src = FakeSource(broken={"BAD"})
    r = runner_factory(src, max_attempts=3)
    r.enqueue(["BAD"])

    for _ in range(5):
        r.run()

    rows = r.failures()
    assert len(rows) == 1
    assert rows[0]["attempts"] == 3, "stops retrying at max_attempts"
    assert "boom" in rows[0]["last_error"]
    assert r.pending() == [], "parked, not retried forever"


def test_failures_are_visible_not_silent(runner_factory):
    src = FakeSource(broken={"BAD"})
    r = runner_factory(src)
    r.enqueue(["GOOD", "BAD"])
    r.run()

    assert [row["symbol"] for row in r.failures()] == ["BAD"]


def test_limit_caps_work_per_invocation(runner_factory):
    """Lets a systemd timer take a bite at a time instead of one huge run."""
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAA", "BBB", "CCC", "DDD"])

    r.run(limit=2)

    assert len(src.calls) == 2
    assert len(r.pending()) == 2


def test_progress_reports_remaining_and_eta(runner_factory):
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAA", "BBB", "CCC"])

    prog = r.run(limit=1)

    assert prog.remaining == 2
    assert "left" in str(prog)


def test_stop_before_any_work_does_nothing(runner_factory):
    src = FakeSource()
    r = runner_factory(src)
    r.enqueue(["AAA"])
    r.request_stop()

    prog = r.run()

    assert src.calls == []
    assert prog.done == 0


# --- helper ------------------------------------------------------------------

def test_symbols_from_events_dedupes_and_sorts():
    events = [
        {"symbol": "msft"}, {"symbol": "AAPL"}, {"symbol": "AAPL"},
        {"symbol": None}, {"symbol": ""},
    ]

    assert symbols_from_events(events) == ["AAPL", "MSFT"]


# --- requeue ---------------------------------------------------------------------

def test_requeue_returns_finished_symbols_to_the_queue(tmp_path):
    """Shipped broken once: the method was appended to the wrong scope, so it
    was never on the class and only failed in CI under --requeue, which no test
    exercised."""
    runner = BackfillRunner(None, tmp_path / "c.db",
                            start=date(2024, 1, 1), end=date(2024, 2, 1))
    runner.enqueue(["AAA", "BBB", "CCC"])
    runner._record_success("AAA", 500, True)
    runner._record_success("BBB", 500, True)

    assert runner.pending() == ["CCC"]
    moved = runner.requeue()

    assert moved == 2
    assert runner.pending() == ["AAA", "BBB", "CCC"]
    runner.close()


def test_requeue_is_a_method_on_the_runner():
    """Pins the actual defect: it parsed fine but was nested inside a
    module-level function, so the attribute never existed."""
    assert callable(getattr(BackfillRunner, "requeue", None))


def test_requeue_resets_attempts_so_parked_failures_retry(tmp_path):
    runner = BackfillRunner(None, tmp_path / "c.db",
                            start=date(2024, 1, 1), end=date(2024, 2, 1))
    runner.enqueue(["BAD"])
    for _ in range(5):
        runner._record_failure("BAD", RuntimeError("vendor said no"))

    assert runner.pending() == [], "parked after max_attempts"
    runner.requeue()

    assert runner.pending() == ["BAD"]
    runner.close()


def test_requeue_on_an_empty_queue_moves_nothing(tmp_path):
    runner = BackfillRunner(None, tmp_path / "c.db",
                            start=date(2024, 1, 1), end=date(2024, 2, 1))
    runner.enqueue(["AAA"])

    assert runner.requeue() == 0, "AAA is still pending, not finished"
    runner.close()


def test_a_runner_without_a_source_can_queue_but_not_fetch(tmp_path):
    """Queueing and progress need no vendor credentials; fetching does. This
    split is what let enqueue-symbols run in a job holding no price keys."""
    runner = BackfillRunner(None, tmp_path / "c.db",
                            start=date(2024, 1, 1), end=date(2024, 2, 1))
    runner.enqueue(["AAA"])

    assert runner.progress().total == 1
    with pytest.raises(PriceError, match="without a price source"):
        runner.run()
    runner.close()
