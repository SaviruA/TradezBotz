"""Tests for pipeline health detection.

The condition being caught is not "a step failed" -- it is "a step has failed
every run for a week behind `continue-on-error`, while the badge stayed green".
Eight of twenty steps carry that flag, so a run can report success having
accomplished almost nothing, and nothing in the output says so.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradezbotz.research.runlog import (
    FAILURE_STREAK_ALERT,
    STALE_AFTER_HOURS,
    RunLog,
    describe,
)


@pytest.fixture
def log(tmp_path):
    with RunLog(tmp_path / "r.db") as r:
        yield r


def run(log, rid, **steps):
    log.start(rid)
    for step, outcome in steps.items():
        log.record(rid, step, outcome)
    log.finish(rid)


# --- the silent-failure case ------------------------------------------------

def test_a_step_failing_every_run_is_flagged(log):
    for i in range(FAILURE_STREAK_ALERT):
        run(log, f"r{i}", bulk="success", intraday="failure")

    text, unhealthy = describe(log)

    assert unhealthy is True
    assert "ALERT" in text
    assert "reporting success while not doing this work" in text


def test_one_failure_is_weather_not_a_finding(log):
    run(log, "r0", bulk="success", intraday="success")
    run(log, "r1", bulk="success", intraday="failure")

    text, unhealthy = describe(log)

    assert unhealthy is False, "a single failure must not cry wolf"


def test_a_recovered_step_stops_alerting(log):
    for i in range(FAILURE_STREAK_ALERT + 2):
        run(log, f"r{i}", intraday="failure")
    run(log, "recovered", intraday="success")

    text, unhealthy = describe(log)

    assert unhealthy is False
    assert "ALERT" not in text


def test_the_streak_counts_only_consecutive_outcomes(log):
    run(log, "a", intraday="failure")
    run(log, "b", intraday="success")
    run(log, "c", intraday="failure")
    run(log, "d", intraday="failure")

    health = {h.step: h for h in log.health()}

    assert health["intraday"].streak == 2, "the success breaks the streak"


# --- the run-never-happened case --------------------------------------------

def test_a_long_gap_since_the_last_run_is_unhealthy(log):
    run(log, "old", bulk="success")
    later = datetime.now(timezone.utc) + timedelta(hours=STALE_AFTER_HOURS + 5)

    text, unhealthy = describe(log, now=later)

    assert unhealthy is True
    assert "STALE" in text
    assert "disabled entirely after 60 days" in text


def test_a_recent_run_is_healthy(log):
    run(log, "recent", bulk="success")

    text, unhealthy = describe(log)

    assert unhealthy is False
    assert "last completed run" in text


def test_an_empty_log_is_not_an_alert(log):
    """A first run, or a reset state blob. Neither is evidence of a problem."""
    text, unhealthy = describe(log)

    assert unhealthy is False
    assert "no previous run recorded" in text


def test_an_unfinished_run_does_not_count_as_completed(log):
    """A run that crashed mid-way must not reset the staleness clock, or a
    pipeline that starts and dies every night looks healthy."""
    log.start("crashed")
    log.record("crashed", "bulk", "failure")

    assert log.hours_since_last_run() is None


# --- bookkeeping ------------------------------------------------------------

def test_recording_the_same_step_twice_in_a_run_overwrites(log):
    log.start("r")
    log.record("r", "intraday", "failure")
    log.record("r", "intraday", "success")

    health = {h.step: h for h in log.health()}

    assert health["intraday"].last_outcome == "success"


def test_starting_a_run_twice_does_not_lose_its_finish(log):
    run(log, "r", bulk="success")
    first = log.last_finished()
    log.start("r")

    assert log.last_finished() == first, "a re-start must not erase completion"
