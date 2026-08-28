"""Tests for the single-instance lock.

This guards a failure that actually happened: two ingests running at once,
pushing EDGAR traffic to ~16 req/s against the SEC's 10/s ceiling, and fighting
over the same SQLite files.
"""

from __future__ import annotations

import os

import pytest

from tradezbotz.lock import LockHeld, SingleInstance


def test_lock_is_acquired_and_released(tmp_path):
    lock = SingleInstance("ingest", tmp_path)

    lock.acquire()
    assert lock.path.exists()

    lock.release()
    assert not lock.path.exists()


def test_second_acquire_is_refused_while_held(tmp_path):
    first = SingleInstance("ingest", tmp_path)
    first.acquire()

    with pytest.raises(LockHeld, match="already active"):
        SingleInstance("ingest", tmp_path).acquire()

    first.release()


def test_error_explains_the_rate_limit_consequence(tmp_path):
    SingleInstance("ingest", tmp_path).acquire()

    with pytest.raises(LockHeld, match="SEC"):
        SingleInstance("ingest", tmp_path).acquire()


def test_different_names_do_not_block_each_other(tmp_path):
    """Ingest and backfill hit different services and may run together."""
    SingleInstance("ingest", tmp_path).acquire()
    SingleInstance("backfill", tmp_path).acquire()  # must not raise


def test_stale_lock_from_a_killed_run_is_reclaimed(tmp_path):
    """A killed process leaves its lockfile behind. That must not wedge the
    pipeline forever -- exactly the situation after force-killing a runaway."""
    stale = tmp_path / "ingest.lock"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("999999")  # a PID that does not exist

    SingleInstance("ingest", tmp_path).acquire()  # must not raise

    assert stale.read_text() == str(os.getpid())


def test_corrupt_lockfile_is_reclaimed(tmp_path):
    (tmp_path / "ingest.lock").write_text("not-a-pid")

    SingleInstance("ingest", tmp_path).acquire()  # must not raise


def test_context_manager_releases_on_exception(tmp_path):
    lock = SingleInstance("ingest", tmp_path)

    with pytest.raises(ValueError):
        with lock:
            raise ValueError("boom")

    assert not lock.path.exists(), "a crash must not wedge the next run"


def test_release_is_safe_when_not_held(tmp_path):
    SingleInstance("ingest", tmp_path).release()  # must not raise
