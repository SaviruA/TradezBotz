"""Tests for the locked train/validation/holdout split."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tradezbotz.research.splits import (
    HoldoutLocked,
    chronological_split,
    filter_events,
)
from tradezbotz.research.trials import TrialRegistry

START, END = date(2024, 9, 1), date(2026, 8, 30)


@pytest.fixture
def split():
    return chronological_split(START, END)


@pytest.fixture
def reg(tmp_path):
    with TrialRegistry(tmp_path / "trials.db") as r:
        yield r


# --- ordering ----------------------------------------------------------------

def test_partitions_are_chronological_and_contiguous(split):
    """Never shuffled: random splits leak the future through overlapping
    horizons and shared regimes."""
    assert split.train.start == START
    assert split.train.end < split.validation.start
    assert split.validation.end < split._holdout.start
    assert split._holdout.end == END


def test_holdout_is_the_most_recent_slice(split):
    """The recent regime is the one a live strategy meets first."""
    assert split._holdout.end == END
    assert split._holdout.start > split.train.end


def test_train_is_the_largest_partition(split):
    assert split.train.days > split.validation.days
    assert split.train.days > split._holdout.days


def test_partition_lookup(split):
    assert split.of(START) == "train"
    assert split.of(END) == "holdout"
    assert split.of(date(2020, 1, 1)) == "outside"
    assert split.of(split.validation.start) == "validation"


def test_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        chronological_split(START, END, validation_frac=0.5, holdout_frac=0.6)
    with pytest.raises(ValueError, match="end must be after"):
        chronological_split(END, START)


# --- the lock ----------------------------------------------------------------

def test_holdout_property_raises(split):
    """Accessing it casually must be impossible, not merely discouraged."""
    with pytest.raises(HoldoutLocked, match="sealed"):
        _ = split.holdout


def test_unlock_requires_hypothesis_and_reason(split, reg):
    with pytest.raises(HoldoutLocked):
        split.unlock_holdout(reg, "", "final check")
    with pytest.raises(HoldoutLocked):
        split.unlock_holdout(reg, "idea", "  ")


def test_unlock_returns_the_range_and_records_access(split, reg):
    rng = split.unlock_holdout(reg, "insider_opportunistic", "declared finalist")

    assert rng.end == END
    assert reg.holdout_accesses("insider_opportunistic") == 1


def test_second_unlock_warns_that_it_is_no_longer_out_of_sample(split, reg, capsys):
    split.unlock_holdout(reg, "idea", "first look")
    split.unlock_holdout(reg, "idea", "after a tweak")

    assert "no longer an out-of-sample test" in capsys.readouterr().out
    assert reg.holdout_accesses("idea") == 2


def test_describe_marks_the_holdout_sealed(split):
    assert "[sealed]" in split.describe()


# --- event filtering ---------------------------------------------------------

def ev(day: date):
    return {"observed_at": datetime(day.year, day.month, day.day, 20, tzinfo=UTC)}


def test_filter_events_by_partition(split):
    events = [ev(START), ev(split.validation.start), ev(END), ev(date(2020, 1, 1))]

    assert len(filter_events(events, split, "train")) == 1
    assert len(filter_events(events, split, "validation")) == 1
    assert len(filter_events(events, split, "holdout")) == 1


def test_filter_uses_observation_date_not_transaction_date(split):
    """Consistent with the point-in-time rule: what mattered is when the
    information became knowable."""
    e = {
        "observed_at": datetime(2026, 8, 20, 20, tzinfo=UTC),  # holdout
        "payload": {"transaction_date": "2024-10-01"},          # train
    }

    assert filter_events([e], split, "holdout") == [e]
    assert filter_events([e], split, "train") == []


def test_filter_accepts_iso_strings(split):
    e = {"observed_at": datetime(2026, 8, 20, 20, tzinfo=UTC).isoformat()}

    assert len(filter_events([e], split, "holdout")) == 1
