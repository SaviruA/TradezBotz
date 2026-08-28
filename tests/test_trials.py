"""Tests for the trial registry and Deflated Sharpe Ratio."""

from __future__ import annotations

import math

import pytest

from tradezbotz.research.trials import (
    ABANDONED,
    COMPLETED,
    PENDING,
    TrialError,
    TrialRegistry,
    annualised_to_per_obs,
    assess,
    deflated_sharpe,
    expected_max_sharpe,
)


@pytest.fixture
def reg(tmp_path):
    with TrialRegistry(tmp_path / "trials.db") as r:
        yield r


# --- registry ----------------------------------------------------------------

def test_register_returns_an_id_and_starts_pending(reg):
    tid = reg.register("insider_opportunistic", "CMP: pattern breaks carry information")

    assert reg.get(tid).outcome == PENDING
    assert reg.count() == 1


def test_rationale_is_mandatory(reg):
    """A stated mechanism carries a better prior than a mined pattern, and can
    be falsified independently of the returns."""
    with pytest.raises(TrialError, match="rationale is required"):
        reg.register("some_idea", "   ")


def test_hypothesis_name_is_mandatory(reg):
    with pytest.raises(TrialError, match="hypothesis name"):
        reg.register("  ", "a reason")


def test_abandoned_trials_still_count(reg):
    """The whole point: an experiment dropped because it looked bad still
    consumed a lottery ticket. Excluding it inflates every later result."""
    a = reg.register("idea_a", "mechanism a")
    reg.register("idea_b", "mechanism b")
    reg.abandon(a, "looked unpromising")

    assert reg.get(a).outcome == ABANDONED
    assert reg.count() == 2


def test_completed_trial_stores_metrics(reg):
    tid = reg.register("idea", "mechanism")
    reg.complete(tid, sharpe=0.08, n_obs=500, n_trades=120, notes="train split")

    t = reg.get(tid)
    assert t.outcome == COMPLETED
    assert t.sharpe == pytest.approx(0.08)
    assert t.n_obs == 500


def test_params_round_trip(reg):
    tid = reg.register("idea", "mechanism", params={"min_notional": 100_000, "h": 5})

    assert reg.get(tid).params == {"min_notional": 100_000, "h": 5}


def test_unknown_trial_returns_none(reg):
    assert reg.get(99999) is None


# --- expected maximum Sharpe -------------------------------------------------

def test_expected_max_sharpe_grows_with_trial_count():
    """Selecting the best of many trials produces a positive Sharpe by
    construction. More trials, higher benchmark to beat."""
    v = 0.002
    assert expected_max_sharpe(1, v) == 0.0
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)


def test_expected_max_sharpe_scales_with_spread():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(100, 0.001)


# --- deflated Sharpe ---------------------------------------------------------

def base(**kw):
    args = dict(observed_sharpe=0.10, n_trials=1, n_obs=500, sharpe_variance=0.002)
    args.update(kw)
    return deflated_sharpe(**args)


def test_dsr_is_a_probability():
    assert 0.0 <= base() <= 1.0


def test_more_trials_lowers_confidence():
    """The core correction: the same backtest means less after more searching."""
    assert base(n_trials=1) > base(n_trials=100) > base(n_trials=1000)


def test_stronger_sharpe_raises_confidence():
    assert base(observed_sharpe=0.02) < base(observed_sharpe=0.15)


def test_longer_sample_raises_confidence():
    assert base(n_obs=100) < base(n_obs=2000)


def test_fat_tails_lower_confidence():
    """Excess kurtosis makes an extreme Sharpe easier to get by luck."""
    assert base(kurtosis=3.0) > base(kurtosis=12.0)


def test_negative_skew_is_penalised():
    assert base(skew=0.5) > base(skew=-0.5)


def test_degenerate_sample_returns_zero():
    assert base(n_obs=1) == 0.0


def test_annualised_conversion():
    """Feeding an annualised Sharpe straight in would inflate it ~sqrt(252)."""
    assert annualised_to_per_obs(1.59) == pytest.approx(0.10, abs=0.005)


# --- assess ------------------------------------------------------------------

def test_assess_uses_the_registry_trial_count(reg):
    for i in range(50):
        tid = reg.register(f"idea_{i}", "mechanism")
        reg.complete(tid, sharpe=0.01 * (i % 7), n_obs=500)

    out = assess(reg, observed_sharpe_annual=1.5, n_obs=500)

    assert out["n_trials"] == 50
    assert 0.0 <= out["deflated_sharpe"] <= 1.0


def test_the_same_result_fails_after_enough_searching(tmp_path):
    """A Sharpe that is convincing on one pre-registered test should not be
    convincing after a thousand attempts. This is the whole point of the file."""
    lonely = TrialRegistry(tmp_path / "a.db")
    tid = lonely.register("only_idea", "mechanism")
    lonely.complete(tid, sharpe=0.09, n_obs=500)

    busy = TrialRegistry(tmp_path / "b.db")
    for i in range(1000):
        t = busy.register(f"idea_{i}", "mechanism")
        busy.complete(t, sharpe=0.09 + 0.001 * (i % 11), n_obs=500)

    single = assess(lonely, 1.6, 500)
    dredged = assess(busy, 1.6, 500)

    assert single["deflated_sharpe"] > dredged["deflated_sharpe"]
    assert single["expected_max_sharpe_annual"] < dredged["expected_max_sharpe_annual"]
    lonely.close(); busy.close()


def test_assess_flags_significance(reg):
    strong = assess(reg, observed_sharpe_annual=6.0, n_obs=2000)
    weak = assess(reg, observed_sharpe_annual=0.2, n_obs=2000)

    assert strong["significant"] is True
    assert weak["significant"] is False


# --- holdout accounting ------------------------------------------------------

def test_holdout_accesses_are_counted(reg):
    assert reg.holdout_accesses() == 0

    reg.record_holdout_access("idea", "final confirmation")

    assert reg.holdout_accesses() == 1
    assert reg.holdout_accesses("idea") == 1
    assert reg.holdout_accesses("other") == 0
