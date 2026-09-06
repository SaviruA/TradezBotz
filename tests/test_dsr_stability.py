"""The DSR is the gate that decides money, so its movements must be explicable.

`opportunistic buy + liquid` at h=60 went DSR 0.550 -> 0.943 across two runs
while its net return got WORSE (+10.88% -> +9.79%) and the trial count ROSE
(2,185 -> 3,259). More trials is supposed to make the bar harder. Something
else moved further in the other direction.

`expected_max_sharpe = sd * f(N)` where `sd` is the standard deviation of every
Sharpe ever registered. Two terms, opposite signs:

    N up    -> f(N) up   -> bar up   -> DSR down
    sd down -> bar down  -> DSR up   for every candidate at once

The second is the dangerous one, because nothing about a given candidate
changed and yet its verdict did. These tests pin the direction of each term so
a future swing can be attributed from the log rather than reconstructed.
"""

from __future__ import annotations

import pytest

from tradezbotz.research.trials import (
    TrialRegistry,
    assess,
    deflated_sharpe,
    expected_max_sharpe,
)


# --- the two terms, and their signs ----------------------------------------

def test_more_trials_raises_the_bar():
    """The whole point of the correction: a wider search must be harder to
    clear."""
    low = expected_max_sharpe(10, 0.04)
    high = expected_max_sharpe(10_000, 0.04)

    assert high > low


def test_a_smaller_spread_of_sharpes_lowers_the_bar():
    """The mechanism behind the swing. Registering many similar Sharpes shrinks
    the variance, and the bar drops for every candidate at once -- without any
    of them having improved."""
    wide = expected_max_sharpe(1_000, 0.25)
    narrow = expected_max_sharpe(1_000, 0.01)

    assert narrow < wide


def test_a_falling_variance_can_outrun_a_rising_trial_count():
    """The exact shape of what happened: trials rose AND the bar fell, because
    the variance fell further. Pinned so the interaction is documented rather
    than rediscovered."""
    before = expected_max_sharpe(2_185, 0.09)
    after = expected_max_sharpe(3_259, 0.01)

    assert after < before


def test_the_same_observed_sharpe_scores_differently_under_each():
    sr, n = 0.15, 500
    strict = deflated_sharpe(sr, n_trials=3_259, n_obs=n, sharpe_variance=0.09)
    loose = deflated_sharpe(sr, n_trials=3_259, n_obs=n, sharpe_variance=0.01)

    assert loose > strict


def test_a_larger_effective_sample_sharpens_the_verdict_either_way():
    """The third term, and the only legitimate reason a row's DSR should move
    between runs on the same data. More observations do not simply raise the
    score -- they make the existing verdict more certain, which cuts BOTH ways:
    above the bar the score rises, below it the score falls."""
    bar = expected_max_sharpe(1_000, 0.04)

    above = _pair(bar + 0.20)
    below = _pair(bar - 0.05)

    assert above["large"] > above["small"], "more evidence should confirm an edge"
    assert below["large"] < below["small"], "more evidence should confirm its absence"


def _pair(sr: float) -> dict:
    return {
        "small": deflated_sharpe(sr, n_trials=1_000, n_obs=100,
                                 sharpe_variance=0.04),
        "large": deflated_sharpe(sr, n_trials=1_000, n_obs=5_000,
                                 sharpe_variance=0.04),
    }


# --- the inputs are reported, so a swing is attributable --------------------

def test_assess_returns_the_inputs_that_produced_its_verdict(tmp_path):
    reg = TrialRegistry(tmp_path / "t.db")
    try:
        for i in range(5):
            tid = reg.register(f"h{i}", "because", dataset=f"d{i}")
            reg.complete(tid, sharpe=0.1 * i, n_obs=100)

        out = assess(reg, observed_sharpe_annual=2.0, n_obs=500)

        assert out["sharpe_population"] == 5
        assert out["sharpe_variance"] > 0
        assert out["n_obs_effective"] == 500
        assert out["observed_sharpe_per_obs"] < out["observed_sharpe_annual"]
    finally:
        reg.close()


def test_the_reported_variance_is_the_one_actually_used(tmp_path):
    """Otherwise the log explains a verdict the code did not reach."""
    reg = TrialRegistry(tmp_path / "t.db")
    try:
        for i in range(6):
            tid = reg.register(f"h{i}", "because", dataset=f"d{i}")
            reg.complete(tid, sharpe=0.2 * i, n_obs=100)

        out = assess(reg, observed_sharpe_annual=2.0, n_obs=400)
        recomputed = deflated_sharpe(
            out["observed_sharpe_per_obs"], n_trials=out["n_trials"],
            n_obs=400, sharpe_variance=out["sharpe_variance"])

        assert recomputed == pytest.approx(out["deflated_sharpe"])
    finally:
        reg.close()


def test_a_registry_with_one_completed_trial_does_not_divide_by_zero(tmp_path):
    reg = TrialRegistry(tmp_path / "t.db")
    try:
        tid = reg.register("only", "because", dataset="d")
        reg.complete(tid, sharpe=0.3, n_obs=100)

        out = assess(reg, observed_sharpe_annual=1.0, n_obs=100)

        assert 0.0 <= out["deflated_sharpe"] <= 1.0
    finally:
        reg.close()


def test_a_zero_variance_registry_does_not_pass_everything(tmp_path):
    """Every registered Sharpe identical gives sd=0 and an expected max of
    zero, which would make the deflation a no-op. The score must still be
    bounded by the sample rather than waved through."""
    reg = TrialRegistry(tmp_path / "t.db")
    try:
        for i in range(10):
            tid = reg.register(f"h{i}", "because", dataset=f"d{i}")
            reg.complete(tid, sharpe=0.25, n_obs=100)

        out = assess(reg, observed_sharpe_annual=0.0, n_obs=100)

        assert out["sharpe_variance"] == pytest.approx(0.0)
        assert out["deflated_sharpe"] <= 0.5
    finally:
        reg.close()
