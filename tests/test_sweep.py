"""Tests for the whole-backlog sweep.

The rule being enforced: nothing is dropped before it is measured, and an
omission has to be visible. A test that was never run must never look the same
as one that was run and failed.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.backtest import (
    BacktestResult,
    all_of,
    everything,
    field_equals,
)
from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.sweep import (
    Candidate,
    SweepError,
    Verdict,
    judge,
    priors_vs_outcomes,
    report,
    sweep,
)
from tradezbotz.research.trials import TrialRegistry


@pytest.fixture
def reg(tmp_path):
    with TrialRegistry(tmp_path / "t.db") as r:
        yield r


def dataset(n=300, edge=0.02, seed=1):
    rng = random.Random(seed)
    labels, payloads = [], []
    for i in range(n):
        r = rng.gauss(edge if i % 2 == 0 else 0.0, 0.03)
        labels.append(Label(
            symbol=f"S{i % 60}", observed_at=datetime(2025, 3, 4, tzinfo=UTC),
            entry_day=date(2025, 1, 1) + timedelta(days=i % 120),
            entry_price=10.0, returns={1: r, 5: r, 20: r},
            coverage=Coverage.COMPLETE))
        payloads.append({"code": "P" if i % 2 == 0 else "S"})
    return labels, payloads


# --- candidates ------------------------------------------------------------------

def test_a_candidate_is_runnable_unless_blocked():
    assert Candidate("a", everything, "r").runnable is True
    assert Candidate("b", everything, "r", blocked_by="no data").runnable is False


def test_blocked_candidates_are_reported_as_untested_not_rejected():
    """The distinction the whole module exists for: never-run must not look the
    same as run-and-failed."""
    cands = [Candidate("built", everything, "r"),
             Candidate("waiting", everything, "r", blocked_by="needs 13D data")]

    text = report([], cands)

    assert "NOT measured" in text
    assert "untested, not rejected" in text
    assert "needs 13D data" in text


# --- the gates --------------------------------------------------------------------

def make(mean=0.02, net=0.01, costed=True, trades=200, sig=True,
         outlier=False):
    return BacktestResult(
        hypothesis="h", horizon=5, trial_id=1, n_events=trades, n_trades=trades,
        mean_return=mean, median_return=mean, stdev=0.03, hit_rate=0.55,
        sharpe_per_trade=0.3, sharpe_annualised=2.0, t_stat=3.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.9, n_trials=10, significant=sig,
        n_symbols=60,
        mean_return_winsorised=(mean * 0.2 if outlier else mean),
        mean_return_net=net, costed=costed, t_stat_clustered=2.5,
        n_effective=trades * 0.8, se_inflation=1.1,
    )


def test_too_few_trades_is_its_own_verdict():
    assert judge(make(trades=5), None) == Verdict.TOO_FEW_TRADES


def test_no_gross_edge_is_reported_as_no_edge():
    """Not as a cost failure it never reached."""
    assert judge(make(mean=-0.01), None) == Verdict.NO_EDGE


def test_costs_can_disqualify():
    assert judge(make(mean=0.005, net=-0.004), None) == Verdict.EATEN_BY_COSTS


def test_a_control_that_performs_equally_disqualifies():
    """This caught a real labeller bug once, where negate(BUY) returned results
    identical to BUY."""
    assert judge(make(mean=0.02), make(mean=0.019)) == Verdict.MATCHES_CONTROL


def test_a_weak_control_does_not_disqualify():
    assert judge(make(mean=0.02), make(mean=0.002)) == Verdict.KEEP


def test_a_tiny_control_sample_is_ignored():
    """A control with 5 trades tells us nothing either way."""
    assert judge(make(mean=0.02), make(mean=0.02, trades=5)) == Verdict.KEEP


def test_outlier_dependence_disqualifies():
    assert judge(make(outlier=True), None) == Verdict.OUTLIER_DEPENDENT


def test_insignificance_disqualifies_last():
    assert judge(make(sig=False), None) == Verdict.NOT_SIGNIFICANT


def test_a_clean_result_is_kept():
    assert judge(make(), None) == Verdict.KEEP


def test_uncosted_results_are_not_failed_on_costs():
    """An uncosted result cannot pass survives_costs, but it must not be
    reported as eaten by costs it was never charged."""
    assert judge(make(costed=False, net=0.0), None) != Verdict.EATEN_BY_COSTS


# --- the sweep ---------------------------------------------------------------------

def test_every_candidate_is_measured_at_every_horizon(reg):
    labels, payloads = dataset()
    cands = [Candidate("buys", field_equals("code", "P"), "r"),
             Candidate("all", everything, "r")]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(1, 5))

    assert len(out) == 4, "2 candidates x 2 horizons"
    assert {a.horizon for a in out} == {1, 5}


def test_blocked_candidates_are_not_run(reg):
    labels, payloads = dataset()
    cands = [Candidate("ok", everything, "r"),
             Candidate("blocked", everything, "r", blocked_by="no data yet")]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,))

    assert [a.name for a in out] == ["ok"]


def test_controls_are_registered_as_trials_too(reg):
    """Conservative by choice: a control does not compete for selection, but
    counting it raises the bar against ourselves."""
    labels, payloads = dataset()

    sweep([Candidate("buys", field_equals("code", "P"), "r")],
          labels, payloads, registry=reg, horizons=(5,))

    assert reg.count() == 2, "signal and control both registered"


def test_the_baseline_gets_no_control(reg):
    """`everything` has no meaningful complement."""
    labels, payloads = dataset()

    sweep([Candidate("all", everything, "r")], labels, payloads,
          registry=reg, horizons=(5,))

    assert reg.count() == 1


def test_costs_reach_the_run(reg):
    labels, payloads = dataset()

    out = sweep([Candidate("buys", field_equals("code", "P"), "r")],
                labels, payloads, registry=reg, horizons=(5,),
                costs=lambda lab: 0.0093)

    assert out[0].result.costed is True
    assert out[0].result.median_cost_bps == pytest.approx(93.0)


def test_trial_count_grows_with_the_sweep(reg):
    """Testing everything raises the DSR bar. That is the honest price."""
    labels, payloads = dataset()
    cands = [Candidate(f"c{i}", field_equals("code", "P"), "r") for i in range(5)]

    sweep(cands, labels, payloads, registry=reg, horizons=(1, 5, 20))

    assert reg.count() == 30, "5 candidates x 3 horizons x (signal + control)"


# --- reporting ----------------------------------------------------------------------

def test_report_shows_failures_not_only_survivors(reg):
    labels, payloads = dataset(edge=-0.05)

    out = sweep([Candidate("bad", field_equals("code", "P"), "r")],
                labels, payloads, registry=reg, horizons=(5,))
    text = report(out)

    assert "no edge" in text or "not significant" in text
    assert "survived every gate" in text


def test_priors_are_checked_against_outcomes(reg):
    labels, payloads = dataset()
    cands = [Candidate("buys", field_equals("code", "P"), "r",
                       prior="expected to work")]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,))
    text = priors_vs_outcomes(out, cands)

    assert "expected to work" in text
    assert "survived" in text


def test_an_unmeasured_prior_is_flagged():
    """A prior that never gets checked is an opinion that acted without being
    measured."""
    cands = [Candidate("never run", everything, "r", prior="will fail",
                       blocked_by="no data")]

    text = priors_vs_outcomes([], cands)

    assert "NOT MEASURED" in text
    assert "stands unchecked" in text


# --- the holdout guard ------------------------------------------------------------

def test_sweeping_the_holdout_is_refused_without_a_declared_finalist(reg):
    """`splits.Split` seals the holdout behind `unlock_holdout`, which records
    the access. Nothing connected that seal to the sweep: passing
    partition="holdout" measured it with no access logged at all."""
    labels, payloads = dataset()
    cands = [Candidate("a", everything, "r"), Candidate("b", everything, "r")]

    with pytest.raises(SweepError) as exc:
        sweep(cands, labels, payloads, registry=reg, partition="holdout")

    assert "no recorded access" in str(exc.value)
    assert "unlock_holdout" in str(exc.value)


def test_the_refused_holdout_sweep_measures_nothing_at_all(reg):
    """The check runs before any measurement. A sweep that would have touched
    the holdout improperly must touch none of it, not the first half."""
    labels, payloads = dataset()
    cands = [Candidate("declared", everything, "r"),
             Candidate("undeclared", everything, "r")]
    reg.record_holdout_access("declared", "finalist")

    with pytest.raises(SweepError):
        sweep(cands, labels, payloads, registry=reg, partition="holdout")

    assert reg.count() == 0


def test_the_holdout_opens_once_every_candidate_is_declared(reg):
    labels, payloads = dataset()
    cands = [Candidate("a", everything, "r", controlled=False)]
    reg.record_holdout_access("a", "declared finalist after the train sweep")

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,),
                partition="holdout")

    assert len(out) == 1
    assert reg.get(out[0].result.trial_id).split == "holdout"


def test_train_and_validation_need_no_declaration(reg):
    labels, payloads = dataset()
    cands = [Candidate("a", everything, "r", controlled=False)]

    for partition in ("train", "validation"):
        out = sweep(cands, labels, payloads, registry=reg, horizons=(5,),
                    partition=partition)
        assert len(out) == 1


# --- controls ---------------------------------------------------------------------

def test_a_selector_that_admits_everything_gets_no_control(reg):
    """The bug this replaces: the check was `cand.selector is not everything`,
    an identity test that only recognised the literal baseline function. Any
    equivalent selector -- a lambda, a wrapper, a filter every event happens to
    pass -- slipped through and had a control run against an empty complement.
    """
    labels, payloads = dataset()
    admits_all = all_of(everything)          # not `everything` by identity
    cands = [Candidate("wrapped", admits_all, "r")]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,))

    assert out[0].control is None
    assert "below the" in out[0].control_note


def test_an_empty_control_does_not_consume_trial_budget(reg):
    """Not cosmetic. Every registered trial raises the Deflated Sharpe bar for
    every other candidate, so a control that could never have said anything
    makes real findings harder to establish in exchange for nothing."""
    labels, payloads = dataset()

    sweep([Candidate("wrapped", all_of(everything), "r")], labels, payloads,
          registry=reg, horizons=(5,))

    assert reg.count() == 1


def test_a_real_selector_still_gets_its_control(reg):
    labels, payloads = dataset()
    cands = [Candidate("buys", field_equals("code", "P"), "r")]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,))

    assert out[0].control is not None
    assert out[0].control.n_trades > 0
    assert out[0].control_note == ""


def test_controlled_false_skips_the_control_and_says_so(reg):
    labels, payloads = dataset()
    cands = [Candidate("baseline", everything, "r", controlled=False)]

    out = sweep(cands, labels, payloads, registry=reg, horizons=(5,))

    assert out[0].control is None
    assert out[0].control_note == "control not meaningful for this candidate"
    assert reg.count() == 1


def test_the_report_distinguishes_no_control_from_a_matching_control(reg):
    """An empty control column reads as 'no control needed' unless it says
    otherwise, and those are opposite claims."""
    labels, payloads = dataset()
    cands = [Candidate("baseline", everything, "r", controlled=False)]

    text = report(sweep(cands, labels, payloads, registry=reg, horizons=(5,)))

    assert "ran without a control" in text
