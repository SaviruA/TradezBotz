"""Tests for the deployment gate.

The gate's job is to refuse. Its most important behaviour is refusing a result
that is statistically sound but not yet justified — an unconfirmed criteria
file, an unconfirmed holdout, or an edge that does not survive the live haircut.
A gate that passes everything sound is not a gate, it is a formality.
"""

from __future__ import annotations

import json

import pytest

from tradezbotz.research.backtest import BacktestResult
from tradezbotz.research.deployment import (
    Criteria,
    DeploymentError,
    GateResult,
    evaluate,
    load,
    report,
    save,
)
from tradezbotz.research.sweep import Assessment, Verdict


def result(mean=0.06, net=0.045, dsr=0.99, trades=500):
    return BacktestResult(
        hypothesis="h", horizon=5, trial_id=1, n_events=trades, n_trades=trades,
        mean_return=mean, median_return=mean, stdev=0.03, hit_rate=0.55,
        sharpe_per_trade=0.4, sharpe_annualised=2.0, t_stat=4.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=dsr, n_trials=200, significant=True,
        n_symbols=120, mean_return_winsorised=mean, mean_return_net=net,
        costed=True, t_stat_clustered=3.5, n_effective=trades * 0.7,
        se_inflation=1.2)


def assessment(res=None, verdict=Verdict.KEEP, coverage=0.6, fallback=0.05,
               control=None):
    return Assessment("cand", 5, res or result(), control, verdict, "",
                      coverage, fallback)


def confirmed(**kw) -> Criteria:
    base = dict(capital_at_risk=25_000.0, max_drawdown_halt=5_000.0,
                confirmed=True, signed_off="test")
    base.update(kw)
    return Criteria(**base)


# --- the refusals that matter -----------------------------------------------

def test_unconfirmed_criteria_refuse_everything():
    """However good the numbers look. The operator has not chosen capital or a
    drawdown halt, so there is no basis for risking anything."""
    out = evaluate(assessment(), Criteria(), holdout_confirmed=True)

    assert out.passed is False
    assert any("not confirmed" in f for f in out.failures)


def test_unset_operator_values_are_named_individually():
    out = evaluate(assessment(), Criteria(confirmed=True), holdout_confirmed=True)

    joined = " ".join(out.failures)
    assert "capital_at_risk" in joined
    assert "max_drawdown_halt" in joined


def test_a_sound_result_without_holdout_confirmation_is_refused():
    out = evaluate(assessment(), confirmed(), holdout_confirmed=False)

    assert out.passed is False
    assert any("holdout" in f for f in out.failures)


def test_a_fully_qualified_candidate_passes():
    out = evaluate(assessment(), confirmed(), holdout_confirmed=True)

    assert out.passed is True, out.failures


# --- the live haircut -------------------------------------------------------

def test_an_edge_that_does_not_survive_halving_is_refused():
    """6% gross against 5.5% costs is profitable as measured and worthless at
    half. The documented base rate for backtest-to-live decay is roughly that
    severe, so this is the criterion doing its job."""
    thin = result(mean=0.06, net=0.005)

    out = evaluate(assessment(thin), confirmed(), holdout_confirmed=True)

    assert out.passed is False
    assert any("haircut" in f for f in out.failures)


def test_a_wide_edge_survives_and_the_margin_is_reported():
    out = evaluate(assessment(result(mean=0.06, net=0.05)), confirmed(),
                   holdout_confirmed=True)

    assert out.passed is True
    assert any("survives the haircut" in n for n in out.notes)


def test_an_uncosted_result_is_refused_outright():
    uncosted = BacktestResult(
        hypothesis="h", horizon=5, trial_id=1, n_events=500, n_trades=500,
        mean_return=0.06, median_return=0.06, stdev=0.03, hit_rate=0.6,
        sharpe_per_trade=0.4, sharpe_annualised=2.0, t_stat=4.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.99, n_trials=200, significant=True,
        costed=False)

    out = evaluate(assessment(uncosted), confirmed(), holdout_confirmed=True)

    assert any("uncosted" in f for f in out.failures)


# --- stricter than the research gates ---------------------------------------

def test_every_deployment_threshold_is_stricter_than_its_research_gate():
    """The gate exists because a result can be perfectly sound and still not
    justify risk. If any threshold were looser, it would add nothing."""
    from tradezbotz.research import sweep

    c = Criteria()
    assert c.min_trades > sweep.MIN_TRADES
    assert c.min_coverage > sweep.MIN_COVERAGE
    assert c.max_fallback_share < sweep.MAX_FALLBACK_SHARE
    assert c.max_control_ratio < sweep.CONTROL_TOLERANCE


def test_a_research_verdict_short_of_keep_is_refused():
    out = evaluate(assessment(verdict=Verdict.NOT_SIGNIFICANT), confirmed(),
                   holdout_confirmed=True)

    assert any("not KEEP" in f for f in out.failures)


def test_thin_coverage_is_refused_even_when_significant():
    out = evaluate(assessment(coverage=0.25), confirmed(), holdout_confirmed=True)

    assert any("coverage" in f for f in out.failures)


def test_a_result_priced_mostly_by_the_fallback_is_refused():
    out = evaluate(assessment(fallback=0.5), confirmed(), holdout_confirmed=True)

    assert any("fallback constant" in f for f in out.failures)


def test_every_failure_is_reported_not_just_the_first():
    """A candidate short on three criteria is a different situation from one
    short on a single threshold, and stopping at the first hides that."""
    bad = result(mean=0.01, net=0.001, dsr=0.2, trades=40)

    out = evaluate(assessment(bad, coverage=0.05, fallback=0.9), Criteria())

    assert len(out.failures) >= 4


def test_a_missing_control_is_a_note_not_a_pass():
    out = evaluate(assessment(), confirmed(), holdout_confirmed=True)

    assert any("no control" in n for n in out.notes)


# --- the committed file -----------------------------------------------------

def test_criteria_round_trip(tmp_path):
    path = tmp_path / "c.json"
    save(confirmed(min_trades=321), path)

    back = load(path)

    assert back.min_trades == 321
    assert back.confirmed is True


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """A typo would silently drop a threshold, which is the one failure a
    criteria file must not have."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"min_trades": 200, "min_tradez": 5}))

    with pytest.raises(DeploymentError, match="unknown criteria"):
        load(path)


def test_a_missing_file_is_refused_with_an_explanation(tmp_path):
    with pytest.raises(DeploymentError, match="must be committed"):
        load(tmp_path / "absent.json")


def test_the_repository_criteria_are_loadable_and_currently_unconfirmed():
    """The committed state: thresholds set blind, operator values unset, so the
    gate refuses everything until a deliberate sign-off."""
    c = load()

    assert c.confirmed is False
    assert set(c.unset_operator_values()) == {"capital_at_risk",
                                              "max_drawdown_halt"}


def test_the_report_says_plainly_that_nothing_can_pass():
    text = report([GateResult("a", 5, False, ("x",))], Criteria())

    assert "CRITERIA UNCONFIRMED" in text
    assert "0 of 1" in text
