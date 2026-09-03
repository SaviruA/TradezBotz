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


def test_the_repository_criteria_are_loadable_and_signed_off():
    c = load()

    assert c.confirmed is True
    assert c.unset_operator_values() == []
    assert c.signed_off.strip(), "a confirmed file must say who decided and why"


def test_the_backtest_is_never_sized_off_the_live_test_stake():
    """The invariant that protects every result. Sizing the backtest off a $100
    stake charges impact for ~$10 orders -- approximately zero -- and every net
    return then flatters any size actually traded later. A backtest may be
    pessimistic relative to reality; it must never be optimistic."""
    c = load()

    assert c.position_notional > c.live_test_capital, (
        "backtest position size must exceed the live test stake, or the cost "
        "model is charging less impact than reality will")


def test_the_live_test_holds_few_enough_positions_to_be_fillable():
    """Only 56% of listed US equities are fractionable, and at $10 a position
    just 24% of cached names are buyable as a whole share. A small stake split
    many ways is mostly unfillable rather than merely small."""
    c = load()

    per_position = c.live_test_capital / max(c.live_test_positions, 1)
    assert per_position >= 20, (
        f"${per_position:.0f} per live-test position is too small to reach most "
        "of the universe even with fractional shares; use fewer positions")


def test_paper_is_sized_like_the_backtest_not_like_the_account():
    """Paper's job is plumbing and live-vs-backtest agreement, and neither is
    served by trading the account's full balance -- that would "fill" orders
    which could never fill live. Matching the backtest size keeps paper,
    research and eventual live describing one strategy."""
    c = load()

    assert c.paper_capital == c.capital_at_risk, (
        "paper capital should mirror the backtest's sizing, not the account "
        "balance and not the live test stake")


def test_the_report_says_plainly_that_nothing_can_pass():
    text = report([GateResult("a", 5, False, ("x",))], Criteria())

    assert "CRITERIA UNCONFIRMED" in text
    assert "0 of 1" in text


# --- concurrency: the assumption that sets position size --------------------

def _labels(n_symbols, n_days=200, per_day=15):
    from datetime import UTC, date, datetime, timedelta

    from tradezbotz.research.labeler import Coverage, Label
    out = []
    for i in range(n_days * per_day):
        out.append(Label(
            symbol=f"S{i % n_symbols}", observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            entry_day=date(2025, 1, 1) + timedelta(days=i % n_days),
            entry_price=10.0, returns={1: 0.01, 5: 0.01, 20: 0.01},
            coverage=Coverage.COMPLETE))
    return out


def test_concurrency_is_measured_not_assumed():
    """The criteria assume 10 concurrent positions. Taking every open-market buy
    at a 5-day horizon holds a median of 97 distinct symbols -- an order of
    magnitude out, which mis-prices impact on every trade."""
    from tradezbotz.research.deployment import concurrent_positions

    stats = concurrent_positions(_labels(n_symbols=97), horizon=5)

    assert stats["median"] > 50
    assert stats["max"] >= stats["median"]


def test_a_concurrency_overrun_is_reported_with_the_real_position_size():
    from tradezbotz.research.deployment import sizing_warning

    msg = sizing_warning(_labels(n_symbols=97), 5, confirmed())

    assert "against an assumed 10" in msg
    assert "larger than the strategy would take" in msg
    assert "cap concurrency and rank" in msg


def test_concurrency_within_the_assumption_is_reported_as_fine():
    from tradezbotz.research.deployment import sizing_warning

    msg = sizing_warning(_labels(n_symbols=4, per_day=2), 1, confirmed())

    assert "within the assumed" in msg


def test_a_label_missing_that_horizon_is_not_counted_as_held():
    """A position is only held if it was actually taken at that horizon.
    Counting labels whose return never resolved would inflate concurrency and
    shrink position size for trades that were never made."""
    from datetime import UTC, date, datetime

    from tradezbotz.research.deployment import concurrent_positions
    from tradezbotz.research.labeler import Coverage, Label

    only_h5 = [Label(symbol="A", observed_at=datetime(2025, 1, 1, tzinfo=UTC),
                     entry_day=date(2025, 1, 1), entry_price=10.0,
                     returns={5: 0.01}, coverage=Coverage.COMPLETE)]

    assert concurrent_positions(only_h5, 5)["median"] == 1
    assert concurrent_positions(only_h5, 20) == {}


def test_concurrency_on_an_empty_label_set_is_silent():
    from tradezbotz.research.deployment import concurrent_positions, sizing_warning

    assert concurrent_positions([], 5) == {}
    assert sizing_warning([], 5, confirmed()) == ""


# --- concurrency measured on a SAMPLE ---------------------------------------

def test_a_sample_cannot_clear_the_concurrency_assumption():
    """The failure this prevents: a 1-in-41 stride sees 1/41 of the events per
    day, so concurrency looks comfortably inside the assumption. That is an
    artefact of sampling read as a property of the strategy, and it arrives
    exactly when the results start being trusted."""
    from tradezbotz.research.deployment import sizing_warning

    msg = sizing_warning(_labels(n_symbols=4, per_day=2), 1, confirmed(),
                         sampled_fraction=0.024)

    assert "LOWER BOUND and not a pass" in msg
    assert "within the assumed" not in msg


def test_a_full_population_within_the_assumption_still_passes_cleanly():
    from tradezbotz.research.deployment import sizing_warning

    msg = sizing_warning(_labels(n_symbols=4, per_day=2), 1, confirmed(),
                         sampled_fraction=1.0)

    assert "within the assumed" in msg


def test_an_overrun_on_a_sample_says_the_true_figure_is_higher():
    from tradezbotz.research.deployment import sizing_warning

    msg = sizing_warning(_labels(n_symbols=97), 5, confirmed(),
                         sampled_fraction=0.024)

    assert "true figure is higher" in msg
    assert "cap concurrency and rank" in msg


def test_the_measured_count_is_never_scaled_up_to_fake_precision():
    """Distinct symbols held saturates rather than growing linearly with event
    count, so a scaled figure would be invented. The number stays what was
    measured; only its interpretation changes."""
    from tradezbotz.research.deployment import concurrent_positions, sizing_warning

    labels = _labels(n_symbols=97)
    full = sizing_warning(labels, 5, confirmed(), sampled_fraction=1.0)
    part = sizing_warning(labels, 5, confirmed(), sampled_fraction=0.02)
    med = concurrent_positions(labels, 5)["median"]

    assert f"median {med} symbols" in full
    assert f"median {med} symbols" in part
