"""Tests for the event-study backtest engine."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime

import pytest

from tradezbotz.research.backtest import (
    BacktestResult,
    all_of,
    any_of,
    compare,
    everything,
    field_equals,
    negate,
    run,
    threshold,
)
from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.trials import TrialRegistry

H = 5


@pytest.fixture
def reg(tmp_path):
    with TrialRegistry(tmp_path / "trials.db") as r:
        yield r


def label(ret=0.01, horizon=H, coverage=Coverage.COMPLETE):
    return Label(
        symbol="AAPL",
        observed_at=datetime(2025, 3, 4, 18, tzinfo=UTC),
        entry_day=date(2025, 3, 5),
        entry_price=100.0,
        returns={horizon: ret},
        coverage=coverage,
    )


def dataset(n=200, ret=0.01, **payload):
    base = {"transaction_code": "P", "acquired_disposed": "A", "notional": 500_000}
    base.update(payload)
    return [base] * n, [label(ret) for _ in range(n)]


# --- selectors ---------------------------------------------------------------

def test_everything_takes_all_trades():
    p, l = dataset(50)
    assert sum(1 for a, b in zip(p, l) if everything(a, b)) == 50


def test_field_equals_filters():
    sel = field_equals("transaction_code", "P")
    assert sel({"transaction_code": "P"}, label()) is True
    assert sel({"transaction_code": "S"}, label()) is False


def test_threshold_ignores_non_numeric():
    sel = threshold("notional", 100_000)
    assert sel({"notional": 500_000}, label()) is True
    assert sel({"notional": 5_000}, label()) is False
    assert sel({"notional": None}, label()) is False
    assert sel({}, label()) is False


def test_all_of_requires_every_condition():
    """The combination case: two weak signals tested together."""
    sel = all_of(field_equals("transaction_code", "P"), threshold("notional", 100_000))

    assert sel({"transaction_code": "P", "notional": 500_000}, label()) is True
    assert sel({"transaction_code": "P", "notional": 1_000}, label()) is False
    assert sel({"transaction_code": "S", "notional": 500_000}, label()) is False


def test_any_of_widens():
    sel = any_of(field_equals("transaction_code", "P"), threshold("notional", 100_000))

    assert sel({"transaction_code": "S", "notional": 500_000}, label()) is True
    assert sel({"transaction_code": "S", "notional": 10}, label()) is False


def test_negate_builds_a_control_group():
    """If a signal works, its complement should not. That comparison is how you
    tell a signal from a property of the whole population."""
    sel = negate(field_equals("transaction_code", "P"))

    assert sel({"transaction_code": "S"}, label()) is True
    assert sel({"transaction_code": "P"}, label()) is False


def test_selectors_nest_arbitrarily():
    sel = all_of(
        any_of(field_equals("a", 1), field_equals("a", 2)),
        negate(threshold("b", 100)),
    )

    assert sel({"a": 1, "b": 5}, label()) is True
    assert sel({"a": 3, "b": 5}, label()) is False
    assert sel({"a": 1, "b": 500}, label()) is False


# --- running -----------------------------------------------------------------

def test_run_registers_a_trial(reg):
    p, l = dataset(100)

    res = run(l, p, hypothesis="baseline", rationale="all events", registry=reg)

    assert reg.count() == 1
    assert reg.get(res.trial_id).hypothesis == "baseline"


def test_rationale_is_enforced_through_the_registry(reg):
    p, l = dataset(10)

    with pytest.raises(Exception, match="rationale"):
        run(l, p, hypothesis="x", rationale="", registry=reg)


def test_statistics_are_computed(reg):
    p, l = dataset(100, ret=0.02)

    res = run(l, p, hypothesis="h", rationale="r", registry=reg)

    assert res.n_trades == 100
    assert res.mean_return == pytest.approx(0.02)
    assert res.hit_rate == 1.0


def test_events_missing_the_horizon_are_skipped(reg):
    payloads = [{"k": 1}] * 4
    labels = [label(0.01, horizon=1), label(0.01, horizon=5),
              label(0.01, horizon=5), label(0.01, horizon=20)]

    res = run(labels, payloads, hypothesis="h", rationale="r",
              registry=reg, horizon=5)

    assert res.n_trades == 2, "only labels carrying h=5 are tradeable"
    assert res.n_events == 4


def test_too_few_trades_abandons_the_trial(reg):
    """Abandoned still counts toward N -- it consumed a lottery ticket."""
    p, l = dataset(1)

    res = run(l, p, hypothesis="thin", rationale="r", registry=reg)

    assert res.n_trades <= 1
    assert res.significant is False
    assert "insufficient" in res.notes
    assert reg.count() == 1


def test_coverage_is_reported_beside_the_result(reg):
    payloads = [{"k": 1}] * 3
    labels = [label(coverage=Coverage.COMPLETE),
              label(coverage=Coverage.DELISTED_DURING_WINDOW),
              label(coverage=Coverage.PARTIAL)]

    res = run(labels, payloads, hypothesis="h", rationale="r", registry=reg)

    assert res.coverage["complete"] == 1
    assert res.coverage["delisted_during_window"] == 1


def test_selector_narrows_the_trade_set(reg):
    payloads = [{"code": "P"}] * 60 + [{"code": "S"}] * 40
    labels = [label(0.02) for _ in range(60)] + [label(-0.01) for _ in range(40)]

    res = run(labels, payloads, hypothesis="buys", rationale="r",
              registry=reg, selector=field_equals("code", "P"))

    assert res.n_trades == 60
    assert res.mean_return == pytest.approx(0.02)


# --- the statistics that matter ----------------------------------------------

def test_pure_noise_is_not_significant(reg):
    """The engine must not bless randomness."""
    rng = random.Random(7)
    payloads = [{"k": 1}] * 400
    labels = [label(rng.gauss(0, 0.03)) for _ in range(400)]

    res = run(labels, payloads, hypothesis="noise", rationale="control",
              registry=reg)

    assert res.significant is False
    assert abs(res.t_stat) < 3


def test_the_same_edge_weakens_as_trials_accumulate(reg):
    """The core discipline: a result means less after more searching."""
    rng = random.Random(11)
    payloads = [{"k": 1}] * 300
    labels = [label(rng.gauss(0.004, 0.03)) for _ in range(300)]

    first = run(labels, payloads, hypothesis="early", rationale="r", registry=reg)
    for i in range(300):
        t = reg.register(f"filler_{i}", "r")
        reg.complete(t, sharpe=rng.gauss(0, 0.06), n_obs=300)
    later = run(labels, payloads, hypothesis="late", rationale="r", registry=reg)

    assert later.n_trials > first.n_trials
    assert later.deflated_sharpe < first.deflated_sharpe


def test_annualised_sharpe_is_not_what_feeds_the_dsr(reg):
    """Feeding an annualised figure to the DSR would inflate it ~sqrt(252).
    The registry must record the per-trade value."""
    rng = random.Random(5)
    payloads = [{"k": 1}] * 200
    labels = [label(rng.gauss(0.005, 0.02)) for _ in range(200)]

    res = run(labels, payloads, hypothesis="h", rationale="r", registry=reg)

    assert res.sharpe_per_trade > 0
    assert abs(res.sharpe_annualised) > abs(res.sharpe_per_trade)
    assert reg.get(res.trial_id).sharpe == pytest.approx(res.sharpe_per_trade)


def test_zero_variance_returns_zero_sharpe_not_infinity(reg):
    """Degenerate but worth pinning: identical returns mean undefined risk, and
    reporting an infinite Sharpe would be worse than reporting none."""
    p, l = dataset(50, ret=0.01)

    res = run(l, p, hypothesis="constant", rationale="r", registry=reg)

    assert res.stdev == 0.0
    assert res.sharpe_per_trade == 0.0
    assert res.significant is False


def test_skew_and_kurtosis_are_measured(reg):
    rng = random.Random(3)
    payloads = [{"k": 1}] * 300
    # One huge winner: strongly right-skewed, fat-tailed.
    rets = [rng.gauss(0, 0.01) for _ in range(299)] + [2.0]
    labels = [label(r) for r in rets]

    res = run(labels, payloads, hypothesis="skewed", rationale="r", registry=reg)

    assert res.skew > 1.0
    assert res.kurtosis > 5.0


# --- reporting ---------------------------------------------------------------

def test_compare_tabulates(reg):
    p, l = dataset(100)
    rs = [run(l, p, hypothesis=f"h{i}", rationale="r", registry=reg) for i in range(3)]

    out = compare(rs)

    assert "hypothesis" in out and "h0" in out and "h2" in out


def test_compare_handles_empty():
    assert compare([]) == "no results"


def test_summary_mentions_the_trial_count(reg):
    p, l = dataset(100)
    res = run(l, p, hypothesis="h", rationale="r", registry=reg)

    assert "trials" in res.summary()


# --- clustering --------------------------------------------------------------

def test_concentrated_sample_is_flagged_as_clustered(reg):
    """Real data hit this: 1,836 trades across 204 symbols, one $1.84 micro-cap
    contributing 154 of them. Those are not independent draws -- they trace one
    price path. The t-stat and DSR both assume independence."""
    payloads = [{"k": 1}] * 200
    labels = [label(0.03) for _ in range(200)]
    for lab_i in range(200):
        object.__setattr__(labels[lab_i], "symbol", "RCG")

    res = run(labels, payloads, hypothesis="clustered", rationale="r", registry=reg)

    assert res.n_symbols == 1
    assert res.clustered is True
    assert "clustered" in res.summary()


def test_diverse_sample_is_not_flagged(reg):
    payloads = [{"k": 1}] * 200
    labels = [label(0.01) for _ in range(200)]
    for i in range(200):
        object.__setattr__(labels[i], "symbol", f"SYM{i}")

    res = run(labels, payloads, hypothesis="diverse", rationale="r", registry=reg)

    assert res.n_symbols == 200
    assert res.clustered is False


def test_small_samples_are_not_called_clustered(reg):
    """Below 30 trades the concentration test says nothing useful."""
    payloads = [{"k": 1}] * 10
    labels = [label(0.01) for _ in range(10)]

    res = run(labels, payloads, hypothesis="tiny", rationale="r", registry=reg)

    assert res.clustered is False
