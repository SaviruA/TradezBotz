"""Tests for the concentration decomposition and skewness-adjusted inference.

Our strongest row reported a mean of +11.61% against a winsorised +2.65%. Those
describe different strategies, and no gate in the sweep separates them: a
significance test asks whether the mean differs from zero, never whether five
trades produced it.

The inference half exists because at a 60-session horizon skewness is the
dominant feature rather than a nuisance. Barber & Lyon (1997) and Lyon, Barber
& Tsai (1999) established that long-horizon buy-and-hold abnormal returns are
strongly positively skewed, that this biases the conventional t DOWNWARD, and
that the remedy is a skewness-adjusted bootstrapped t. The bias runs in our
favour, which is exactly why using the right test matters rather than keeping
the flattering one.
"""

from __future__ import annotations

import math

import pytest

from tradezbotz.research.concentration import (
    CONCENTRATED_ABOVE,
    analyse,
)


# --- concentration ----------------------------------------------------------

def test_a_handful_of_winners_carrying_the_mean_is_flagged():
    """The case this exists for: 95 flat trades and 5 multi-baggers produce a
    healthy mean and a strategy that is really five lottery tickets."""
    returns = [0.0] * 95 + [3.0] * 5

    out = analyse(returns)

    assert out.top_share[5] == pytest.approx(1.0)
    assert out.concentrated is True
    assert "CONCENTRATED" in out.describe()


def test_a_broad_edge_is_not_flagged():
    returns = [0.02] * 500

    out = analyse(returns)

    assert out.top_share[5] < CONCENTRATED_ABOVE
    assert out.concentrated is False
    assert "CONCENTRATED" not in out.describe()


def test_the_median_exposes_what_the_mean_hides():
    returns = [0.0] * 95 + [3.0] * 5

    out = analyse(returns)

    assert out.mean > 0.14
    assert out.median == 0.0


def test_the_trimmed_mean_sits_between_mean_and_median():
    returns = [0.01] * 90 + [2.0] * 10

    out = analyse(returns)

    assert out.median <= out.trimmed_mean <= out.mean


def test_the_positive_share_is_reported_separately_from_the_mean():
    """Hit rate and mean answer different questions, and a strategy that is
    right 20% of the time can still be excellent -- or a lottery."""
    out = analyse([0.0] * 80 + [1.0] * 20)

    assert out.positive_share == pytest.approx(0.20)


def test_a_non_positive_total_gives_no_share_rather_than_a_wild_ratio():
    """Against a total at or below zero the contribution ratio is undefined;
    reporting a number there would be arithmetic theatre."""
    out = analyse([-0.5] * 50 + [0.1] * 50)

    assert all(math.isnan(v) for v in out.top_share.values())


def test_too_few_trades_to_decompose_returns_nothing():
    assert analyse([0.01] * 9) is None


def test_an_empty_series_is_not_a_crash():
    assert analyse([]) is None


# --- skewness-adjusted inference -------------------------------------------

def test_the_skew_adjustment_raises_t_on_a_positively_skewed_series():
    """The published direction. Positive skew biases the conventional t
    downward, so the corrected statistic is larger -- and pretending otherwise
    would understate a real long-horizon effect."""
    import statistics

    returns = [0.0] * 90 + [1.0] * 10
    plain_t = (statistics.fmean(returns) /
               (statistics.pstdev(returns) / math.sqrt(len(returns))))

    out = analyse(returns)

    assert out.skew_adjusted_t > plain_t


def test_a_symmetric_series_is_barely_adjusted():
    """No skew, no correction -- otherwise the adjustment would be inventing
    significance rather than repairing a known bias."""
    import statistics

    returns = [0.01, -0.01] * 100 + [0.02] * 5
    plain_t = (statistics.fmean(returns) /
               (statistics.pstdev(returns) / math.sqrt(len(returns))))

    out = analyse(returns)

    assert abs(out.skew_adjusted_t - plain_t) < 0.5


def test_the_bootstrap_interval_brackets_the_mean():
    out = analyse([0.02 + 0.001 * i for i in range(200)])

    lo, hi = out.bootstrap_ci
    assert lo < out.mean < hi


def test_the_bootstrap_interval_is_wide_when_the_edge_is_a_few_names():
    """The interval is the honest summary: a mean carried by five trades is
    barely distinguishable from zero once resampled."""
    concentrated = analyse([0.0] * 195 + [4.0] * 5)
    broad = analyse([0.10] * 200)

    conc_width = concentrated.bootstrap_ci[1] - concentrated.bootstrap_ci[0]
    broad_width = broad.bootstrap_ci[1] - broad.bootstrap_ci[0]

    assert conc_width > broad_width * 10


def test_the_bootstrap_is_reproducible():
    """A diagnostic that moves between runs cannot be compared across runs."""
    xs = [0.01 * (i % 7) for i in range(300)]

    assert analyse(xs).bootstrap_ci == analyse(xs).bootstrap_ci


def test_a_zero_variance_series_does_not_divide_by_zero():
    out = analyse([0.05] * 50)

    assert out.skew_adjusted_t == 0.0


# --- it reaches the report --------------------------------------------------

def test_the_description_cites_its_source():
    """A correction this consequential must carry its provenance."""
    assert "Lyon/Barber/Tsai 1999" in analyse([0.01] * 50).describe()


def test_a_backtest_result_carries_the_decomposition():
    from tradezbotz.research.backtest import BacktestResult

    r = BacktestResult(
        hypothesis="h", horizon=60, trial_id=1, n_events=10, n_trades=10,
        mean_return=0.1, median_return=0.1, stdev=0.1, hit_rate=0.5,
        sharpe_per_trade=0.1, sharpe_annualised=0.1, t_stat=1.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.5, n_trials=10, significant=False)

    assert r.concentration is None  # default; populated by `run`


# --- the verdict now measures rather than infers ----------------------------

def test_the_winsor_bound_scales_with_holding_period():
    """Welch's +/-15% describes SHORT horizon returns. Applied unscaled to a
    60-session hold it clips ordinary variation -- a 3-month microcap move past
    15% is unremarkable."""
    from tradezbotz.research.backtest import WINSOR_LIMIT, winsor_limit_for

    assert winsor_limit_for(5) == pytest.approx(WINSOR_LIMIT)
    assert winsor_limit_for(60) > winsor_limit_for(20) > winsor_limit_for(5)
    assert winsor_limit_for(1) < WINSOR_LIMIT


def test_a_degenerate_horizon_does_not_produce_a_zero_cap():
    """A zero cap would winsorise every return to zero and report every row as
    outlier-dependent."""
    from tradezbotz.research.backtest import WINSOR_LIMIT, winsor_limit_for

    assert winsor_limit_for(0) == WINSOR_LIMIT


def test_the_broad_row_and_the_lottery_row_now_get_different_answers():
    """The regression, in the terms the real run produced it. `buy + liquid`
    had a top-5 share of 13.8% and `buy + bb_below_lower` had 66.7%, and the
    old winsor-ratio proxy gave both the identical verdict."""
    from tradezbotz.research.backtest import BacktestResult

    def _result(returns):
        import statistics
        return BacktestResult(
            hypothesis="h", horizon=60, trial_id=1, n_events=len(returns),
            n_trades=len(returns), mean_return=statistics.fmean(returns),
            median_return=statistics.median(returns), stdev=0.1, hit_rate=0.5,
            sharpe_per_trade=0.1, sharpe_annualised=0.1, t_stat=1.0, skew=0.0,
            kurtosis=3.0, deflated_sharpe=0.5, n_trials=10, significant=False,
            mean_return_winsorised=0.02,
            concentration=analyse(returns))

    broad = _result([0.05] * 200)
    lottery = _result([0.0] * 195 + [4.0] * 5)

    assert broad.outlier_dependent is False
    assert lottery.outlier_dependent is True


def test_a_result_too_small_to_decompose_falls_back_to_the_winsor_ratio():
    """Below the decomposition floor there is nothing to measure, and the old
    proxy is better than no check at all."""
    from tradezbotz.research.backtest import BacktestResult

    r = BacktestResult(
        hypothesis="h", horizon=60, trial_id=1, n_events=5, n_trades=5,
        mean_return=0.10, median_return=0.10, stdev=0.1, hit_rate=0.5,
        sharpe_per_trade=0.1, sharpe_annualised=0.1, t_stat=1.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.5, n_trials=10, significant=False,
        mean_return_winsorised=0.02, concentration=None)

    assert r.outlier_dependent is True


def test_an_undefined_contribution_ratio_falls_back_rather_than_passing():
    """Against a non-positive total the share is NaN, and NaN must not be read
    as "not concentrated"."""
    from tradezbotz.research.backtest import BacktestResult

    losers = [-0.5] * 50 + [0.1] * 50
    r = BacktestResult(
        hypothesis="h", horizon=60, trial_id=1, n_events=100, n_trades=100,
        mean_return=-0.2, median_return=-0.2, stdev=0.1, hit_rate=0.5,
        sharpe_per_trade=0.1, sharpe_annualised=0.1, t_stat=1.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.5, n_trials=10, significant=False,
        mean_return_winsorised=-0.01, concentration=analyse(losers))

    # Falls through to the winsor ratio rather than silently returning False.
    assert r.outlier_dependent is True
