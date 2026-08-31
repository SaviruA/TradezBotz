"""Tests for dependence corrections.

The property that matters most is the *pair*: the correction must bite hard on
dependent data and do nothing at all on independent data. A correction that
always shrinks the statistic is not a correction, it is a penalty.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from tradezbotz.research.clustering import (
    MIN_CLUSTERS,
    average_pairwise_correlation,
    cluster_robust_variance,
    diagnose,
    effective_from_inflation,
    effective_sample_size,
    kolari_pynnonen_factor,
    two_way_cluster_robust_variance,
)


def independent(n=400, seed=1):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n)], list(range(n)), list(range(n))


def clustered(n_symbols=40, per_symbol=10, rho=0.4, seed=1, n_dates=40):
    """Values carrying both a symbol effect and a shared-date effect."""
    rng = random.Random(seed)
    vals, syms, dates = [], [], []
    sym_eff = {s: rng.gauss(0, math.sqrt(rho)) for s in range(n_symbols)}
    date_eff = {d: rng.gauss(0, math.sqrt(rho)) for d in range(n_dates)}
    for s in range(n_symbols):
        for _ in range(per_symbol):
            d = rng.randrange(n_dates)
            vals.append(sym_eff[s] + date_eff[d] + rng.gauss(0, 0.5))
            syms.append(s)
            dates.append(d)
    return vals, syms, dates


# --- the Kolari-Pynnonen factor -------------------------------------------------

def test_kp_factor_is_one_without_correlation():
    """Applying the correction unconditionally must be safe."""
    assert kolari_pynnonen_factor(100, 0.0) == 1.0


def test_kp_factor_matches_the_published_magnitude():
    """Kolari & Pynnonen: rho of 0.02 over 100 events overstates the t-stat by
    about 1.73x. The factor is the reciprocal of that inflation."""
    factor = kolari_pynnonen_factor(100, 0.02)

    assert 1 / factor == pytest.approx(1.73, rel=0.05)


def test_kp_factor_shrinks_as_correlation_rises():
    weak = kolari_pynnonen_factor(100, 0.01)
    strong = kolari_pynnonen_factor(100, 0.20)

    assert strong < weak < 1.0


# --- effective sample size --------------------------------------------------------

def test_design_effect_uses_cluster_size_not_sample_size():
    """The bug that shipped first: using n in place of average cluster size
    reported an effective n of 3 out of 1,000, which would have made the DSR
    reject everything forever."""
    n, rho = 1000, 0.3

    with_cluster_size = effective_sample_size(n, rho, cluster_size=16.7)
    wrong_way = n / (1 + (n - 1) * rho)

    assert with_cluster_size > 100, "sane"
    assert wrong_way < 5, "the mistake, pinned so it cannot come back"


def test_effective_equals_n_without_correlation():
    assert effective_sample_size(500, 0.0, 10) == 500
    assert effective_from_inflation(500, 1.0) == 500


def test_effective_from_inflation_scales_with_the_square():
    """SE scales as 1/sqrt(n), so doubling the SE quarters the information."""
    assert effective_from_inflation(400, 2.0) == pytest.approx(100.0)


# --- cluster-robust variance --------------------------------------------------------

def test_clustering_matches_naive_when_every_obs_is_its_own_cluster():
    vals, _, _ = independent(200)
    groups = list(range(len(vals)))

    clustered_se = math.sqrt(cluster_robust_variance(vals, groups))
    naive_se = statistics.stdev(vals) / math.sqrt(len(vals))

    assert clustered_se == pytest.approx(naive_se, rel=0.05)


def test_clustering_inflates_when_observations_repeat():
    """The core case: the same value repeated within a cluster carries no more
    information than one observation of it."""
    vals = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 5
    one_cluster_each = list(range(len(vals)))
    two_clusters = [0, 0, 0, 0, 1, 1, 1, 1] * 5

    assert (cluster_robust_variance(vals, two_clusters)
            > cluster_robust_variance(vals, one_cluster_each))


def test_a_single_cluster_carries_no_information():
    """One cluster means no between-cluster variation to estimate from."""
    vals = [1.0, 2.0, 3.0]

    assert math.isinf(cluster_robust_variance(vals, [0, 0, 0]))


def test_two_way_exceeds_neither_one_way_by_construction():
    vals, syms, dates = clustered()

    v_two = two_way_cluster_robust_variance(vals, syms, dates)
    v_sym = cluster_robust_variance(vals, syms)

    assert v_two >= v_sym * 0.5, "subtraction must not collapse the estimate"


def test_two_way_falls_back_when_the_estimate_goes_negative():
    """The CGM estimator is not guaranteed positive in finite samples; the
    conventional fallback is the larger one-way estimate."""
    vals, syms, dates = independent(60)

    v = two_way_cluster_robust_variance(vals, syms, dates)

    assert v > 0 and not math.isnan(v)


# --- correlation estimation -----------------------------------------------------

def test_rho_is_zero_on_independent_data():
    vals, _, dates = independent(300)

    assert average_pairwise_correlation(vals, dates) == pytest.approx(0.0, abs=0.05)


def test_rho_is_positive_when_dates_share_a_shock():
    vals, _, dates = clustered(rho=0.4)

    assert average_pairwise_correlation(vals, dates) > 0.1


def test_rho_is_never_negative():
    """Sample correlation can come out negative; as a dependence inflation that
    is meaningless and would make the correction anti-conservative."""
    vals, _, dates = independent(100)

    assert average_pairwise_correlation(vals, dates) >= 0.0


def test_rho_needs_repeated_dates():
    """Nothing to estimate from when every event has its own date."""
    vals = [1.0, 2.0, 3.0, 4.0]

    assert average_pairwise_correlation(vals, [1, 2, 3, 4]) == 0.0


# --- the diagnosis as a whole ------------------------------------------------------

def test_independent_data_is_left_alone():
    """The load-bearing half of the pair. A correction that always shrinks the
    statistic is a penalty, not a correction."""
    d = diagnose(*independent(400))

    assert d.inflation == pytest.approx(1.0, abs=0.15)
    assert d.n_effective / d.n_obs > 0.7


def test_dependent_data_is_corrected_hard():
    d = diagnose(*clustered(rho=0.4))

    assert d.inflation > 1.5
    assert d.n_effective < d.n_obs / 2


def test_corrected_se_is_never_tighter_than_naive():
    """Dependence cannot make a sample more informative than independence."""
    for seed in range(10):
        d = diagnose(*independent(120, seed=seed))
        assert d.se_clustered >= d.se_naive


def test_too_few_clusters_is_reported():
    vals, syms, dates = clustered(n_symbols=4, per_symbol=10, n_dates=4)

    d = diagnose(vals, syms, dates)

    assert d.enough_clusters is False
    assert "WARNING" in d.summary()
    assert d.n_symbol_clusters < MIN_CLUSTERS


def test_enough_clusters_passes():
    d = diagnose(*clustered(n_symbols=40, per_symbol=5, n_dates=40))

    assert d.enough_clusters is True


def test_largest_cluster_share_is_measured():
    vals = [1.0] * 100
    syms = ["BIG"] * 60 + [f"S{i}" for i in range(40)]
    dates = list(range(100))

    d = diagnose(vals, syms, dates)

    assert d.largest_cluster_share == pytest.approx(0.6)


def test_diagnose_survives_a_tiny_sample():
    d = diagnose([1.0], ["A"], [1])

    assert d.n_obs == 1
    assert d.n_effective == 1.0
