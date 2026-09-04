"""Tests for portfolio construction.

Everything upstream is an event study: every qualifying event is a trade, no
capital constraint, measured concurrency of 591 distinct symbols at h=5 against
an assumed 10. That is not a strategy, and a capacity constraint is not a
detail bolted on at the end -- it changes which trades you take and therefore
what you measured.

The properties pinned here are the ones that would let a capped backtest
flatter itself: taking a trade with no free slot, re-entering a name already
held, releasing a slot early, and -- most importantly -- reporting a ranked
result without its arrival-order control.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.portfolio import compare, simulate


def _label(symbol, day, ret=0.01, horizon=5):
    return Label(symbol=symbol, observed_at=datetime(2020, 1, 1, tzinfo=UTC),
                 entry_day=day, entry_price=10.0, returns={horizon: ret},
                 coverage=Coverage.COMPLETE)


def _signals(n, day=date(2020, 1, 1), ret=0.01, horizon=5):
    """n signals all firing on the same day."""
    labels = [_label(f"S{i}", day, ret, horizon) for i in range(n)]
    return [{"i": i} for i in range(n)], labels


# --- the capacity constraint binds -----------------------------------------

def test_only_the_capacity_is_taken_and_the_rest_are_refused():
    payloads, labels = _signals(50)

    out = simulate(payloads, labels, 5, max_positions=10)

    assert out.n_taken == 10
    assert out.n_rejected_no_slot == 40
    assert out.n_signals == 50


def test_participation_is_reported_because_it_changes_the_strategy():
    """A strategy taking 5% of its signals is not the one measured upstream,
    however similar the per-trade return looks."""
    payloads, labels = _signals(200)

    out = simulate(payloads, labels, 5, max_positions=10)

    assert out.participation == pytest.approx(0.05)
    assert "took 10 of 200 signals (5.0%)" in out.describe()


def test_a_slot_frees_only_after_the_holding_period():
    """Two signals a day apart with a 5-session hold: the second must wait."""
    payloads = [{}, {}]
    labels = [_label("A", date(2020, 1, 1)), _label("B", date(2020, 1, 2))]

    out = simulate(payloads, labels, 5, max_positions=1)

    assert out.n_taken == 1
    assert out.n_rejected_no_slot == 1


def test_a_slot_freed_by_an_exit_is_reused():
    payloads = [{}, {}]
    labels = [_label("A", date(2020, 1, 1)), _label("B", date(2020, 3, 1))]

    out = simulate(payloads, labels, 5, max_positions=1)

    assert out.n_taken == 2
    assert out.n_rejected_no_slot == 0


def test_a_symbol_already_held_is_not_re_entered():
    """Doubling into a name is a different strategy with different risk, and
    allowing it silently would let one issuer consume the whole book."""
    payloads = [{}, {}]
    labels = [_label("A", date(2020, 1, 1)), _label("A", date(2020, 1, 2))]

    out = simulate(payloads, labels, 5, max_positions=10)

    assert out.n_taken == 1
    # Refused for being a duplicate, not for want of a slot -- nine were free.
    assert out.n_rejected_no_slot == 0


def test_peak_concurrency_never_exceeds_the_cap():
    payloads, labels = _signals(100)

    out = simulate(payloads, labels, 20, max_positions=7)

    assert out.peak_concurrent <= 7


# --- the ranking, and its control ------------------------------------------

def test_ranking_takes_the_best_signals_when_slots_are_scarce():
    payloads = [{"score": 0.0}, {"score": 9.0}]
    labels = [_label("A", date(2020, 1, 1), ret=0.01),
              _label("B", date(2020, 1, 1), ret=0.50)]

    out = simulate(payloads, labels, 5, max_positions=1,
                   rank=lambda p, l: p["score"])

    assert out.returns == [0.50]


def test_arrival_order_is_the_default_and_ignores_the_score():
    payloads = [{"score": 0.0}, {"score": 9.0}]
    labels = [_label("A", date(2020, 1, 1), ret=0.01),
              _label("B", date(2020, 1, 1), ret=0.50)]

    out = simulate(payloads, labels, 5, max_positions=1)

    assert out.returns == [0.01]


def test_a_ranking_is_always_reported_against_arrival_order():
    """The only comparison that isolates the ranking. Against the uncapped
    study a ranked portfolio differs in two ways at once -- fewer trades AND a
    different selection -- so attributing the difference to skill is
    unfounded."""
    payloads = [{"score": float(i)} for i in range(20)]
    labels = [_label(f"S{i}", date(2020, 1, 1), ret=i / 100.0)
              for i in range(20)]

    text = compare(payloads, labels, 5, max_positions=3,
                   rank=lambda p, l: p["score"], rank_name="conviction")

    assert "conviction" in text
    assert "arrival order" in text
    assert "worth" in text


def test_a_worthless_ranking_is_called_out_rather_than_reported_neutrally():
    payloads = [{"score": -float(i)} for i in range(20)]
    labels = [_label(f"S{i}", date(2020, 1, 1), ret=i / 100.0)
              for i in range(20)]

    text = compare(payloads, labels, 5, max_positions=3,
                   rank=lambda p, l: p["score"])

    assert "carries nothing here" in text


def test_equal_scores_keep_arrival_order():
    """An unranked tie must not depend on dict iteration order, or the result
    changes between runs for no reason."""
    payloads = [{"score": 1.0} for _ in range(5)]
    labels = [_label(f"S{i}", date(2020, 1, 1), ret=i / 100.0)
              for i in range(5)]

    out = simulate(payloads, labels, 5, max_positions=2,
                   rank=lambda p, l: p["score"])

    assert out.returns == [0.0, 0.01]


# --- what is not eligible ---------------------------------------------------

def test_a_label_without_that_horizon_is_not_a_signal():
    payloads = [{}]
    labels = [_label("A", date(2020, 1, 1), horizon=5)]

    assert simulate(payloads, labels, 20, max_positions=10).n_signals == 0


def test_a_label_without_a_symbol_cannot_occupy_a_slot():
    payloads = [{}]
    labels = [_label("", date(2020, 1, 1))]

    assert simulate(payloads, labels, 5, max_positions=10).n_signals == 0


def test_an_empty_population_is_not_a_crash():
    out = simulate([], [], 5, max_positions=10)

    assert out.n_taken == 0
    assert out.participation == 0.0
    assert out.mean_return == 0.0


def test_a_zero_capacity_is_refused_rather_than_silently_taking_nothing():
    payloads, labels = _signals(5)

    with pytest.raises(ValueError, match="must be positive"):
        simulate(payloads, labels, 5, max_positions=0)
