"""Tests for routine vs opportunistic insider classification."""

from __future__ import annotations

from datetime import date

import pytest

from tradezbotz.research.classify import (
    InsiderClass,
    PriorTrade,
    RoutineClassifier,
    score,
)


def trades(cik: str, *pairs: tuple[int, int]) -> list[PriorTrade]:
    return [PriorTrade(cik, date(year, month, 15)) for year, month in pairs]


def test_same_month_three_years_running_is_routine():
    """A December buyer every December is doing tax planning, not forecasting."""
    clf = RoutineClassifier()
    clf.add_history(trades("A", (2021, 12), (2022, 12), (2023, 12)))

    assert clf.classify("A", date(2024, 12, 10)) is InsiderClass.ROUTINE


def test_trade_off_the_usual_month_is_opportunistic():
    """Same insider, same history -- but a March buy breaks the pattern."""
    clf = RoutineClassifier()
    clf.add_history(trades("A", (2021, 12), (2022, 12), (2023, 12)))

    assert clf.classify("A", date(2024, 3, 10)) is InsiderClass.OPPORTUNISTIC


def test_broken_streak_is_not_routine():
    """2021 and 2023 with a gap is not an established calendar."""
    clf = RoutineClassifier()
    clf.add_history(trades("A", (2021, 12), (2023, 12), (2020, 6)))

    assert clf.classify("A", date(2024, 12, 10)) is InsiderClass.OPPORTUNISTIC


def test_thin_history_is_unknown_not_opportunistic():
    """The trap: treating first-time filers as opportunistic would flood the
    signal population with insiders we know nothing about."""
    clf = RoutineClassifier()
    clf.add_history(trades("A", (2023, 12)))

    assert clf.classify("A", date(2024, 3, 10)) is InsiderClass.UNKNOWN


def test_unseen_insider_is_unknown():
    assert RoutineClassifier().classify("NOBODY", date(2024, 3, 10)) is InsiderClass.UNKNOWN


def test_future_trades_in_same_month_do_not_count():
    """Only years strictly before the trade may establish the baseline."""
    clf = RoutineClassifier()
    clf.add_history(trades("A", (2024, 12), (2025, 12), (2026, 12), (2020, 1), (2021, 2)))

    # Classifying the 2024 December trade: prior Decembers are none.
    assert clf.classify("A", date(2024, 12, 10)) is InsiderClass.OPPORTUNISTIC


# --- scoring -----------------------------------------------------------------

def base_score(**overrides):
    kwargs = dict(
        symbol="AAPL",
        owner_cik="A",
        owner_name="DOE JANE",
        insider_class=InsiderClass.OPPORTUNISTIC,
        is_open_market_buy=True,
        is_officer=True,
        is_director=False,
        officer_title="Chief Executive Officer",
        notional=750_000.0,
    )
    kwargs.update(overrides)
    return score(**kwargs)


def test_non_purchase_scores_zero():
    """Grants, option exercises and tax withholding carry no conviction."""
    s = base_score(is_open_market_buy=False)

    assert s.conviction == 0.0
    assert "not an open-market purchase" in s.reasons


def test_opportunistic_ceo_buy_outscores_routine_equivalent():
    opportunistic = base_score(insider_class=InsiderClass.OPPORTUNISTIC)
    routine = base_score(insider_class=InsiderClass.ROUTINE)

    assert opportunistic.conviction > routine.conviction


def test_unknown_scores_between_routine_and_opportunistic():
    unknown = base_score(insider_class=InsiderClass.UNKNOWN)

    assert (
        base_score(insider_class=InsiderClass.ROUTINE).conviction
        < unknown.conviction
        < base_score(insider_class=InsiderClass.OPPORTUNISTIC).conviction
    )


def test_token_purchases_are_penalised():
    """A $3k buy by a CEO is optics, not a position."""
    assert base_score(notional=3_000.0).conviction < base_score(notional=750_000.0).conviction


def test_conviction_stays_bounded():
    strongest = base_score(notional=50_000_000.0)
    weakest = base_score(
        insider_class=InsiderClass.ROUTINE,
        is_officer=False,
        is_director=False,
        officer_title=None,
        notional=500.0,
    )

    assert 0.0 <= weakest.conviction <= strongest.conviction <= 1.0


def test_reasons_are_always_populated():
    assert base_score().reasons
