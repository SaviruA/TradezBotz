"""Tests for the twelve Tigers raised by the 2026-09-01 pre-mortem.

Each test names the Tiger it closes. They are grouped here rather than scattered
into the module suites because the register is the thing being discharged, and a
reviewer checking "was T4 actually fixed" should not have to hunt.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.backtest import BacktestResult, everything, field_equals
from tradezbotz.research.costs import CostTable
from tradezbotz.research.edgar import candidate_symbols, normalise_symbol
from tradezbotz.research.features import FeatureBuilder
from tradezbotz.research.labeler import Coverage, Label
from tradezbotz.research.prices import Bar, Series
from tradezbotz.research.sweep import (
    MAX_FALLBACK_SHARE,
    MIN_COVERAGE,
    Candidate,
    Verdict,
    judge,
    report,
    sweep,
)
from tradezbotz.research.trials import TrialRegistry


@pytest.fixture
def reg(tmp_path):
    with TrialRegistry(tmp_path / "t.db") as r:
        yield r


def make(mean=0.02, net=0.01, trades=200, sig=True):
    return BacktestResult(
        hypothesis="h", horizon=5, trial_id=1, n_events=trades, n_trades=trades,
        mean_return=mean, median_return=mean, stdev=0.03, hit_rate=0.55,
        sharpe_per_trade=0.3, sharpe_annualised=2.0, t_stat=3.0, skew=0.0,
        kurtosis=3.0, deflated_sharpe=0.9, n_trials=10, significant=sig,
        n_symbols=60, mean_return_winsorised=mean, mean_return_net=net,
        costed=True, t_stat_clustered=2.5, n_effective=trades * 0.8,
        se_inflation=1.1)


# --- T1 / T2: the registry counts experiments, not executions ---------------

def test_t1_a_repeated_sweep_does_not_inflate_the_trial_count(reg):
    """The bar rose every night with no new hypothesis tested. ~200 trials per
    scheduled run, ~6,000 a month, and expected_max_sharpe grows in n_trials."""
    for _ in range(30):
        for cand in ("buy", "officer buy", "near_high"):
            reg.register(cand, "because", params={"horizon": 5},
                         split="train", dataset="fp-v1")

    assert reg.count() == 3, "N for the DSR is distinct experiments"
    assert reg.executions() == 90, "re-runs are still visible"


def test_t2_a_genuinely_new_dataset_does_count_as_a_new_trial(reg):
    """A second look at fresh data IS a second chance to get lucky, so the
    deduplication must not swallow it."""
    reg.register("buy", "because", params={"horizon": 5}, dataset="fp-v1")
    reg.register("buy", "because", params={"horizon": 5}, dataset="fp-v2")

    assert reg.count() == 2


def test_t2_differing_horizon_or_partition_are_different_trials(reg):
    reg.register("buy", "r", params={"horizon": 1}, dataset="fp")
    reg.register("buy", "r", params={"horizon": 5}, dataset="fp")
    reg.register("buy", "r", params={"horizon": 5}, split="validation",
                 dataset="fp")

    assert reg.count() == 3


def test_t1_opting_out_restores_append_behaviour(reg):
    """A caller with no stable notion of its dataset must not be silently
    deduplicated into one trial."""
    for _ in range(4):
        reg.register("buy", "r", params={"horizon": 5})

    assert reg.count() == 4


def test_t1_a_registry_predating_the_columns_still_opens(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT, hypothesis TEXT NOT NULL,
            rationale TEXT NOT NULL, params TEXT NOT NULL, split TEXT NOT NULL,
            sharpe REAL, n_obs INTEGER, n_trades INTEGER, skew REAL,
            kurtosis REAL, outcome TEXT NOT NULL, notes TEXT,
            created_at TEXT NOT NULL);
        CREATE TABLE holdout_access (
            id INTEGER PRIMARY KEY AUTOINCREMENT, hypothesis TEXT NOT NULL,
            reason TEXT NOT NULL, accessed_at TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO trials (hypothesis, rationale, params, split, outcome, "
        "created_at) VALUES ('old','r','{}','train','completed','2026-01-01')")
    conn.commit()
    conn.close()

    r = TrialRegistry(path)
    r.register("new", "r", params={"horizon": 5}, dataset="fp")
    n, runs = r.count(), r.executions()
    r.close()

    assert n == 2, "the pre-existing row survives and still counts toward N"
    assert runs == 2


# --- T4: the cost gate must be a measurement --------------------------------

def test_t4_a_result_costed_mostly_by_the_fallback_is_refused():
    verdict = judge(make(), None, fallback_share=0.9, coverage=1.0)

    assert verdict == Verdict.COST_NOT_MEASURED


def test_t4_a_result_costed_from_real_estimates_passes_the_gate():
    verdict = judge(make(), None, fallback_share=0.05, coverage=1.0)

    assert verdict == Verdict.KEEP


def test_t4_the_fallback_boundary_is_where_it_is_documented():
    assert judge(make(), None, fallback_share=MAX_FALLBACK_SHARE - 0.01,
                 coverage=1.0) == Verdict.KEEP
    assert judge(make(), None, fallback_share=MAX_FALLBACK_SHARE + 0.01,
                 coverage=1.0) == Verdict.COST_NOT_MEASURED


# --- T5: coverage gates before any statistic is read ------------------------

def test_t5_a_thinly_covered_population_is_refused_before_anything_else():
    """0.28% coverage was the local reality. Without this gate the first
    candidate to clear the trade floor is selected by data availability."""
    verdict = judge(make(), None, coverage=0.0028)

    assert verdict == Verdict.THIN_COVERAGE


def test_t5_coverage_is_checked_before_the_trade_count():
    """Ordering matters for the reader: 'thin coverage' tells you to fetch more
    data, 'too few trades' tells you the filter is too narrow."""
    thin_and_few = judge(make(trades=2), None, coverage=0.01)

    assert thin_and_few == Verdict.THIN_COVERAGE


def test_t5_adequate_coverage_passes():
    assert judge(make(), None, coverage=MIN_COVERAGE + 0.01) == Verdict.KEEP


# --- T5 / T9: the report states its own provenance --------------------------

def test_t9_the_report_carries_a_coverage_column_and_a_provenance_line(reg):
    labels = [Label(symbol=f"S{i%40}", observed_at=datetime(2025, 3, 4, tzinfo=UTC),
                    entry_day=date(2025, 1, 1) + timedelta(days=i % 100),
                    entry_price=10.0, returns={5: 0.01},
                    coverage=Coverage.COMPLETE) for i in range(200)]
    payloads = [{"code": "P"} for _ in labels]

    out = sweep([Candidate("a", everything, "r", controlled=False)],
                labels, payloads, registry=reg, horizons=(5,),
                coverage=0.05, fallback_share=0.8)
    text = report(out)

    assert "cov" in text
    assert "coverage 5.0%" in text
    assert "80% of trades charged the fallback" in text
    assert "nothing here is a finding" in text


# --- T8: dual-class filings expose every class ------------------------------

def test_t8_both_share_classes_are_recoverable():
    """normalise_symbol takes the first, which is wrong whenever the filer wrote
    the illiquid class first -- PARAA is the class A line, PARA is the one that
    trades."""
    assert candidate_symbols("PARAA,PARA") == ["PARAA", "PARA"]
    assert candidate_symbols("GEF,GEF.B") == ["GEF", "GEF.B"]
    assert candidate_symbols("MOGA/MOGB") == ["MOGA", "MOGB"]


def test_t8_the_first_candidate_still_matches_the_old_behaviour():
    """Callers with no liquidity data keep the previous answer."""
    for raw in ("PARAA,PARA", "GEF,GEF.B", "NYSE: KRC", "N O G", "AAPL"):
        assert candidate_symbols(raw)[0] == normalise_symbol(raw)


def test_t8_a_single_class_filing_yields_one_candidate():
    assert candidate_symbols("AAPL") == ["AAPL"]
    assert candidate_symbols("N/A") == []


# --- T10: a position size means impact is actually charged ------------------

class Bars:
    """Bars with a real price path.

    Constant closes give zero daily volatility, and market impact is
    proportional to volatility -- so a degenerate fixture would charge zero
    impact for any position size and the sizing test would pass vacuously
    whether or not the code worked.
    """

    def __init__(self, n=300, price=10.0, volume=100_000.0):
        import math
        self.bars = tuple(
            self._bar(date(2023, 1, 1) + timedelta(days=i),
                      price * (1.0 + 0.02 * math.sin(i * 0.7)), volume)
            for i in range(n))

    @staticmethod
    def _bar(day, p, volume):
        return Bar(day=day, open=p, high=p * 1.02, low=p * 0.98, close=p,
                   volume=volume)

    def get(self, symbol, start, end, basis=None):
        rows = tuple(b for b in self.bars if start <= b.day <= end)
        return Series(symbol=symbol, bars=rows, requested_start=start,
                      requested_end=end)


def label_at(day=date(2023, 9, 14), price=10.0):  # noqa: D401
    return Label(symbol="AAA", observed_at=datetime(2023, 9, 13, tzinfo=UTC),
                 entry_day=day, entry_price=price, returns={5: 0.01},
                 coverage=Coverage.COMPLETE)


def test_t10_an_unsized_table_charges_no_market_impact():
    """The prior behaviour, now labelled rather than silent: every net return
    was an upper bound because participation was always zero."""
    table = CostTable(Bars(), capital_per_trade=0.0)
    table(label_at())

    assert "UNSIZED" in table.summary()


def test_t10_a_sized_table_charges_more_than_an_unsized_one():
    small = CostTable(Bars(), capital_per_trade=0.0)
    large = CostTable(Bars(), capital_per_trade=5_000_000.0)

    assert large(label_at()) > small(label_at())


def test_t10_an_oversized_position_is_flagged_infeasible():
    """Half a billion dollars against 100k shares a day is not a fill."""
    table = CostTable(Bars(), capital_per_trade=500_000_000.0)
    table(label_at())

    assert table.infeasible == 1
    assert "not executable in one session" in table.summary()


def test_t4_fallback_rate_is_reported_separately_from_the_summary():
    """sweep.judge gates on this number, so it has to be readable without
    parsing prose."""
    table = CostTable(Bars(n=0))
    table(label_at())

    assert table.fallback_rate() == 1.0


# --- T11: the survivorship ratio carries its window -------------------------

def test_t11_survivorship_without_a_window_says_so():
    from tradezbotz.research.assets import DELISTED, LISTED, OTC, UNKNOWN, describe

    text = describe({LISTED: 80, DELISTED: 20, OTC: 0, UNKNOWN: 0})

    assert "WINDOW UNSTATED" in text
    assert "not a ratio over thirteen" in text


def test_t11_a_stated_window_is_printed_with_the_ratio():
    from tradezbotz.research.assets import DELISTED, LISTED, OTC, UNKNOWN, describe

    text = describe({LISTED: 80, DELISTED: 20, OTC: 0, UNKNOWN: 0},
                    window="2024-01-03 to 2026-08-28")

    assert "over 2024-01-03 to 2026-08-28" in text
    assert "WINDOW UNSTATED" not in text


# --- T12: memoisation is keyed on the day, not just the symbol --------------

class OneSymbol:
    """Rising prices, so features genuinely differ between entry days."""

    def get(self, symbol, start, end, basis=None):
        rows = tuple(
            Bar(day=date(2024, 1, 1) + timedelta(days=i), open=10.0 + i * 0.1,
                high=10.2 + i * 0.1, low=9.9 + i * 0.1, close=10.0 + i * 0.1,
                volume=1000.0)
            for i in range(400)
            if start <= date(2024, 1, 1) + timedelta(days=i) <= end)
        return Series(symbol=symbol, bars=rows, requested_start=start,
                      requested_end=end)


def test_t12_two_entry_days_on_one_symbol_get_different_features():
    """If the memo key were the symbol alone, every event on a name would share
    one feature set -- a bug that would produce coherent output on every row
    rather than an error anywhere."""
    builder = FeatureBuilder(OneSymbol())
    early = builder.features(Label(
        symbol="AAA", observed_at=datetime(2024, 5, 1, tzinfo=UTC),
        entry_day=date(2024, 5, 1), entry_price=10.0, returns={5: 0.01},
        coverage=Coverage.COMPLETE))
    late = builder.features(Label(
        symbol="AAA", observed_at=datetime(2024, 12, 1, tzinfo=UTC),
        entry_day=date(2024, 12, 1), entry_price=10.0, returns={5: 0.01},
        coverage=Coverage.COMPLETE))

    assert early and late
    assert early["distance_from_high"] != late["distance_from_high"]


# --- T4 regression: the gate must see the REAL fallback share ---------------

def test_fallback_rate_is_zero_before_anything_is_charged():
    """The bug the first CI run exposed. `fallback_rate()` reports 0.0 until
    charges happen, so passing it as an argument to sweep() evaluated it too
    early and the COST_NOT_MEASURED gate could never fire. The run printed
    '0% of trades charged the fallback' beside a summary saying 99.5%, and six
    candidates passed a cost gate that had not been applied."""
    table = CostTable(Bars(n=0))

    assert table.fallback_rate() == 0.0, "no charges yet"

    table(label_at())

    assert table.fallback_rate() == 1.0, "the rate only exists after charging"


def test_charging_every_label_first_yields_the_share_the_gate_needs():
    """The fix: a pre-pass over the labels, so the share is known before the
    sweep rather than after it."""
    table = CostTable(Bars(n=0))
    labels = [label_at(date(2023, 9, 5)), label_at(date(2023, 10, 5))]

    for lab in labels:
        table(lab)

    assert table.fallback_rate() > 0
    assert judge(make(), None, fallback_share=table.fallback_rate(),
                 coverage=1.0) == Verdict.COST_NOT_MEASURED


def test_the_report_shows_the_winsorised_mean_beside_the_raw_one(reg):
    """A +35% mean that winsorises to +3% is a few microcap moonshots, not a
    strategy. The gap is the finding, and a reader seeing only the raw column
    has no way to tell."""
    from datetime import UTC as _UTC

    labels = [Label(symbol=f"S{i%40}", observed_at=datetime(2025, 3, 4, tzinfo=_UTC),
                    entry_day=date(2025, 1, 1) + timedelta(days=i % 100),
                    entry_price=10.0, returns={5: (3.0 if i == 0 else 0.001)},
                    coverage=Coverage.COMPLETE) for i in range(200)]
    out = sweep([Candidate("a", everything, "r", controlled=False)],
                labels, [{} for _ in labels], registry=reg, horizons=(5,),
                coverage=0.8)
    text = report(out)

    assert "w-mean" in text
    raw = out[0].result.mean_return
    wins = out[0].result.mean_return_winsorised
    assert raw > wins * 3, "one 300% move should dominate the raw mean"
