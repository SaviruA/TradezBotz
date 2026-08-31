"""Tests for 13F holdings, 13D/G stakes, and filer ranking.

Two things carry the weight: that the *diff* between quarters is what gets
measured rather than the holding, and that a trailing-return ranking is checked
for persistence before it is believed. The second is the error the Ziobrowski
replications diagnosed -- an edge that exists only in the aggregate, driven by a
few large trades, and vanishes when members are weighted equally.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tradezbotz.research.holdings import (
    THIRTEEN_F_LAG_DAYS,
    Filing13F,
    FilerScore,
    Holding,
    Stake,
    persistence,
    position_changes,
    rank_by_trailing_return,
)

WHEN = datetime(2026, 5, 15, 18, 0, tzinfo=UTC)


def filing(holdings, cik="1045810", period=date(2026, 3, 31), when=WHEN):
    return Filing13F(accession="0001-26-1", filer_cik=cik, filer_name="NVIDIA CORP",
                     period_end=period, observed_at=when,
                     holdings=tuple(holdings))


def h(cusip, shares, value, issuer="ISS"):
    return Holding(cusip=cusip, issuer=issuer, shares=shares, value_usd=value)


# --- position diffs ------------------------------------------------------------------

def test_a_brand_new_position_is_distinguished_from_an_add():
    """The cloning literature puts the edge in high-conviction NEW positions,
    not in incremental adds, so collapsing them would discard the signal."""
    prev = filing([h("AAA", 100, 1000)])
    cur = filing([h("AAA", 150, 1500), h("BBB", 50, 500)])

    kinds = {c.cusip: c.kind for c in position_changes(prev, cur)}

    assert kinds["BBB"] == "new"
    assert kinds["AAA"] == "add"


def test_trims_and_exits_are_detected():
    prev = filing([h("AAA", 100, 1000), h("BBB", 80, 800)])
    cur = filing([h("AAA", 40, 400)])

    kinds = {c.cusip: c.kind for c in position_changes(prev, cur)}

    assert kinds["AAA"] == "trim"
    assert kinds["BBB"] == "exit"


def test_an_unchanged_position_carries_no_information():
    """A manager who has held a name for six quarters is telling us nothing new."""
    prev = filing([h("AAA", 100, 1000)])
    cur = filing([h("AAA", 100, 1100)])   # value moved, share count did not

    assert position_changes(prev, cur) == []


def test_the_first_filing_makes_everything_new():
    cur = filing([h("AAA", 100, 1000), h("BBB", 50, 500)])

    changes = position_changes(None, cur)

    assert {c.kind for c in changes} == {"new"}
    assert len(changes) == 2


def test_weight_measures_conviction():
    """A 15% position is a statement; 0.1% is index-hugging."""
    cur = filing([h("BIG", 100, 9000), h("SMALL", 10, 1000)])

    by = {c.cusip: c.weight for c in position_changes(None, cur)}

    assert by["BIG"] == pytest.approx(0.9)
    assert by["SMALL"] == pytest.approx(0.1)


def test_delta_is_signed():
    prev = filing([h("AAA", 100, 1000)])
    cur = filing([h("AAA", 40, 400)])

    assert position_changes(prev, cur)[0].delta == -60


# --- the lag -------------------------------------------------------------------------

def test_the_45_day_lag_is_measured_not_assumed():
    """Any strategy on 13F has to reason about staleness explicitly."""
    f = filing([h("AAA", 1, 1)], period=date(2026, 3, 31),
               when=datetime(2026, 5, 15, tzinfo=UTC))

    assert f.lag_days == 45
    assert THIRTEEN_F_LAG_DAYS == 45


def test_events_are_emitted_per_position():
    """One event per holding, or it could never be joined to a symbol."""
    events = list(filing([h("AAA", 1, 600), h("BBB", 2, 400)]).to_events())

    assert len(events) == 2
    assert {e.payload["cusip"] for e in events} == {"AAA", "BBB"}
    assert events[0].payload["weight"] == pytest.approx(0.6)


def test_event_ids_do_not_collide_across_positions():
    events = list(filing([h("AAA", 1, 1), h("BBB", 2, 2)]).to_events())

    assert events[0].external_id != events[1].external_id


def test_holding_events_are_point_in_time():
    for e in filing([h("AAA", 1, 1)]).to_events():
        assert e.occurred_at <= e.observed_at
        assert e.observed_at.tzinfo is not None


# --- 13D vs 13G ----------------------------------------------------------------------

def test_13d_signals_intent_and_13g_does_not():
    d = Stake("a", "1", "ACTIVIST LP", "TARGET INC", "SC 13D", WHEN)
    g = Stake("b", "1", "INDEX FUND", "TARGET INC", "SC 13G", WHEN)

    assert d.activist is True
    assert g.activist is False


def test_an_amendment_is_flagged():
    s = Stake("a", "1", "F", "T", "SC 13D/A", WHEN)

    assert s.to_event().payload["amendment"] is True
    assert s.activist is True, "an amended 13D is still a 13D"


# --- ranking and, crucially, persistence -------------------------------------------

def score(cik, ret, n=50):
    return FilerScore(cik, f"F{cik}", n, ret, date(2020, 1, 1), date(2022, 1, 1))


def test_ranking_orders_by_return():
    out = rank_by_trailing_return([score("a", 0.05), score("b", 0.20),
                                   score("c", 0.10)])

    assert [s.filer_cik for s in out] == ["b", "c", "a"]


def test_thin_filers_are_excluded_from_the_ranking():
    """A filer with three positions can top a leaderboard on luck alone, and a
    ranking dominated by tiny samples is how the aggregate Ziobrowski result
    appeared while the equal-weighted one did not."""
    out = rank_by_trailing_return([score("lucky", 0.90, n=3), score("real", 0.10, n=50)])

    assert [s.filer_cik for s in out] == ["real"]


def test_persistence_detects_a_ranking_that_holds():
    early = [score(c, r) for c, r in [("a", .3), ("b", .2), ("c", .1), ("d", .0)]]
    late = [score(c, r) for c, r in [("a", .28), ("b", .18), ("c", .09), ("d", .01)]]

    out = persistence(early, late, top_n=2)

    assert out["top_n_overlap_rate"] == 1.0
    assert out["rank_correlation"] > 0.8
    assert out["edge_over_field"] > 0


def test_persistence_exposes_a_ranking_that_is_noise():
    """The check that has to pass before any ranking is trusted: if the early
    leaders do not lead later, the leaderboard recorded luck."""
    early = [score(c, r) for c, r in [("a", .3), ("b", .2), ("c", .1), ("d", .0)]]
    late = [score(c, r) for c, r in [("a", .0), ("b", .1), ("c", .2), ("d", .3)]]

    out = persistence(early, late, top_n=2)

    assert out["top_n_overlap_rate"] == 0.0
    assert out["rank_correlation"] < 0
    assert out["edge_over_field"] < 0, "following the early leaders lost to the field"


def test_persistence_reports_what_following_would_have_earned():
    early = [score(c, r) for c, r in [("a", .5), ("b", .1)]]
    late = [score(c, r) for c, r in [("a", .04), ("b", .06)]]

    out = persistence(early, late, top_n=1)

    assert out["followed_mean_return"] == pytest.approx(0.04)
    assert out["all_mean_return"] == pytest.approx(0.05)


def test_persistence_survives_no_overlap():
    early = [score("a", .3), score("b", .2)]
    late = [score("x", .1), score("y", .2)]

    out = persistence(early, late)

    assert out["shared_filers"] == 0
    assert out["rank_correlation"] == 0.0


# --- House PTR parsing ----------------------------------------------------------

from tradezbotz.research.holdings import (  # noqa: E402
    CongressTrade, HouseIndexRow, parse_amount, parse_house_ptr, _reflow,
)

ROW = HouseIndexRow(last="Allen", first="Richard W.", filing_type="P",
                    state_district="GA12", year="2025",
                    filed=date(2025, 1, 16), doc_id="20026537")

# Exactly how the extractor emits it: ticker and type on separate lines, and the
# amount bracket wrapped mid-range.
REAL_SHAPE = """Name: Hon. Richard W. Allen
SP Rollins, Inc. Common Stock (ROL)
[ST]
P 12/12/2024 01/08/2025 $15,001 -
$50,000
"""


def test_wrapped_records_are_reflowed_before_parsing():
    """Both live defects in one fixture: without reflow the ticker is missed and
    a $15,001-$50,000 trade is recorded as $15,001-$15,001."""
    trades = parse_house_ptr(REAL_SHAPE, ROW)

    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "ROL"
    assert (t.amount_low, t.amount_high) == (15001.0, 50000.0)


def test_ownership_is_extracted():
    """A member's own account is a stronger signal than a spouse's."""
    assert parse_house_ptr(REAL_SHAPE, ROW)[0].owner == "spouse"


def test_self_is_the_default_owner():
    text = "Apple Inc (AAPL)\n[ST]\nP 01/02/2025 01/10/2025 $1,001 - $15,000\n"

    assert parse_house_ptr(text, ROW)[0].owner == "self"


def test_treasuries_yield_no_ticker_rather_than_a_wrong_one():
    """Government securities carry a CUSIP, not a symbol. Half of a typical
    filing is these, and inventing a ticker would be worse than omitting one."""
    text = "US T NOTE 12/15/26 (91282CJP7)\n[GS]\nP 12/03/2024 12/20/2024 $100,001 - $250,000\n"

    t = parse_house_ptr(text, ROW)[0]

    assert t.symbol is None
    assert t.asset_type == "GS"


def test_purchases_and_sales_are_distinguished():
    buy = "X Corp (XXX)\n[ST]\nP 01/02/2025 01/10/2025 $1,001 - $15,000\n"
    sell = "X Corp (XXX)\n[ST]\nS 01/02/2025 01/10/2025 $1,001 - $15,000\n"

    assert parse_house_ptr(buy, ROW)[0].is_purchase is True
    assert parse_house_ptr(sell, ROW)[0].is_purchase is False


def test_a_partial_sale_is_still_a_sale():
    text = "X Corp (XXX)\n[ST]\nS (partial) 01/02/2025 01/10/2025 $1,001 - $15,000\n"

    trades = parse_house_ptr(text, ROW)

    assert len(trades) == 1 and trades[0].code == "S"


def test_unparseable_lines_are_dropped_not_guessed():
    """These are disclosures. A wrong ticker is worse than a missing row."""
    assert parse_house_ptr("garbage\nmore garbage\n", ROW) == []


def test_amount_is_a_bracket_never_a_midpoint():
    """The filing discloses a range; collapsing it to one number would invent
    precision that was never disclosed."""
    assert parse_amount("$1,001 - $15,000") == (1001.0, 15000.0)
    assert parse_amount("$15,000") == (15000.0, 15000.0)
    assert parse_amount("no numbers here") == (0.0, 0.0)


def test_reflow_joins_only_continuations():
    out = _reflow(["P 01/02/2025 01/10/2025 $15,001 -", "$50,000", "Next Asset (AAA)"])

    assert out == ["P 01/02/2025 01/10/2025 $15,001 - $50,000", "Next Asset (AAA)"]


# --- the point-in-time rule for congress -------------------------------------------

def test_observed_at_is_the_filing_date_not_the_trade_date():
    """The transaction is up to 45 days older than the disclosure. Using the
    trade date would be lookahead of exactly the kind the STOCK Act lag creates."""
    t = parse_house_ptr(REAL_SHAPE, ROW)[0]
    ev = t.to_event()

    assert ev.observed_at.date() == date(2025, 1, 16), "filing date"
    assert ev.occurred_at.date() == date(2024, 12, 12), "trade date"
    assert ev.occurred_at < ev.observed_at


def test_disclosure_lag_is_measured():
    assert parse_house_ptr(REAL_SHAPE, ROW)[0].disclosure_lag_days == 35


def test_congress_event_ids_do_not_collide():
    """One filing can disclose several trades in the same name on different
    dates, and two members can file the same ticker the same day."""
    a = CongressTrade("D1", "M", "GA12", "AAA", "A", "P", "self",
                      date(2025, 1, 2), date(2025, 2, 1), 1.0, 2.0).to_event()
    b = CongressTrade("D1", "M", "GA12", "AAA", "A", "P", "self",
                      date(2025, 1, 3), date(2025, 2, 1), 1.0, 2.0).to_event()
    c = CongressTrade("D2", "M", "GA12", "AAA", "A", "P", "self",
                      date(2025, 1, 2), date(2025, 2, 1), 1.0, 2.0).to_event()

    assert len({a.external_id, b.external_id, c.external_id}) == 3


def test_only_ptr_rows_carry_transactions():
    annual = HouseIndexRow("X", "Y", "D", "GA12", "2025", date(2025, 1, 1), "1")
    ptr = HouseIndexRow("X", "Y", "P", "GA12", "2025", date(2025, 1, 1), "2")

    assert annual.is_ptr is False
    assert ptr.is_ptr is True


def test_a_trade_dated_after_its_own_disclosure_is_dropped():
    """Real filings contain these: a live 2025 filing disclosed a trade dated
    2026-12-26, because members type the dates by hand. The event store rejects
    such a row outright, so one typo would otherwise end a whole ingest -- and
    a date known to be wrong cannot be used to place an entry anyway."""
    text = "X Corp (XXX)\n[ST]\nP 12/26/2026 12/28/2026 $1,001 - $15,000\n"

    assert parse_house_ptr(text, ROW) == []


def test_a_normal_backdated_trade_still_parses():
    """The guard must reject only the impossible direction."""
    text = "X Corp (XXX)\n[ST]\nP 12/12/2024 01/08/2025 $1,001 - $15,000\n"

    assert len(parse_house_ptr(text, ROW)) == 1
