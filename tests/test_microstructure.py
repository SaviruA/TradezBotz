"""Tests for volume profile and order flow.

Two things carry most of the weight here: that the value area follows the
Market Profile construction rather than a plausible-looking approximation of it,
and that merging sessions with different bucket widths does not manufacture a
point of control that no session actually had.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradezbotz.research.intraday import (
    EASTERN,
    MinuteBar,
    ProfileStore,
    SessionProfile,
    group_by_session,
)
from tradezbotz.research.microstructure import (
    above_poc,
    above_value_area,
    below_value_area,
    build_profile,
    compare_classifiers,
    cumulative_delta,
    delta_divergence,
    delta_ratio,
    in_low_volume_node,
    lee_ready,
    merge_profiles,
    positive_delta,
    tick_rule_delta,
    with_exact_flow,
)


def minutes(prices, volumes=None, day=date(2025, 3, 4), start_hour=14):
    """Minute bars at a flat price each, UTC (14:00Z is 09:00 ET winter)."""
    out = []
    for i, p in enumerate(prices):
        v = volumes[i] if volumes else 1_000.0
        ts = datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=UTC) + timedelta(minutes=i)
        out.append(MinuteBar(ts=ts, open=p, high=p, low=p, close=p, volume=v, vwap=p))
    return out


# --- session reduction --------------------------------------------------------

def test_build_profile_puts_volume_at_the_traded_price():
    bars = minutes([10.0, 20.0], volumes=[100.0, 300.0])

    p = build_profile("TEST", date(2025, 3, 4), bars)

    assert p.low == 10.0 and p.high == 20.0
    assert p.volume == 400.0
    assert p.vwap == pytest.approx((10 * 100 + 20 * 300) / 400)


def test_build_profile_uses_minute_vwap_not_close():
    """A minute that ran 10 to 11 traded across the range; assigning all of its
    volume to 11 walks the point of control toward wherever minutes ended."""
    bar = MinuteBar(ts=datetime(2025, 3, 4, 15, 0, tzinfo=UTC), open=10, high=11,
                    low=10, close=11, volume=1000, vwap=10.2)

    p = build_profile("TEST", date(2025, 3, 4), [bar])

    assert p.vwap == pytest.approx(10.2), "not 11.0"


def test_build_profile_returns_none_on_an_empty_session():
    assert build_profile("TEST", date(2025, 3, 4), []) is None


def test_build_profile_survives_a_flat_session():
    """A halted or one-price session has zero range; bucketing must not divide
    by zero."""
    p = build_profile("TEST", date(2025, 3, 4), minutes([5.0] * 10))

    assert p is not None
    assert sum(p.histogram) == pytest.approx(10_000.0)


# --- volume profile -----------------------------------------------------------

def test_poc_is_the_heaviest_price():
    bars = minutes([10.0, 11.0, 12.0], volumes=[100.0, 5_000.0, 100.0])
    p = build_profile("T", date(2025, 3, 4), bars)

    vp = merge_profiles([p])

    assert vp.poc == pytest.approx(11.0, abs=0.1)


def test_value_area_holds_about_seventy_percent():
    bars = minutes([10.0 + i * 0.1 for i in range(40)],
                   volumes=[100.0 + (20 - abs(20 - i)) * 50 for i in range(40)])
    vp = merge_profiles([build_profile("T", date(2025, 3, 4), bars)])

    inside = sum(v for i, v in enumerate(vp.buckets)
                 if vp.value_area_low <= vp.low + vp.bucket_width * (i + 0.5) <= vp.value_area_high)

    assert 0.60 <= inside / vp.total_volume <= 0.85


def test_value_area_brackets_the_poc():
    bars = minutes([10.0 + i * 0.1 for i in range(30)],
                   volumes=[100.0 + (15 - abs(15 - i)) * 80 for i in range(30)])
    vp = merge_profiles([build_profile("T", date(2025, 3, 4), bars)])

    assert vp.value_area_low <= vp.poc <= vp.value_area_high


def test_merging_preserves_total_volume():
    """Redistribution across a new grid must not create or destroy volume."""
    a = build_profile("T", date(2025, 3, 4), minutes([10.0, 11.0], volumes=[500.0, 500.0]))
    b = build_profile("T", date(2025, 3, 5), minutes([12.0, 20.0], volumes=[700.0, 300.0]))

    vp = merge_profiles([a, b])

    assert vp.total_volume == pytest.approx(2000.0)
    assert vp.sessions == 2


def test_merging_spans_the_union_of_ranges():
    a = build_profile("T", date(2025, 3, 4), minutes([10.0, 12.0]))
    b = build_profile("T", date(2025, 3, 5), minutes([30.0, 40.0]))

    vp = merge_profiles([a, b])

    assert vp.low == 10.0 and vp.high == 40.0


def test_merge_finds_the_heavier_session():
    thin = build_profile("T", date(2025, 3, 4), minutes([10.0], volumes=[100.0]))
    thick = build_profile("T", date(2025, 3, 5), minutes([50.0], volumes=[90_000.0]))

    vp = merge_profiles([thin, thick])

    assert vp.poc > 40.0, "the point of control follows the volume"


def test_merge_of_nothing_is_none():
    assert merge_profiles([]) is None


def test_position_of_locates_price_in_the_range():
    vp = merge_profiles([build_profile("T", date(2025, 3, 4), minutes([10.0, 20.0]))])

    assert vp.position_of(15.0) == pytest.approx(0.5)


# --- order flow ---------------------------------------------------------------

def test_tick_rule_signs_by_direction():
    bars = minutes([10.0, 11.0, 12.0], volumes=[100.0, 200.0, 300.0])

    delta, unsigned = tick_rule_delta(bars)

    assert delta == pytest.approx(500.0), "two up minutes"
    assert unsigned == pytest.approx(100.0), "the first has no prior tick"


def test_tick_rule_reports_unchanged_minutes_rather_than_guessing():
    """On thin names unchanged minutes are a large share of the session, and
    assigning them a side would overstate how much flow we actually observed."""
    bars = minutes([10.0, 10.0, 10.0], volumes=[100.0, 100.0, 100.0])

    delta, unsigned = tick_rule_delta(bars)

    assert delta == 0.0
    assert unsigned == pytest.approx(300.0)


def test_tick_rule_nets_out_a_round_trip():
    bars = minutes([10.0, 11.0, 10.0], volumes=[50.0, 200.0, 200.0])

    delta, _ = tick_rule_delta(bars)

    assert delta == pytest.approx(0.0)


def test_lee_ready_signs_against_the_midpoint():
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 10.10}]
    trades = [{"t": "2025-03-04T14:00:01Z", "p": 10.10, "s": 100},   # at ask -> buy
              {"t": "2025-03-04T14:00:02Z", "p": 10.00, "s": 40}]    # at bid -> sell

    delta, unsigned = lee_ready(trades, quotes)

    assert delta == pytest.approx(60.0)
    assert unsigned == 0.0


def test_lee_ready_falls_back_to_the_tick_rule_at_the_midpoint():
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 10.10}]
    trades = [{"t": "2025-03-04T14:00:01Z", "p": 10.05, "s": 100},   # midpoint, no prior
              {"t": "2025-03-04T14:00:02Z", "p": 10.05, "s": 50}]    # midpoint, unchanged

    delta, unsigned = lee_ready(trades, quotes)

    assert delta == 0.0
    assert unsigned == pytest.approx(150.0)


def test_lee_ready_uses_the_prevailing_quote_not_a_later_one():
    """Causality inside the session: a quote stamped after the print must not
    classify it."""
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 10.10},
              {"t": "2025-03-04T14:00:09Z", "bp": 20.0, "ap": 20.10}]
    trades = [{"t": "2025-03-04T14:00:05Z", "p": 10.10, "s": 100}]

    delta, _ = lee_ready(trades, quotes)

    assert delta == pytest.approx(100.0), "buy against the 10.05 midpoint"


def test_lee_ready_ignores_a_crossed_quote():
    """A crossed book (ask below bid) is bad data, not a signable midpoint."""
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 11.0, "ap": 10.0}]
    trades = [{"t": "2025-03-04T14:00:01Z", "p": 10.5, "s": 100}]

    delta, unsigned = lee_ready(trades, quotes)

    assert unsigned == pytest.approx(100.0)


def test_lee_ready_with_no_trades():
    assert lee_ready([], []) == (0.0, 0.0)


def test_compare_classifiers_flags_a_sign_disagreement():
    """The check that must run before any delta result is believed.

    Minute closes rise, so the tick rule reads net buying; every print sits below
    the quote midpoint, so Lee-Ready reads net selling. Both cover the same
    window, so the disagreement is real rather than a sampling artefact -- which
    is exactly what the live data showed on XELB and RCG.
    """
    bars = minutes([10.0, 11.0, 12.0], volumes=[100.0] * 3)       # tick rule: +200
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 14.0}]
    trades = [{"t": "2025-03-04T14:00:30Z", "p": 10.1, "s": 200},  # below mid -> sell
              {"t": "2025-03-04T14:02:30Z", "p": 10.2, "s": 200}]

    out = compare_classifiers(bars, trades, quotes)

    assert out["minute_delta"] > 0 and out["exact_delta"] < 0
    assert out["same_sign"] == 0.0


def test_compare_classifiers_agrees_when_it_should():
    bars = minutes([10.0, 11.0, 12.0], volumes=[100.0, 100.0, 100.0])
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 10.2}]
    trades = [{"t": "2025-03-04T14:00:30Z", "p": 10.2, "s": 500},
              {"t": "2025-03-04T14:02:30Z", "p": 10.2, "s": 500}]

    assert compare_classifiers(bars, trades, quotes)["same_sign"] == 1.0


def test_delta_ratio_is_scale_free():
    """A micro-cap and a mega-cap with the same imbalance must read the same."""
    small = SessionProfile("S", date(2025, 3, 4), 1, 2, 1_000, 1.5, (1_000.0,), 200.0, 0, 10)
    large = SessionProfile("L", date(2025, 3, 4), 1, 2, 10_000_000, 1.5,
                           (10_000_000.0,), 2_000_000.0, 0, 10)

    assert delta_ratio([small]) == pytest.approx(delta_ratio([large]))


def test_cumulative_delta_accumulates():
    ps = [SessionProfile("T", date(2025, 3, i + 1), 1, 2, 100, 1.5, (100.0,), d, 0, 10)
          for i, d in enumerate([10.0, -4.0, 6.0])]

    assert cumulative_delta(ps) == pytest.approx([10.0, 6.0, 12.0])


def test_delta_divergence_needs_price_down_and_flow_not():
    def p(day, low, delta):
        return SessionProfile("T", date(2025, 3, day), low, low + 1, 100, low + 0.5,
                              (100.0,), delta, 0, 10)

    # price makes lower lows; cumulative delta does not
    diverging = [p(1, 10, -50.0), p(2, 9, -50.0), p(3, 8, 60.0), p(4, 7, 60.0)]
    confirming = [p(1, 10, -10.0), p(2, 9, -10.0), p(3, 8, -60.0), p(4, 7, -60.0)]

    assert delta_divergence(diverging) is True
    assert delta_divergence(confirming) is False


def test_delta_divergence_needs_enough_history():
    assert delta_divergence([]) is False


# --- selectors ----------------------------------------------------------------

def profiles_10_to_20():
    """A 10-20 session with volume peaked at 15.

    Deliberately not uniform: a flat distribution has no meaningful point of
    control, and testing selectors against one measures bucket rounding rather
    than the selector.
    """
    prices = [10.0 + i * 0.25 for i in range(41)]
    volumes = [100.0 + (20 - abs(20 - i)) * 500 for i in range(41)]
    return [build_profile("T", date(2025, 3, 4), minutes(prices, volumes=volumes))]


def test_above_poc_selector():
    ps = profiles_10_to_20()

    assert above_poc(ps, 19.0) is True
    assert above_poc(ps, 11.0) is False


def test_value_area_selectors_are_mutually_exclusive():
    ps = profiles_10_to_20()

    assert below_value_area(ps, 10.0) is True
    assert above_value_area(ps, 10.0) is False
    assert above_value_area(ps, 20.0) is True
    assert below_value_area(ps, 20.0) is False


def test_low_volume_node_selector():
    """Heavy trade at 10, a thin tail up to 20."""
    bars = minutes([10.0] * 40 + [10.0 + i * 0.25 for i in range(1, 41)],
                   volumes=[10_000.0] * 40 + [1.0] * 40)
    ps = [build_profile("T", date(2025, 3, 4), bars)]

    assert in_low_volume_node(ps, 19.0) is True
    assert in_low_volume_node(ps, 10.0) is False


def test_positive_delta_selector():
    heavy = SessionProfile("T", date(2025, 3, 4), 1, 2, 1_000, 1.5, (1_000.0,), 500.0, 0, 10)
    light = SessionProfile("T", date(2025, 3, 4), 1, 2, 1_000, 1.5, (1_000.0,), 10.0, 0, 10)

    assert positive_delta([heavy]) is True
    assert positive_delta([light]) is False


def test_selectors_are_safe_with_no_profiles():
    assert above_poc([], 10.0) is False
    assert below_value_area([], 10.0) is False
    assert in_low_volume_node([], 10.0) is False
    assert positive_delta([]) is False


# --- session grouping and storage ---------------------------------------------

def test_group_by_session_uses_eastern_not_utc():
    """A 20:30 ET bar is 00:30 UTC the next day. Grouping on UTC would split the
    session and merge halves of different days."""
    late = MinuteBar(ts=datetime(2025, 3, 5, 0, 30, tzinfo=UTC), open=1, high=1,
                     low=1, close=1, volume=10)
    early = MinuteBar(ts=datetime(2025, 3, 4, 15, 0, tzinfo=UTC), open=1, high=1,
                      low=1, close=1, volume=10)

    grouped = group_by_session([late, early])

    assert set(grouped) == {date(2025, 3, 4)}, "both belong to the 4th ET"


def test_regular_hours_filter():
    pre = MinuteBar(ts=datetime(2025, 3, 4, 13, 0, tzinfo=UTC), open=1, high=1,
                    low=1, close=1, volume=1)      # 08:00 ET
    rth = MinuteBar(ts=datetime(2025, 3, 4, 15, 0, tzinfo=UTC), open=1, high=1,
                    low=1, close=1, volume=1)      # 10:00 ET

    assert pre.in_regular_hours() is False
    assert rth.in_regular_hours() is True


def test_store_round_trips_a_profile(tmp_path):
    store = ProfileStore(tmp_path / "profiles.db")
    p = build_profile("TEST", date(2025, 3, 4), minutes([10.0, 11.0], volumes=[1.0, 2.0]))

    store.put(p)
    back = store.get("TEST", date(2025, 3, 4))

    assert back.symbol == "TEST"
    assert back.volume == pytest.approx(p.volume)
    assert back.histogram == pytest.approx(p.histogram)
    assert back.delta == pytest.approx(p.delta)
    store.close()


def test_store_distinguishes_empty_from_never_fetched(tmp_path):
    """Without this split a backfill re-requests every empty day forever."""
    store = ProfileStore(tmp_path / "profiles.db")

    assert store.was_fetched("QUIET", date(2025, 3, 4)) is False
    store.mark_fetched("QUIET", date(2025, 3, 4))

    assert store.was_fetched("QUIET", date(2025, 3, 4)) is True
    assert store.get("QUIET", date(2025, 3, 4)) is None
    store.close()


def test_store_range_is_ordered(tmp_path):
    store = ProfileStore(tmp_path / "profiles.db")
    for d in (7, 5, 6):
        store.put(build_profile("T", date(2025, 3, d), minutes([10.0])))

    got = store.range("T", date(2025, 3, 1), date(2025, 3, 31))

    assert [p.day.day for p in got] == [5, 6, 7]
    store.close()


# --- timestamp parsing --------------------------------------------------------

def test_parse_ts_handles_nanoseconds():
    from tradezbotz.research.microstructure import parse_ts

    ts = parse_ts("2026-08-24T13:19:58.267405182Z")

    assert ts.year == 2026 and ts.microsecond == 267405


def test_parse_ts_orders_a_bare_second_before_its_fraction():
    """The reason string comparison cannot be used: '.' sorts below 'Z', so
    plain '...:58Z' would compare as later than '...:58.267Z'."""
    from tradezbotz.research.microstructure import parse_ts

    bare = parse_ts("2026-08-24T13:19:58Z")
    fractional = parse_ts("2026-08-24T13:19:58.267405182Z")

    assert bare < fractional
    assert "2026-08-24T13:19:58Z" > "2026-08-24T13:19:58.267405182Z", "the trap"


def test_lee_ready_pairs_a_bare_second_trade_with_the_right_quote():
    """Regression: with string ordering this trade picked up the later quote."""
    quotes = [{"t": "2026-08-24T13:19:58.100000000Z", "bp": 10.0, "ap": 10.2},
              {"t": "2026-08-24T13:19:59.000000000Z", "bp": 50.0, "ap": 50.2}]
    trades = [{"t": "2026-08-24T13:19:58.500000000Z", "p": 10.2, "s": 100}]

    delta, unsigned = lee_ready(trades, quotes)

    assert delta == pytest.approx(100.0), "buy against the 10.10 midpoint"
    assert unsigned == 0.0


def test_compare_classifiers_aligns_the_window():
    """A truncated trade page must not be compared against a whole session of
    bars -- that reports a sampling artefact as a disagreement."""
    bars = minutes([10.0, 11.0, 12.0, 1.0], volumes=[100.0] * 4)
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 10.2}]
    trades = [{"t": "2025-03-04T14:00:30Z", "p": 10.2, "s": 100},
              {"t": "2025-03-04T14:01:30Z", "p": 10.2, "s": 100}]

    out = compare_classifiers(bars, trades, quotes)

    assert out["minute_volume"] == pytest.approx(200.0), "the 3rd and 4th bars are outside"


# --- classifier provenance ----------------------------------------------------

def test_build_profile_records_the_cheap_classifier():
    p = build_profile("T", date(2025, 3, 4), minutes([10.0, 11.0]))

    assert p.flow_method == "tick_minute"


def test_with_exact_flow_replaces_the_delta_and_the_label():
    p = build_profile("T", date(2025, 3, 4), minutes([10.0, 11.0], volumes=[100.0, 100.0]))
    quotes = [{"t": "2025-03-04T14:00:00Z", "bp": 10.0, "ap": 12.0}]
    trades = [{"t": "2025-03-04T14:00:30Z", "p": 10.1, "s": 200}]   # below mid -> sell

    exact = with_exact_flow(p, trades, quotes)

    assert p.delta > 0 and exact.delta < 0
    assert exact.flow_method == "lee_ready"
    assert exact.histogram == p.histogram, "the profile itself is untouched"


def test_aggregating_mixed_classifiers_raises():
    """They agree on sign about one time in four, so summing them would produce
    a number belonging to neither."""
    cheap = build_profile("T", date(2025, 3, 4), minutes([10.0, 11.0]))
    exact = with_exact_flow(build_profile("T", date(2025, 3, 5), minutes([10.0, 11.0])), [], [])

    with pytest.raises(ValueError, match="mixed flow classifiers"):
        delta_ratio([cheap, exact])
    with pytest.raises(ValueError, match="mixed flow classifiers"):
        cumulative_delta([cheap, exact])


def test_store_round_trips_the_flow_method(tmp_path):
    store = ProfileStore(tmp_path / "p.db")
    p = build_profile("T", date(2025, 3, 4), minutes([10.0, 11.0]))
    store.put(with_exact_flow(p, [], []))

    assert store.get("T", date(2025, 3, 4)).flow_method == "lee_ready"
    store.close()
