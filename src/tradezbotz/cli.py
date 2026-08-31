"""Command line entry points.

    python -m tradezbotz ingest-edgar --days 30
    python -m tradezbotz enqueue-symbols
    python -m tradezbotz backfill [--limit N]
    python -m tradezbotz status

Every command is safe to re-run. Ingestion is idempotent by filing accession,
and the backfill is checkpointed per symbol, so a cron entry that overlaps a
still-running job wastes time but cannot corrupt state.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEFAULT_STATE = Path(os.environ.get("TRADEZBOTZ_STATE", "state"))
EVENTS_DB = "events.db"
BARS_DB = "bars.db"
CHECKPOINT_DB = "backfill.db"
PROFILES_DB = "profiles.db"
KIND_INSIDER = "insider_transaction"

#: Price history ceiling: events older than this cannot be labelled, because no
#: price data exists to measure their outcome.
#:
#: This was 730 for as long as Massive was the price vendor, whose free tier
#: serves roughly two years regardless of what is asked for. Alpaca serves 2016
#: onward, so the ceiling moved with the vendor -- and the old value was doing
#: real damage in the meantime: of 183,011 stored insider events, only **1,774**
#: fell inside a 730-day window. The other 99% were ingested, stored, and then
#: excluded from every backtest by a constant describing a vendor we no longer
#: use for this.
#:
#: The effect on dependence is the reason this matters beyond raw count. Events
#: were concentrated into two years of trading days, so the date dimension had
#: few distinct clusters and cross-sectional correlation was correspondingly
#: severe. Spreading the same events over ten years multiplies the distinct
#: dates by roughly five, which is the single largest lever available on the
#: effective sample size.
PRICE_WINDOW_DAYS = 3800  # ~2016 onward, matching ALPACA_HISTORY_DAYS

#: What the free Massive tier actually served. Kept because `--vendor massive`
#: still works and its ceiling is real; it is no longer the default.
MASSIVE_WINDOW_DAYS = 730

#: How much EDGAR history to ingest. Deliberately deeper than the price window:
#: the routine/opportunistic classifier needs 3+ years of an insider's *prior*
#: filings, so an ingest that merely matches the labelling window leaves the
#: earliest three years of events UNKNOWN and the filter carrying the actual
#: edge never fires on them.
#:
#: Derived rather than hardcoded so the relationship cannot drift. It was 1825
#: when the price window was 730; when the window moved to 3800 this had to move
#: with it, and a second constant would have silently failed to.
#:
#: EDGAR history is free and unbounded, so the only cost of going deeper is
#: ingest time -- roughly nine minutes per uncached quarter, once.
BASELINE_LEAD_DAYS = 3 * 365  # MIN_YEARS_FOR_ROUTINE in research.classify
BASELINE_DAYS = PRICE_WINDOW_DAYS + BASELINE_LEAD_DAYS  # ~13 years

#: Alpaca's consolidated feed serves minute and daily bars back to 2016, against
#: Massive's two years. Verified by probe, not read off a pricing page. This is
#: what makes a real holdout possible: two years of history leaves almost
#: nothing to hold out, and a holdout you cannot afford to lock is not one.
ALPACA_HISTORY_DAYS = 3800  # ~2016 onward

#: Retained for callers that predate the split.
HISTORY_DAYS = PRICE_WINDOW_DAYS


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so cron and systemd need no shell wrapper."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_ingest_edgar(args: argparse.Namespace) -> int:
    from .lock import SingleInstance
    from .research.edgar import EdgarClient, ingest_day
    from .research.eventstore import EventStore

    # Two concurrent ingests push EDGAR traffic past the SEC's 10 req/s limit.
    lock = SingleInstance("ingest", DEFAULT_STATE)

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))
    price_cutoff = date.today() - timedelta(days=PRICE_WINDOW_DAYS)
    if start < price_cutoff:
        print(
            f"note: filings before {price_cutoff} have no price data and cannot be "
            "labelled. They are ingested on purpose, as baselines for the "
            "routine/opportunistic classifier.",
            file=sys.stderr,
        )

    lock.acquire()
    client = EdgarClient()
    # Prove the User-Agent works once, so a later 403 can be read as "index not
    # published yet" rather than "credentials rejected". Fail fast if it doesn't.
    client.verify_access()
    store = EventStore(DEFAULT_STATE / EVENTS_DB)
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None

    # Newest first: the most recent filings are the ones a daily run needs, and
    # a sliced backfill that never finishes still keeps the recent window fresh.
    days = [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]
    days.reverse()

    total_new = skipped = processed = failed = 0
    for day in days:
        if deadline and time.monotonic() > deadline:
            print(f"\ntime budget reached; {len(days) - processed - skipped} days left")
            break
        if not args.force and store.day_ingested("sec_form4", day):
            skipped += 1
            continue
        try:
            events, rejected = [], 0
            for txn in ingest_day(client, day):
                # Isolate per transaction: one malformed filing among ~1000 must
                # not cost the whole day's fetch.
                try:
                    events.append(txn.to_event())
                except Exception:  # noqa: BLE001
                    rejected += 1
        except Exception as exc:  # noqa: BLE001
            # One unavailable day must not end a multi-hour run. The day stays
            # unmarked, so the next invocation retries it.
            failed += 1
            print(f"{day}  FAILED  {type(exc).__name__}: {exc}"[:160], flush=True)
            continue
        new = store.record_many(events)
        if events:
            store.mark_day_ingested("sec_form4", day, len(events))
        total_new += new
        processed += 1
        note = f"  rejected {rejected}" if rejected else ""
        print(
            f"{day}  filings->events {len(events):5d}  new {new:5d}{note}", flush=True
        )

    print(f"\ndays processed {processed}, already done {skipped}, failed {failed}")
    print(f"total new events: {total_new}")
    print(f"event store total: {store.count()}")
    store.close()
    return 0


def cmd_ingest_bulk(args: argparse.Namespace) -> int:
    """Load baselines from the SEC's quarterly Form 345 archives.

    ~20 downloads for five years, versus roughly half a million individual
    filing fetches. Stops short of the price window so it cannot collide with
    the timed per-filing path, which mints different external_ids.
    """
    from .lock import SingleInstance
    from .research.bulk import download_quarter, events_from_archive, quarters_between
    from .research.edgar import EdgarClient
    from .research.eventstore import EventStore
    from .research.submissions import (
        SubmissionsCache, SubmissionsClient, upgrade_precision,
    )

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))
    cutoff = args.before or (date.today() - timedelta(days=MASSIVE_WINDOW_DAYS))

    # flush: these run under nohup and systemd, where an unflushed header makes
    # a working job look hung for the ~9 minutes a first quarter takes.
    print(f"loading {start} -> {cutoff}", flush=True)
    kind = "exact (submissions API)" if args.timed else "date-only"
    print(f"timestamps: {kind}\n", flush=True)

    lock = SingleInstance("ingest", DEFAULT_STATE)
    lock.acquire()
    try:
        client = EdgarClient()
        client.verify_access()
        store = EventStore(DEFAULT_STATE / EVENTS_DB)
        archives = DEFAULT_STATE / "bulk"
        subs_cache = SubmissionsCache(DEFAULT_STATE / "submissions.db")
        subs = SubmissionsClient(client, subs_cache)
        total = 0
        deadline = (
            time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
        )
        # Newest quarter first. quarters_between returns oldest-first, and a
        # time-boxed run then spends its whole budget on deep baselines and
        # never reaches the labelling window -- which is the only part that
        # yields measurable results. Run #2 loaded 2021Q3..2025Q1 and stopped,
        # leaving 2.5% of 1.4M events labellable. Order is irrelevant to
        # correctness, so prioritise the window we can actually measure.
        for year, quarter in reversed(quarters_between(start, min(end, cutoff))):
            if deadline and time.monotonic() > deadline:
                # CI runs are time-boxed. Quarters already stored are skipped on
                # the next invocation, so stopping here loses nothing.
                print("time budget reached; remaining quarters left for next run",
                      flush=True)
                break
            try:
                path = download_quarter(client, year, quarter, archives)
            except FileNotFoundError:
                print(f"{year}Q{quarter}  not published yet")
                continue
            events = list(events_from_archive(path, before=cutoff))

            if args.timed and events:
                # Bulk archives carry only a filing date. One request per issuer
                # recovers the exact acceptance time for all of that issuer's
                # filings -- ~4,200 issuers a quarter against ~27,000 filings,
                # so this costs minutes rather than hours.
                ciks = {e.payload.get("issuer_cik") for e in events}
                ciks.discard(None)
                fetched = subs.load_ciks(sorted(ciks))
                events = list(upgrade_precision(events, subs_cache))
                timed = sum(1 for e in events if e.payload.get("precision") == "timed")
                print(f"{year}Q{quarter}  issuers {len(ciks):5,d} "
                      f"(fetched {fetched:5,d})  timed {timed:7,d}/{len(events):,d}",
                      flush=True)

            new = store.record_many(events)
            total += new
            print(f"{year}Q{quarter}  events {len(events):7,d}  new {new:7,d}", flush=True)
        print(f"\ntotal new baseline events: {total:,}")
        print(f"event store total: {store.count():,}")
        store.close()
    finally:
        lock.release()
    return 0


def cmd_ingest_sentiment(args: argparse.Namespace) -> int:
    """Snapshot Reddit mention counts.

    ApeWisdom serves only a current snapshot -- no history exists at any price,
    verified against the live API. So this cannot be backfilled, only
    accumulated: every day it does not run is a day of history lost for good.
    """
    from .research.apewisdom import DEFAULT_FILTERS, ApeWisdomClient, to_events
    from .research.eventstore import EventStore

    client = ApeWisdomClient()
    store = EventStore(DEFAULT_STATE / EVENTS_DB)
    filters = args.filters or list(DEFAULT_FILTERS)
    total = 0
    for name in filters:
        try:
            mentions = client.mentions(name)
        except Exception as exc:  # noqa: BLE001
            # An unofficial free API must never fail the pipeline that carries
            # the insider signal.
            print(f"{name:16s} FAILED {type(exc).__name__}: {exc}"[:140], flush=True)
            continue
        new = store.record_many(list(to_events(mentions)))
        total += new
        print(f"{name:16s} tickers {len(mentions):5d}  new {new:5d}", flush=True)

    print(f"total new sentiment events: {total}", flush=True)
    store.close()
    return 0


def cmd_enqueue_symbols(args: argparse.Namespace) -> int:
    from .research.backfill import symbols_from_events
    from .research.eventstore import EventStore

    store = EventStore(DEFAULT_STATE / EVENTS_DB)
    now = datetime.now(timezone.utc)
    # Only symbols inside the price window are worth fetching. Baseline-only
    # filings can never be labelled, and at 5 requests/minute queueing them
    # would spend days of budget on data with no possible outcome.
    since = now - timedelta(days=args.price_window)
    labellable = list(store.as_of(now, since=since))
    if args.buys_only:
        # Open-market purchases are the signal; they touch ~35% of the symbols
        # that all transaction codes do. At 5 requests/minute that is the
        # difference between a 5-hour and a 15-hour backfill.
        labellable = [
            e for e in labellable
            if e["payload"].get("transaction_code") == "P"
            and e["payload"].get("acquired_disposed") == "A"
        ]
    all_events = store.count(KIND_INSIDER)
    symbols = symbols_from_events(labellable)
    store.close()

    runner = _make_runner(need_source=False)
    added = runner.enqueue(symbols)
    print(
        f"{all_events} stored events; {len(labellable)} inside the "
        f"{args.price_window}-day price window"
    )
    print(f"{len(symbols)} distinct labellable symbols; {added} newly queued")
    print(runner.progress())
    runner.close()
    return 0


def _make_runner(limit_per_minute: int | None = None, *, vendor: str = "alpaca",
                 history_days: int | None = None, need_source: bool = True):
    """Build the price backfill.

    Alpaca by default. The reason is arithmetic: Massive's free tier allows 5
    requests/minute, which measured out at 242 symbols/hour and made a
    3,373-symbol universe a fourteen-hour job that never finished inside a
    six-hour CI run. Alpaca allows 200/minute and serves history back to 2016
    against Massive's two years. Even fetching both adjustment bases -- two
    requests per symbol instead of one -- it is roughly twenty times faster.

    Massive stays reachable with `--vendor massive`, and remains a crosscheck
    source. It is no longer the bottleneck for coverage.
    """
    from .research.backfill import BackfillRunner
    from .research.prices import (
        ALPACA_REQUESTS_PER_MINUTE,
        DEFAULT_REQUESTS_PER_MINUTE,
        DualBasisSource,
        MassivePriceSource,
        PriceCache,
    )

    cache = PriceCache(DEFAULT_STATE / BARS_DB)
    if not need_source:
        # Queueing and status touch only the checkpoint. Constructing a vendor
        # client would demand credentials those commands have no use for, which
        # is what broke the pipeline after the switch to Alpaca.
        source = None
        window = history_days or ALPACA_HISTORY_DAYS
    elif vendor == "massive":
        source = MassivePriceSource(
            cache=cache,
            per_minute=limit_per_minute or DEFAULT_REQUESTS_PER_MINUTE,
        )
        window = MASSIVE_WINDOW_DAYS
    else:
        source = DualBasisSource.alpaca(
            cache=cache,
            per_minute=limit_per_minute or ALPACA_REQUESTS_PER_MINUTE,
        )
        window = history_days or ALPACA_HISTORY_DAYS

    end = date.today()
    return BackfillRunner(
        source,
        DEFAULT_STATE / CHECKPOINT_DB,
        start=end - timedelta(days=window),
        end=end,
    )


def cmd_backfill(args: argparse.Namespace) -> int:
    from .lock import SingleInstance

    # Concurrent backfills would double the request rate and race on the cache.
    with SingleInstance("backfill", DEFAULT_STATE):
        return _run_backfill(args)


def _run_backfill(args: argparse.Namespace) -> int:
    runner = _make_runner(args.per_minute, vendor=args.vendor,
                          history_days=args.days)
    if args.requeue:
        n = runner.requeue()
        print(f"requeued {n} finished symbols: what 'done' means has changed")
    runner.install_signal_handlers()

    def report(symbol: str, prog) -> None:
        print(f"[{prog.done + prog.failed:5d}] {symbol:8s} {prog}", flush=True)

    from .research.watchlist import Watchlist
    watched = Watchlist().symbols()
    if watched:
        print(f"watchlist: {len(watched)} symbols fetched first "
              f"({', '.join(watched[:8])}{'...' if len(watched) > 8 else ''})")

    print(f"starting: {runner.progress()}")
    prog = runner.run(limit=args.limit, on_progress=report, priority=watched)
    print(f"\nfinished: {prog}")

    failures = runner.failures()
    if failures:
        print(f"\n{len(failures)} parked failures:")
        for row in failures[:20]:
            print(f"  {row['symbol']:8s} attempts={row['attempts']}  {row['last_error'][:90]}")
    runner.close()
    return 0


def cmd_crosscheck(args: argparse.Namespace) -> int:
    """Compare Massive against Alpaca on symbols we already hold.

    Free data ships without error bars: a single source returns a close for an
    illiquid micro-cap with exactly the same confidence as one for Apple.
    Two independent sources supply the missing information -- where they agree
    the bar is probably fine, where they diverge the name is too thin for either
    to be trusted.

    This decides whether Alpaca's deeper history (7+ years, against Massive's 2)
    is usable for the slow indicators, rather than assuming either way.
    """
    from .research.crosscheck import (
        adjudicate, compare, summarise, summarise_adjudications,
    )
    from .research.prices import AlpacaPriceSource, OpenBBPriceSource, PriceCache

    cache = PriceCache(DEFAULT_STATE / BARS_DB)
    symbols = cache.symbols()[: args.limit] if args.limit else cache.symbols()
    if not symbols:
        print("no cached symbols to compare; run backfill first")
        return 1

    alpaca = AlpacaPriceSource(per_minute=args.per_minute)
    # A third, independent opinion. Two sources can only ever say "one of you is
    # wrong"; three can say which one. Optional -- it needs an extra package.
    referee = OpenBBPriceSource() if args.three_way else None
    end = date.today()
    start = end - timedelta(days=PRICE_WINDOW_DAYS)

    results, worst, verdicts = [], [], []
    for i, symbol in enumerate(symbols, 1):
        primary = cache.get(symbol, start, end)
        if not primary.bars:
            continue
        try:
            secondary = alpaca.daily_bars(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol:8s} fetch failed: {type(exc).__name__}", flush=True)
            continue
        d = compare(primary, secondary)
        results.append(d)
        if d.median_rel_diff and d.median_rel_diff > 0.02:
            worst.append(d)
        # Only pay for the referee where the first two actually disagree.
        # Symbols they already agree on need no tiebreak.
        if referee is not None and not d.trustworthy:
            try:
                third = referee.daily_bars(symbol, start, end)
                verdicts.append(adjudicate(primary, secondary, third))
            except Exception as exc:  # noqa: BLE001
                print(f"  {symbol:8s} referee failed: {type(exc).__name__}", flush=True)
        if i % 25 == 0:
            print(f"  ...{i}/{len(symbols)}", flush=True)

    print()
    for k, v in summarise(results).items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

    if worst:
        print(f"\n  {len(worst)} symbols where the sources materially disagree:")
        for d in sorted(worst, key=lambda x: -(x.median_rel_diff or 0))[:15]:
            print(f"    {d.symbol:8s} median {d.median_rel_diff:6.2%}  "
                  f"max {d.max_rel_diff:6.2%}  over {d.overlapping_days} days")
        print("\n  Treat these as low-confidence: exclude them in a sensitivity")
        print("  check rather than silently trusting either source.")

    if verdicts:
        print("\n  Third-source adjudication of the disputed symbols:")
        for k, v in summarise_adjudications(verdicts).items():
            print(f"    {k:22s} {v}")
        settled = [v for v in verdicts
                   if v.trustworthy_source in ("primary", "secondary")]
        if settled:
            print("\n    symbol    believe   pairwise medians")
            for v in sorted(settled, key=lambda x: x.symbol)[:20]:
                name = {"primary": "massive", "secondary": "alpaca"}[v.trustworthy_source]
                print(f"    {v.symbol:8s}  {name:8s}  "
                      f"M-A {v.primary_secondary:6.2%}  "
                      f"M-Y {v.primary_referee:6.2%}  "
                      f"A-Y {v.secondary_referee:6.2%}")
            print("\n    Per symbol, not globally. Neither vendor is uniformly")
            print("    correct, so there is no single source to switch to.")
    cache.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .research.eventstore import EventStore
    from .research.prices import PriceCache

    events_path = DEFAULT_STATE / EVENTS_DB
    if events_path.exists():
        store = EventStore(events_path)
        print(f"events           : {store.count()} ({store.count('insider_transaction')} insider)")
        store.close()
    else:
        print("events           : no store yet")

    bars_path = DEFAULT_STATE / BARS_DB
    if bars_path.exists():
        from .research.prices import BASES
        cache = PriceCache(bars_path)
        print(f"symbols cached   : {len(cache.symbols())}")
        # Per basis, because a symbol holding only one is half-migrated and
        # nothing else in the data would say so.
        for basis in BASES:
            print(f"  basis {basis:8s}   : {len(cache.symbols(basis))}")
        cache.close()
    else:
        print("symbols cached   : none")

    profiles_path = DEFAULT_STATE / PROFILES_DB
    if profiles_path.exists():
        from .research.intraday import ProfileStore
        store = ProfileStore(profiles_path)
        print(f"intraday sessions: {store.count():,} over {len(store.symbols())} symbols")
        store.close()
    else:
        print("intraday sessions: none")

    ckpt = DEFAULT_STATE / CHECKPOINT_DB
    if ckpt.exists():
        runner = _make_runner(need_source=False)
        print(f"backfill         : {runner.progress()}")
        print(f"parked failures  : {len(runner.failures())}")
        runner.close()
    else:
        print("backfill         : not started")
    return 0


def cmd_backfill_intraday(args: argparse.Namespace) -> int:
    """Reduce sessions to volume profiles and order flow.

    Volume profile and order flow cannot be computed from daily bars at all, so
    this is the only path that makes either testable. Each session is fetched
    once, reduced to a compact histogram plus flow statistics, and stored -- raw
    minute bars are never kept, because the universe over a couple of years is
    on the order of 10^8 of them.

    Two cost profiles, deliberately separated:

      minute bars   batched across symbols, so one request covers up to 100
                    names. Cheap enough to run over the whole universe.
      trades+quotes one symbol at a time, and the only way to get order flow
                    that is actually order flow. `--exact` opts in.

    The `--exact` distinction is not a performance knob. Measured against real
    prints on four small caps, minute-bar tick-rule delta agreed with tick-level
    Lee-Ready on *sign* once in four. Without `--exact` the stored delta is
    labelled `tick_minute` and should be read as "direction of minute closes",
    which is a different hypothesis rather than a cheaper version of this one.
    """
    from .lock import SingleInstance
    from .research.intraday import (
        AlpacaIntradaySource,
        ProfileStore,
        SIP_DELAY_MINUTES,
        group_by_session,
    )
    from .research.microstructure import build_profile, with_exact_flow
    from .research.prices import PriceCache

    # Its own lock, not the shared "ingest" one: this path talks to Alpaca, not
    # EDGAR, so it can safely run alongside a filing ingest.
    lock = SingleInstance("intraday", DEFAULT_STATE)
    lock.acquire()
    try:
        cache = PriceCache(DEFAULT_STATE / BARS_DB)
        symbols = cache.symbols()
        cache.close()
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        if not symbols:
            print("no cached symbols; run backfill first")
            return 1
        # Watched names first, so a time-boxed run always covers them.
        from .research.watchlist import Watchlist, prioritise
        watched = Watchlist().symbols()
        if watched:
            symbols = prioritise(symbols, watched)
            print(f"watchlist: {len(watched)} symbols reduced first")
        symbols = symbols[: args.limit] if args.limit else symbols

        store = ProfileStore(DEFAULT_STATE / PROFILES_DB)
        source = AlpacaIntradaySource()

        # Never request into the SIP embargo; `end` is exclusive.
        latest = (datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MINUTES)).date()
        end = min(date.today(), latest)
        start = end - timedelta(days=args.days)

        deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
        built = skipped = exact_done = 0

        for i in range(0, len(symbols), args.batch):
            if deadline and time.monotonic() > deadline:
                print("time budget reached; state is checkpointed")
                break
            chunk = symbols[i : i + args.batch]
            try:
                bars_by_symbol = source.minute_bars(chunk, start, end)
            except Exception as exc:                      # noqa: BLE001
                print(f"  batch {i // args.batch}: {type(exc).__name__}: {exc}")
                continue

            for symbol, bars in bars_by_symbol.items():
                for day, session in group_by_session(bars).items():
                    if store.was_fetched(symbol, day):
                        skipped += 1
                        continue
                    profile = build_profile(symbol, day, session)
                    if profile is None:
                        # A real session with no prints. Recorded as attempted so
                        # the next run does not ask again.
                        store.mark_fetched(symbol, day)
                        continue
                    if args.exact:
                        try:
                            trades = source.trades(symbol, day, limit_pages=args.trade_pages)
                            quotes = source.quotes(symbol, day, limit_pages=args.quote_pages)
                            if trades:
                                profile = with_exact_flow(profile, trades, quotes)
                                exact_done += 1
                        except Exception as exc:          # noqa: BLE001
                            print(f"  {symbol} {day}: flow unavailable ({exc})")
                    store.put(profile)
                    built += 1

            print(f"  {min(i + args.batch, len(symbols))}/{len(symbols)} symbols  "
                  f"sessions built {built:,}  skipped {skipped:,}"
                  + (f"  exact flow {exact_done:,}" if args.exact else ""))

        print(f"\nstored sessions : {store.count():,}")
        print(f"symbols covered : {len(store.symbols())}")
        store.close()
        return 0
    finally:
        lock.release()


def cmd_ingest_filings(args: argparse.Namespace) -> int:
    """Ingest 8-K material events and 424B offerings.

    These exist because news does not cover our universe. Measured against the
    Benzinga feed, NVDA drew 200+ articles in a month while XELB and ARI drew
    zero -- yet XELB filed eleven 8-Ks that year and ARI six. Disclosure
    obligations do not scale with market cap; journalist attention does.
    """
    from .lock import SingleInstance
    from .research.edgar import EdgarClient
    from .research.eventstore import EventStore
    from .research.filings import FORMS_424B, ingest_day

    forms: tuple[str, ...] = ()
    if args.kind in ("both", "8-K"):
        forms += ("8-K",)
    if args.kind in ("both", "424B"):
        forms += FORMS_424B

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))

    lock = SingleInstance("ingest", DEFAULT_STATE)
    lock.acquire()
    try:
        client = EdgarClient()
        client.verify_access()
        store = EventStore(DEFAULT_STATE / EVENTS_DB)
        deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None

        # Newest first, same reasoning as the Form 4 path: a time-boxed run that
        # never finishes should still keep the recent window fresh.
        days = [
            start + timedelta(days=i)
            for i in range((end - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5
        ]
        days.reverse()

        total = skipped = 0
        for day in days:
            if deadline and time.monotonic() > deadline:
                print("time budget reached; remaining days left for next run",
                      flush=True)
                break
            marker = f"sec_filings_{args.kind}"
            if not args.force and store.day_ingested(marker, day):
                skipped += 1
                continue
            events = []
            for parsed in ingest_day(client, day, forms):
                try:
                    events.append(parsed.to_event())
                except Exception:  # noqa: BLE001
                    continue
            new = store.record_many(events)
            if events:
                store.mark_day_ingested(marker, day, len(events))
            total += new
            print(f"{day}  filings {len(events):5d}  new {new:5d}", flush=True)

        print(f"\ndays done, {skipped} already ingested")
        print(f"total new events: {total}")
        print(f"event store total: {store.count()}")
        store.close()
        return 0
    finally:
        lock.release()


def cmd_watch(args: argparse.Namespace) -> int:
    """Manage the watchlist.

    The watchlist changes fetch priority and reporting. It never changes the
    measured universe -- see research/watchlist.py for why that firewall exists.
    """
    from .research.watchlist import Watchlist, WatchlistError

    wl = Watchlist(args.file)

    if args.action == "add":
        try:
            entry = wl.add(args.symbol, args.reason or "")
        except WatchlistError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"watching {entry.symbol}: {entry.reason}")
        print(f"saved to {wl.path}")
        return 0

    if args.action == "remove":
        if wl.remove(args.symbol):
            print(f"removed {args.symbol.upper()}")
            return 0
        print(f"{args.symbol.upper()} was not on the list", file=sys.stderr)
        return 1

    entries = wl.load()
    if not entries:
        print("watchlist is empty")
        print(f"  add one:  python -m tradezbotz watch add NVDA --reason \"...\"")
        return 0

    if args.action == "list":
        print(f"{len(entries)} watched symbols  ({wl.path})\n")
        for e in entries:
            print(f"  {e.symbol:8s} added {e.added}  {e.reason}")
        return 0

    # status: what we actually hold for each watched name
    from .research.prices import BASES, PriceCache
    from .research.intraday import ProfileStore

    cache = PriceCache(DEFAULT_STATE / BARS_DB)
    profiles_path = DEFAULT_STATE / PROFILES_DB
    store = ProfileStore(profiles_path) if profiles_path.exists() else None
    end = date.today()
    start = end - timedelta(days=ALPACA_HISTORY_DAYS)

    print(f"{'symbol':8}{'bases':>14}{'bars':>8}{'from':>12}{'sessions':>10}  reason")
    print("-" * 78)
    for e in entries:
        held = cache.bases(e.symbol)
        series = cache.get(e.symbol, start, end) if held else None
        n = len(series.bars) if series else 0
        first = series.bars[0].day.isoformat() if series and series.bars else "-"
        sessions = len(store.range(e.symbol, start, end)) if store else 0
        print(f"{e.symbol:8}{'/'.join(held) or 'none':>14}{n:>8}{first:>12}"
              f"{sessions:>10}  {e.reason[:30]}")
    uncovered = [e.symbol for e in entries if not cache.bases(e.symbol)]
    cache.close()
    if store:
        store.close()
    if uncovered:
        print(f"\n  {len(uncovered)} watched symbols have no bars yet: "
              f"{', '.join(uncovered)}")
        print("  They are queued first on the next backfill.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradezbotz")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest-edgar", help="pull Form 4 filings into the event store")
    ing.add_argument("--days", type=int, default=BASELINE_DAYS,
                     help=f"EDGAR history to ingest (default {BASELINE_DAYS}, ~5y). "
                          "Deeper than the price window on purpose: classifier "
                          "baselines need 3+ years per insider.")
    ing.add_argument("--start", type=date.fromisoformat)
    ing.add_argument("--end", type=date.fromisoformat)
    ing.add_argument("--max-minutes", type=float,
                     help="stop cleanly after N minutes (for sliced CI runs)")
    ing.add_argument("--force", action="store_true",
                     help="re-pull days already marked complete")
    ing.set_defaults(func=cmd_ingest_edgar)

    blk = sub.add_parser("ingest-bulk",
                         help="load baselines from SEC quarterly archives (fast)")
    blk.add_argument("--days", type=int, default=BASELINE_DAYS,
                     help=f"history to load (default {BASELINE_DAYS}, ~5y)")
    blk.add_argument("--start", type=date.fromisoformat)
    blk.add_argument("--end", type=date.fromisoformat)
    blk.add_argument("--max-minutes", type=float,
                     help="stop cleanly after N minutes (for sliced CI runs)")
    blk.add_argument("--timed", action="store_true",
                     help="join exact acceptance times from the submissions API. "
                          "Required for the labelling window; unnecessary for deep "
                          "baselines, where only transaction dates matter.")
    blk.add_argument("--before", type=date.fromisoformat,
                     help="stop before this date; defaults to the price window "
                          "start, leaving recent filings to the timed path")
    blk.set_defaults(func=cmd_ingest_bulk)

    snt = sub.add_parser("ingest-sentiment",
                         help="snapshot Reddit mention counts (accumulates only)")
    snt.add_argument("--filters", nargs="*",
                     help="community filters; defaults to all-stocks, "
                          "wallstreetbets, stocks")
    snt.set_defaults(func=cmd_ingest_sentiment)

    enq = sub.add_parser("enqueue-symbols",
                         help="queue symbols from events that can actually be labelled")
    enq.add_argument("--buys-only", action="store_true",
                     help="queue only symbols with an open-market purchase")
    enq.add_argument("--price-window", type=int, default=PRICE_WINDOW_DAYS,
                     help=f"days of price history available (default {PRICE_WINDOW_DAYS})")
    enq.set_defaults(func=cmd_enqueue_symbols)

    bf = sub.add_parser("backfill", help="fetch daily bars for queued symbols")
    bf.add_argument("--limit", type=int, help="stop after N symbols this run")
    bf.add_argument("--per-minute", type=int, help="override the request budget")
    bf.add_argument("--vendor", choices=("alpaca", "massive"), default="alpaca",
                    help="alpaca (default) stores both adjustment bases at "
                         "200 req/min; massive is price-only at 5 req/min")
    bf.add_argument("--days", type=int,
                    help="history window; defaults to 2016 onward for alpaca, "
                         "730 days for massive")
    bf.add_argument("--requeue", action="store_true",
                    help="return finished symbols to the queue. Needed after a "
                         "vendor or basis change, where 'done' now means "
                         "something different than it did")
    bf.set_defaults(func=cmd_backfill)

    xc = sub.add_parser("crosscheck",
                        help="compare Massive vs Alpaca on cached symbols")
    xc.add_argument("--limit", type=int, help="stop after N symbols")
    xc.add_argument("--per-minute", type=int, default=180,
                    help="Alpaca allows 200/min on the free plan")
    xc.add_argument("--three-way", action="store_true",
                    help="bring in a third source (OpenBB/Yahoo) to adjudicate "
                         "symbols where Massive and Alpaca disagree; needs "
                         "`pip install openbb-yfinance`")
    xc.set_defaults(func=cmd_crosscheck)

    itd = sub.add_parser("backfill-intraday",
                         help="reduce sessions to volume profiles and order flow")
    itd.add_argument("--days", type=int, default=180,
                     help="how far back to reduce sessions")
    itd.add_argument("--limit", type=int, default=0, help="cap symbols processed")
    itd.add_argument("--symbols", default="", help="comma-separated override")
    itd.add_argument("--batch", type=int, default=50,
                     help="symbols per multi-symbol bar request")
    itd.add_argument("--minutes", type=int, default=0,
                     help="stop after N minutes; state is checkpointed")
    itd.add_argument("--exact", action="store_true",
                     help="classify flow from trades and quotes (Lee-Ready) "
                          "instead of minute closes; far slower and the only "
                          "version that actually measures order flow")
    itd.add_argument("--trade-pages", type=int, default=3,
                     help="cap trade pagination per session under --exact")
    itd.add_argument("--quote-pages", type=int, default=6,
                     help="cap quote pagination per session under --exact")
    itd.set_defaults(func=cmd_backfill_intraday)

    fil = sub.add_parser("ingest-filings",
                         help="pull 8-K material events and 424B offerings")
    fil.add_argument("--days", type=int, default=365, help="how far back to go")
    fil.add_argument("--start", type=date.fromisoformat)
    fil.add_argument("--end", type=date.fromisoformat)
    fil.add_argument("--kind", choices=("both", "8-K", "424B"), default="both")
    fil.add_argument("--max-minutes", type=int, default=0,
                     help="stop after N minutes; days are checkpointed")
    fil.add_argument("--force", action="store_true",
                     help="re-ingest days already marked done")
    fil.set_defaults(func=cmd_ingest_filings)

    wt = sub.add_parser("watch",
                        help="symbols to pay special attention to (fetch "
                             "priority and reporting only, never the backtest "
                             "universe)")
    wt.add_argument("action", choices=("list", "add", "remove", "status"),
                    nargs="?", default="list")
    wt.add_argument("symbol", nargs="?", default="")
    wt.add_argument("--reason", help="why this symbol is worth watching "
                                     "(required when adding)")
    wt.add_argument("--file", default="watchlist.yml",
                    help="watchlist location (versioned in the repo by design)")
    wt.set_defaults(func=cmd_watch)

    st = sub.add_parser("status", help="show pipeline state")
    st.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    from .lock import LockHeld

    _load_dotenv()
    DEFAULT_STATE.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LockHeld as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
