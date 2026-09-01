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
import hashlib
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
#: Append-only log of every backtest ever run against this dataset. It lives in
#: the encrypted state blob with everything else, because a trial count that
#: resets is a Deflated Sharpe bar that resets, and the whole correction depends
#: on the count being cumulative across runs.
TRIALS_DB = "trials.db"
#: Cached SEC XBRL company facts, one gzipped row per issuer.
FACTS_DB = "facts.db"
#: Alpaca's US equity asset list, including inactive listings. What makes
#: "delisted" distinguishable from "never existed" and from "OTC".
ASSETS_DB = "assets.db"
#: Daily geopolitical risk (Caldara & Iacoviello). One world-level series,
#: so it conditions every event rather than describing any one symbol.
MACRO_DB = "macro.db"
#: Per-run step outcomes. A pipeline that reports success while a third of
#: its steps have silently failed for a week is worse than one that fails.
RUNLOG_DB = "runlog.db"
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

    # OTC names return zero bars on the SIP feed rather than an error, and the
    # `otc` feed is 403 on this plan -- both verified by probe. So they cost a
    # request each and land in the same "no data" bucket as a genuine gap.
    #
    # Skipping is opt-in, not the default. A name on OTC today may have been
    # listed for most of the window, and its history is real and already
    # fetched; excluding on today's tag would delete data we hold and introduce
    # survivorship bias in a cleanup step.
    assets_path = DEFAULT_STATE / ASSETS_DB
    if assets_path.exists():
        from .research.assets import OTC, AssetCatalog, describe

        catalog = AssetCatalog(assets_path)
        if catalog.count():
            print(f"universe composition ({len(symbols):,} symbols):")
            print(describe(catalog.breakdown(symbols)))
            if args.skip_otc:
                before = len(symbols)
                symbols = [s for s in symbols if catalog.classify(s) != OTC]
                print(f"\n--skip-otc dropped {before - len(symbols):,} symbols")
        catalog.close()
    elif args.skip_otc:
        print("--skip-otc ignored: no asset catalog yet, run ingest-assets",
              file=sys.stderr)

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
        # Enqueue as well as prioritise. `prioritise` only reorders, by design --
        # it must never shrink the queue -- but that left a watched symbol which
        # was never queued permanently unfetched, while `watch status` claimed it
        # would be picked up next run. Adding to the FETCH queue is safe: it
        # obtains bars, and bars are not events. The research universe still
        # comes only from ingested filings.
        added = runner.enqueue(watched)
        if added:
            print(f"watchlist: queued {added} watched symbols that were absent")
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
        untimed = store.count_untimed()
        if untimed:
            print(f"  without timing   : {untimed:,} -- these predate the "
                  f"session-sequence fields and block the intraday sweep test.\n"
                  f"  Refetch with `backfill-intraday --refresh-untimed`; they "
                  f"cannot be repaired in place.")
        store.close()
    else:
        print("intraday sessions: none")

    assets_path = DEFAULT_STATE / ASSETS_DB
    if assets_path.exists():
        from .research.assets import AssetCatalog, describe
        from .research.eventstore import EventStore as _ES

        catalog = AssetCatalog(assets_path)
        events_path2 = DEFAULT_STATE / EVENTS_DB
        if catalog.count() and events_path2.exists():
            st = _ES(events_path2)
            syms = [r[0] for r in st._conn.execute(
                "SELECT DISTINCT symbol FROM events WHERE kind = ? "
                "AND symbol IS NOT NULL", (KIND_INSIDER,))]
            st.close()
            print(f"universe          : {len(syms):,} symbols")
            print(describe(catalog.breakdown(syms)))
        catalog.close()
    else:
        print("universe          : unclassified; run ingest-assets")

    trials_path = DEFAULT_STATE / TRIALS_DB
    if trials_path.exists():
        from .research.trials import TrialRegistry
        reg = TrialRegistry(trials_path)
        n, runs = reg.count(), reg.executions()
        last = reg._conn.execute(
            "SELECT MAX(last_run_at) FROM trials").fetchone()[0]
        reg.close()
        print(f"trials            : {n:,} distinct ({runs:,} executions)")
        print(f"  last measured   : {last or 'never'}")
    else:
        # T7: measure runs with continue-on-error, so a silent permanent
        # failure would otherwise look identical to a healthy pipeline.
        print("trials            : NONE -- `measure` has never completed. "
              "It runs with continue-on-error, so a persistent failure is "
              "silent; check the measure step's log.")

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

            # Accumulated across the batch and written under one commit. The
            # per-session `put` committed twice each, at 442 sessions/s against
            # 62,000/s batched -- which is most of why this step overran a
            # 15-minute budget by 36.
            pending: list = []
            empty: list = []
            overran = False
            for symbol, bars in bars_by_symbol.items():
                # The deadline was checked only between batches, and a batch is
                # 50 symbols x ~124 sessions. One batch could therefore run for
                # tens of minutes past the budget, which is exactly what
                # happened. Checked per symbol now, so the overrun is bounded by
                # one symbol rather than one batch.
                if deadline and time.monotonic() > deadline:
                    overran = True
                    break
                for day, session in group_by_session(bars).items():
                    if store.was_fetched(symbol, day):
                        # --refresh-untimed rebuilds only what is actually
                        # missing the session-sequence fields. Those cannot be
                        # filled in from the store -- minute bars are never kept
                        # -- so the session has to come down again, and scoping
                        # the refetch to the rows that need it is the difference
                        # between minutes and a full re-reduction.
                        stale = False
                        if args.refresh_untimed:
                            held = store.get(symbol, day)
                            stale = held is not None and not held.has_timing
                        if not stale:
                            skipped += 1
                            continue
                    profile = build_profile(symbol, day, session)
                    if profile is None:
                        # A real session with no prints. Recorded as attempted so
                        # the next run does not ask again.
                        empty.append((symbol, day))
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
                    pending.append(profile)
                    built += 1

            store.put_many(pending)
            store.mark_many_fetched(empty)

            print(f"  {min(i + args.batch, len(symbols))}/{len(symbols)} symbols  "
                  f"sessions built {built:,}  skipped {skipped:,}"
                  + (f"  exact flow {exact_done:,}" if args.exact else ""))
            if overran:
                print("time budget reached mid-batch; state is checkpointed")
                break

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


def cmd_ingest_holdings(args: argparse.Namespace) -> int:
    """Ingest 13F holdings, 13D/G stakes, and House congressional trades.

    Three disclosure regimes with different lags: 13F is 45 days stale by
    construction, 13D is five business days, and House PTRs run 14-45 days.
    Each records its own lag on every event so a strategy has to reason about
    staleness rather than forget it.
    """
    from .lock import SingleInstance
    from .research.edgar import EdgarClient
    from .research.eventstore import EventStore
    from .research.holdings import (
        FORMS_13DG, FORMS_13F, HOUSE_BULK, extract_pdf_text, ingest_day,
        parse_house_index, parse_house_ptr,
    )

    lock = SingleInstance("ingest", DEFAULT_STATE)
    lock.acquire()
    try:
        store = EventStore(DEFAULT_STATE / EVENTS_DB)
        deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
        total = 0

        if args.kind in ("both", "sec"):
            client = EdgarClient()
            client.verify_access()
            end = args.end or date.today()
            start = args.start or (end - timedelta(days=args.days))
            days = [start + timedelta(days=i)
                    for i in range((end - start).days + 1)
                    if (start + timedelta(days=i)).weekday() < 5]
            days.reverse()
            forms = FORMS_13F + FORMS_13DG
            for day in days:
                if deadline and time.monotonic() > deadline:
                    print("time budget reached; remaining days left for next run",
                          flush=True)
                    break
                marker = "sec_holdings"
                if not args.force and store.day_ingested(marker, day):
                    continue
                events = []
                for parsed in ingest_day(client, day, forms):
                    try:
                        made = (list(parsed.to_events())
                                if hasattr(parsed, "to_events") else [parsed.to_event()])
                        events.extend(made)
                    except Exception:  # noqa: BLE001
                        continue
                new = store.record_many(events)
                if events:
                    store.mark_day_ingested(marker, day, len(events))
                total += new
                if events:
                    print(f"{day}  13F/13D events {len(events):6d}  new {new:6d}",
                          flush=True)

        if args.kind in ("both", "house"):
            import requests
            from .research.edgar import _user_agent
            ua = {"User-Agent": _user_agent()}
            for year in range(date.today().year, date.today().year - args.years, -1):
                if deadline and time.monotonic() > deadline:
                    break
                try:
                    blob = requests.get(HOUSE_BULK.format(year=year),
                                        headers=ua, timeout=120)
                    if blob.status_code != 200:
                        continue
                    rows = [r for r in parse_house_index(blob.content) if r.is_ptr]
                except Exception as exc:  # noqa: BLE001
                    print(f"{year}: House index unavailable ({exc})", flush=True)
                    continue
                print(f"{year}: {len(rows)} periodic transaction reports", flush=True)
                for row in rows:
                    if deadline and time.monotonic() > deadline:
                        break
                    marker = f"house_{row.doc_id}"
                    if not args.force and store.day_ingested(marker, row.filed):
                        continue
                    try:
                        pdf = requests.get(row.pdf_url, headers=ua, timeout=60)
                        if pdf.status_code != 200:
                            continue
                        trades = parse_house_ptr(extract_pdf_text(pdf.content), row)
                    except Exception:  # noqa: BLE001
                        # One unreadable PDF must not end the run; the doc stays
                        # unmarked so a later invocation retries it.
                        continue
                    events = [t.to_event() for t in trades]
                    total += store.record_many(events)
                    store.mark_day_ingested(marker, row.filed, len(events))

        print(f"\ntotal new events: {total}")
        print(f"event store total: {store.count()}")
        store.close()
        return 0
    finally:
        lock.release()


def cmd_runlog(args: argparse.Namespace) -> int:
    """Record a step outcome, or report pipeline health and fail if it is bad.

    Two failure modes this closes. A run that never happens -- GitHub delays and
    drops scheduled runs under load, and disables them entirely after 60 days
    without repository activity, none of which raises a failure. And a run that
    happens while doing nothing, because eight of twenty steps carry
    `continue-on-error` and can fail on every run for weeks behind a green badge.
    """
    from .research.runlog import RunLog, describe

    log = RunLog(DEFAULT_STATE / RUNLOG_DB)
    try:
        if args.action == "start":
            log.start(args.run_id)
            gap = log.hours_since_last_run()
            if gap is not None:
                print(f"last completed run {gap:.1f}h ago")
            return 0

        if args.action == "record":
            log.record(args.run_id, args.step, args.outcome)
            return 0

        if args.action == "finish":
            log.finish(args.run_id)
            return 0

        text, unhealthy = describe(log)
        print(text)
        if unhealthy and args.strict:
            print("\npipeline health check FAILED", file=sys.stderr)
            return 1
        return 0
    finally:
        log.close()


def cmd_ingest_macro(args: argparse.Namespace) -> int:
    """Refresh the daily geopolitical risk series.

    One download covering 1985 to now, so this costs seconds and is worth
    running every pass. The series is republished when its methodology moves,
    which is why every row carries the time it was fetched.
    """
    from .research.macro import MacroStore, fetch_gpr

    store = MacroStore(DEFAULT_STATE / MACRO_DB)
    rows = fetch_gpr()
    n = store.put_many(rows)
    span = store.span()
    print(f"geopolitical risk: {n:,} daily observations")
    if span:
        print(f"  covering {span[0]} to {span[1]}")
    print(f"  fetched at {store.fetched_at()}")
    store.close()
    return 0


def cmd_ingest_assets(args: argparse.Namespace) -> int:
    """Refresh the local asset catalog and report universe composition.

    One request returns all 33,468 US equities including inactive ones, so this
    costs seconds and is worth running every pipeline pass -- an asset's status
    changes and the current answer is the one wanted.
    """
    from .research.assets import (
        RESOLVE_PER_MINUTE,
        UNKNOWN,
        AssetCatalog,
        describe,
        fetch_assets,
        resolve_unknown,
    )
    from .research.eventstore import EventStore

    catalog = AssetCatalog(DEFAULT_STATE / ASSETS_DB)
    if not args.resolve_only:
        n = catalog.put_many(fetch_assets())
        print(f"catalog refreshed: {n:,} US equities")

    events_path = DEFAULT_STATE / EVENTS_DB
    if not events_path.exists():
        catalog.close()
        return 0

    store = EventStore(events_path)
    symbols = [
        r[0] for r in store._conn.execute(
            "SELECT DISTINCT symbol FROM events "
            "WHERE kind = ? AND symbol IS NOT NULL", (KIND_INSIDER,))
    ]
    store.close()

    print(f"\nuniverse composition ({len(symbols):,} distinct symbols):")
    print(describe(catalog.breakdown(symbols)))

    # The bulk list omits recently delisted names -- the exact bucket a
    # survivorship check depends on. Without this pass the composition above
    # reports them as `unknown` and survivorship reads far too high.
    if args.resolve:
        absent = catalog.known_absent()
        pending = sum(1 for s in symbols
                      if catalog.classify(s) == UNKNOWN and s.upper() not in absent)
        print(f"\nresolving {pending:,} unclassified symbols individually "
              f"(~{pending / RESOLVE_PER_MINUTE:.0f} min at "
              f"{RESOLVE_PER_MINUTE}/min)")
        stats = resolve_unknown(
            catalog, symbols, limit=args.resolve_limit,
            on_progress=lambda i, n, st: print(
                f"  {i:,}/{n:,}  resolved {st['resolved']:,}  "
                f"absent {st['absent']:,}", flush=True))
        print(f"  resolved {stats['resolved']:,}, confirmed absent "
              f"{stats['absent']:,}, failed {stats['failed']:,}")
        print("\nuniverse composition after resolution:")
        print(describe(catalog.breakdown(symbols)))

    catalog.close()
    return 0


def cmd_repair_symbols(args: argparse.Namespace) -> int:
    """Re-normalise issuer tickers already in the event store.

    `normalise_symbol` runs at ingest, but the store holds millions of events
    written before it existed, and re-ingesting them costs hours. This rewrites
    the symbol column in place instead.

    Safe to re-run: normalisation is idempotent, and `external_id` is untouched
    so nothing about event identity or the point-in-time guarantee changes.
    Symbols that normalise to nothing are left exactly as they are -- they are
    already excluded from the queue by the same validity check, and blanking
    them would destroy the only record of what the filer actually wrote.
    """
    from .research.edgar import normalise_symbol
    from .research.eventstore import EventStore

    store = EventStore(DEFAULT_STATE / EVENTS_DB)
    conn = store._conn
    rows = list(conn.execute(
        "SELECT DISTINCT symbol FROM events WHERE symbol IS NOT NULL"))

    repairs: list[tuple[str, str]] = []
    unusable: list[str] = []
    for (raw,) in rows:
        fixed = normalise_symbol(raw)
        if not fixed:
            if raw:
                unusable.append(raw)
        elif fixed != raw:
            repairs.append((raw, fixed))

    print(f"{len(rows):,} distinct symbols")
    print(f"  {len(repairs):,} need repair")
    print(f"  {len(unusable):,} are unusable and stay as filed")

    moved = 0
    for raw, fixed in repairs:
        n = conn.execute("SELECT COUNT(*) FROM events WHERE symbol = ?",
                         (raw,)).fetchone()[0]
        moved += n
        if args.dry_run:
            print(f"    {raw!r:<24} -> {fixed:<10} {n:>6,} events")
            continue
        conn.execute("UPDATE events SET symbol = ? WHERE symbol = ?",
                     (fixed, raw))
    if not args.dry_run:
        conn.commit()

    verb = "would move" if args.dry_run else "moved"
    print(f"\n{verb} {moved:,} events onto a usable ticker")
    if unusable and args.dry_run:
        print(f"\nunusable (left alone): {', '.join(repr(u) for u in unusable[:20])}")
    store.close()
    return 0


def cmd_ingest_fundamentals(args: argparse.Namespace) -> int:
    """Cache XBRL company facts for every issuer we hold events on.

    Measurement has to be offline: the SEC allows 8 requests a second, and
    fetching inside a backtest would both take hours and make a result's
    coverage depend on when it was run rather than on what was filed.

    One request per issuer covers every concept and every period they have ever
    reported, so this is a few thousand requests once, then incremental.
    """
    from .lock import SingleInstance
    from .research.eventstore import EventStore
    from .research.fundamentals import FactsCache, XbrlClient

    # Shares the "ingest" lock: this hits the same SEC host under the same 10
    # req/s ceiling as the EDGAR ingests, and two of them at once would breach
    # it. `acquire` returns None and raises LockHeld, which `main` catches --
    # writing `if not lock.acquire()` here would be true on every single run and
    # the command would never execute.
    lock = SingleInstance("ingest", DEFAULT_STATE)
    lock.acquire()
    try:
        store = EventStore(DEFAULT_STATE / EVENTS_DB)
        cutoff = datetime.now(timezone.utc)
        since = cutoff - timedelta(days=args.days)
        ciks: list[str] = []
        seen: set[str] = set()
        for row in store.as_of(cutoff, kind=KIND_INSIDER, since=since):
            cik = str(row["payload"].get("issuer_cik") or "").lstrip("0")
            if cik and cik not in seen:
                seen.add(cik)
                ciks.append(cik)
        store.close()
        print(f"{len(ciks):,} distinct issuers in the last {args.days} days")

        cache = FactsCache(DEFAULT_STATE / FACTS_DB)
        client = XbrlClient()
        deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
        fetched = skipped = failed = 0
        for i, cik in enumerate(ciks):
            if deadline and time.monotonic() > deadline:
                print("time budget reached; the cache is incremental")
                break
            if not args.force and cache.has(cik):
                skipped += 1
                continue
            try:
                raw = client.company_facts(cik)
            except Exception as exc:  # noqa: BLE001
                # One unreachable issuer must not end the run. It stays
                # uncached, so the next invocation retries it.
                failed += 1
                if failed <= 5:
                    print(f"  CIK {cik}: {type(exc).__name__}: {exc}")
                continue
            if raw:
                cache.put(cik, raw)
                fetched += 1
            if fetched and fetched % 250 == 0:
                print(f"  {i + 1:,}/{len(ciks):,}  cached {fetched:,}", flush=True)

        print(f"\nfetched {fetched:,}, already held {skipped:,}, failed {failed:,}")
        print(f"cache now holds {cache.count():,} issuers")
        cache.close()
        return 0
    finally:
        lock.release()


def cmd_measure(args: argparse.Namespace) -> int:
    """Run the backlog against the labelled event population.

    This is the command everything else exists to feed, and it did not exist.
    Every module below it was built and tested -- the labeller, the cost model,
    the clustering corrections, the trial registry, the sweep -- and nothing
    called them together, so the honest description of the project's state was
    not "nothing has been measured yet because coverage is thin". It was "there
    is no code path that measures anything".

    Five stages, each of which can fail loudly rather than quietly degrading:

      load      events the store says were knowable at the cutoff
      split     chronologically; the holdout stays sealed unless declared
      label     forward returns from cached bars only, never a live fetch
      enrich    point-in-time indicator values, evaluated before the entry bar
      sweep     every candidate at every horizon, with costs charged per trade
    """
    from .research.candidates import all_candidates
    from .research.costs import CostModel, CostTable
    from .research.eventstore import EventStore
    from .research.features import FeatureBuilder
    from .research.labeler import Labeller, coverage_report
    from .research.prices import BASIS_PRICE, BASIS_TOTAL, CachedOnlySource, PriceCache
    from .research.splits import chronological_split, filter_events
    from .research.sweep import DEFAULT_HORIZONS, SweepError, priors_vs_outcomes
    from .research.sweep import report as sweep_report
    from .research.sweep import sweep
    from .research.trials import TrialRegistry

    events_path = DEFAULT_STATE / EVENTS_DB
    bars_path = DEFAULT_STATE / BARS_DB
    for path, what in ((events_path, "event store"), (bars_path, "price cache")):
        if not path.exists():
            print(f"no {what} at {path}; run the ingest and backfill first",
                  file=sys.stderr)
            return 2

    cutoff = args.as_of or datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    window_start = cutoff - timedelta(days=args.window)

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    if not horizons:
        horizons = DEFAULT_HORIZONS

    store = EventStore(events_path)
    rows = list(store.as_of(cutoff, kind=args.kind, since=window_start))
    store.close()
    print(f"{len(rows):,} {args.kind} events knowable at {cutoff.date()} "
          f"within {args.window} days")
    if not rows:
        print("nothing to measure", file=sys.stderr)
        return 1

    # Split the range the data actually covers, not the range that was asked
    # for. Splitting the request instead is degenerate whenever ingestion has
    # not caught up with it: asking for 3800 days against a store holding two
    # years of recent filings put every single event in the holdout and left
    # train empty, which reads like "no events" rather than "the window is
    # wrong".
    #
    # The cost is that the boundaries move as history accumulates, so "train"
    # is not the same set between runs. That is tolerable for train and
    # validation and is NOT tolerable for the holdout, which is why
    # --split-start and --split-end exist: pin them before declaring a finalist,
    # and the sealed window stops drifting under it.
    observed = [datetime.fromisoformat(r["observed_at"]) for r in rows]
    split_start = args.split_start or min(observed).date()
    split_end = args.split_end or max(observed).date()
    split = chronological_split(split_start, split_end)
    print(split.describe())
    if not (args.split_start and args.split_end):
        print("  (boundaries derived from the data range; pin them with "
              "--split-start/--split-end before any holdout work)")

    rows = filter_events(rows, split, args.partition)
    print(f"{len(rows):,} in the {args.partition} partition")
    if not rows:
        print(f"no events in the {args.partition} partition", file=sys.stderr)
        return 1
    if args.limit:
        # Head, not a sample: the split is chronological and taking a random
        # subset would scatter the trades across the window while claiming the
        # sample size of a contiguous one.
        rows = rows[: args.limit]
        print(f"limited to the first {len(rows):,}")

    events = [
        {"symbol": r["symbol"], "observed_at": r["observed_at"]} for r in rows
    ]
    payloads = [r["payload"] for r in rows]

    basis = BASIS_TOTAL if args.basis == "total" else BASIS_PRICE
    cache = PriceCache(bars_path)
    source = CachedOnlySource(cache, basis=basis)

    started = time.monotonic()
    labels = Labeller(source, horizons=horizons).label(events)
    print(f"labelled in {time.monotonic() - started:.0f}s -- {source.summary()}")

    cov = coverage_report(labels)
    print("coverage: " + ", ".join(f"{k}={v}" for k, v in cov.items()))

    # Coverage BY CLASSIFICATION, because the aggregate number hides the shape
    # of the loss. Delisted issuers stay in the event store but their prices
    # were often never fetched, so they label NO_DATA and drop out silently --
    # and the names that drop out are exactly the ones whose absence biases a
    # backtest upward. An aggregate 40% coverage made of 95% listed and 2%
    # delisted is a survivorship problem; the same 40% spread evenly is not.
    assets_path = DEFAULT_STATE / ASSETS_DB
    if assets_path.exists():
        from .research.assets import AssetCatalog

        catalog = AssetCatalog(assets_path)
        if catalog.count():
            buckets: dict[str, list[int]] = {}
            for lab in labels:
                cls = catalog.classify(lab.symbol) if lab.symbol else "unknown"
                seen, ok = buckets.setdefault(cls, [0, 0])
                buckets[cls] = [seen + 1, ok + (1 if lab.returns else 0)]
            print("labelled share by classification:")
            for cls in sorted(buckets):
                seen, ok = buckets[cls]
                print(f"  {cls:<10} {ok:>7,} of {seen:>7,}  "
                      f"({ok / seen if seen else 0:>6.1%})")
            listed = buckets.get("listed", [0, 0])
            delisted = buckets.get("delisted", [0, 0])
            lr = listed[1] / listed[0] if listed[0] else 0.0
            dr = delisted[1] / delisted[0] if delisted[0] else 0.0
            if listed[0] and delisted[0] and lr > 0 and dr < lr * 0.5:
                print(f"  WARNING: delisted names label at {dr:.1%} against "
                      f"{lr:.1%} for listed ones. The backtest population is "
                      "skewed toward survivors by roughly that gap, and returns "
                      "measured on it are biased upward by an amount nothing "
                      "here can estimate. Backfill the delisted symbols.")
        catalog.close()

    # The specific trap this catches: the default basis is total-return on the
    # merits, but a cache filled before the dual-basis change holds price-only
    # bars. Every symbol then misses, coverage collapses, and the sweep reports
    # "too few trades" across the board -- which looks like a finding about the
    # strategies and is a finding about the cache.
    other = BASIS_PRICE if basis == BASIS_TOTAL else BASIS_TOTAL
    held, held_other = len(cache.symbols(basis)), len(cache.symbols(other))
    if held_other > held * 2:
        print(f"\nWARNING: the cache holds {held:,} symbols on the {basis} basis "
              f"and {held_other:,} on {other}. This run is measuring the thinner "
              f"one.\n  Either pass --basis {other}, or refill with "
              f"`backfill --requeue`, which is what makes 'done' mean both "
              f"bases.\n  Low coverage below is about the cache, not about the "
              f"strategies.")

    if args.features:
        started = time.monotonic()
        builder = FeatureBuilder(cache, basis=basis)
        payloads = builder.enrich(payloads, labels)
        print(f"enriched in {time.monotonic() - started:.0f}s -- "
              f"{builder.summary()}")

    # The other three data families. Each was blocked for one reason -- no path
    # from the family to the payload -- and each is unblocked by the same shape
    # of join. Every one is optional and silent when its store is absent, so a
    # partial state produces fewer features rather than a crash.
    if args.joins:
        from .research.joins import (
            FundamentalsJoin,
            HoldingsJoin,
            MacroJoin,
            ProfileJoin,
            enrich_all,
        )

        started = time.monotonic()
        active = []

        profiles_path = DEFAULT_STATE / PROFILES_DB
        if profiles_path.exists():
            from .research.intraday import ProfileStore
            pstore = ProfileStore(profiles_path)
            if pstore.count():
                active.append(ProfileJoin(pstore, price_cache=cache, basis=basis))
            else:
                pstore.close()
                print("profiles: store is empty; run backfill-intraday")
        else:
            print("profiles: no store yet; run backfill-intraday")

        hstore = EventStore(events_path)
        active.append(HoldingsJoin(hstore))

        macro_path = DEFAULT_STATE / MACRO_DB
        if macro_path.exists():
            from .research.macro import MacroStore
            mstore = MacroStore(macro_path)
            if mstore.count():
                active.append(MacroJoin(mstore))
            else:
                mstore.close()
                print("macro: store is empty; run ingest-macro")
        else:
            print("macro: no store yet; run ingest-macro")

        facts_path = DEFAULT_STATE / FACTS_DB
        if facts_path.exists():
            from .research.fundamentals import FactsCache
            fcache = FactsCache(facts_path)
            if fcache.count():
                active.append(FundamentalsJoin(fcache, price_cache=cache,
                                               basis=basis))
            else:
                fcache.close()
                print("fundamentals: cache is empty; run ingest-fundamentals")
        else:
            print("fundamentals: no cache yet; run ingest-fundamentals")

        payloads = enrich_all(payloads, labels, *active)
        print(f"joined in {time.monotonic() - started:.0f}s")
        for join in active:
            print(f"  {join.summary()}")
        hstore.close()

    costs = None
    if args.costs:
        # `capital` turns the cost model from a spread-only estimate into one
        # that charges market impact. Without a position size, participation is
        # zero and impact is zero, so every net figure was an upper bound on
        # performance rather than an estimate of it. Sizing is still crude --
        # equal notional per trade -- but a stated crude size beats an implied
        # size of zero.
        costs = CostTable(cache, model=CostModel(), basis=basis,
                          capital_per_trade=args.capital)

    # Identity of this dataset, so a nightly re-run of the same sweep updates
    # its trials instead of appending new ones. Without this the Deflated Sharpe
    # bar climbed every night with no new hypothesis having been tested.
    #
    # The fingerprint deliberately includes what would make the same hypothesis
    # a genuinely new look: the events measured, the window, the partition, the
    # price basis and the horizons. Change any of those and it IS a new trial.
    fingerprint = hashlib.sha256("|".join([
        str(len(rows)), str(min(observed).date()), str(max(observed).date()),
        args.partition, basis, args.horizons, args.kind,
        str(args.features), str(args.joins), str(args.costs),
    ]).encode("utf-8")).hexdigest()[:16]
    print(f"dataset fingerprint: {fingerprint}")

    registry = TrialRegistry(DEFAULT_STATE / TRIALS_DB)
    cands = all_candidates(with_features=args.features,
                           with_joins=args.joins)
    runnable = sum(1 for c in cands if c.runnable)
    print(f"\nsweeping {runnable} candidates x {len(horizons)} horizons "
          f"= {runnable * len(horizons)} trials, on top of "
          f"{registry.count()} already registered")

    try:
        # Coverage is the labellable share -- the honest denominator for "what
        # population does this result describe". fallback_share is filled after
        # the sweep, since CostTable only learns it while charging.
        coverage = float(cov.get("complete_rate", 0.0) or 0.0)
        assessments = sweep(
            cands, labels, payloads, registry=registry, horizons=horizons,
            costs=costs, partition=args.partition, dataset=fingerprint,
            coverage=coverage,
            fallback_share=(costs.fallback_rate() if costs else 0.0),
        )
    except SweepError as exc:
        print(str(exc), file=sys.stderr)
        registry.close()
        cache.close()
        return 2

    print()
    print(sweep_report(assessments, cands))
    print()
    if costs is not None:
        print(costs.summary())
    else:
        print("UNCOSTED: --no-costs was passed. Gross returns only, and a gross "
              "edge on this universe is not a result.")
    print()
    print(priors_vs_outcomes(assessments, cands))

    registry.close()
    cache.close()
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
    enq.add_argument("--skip-otc", action="store_true",
                     help="drop symbols the asset catalog marks OTC. Off by "
                          "default on purpose: a name on OTC today may have "
                          "been listed during the window, and its history is "
                          "real. Skipping on today's tag deletes data we hold "
                          "and introduces survivorship bias in the cleanup step")
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
    itd.add_argument("--refresh-untimed", action="store_true",
                     help="refetch sessions stored before the session-sequence "
                          "fields existed. They cannot be repaired in place -- "
                          "minute bars are not kept -- and `status` reports how "
                          "many are outstanding")
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

    hld = sub.add_parser("ingest-holdings",
                         help="13F holdings, 13D/G stakes, House congress trades")
    hld.add_argument("--kind", choices=("both", "sec", "house"), default="both")
    hld.add_argument("--days", type=int, default=120,
                     help="how far back to scan EDGAR daily indexes")
    hld.add_argument("--years", type=int, default=2,
                     help="how many years of House bulk indexes to walk")
    hld.add_argument("--start", type=date.fromisoformat)
    hld.add_argument("--end", type=date.fromisoformat)
    hld.add_argument("--max-minutes", type=int, default=0)
    hld.add_argument("--force", action="store_true")
    hld.set_defaults(func=cmd_ingest_holdings)

    ms = sub.add_parser("measure",
                        help="run the whole candidate backlog against labelled "
                             "events and print the verdicts")
    ms.add_argument("--as-of", type=datetime.fromisoformat,
                    help="point-in-time cutoff; nothing observed after this is "
                         "visible. Defaults to now")
    ms.add_argument("--window", type=int, default=PRICE_WINDOW_DAYS,
                    help=f"days of history to measure over (default "
                         f"{PRICE_WINDOW_DAYS}, the price window)")
    ms.add_argument("--kind", default=KIND_INSIDER,
                    help="event kind to measure (default insider_transaction)")
    ms.add_argument("--partition", default="train",
                    choices=("train", "validation", "holdout"),
                    help="holdout is refused unless every candidate has a "
                         "recorded unlock; that is deliberate")
    ms.add_argument("--split-start", type=date.fromisoformat,
                    help="pin the split's first day. Without it the boundaries "
                         "follow the data and move as history accumulates, "
                         "which a sealed holdout cannot tolerate")
    ms.add_argument("--split-end", type=date.fromisoformat,
                    help="pin the split's last day")
    ms.add_argument("--horizons", default="1,5,20",
                    help="comma-separated holding periods in sessions")
    ms.add_argument("--basis", choices=("total", "price"), default="total",
                    help="adjustment basis; total-return is the default because "
                         "a price-only series fabricates a loss on every "
                         "ex-dividend date")
    ms.add_argument("--limit", type=int, default=0,
                    help="cap events measured, taken chronologically from the "
                         "start of the partition")
    ms.add_argument("--no-features", dest="features", action="store_false",
                    help="skip indicator enrichment. Drops the indicator "
                         "candidates from the sweep entirely rather than "
                         "reporting them as zero-trade measurements")
    ms.add_argument("--capital", type=float, default=25_000.0,
                    help="notional per position, used to charge market impact. "
                         "Zero leaves participation at zero, which makes every "
                         "net return an upper bound rather than an estimate")
    ms.add_argument("--no-joins", dest="joins", action="store_false",
                    help="skip the intraday, holdings and fundamentals joins. "
                         "Their candidates then report zero trades, which reads "
                         "as measured-and-empty rather than not-joined")
    ms.add_argument("--no-costs", dest="costs", action="store_false",
                    help="do not charge transaction costs. Every result is then "
                         "gross, and no gross result survives the cost gate by "
                         "default -- this is for diagnosing the sweep, not for "
                         "producing a finding")
    ms.set_defaults(func=cmd_measure, features=True, costs=True, joins=True)

    rl = sub.add_parser("runlog",
                        help="record step outcomes and check pipeline health")
    rl.add_argument("action", choices=("start", "record", "finish", "report"))
    rl.add_argument("--run-id", default="local")
    rl.add_argument("--step", default="")
    rl.add_argument("--outcome", default="success")
    rl.add_argument("--strict", action="store_true",
                    help="exit non-zero when the pipeline is unhealthy, so a "
                         "silently broken run stops reporting success")
    rl.set_defaults(func=cmd_runlog)

    mac = sub.add_parser("ingest-macro",
                         help="refresh the daily geopolitical risk series "
                              "(Caldara & Iacoviello, 1985 onward)")
    mac.set_defaults(func=cmd_ingest_macro)

    ast = sub.add_parser("ingest-assets",
                         help="refresh the asset catalog; classifies the "
                              "universe as listed / delisted / OTC / unknown")
    ast.add_argument("--resolve", action="store_true",
                     help="look up symbols the bulk list omits, one request "
                          "each. The bulk list drops recently delisted names, "
                          "which is exactly the bucket survivorship depends on")
    ast.add_argument("--resolve-only", action="store_true",
                     help="skip the bulk refresh and only resolve gaps")
    ast.add_argument("--resolve-limit", type=int, default=0,
                     help="cap lookups this run; the catalog is incremental")
    ast.set_defaults(func=cmd_ingest_assets)

    rep = sub.add_parser("repair-symbols",
                         help="re-normalise issuer tickers already stored")
    rep.add_argument("--dry-run", action="store_true",
                     help="report what would change without writing")
    rep.set_defaults(func=cmd_repair_symbols)

    fnd = sub.add_parser("ingest-fundamentals",
                         help="cache SEC XBRL company facts for held issuers")
    fnd.add_argument("--days", type=int, default=PRICE_WINDOW_DAYS,
                     help="how far back to collect issuers from the event store")
    fnd.add_argument("--max-minutes", type=float, default=0,
                     help="stop cleanly after N minutes; the cache is incremental")
    fnd.add_argument("--force", action="store_true",
                     help="refetch issuers already cached")
    fnd.set_defaults(func=cmd_ingest_fundamentals)

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
