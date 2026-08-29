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
KIND_INSIDER = "insider_transaction"

#: Free-tier price history ceiling. Events older than this can never be
#: labelled, because no price data exists to measure their outcome.
PRICE_WINDOW_DAYS = 730

#: How much EDGAR history to ingest by default. Deliberately deeper than the
#: price window: the routine/opportunistic classifier needs 3+ years of an
#: insider's prior filings, so a 2-year ingest leaves every insider UNKNOWN and
#: the filter carrying the actual edge never fires. EDGAR history is free and
#: unbounded; the price window is not. These two limits are unrelated and must
#: not be collapsed into one number.
BASELINE_DAYS = 1825  # ~5 years

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
    cutoff = args.before or (date.today() - timedelta(days=PRICE_WINDOW_DAYS))

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

    runner = _make_runner()
    added = runner.enqueue(symbols)
    print(
        f"{all_events} stored events; {len(labellable)} inside the "
        f"{args.price_window}-day price window"
    )
    print(f"{len(symbols)} distinct labellable symbols; {added} newly queued")
    print(runner.progress())
    runner.close()
    return 0


def _make_runner(limit_per_minute: int | None = None):
    from .research.backfill import BackfillRunner
    from .research.prices import DEFAULT_REQUESTS_PER_MINUTE, MassivePriceSource, PriceCache

    source = MassivePriceSource(
        cache=PriceCache(DEFAULT_STATE / BARS_DB),
        per_minute=limit_per_minute or DEFAULT_REQUESTS_PER_MINUTE,
    )
    end = date.today()
    return BackfillRunner(
        source,
        DEFAULT_STATE / CHECKPOINT_DB,
        start=end - timedelta(days=PRICE_WINDOW_DAYS),
        end=end,
    )


def cmd_backfill(args: argparse.Namespace) -> int:
    from .lock import SingleInstance

    # Concurrent backfills would double the request rate and race on the cache.
    with SingleInstance("backfill", DEFAULT_STATE):
        return _run_backfill(args)


def _run_backfill(args: argparse.Namespace) -> int:
    runner = _make_runner(args.per_minute)
    runner.install_signal_handlers()

    def report(symbol: str, prog) -> None:
        print(f"[{prog.done + prog.failed:5d}] {symbol:8s} {prog}", flush=True)

    print(f"starting: {runner.progress()}")
    prog = runner.run(limit=args.limit, on_progress=report)
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
    from .research.crosscheck import compare, summarise
    from .research.prices import AlpacaPriceSource, PriceCache

    cache = PriceCache(DEFAULT_STATE / BARS_DB)
    symbols = cache.symbols()[: args.limit] if args.limit else cache.symbols()
    if not symbols:
        print("no cached symbols to compare; run backfill first")
        return 1

    alpaca = AlpacaPriceSource(per_minute=args.per_minute)
    end = date.today()
    start = end - timedelta(days=PRICE_WINDOW_DAYS)

    results, worst = [], []
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
        cache = PriceCache(bars_path)
        print(f"symbols cached   : {len(cache.symbols())}")
        cache.close()
    else:
        print("symbols cached   : none")

    ckpt = DEFAULT_STATE / CHECKPOINT_DB
    if ckpt.exists():
        runner = _make_runner()
        print(f"backfill         : {runner.progress()}")
        print(f"parked failures  : {len(runner.failures())}")
        runner.close()
    else:
        print("backfill         : not started")
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
    bf.set_defaults(func=cmd_backfill)

    xc = sub.add_parser("crosscheck",
                        help="compare Massive vs Alpaca on cached symbols")
    xc.add_argument("--limit", type=int, help="stop after N symbols")
    xc.add_argument("--per-minute", type=int, default=180,
                    help="Alpaca allows 200/min on the free plan")
    xc.set_defaults(func=cmd_crosscheck)

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
