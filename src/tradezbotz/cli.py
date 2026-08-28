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

#: Free-tier price history ceiling. Ingesting events older than this produces
#: rows that can never be labelled.
HISTORY_DAYS = 730


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
    from .research.edgar import EdgarClient, ingest_day
    from .research.eventstore import EventStore

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))
    if (date.today() - start).days > HISTORY_DAYS:
        print(
            f"note: {start} is beyond the {HISTORY_DAYS}-day price history window; "
            "those events will ingest but cannot be labelled.",
            file=sys.stderr,
        )

    client = EdgarClient()
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

    total_new = skipped = processed = 0
    for day in days:
        if deadline and time.monotonic() > deadline:
            print(f"\ntime budget reached; {len(days) - processed - skipped} days left")
            break
        if not args.force and store.day_ingested("sec_form4", day):
            skipped += 1
            continue
        events = [t.to_event() for t in ingest_day(client, day)]
        new = store.record_many(events)
        store.mark_day_ingested("sec_form4", day, len(events))
        total_new += new
        processed += 1
        print(f"{day}  filings->events {len(events):5d}  new {new:5d}", flush=True)

    print(f"\ndays processed {processed}, already done {skipped}")
    print(f"total new events: {total_new}")
    print(f"event store total: {store.count()}")
    store.close()
    return 0


def cmd_enqueue_symbols(args: argparse.Namespace) -> int:
    from .research.backfill import symbols_from_events
    from .research.eventstore import EventStore

    store = EventStore(DEFAULT_STATE / EVENTS_DB)
    events = list(store.as_of(datetime.now(timezone.utc)))
    symbols = symbols_from_events(events)
    store.close()

    runner = _make_runner()
    added = runner.enqueue(symbols)
    print(f"{len(symbols)} distinct symbols in event store; {added} newly queued")
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
        start=end - timedelta(days=HISTORY_DAYS),
        end=end,
    )


def cmd_backfill(args: argparse.Namespace) -> int:
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
    ing.add_argument("--days", type=int, default=7, help="lookback when --start omitted")
    ing.add_argument("--start", type=date.fromisoformat)
    ing.add_argument("--end", type=date.fromisoformat)
    ing.add_argument("--max-minutes", type=float,
                     help="stop cleanly after N minutes (for sliced CI runs)")
    ing.add_argument("--force", action="store_true",
                     help="re-pull days already marked complete")
    ing.set_defaults(func=cmd_ingest_edgar)

    enq = sub.add_parser("enqueue-symbols", help="queue every symbol seen in events")
    enq.set_defaults(func=cmd_enqueue_symbols)

    bf = sub.add_parser("backfill", help="fetch daily bars for queued symbols")
    bf.add_argument("--limit", type=int, help="stop after N symbols this run")
    bf.add_argument("--per-minute", type=int, help="override the request budget")
    bf.set_defaults(func=cmd_backfill)

    st = sub.add_parser("status", help="show pipeline state")
    st.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    DEFAULT_STATE.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
