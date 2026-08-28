# TradezBotz

A research-first trading system. The order of work is deliberate:

**Phase 0 — measure** (in progress) · Phase 1 — backtest · Phase 2 — combine · Phase 3 — paper · Phase 4 — live, small

Nothing trades until a signal has survived the phase before it.

## Why this order

Paper trading validates *plumbing* — do orders fill, does reconnect work, are
halts handled. It does not validate *edge*. Event-driven signals are trade-sparse:
two weeks might produce 10–20 trades, and at that sample size a 55% hit rate is
statistically indistinguishable from a coin flip. Edge validation needs years of
history and hundreds of events, which is what Phase 0 and 1 are for.

## The two rules the code enforces

**1. Point-in-time or nothing.** A backtest evaluating time *T* may only see rows
whose `observed_at <= T`. For our signals the gap between when something happened
and when it became knowable is large and systematic:

| Signal | Happens | Becomes public |
|---|---|---|
| SEC Form 4 | transaction date | within 2 business days |
| Congressional PTR | transaction date | 45-day deadline; **median 26 days** |

Keying a backtest on transaction date instead of dissemination time manufactures
returns that were never available to anyone. `research/eventstore.py` exists
solely to make that mistake hard, and `tests/test_eventstore.py` pins the
behaviour.

**2. Count your trials.** Sweep 200 strategy variants and several will look
excellent by luck. The [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
(Bailey & López de Prado) corrects for this, but requires the *number of trials*
as an input — so every backtest run must be logged. That is an architectural
requirement, not a reporting step.

## What exists now

```
src/tradezbotz/research/
  eventstore.py   Point-in-time, append-only event store. Revisions never
                  leak backwards, so restatements can't rewrite the past.
  edgar.py        SEC Form 4 ingestion. Separates transaction date from
                  dissemination time, honours the 22:00 ET Form 4 cutoff and
                  rolls late filings to the next business morning.
  classify.py     Routine vs opportunistic insider classification, after
                  Cohen, Malloy & Pomorski.
  prices.py       PriceSource protocol + Massive adapter. Rate limiter, on-disk
                  bar cache (reproducibility: a dividend paid tomorrow must not
                  change yesterday's backtest), and delisting detection.
  labeler.py      Forward returns. Entry is the next tradeable OPEN, never the
                  signal-day close; delistings are recorded, never dropped.
  backfill.py     Resumable, checkpointed symbol queue. Survives Ctrl-C, VM
                  reboot and vendor outages; one bad ticker cannot end a run.
src/tradezbotz/
  cli.py          ingest-edgar / enqueue-symbols / backfill / status
  config.py       Config + the three-signal live-trading gate (used from Phase 3).
.github/workflows/
  pipeline.yml    Scheduled sliced run: restore state -> ingest -> backfill ->
                  re-encrypt -> save state. The hosting path in use.
deploy/
  setup_vm.sh     Idempotent provisioning for a VM (GCP e2-micro, Pi, or any
                  Debian box). Not required for the Actions path.
  *.service       systemd units: daily ingest timer + long-running backfill.
  README.md       VM deploy walkthrough, for Phase 3 or if you outgrow Actions.
```

## How it runs

The pipeline runs on **GitHub Actions**, not a server. Each scheduled run takes a
bounded bite of work and hands state to the next run through the Actions cache.
This works only because both long jobs are checkpointed: EDGAR ingestion by day,
the price backfill by symbol.

Constraints this design is shaped around:

| Limit | Consequence |
|---|---|
| 6h job cap | runs are time-boxed to 330 min, well clear of it |
| cache is immutable | each run writes a new key, restores newest by prefix |
| cache evicted after 7 days unused | daily schedule keeps it warm |
| schedules delayed/dropped at peak | cron is at :17, not the top of the hour |

**State is encrypted before it leaves the runner.** This repo is public, and
Actions caches and artifacts on a public repo are readable by anyone.
`state/bars.db` holds vendor-licensed price data that must not be redistributed,
so the whole archive is AES256-encrypted with a passphrase held in repository
secrets.

### Required repository secrets

| Secret | Value |
|---|---|
| `SEC_USER_AGENT` | contact string with a real email; the SEC blocks requests without one |
| `MASSIVE_API_KEY` | Massive REST key |
| `STATE_PASSPHRASE` | any long random string; **losing it means losing all accumulated state** |

### Why the routine/opportunistic split matters

Cohen, Malloy & Pomorski ([NBER w16454](https://www.nber.org/system/files/working_papers/w16454/w16454.pdf))
found that insiders who trade the same calendar month year after year carry
essentially no information — that's tax and diversification behaviour. The
abnormal returns lived entirely in trades that *broke* an insider's own pattern
(82bp/month value-weighted, 180bp/month equal-weighted).

Two caveats are baked into the code as comments: the sample is 1986–2007, so
assume material decay since publication; and the equal-weighted figure implies
small caps, where spreads and capacity will erode a lot of it. The conviction
weights in `classify.score()` are a **starting hypothesis to be tested, not a
fitted model**.

## Data sources — what the free tier actually gives us

Verified against a live key on 2026-08-28, not taken from marketing pages:

| Capability | Status |
|---|---|
| REST daily aggregates | works, **capped at exactly 2 years** (498 trading days) |
| REST rate limit | 5 req/min, enforced hard (429) |
| Delisted tickers via `/v3/reference/tickers?active=false` | returns results |
| Flat Files (S3) listing | works, catalogue visible back to **2003** |
| Flat Files (S3) download | **403 on every object, all eras — paid feature** |

Two consequences:

**The 2-year cap is the binding constraint.** Form 4 volume is high, so two years
still yields tens of thousands of events — sample size on *events* is fine. What
we lose is *regime* diversity: two years is one market environment. A signal that
works only in this regime will look identical to one that works generally, and
nothing in the data can tell them apart. Treat Phase 1 results accordingly.

**Survivorship handling works — confirmed end to end.** `AACB`, delisted
2026-08-20, returns 248 daily bars ending 2026-08-19, and the labeller classifies
an event ten sessions before the delisting as `delisted_during_window` with the
short horizons still resolved. So failed companies stay in the dataset as
countable outcomes instead of vanishing.

One caveat found while proving this: the reference endpoint defaults to
`active=true`, so a delisted ticker returns *zero results* rather than
`active: false`. Trusting the first lookup reports `None`, and the labeller then
downgrades a real delisting to an ordinary coverage gap — reintroducing
survivorship bias at the exact point built to catch it. `is_active()` now falls
back to an explicit inactive lookup, with a regression test.

Flat Files would solve most of this — SIP consolidated data, full history, and
each daily file is a snapshot of everything that traded that day, which is
survivorship-bias-free by construction. It is the obvious first thing to pay
for if this project earns it.

## Setup

```bash
python -m pip install -r requirements.txt
```

The SEC requires a descriptive User-Agent containing a real contact email and
blocks clients that omit it. Set your own — it is sent to sec.gov on every
request, so it is yours to choose:

```bash
cp .env.example .env   # then edit SEC_USER_AGENT
```

```bash
python -m pytest -q
```

## Status

72 tests passing. The price adapter and labeller have been exercised against the
live Massive API end to end (see the delisting result above).

No live EDGAR fetch has been run yet — that needs your `SEC_USER_AGENT` set
first, since the SEC requires a real contact email in the header.

## Not yet built

- Congressional PTR ingester (Senate Stock Watcher works; the House Stock
  Watcher S3 bucket has returned 403 since early 2026, so House data needs the
  Clerk's portal directly)
- News / executive-mention event logger — **logged and measured, not traded**,
  until there's evidence of an edge
- Copy-trading signal source. Collective2 and Darwinex both publish *audited*
  track records and expose APIs, unlike leaderboard apps. Same rule as the news
  leg: ingest positions as events with a real `observed_at`, label forward
  returns, and let the harness decide whether the signal survives — rather than
  trusting a leaderboard that is selected on the outcome being predicted.
- Backtest engine, trial registry, and Deflated Sharpe reporting
- Broker layer, risk guard, live loop (Phases 3–4)

## Scope

This repo builds and tests infrastructure. It does not constitute investment
advice, and none of the numbers cited above are expected returns.
