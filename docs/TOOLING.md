# Third-party tooling: what we adopted and what we refused

Evaluated 2026-08-30. Each entry records the decision *and the condition that
would reverse it*, because "maybe later" with no trigger is how a library gets
re-litigated every few months.

| project | decision | licence |
| --- | --- | --- |
| OpenBB | **adopted** as a crosscheck referee | AGPL-3.0 |
| awesome-systematic-trading | **adopted** as backlog input | MIT / CC-BY |
| NautilusTrader | deferred to execution | LGPL-3.0 |
| Kronos | refused, with conditions | MIT |
| TradingAgents | **refused on principle** | Apache-2.0 |
| Trading-R1 | refused | unstated |
| OpenAlice | refused, redundant | AGPL-3.0 |
| paperclip | refused, out of scope | MIT |
| worldmonitor | refused as a data source; possible ops surface later | AGPL-3.0 |

---

## Adopted

### OpenBB — third price opinion

Added as `prices.OpenBBPriceSource`, reached by `crosscheck --three-way`.
Optional dependency (`pip install openbb-yfinance`); the pipeline runs without it.

It exists to break a deadlock. Two sources can only say "one of you is wrong."
For weeks that left Alpaca's deeper history unusable *on principle rather than
on evidence*, because we could not say which vendor to believe.

It paid for itself immediately, and not in the way expected — see
[PRICE-BASIS.md](PRICE-BASIS.md). The headline: of 24 disputed symbols, **15
were not errors at all**, and the naive two-source read had been calling them
data-quality failures.

Not a candidate for system of record. Yahoo is derived and unaudited; the only
property being used is that it is *independent* of the other two.

**Licence:** AGPL copyleft triggers on distribution or network service. We do
neither — private repo, CI job, no users. If that changes, this needs revisiting.

### awesome-systematic-trading — backlog input

A curated index, not code, so there is nothing to depend on. Mined into
[STRATEGIES.md](STRATEGIES.md): ten new candidate hypotheses plus two methods
worth stealing (cluster-robust confidence intervals, factor IC analysis).

Zero integration cost, and it directly serves the "test every strategy" rule.

---

## Deferred

### NautilusTrader — revisit at live execution

The most serious engineering of the eight: Rust core, nanosecond event-driven
backtests, and the same strategy code in research and production. That last
property is real value we do not have.

**Why not now.** It would *replace* our backtest engine, and with it the trial
registry, Deflated Sharpe, clustering and winsorisation checks. That machinery
is the actual differentiator here — a faster engine that does not count trials
would be a downgrade, not an upgrade.

**Revisit when all of these hold:**
1. We have a signal that survives the holdout, so execution is the bottleneck
2. We need multi-venue or intraday order precision — for daily next-open entries
   against one broker, driving Alpaca directly is simpler and already built
3. Its Alpaca adapter lands (RFC #3374, open since January 2026)

If only (1) is true, keep our own execution path.

---

## Refused

### Kronos — conditions for reconsidering

Real work: AAAI 2026 paper, 12B K-line records from 45 exchanges, MIT licensed,
and the reported gains over time-series foundation-model baselines are large.

**Why not.** Two reasons, and the second is the binding one:

1. The headline numbers are **RankIC and MAE — forecasting metrics, not P&L**.
   Forecasting skill is necessary but nowhere near sufficient, and the paper
   does not claim otherwise.
2. Neither the repo nor the paper documents a **date-based train/test split**,
   so we cannot establish the model never saw our evaluation window. That is
   unfalsifiable from the outside, which makes it the same category of problem
   as the LLM agents below.

There is also a shape mismatch: it is built for dense sequences, and our signal
is ~4,900 sparse events concentrated in small caps.

**Reconsider if:** the authors publish a dated train/test protocol we can check
against our window, *and* someone demonstrates value on sparse event data rather
than continuous forecasting.

### TradingAgents, Trading-R1 — refused on principle

This is the firmest decision here, and it is worth stating why at length,
because the star count invites revisiting it.

The TradingAgents paper claims no lookahead: agents see only data available up
to each trading day. That is true of the **pipeline** and irrelevant to the
**model**. An LLM with a 2025+ cutoff has already read how every liquid US
equity moved through 2024.

The 2026 literature now treats this as its own failure class. *Look-Ahead-Freedom
as Temporal Non-Interference* (arXiv 2607.04958) calls it "a form of look-ahead
that has no analogue in classical backtesting" and states that **"inspecting the
pipeline is not even sufficient to rule leakage out."**

That is specifically fatal *for us*. Every guardrail in this repo —
`observed_at` vs `occurred_at`, the trial registry, the Deflated Sharpe, the
locked holdout — assumes bias is detectable by auditing the data path. LLM
leakage lives in the weights, and defeats all of it silently. Adopting these
would add an unmeasurable bias to a system whose entire premise is that bias is
measurable.

Worth noting: that paper's "reference time vs availability" separation is
exactly the `occurred_at` / `observed_at` split already in `eventstore.py`. We
arrived at the formalism independently.

Trading-R1 is additionally a placeholder — one commit, no code, no licence.

**This does not forbid LLM use anywhere.** Generating hypotheses, reading
filings, writing code are all fine — the model is not being asked to predict a
price whose answer it may already know. The refusal is narrow: **no LLM output
may be an input to a backtested trading decision.**

### OpenAlice — redundant

A workspace for coding agents to trade: files, issues, market tools,
approval-gated execution. It explicitly "works with Claude Code."

We *are* the coding agent, and this repo already has the workspace, the state
store, and the approval gate. Adopting it would add a layer between us and code
we already control, and its execution layer is self-described as beta.

The one idea worth borrowing is approval-gated execution as a first-class
primitive, when we reach live trading.

### paperclip — out of scope

Multi-agent orchestration for running a business. Not a trading tool; no overlap
with anything here beyond both involving agents.

---

## worldmonitor — refused as a data source

Evaluated 2026-09-01. [koala73/worldmonitor](https://github.com/koala73/worldmonitor),
AGPL-3.0, TypeScript.

**The community backing is real and was worth checking rather than assuming.**
85,288 stars, 12,841 forks, 494 watchers, pushed the same day it was evaluated,
a dozen-plus human contributors, SDKs on npm / PyPI / RubyGems / pkg.go.dev, an
MCP server and an active Discord. This is not a hobby repository.

**It is a dashboard, not a data source, and that distinction decides it.** The
market data is aggregated from upstream vendors we already have opinions about:

    query1.finance.yahoo.com   Yahoo -- already our third crosscheck referee,
                               price-only and restated, never a returns source
    finnhub.io                 the insider, earnings and profile endpoints
    alphavantage.co            quotes
    api.coingecko.com          crypto

Four specific reasons it cannot feed a backtest here:

**1. Its insider data filters on the wrong date.** `get-insider-transactions.ts`
cuts its six-month window on `tx.transactionDate` -- when the insider traded --
not on when the filing was disseminated. For a live "recent insider activity"
panel that is the right choice. For a backtest it is precisely the lookahead
this system is built to prevent, and it would be inherited silently.

**2. It is a derived source where we already hold the primary.** Its insider
feed is Finnhub's, and Finnhub's is the SEC's. We ingest EDGAR directly and get
the `filed` date, which is the whole basis of `observed_at`.

**3. There is no history.** Cache TTLs across the market service run from 60
seconds to 90 days. It is a cache, not an archive. Our labelling window is 3,800
days, and a real-time feed can only ever be accumulated forward -- the same
structural limit already recorded against ApeWisdom.

**4. The analysis is LLM-generated.** `analyze-stock.ts` is model-backed, which
runs into the standing refusal on LLM output reaching a backtest: the lookahead
lives in the weights, where no audit of the data path can find it.

**Licence.** AGPL-3.0 with the network clause. This repository is public and
currently carries no licence of its own, so adopting worldmonitor code would be
a decision to license this system AGPL rather than a blocker -- but it is a
decision, not a detail.

**What is genuinely novel in it**, and unavailable to us elsewhere: COT
positioning, ETF flows, the Country Instability Index, physical premiums. All of
it is real alternative data. None of it is archived point-in-time, none covers
microcaps, and each would need years of forward accumulation before it could be
tested. That is a reason to note them, not to adopt the platform.

**What would reverse this:** a live deployment needing an operational monitoring
surface. Watching positions and geopolitical context in real time is a different
job from research, it has no point-in-time requirement, and worldmonitor is a
strong candidate for it. We have nothing deployed, so that job does not exist
yet.
