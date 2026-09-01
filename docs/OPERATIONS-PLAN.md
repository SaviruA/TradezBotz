# From research to decisions: the operating plan

**Status: proposed, revised 2026-09-01. Nothing in Phases 1–4 is built.**

The system ingests events, labels them, and measures candidates. There is no
code path from "a Form 4 arrived this morning" to "here is what to do about
it." This is the plan for that path.

Revised after a pre-mortem and a review of how other practitioners handle the
same problems. Four issues found in the first draft changed its shape rather
than adding caveats, and they are marked **[revised]** below.

---

## The constraint this plan is built around

**No strategy has cleared a gate yet.** `measure` ran for the first time today,
and on current coverage every row reports `population too thinly covered to
generalise`. A decision system acting on that would be acting on nothing.

That is not a reason to defer the work. It is a reason to sequence it so the
parts that need no validation get built now, the parts that do are gated, and
the threshold is chosen **before** any result is visible.

1. **Surveillance needs no validation.** Surfacing today's events with their
   point-in-time features is useful whether or not any strategy works, and it is
   the fastest way to discover the live path disagreeing with the backtest.
2. **The bar must be set blind.** Deciding "good enough to trade" after seeing
   which candidate won is how goalposts move. Pre-mortem elephant E2 is this
   hazard, and now is the only moment it can be closed honestly.
3. **Paper trading measures some costs and cannot measure the expensive one.**
   [revised] See Phase 3.

---

## The one architectural rule

**Live scoring and backtest scoring must run the same code.**

Not equivalent code. The same functions: `features.FeatureBuilder`,
`joins.*Join`, `candidates.*`. A separate "live" implementation will drift from
the research one and nothing will report the drift — results simply stop
corresponding to reality.

The difference between research and live is **only the cutoff**: the backtest
asks "what was knowable at entry day D" for many past D; the monitor asks it for
D = today. `EventStore.as_of` already expresses that, so the monitor is a
different argument to existing code, not a second pipeline. Anything that cannot
be expressed that way does not belong in the live path.

---

## What other people do about these problems

Grounding for the choices below, since each was a place the first draft was
wrong.

**Sharpe collapse is expected, not bad luck.** A backtest Sharpe of 4 arriving
live at 0.5 is [the predictable result of biases stacking, each shaving the
number](https://www.techinterview.org/post/3233477314/why-backtest-sharpe-collapses-live/),
with a couple able to erase it. Planning for a haircut is standard; planning for
the backtest number is not.

**Paper trading is not the impact test.** [Paper fills are
simulated](https://referentiallabs.com/blog/backtesting-vs-paper-trading/) — no
real order touches the book, so no impact occurs. Practitioners model impact
explicitly and calibrate against their *own* fills; the only real validation is
small live orders. This split Phase 3 in two.

**The answer to selection pressure is more corrections, not fewer looks.**
[`ml4t/diagnostic`](https://github.com/ml4t/diagnostic) ships the Deflated
Sharpe with correlation-adjusted effective K alongside PBO and FDR. We have DSR
and two-way clustering; PBO is a gap worth noting.

**Scheduled CI needs a dead man's switch.** GitHub runs schedules best-effort,
and [disables them entirely after 60 days without repository
activity](https://dev.to/krissv/monitoring-github-actions-scheduled-workflows-a-practical-guide-31h7).
Monitor expected execution, not failures. Built as `runlog.py`.

---

## Phase 0 — Set the bar, before any result exists  ✅ DONE 2026-09-01

Built as `research/deployment.py` with the thresholds committed in
[`deployment-criteria.json`](deployment-criteria.json) and explained in
[DEPLOYMENT-CRITERIA.md](DEPLOYMENT-CRITERIA.md). Set while `measure` had run
once and every row reported thin coverage, so none of it could have been fitted
to a result.

**Status: UNCONFIRMED.** The gate refuses everything until the operator sets
capital, drawdown halt and position ceiling. That is the intended state, not an
omission.

Write `docs/DEPLOYMENT-CRITERIA.md` and commit it as a machine-checked gate.
Values below are proposals, not defaults to accept quietly.

| criterion | proposed | why |
| --- | --- | --- |
| Deflated Sharpe | ≥ 0.95 | already the `significant` gate |
| Trades | ≥ 200 on train | 30 is the floor for arithmetic, not for conviction |
| Coverage | ≥ 40% | above the 20% `MIN_COVERAGE` gate |
| Cost fallback share | ≤ 10% | tighter than the 25% gate; this decides money |
| Control separation | control ≤ 50% of signal mean | tighter than `CONTROL_TOLERANCE` |
| Holdout | confirmed **once**, declared in advance | `splits.unlock_holdout` enforces the ritual |
| **Expected live haircut** | **assume ≥ 50% of backtest edge is lost** | [revised] the documented base rate; if the strategy is not viable at half, it is not viable |
| Capital at risk | to decide | drives sizing and therefore impact |
| Max drawdown → halt | to decide | the number at which the system stops itself |
| Max position | ≤ 5% ADV | `costs.MAX_PARTICIPATION` flags 10% as infeasible |

The last three are the operator's and cannot be defaulted. Without a drawdown
halt there is no plan for being wrong, only a plan for being right.

---

## Phase 1 — Surveillance, not ranking **[revised]**

Cost: small. Prerequisite: Phase 0. **Buildable now.**

The first draft had the monitor rank events by which candidates fire, including
candidates that have never cleared a gate. That is an untracked selection
channel: watching `buy + near_high` fire every morning builds conviction with no
evidence, and nothing enters the trial registry. The whole apparatus exists to
prevent exactly that, and the monitor would have routed around it.

So the scope narrows. `python -m tradezbotz monitor` reports **events and their
point-in-time features** — data you could read out of the event store anyway,
assembled and made legible. It does **not** rank by candidate conviction.

    load    events with observed_at inside the last N days
    enrich  the same FeatureBuilder and joins that measure uses
    report  symbol, event, features, regime, estimated cost, staleness

Candidate scoring switches on **only for candidates that have passed Phase 0**.
Until one has, the monitor is a surveillance tool and says so. A `--exploratory`
flag may show unvalidated candidate matches, and must stamp the output with what
looking at it costs.

Two requirements easy to omit:

**Record the briefing before the outcome.** Same discipline as the trial
registry. A monitor whose past calls are not retrievable cannot be evaluated and
will be remembered selectively.

**Report staleness.** EDGAR filings arrive on a lag and the pipeline runs
nightly, so an event may be days old before it is seen. "Insider bought" and
"insider bought, and we noticed on day four" are different trades.

---

## Phase 2 — Propose

Cost: moderate. Prerequisite: at least one candidate past Phase 0.

Turn a scored event into an explicit, falsifiable proposal:

    symbol, side, entry rule, size, exit rule, stop, expiry, and the
    candidate + verdict + trial id that justifies it

Recorded **before** the open, never amended after. That store is what makes the
system measurable: realised against predicted, per proposal.

This is where Tiger T10 closes. Sizing exists as `--capital`; **exits do not
exist at all**. The backtest's horizons are fixed holding periods — a legitimate
research choice and an inadequate trading rule. Phase 2 must state the exit
explicitly, and the backtest must then be re-run using that same exit, or the
research and the trading are measuring different strategies.

---

## Phase 3a — Paper: plumbing and spread **[revised]**

Cost: moderate. Prerequisite: Phase 2.

Execute proposals against the Alpaca paper account. It validates two things
honestly:

**Plumbing.** Symbol resolution, market hours, halts, corporate actions,
rejected orders. Each is a silent failure in a backtest and a loud one in an
order book.

**Live-vs-backtest agreement.** Run the same day through the monitor and the
backtest and confirm they select the same events. Disagreement is a breach of
the shared-code rule, and finding it here is the point.

**What it cannot validate: market impact.** No real order touches the book, so
no impact occurs. The first draft called this the most valuable phase for
replacing our 93bps constant — that was half right. Paper gives spread and
plumbing. It cannot give the impact term, which is precisely the part that
matters for microcaps where a position is a meaningful share of daily volume.

Minimum before 3b: 60 sessions.

## Phase 3b — Minimum-size live: impact **[new]**

Prerequisite: Phase 3a clean.

The smallest live orders that clear broker minimums, on real money, for the sole
purpose of measuring realised slippage against the modelled figure. Not a
profit-seeking phase — the position sizes are too small to matter and are
supposed to be.

This is the only mechanism that tests the impact term. It needs its own risk
limit (a fixed total capital, small enough that losing all of it changes
nothing) and its output is a single number: realised cost against modelled cost.
If the model is optimistic, every backtest result is too high, and Phase 4 does
not open.

---

## Phase 4 — Live

Prerequisite: every Phase 0 criterion, holdout confirmation, and a Phase 3b run
whose realised costs did not exceed the modelled ones.

Non-negotiable:

- a drawdown halt that stops the system without asking
- position limits enforced in code, not intent
- every order traceable to a recorded proposal, every proposal to a candidate
  with a verdict and a trial id
- a kill switch that does not depend on the pipeline being healthy

**Execution stays manual.** The system proposes; a person places the order.
Automating that is a separate decision needing its own evidence.

---

## Accounting that must not be skipped **[revised]**

**Selection across phases is selection.** If a candidate passes Phase 0, fails
in paper, and we move to the next one, that is another layer of choosing on
data. It must reach the trial count, or the Deflated Sharpe understates the
search that actually happened. Phase 2 onward should register each promotion as
a trial against the relevant partition.

**PBO is a gap.** We correct with DSR and two-way clustering. The probability of
backtest overfitting is a complementary check other practitioners run and we do
not.

---

## What this plan does not solve

- **T6, survivorship, gets worse live.** Delisted issuers are still largely
  unpriced, and a live system only ever encounters currently-listed names — so
  the bias becomes structurally invisible rather than merely unmeasured.
- **E1, displacement.** Building a decision layer before any signal is validated
  is a way of feeling productive. Phases 0 and 1 are defensible now because
  neither depends on a validated signal. **Phases 2–4 are not, and must not
  start until one exists.**
- **E5, single reviewer.** More moving parts, same one pair of eyes, now with
  money downstream.

---

## Sequencing

| phase | prerequisite | start |
| --- | --- | --- |
| 0 — set the bar | none | **done, awaiting operator sign-off** |
| 1 — surveillance | Phase 0 | now |
| 2 — propose | one candidate past Phase 0 | not yet |
| 3a — paper | Phase 2 | not yet |
| 3b — minimum-size live | 3a clean, 60 sessions | not yet |
| 4 — live | Phase 0 met + 3b costs within model | not yet |

Phase 0 is time-critical for a reason that has nothing to do with engineering:
it is only honest while the results are unknown. Every night `measure` runs,
that window narrows.
