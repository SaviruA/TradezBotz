# From research to decisions: the operating plan

**Status: proposed, 2026-09-01. Nothing here is built.**

The system ingests events, labels them, and measures candidates. There is no
code path from "a Form 4 arrived this morning" to "here is what to do about
it." This is the plan for that path.

---

## The constraint this plan is built around

**No strategy has cleared a gate yet.** `measure` ran for the first time today,
and on current coverage every row reports `population too thinly covered to
generalise`. A decision system acting on that would be acting on nothing.

That is not a reason to defer the work. It is a reason to sequence it so the
parts that need no validation get built now, the parts that do are gated, and
the threshold is chosen **before** any result is visible.

Three things follow, and they shape everything below:

1. **Monitoring needs no validation.** Detecting a new event, scoring it
   point-in-time and surfacing it is useful whether or not any strategy works.
   It is also the fastest way to find out that the live path disagrees with the
   backtest.
2. **The bar must be set blind.** Deciding "good enough to trade" after seeing
   which candidate won is how goalposts move. Elephant E2 in the pre-mortem is
   this exact hazard, and the only moment it can be closed honestly is now.
3. **Paper trading is a measurement, not a rehearsal.** Its value is not
   practice. It is the only way to replace our assumed 93bps cost constant with
   observed fills — the number Tiger T4 says currently decides KEEP.

---

## The one architectural rule

**Live scoring and backtest scoring must run the same code.**

Not equivalent code. The same functions: `features.FeatureBuilder`,
`joins.*Join`, `candidates.*`. A separate "live" implementation will drift from
the research one, and nothing will report the drift — results will simply stop
corresponding to reality.

The difference between research and live is **only the cutoff**: the backtest
asks "what was knowable at entry day D" for many past D; the monitor asks the
same question for D = today. `EventStore.as_of` already expresses that, so the
monitor is a different argument to existing code, not a different pipeline.

Anything that cannot be expressed that way does not belong in the live path.

---

## Phase 0 — Set the bar, before any result exists

Cost: an afternoon of decisions. Prerequisite: none. **Do this first.**

Write `docs/DEPLOYMENT-CRITERIA.md` and commit it. It becomes a machine-checked
gate, not a document of intent. Every threshold below needs a number chosen by
the operator; the values shown are proposals, not defaults to accept quietly.

| criterion | proposed | why it is here |
| --- | --- | --- |
| Deflated Sharpe | ≥ 0.95 | already the `significant` gate |
| Trades | ≥ 200 on train | 30 is the floor for arithmetic, not for conviction |
| Coverage | ≥ 40% | above the 20% `MIN_COVERAGE` gate; a strategy read off half the population |
| Cost fallback share | ≤ 10% | tighter than the 25% gate, since this decides money |
| Control separation | control mean ≤ 50% of signal | tighter than `CONTROL_TOLERANCE` |
| Holdout | confirmed **once**, declared in advance | `splits.unlock_holdout` already enforces the ritual |
| Capital at risk | to decide | drives sizing and therefore market impact |
| Max drawdown → halt | to decide | the number at which the system stops itself |
| Max position | ≤ 5% ADV | `costs.MAX_PARTICIPATION` already flags 10% as infeasible |

The last three are genuinely the operator's and cannot be defaulted. Without a
drawdown halt there is no plan for being wrong, only a plan for being right.

---

## Phase 1 — Monitor

Cost: small. Prerequisite: Phase 0. **Buildable now, useful now.**

`python -m tradezbotz monitor` — a daily briefing, no orders.

    load    events with observed_at inside the last N days
    enrich  the same FeatureBuilder and joins measure uses
    score   against candidates, flagged by their current verdict
    rank    by conviction, with the evidence beside each row
    emit    a briefing, and record it

Each row states the symbol, the event, which candidates fire, the current
geopolitical regime, the estimated round-trip cost, and — most importantly —
**the verdict of every candidate it matches.** A row matching only candidates
that have never cleared a gate is labelled as such rather than presented as a
recommendation.

Two things it must do that are easy to omit:

**Record the briefing before the outcome.** Same discipline as the trial
registry. A monitoring system whose past calls are not retrievable cannot be
evaluated, and will be remembered selectively.

**Report staleness.** EDGAR filings arrive on a lag and the pipeline runs
nightly, so an event may be three days old before it is seen. The briefing must
say how old each event is, because "insider bought" and "insider bought, and we
noticed on day four" are different trades.

---

## Phase 2 — Propose

Cost: moderate. Prerequisite: at least one candidate past Phase 0.

Turn a scored event into an explicit, falsifiable proposal:

    symbol, side, entry rule, size, exit rule, stop, expiry, and the
    candidate + verdict that justifies it

Recorded in a `proposals` store **before** the market opens, never amended
after. That store is what makes the whole thing measurable: realised outcome
against predicted outcome, per proposal.

This is also where Tiger T10 gets closed. Sizing exists as `--capital` in the
cost model; exits do not exist at all. The horizons in the backtest are fixed
holding periods, which is a legitimate research choice and an inadequate trading
rule. Phase 2 must state the exit explicitly — time-based, target, stop, or a
combination — and then the backtest must be re-run using that same exit, or the
research and the trading are measuring different strategies.

---

## Phase 3 — Paper

Cost: moderate. Prerequisite: Phase 2. **The most valuable phase.**

Execute proposals against the Alpaca paper account (credentials already held).
Its purpose is measurement, in three specific things:

**Cost model validation.** We charge a 93bps constant to most trades and Tiger
T4 says that constant currently decides KEEP. Paper fills give observed slippage
against the modelled figure. If the model is optimistic, every backtest result
is too high and we would learn it here rather than with money.

**Plumbing validation.** Symbol resolution, market hours, halts, corporate
actions, rejected orders. Every one is a silent failure in a backtest and a
loud one in an order book.

**Live-vs-backtest agreement.** Run the same day through the monitor and through
the backtest, and confirm they select the same events. Any disagreement is a
bug in the shared-code rule above, and finding it here is the entire point.

Minimum before Phase 4: **60 sessions** and enough proposals to compare
distributions rather than anecdotes.

---

## Phase 4 — Live

Prerequisite: every Phase 0 criterion met, including holdout confirmation, plus
a Phase 3 run whose realised costs did not exceed the modelled ones.

Non-negotiable at this stage:

- a drawdown halt that stops the system without asking
- position limits enforced in code, not in intent
- every order traceable to a recorded proposal, and every proposal to a
  candidate with a verdict and a trial id
- a kill switch that does not depend on the pipeline being healthy

**Execution is not automated at this phase.** The system proposes; a person
places the order. Automating execution is a separate decision requiring its own
evidence, and nothing here depends on it.

---

## What this plan does not solve

The pre-mortem's open items remain open and are not fixed by building this:

- **T6, survivorship.** Delisted issuers are still largely unpriced. A live
  system will not notice, because it only ever sees currently-listed names —
  which is precisely how the bias stays invisible.
- **E1, displacement.** Building a decision layer before any signal has been
  validated is a way of feeling productive. Phase 0 and Phase 1 are defensible
  now because neither depends on a validated signal. **Phases 2 through 4 are
  not, and should not start until one exists.**
- **E5, single reviewer.** More moving parts, same one pair of eyes, and now
  with money downstream.

---

## Sequencing

| phase | prerequisite | start |
| --- | --- | --- |
| 0 — set the bar | none | **now, and before any result is read** |
| 1 — monitor | Phase 0 | now |
| 2 — propose | one candidate past Phase 0 | not yet |
| 3 — paper | Phase 2 | not yet |
| 4 — live | Phase 0 met + 60 paper sessions | not yet |

Phase 0 is time-critical for a reason that has nothing to do with engineering:
it is only honest while the results are unknown. Every day `measure` runs, that
window narrows.
