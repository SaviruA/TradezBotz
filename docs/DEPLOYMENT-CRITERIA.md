# Deployment criteria

**Set 2026-09-01, before any candidate had cleared a research gate.**

That timing is the point. Deciding what counts as good enough *after* seeing
which candidate won turns a threshold into a rationalisation. On the day this
was written, `measure` had run exactly once and every row reported
`population too thinly covered to generalise` — so none of these numbers could
have been chosen to fit a result, because there were no results.

The machine-readable copy is [`deployment-criteria.json`](deployment-criteria.json),
read by `research/deployment.py`. This file explains it.

---

## The enforcement mechanism

**Not that the numbers cannot change — that changing them leaves a record.**

The criteria are committed JSON. Lowering a threshold is a commit, with a diff
and a message, sitting in history next to the result that motivated it. Anyone
reading later can see both. That is stronger than a constant buried in code and
more honest than pretending a bar is immutable.

If a threshold here turns out to be wrong, change it and say why. What must not
happen is changing it quietly, in the same session as the result that made it
inconvenient.

---

## Status: UNCONFIRMED

`confirmed: false`. **Nothing can pass the gate**, however good it looks, until
three figures below are chosen. They are the operator's and cannot be
defaulted — a wrong guess by me would be worse than a blank.

---

## Statistical criteria

Every one is stricter than the corresponding research gate. `sweep.judge`
decides whether a measurement *means* anything; this decides whether it
justifies risk. A result can be perfectly sound and still not worth trading.

| criterion | value | research gate | why stricter |
| --- | --- | --- | --- |
| `min_deflated_sharpe` | 0.95 | same | already the significance bar |
| `min_trades` | 200 | 30 | 30 is the floor for arithmetic, not for conviction |
| `min_coverage` | 40% | 20% | a result read off a fifth of the population is not a result to fund |
| `max_fallback_share` | 10% | 25% | this one decides money, and the fallback is a constant somebody chose |
| `max_control_ratio` | 50% | 70% | if the complement keeps half the edge, most of it belongs to the population |
| `require_holdout_confirmation` | true | — | `splits.unlock_holdout` already enforces the ritual; this makes it mandatory |

## The live haircut — the criterion with no research counterpart

`assumed_live_haircut: 0.50`

The strategy must remain profitable after **half its gross edge is deleted**.

This is not pessimism. A backtest Sharpe of 4 arriving live at 0.5 is the
documented base rate rather than bad luck: biases stack, each one shaving the
number. We already correct for several of them, and the ones we cannot correct
for — impact at real size, adverse selection on fills, regime change between the
sample and next year — all point the same way.

So the test is not "is this profitable as measured" but "is it still profitable
at half". **If it is not viable at half, it is not viable.**

---

## The three the operator must set

These are unset, and the gate refuses everything while they are.

### `capital_at_risk`

Total capital the strategy may deploy. **Not a footnote — an input to the cost
model.** Position size determines participation, participation determines market
impact, and impact is the term that decides whether a microcap edge survives
contact with the book. Setting this at $10k and $500k produces materially
different net returns from the same signal.

### `max_drawdown_halt`

Peak-to-trough loss at which the system stops itself, without asking and without
a discretionary override. Without this there is no plan for being wrong — only a
plan for being right, which is not a plan.

Worth choosing as a number that would be genuinely tolerable to lose, not one
that sounds brave.

### `max_participation`

Currently `0.05` — 5% of average daily volume per position. Proposed rather than
required, since `costs.MAX_PARTICIPATION` already flags 10% as not executable in
one session and 5% leaves room. Change it if the capital figure makes it
binding.

---

## How to confirm

Edit [`deployment-criteria.json`](deployment-criteria.json): set the three
values, set `confirmed: true`, and put a real note in `signed_off` — who
decided, when, and on what basis. Commit it as its own change, so the decision
is legible in history rather than mixed into unrelated work.

---

## What the gate does not check

**Survivorship.** Tiger T6 is open: delisted issuers are largely unpriced, so
the measured population skews to survivors and every return is biased upward by
an amount nothing here can estimate. No threshold catches that — only the
backfill does.

**Whether the exit rule was backtested.** The research horizons are fixed
holding periods. If a live exit rule differs from them, the research and the
trading are measuring different strategies and this gate will not notice.

**Anything about Phase 3b.** Realised-versus-modelled cost is a separate gate,
checked after minimum-size live orders, not here.
