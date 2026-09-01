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

## Status: CONFIRMED 2026-09-02

The operator figures are set. The gate now refuses on merit rather than on
missing configuration.

## Two capital numbers, and why they are not the same

The single most important distinction in this file.

| | value | job |
| --- | --- | --- |
| `capital_at_risk` | $25,000 over 10 positions | **sizes the backtest.** What the cost model charges impact against |
| `paper_capital` | $25,000, same sizing | **Phase 3a.** Mirrors the backtest so paper describes the same strategy |
| `live_test_capital` | $100 across 4 positions | **Phase 3b.** The real stake, measuring realised slippage |

They must not be the same number, and the direction of the error matters.

Sizing the backtest off the $100 stake would charge impact for ~$10 orders,
which is approximately zero — and every net return would then **flatter any size
actually traded later**. A backtest may be pessimistic relative to reality. It
must never be optimistic. So it is sized off the capital that might be
*reached*, not the capital being started with.

When unsure, round `capital_at_risk` **up**. Higher makes the backtest charge
more cost and reject more candidates; lower makes it generous. The asymmetry is
the whole point.

### Paper sizing — corrected

An earlier draft said paper must mirror the *live test* stake. That was too
strict and is withdrawn. Phase 3a's jobs are plumbing and live-vs-backtest
agreement, and neither depends on trading $100.

What paper sizing must avoid is the opposite error: deploying the account's full
$100,000 per position would "fill" orders that could never fill live, which is
false confidence rather than a test. And using the $100 stake would exercise
none of the portfolio mechanics — concurrent positions, buying power, ordering —
that paper exists to shake out.

So paper mirrors **the backtest**: $2,500 a position, ten positions, $25,000 of
the available $100,000. Paper, research and eventual live then all describe one
strategy. The remaining balance is headroom, not a target.

### The live test splits into four, and that is measured

$100 across 4 positions is $25 each. Against the cached universe:

| stake | whole share | + fractional | reachable |
| --- | --- | --- | --- |
| $10 | 22% | 168 names | **96%** |
| $25 | 37% | 138 names | **98%** |
| $50 | 55% | 100 names | **100%** |

**92% of the cached universe is fractionable**, far above the 56% across all
listed US equities — because the backfill has reached the names that actually
trade. So $25 positions are fine.

One caveat: the cached universe currently skews liquid. As the backfill reaches
thinner microcaps the fractionable share will fall, and $25 positions may become
unreachable on part of the universe. Worth re-measuring before the live test
rather than assuming this holds.

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

## The operator figures, as set

### `capital_at_risk` — $25,000 over 10 positions

**Not a footnote — an input to the cost model.** Position size determines
participation, participation determines impact, and impact decides whether a
microcap edge survives contact with the book.

$2,500 per position measured against real cached bars costs 139bps round trip on
a thin name and 95bps on a median one — within a few basis points of the
smallest size tested, and comfortably fillable. Impact turns out to be
second-order at these sizes; spread dominates. The cautious choice is therefore
nearly free, which is a good property to have.

### `max_drawdown_halt` — $5,000 (20%)

Peak-to-trough loss at which the system stops itself, without asking and without
a discretionary override. Without this there is no plan for being wrong — only a
plan for being right, which is not a plan.

**Nothing enforces this yet**, because there is no live system to halt. It is a
recorded commitment that Phase 4 must implement and that the gate refuses to
pass without. Its value today is that it was decided in advance rather than
during a drawdown.

### `max_participation`

Currently `0.05` — 5% of average daily volume per position. Proposed rather than
required, since `costs.MAX_PARTICIPATION` already flags 10% as not executable in
one session and 5% leaves room. Change it if the capital figure makes it
binding.

---

## How to change it

Edit [`deployment-criteria.json`](deployment-criteria.json) and commit it as its
own change with a message saying why. That is the whole enforcement mechanism:
the diff sits in history next to whatever motivated it.

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
