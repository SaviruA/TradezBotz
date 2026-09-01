# Pre-Mortem: the first strategy this system says survived

| Field | Value |
|-------|-------|
| **Thing being stress-tested** | Deploying capital on the first candidate `measure` reports as KEEP |
| **Failure horizon** | 14 days after that deployment |
| **Pre-mortem date** | 2026-09-01 |
| **Method** | Gary Klein prospective hindsight; Tiger / Paper Tiger / Elephant |
| **Participants** | One operator, one agent — see Elephant E4 |

## Assumptions, stated because the clarifying round was skipped

1. The "launch" is real capital on a strategy this pipeline selected. If the actual
   goal is only to produce a defensible research result, the six launch-blocking
   Tigers drop to four — T5 and T12 are about tradeability, not about truth.
2. There is no launch date. The horizon is "whenever `measure` first returns KEEP",
   which on current coverage is weeks away.
3. "Failed" means the live result was materially worse than the backtest, not that
   the strategy lost money. A losing strategy that lost exactly as much as the
   backtest predicted is a success of the method.

---

## Summary

| Metric | Count |
|--------|:-----:|
| Total risks | 22 |
| **Tigers** | **12** |
| — Launch-blocking | 6 |
| — Fast-follow | 4 |
| — Track | 2 |
| Paper Tigers | 5 |
| Elephants | 5 |

**The single most important line in this document:** four of the six
launch-blocking Tigers are defects introduced during the last two days of work,
by the same process that produced the tests asserting the system is correct.

**Retraction, same day.** Elephant E3 originally claimed the expected outcome was
that nothing survives the gates. Challenged, computed, and withdrawn — the cost
and significance hurdles are clearable by roughly an order of magnitude against
documented effect sizes. See E3 for the numbers and for why the wrong version
was written.

---

## Tigers — launch-blocking

| # | Risk | Evidence | Mitigation | Decision by |
|---|------|----------|-----------|-------------|
| T1 | The Deflated Sharpe bar rises every night with no new hypothesis tested. `assess()` reads `registry.count()` — every trial ever — and `measure` re-registers ~200 trials per scheduled run | `trials.py:287`; 54 candidates × 3 horizons × ~2 with controls ≈ 200/night ≈ 6,000/month; `expected_max_sharpe` grows monotonically in `n_trials` | Deduplicate the registry on (hypothesis, horizon, partition, dataset fingerprint). A re-run of an identical trial must update, not append | Before the next `measure` run |
| T2 | Repetition is counted as search breadth, so the correction penalises re-running rather than the number of distinct hypotheses tried | `backtest.run()` registers unconditionally. Bailey & López de Prado count distinct configurations, not executions | Same fix as T1. Additionally record a dataset fingerprint so a genuinely new dataset *does* count as a new trial | Before the next `measure` run |
| T3 | **The holdout is not sealed.** Split boundaries derive from the observed data range, which moves every night as events accumulate | `cmd_measure` derives `split_start`/`split_end` from min/max `observed_at`; the workflow passes neither `--split-start` nor `--split-end`. I added those flags today and did not wire them | Pin both in the workflow to fixed dates and commit them. A moving holdout boundary is not a holdout | Before the next `measure` run |
| T4 | The cost gate is decided by a constant, not a measurement, for most trades — `CostTable` falls back to a flat 93bps below 21 prior bars | `costs.py FALLBACK_COST_BPS`; local run charged the fallback on effectively every trade (2 EDGE estimates total) | Report the fallback share beside every result and refuse KEEP above a threshold (suggest 25%). A cost gate decided by my constant is not a cost gate | Before any KEEP is believed |
| T5 | The first candidate to clear the 30-trade floor will be whichever has coverage, not whichever is strongest | Local: coverage 0.28%, every candidate TOO_FEW_TRADES. CI: 4,875 of 7,342 symbols cached; intraday covers 180 of 3,800 days | Add a coverage column to `report()` and require a minimum coverage before KEEP. Do not read the first survivor as the best candidate | Before the first KEEP |
| T6 | The population is biased toward survivors: delisted issuers stay in the event store but their prices were never fetched, so they label NO_DATA and vanish | 641 delisted and 316 OTC symbols found; only 11/1,001 formerly-unknown and 4/244 OTC have cached bars | Attempt a backfill for every delisted symbol, then report labelled-vs-unlabelled by classification. If delisted names cannot be priced, state the bias size rather than carrying it silently | Before the first KEEP |

## Tigers — fast-follow

| # | Risk | Evidence | Plan |
|---|------|----------|------|
| T7 | `measure` fails silently forever — `continue-on-error` plus a 20-minute timeout on the last substantive step | Workflow step config; it has never run once in CI | Assert in `status` that a `measure` result exists and is newer than the newest event; fail the run if not |
| T8 | Dual-class tickers resolve to the first symbol the filer wrote, which can be the illiquid class | `PARAA,PARA` → `PARAA`, the class A line. 42 symbols recovered this way | Prefer the class with greater cached volume where both exist; otherwise keep first-wins and log the choice |
| T9 | Microstructure candidates measured on 180 days appear beside full-window candidates in one ranked table | 343,125 sessions at `--days 180`; full depth ~7.6M and ~10 runs away | Coverage column in `report()` (shared with T5), or deepen intraday before trusting those six rows |
| T10 | No position sizing or exit rule exists, so KEEP describes a signal, not a strategy | No sizing module; `measure` passes no `shares`, so market impact is never charged | Decide capital and sizing, then re-cost with impact. Until then read every net figure as an upper bound |

## Tigers — track

| # | Risk | Evidence |
|---|------|----------|
| T11 | The 84.4% survivorship figure covers 2.5 local years and will be quoted as if it described 13 years of CI history | Local store spans 2024-01→2026-08; CI holds ~13 years. `describe()` prints the ratio without its window |
| T12 | Feature memoisation on (symbol, entry_day) means a truncation bug would apply uniformly and produce coherent, wrong output rather than an error | `FeatureBuilder._memo`; `prior_bars` is the single enforcement point |

---

## Paper Tigers — named so they are not mitigated

| Risk | Why it is not a Tiger |
|------|----------------------|
| Alpaca changes or revokes SIP entitlement | Stable across every run since the switch; `crosscheck --three-way` detects divergence; bars are cached so history survives |
| State blob outgrows the Actions cache | Measured: 78 B/session, ~600 MB at full depth against a 10 GB ceiling |
| SEC rate-limits or blocks the ingest | Self-limited to 8 req/s against a 10/s ceiling, contact User-Agent, weeks without a block |
| Excluding OTC discards signal | 6.7% of symbols, 4 of 244 with bars, and the `otc` feed is 403 — there is no data to discard |
| A competitor arbitrages the insider signal first | Decades-old, publicly documented. If it is arbitraged away that is a finding this system measures, not an external threat |

---

## Elephants

**E1 — The apparatus may be displacement activity.** 756 tests, 15 commands and
five statistical-correction modules exist; not one strategy has been measured.
Every earlier "nothing has been measured yet" was blamed on coverage when in
fact no code path existed to measure with.

**E2 — Nobody has said what result would be good enough to trade.** No target
Sharpe, no capital at risk, no maximum drawdown, no decision rule. `judge()`
returns KEEP or a rejection reason; KEEP has no agreed consequence. Goalposts
that are undefined before the result can move after it.

**E3 — RETRACTED 2026-09-01, same day, on challenge. The original claim was that
"the honest expected outcome is that nothing survives." That was asserted, not
computed, and computing it shows the opposite.**

The hurdles we can actually calculate are clearable:

| hurdle | required mean per trade | documented effect |
|---|---|---|
| Transaction costs alone | 0.93% | 6.3% mean CAR, microcap insider buys after >10% gains (arXiv 2602.06198) |
| DSR @ 200 trials, effective n 2,000 | 0.94% | same |
| DSR @ 200 trials, effective n 30 | 1.83% | same |
| DSR @ 200 trials, fat tails (skew 2, kurt 12) | 0.92–1.23% | same |

Costs are roughly 15% of the documented effect, not a wall. And T1's trial
inflation moves the bar **logarithmically**, not catastrophically — 0.94% at 200
trials to 1.32% at 20,000. T1 is a real defect and still worth fixing, but the
original text called the bar "unreachable through a bug", and that was wrong by
about an order of magnitude in implication.

**What should genuinely be expected to fail, and by design:** the ~40 indicator
candidates — RSI oversold, MACD cross, Bollinger tags, engulfing candles, the
sweeps. Their recorded priors say "no edge" and those priors are well-founded;
an edge in the most-tested indicators in existence would have been arbitraged
decades ago. They are in the sweep as **controls**, not as hopes. Conflating
"the controls will fail" with "the project will fail" is the error the original
E3 made.

**What remains genuinely uncertain** is not the effect size but the **coverage** —
0.28% locally, incomplete in CI — and whether the documented effect survives our
specific universe and corrections. That is a data problem under active repair,
not a verdict on the signal.

**Why the wrong version got written, which is the actual elephant.** A
pre-mortem asks "why did it fail" and structurally rewards pessimism. "Nothing
will survive" feels rigorous, is safe to say, and cannot embarrass anyone. It
was generated from the framing rather than from evidence and went unchecked
until the operator asked for the reasoning. That is Red Flag 1 from this skill's
own reference — a Paper Tiger wearing a Tiger's clothes — committed inside the
document meant to catch it.

The residual real risk is the one the retraction does not remove: **there is
still no definition of what result would be good enough to trade** (E2), so
neither optimism nor pessimism is currently falsifiable.

**E4 — Every audit has been run by the author.** The defects fixed in the last
two days — `requeue` never attaching to its class, `lock.acquire()` misuse that
would have made a command never run, the budget arithmetic — were all caught by
accident or by measurement, not by the review process that had already passed
them.

**E5 — A plausible wrong answer is indistinguishable from a right one here.**
Every failure mode fixed recently produced *coherent output*: silent lookahead,
inflated survivorship, order-free reductions, a control measuring nothing. None
of them errored. One reviewer on an AI-authored quant codebase has no
independent replication to catch the next one.

---

## What this changes

Do not read the first KEEP as a finding. On current evidence the first KEEP is
more likely to be an artefact of T1–T6 than a strategy — note that this is a
claim about **measurement defects**, not about the signal. The retraction in E3
matters here: there is no computed reason to expect the underlying effect to be
absent, and the reasons to distrust an early result are all fixable.

The cheapest high-value fixes are **T3 and T1**: pin the split dates in the
workflow, and deduplicate the trial registry. Both are small, both are defects
introduced this week, and until they are fixed every subsequent result is
measured against a moving holdout and an inflating significance bar.

**Review cadence:** this document decays. Re-run before the first KEEP, and
again before any capital is deployed.
