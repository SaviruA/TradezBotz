# Massive vs Alpaca: it was never an adjustment bug

**This corrects an earlier finding.** We recorded that Massive and Alpaca
"disagree systematically on corporate-action adjustment," that only 54% of
symbols agreed, and that Alpaca's deeper history was therefore unusable. The
first claim was wrong about its cause, and the conclusion that followed from it
was wrong too.

## What the third source showed

Adding Yahoo (via OpenBB) as an independent referee, over 24 disputed symbols:

```
adjustment_basis       15     <- not errors
secondary_outlier       5     <- genuine faults
no_majority             2
insufficient_overlap    2
```

**Fifteen of the twenty-four were not errors at all.** Massive and Yahoo agreed
to `0.00%` on every one of them, and Alpaca sat below both.

## The cause

Alpaca is **total-return adjusted**. Massive and Yahoo are **price-only**.

Checked directly on 2024-09-03 against actual dividend history:

| symbol | Massive | Alpaca | Yahoo | gap | dividends since |
| --- | --- | --- | --- | --- | --- |
| ARI | 10.430 | 5.663 | 10.430 | 4.77 | **5.50** |
| ABR | 13.260 | 10.330 | 13.260 | 2.93 | **2.40** |
| AMAT | 183.37 | 180.48 | 183.37 | 2.89 | **3.70** |
| AAPL | 222.77 | 220.92 | 222.77 | 1.85 | **2.08** |

Alpaca is lower on *every* symbol, by approximately the cumulative distribution.
That is not a bug in either vendor — it is two different, both-correct answers to
two different questions. `adjustment=all` means what it says.

The ordering explains the original 54% figure too: the worst "disagreements"
were ARI, ABR, ASGI — a mortgage REIT, a commercial REIT, and a closed-end fund.
The highest-yielding names, which is exactly what a dividend-basis difference
predicts and what a random data fault does not.

## How the code tells them apart

`crosscheck._looks_like_dividend_adjustment` requires two signatures together:

1. **One-sided** — the candidate is below the price-only series on ≥90% of days.
   Dividends only ever mark history down. A wrong split factor can go either way.
2. **Monotonically converging** — the gap is the sum of distributions *since*
   each day, so it shrinks to nothing at the present. Tested by rank correlation
   of the gap against time (≤ −0.7), not by comparing two windows: the gap is a
   fraction of price, and on a name whose price collapsed, a ratio test is
   dominated by the price path rather than the dividend. ARI pays ~98% of its
   price in distributions over the window *and* fell hard — the ratio test called
   it a vendor fault, the rank test reads the monotone decline correctly.

A constant ratio that never converges stays flagged. That is the XELB case: a
clean 3.004 throughout, Alpaca *higher* not lower, with Massive and Yahoo
agreeing at 7.03. **That one is still a genuine split-adjustment fault**, and the
original finding was right about it — just wrong to generalise from it.

## What actually follows

1. **Alpaca's history is not disqualified.** The reason we withheld it has
   mostly evaporated. What remains is a handful of real per-symbol faults, now
   individually identified rather than assumed.
2. **Never mix bases in one return calculation** — this rule survives, but for a
   sharper reason than before. Mixing price-only and total-return series does not
   produce a small error, it produces a fake return on every ex-dividend date.
3. **Which basis do *we* want?** An open question, not yet decided. We measure
   forward returns over 1/5/20 days after a filing. On a price-only series a
   dividend ex-date shows as a ~1% loss that no holder actually suffered. Total
   return is arguably the more honest measure of what a trader earned. The
   labeller currently uses Massive, i.e. price-only, which is at least internally
   consistent — but it is a choice we made by accident rather than on purpose.
4. **A two-source crosscheck cannot tell a fault from a definition.** That is the
   general lesson. Disagreement was treated as evidence of error when it was
   evidence of a different question being answered.

## Reproducing

```bash
python -m tradezbotz crosscheck --three-way --limit 60
```

Needs `pip install openbb-yfinance`.
