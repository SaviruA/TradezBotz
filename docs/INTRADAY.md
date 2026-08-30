# Intraday data: what the account actually has

Probed directly against Alpaca on 2026-08-30 with the paper credentials in
`.env`. Everything below is measured, not read off a pricing page.

## The finding

The free Alpaca plan serves the **full consolidated tape (SIP)**, not just IEX.
This was not what the code assumed.

Same symbol, same one-minute window (AAPL, 2026-08-24 14:00–14:01 UTC):

| feed | trades | shares | distinct venues |
| --- | --- | --- | --- |
| `iex` | 147 | 10,071 | 1 |
| `sip` | 4,275 | 209,697 | 17 |

**IEX carried 4.8% of consolidated volume in that minute.** Daily bars show the
same gap: 1,160,745 shares on `iex` against 38,414,225 on `sip` for 2026-08-17.

History depth, same probe:

| feed | 1-minute bars | daily bars |
| --- | --- | --- |
| `iex` | 2021 onward | 2021 onward |
| `sip` | **2016 onward** | **2016 onward** |

Trades (`/v2/stocks/{s}/trades`) and quotes (`/v2/stocks/{s}/quotes`) both
return data on the free plan.

## The one restriction

SIP data inside the last 15 minutes is refused:

```
sip trades from 2m ago:  403 {"message":"subscription does not permit querying recent SIP data"}
sip trades from 20m ago: 200
```

This is irrelevant to everything we currently do. Backtests read history, and
the live strategy enters at the next open, not within 15 minutes of a print. It
would only bite if we ever wanted genuine intraday execution.

## What this changes

1. **`AlpacaPriceSource` defaulted to `feed="iex"`.** Changed to `"sip"`. Every
   comparison we have run to date compared Massive's consolidated bars against a
   ~5%-of-volume single-venue sample.

2. **The crosscheck disagreement numbers need re-running.** We attributed the
   ~54% agreement rate to corporate-action adjustment conflicts between vendors.
   Some of it is real — XELB's clean 3.004 split ratio is a genuine adjustment
   difference and that finding stands — but an unknown share of the rest is
   simply IEX not being the market. The adjustment conflict is now a weaker
   claim than the README states, and the rerun on `sip` is what settles it.
   Until then, treat the 54% figure as an upper bound on real disagreement.

3. **Volume profile and order flow are unblocked**, on data rather than in
   principle. Both still need an intraday fetch and store path that does not
   exist yet — `daily_bars` is the only method on the `PriceSource` protocol,
   and the SQLite cache has one row per symbol-day.

4. **History roughly triples.** 2016 rather than 2021 on SIP, against Massive's
   2 years, which materially changes how much out-of-sample room the holdout
   split has.

## Built on this (2026-08-30)

- `research/intraday.py` -- batched minute-bar fetch, trades/quotes, and a
  `ProfileStore` holding reduced sessions at ~600 bytes each
- `research/microstructure.py` -- volume profile (POC, 70% value area, low-volume
  nodes), tick-rule and Lee-Ready flow classification, and `compare_classifiers`
- `tradezbotz backfill-intraday [--exact]`

Two bugs the live probe caught that unit tests would not have:

1. **Date-granular embargo guard.** A window ending "today" is expanded by the
   API to cover today's session, breaching the 15-minute SIP embargo and
   returning 403 for the *entire batch* -- so a valid month-long backfill died on
   its most recent day. Now clamped to an explicit `now - 16m` timestamp, which
   also keeps today's session usable right up to the embargo.
2. **Lexicographic timestamp ordering.** Alpaca omits the fractional part when
   it is zero, and `.` sorts below `Z`, so `...:58Z` compared as *later* than
   `...:58.267Z`. Quote lookup used string comparison, which silently paired some
   prints with a quote that came after them. Now parsed to datetimes.

## Next build

- run `backfill-intraday --exact` across the universe and wire it into
  `pipeline.yml`
- re-run `crosscheck` on `sip` and correct the README's adjustment section
- register the new hypotheses as trials and measure them, individually and paired
