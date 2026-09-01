"""Trial registry and the Deflated Sharpe Ratio.

Every backtest is a trial, and the number of trials determines what a result
means. Simulated on 500 trading days -- our whole price window -- with strategies
that have *no* real edge:

     1 trial   ->  best Sharpe 0.01   (95th pct 1.17)
     3 trials  ->  best Sharpe 0.62   (95th pct 1.55)
    90 trials  ->  best Sharpe 1.77   (95th pct 2.33)
   270 trials  ->  best Sharpe 2.02   (95th pct 2.52)
  1000 trials  ->  best Sharpe 2.33   (95th pct 2.81)

Testing a thousand ideas does not merely risk a false positive: it moves the
significance bar to 2.81, past where almost anything genuine lives. Uncounted
trials therefore destroy the ability to recognise a real finding, not just the
ability to reject a fake one.

Bailey & López de Prado's Deflated Sharpe Ratio corrects for this, but it needs
the true trial count as an input. That is why logging is a hard requirement
rather than a reporting convenience -- and why abandoned trials must be logged
too. A trial you ran and discarded because it looked bad still consumed a
lottery ticket.

    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
"""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

#: Euler-Mascheroni constant, used in the expected-maximum-Sharpe term.
EULER_GAMMA = 0.5772156649015329

_N = NormalDist()

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis   TEXT    NOT NULL,
    rationale    TEXT    NOT NULL,
    params       TEXT    NOT NULL,
    split        TEXT    NOT NULL,
    sharpe       REAL,
    n_obs        INTEGER,
    n_trades     INTEGER,
    skew         REAL,
    kurtosis     REAL,
    outcome      TEXT    NOT NULL,
    notes        TEXT,
    created_at   TEXT    NOT NULL,
    -- Identity of the experiment, not of the execution. See `register`.
    fingerprint  TEXT,
    runs         INTEGER NOT NULL DEFAULT 1,
    last_run_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_hypothesis ON trials(hypothesis);
-- The fingerprint index is created in _migrate, NOT here. On a database that
-- predates the column, executescript runs before the ALTER TABLE and the index
-- would reference a column that does not exist yet.

-- Every holdout read is recorded. The holdout is only meaningful while it stays
-- untouched, so a count above one per finalist is itself a finding.
CREATE TABLE IF NOT EXISTS holdout_access (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis TEXT NOT NULL,
    reason     TEXT NOT NULL,
    accessed_at TEXT NOT NULL
);
"""

PENDING = "pending"
COMPLETED = "completed"
ABANDONED = "abandoned"


class TrialError(RuntimeError):
    pass


@dataclass(frozen=True)
class Trial:
    trial_id: int
    hypothesis: str
    rationale: str
    params: dict[str, Any]
    split: str
    sharpe: float | None
    n_obs: int | None
    outcome: str


class TrialRegistry:
    """Append-only log of every backtest ever run against this dataset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the identity columns to a registry that predates them.

        Existing rows keep fingerprint NULL, so they neither collide with new
        trials nor get deduplicated retroactively. That is the honest handling:
        those rows were real executions and we cannot now reconstruct which of
        them were re-runs of each other.
        """
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(trials)")}
        for name, decl in (("fingerprint", "TEXT"),
                           ("runs", "INTEGER NOT NULL DEFAULT 1"),
                           ("last_run_at", "TEXT")):
            if name not in have:
                self._conn.execute(f"ALTER TABLE trials ADD COLUMN {name} {decl}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trials_fingerprint "
            "ON trials(fingerprint) WHERE fingerprint IS NOT NULL")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def register(
        self,
        hypothesis: str,
        rationale: str,
        *,
        params: dict[str, Any] | None = None,
        split: str = "train",
        dataset: str = "",
    ) -> int:
        """Record a trial *before* running it, and return its id.

        Registering first is deliberate. If trials were logged on completion, an
        experiment abandoned halfway -- the ones most likely to have looked
        unpromising -- would vanish from the count and inflate every later result.

        A rationale is mandatory: a hypothesis with a stated mechanism carries a
        better prior than a mined pattern, and it can be falsified twice over.

        **Re-running an identical trial updates it rather than appending.** The
        Deflated Sharpe penalises the BREADTH of a search -- how many distinct
        configurations were tried -- not how many times each was executed.
        Testing one hypothesis on Monday and again on Tuesday is one hypothesis,
        and counting it twice inflates N without any corresponding increase in
        the opportunity to get lucky.

        This mattered concretely. `measure` is a scheduled nightly job sweeping
        ~200 candidate-horizon combinations, so appending on every execution
        added ~6,000 phantom trials a month and raised the significance bar for
        every real result with no new search having occurred.

        Identity is (hypothesis, params, split, dataset). `dataset` is the
        caller's fingerprint of the data the trial ran against: when the data
        genuinely changes, the same hypothesis IS a new trial, because a second
        look at fresh data is a second chance to get lucky. Passing "" opts out
        and restores append-on-every-call, which is only correct for a caller
        that has no stable notion of its dataset.
        """
        if not hypothesis.strip():
            raise TrialError("hypothesis name is required")
        if not rationale.strip():
            raise TrialError(
                "a rationale is required: state the mechanism you expect to "
                "produce returns, before seeing whether it does"
            )
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(params or {}, sort_keys=True, default=str)
        fingerprint = None
        if dataset:
            fingerprint = hashlib.sha256(
                "\x1f".join((hypothesis.strip(), params_json, split, dataset))
                .encode("utf-8")
            ).hexdigest()
            row = self._conn.execute(
                "SELECT trial_id FROM trials WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is not None:
                # Same experiment, same data, run again. Count the re-run for
                # visibility but do not add to N.
                self._conn.execute(
                    "UPDATE trials SET runs = runs + 1, last_run_at = ?, "
                    "outcome = ? WHERE trial_id = ?",
                    (now, PENDING, int(row["trial_id"])),
                )
                self._conn.commit()
                return int(row["trial_id"])

        cur = self._conn.execute(
            "INSERT INTO trials (hypothesis, rationale, params, split, outcome, "
            "created_at, fingerprint, runs, last_run_at) "
            "VALUES (?,?,?,?,?,?,?,1,?)",
            (
                hypothesis.strip(),
                rationale.strip(),
                params_json,
                split,
                PENDING,
                now,
                fingerprint,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def complete(
        self,
        trial_id: int,
        *,
        sharpe: float,
        n_obs: int,
        n_trades: int | None = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        notes: str = "",
    ) -> None:
        self._conn.execute(
            "UPDATE trials SET sharpe=?, n_obs=?, n_trades=?, skew=?, kurtosis=?, "
            "outcome=?, notes=? WHERE trial_id=?",
            (sharpe, n_obs, n_trades, skew, kurtosis, COMPLETED, notes, trial_id),
        )
        self._conn.commit()

    def abandon(self, trial_id: int, notes: str = "") -> None:
        """Mark a trial abandoned. It still counts toward N."""
        self._conn.execute(
            "UPDATE trials SET outcome=?, notes=? WHERE trial_id=?",
            (ABANDONED, notes, trial_id),
        )
        self._conn.commit()

    def count(self) -> int:
        """Distinct trials, including abandoned ones. This is N for the DSR.

        One row per (hypothesis, params, split, dataset), however many times it
        has been executed -- see `register`. `executions` reports the raw count
        for operational visibility.
        """
        return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def executions(self) -> int:
        """Total runs across all trials. Exceeds `count` when a sweep repeats."""
        return int(self._conn.execute(
            "SELECT COALESCE(SUM(runs), 0) FROM trials").fetchone()[0])

    def sharpes(self) -> list[float]:
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT sharpe FROM trials WHERE sharpe IS NOT NULL"
            )
        ]

    def get(self, trial_id: int) -> Trial | None:
        row = self._conn.execute(
            "SELECT * FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            return None
        return Trial(
            trial_id=row["trial_id"], hypothesis=row["hypothesis"],
            rationale=row["rationale"], params=json.loads(row["params"]),
            split=row["split"], sharpe=row["sharpe"], n_obs=row["n_obs"],
            outcome=row["outcome"],
        )

    def record_holdout_access(self, hypothesis: str, reason: str) -> int:
        self._conn.execute(
            "INSERT INTO holdout_access (hypothesis, reason, accessed_at) "
            "VALUES (?,?,?)",
            (hypothesis, reason, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM holdout_access WHERE hypothesis = ?",
                (hypothesis,),
            ).fetchone()[0]
        )

    def holdout_accesses(self, hypothesis: str | None = None) -> int:
        if hypothesis:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM holdout_access WHERE hypothesis = ?",
                (hypothesis,),
            )
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM holdout_access")
        return int(cur.fetchone()[0])


# --- Deflated Sharpe Ratio ---------------------------------------------------

def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across `n_trials` strategies with no real edge.

    This is the benchmark a candidate must beat -- not zero. Selecting the best
    of many trials produces a positive Sharpe by construction, and this
    quantifies how positive.
    """
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(max(sharpe_variance, 0.0))
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the strategy's true Sharpe exceeds zero, after correcting for
    selection across `n_trials`, sample length, skew and fat tails.

    `observed_sharpe` must be **per-observation**, not annualised. Feeding an
    annualised figure inflates the result by roughly sqrt(252).

    Returns a probability in [0, 1]. Values below ~0.95 do not support a claim
    of edge.
    """
    if n_obs < 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    if denom <= 0:
        return 0.0
    z = (observed_sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _N.cdf(z)


def annualised_to_per_obs(annual_sharpe: float, periods_per_year: int = 252) -> float:
    """Convert an annualised Sharpe to the per-observation form the DSR needs."""
    return annual_sharpe / math.sqrt(periods_per_year)


def assess(
    registry: TrialRegistry,
    observed_sharpe_annual: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> dict[str, float | int | bool]:
    """Score a candidate against every trial the registry has ever seen."""
    n_trials = max(registry.count(), 1)
    sharpes = registry.sharpes()
    if len(sharpes) >= 2:
        mean = sum(sharpes) / len(sharpes)
        variance = sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1)
    else:
        # No spread observed yet. Fall back to the sampling variance of a Sharpe
        # estimate under the null, which is ~1/n_obs per observation.
        variance = 1.0 / max(n_obs, 1)

    sr = annualised_to_per_obs(observed_sharpe_annual, periods_per_year)
    dsr = deflated_sharpe(
        sr, n_trials=n_trials, n_obs=n_obs,
        sharpe_variance=variance, skew=skew, kurtosis=kurtosis,
    )
    return {
        "n_trials": n_trials,
        "observed_sharpe_annual": observed_sharpe_annual,
        "expected_max_sharpe_annual": expected_max_sharpe(n_trials, variance)
        * math.sqrt(periods_per_year),
        "deflated_sharpe": dsr,
        "significant": dsr >= 0.95,
    }
