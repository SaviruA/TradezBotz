"""Tests for repairing mangled tickers in the backfill queue.

`repair-symbols` fixed the event store, which is where symbols come from -- but
the queue is a separate table with its own rows. A ticker already enqueued as
'"OMEX"' stayed enqueued that way, burned its three attempts against a vendor
that cannot parse it, and parked. CI carried 169 such rows after the event store
was clean.
"""

from __future__ import annotations

from tradezbotz.research.backfill import repair_queue
from tradezbotz.research.edgar import normalise_symbol


class FakeRunner:
    """Minimal stand-in exposing the connection repair_queue needs."""

    def __init__(self, tmp_path, rows):
        import sqlite3

        self._conn = sqlite3.connect(tmp_path / "b.db")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE backfill (symbol TEXT PRIMARY KEY, status TEXT NOT NULL,"
            " attempts INTEGER NOT NULL DEFAULT 0, bars INTEGER, is_active INTEGER,"
            " last_error TEXT, updated_at TEXT)")
        for sym, status in rows:
            self._conn.execute(
                "INSERT INTO backfill (symbol, status, attempts, updated_at) "
                "VALUES (?,?,3,'now')", (sym, status))
        self._conn.commit()

    def symbols(self):
        return sorted(r[0] for r in self._conn.execute("SELECT symbol FROM backfill"))

    def status_of(self, sym):
        r = self._conn.execute(
            "SELECT status FROM backfill WHERE symbol = ?", (sym,)).fetchone()
        return r["status"] if r else None

    def close(self):
        self._conn.close()


def test_a_mangled_ticker_is_replaced_by_its_repaired_form(tmp_path):
    r = FakeRunner(tmp_path, [('"OMEX"', "failed"), ("NYSE: KRC", "failed")])

    stats = repair_queue(r, normalise_symbol)

    assert stats["repaired"] == 2
    assert r.symbols() == ["KRC", "OMEX"]
    assert r.status_of("OMEX") == "pending", "repaired rows retry"
    r.close()


def test_an_unusable_ticker_is_dropped_rather_than_retried_forever(tmp_path):
    r = FakeRunner(tmp_path, [("N/A", "failed"), ("--", "failed")])

    stats = repair_queue(r, normalise_symbol)

    assert stats["dropped"] == 2
    assert r.symbols() == []
    r.close()


def test_a_clean_ticker_is_left_completely_alone(tmp_path):
    """Including its status. Resetting a finished symbol to pending would
    refetch thousands of symbols for no reason."""
    r = FakeRunner(tmp_path, [("AAPL", "done"), ("MSFT", "done")])

    stats = repair_queue(r, normalise_symbol)

    assert stats["already_ok"] == 2
    assert r.status_of("AAPL") == "done"
    r.close()


def test_repairing_onto_an_existing_symbol_keeps_the_existing_status(tmp_path):
    """'(SIRI)' and 'SIRI' can both be queued. The mangled one must go without
    resetting the good one, which may already be fetched."""
    r = FakeRunner(tmp_path, [("SIRI", "done"), ("(SIRI)", "failed")])

    repair_queue(r, normalise_symbol)

    assert r.symbols() == ["SIRI"]
    assert r.status_of("SIRI") == "done", "the finished row survives untouched"
    r.close()


def test_repair_is_idempotent(tmp_path):
    r = FakeRunner(tmp_path, [('"OMEX"', "failed")])

    repair_queue(r, normalise_symbol)
    second = repair_queue(r, normalise_symbol)

    assert second["repaired"] == 0
    assert r.symbols() == ["OMEX"]
    r.close()
