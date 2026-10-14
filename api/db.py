"""SQLite persistence for application metadata (stdlib sqlite3, one file under results/app/).

Stored here: experiment records (configuration, status, progress, aggregated results), audit events, settings
and generated-report metadata. Never stored here: raw survey rows, split indices or model weights.
Each call opens its own connection, so the module is safe to use from the request threads and the experiment
worker thread at the same time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    config TEXT NOT NULL,
    progress TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
"""

_write_lock = threading.Lock()


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("config", "progress", "result", "detail", "meta", "value"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    def health(self) -> dict[str, Any]:
        try:
            with self.connect() as con:
                counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                          for t in ("experiments", "audit_events", "settings", "reports")}
            return {"ok": True, "path": str(self.path), "size_bytes": self.path.stat().st_size, **counts}
        except sqlite3.Error as exc:  # pragma: no cover - only on a corrupted file
            return {"ok": False, "path": str(self.path), "error": str(exc)}

    # ------------------------------------------------------------------ experiments
    def create_experiment(self, exp_id: str, name: str, algorithm: str, config: dict) -> dict:
        with _write_lock, self.connect() as con:
            con.execute("INSERT INTO experiments (id, name, algorithm, status, created_at, config) VALUES (?,?,?,?,?,?)",
                        (exp_id, name, algorithm, "queued", time.time(), json.dumps(config)))
        return self.get_experiment(exp_id)  # type: ignore[return-value]

    def update_experiment(self, exp_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k} = ?")
            vals.append(json.dumps(v) if k in ("progress", "result", "config") else v)
        with _write_lock, self.connect() as con:
            con.execute(f"UPDATE experiments SET {', '.join(cols)} WHERE id = ?", (*vals, exp_id))

    def get_experiment(self, exp_id: str) -> dict | None:
        with self.connect() as con:
            return _row(con.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone())

    def list_experiments(self, limit: int = 200) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]  # type: ignore[misc]

    def running_experiment(self) -> dict | None:
        with self.connect() as con:
            r = con.execute("SELECT * FROM experiments WHERE status IN ('queued','running') ORDER BY created_at LIMIT 1").fetchone()
        return _row(r)

    def request_cancel(self, exp_id: str) -> None:
        with _write_lock, self.connect() as con:
            con.execute("UPDATE experiments SET cancel_requested = 1 WHERE id = ?", (exp_id,))

    def cancel_requested(self, exp_id: str) -> bool:
        with self.connect() as con:
            r = con.execute("SELECT cancel_requested FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        return bool(r and r[0])

    def mark_interrupted(self) -> int:
        """Experiments left 'running' by a previous process cannot be resumed; mark them failed on startup."""
        with _write_lock, self.connect() as con:
            cur = con.execute("UPDATE experiments SET status = 'failed', finished_at = ?, "
                              "error = 'backend restarted while the experiment was running' "
                              "WHERE status IN ('queued','running')", (time.time(),))
            return cur.rowcount

    def delete_experiment(self, exp_id: str) -> bool:
        with _write_lock, self.connect() as con:
            return con.execute("DELETE FROM experiments WHERE id = ? AND status NOT IN ('queued','running')", (exp_id,)).rowcount > 0

    # ------------------------------------------------------------------ audit
    def audit(self, event: str, **detail: Any) -> None:
        with _write_lock, self.connect() as con:
            con.execute("INSERT INTO audit_events (ts, event, detail) VALUES (?,?,?)",
                        (time.time(), event, json.dumps(detail, default=str)))

    def audit_events(self, limit: int = 100) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------ settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as con:
            r = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value: Any) -> None:
        with _write_lock, self.connect() as con:
            con.execute("INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (key, json.dumps(value), time.time()))

    def all_settings(self) -> dict[str, Any]:
        with self.connect() as con:
            rows = con.execute("SELECT key, value FROM settings").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    # ------------------------------------------------------------------ reports
    def add_report(self, rep_id: str, kind: str, title: str, filename: str, media_type: str, size: int, meta: dict) -> dict:
        with _write_lock, self.connect() as con:
            con.execute("INSERT INTO reports (id, kind, title, created_at, filename, media_type, size_bytes, meta) VALUES (?,?,?,?,?,?,?,?)",
                        (rep_id, kind, title, time.time(), filename, media_type, size, json.dumps(meta, default=str)))
        return self.get_report(rep_id)  # type: ignore[return-value]

    def get_report(self, rep_id: str) -> dict | None:
        with self.connect() as con:
            return _row(con.execute("SELECT * FROM reports WHERE id = ?", (rep_id,)).fetchone())

    def list_reports(self, limit: int = 100) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]  # type: ignore[misc]
