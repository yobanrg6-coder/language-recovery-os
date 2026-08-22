"""
SQLite persistence for Jobs -- same plain-functions, own-connection-per-call
pattern as store/build_pack_store.py in the sibling project ScopeCouncil.
Documented limitation, same as the sibling projects: state is lost on a
Cloud Run cold start; Firestore is the upgrade path if this needs to
survive across instances. The whole job (sources, claims, archive analysis)
is stored as one JSON blob per row -- there is no cross-job querying need
for the demo beyond "list recent" and "get by id".
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from agents.schemas import Job

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "jobs.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    job_json TEXT NOT NULL
);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def save_job(db_path: str, job: Job) -> None:
    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (id, name, status, created_at, job_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job.job_id, job.name, job.status.value, job.created_at, job.model_dump_json()),
        )
        conn.commit()


def get_job(db_path: str, job_id: str) -> Job | None:
    with closing(get_connection(db_path)) as conn:
        row = conn.execute("SELECT job_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row["job_json"]) if row else None


def list_recent_jobs(db_path: str, limit: int = 20) -> list[dict]:
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, name, status, created_at FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_job(db_path: str, job_id: str) -> None:
    with closing(get_connection(db_path)) as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
