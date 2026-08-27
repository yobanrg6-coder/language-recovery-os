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

from agents.schemas import Job, JobStatus

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "jobs.db")

# A job can only be picked up for a pipeline run from one of these states.
# RUNNING / WAITING_HUMAN / COMPLETED are deliberately excluded: re-running
# rebuilds job.claims from scratch and would wipe any human validations.
CLAIMABLE_FOR_PROCESSING = {JobStatus.QUEUED.value, JobStatus.FAILED.value}

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


def try_claim_job(db_path: str, job_id: str) -> Job | None:
    """Atomically flip a job to RUNNING inside one write-locked transaction
    and return the updated Job, or None if the job is missing or is not in a
    CLAIMABLE_FOR_PROCESSING state (i.e. a concurrent request -- double
    click, EventSource auto-reconnect -- already claimed it).

    This is the only race-safe gate against two pipeline runs on the same
    job: reading the status into memory in the request handler and checking
    it later, once the SSE body has started streaming, leaves a window where
    both requests see QUEUED and both run the full pipeline (double Gemini
    spend, racing job writes that can clobber a human validation).
    BEGIN IMMEDIATE takes the write lock up front so a second caller blocks,
    then reads RUNNING and backs off."""
    with closing(get_connection(db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT job_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            job = Job.model_validate_json(row["job_json"])
            if job.status.value not in CLAIMABLE_FOR_PROCESSING:
                conn.rollback()
                return None
            running = job.model_copy(update={"status": JobStatus.RUNNING})
            conn.execute(
                "UPDATE jobs SET status = ?, job_json = ? WHERE id = ?",
                (running.status.value, running.model_dump_json(), job_id),
            )
            conn.commit()
            return running
        except BaseException:
            conn.rollback()
            raise


def mutate_job(db_path: str, job_id: str, fn) -> Job | None:
    """Read-modify-write a job inside one write-locked transaction. `fn` is
    called with the current Job and must return the new Job to persist (or
    None to abort without writing). Returns the persisted Job, or None if the
    job doesn't exist / `fn` aborted. `fn` may raise (e.g. an HTTPException):
    the transaction is rolled back and the exception propagates.

    Use this for any endpoint that reads a job, changes part of it, and
    writes it back: save_job() overwrites the whole JSON blob, so two such
    endpoints running concurrently (e.g. two validators confirming different
    claims of the same job) would lose one another's change on a plain
    get -> edit -> save. BEGIN IMMEDIATE serializes them."""
    with closing(get_connection(db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT job_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            new_job = fn(Job.model_validate_json(row["job_json"]))
            if new_job is None:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE jobs SET status = ?, job_json = ? WHERE id = ?",
                (new_job.status.value, new_job.model_dump_json(), job_id),
            )
            conn.commit()
            return new_job
        except BaseException:
            conn.rollback()
            raise


def get_job(db_path: str, job_id: str) -> Job | None:
    with closing(get_connection(db_path)) as conn:
        row = conn.execute("SELECT job_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row["job_json"]) if row else None


def list_recent_jobs(db_path: str, limit: int = 20) -> list[dict]:
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, name, status, created_at FROM jobs ORDER BY created_at DESC LIMIT ?",
            (max(1, limit),),  # SQLite treats a negative LIMIT as unbounded
        ).fetchall()
        return [dict(row) for row in rows]


def delete_job(db_path: str, job_id: str) -> None:
    with closing(get_connection(db_path)) as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
