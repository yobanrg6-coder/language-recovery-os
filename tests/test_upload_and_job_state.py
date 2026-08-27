"""
Regression tests for 3 findings from the 2026-08-23 audit
(see BITACORA_PROYECTO.md), all in web_app/app.py:

1. Orphaned upload files: a file that fails the size limit mid-manifest
   must not leave earlier files in the loop stranded on disk with no job
   record pointing to them.
2. Filename collision: two sources in the same manifest with the same
   original filename must not silently overwrite each other on disk.
3. Status race: a job's status must be persisted as RUNNING *before* the
   pipeline starts, so a duplicate /process-stream call (double click,
   EventSource auto-reconnect) is rejected by _ALREADY_PROCESSED_STATUSES
   instead of racing a second pipeline execution.

Isolated from the real data/jobs.db and data/uploads by monkeypatching
job_store.DEFAULT_DB_PATH and app.UPLOAD_ROOT to a pytest tmp_path.
"""

import asyncio
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.schemas import Claim, ClaimStatus, ClaimType, Job, JobStatus
from store import job_store
from web_app import app as app_module


def _job_with_pending_claims(db_path, *claim_ids):
    job = Job(
        job_id="job-" + "-".join(claim_ids),
        name="test",
        status=JobStatus.WAITING_HUMAN,
        claims=[
            Claim(
                claim_id=cid, job_id="j", claim_type=ClaimType.TRANSCRIPTION,
                source_id="s", value=f"value-{cid}", status=ClaimStatus.NEEDS_VALIDATION,
            )
            for cid in claim_ids
        ],
        created_at="2026-08-26T00:00:00+00:00",
    )
    job_store.save_job(db_path, job)
    return job


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "DEFAULT_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(app_module.job_store, "DEFAULT_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(app_module, "UPLOAD_ROOT", tmp_path / "uploads")
    app_module._request_log.clear()
    return TestClient(app_module.app)


def _manifest(entries):
    return json.dumps(entries)


def test_oversized_file_cleans_up_earlier_uploads_in_same_manifest(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 32)
    manifest = _manifest(
        [
            {"source_type": "text", "access_level": "PUBLIC"},
            {"source_type": "text", "access_level": "PUBLIC"},
        ]
    )
    small_file = io.BytesIO(b"a small valid file")
    too_big = io.BytesIO(b"x" * (app_module.MAX_UPLOAD_BYTES + 1))

    response = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[
            ("files", ("first.txt", small_file, "text/plain")),
            ("files", ("second.txt", too_big, "text/plain")),
        ],
    )

    assert response.status_code == 413
    upload_dirs = list((app_module.UPLOAD_ROOT).glob("*")) if app_module.UPLOAD_ROOT.exists() else []
    assert upload_dirs == [], f"expected no orphaned upload dirs, found {upload_dirs}"


def test_duplicate_filenames_do_not_overwrite_each_other(client):
    manifest = _manifest(
        [
            {"source_type": "text", "access_level": "PUBLIC"},
            {"source_type": "text", "access_level": "PUBLIC"},
        ]
    )
    response = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[
            ("files", ("audio.webm", io.BytesIO(b"content-A"), "text/plain")),
            ("files", ("audio.webm", io.BytesIO(b"content-B"), "text/plain")),
        ],
    )

    assert response.status_code == 200
    job = response.json()
    filenames = [s["filename"] for s in job["sources"]]
    assert len(set(filenames)) == 2, f"filenames collided: {filenames}"

    upload_dir = app_module.UPLOAD_ROOT / job["job_id"]
    saved_files = sorted(p.name for p in upload_dir.iterdir())
    assert len(saved_files) == 2
    contents = {p.read_bytes() for p in upload_dir.iterdir()}
    assert contents == {b"content-A", b"content-B"}


class _FakeOrchestrator:
    """Stands in for RecoveryOrchestrator: no real Gemini/network call, just
    reports what job status was already persisted by the time the pipeline
    started, which is exactly what the race-condition fix controls."""

    def __init__(self, api_key=None):
        self.api_key = api_key

    async def build_recovery_stream(self, job, upload_dir):
        persisted = job_store.get_job(job_store.DEFAULT_DB_PATH, job.job_id)
        yield {"type": "status_seen_at_pipeline_start", "status": persisted.status.value}


def test_try_claim_job_is_a_one_shot_gate(client):
    """2026-08-26 audit M1: only the first claimer flips the job to RUNNING;
    a concurrent second claim (double click / EventSource reconnect) gets
    None instead of racing a second pipeline run."""
    manifest = _manifest([{"source_type": "text", "access_level": "PUBLIC"}])
    job_id = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[("files", ("note.txt", io.BytesIO(b"hello"), "text/plain"))],
    ).json()["job_id"]

    first = job_store.try_claim_job(job_store.DEFAULT_DB_PATH, job_id)
    assert first is not None and first.status == JobStatus.RUNNING

    second = job_store.try_claim_job(job_store.DEFAULT_DB_PATH, job_id)
    assert second is None, "a RUNNING job must not be claimable again"

    # unknown id is also unclaimable, not an error
    assert job_store.try_claim_job(job_store.DEFAULT_DB_PATH, "does-not-exist") is None

    # a FAILED job IS retryable
    job_store.save_job(
        job_store.DEFAULT_DB_PATH,
        first.model_copy(update={"status": JobStatus.FAILED}),
    )
    retried = job_store.try_claim_job(job_store.DEFAULT_DB_PATH, job_id)
    assert retried is not None and retried.status == JobStatus.RUNNING


def test_process_stream_persists_running_before_pipeline_starts(client, monkeypatch):
    manifest = _manifest([{"source_type": "text", "access_level": "PUBLIC"}])
    create_response = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[("files", ("note.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    job_id = create_response.json()["job_id"]

    monkeypatch.setattr(app_module, "RecoveryOrchestrator", _FakeOrchestrator)

    with client.stream("GET", f"/api/jobs/{job_id}/process-stream", params={"api_key": "fake-key-not-used"}) as response:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                break

    assert events[0] == {"type": "status_seen_at_pipeline_start", "status": "RUNNING"}

    final_job = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
    assert final_job.status == JobStatus.RUNNING


class _CancellingOrchestrator:
    """Yields one event, then raises CancelledError - simulating the client
    disconnecting (tab closed / EventSource dropped) or Cloud Run hitting its
    request deadline while the pipeline streams."""

    def __init__(self, api_key=None):
        self.api_key = api_key

    async def build_recovery_stream(self, job, upload_dir):
        yield {"type": "status", "message": "started"}
        raise asyncio.CancelledError()


def test_cancelled_stream_marks_job_failed_and_leaves_it_retryable(client, monkeypatch):
    """2026-08-26 audit A2: CancelledError is a BaseException, so the generic
    `except Exception` never persisted FAILED and the job stayed stuck at
    RUNNING - which _ALREADY_PROCESSED_STATUSES then blocks forever."""
    manifest = _manifest([{"source_type": "text", "access_level": "PUBLIC"}])
    create_response = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[("files", ("note.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    job_id = create_response.json()["job_id"]

    monkeypatch.setattr(app_module, "RecoveryOrchestrator", _CancellingOrchestrator)

    try:
        with client.stream("GET", f"/api/jobs/{job_id}/process-stream", params={"api_key": "fake-key"}) as response:
            for _ in response.iter_lines():
                pass
    except Exception:
        pass  # the re-raised CancelledError may surface through the test transport

    failed_job = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
    assert failed_job.status == JobStatus.FAILED

    # FAILED is not a blocked status, so the job can still be re-processed in
    # place instead of being wedged forever.
    monkeypatch.setattr(app_module, "RecoveryOrchestrator", _FakeOrchestrator)
    with client.stream("GET", f"/api/jobs/{job_id}/process-stream", params={"api_key": "fake-key"}) as response:
        first = None
        for line in response.iter_lines():
            if line.startswith("data: "):
                first = json.loads(line[len("data: "):])
                break
    assert first["type"] == "status_seen_at_pipeline_start"


def _validate(client, job_id, claim_id, value="confirmed", name="Ana", role="linguist", notes=None):
    return client.post(
        f"/api/jobs/{job_id}/claims/{claim_id}/validate",
        json={"validator_name": name, "validator_role": role, "decision_value": value, "notes": notes},
    )


def test_validation_is_persisted_as_an_audit_record(client):
    """2026-08-26 audit M3: the Validation (who/when/notes) used to be
    returned once and dropped - Job had nowhere to keep it."""
    db = job_store.DEFAULT_DB_PATH
    job = _job_with_pending_claims(db, "c1", "c2")

    r = _validate(client, job.job_id, "c1", value="kuyfi", name="Marta", role="community elder", notes="checked against grandmother's usage")
    assert r.status_code == 200

    reloaded = job_store.get_job(db, job.job_id)
    assert len(reloaded.validations) == 1
    v = reloaded.validations[0]
    assert (v.claim_id, v.validator_name, v.validator_role, v.decision_value) == \
        ("c1", "Marta", "community elder", "kuyfi")
    assert v.notes == "checked against grandmother's usage" and v.timestamp
    # job still WAITING_HUMAN because c2 is still pending
    assert reloaded.status == JobStatus.WAITING_HUMAN

    # second validation appends, and the last pending claim flips job -> COMPLETED
    _validate(client, job.job_id, "c2", value="pu")
    reloaded = job_store.get_job(db, job.job_id)
    assert [x.claim_id for x in reloaded.validations] == ["c1", "c2"]
    assert reloaded.status == JobStatus.COMPLETED


def test_validate_rejects_non_pending_claim_with_409(client):
    db = job_store.DEFAULT_DB_PATH
    job = _job_with_pending_claims(db, "c1")
    assert _validate(client, job.job_id, "c1").status_code == 200
    # already COMMUNITY_VALIDATED now
    again = _validate(client, job.job_id, "c1")
    assert again.status_code == 409
    # and no phantom second audit record was written
    assert len(job_store.get_job(db, job.job_id).validations) == 1


def test_validate_unknown_job_and_claim_are_404(client):
    db = job_store.DEFAULT_DB_PATH
    job = _job_with_pending_claims(db, "c1")
    assert _validate(client, "no-such-job", "c1").status_code == 404
    assert _validate(client, job.job_id, "no-such-claim").status_code == 404


def test_second_process_stream_call_is_rejected_while_running(client, monkeypatch):
    manifest = _manifest([{"source_type": "text", "access_level": "PUBLIC"}])
    create_response = client.post(
        "/api/jobs",
        data={"manifest": manifest},
        files=[("files", ("note.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    job_id = create_response.json()["job_id"]
    job = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
    job_store.save_job(job_store.DEFAULT_DB_PATH, job.model_copy(update={"status": JobStatus.RUNNING}))

    monkeypatch.setattr(app_module, "RecoveryOrchestrator", _FakeOrchestrator)

    with client.stream("GET", f"/api/jobs/{job_id}/process-stream", params={"api_key": "fake-key-not-used"}) as response:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                break

    assert events[0]["type"] == "error"
    assert "already" in events[0]["message"]
