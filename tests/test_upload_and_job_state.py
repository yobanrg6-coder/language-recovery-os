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

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.schemas import JobStatus
from store import job_store
from web_app import app as app_module


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
