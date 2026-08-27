"""
FastAPI Backend - Language Recovery OS
Serves the web UI, an upload endpoint, an SSE streaming endpoint for the
recovery pipeline, and a human-validation endpoint that resumes a
WAITING_HUMAN job.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("languagerecoveryos")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from agents.orchestrator import RecoveryOrchestrator
from agents.schemas import AccessLevel, Job, JobStatus, Source, SourceType, Validation
from agents.scoring import apply_community_validation
from store import job_store

load_dotenv()

app = FastAPI(
    title="Language Recovery OS",
    description="Turns fragmented language archives into living, traceable, community-validated knowledge. "
    "Every claim carries evidence, confidence and provenance; the accept/escalate decision is always "
    "computed in Python, never asserted by a model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Same rationale as the sibling hackathon projects: --allow-unauthenticated
# on Cloud Run is required so judges can reach the demo, and this endpoint
# falls back to the server's own GEMINI_API_KEY when a caller sends none -
# without a limiter, anyone with the URL could script unlimited free calls.
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, deque] = defaultdict(deque)


def _get_client_ip(request: Request) -> str:
    """X-Forwarded-For is a chain built hop by hop, and Cloud Run does NOT
    strip whatever a client already sent - confirmed live in the sibling
    project Trusted Hire Mexico on 2026-08-17. The trustworthy entry is the
    LAST one, appended by Cloud Run's own edge."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    # The 2026-08-19 fix only pruned the CALLING ip's own entry, so a
    # one-time visitor's deque (the dominant traffic pattern on a public,
    # unauthenticated hackathon demo link - most judges/crawlers hit it once
    # and never come back) was never revisited and stayed in _request_log
    # forever. Found 2026-08-22: sweep every tracked IP's expired entries on
    # each call, not just the current one - O(distinct IPs seen), trivially
    # cheap at demo traffic volumes.
    stale_ips = []
    for ip, log in _request_log.items():
        while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
            log.popleft()
        if not log:
            stale_ips.append(ip)
    for ip in stale_ips:
        del _request_log[ip]

    log = _request_log[client_ip]
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    log.append(now)


static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

UPLOAD_ROOT = Path(PROJECT_ROOT) / "data" / "uploads"
DEMO_DATA_ROOT = Path(PROJECT_ROOT) / "demo_data"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60MB - generous for the demo audio + PDFs, not unbounded
MAX_FILES_PER_JOB = 25  # an archive job is a handful of documents, not a bulk dump

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    return _FILENAME_SAFE_RE.sub("_", base) or "file"


# Curated one-click demo archive (master spec section 6: "El usuario pulsa
# una vez"). Corpus source points at one of 3 curated aligned pairs kept
# under demo_data/corpus_repo/translation-clean/ -- see demo_data/SOURCES.md
# for licensing/citation of every demo file and why this one was picked.
DEMO_SOURCES: list[tuple[str, SourceType, str]] = [
    ("dictionary_augusta_1916_ocr.txt", SourceType.DICTIONARY,
     "Augusta, Diccionario Araucano-Español y Español-Araucano, 1916 (archive.org) - dominio público"),
    ("grammar_augusta_1903_ocr.txt", SourceType.GRAMMAR,
     "Augusta, Gramática Araucana, 1903 (archive.org) - dominio público"),
    ("corpus_repo/translation-clean/nmlch-pmope1.txt", SourceType.CORPUS,
     "AVENUE Mapudungun corpus (CMU + Chile Min. Educación + U. La Frontera), cleaned by Duan et al. 2020 - CC BY-NC-SA 3.0"),
    ("audio_victor_wikitongues_mapudungun.webm", SourceType.AUDIO,
     "Wikitongues, hablante real de Mapudungun (Wikimedia Commons) - CC BY 3.0"),
]


class ValidationRequest(BaseModel):
    validator_name: str
    validator_role: str
    decision_value: str
    notes: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        async with aiofiles.open(index_file, "r", encoding="utf-8") as f:
            return await f.read()
    return "<h1>Language Recovery OS - Web Dashboard Loading...</h1>"


@app.get("/api/health")
async def health_check():
    return {"status": "online", "service": "Language Recovery OS", "model": os.getenv("MODEL", "gemini-3.5-flash-lite")}


@app.get("/api/jobs/recent")
async def recent_jobs(limit: int = 20):
    # clamp to [1, 50] - a negative limit reaches SQLite as `LIMIT -1`, which
    # it interprets as "no limit" and dumps every row.
    return job_store.list_recent_jobs(job_store.DEFAULT_DB_PATH, limit=max(1, min(limit, 50)))


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


@app.post("/api/jobs")
async def create_job(
    request: Request,
    job_name: str = Form(default="Archive Upload"),
    manifest: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """manifest is a JSON array, one entry per file in `files`, in the same
    order: [{"source_type": "audio", "access_level": "PUBLIC", "citation": "..."}]."""
    _check_rate_limit(_get_client_ip(request))
    try:
        entries = json.loads(manifest)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"manifest is not valid JSON: {exc}") from exc
    if len(entries) != len(files):
        raise HTTPException(status_code=400, detail="manifest length must match number of files")
    if len(files) > MAX_FILES_PER_JOB:
        raise HTTPException(status_code=400, detail=f"Too many files: max {MAX_FILES_PER_JOB} per job")

    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    sources: list[Source] = []
    try:
        for i, (entry, upload) in enumerate(zip(entries, files, strict=True)):
            try:
                source_type = SourceType(entry["source_type"])
                access_level = AccessLevel(entry.get("access_level", "PUBLIC"))
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"manifest entry {i} invalid: {exc}") from exc

            safe_name = f"{i}_{_safe_filename(upload.filename or f'source_{i}')}"
            dest_path = upload_dir / safe_name
            size = 0
            async with aiofiles.open(dest_path, "wb") as out:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        await out.close()
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail=f"File '{safe_name}' exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
                    await out.write(chunk)

            sources.append(
                Source(
                    source_id=f"src_{i}_{uuid.uuid4().hex[:8]}",
                    filename=safe_name,
                    source_type=source_type,
                    access_level=access_level,
                    citation=entry.get("citation"),
                )
            )
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    job = Job(
        job_id=job_id, name=job_name, status=JobStatus.QUEUED, sources=sources,
        created_at=datetime.now(UTC).isoformat(),
    )
    job_store.save_job(job_store.DEFAULT_DB_PATH, job)
    return job.model_dump()


@app.post("/api/jobs/demo")
async def create_demo_job(request: Request):
    """One-click demo archive - master spec section 6. Copies the curated
    public-domain/CC-licensed demo_data files into a fresh job's upload dir."""
    _check_rate_limit(_get_client_ip(request))

    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    sources: list[Source] = []
    for i, (relative_path, source_type, citation) in enumerate(DEMO_SOURCES):
        src_path = DEMO_DATA_ROOT / relative_path
        if not src_path.exists():
            raise HTTPException(status_code=500, detail=f"Demo data file missing on server: {relative_path}")
        # Same i-prefix as /api/jobs so two demo sources with the same
        # basename could never overwrite each other on disk (the 4 curated
        # names differ today, but keep the two paths consistent).
        safe_name = f"{i}_{_safe_filename(Path(relative_path).name)}"
        shutil.copyfile(src_path, upload_dir / safe_name)
        sources.append(
            Source(
                source_id=f"src_{i}_{uuid.uuid4().hex[:8]}",
                filename=safe_name,
                source_type=source_type,
                access_level=AccessLevel.PUBLIC,
                citation=citation,
            )
        )

    job = Job(
        job_id=job_id, name="Mapudungun Archive Demo", status=JobStatus.QUEUED, sources=sources,
        created_at=datetime.now(UTC).isoformat(),
    )
    job_store.save_job(job_store.DEFAULT_DB_PATH, job)
    return job.model_dump()


@app.get("/api/jobs/{job_id}/process-stream")
async def process_stream(job_id: str, request: Request, api_key: str | None = None):
    """SSE streaming endpoint for the full recovery pipeline. GET (not POST)
    so the browser's native EventSource can call it directly with no extra
    JS fetch/stream plumbing."""
    _check_rate_limit(_get_client_ip(request))
    job = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    upload_dir = UPLOAD_ROOT / job_id

    async def event_generator():
        if not resolved_key:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Missing GEMINI_API_KEY: set it in .env or pass it as a query param.'}, ensure_ascii=False)}\n\n"
            return

        # Atomically claim the job (QUEUED/FAILED -> RUNNING in one
        # write-locked transaction). A stale in-memory status check here let
        # a double click or an EventSource auto-reconnect start the pipeline
        # twice on the same job - double Gemini spend, and re-running rebuilds
        # job.claims from scratch, wiping any COMMUNITY_VALIDATED claims a
        # human already confirmed. If the claim fails the job is already
        # running or finished; it must be re-created to run again.
        running_job = job_store.try_claim_job(job_store.DEFAULT_DB_PATH, job_id)
        if running_job is None:
            current = job_store.get_job(job_store.DEFAULT_DB_PATH, job_id)
            status_value = current.status.value if current else "unknown"
            being_processed = current is not None and current.status == JobStatus.RUNNING
            message = (
                f"This job already {'is being processed' if being_processed else 'has results'} "
                f"(status={status_value}). Re-processing in place would discard any human validations "
                "already recorded on it - create a new job to run the pipeline again."
            )
            yield f"data: {json.dumps({'type': 'error', 'message': message}, ensure_ascii=False)}\n\n"
            return

        try:
            orchestrator = RecoveryOrchestrator(api_key=resolved_key)
        except Exception:
            logger.exception("Failed to initialize RecoveryOrchestrator")
            job_store.save_job(job_store.DEFAULT_DB_PATH, running_job.model_copy(update={"status": JobStatus.FAILED}))
            yield f"data: {json.dumps({'type': 'error', 'message': 'Could not initialize the orchestrator. Check that the Gemini API key is valid.'}, ensure_ascii=False)}\n\n"
            return

        try:
            async for event in orchestrator.build_recovery_stream(running_job, upload_dir):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected (tab closed, EventSource dropped) or Cloud
            # Run hit its request deadline mid-stream. CancelledError is a
            # BaseException in Python 3.8+, so the generic handler below never
            # ran - the job was left stuck at RUNNING, which then blocks it
            # from ever being retried. Persist FAILED (a claimable state, so
            # a re-run is allowed) and re-raise: never swallow a cancellation.
            logger.warning("Recovery stream cancelled for job %s; marking FAILED", job_id)
            job_store.save_job(job_store.DEFAULT_DB_PATH, running_job.model_copy(update={"status": JobStatus.FAILED}))
            raise
        except Exception:
            logger.exception("Recovery pipeline failed")
            job_store.save_job(job_store.DEFAULT_DB_PATH, running_job.model_copy(update={"status": JobStatus.FAILED}))
            error_event = {"type": "error", "message": "The agent pipeline failed during execution. Check the server logs for technical detail."}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        # Explicit charset: Starlette does not add one for a raw
        # media_type=..., and the stream carries non-ASCII transcriptions
        # (accented text, diacritics) as UTF-8 via ensure_ascii=False.
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{job_id}/claims/{claim_id}/validate")
async def validate_claim(job_id: str, claim_id: str, req: ValidationRequest, request: Request):
    """Human validation - the only place a claim can move to
    COMMUNITY_VALIDATED. Recomputing here (rather than trusting the client)
    is what lets a WAITING_HUMAN job resume: if this was the last pending
    claim, the job flips to COMPLETED in the same call (master spec H5)."""
    _check_rate_limit(_get_client_ip(request))

    validation = Validation(
        claim_id=claim_id, validator_name=req.validator_name, validator_role=req.validator_role,
        decision_value=req.decision_value, notes=req.notes, timestamp=datetime.now(UTC).isoformat(),
    )

    def _apply(job: Job) -> Job:
        # Runs inside job_store.mutate_job's write-locked transaction, so the
        # read (claim status), the recompute (still-pending -> job status)
        # and the write are one atomic step - two validators confirming
        # different claims of the same job can no longer clobber each other.
        claim_index = next((i for i, c in enumerate(job.claims) if c.claim_id == claim_id), None)
        if claim_index is None:
            raise HTTPException(status_code=404, detail="Claim not found")

        current_status = job.claims[claim_index].status.value
        if current_status not in ("NEEDS_VALIDATION", "CONFLICTED"):
            # Without this, POSTing here on an already-SUPPORTED/HYPOTHESIS/
            # COMMUNITY_VALIDATED claim silently overwrites its value and
            # forces it to COMMUNITY_VALIDATED, bypassing the deterministic
            # confidence engine - found 2026-08-22, since job_id/claim_id are
            # both visible in ordinary API responses and this endpoint has no
            # auth.
            raise HTTPException(
                status_code=409,
                detail=f"Claim is '{current_status}', not pending human validation - it cannot be re-validated.",
            )

        updated_claim = apply_community_validation(
            job.claims[claim_index].model_copy(update={"value": req.decision_value})
        )
        new_claims = list(job.claims)
        new_claims[claim_index] = updated_claim
        still_pending = any(c.status.value in ("NEEDS_VALIDATION", "CONFLICTED") for c in new_claims)
        new_status = JobStatus.WAITING_HUMAN if still_pending else JobStatus.COMPLETED
        return job.model_copy(
            update={
                "claims": new_claims,
                "status": new_status,
                "validations": [*job.validations, validation],
            }
        )

    updated_job = job_store.mutate_job(job_store.DEFAULT_DB_PATH, job_id, _apply)
    if updated_job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job": updated_job.model_dump(), "validation": validation.model_dump()}


# run.py is the one real entrypoint (fail-fast config check, configurable
# host/port, and what the Dockerfile actually runs) - this module is meant to
# be imported, not executed directly.
