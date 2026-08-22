"""
RecoveryOrchestrator

Pipeline (mirrors the master spec's section 6/20 flow, with the recortes de
scope from BITACORA_PROYECTO.md already applied):

1. ArchiveAgent inventories every uploaded source from a short text excerpt
   (governance-cleared sources only) and proposes a workflow plan.
2. For each audio source, the Transcription stage of ArchiveAgent reads the
   audio directly (Gemini multimodal, one prompt -- no dedicated ASR
   pipeline) and proposes segmented, ranked transcription hypotheses. The
   top candidate per segment becomes a DISCOVERED transcription Claim.
3. For each claim, agents/retrieval.py finds candidate snippets in the
   archive's dictionary/grammar/corpus sources in plain Python (no managed
   RAG Engine), then EvidenceAgent judges each snippet's stance and support
   strength.
4. LinguistAgent proposes a meaning/lemma/morphemes hypothesis grounded in
   the judged evidence.
5. ConflictAgent checks the judged evidence and any variants LinguistAgent
   noticed for genuine contradictions.
6. Python (agents/scoring.py) combines every signal above into a final
   confidence and status -- ACCEPT / hypothesis / human validation -- never
   the LLM. GovernanceAgent (agents/governance_agent.py) gates every model
   call by the source's declared access_level before any of the above ever
   touches its content.

Human validation happens out-of-band via web_app/app.py's separate
validate endpoint, which calls agents/scoring.py::apply_community_validation
directly and re-saves the job -- that is what "resumes" a WAITING_HUMAN job
(master spec H5: "a validation changes state and resumes the workflow").
"""

import asyncio
import logging
import os
import re
import sys
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.archive_agent import create_archive_agent, create_transcription_agent
from agents.conflict_agent import create_conflict_agent
from agents.evidence_agent import create_evidence_agent
from agents.governance_agent import GovernanceBlockedError, can_send_to_model, require_sendable
from agents.linguist_agent import create_linguist_agent
from agents.retrieval import search_all
from agents.schemas import (
    ArchiveAnalysis,
    Claim,
    ClaimStatus,
    ClaimType,
    ConflictCheckOutput,
    Evidence,
    EvidenceJudgmentOutput,
    Job,
    JobStatus,
    LinguistProposal,
    Source,
    SourceType,
    TranscriptionOutput,
)
from agents.scoring import score_claim
from store import job_store

load_dotenv()
logger = logging.getLogger("languagerecoveryos.orchestrator")

T = TypeVar("T", bound=BaseModel)

MAX_AGENT_RETRIES = 2
AGENT_CALL_TIMEOUT_SECONDS = 35.0
# Multimodal audio calls carry the full recording as input, not just a text
# prompt -- given the same reasoning as ScopeCouncil's GEMMA_CALL_TIMEOUT
# (a heavier generation task needs more budget than a plain-text call, found
# live 18-ago on that sibling project), this stage gets a longer allowance.
TRANSCRIPTION_CALL_TIMEOUT_SECONDS = 70.0
APP_NAME = "languagerecoveryos"

MAX_EXCERPT_CHARS = 6000

_FENCE_PREFIX_RE = re.compile(r"^```(?:json)?\s*\n?")
_FENCE_SUFFIX_RE = re.compile(r"\n?```\s*$")

_AUDIO_MIME_BY_EXT = {
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    stripped = _FENCE_PREFIX_RE.sub("", stripped, count=1)
    stripped = _FENCE_SUFFIX_RE.sub("", stripped, count=1)
    return stripped.strip()


class AgentExecutionError(RuntimeError):
    """Raised when an ADK agent cannot produce a valid structured response after retries."""


class ConfigurationError(RuntimeError):
    """Raised when the orchestrator is misconfigured (e.g. missing API key)."""


class RecoveryOrchestrator:
    def __init__(self, api_key: str | None = None, model: str | None = None, db_path: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ConfigurationError("GEMINI_API_KEY is not set. Provide it via .env or the request's api_key field.")
        # Not written to os.environ - built fresh per HTTP request in an
        # async server handling concurrent requests, same reasoning as the
        # sibling projects' orchestrators.
        self.model_name = model or os.getenv("MODEL", "gemini-flash-lite-latest")
        self.db_path = db_path or job_store.DEFAULT_DB_PATH

    async def _run_agent(
        self, agent, parts: list, output_model: type[T], label: str, timeout_seconds: float = AGENT_CALL_TIMEOUT_SECONDS
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(1, MAX_AGENT_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self._run_agent_once(agent, parts, output_model, label),
                    timeout=timeout_seconds,
                )
            except TimeoutError as exc:
                # Without this, a stalled model connection never raises and
                # this call would hang the request indefinitely -- same real
                # bug already found and fixed in Trusted Hire Mexico and
                # ScopeCouncil (a hung Gemma/Gemini call blocking a Cloud Run
                # request for minutes).
                last_error = exc
                logger.warning("%s attempt %d/%d timed out after %.0fs", label, attempt, MAX_AGENT_RETRIES, timeout_seconds)
                if attempt < MAX_AGENT_RETRIES:
                    await asyncio.sleep(1.0 * attempt)
            except Exception as exc:  # noqa: BLE001 - ADK/Gemini exception types vary by failure mode
                last_error = exc
                logger.warning("%s attempt %d/%d failed: %s", label, attempt, MAX_AGENT_RETRIES, exc)
                if attempt < MAX_AGENT_RETRIES:
                    await asyncio.sleep(1.0 * attempt)
        raise AgentExecutionError(f"{label} failed after {MAX_AGENT_RETRIES} attempts: {last_error}") from last_error

    async def _run_agent_once(self, agent, parts: list, output_model: type[T], label: str) -> T:
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        user_id = "languagerecoveryos-user"
        session_id = f"{label}-{uuid.uuid4().hex[:10]}"

        await runner.session_service.create_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)

        content = types.Content(role="user", parts=parts)
        final_event = None
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                final_event = event

        if final_event is None:
            raise AgentExecutionError(f"{label}: agent produced no final response")

        if getattr(final_event, "output", None) is not None:
            try:
                return output_model.model_validate(final_event.output)
            except ValidationError:
                pass

        text = None
        if final_event.content and final_event.content.parts:
            text = final_event.content.parts[-1].text
        if not text:
            raise AgentExecutionError(f"{label}: agent response had no parsable text or output")

        try:
            return output_model.model_validate_json(_strip_markdown_fence(text))
        except ValidationError as exc:
            raise AgentExecutionError(f"{label}: response did not match {output_model.__name__} schema: {exc}") from exc

    async def _run_text_agent(self, agent, prompt: str, output_model: type[T], label: str, timeout_seconds: float = AGENT_CALL_TIMEOUT_SECONDS) -> T:
        return await self._run_agent(agent, [types.Part(text=prompt)], output_model, label, timeout_seconds)

    def _searchable_sources(self, sources: list[Source], upload_dir: Path) -> list[tuple[str, Path, str]]:
        searchable = []
        for source in sources:
            if source.source_type == SourceType.AUDIO or not can_send_to_model(source):
                continue
            searchable.append((source.source_id, upload_dir / source.filename, source.source_id))
        return searchable

    async def build_recovery_stream(self, job: Job, upload_dir: Path) -> AsyncGenerator[dict[str, Any], None]:
        job = job.model_copy(update={"status": JobStatus.RUNNING})

        # -------------------------------------------------------------
        # STAGE 1: ArchiveAgent inventory (governance-cleared sources only)
        # -------------------------------------------------------------
        yield {"type": "status", "agent": "ArchiveAgent", "stage": 1, "job_id": job.job_id,
               "message": "Reading the archive materials and proposing a workflow..."}

        archive_agent = create_archive_agent(model_name=self.model_name, api_key=self.api_key)
        inventory_prompt = self._build_inventory_prompt(job.sources, upload_dir)
        try:
            archive_analysis: ArchiveAnalysis = await self._run_text_agent(
                archive_agent, inventory_prompt, ArchiveAnalysis, "ArchiveAgent"
            )
        except AgentExecutionError:
            logger.exception("ArchiveAgent failed; continuing with a minimal analysis")
            archive_analysis = ArchiveAnalysis(
                archive_summary="Archive analysis unavailable this run.", inventory=[], proposed_workflow=[]
            )
        job = job.model_copy(update={"archive_analysis": archive_analysis})
        yield {"type": "agent_result", "agent": "ArchiveAgent", "stage": 1, "job_id": job.job_id,
               "data": archive_analysis.model_dump()}

        # -------------------------------------------------------------
        # STAGE 2: Transcription (per audio source, multimodal)
        # -------------------------------------------------------------
        claims: list[Claim] = []
        audio_sources = [s for s in job.sources if s.source_type == SourceType.AUDIO]
        for source in audio_sources:
            yield {"type": "status", "agent": "TranscriptionAgent", "stage": 2, "job_id": job.job_id,
                   "message": f"Transcribing '{source.filename}' directly with Gemini..."}
            try:
                require_sendable(source)
            except GovernanceBlockedError as exc:
                yield {"type": "status", "agent": "GovernanceAgent", "stage": 2, "job_id": job.job_id,
                       "message": str(exc)}
                continue

            audio_path = upload_dir / source.filename
            try:
                transcription = await self._transcribe_audio(source, audio_path)
            except (AgentExecutionError, FileNotFoundError) as exc:
                logger.exception("Transcription failed for %s", source.filename)
                yield {"type": "status", "agent": "TranscriptionAgent", "stage": 2, "job_id": job.job_id,
                       "message": f"Transcription failed for '{source.filename}': {exc}"}
                continue

            yield {"type": "agent_result", "agent": "TranscriptionAgent", "stage": 2, "job_id": job.job_id,
                   "data": transcription.model_dump()}

            for hyp in transcription.hypotheses:
                if not hyp.candidates:
                    continue
                top = max(hyp.candidates, key=lambda c: c.confidence)
                claims.append(
                    Claim(
                        claim_id=uuid.uuid4().hex[:12],
                        job_id=job.job_id,
                        claim_type=ClaimType.TRANSCRIPTION,
                        source_id=source.source_id,
                        segment_label=hyp.segment_label,
                        value=top.text,
                        status=ClaimStatus.EXTRACTED,
                        transcription_confidence=top.confidence,
                    )
                )

        # -------------------------------------------------------------
        # STAGE 3-6: per claim -- evidence, linguist, conflict, scoring
        # -------------------------------------------------------------
        searchable = self._searchable_sources(job.sources, upload_dir)
        scored_claims: list[Claim] = []
        for claim in claims:
            yield {"type": "status", "agent": "EvidenceAgent", "stage": 3, "job_id": job.job_id,
                   "message": f"Searching the archive for evidence on '{claim.value}'..."}
            evidence = await self._gather_evidence(claim.value, searchable)
            yield {"type": "agent_result", "agent": "EvidenceAgent", "stage": 3, "job_id": job.job_id,
                   "data": {"claim_id": claim.claim_id, "evidence": [e.model_dump() for e in evidence]}}

            yield {"type": "status", "agent": "LinguistAgent", "stage": 4, "job_id": job.job_id,
                   "message": f"Proposing a meaning hypothesis for '{claim.value}'..."}
            linguist_proposal = await self._propose_meaning(claim.value, evidence)
            yield {"type": "agent_result", "agent": "LinguistAgent", "stage": 4, "job_id": job.job_id,
                   "data": {"claim_id": claim.claim_id, "linguist_proposal": linguist_proposal.model_dump() if linguist_proposal else None}}

            yield {"type": "status", "agent": "ConflictAgent", "stage": 5, "job_id": job.job_id,
                   "message": f"Checking '{claim.value}' for cross-source conflicts..."}
            conflict_result = await self._check_conflict(claim.value, evidence, linguist_proposal)
            if conflict_result.conflicts:
                yield {"type": "agent_result", "agent": "ConflictAgent", "stage": 5, "job_id": job.job_id,
                       "data": {"claim_id": claim.claim_id, **conflict_result.model_dump()}}

            yield {"type": "status", "agent": "ScoringEngine", "stage": 6, "job_id": job.job_id,
                   "message": "Combining signals into a final confidence (Python, not a model)..."}
            scored = score_claim(
                claim.model_copy(update={"linguist_proposal": linguist_proposal}),
                claim.transcription_confidence,
                evidence,
                conflict_result.conflicts,
            )
            scored_claims.append(scored)
            yield {"type": "agent_result", "agent": "ScoringEngine", "stage": 6, "job_id": job.job_id,
                   "data": scored.model_dump()}

        needs_human = any(c.status in (ClaimStatus.NEEDS_VALIDATION, ClaimStatus.CONFLICTED) for c in scored_claims)
        final_status = JobStatus.WAITING_HUMAN if needs_human else JobStatus.COMPLETED
        job = job.model_copy(update={"claims": scored_claims, "status": final_status})
        job_store.save_job(self.db_path, job)

        auto_accepted = sum(1 for c in scored_claims if c.status == ClaimStatus.SUPPORTED)
        pending = sum(1 for c in scored_claims if c.status in (ClaimStatus.NEEDS_VALIDATION, ClaimStatus.CONFLICTED))
        yield {
            "type": "complete",
            "job_id": job.job_id,
            "message": f"{len(scored_claims)} claim(s) processed - {auto_accepted} auto-supported, {pending} need human validation.",
            "data": job.model_dump(),
        }

    def _build_inventory_prompt(self, sources: list[Source], upload_dir: Path) -> str:
        blocks = []
        for source in sources:
            if not can_send_to_model(source):
                blocks.append(f"- [redacted] ({source.source_type.value}) -- BLOCKED by governance ({source.access_level.value}), no excerpt provided")
                continue
            if source.source_type == SourceType.AUDIO:
                blocks.append(f"- {source.filename} (audio, type={source.source_type.value}) -- excerpt not applicable")
                continue
            path = upload_dir / source.filename
            excerpt = ""
            if path.exists():
                excerpt = path.read_text(encoding="utf-8", errors="replace")[:MAX_EXCERPT_CHARS]
            blocks.append(
                f"- source_id={source.source_id} filename={source.filename} type={source.source_type.value}\n"
                f"  excerpt:\n{excerpt}\n"
            )
        return "Archive sources:\n\n" + "\n".join(blocks)

    async def _transcribe_audio(self, source: Source, audio_path: Path) -> TranscriptionOutput:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        mime_type = _AUDIO_MIME_BY_EXT.get(audio_path.suffix.lower(), "audio/webm")
        audio_bytes = audio_path.read_bytes()
        transcription_agent = create_transcription_agent(model_name=self.model_name, api_key=self.api_key)
        prompt_text = (
            f"Audio source: {source.filename}. Transcribe and segment this recording following your instructions."
        )
        parts = [types.Part(text=prompt_text), types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)]
        output = await self._run_agent(
            transcription_agent, parts, TranscriptionOutput, "TranscriptionAgent",
            timeout_seconds=TRANSCRIPTION_CALL_TIMEOUT_SECONDS,
        )
        return output.model_copy(update={"source_id": source.source_id})

    async def _gather_evidence(self, claim_value: str, searchable: list[tuple[str, Path, str]]) -> list[Evidence]:
        snippets = search_all(claim_value, searchable)
        if not snippets:
            return []
        snippet_block = "\n\n".join(
            f"source_id={s.source_id} locator={s.locator}\n{s.snippet}" for s in snippets
        )
        evidence_agent = create_evidence_agent(model_name=self.model_name, api_key=self.api_key)
        prompt = f"Claim value: {claim_value}\n\nCandidate snippets found by search:\n\n{snippet_block}"
        try:
            result: EvidenceJudgmentOutput = await self._run_text_agent(
                evidence_agent, prompt, EvidenceJudgmentOutput, "EvidenceAgent"
            )
        except AgentExecutionError:
            logger.exception("EvidenceAgent failed for claim '%s'; continuing with no evidence", claim_value)
            return []
        return result.evidence

    async def _propose_meaning(self, claim_value: str, evidence: list[Evidence]) -> LinguistProposal | None:
        evidence_block = "\n".join(
            f"- [{e.stance.value}, score={e.support_score:.2f}] {e.source_id} ({e.locator}): {e.snippet}"
            for e in evidence
        ) or "(no evidence found)"
        linguist_agent = create_linguist_agent(model_name=self.model_name, api_key=self.api_key)
        prompt = f"Transcribed form: {claim_value}\n\nJudged evidence:\n{evidence_block}"
        try:
            return await self._run_text_agent(linguist_agent, prompt, LinguistProposal, "LinguistAgent")
        except AgentExecutionError:
            logger.exception("LinguistAgent failed for claim '%s'", claim_value)
            return None

    async def _check_conflict(
        self, claim_value: str, evidence: list[Evidence], linguist_proposal: LinguistProposal | None
    ) -> ConflictCheckOutput:
        evidence_block = "\n".join(
            f"- [{e.stance.value}] {e.source_id}: {e.snippet}" for e in evidence
        ) or "(no evidence found)"
        variants = ", ".join(linguist_proposal.variants) if linguist_proposal and linguist_proposal.variants else "(none noted)"
        conflict_agent = create_conflict_agent(model_name=self.model_name, api_key=self.api_key)
        prompt = f"Claim value: {claim_value}\n\nJudged evidence:\n{evidence_block}\n\nVariants noted by Linguist Agent: {variants}"
        try:
            return await self._run_text_agent(conflict_agent, prompt, ConflictCheckOutput, "ConflictAgent")
        except AgentExecutionError:
            logger.exception("ConflictAgent failed for claim '%s'; assuming no conflict", claim_value)
            return ConflictCheckOutput(has_conflict=False, conflicts=[], resolution_note=None)
