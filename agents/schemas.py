"""
Pydantic contracts for Language Recovery OS.

Same central rule as the sibling hackathon projects (TopicAhead, Trusted
Hire Mexico, ScopeCouncil): no LLM output schema ever includes the final
accept/escalate decision. An agent may transcribe, judge evidence stance,
propose a meaning, or flag a conflict -- it never gets to assert
SUPPORTED/NEEDS_VALIDATION itself. Combined confidence and the resulting
claim status are always computed afterward, in plain Python
(agents/scoring.py), against the fixed thresholds from the project's master
spec (section 274-292): >=0.85 auto-accept, 0.65-0.84 supported hypothesis,
<0.65 human validation required.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums shared across the pipeline
# --------------------------------------------------------------------------


class SourceType(str, Enum):
    AUDIO = "audio"
    DICTIONARY = "dictionary"
    GRAMMAR = "grammar"
    CORPUS = "corpus"
    TEXT = "text"


class AccessLevel(str, Enum):
    PUBLIC = "PUBLIC"
    COMMUNITY_ONLY = "COMMUNITY_ONLY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    RESTRICTED = "RESTRICTED"
    SACRED_DO_NOT_PROCESS = "SACRED_DO_NOT_PROCESS"


class ClaimType(str, Enum):
    TRANSCRIPTION = "transcription"
    LEXEME = "lexeme"


class ClaimStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    EXTRACTED = "EXTRACTED"
    HYPOTHESIS = "HYPOTHESIS"
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    COMMUNITY_VALIDATED = "COMMUNITY_VALIDATED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# --------------------------------------------------------------------------
# Source / archive inventory
# --------------------------------------------------------------------------


class Source(BaseModel):
    source_id: str
    filename: str
    source_type: SourceType
    # Set by the human uploader at upload time, never inferred by an LLM --
    # GovernanceAgent (agents/governance_agent.py) enforces this before any
    # content from a RESTRICTED/SACRED_DO_NOT_PROCESS source is ever sent to
    # a model. Defaults to PUBLIC only because every demo_data source is
    # verified public-domain or CC-licensed (see demo_data/SOURCES.md); a
    # real deployment must never default an unlabeled source to PUBLIC.
    access_level: AccessLevel = AccessLevel.PUBLIC
    citation: str | None = None


class ArchiveInventoryItem(BaseModel):
    source_id: str
    material_kind: str
    summary: str
    estimated_scope: str = Field(description="e.g. '213 pages', '18m 43s', '243 aligned sentence pairs'")


class ArchiveAnalysis(BaseModel):
    archive_summary: str
    inventory: list[ArchiveInventoryItem] = Field(default_factory=list)
    proposed_workflow: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Transcription (Gemini multimodal directly on the audio file -- recorte de
# scope #3 in BITACORA_PROYECTO.md: no dedicated ASR/segmentation pipeline)
# --------------------------------------------------------------------------


class TranscriptionCandidate(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, description="Model's own estimate -- one signal into scoring, never a decision")


class TranscriptionHypothesis(BaseModel):
    segment_label: str = Field(description="e.g. '00:17-00:24'")
    start_seconds: float
    end_seconds: float
    candidates: list[TranscriptionCandidate] = Field(default_factory=list)
    unresolved_region: str | None = None


class TranscriptionOutput(BaseModel):
    # Defaulted: the orchestrator always overwrites this with the real
    # Source.source_id after the call (agents/orchestrator.py::
    # _transcribe_audio), so requiring it here only forced the model to
    # invent a value it has no way to know.
    source_id: str = ""
    hypotheses: list[TranscriptionHypothesis] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Evidence (retrieval itself is plain Python -- agents/retrieval.py,
# recorte de scope #2: no managed RAG Engine. The LLM only judges the
# snippets Python already found.)
# --------------------------------------------------------------------------


class EvidenceStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED = "related"


class Evidence(BaseModel):
    source_id: str
    locator: str = Field(description="e.g. 'corpus:nmlch-pmope1#0014' or 'dictionary:line 842'")
    snippet: str
    stance: EvidenceStance
    support_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class EvidenceJudgmentOutput(BaseModel):
    claim_value: str
    evidence: list[Evidence] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Linguist proposal
# --------------------------------------------------------------------------


class LinguistProposal(BaseModel):
    lemma: str | None = None
    meaning_translation: str
    part_of_speech: str | None = None
    morphemes: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    rationale: str


# --------------------------------------------------------------------------
# Conflict detection
# --------------------------------------------------------------------------


class ConflictNote(BaseModel):
    source_id: str
    alternative_value: str
    likely_cause: str = Field(description="e.g. 'orthographic reform', 'regional variant', 'transcription inconsistency'")


class ConflictCheckOutput(BaseModel):
    has_conflict: bool
    conflicts: list[ConflictNote] = Field(default_factory=list)
    resolution_note: str | None = None


# --------------------------------------------------------------------------
# Computed in Python (agents/scoring.py) -- never asserted by an LLM
# --------------------------------------------------------------------------


class Claim(BaseModel):
    claim_id: str
    job_id: str
    claim_type: ClaimType
    source_id: str
    segment_label: str | None = None
    value: str
    status: ClaimStatus = ClaimStatus.DISCOVERED
    transcription_confidence: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[ConflictNote] = Field(default_factory=list)
    linguist_proposal: LinguistProposal | None = None
    combined_confidence: float | None = None
    recommended_action: str | None = None  # "auto_accept" | "keep_as_hypothesis" | "human_validation"
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)


class Validation(BaseModel):
    claim_id: str
    validator_name: str
    validator_role: str
    decision_value: str
    notes: str | None = None
    timestamp: str


# --------------------------------------------------------------------------
# User input / job
# --------------------------------------------------------------------------


class ProcessArchiveRequest(BaseModel):
    job_name: str = Field(default="Archive Demo", max_length=200)
    api_key: str | None = None


class Job(BaseModel):
    job_id: str
    name: str
    status: JobStatus = JobStatus.QUEUED
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    # Append-only audit trail of every human validation applied to this
    # job's claims -- who, what value, when. Kept on the Job (not the Claim)
    # so the record survives even if a claim is later re-scored, and so the
    # provenance guarantee covers the human step too, not just the agents'.
    validations: list[Validation] = Field(default_factory=list)
    archive_analysis: ArchiveAnalysis | None = None
    created_at: str
