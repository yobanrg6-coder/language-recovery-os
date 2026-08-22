"""
ArchiveAgent -- two responsibilities, one file, per BITACORA_PROYECTO.md's
recorte de scope #3 ("ASR/Speech-Segmentation Agent dedicado -> Gemini
multimodal directo"): the master spec's separate Archive Agent (7.2) and
Transcription/Segmentation Agent (7.3/7.4) collapse into one module because
both are "read a raw source and describe what's in it" -- the only
difference is that one reads text/PDF sources and the other reads audio
directly with Gemini's multimodal input, in a single prompt, instead of a
dedicated ASR pipeline (which the sibling projects' timeline does not allow
building in days, per the master spec's own section 22 caveat).

Neither function decides what to do with the material -- EvidenceAgent,
LinguistAgent, ConflictAgent and the Python confidence engine do that.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.schemas import ArchiveAnalysis, TranscriptionOutput

INVENTORY_INSTRUCTION = """You are the Archive Agent in Language Recovery OS. A community, linguist or
archive has handed you a set of language documentation materials (dictionaries, grammars, aligned
corpora, field notes) and wants them turned into a workflow plan. You do NOT transcribe audio here --
that is a separate call. You do NOT propose word meanings or accept/reject anything.

Instructions:
1. For each source you are given (filename, type, and a short text excerpt), classify its
   `material_kind` (e.g. "bilingual dictionary", "grammar reference", "aligned oral-history corpus",
   "field recording") and write a one or two sentence `summary` of what it actually contains, grounded
   in the excerpt you were given -- do not invent content beyond what the excerpt supports.
2. Give a rough `estimated_scope` per source (e.g. "213 pages", "243 aligned sentence pairs") based on
   whatever size signal you were given (line/page counts) -- state your best estimate, do not fabricate
   precision you don't have.
3. Write `archive_summary`: two or three sentences describing the archive as a whole -- what kind of
   materials, what language documentation this appears to represent.
4. Propose `proposed_workflow`: an ordered list of 4-8 short steps this system would take next (e.g.
   "Build document index", "Generate transcription hypotheses for the audio source", "Cross-check
   lexical matches against the dictionary", "Route uncertain segments for validation"). This is a plan
   description, not something you execute.
5. Output strictly conforming to the provided schema. Never propose BUILD/ACCEPT/REJECT decisions --
   that is not part of your schema and not your job.
"""

TRANSCRIPTION_INSTRUCTION = """You are the Transcription stage of the Archive Agent in Language Recovery
OS, working on a single audio recording in a low-resource, under-documented language. You are given the
audio directly (multimodal input) plus whatever textual context (dictionary/grammar excerpts) you were
handed for grounding.

Instructions:
1. Break the audio into a small number of speech segments (do not over-segment -- a handful of
   meaningful chunks, not one per word). For each segment give `segment_label` as "MM:SS-MM:SS",
   `start_seconds` and `end_seconds`.
2. For each segment, propose 1-3 ranked `candidates`: your best-guess transcription `text` in the
   source language as you hear it, each with your own honest `confidence` (0.0-1.0) reflecting how
   sure you actually are -- do not default everything to a high number. If you are only confident about
   part of a segment, still give your best full-segment transcription but keep the confidence low and
   describe the uncertain part in `unresolved_region` (e.g. "token 3" or a short description).
3. This confidence is ONE signal that Python combines with independent evidence lookup later -- it is
   never the final word. Be honest rather than reassuring.
4. Output strictly conforming to the provided schema. Do not translate or explain the meaning of what
   you transcribe -- that is the Linguist Agent's job, not yours.
"""


def create_archive_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("MODEL", "gemini-flash-lite-latest")
    gemini_kwargs: dict = {"model": model}
    if api_key:
        gemini_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="archive_agent",
        description="Inventories and classifies the uploaded archive materials and proposes a workflow plan.",
        model=Gemini(**gemini_kwargs),
        instruction=INVENTORY_INSTRUCTION,
        output_schema=ArchiveAnalysis,
    )


def create_transcription_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("MODEL", "gemini-flash-lite-latest")
    gemini_kwargs: dict = {"model": model}
    if api_key:
        gemini_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="transcription_agent",
        description="Reads an audio recording directly (Gemini multimodal input) and proposes segmented, "
        "ranked transcription hypotheses with self-reported confidence.",
        model=Gemini(**gemini_kwargs),
        instruction=TRANSCRIPTION_INSTRUCTION,
        output_schema=TranscriptionOutput,
    )
