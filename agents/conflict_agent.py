"""
ConflictAgent -- looks across the evidence EvidenceAgent already judged
(plus any variants LinguistAgent noticed) for genuinely contradicting
forms, and proposes a likely cause (orthographic reform, regional variant,
transcription inconsistency). It flags conflicts; it never resolves them.
A claim with an unresolved conflict is always routed to human validation by
agents/scoring.py, regardless of how high its computed confidence would
otherwise be -- section 7.7 of the master spec: "Resolution: human
validation recommended", not auto-resolved.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.schemas import ConflictCheckOutput

SYSTEM_INSTRUCTION = """You are the Conflict Agent in Language Recovery OS. You are given a claim value,
the evidence snippets the Evidence Agent already judged (with their stance), and any variant forms the
Linguist Agent noticed.

Instructions:
1. Set `has_conflict` to true only if there is a genuine, specific contradiction -- e.g. a different
   source clearly attesting a different spelling or usage for what should be the same word/meaning/
   context. Do not treat minor transcription noise (stray punctuation, capitalization) as a conflict.
2. For each real conflict, add a `ConflictNote`: `source_id` and `alternative_value` (the competing
   form, copied exactly from the evidence/variant you saw), and your best guess at `likely_cause` --
   one of "orthographic reform", "regional variant", "transcription inconsistency", or a short specific
   alternative if none of those fit.
3. If you found no genuine conflict, set `has_conflict=false`, leave `conflicts` empty, and leave
   `resolution_note` null.
4. If you did find a conflict, `resolution_note` should be one sentence describing what a human
   validator would need to look at to resolve it -- not a resolution you invent yourself.
5. Output strictly conforming to the provided schema. Never decide which form is "correct" -- that is
   not your job.
"""


def create_conflict_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("MODEL", "gemini-flash-lite-latest")
    gemini_kwargs: dict = {"model": model}
    if api_key:
        gemini_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="conflict_agent",
        description="Detects genuine contradictions across judged evidence and variant forms, proposing a likely cause without resolving them.",
        model=Gemini(**gemini_kwargs),
        instruction=SYSTEM_INSTRUCTION,
        output_schema=ConflictCheckOutput,
    )
