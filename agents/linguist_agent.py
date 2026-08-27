"""
LinguistAgent -- proposes lemma, part of speech, morphemes, variants and a
provisional translation for a transcription claim, always grounded in the
evidence EvidenceAgent already judged. Its output is a hypothesis like
everything else upstream of agents/scoring.py: it never marks a claim
accepted, published, or validated.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.schemas import LinguistProposal

SYSTEM_INSTRUCTION = """You are the Linguist Agent in Language Recovery OS. You are given a transcribed
form (a candidate word or short phrase in a low-resource language), plus the evidence snippets the
Evidence Agent already judged as supporting/contradicting/related, drawn from a dictionary, grammar
and/or corpus for the same language.

Instructions:
1. Propose `meaning_translation`: your best provisional translation/gloss, grounded specifically in the
   supporting evidence you were given -- if the evidence is thin, say so plainly in `rationale` rather
   than inventing confident meaning from general language knowledge.
2. If the evidence or your linguistic judgment supports it, propose `lemma` (dictionary citation form),
   `part_of_speech`, and any `morphemes` you can identify (e.g. root + suffixes) -- leave these null/
   empty rather than guessing when you don't have real support.
3. List `variants`: alternate spellings/forms you noticed across the evidence snippets (this feeds the
   Conflict Agent -- do not resolve or hide a variant, report it).
4. Write `rationale`: 1-3 sentences citing which evidence specifically backs your proposal.
5. Output strictly conforming to the provided schema. Do not assign a confidence score or accept/reject
   status -- that is computed elsewhere, never by you.
"""


def create_linguist_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("MODEL", "gemini-3.5-flash-lite")
    gemini_kwargs: dict = {"model": model}
    if api_key:
        gemini_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="linguist_agent",
        description="Proposes lemma, part of speech, morphemes and provisional meaning for a transcribed form, grounded in judged evidence.",
        model=Gemini(**gemini_kwargs),
        instruction=SYSTEM_INSTRUCTION,
        output_schema=LinguistProposal,
    )
