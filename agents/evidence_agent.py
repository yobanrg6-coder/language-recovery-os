"""
EvidenceAgent -- judges snippets that agents/retrieval.py already found in
plain Python (recorte de scope #2: no managed RAG Engine). It never
searches on its own and never decides whether a claim is accepted; it only
labels each candidate snippet as supporting, contradicting or merely
related, with a support_score and a short rationale. Python
(agents/scoring.py) is the only place a claim's final confidence and status
get decided, same discipline as the sibling projects' scoring engines.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.schemas import EvidenceJudgmentOutput

SYSTEM_INSTRUCTION = """You are the Evidence Agent in Language Recovery OS. You are given a candidate
claim value (e.g. a proposed transcription of an audio segment, or a proposed word form) and a list of
candidate snippets that a plain-code search already retrieved from the archive's dictionary, grammar
and corpus sources. You did not search for these yourself -- your only job is to judge each one.

Instructions:
1. For every snippet you were given, decide its `stance` relative to the claim value:
   - "supports": the snippet contains this form (or a clear match) used in a way consistent with the
     claim.
   - "contradicts": the snippet shows a genuinely different, conflicting form or usage for what should
     be the same word/segment (e.g. a different spelling attested for the same meaning/context).
   - "related": relevant context but neither clearly supporting nor contradicting.
2. Give each snippet an honest `support_score` (0.0-1.0): how strongly this specific snippet backs the
   claim value. A loose/partial match should score low even if you mark it "supports" -- do not inflate
   scores to make the evidence look stronger than it is.
3. Write a one-sentence `rationale` per snippet explaining what in the snippet drove your stance/score.
4. Copy `source_id` and `locator` from the snippet you were given exactly as provided -- do not
   renumber or invent locators.
5. If a snippet is not actually relevant at all, omit it from your output rather than forcing a stance.
6. Output strictly conforming to the provided schema. Never state whether the claim overall should be
   accepted -- that is not part of your schema and is computed elsewhere.
"""


def create_evidence_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("MODEL", "gemini-flash-lite-latest")
    gemini_kwargs: dict = {"model": model}
    if api_key:
        gemini_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="evidence_agent",
        description="Judges pre-retrieved snippets for stance (supports/contradicts/related) and support strength.",
        model=Gemini(**gemini_kwargs),
        instruction=SYSTEM_INSTRUCTION,
        output_schema=EvidenceJudgmentOutput,
    )
