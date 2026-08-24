"""
GemmaConflictAgent -- an independent second opinion on the same judged
evidence and variant forms, using a different model family entirely (Gemma,
not Gemini).

Same discipline as the sibling hackathon projects (Trusted Hire Mexico's
GemmaVerifierAgent, ScopeCouncil's GemmaScopeAgent): no single point of
failure gets to decide whether two sources genuinely contradict each other.
The Conflict Agent (Gemini) already does this check -- this agent asks the
identical question of a genuinely different model, so a real conflict
Gemini's reading missed still has a real chance of being caught
(agents/scoring.py::merge_conflict_checks). This matters directly for
this project's central safety rule (section 7.7 of the master spec): a
claim with an unresolved conflict is always routed to human validation,
regardless of how high its computed confidence would otherwise be -- a
missed conflict here is a missed human-review routing, not just a cosmetic
gap.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.conflict_agent import SYSTEM_INSTRUCTION
from agents.schemas import ConflictCheckOutput

DEFAULT_GEMMA_MODEL = "gemma-4-26b-a4b-it"


def create_gemma_conflict_agent(model_name: str | None = None, api_key: str | None = None) -> LlmAgent:
    model = model_name or os.getenv("GEMMA_MODEL", DEFAULT_GEMMA_MODEL)
    gemma_kwargs: dict = {"model": model}
    if api_key:
        gemma_kwargs["client_kwargs"] = {"api_key": api_key}
    return LlmAgent(
        name="gemma_conflict_agent",
        description="Independently re-checks the same judged evidence and variant forms for genuine "
        "contradictions, on a different model family than the Conflict Agent.",
        model=Gemini(**gemma_kwargs),
        # Deliberately the exact same instruction as the Conflict Agent -
        # this has to be a genuine independent replication of the same
        # question, not a differently-tuned agent, or agreement/disagreement
        # between the two would not mean anything.
        instruction=SYSTEM_INSTRUCTION,
        output_schema=ConflictCheckOutput,
    )
