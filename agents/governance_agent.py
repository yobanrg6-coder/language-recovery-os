"""
GovernanceAgent -- deliberately NOT an LlmAgent.

Section 7.11 of the master spec describes it as applying access rules
(PUBLIC / COMMUNITY_ONLY / RESEARCH_ONLY / RESTRICTED / SACRED_DO_NOT_PROCESS)
and blocking restricted material from being sent to an external model or
published. That is an enforcement decision, and the project's one golden
rule (agents/schemas.py module docstring, inherited from ScopeCouncil and
Trusted Hire Mexico) is that no LLM ever gets to make the final call on
anything that gates what happens next. access_level is set by the human who
uploads a source (agents/schemas.py::Source.access_level), not inferred by
a model -- this module only ever enforces what a human already declared.
"""

from __future__ import annotations

from agents.schemas import AccessLevel, Source

_BLOCKED_FROM_MODEL_CALLS = {AccessLevel.RESTRICTED, AccessLevel.SACRED_DO_NOT_PROCESS}
_BLOCKED_FROM_PUBLICATION = {
    AccessLevel.RESTRICTED,
    AccessLevel.SACRED_DO_NOT_PROCESS,
    AccessLevel.COMMUNITY_ONLY,
    AccessLevel.RESEARCH_ONLY,
}


class GovernanceBlockedError(RuntimeError):
    """Raised when the pipeline tries to send a restricted source's content to a model."""


def can_send_to_model(source: Source) -> bool:
    return source.access_level not in _BLOCKED_FROM_MODEL_CALLS


def require_sendable(source: Source) -> None:
    if not can_send_to_model(source):
        raise GovernanceBlockedError(
            f"Source '{source.source_id}' is marked {source.access_level.value} -- "
            "governance rules block sending its content to any model, including for retrieval or judging."
        )


def can_publish(source: Source) -> bool:
    """PUBLIC-only, on purpose: publication is a stricter bar than model access.
    A COMMUNITY_ONLY or RESEARCH_ONLY source can still be processed to build
    the internal knowledge graph, but nothing derived from it is exported/
    published automatically (master spec section 10: 'no publicación
    automática')."""
    return source.access_level not in _BLOCKED_FROM_PUBLICATION
