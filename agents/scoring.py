"""
Deterministic confidence engine -- the actual answer to "why not just trust
the model's own confidence number". Combines transcription confidence,
judged-evidence support, and cross-source agreement into a single combined
confidence, applies the fixed thresholds from the master spec (section
274-292: >=0.85 auto-accept, 0.65-0.84 supported hypothesis, <0.65 human
validation), and enforces four hard rules an LLM is never trusted to apply to itself:

  1. A claim with zero supporting evidence can never auto-accept, no matter
     how confident the transcription step was (master spec principle #1,
     section 32: "Evidence before confidence"; also section 35's
     test_claim_without_evidence_cannot_auto_accept).
  2. A claim with an unresolved conflict (ConflictAgent found one) always
     routes to human validation, regardless of score (section 7.7).
  3. Auto-accept requires supporting evidence from at least two distinct
     sources -- one agent's confident "supports" on one snippet is never
     enough by itself; below two sources the best a claim can be is a
     hypothesis.
  4. If the Evidence Agent judged any snippet as *contradicting* the claim,
     auto-accept is blocked even when the Conflict Agent raised nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.schemas import Claim, ClaimStatus, ConflictCheckOutput, ConflictNote, Evidence, EvidenceStance

ACTION_AUTO_ACCEPT = "auto_accept"
ACTION_KEEP_HYPOTHESIS = "keep_as_hypothesis"
ACTION_HUMAN_VALIDATION = "human_validation"

AUTO_ACCEPT_THRESHOLD = 0.85
SUPPORTED_HYPOTHESIS_THRESHOLD = 0.65

# Confidence a claim receives once a human validator has confirmed it --
# fixed, not recomputed from the original (possibly weak) evidence, matching
# the master spec's own worked example (section 20, step 7: "Confidence:
# 96%, Community verified").
COMMUNITY_VALIDATED_CONFIDENCE = 0.96


@dataclass(frozen=True)
class ConfidenceWeights:
    transcription: float = 0.35
    evidence_support: float = 0.35
    cross_source_agreement: float = 0.30
    conflict_penalty: float = 0.20


def _evidence_support_avg(evidence: list[Evidence]) -> float:
    supporting = [e.support_score for e in evidence if e.stance == EvidenceStance.SUPPORTS]
    return sum(supporting) / len(supporting) if supporting else 0.0


def _cross_source_agreement(evidence: list[Evidence]) -> float:
    supporting_sources = {e.source_id for e in evidence if e.stance == EvidenceStance.SUPPORTS}
    # 3 independent sources agreeing is treated as full agreement -- matches
    # the master spec's own demo example (section 9: dictionary + grammar +
    # corpus + speaker recording all lining up).
    return min(1.0, len(supporting_sources) / 3.0)


def score_claim(
    claim: Claim,
    transcription_confidence: float | None,
    evidence: list[Evidence],
    conflicts: list[ConflictNote],
    weights: ConfidenceWeights | None = None,
    has_conflict: bool = False,
) -> Claim:
    weights = weights or ConfidenceWeights()

    transcription = transcription_confidence if transcription_confidence is not None else 0.0
    evidence_support = _evidence_support_avg(evidence)
    cross_source = _cross_source_agreement(evidence)
    # ConflictCheckOutput.has_conflict is a separate LLM-set bool from its
    # own `conflicts` list (agents/schemas.py has no validator forcing them
    # to agree) - trusting only len(conflicts) meant a model returning
    # has_conflict=True with an empty conflicts list slipped straight past
    # Rule 2 ("an unresolved conflict always routes to human validation").
    # Found 2026-08-22; ORing both is the conservative fix - either signal
    # saying "conflict" is enough to block auto-accept.
    has_conflict = has_conflict or len(conflicts) > 0

    combined = (
        weights.transcription * transcription
        + weights.evidence_support * evidence_support
        + weights.cross_source_agreement * cross_source
    )
    if has_conflict:
        combined -= weights.conflict_penalty
    combined = max(0.0, min(1.0, combined))

    has_any_evidence = len(evidence) > 0
    has_supporting_evidence = evidence_support > 0.0
    supporting_source_count = len({e.source_id for e in evidence if e.stance == EvidenceStance.SUPPORTS})
    has_contradicting_evidence = any(e.stance == EvidenceStance.CONTRADICTS for e in evidence)

    # Rule 3: auto-accept requires corroboration from at least TWO distinct
    # sources. One agent judging one snippet as "strongly supports", however
    # confident, is never enough on its own -- the accept decision is
    # deterministic, but its inputs are still LLM judgments, so the floor is
    # multi-source agreement, not a single high number. (The weighted score
    # already trends this way; making it an explicit floor keeps the
    # guarantee even if the weights are ever retuned.)
    # Rule 4: any snippet the Evidence Agent itself judged as *contradicting*
    # the claim blocks auto-accept regardless of score -- it drops to a
    # hypothesis a human weighs, even when the Conflict Agent raised nothing.
    can_auto_accept = (
        combined >= AUTO_ACCEPT_THRESHOLD
        and supporting_source_count >= 2
        and not has_contradicting_evidence
    )

    if has_conflict:
        status = ClaimStatus.CONFLICTED
        action = ACTION_HUMAN_VALIDATION
    elif not has_any_evidence or not has_supporting_evidence:
        # Rule 1: no evidence, no auto-accept -- forced to human review even
        # if transcription confidence alone would have cleared 0.85.
        status = ClaimStatus.NEEDS_VALIDATION
        action = ACTION_HUMAN_VALIDATION
    elif can_auto_accept:
        status = ClaimStatus.SUPPORTED
        action = ACTION_AUTO_ACCEPT
    elif combined >= SUPPORTED_HYPOTHESIS_THRESHOLD:
        status = ClaimStatus.HYPOTHESIS
        action = ACTION_KEEP_HYPOTHESIS
    else:
        status = ClaimStatus.NEEDS_VALIDATION
        action = ACTION_HUMAN_VALIDATION

    return claim.model_copy(
        update={
            "status": status,
            "combined_confidence": round(combined, 4),
            "recommended_action": action,
            "evidence": evidence,
            "conflicts": conflicts,
            "transcription_confidence": transcription_confidence,
            "confidence_breakdown": {
                "transcription": round(transcription, 4),
                "evidence_support": round(evidence_support, 4),
                "cross_source_agreement": round(cross_source, 4),
                "conflict_penalty_applied": weights.conflict_penalty if has_conflict else 0.0,
                "supporting_sources": float(supporting_source_count),
                "contradicting_evidence": 1.0 if has_contradicting_evidence else 0.0,
            },
        }
    )


def merge_conflict_checks(
    primary: ConflictCheckOutput, secondary: "ConflictCheckOutput | None"
) -> ConflictCheckOutput:
    """
    Combines the Conflict Agent's (Gemini) and Gemma Conflict Agent's (Gemma)
    independent reads of the same judged evidence into one attributed
    result -- same "union, not intersection" principle as Trusted Hire
    Mexico's merge_scam_flags: a conflict either model raises is kept, never
    silently dropped, because a false negative here is worse than an extra
    claim routed to human validation. has_conflict is the OR of both models'
    own flags and of whether either produced any ConflictNote, matching the
    same defensive OR already applied in score_claim() against a single
    model disagreeing with its own conflicts list.

    Falls back to the primary's read alone when Gemma is unavailable (see
    orchestrator.py's degrade-safe Gemma call).
    """
    if secondary is None:
        return primary

    merged_conflicts: list[ConflictNote] = list(primary.conflicts)
    seen = {(c.source_id, c.alternative_value) for c in merged_conflicts}
    for note in secondary.conflicts:
        key = (note.source_id, note.alternative_value)
        if key not in seen:
            merged_conflicts.append(note)
            seen.add(key)

    has_conflict = primary.has_conflict or secondary.has_conflict or bool(merged_conflicts)

    resolution_note = primary.resolution_note or secondary.resolution_note
    if primary.resolution_note and secondary.resolution_note and primary.resolution_note != secondary.resolution_note:
        resolution_note = f"{primary.resolution_note} (Gemini) / {secondary.resolution_note} (Gemma)"

    return ConflictCheckOutput(has_conflict=has_conflict, conflicts=merged_conflicts, resolution_note=resolution_note)


def apply_community_validation(claim: Claim) -> Claim:
    """Called once a human validator has picked/confirmed a value for a claim
    that was routed to NEEDS_VALIDATION or CONFLICTED (agents/scoring.py is
    the only place that ever moves a claim into COMMUNITY_VALIDATED --
    web_app/app.py's validation endpoint calls this, it never sets the
    status field itself)."""
    return claim.model_copy(
        update={
            "status": ClaimStatus.COMMUNITY_VALIDATED,
            "combined_confidence": COMMUNITY_VALIDATED_CONFIDENCE,
            "recommended_action": "community_validated",
            "conflicts": [],
        }
    )
