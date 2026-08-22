"""
Tests for the deterministic confidence engine (agents/scoring.py) -- the
"tests clave" named in the master spec section 35, adapted to this project's
actual schema. These need no API key: scoring.py takes already-produced
Evidence/ConflictNote objects, never calling a model itself.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.schemas import Claim, ClaimStatus, ClaimType, ConflictNote, Evidence, EvidenceStance
from agents.scoring import (
    ACTION_AUTO_ACCEPT,
    ACTION_HUMAN_VALIDATION,
    ACTION_KEEP_HYPOTHESIS,
    apply_community_validation,
    score_claim,
)


def _base_claim() -> Claim:
    return Claim(
        claim_id="c1", job_id="j1", claim_type=ClaimType.TRANSCRIPTION,
        source_id="audio_1", segment_label="00:00-00:05", value="mari",
    )


def test_claim_without_evidence_cannot_auto_accept():
    claim = _base_claim()
    scored = score_claim(claim, transcription_confidence=0.99, evidence=[], conflicts=[])
    assert scored.status == ClaimStatus.NEEDS_VALIDATION
    assert scored.recommended_action == ACTION_HUMAN_VALIDATION


def test_high_confidence_with_strong_evidence_auto_accepts():
    claim = _base_claim()
    evidence = [
        Evidence(source_id="dict_1", locator="L1", snippet="mari - diez", stance=EvidenceStance.SUPPORTS, support_score=0.95, rationale="exact match"),
        Evidence(source_id="grammar_1", locator="L2", snippet="mari used as numeral", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="grammar-consistent"),
        Evidence(source_id="corpus_1", locator="L3", snippet="mari kechu", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="corpus occurrence"),
    ]
    scored = score_claim(claim, transcription_confidence=0.95, evidence=evidence, conflicts=[])
    assert scored.status == ClaimStatus.SUPPORTED
    assert scored.recommended_action == ACTION_AUTO_ACCEPT
    assert scored.combined_confidence >= 0.85


def test_moderate_evidence_keeps_hypothesis_not_auto_accept():
    # A single supporting source caps cross_source_agreement at 1/3 (see
    # scoring.py::_cross_source_agreement), so both remaining signals need to
    # be fairly strong to still clear the 0.65 hypothesis floor -- 0.85/0.85
    # lands at 0.695, deliberately below the 0.85 auto-accept bar.
    claim = _base_claim()
    evidence = [
        Evidence(source_id="dict_1", locator="L1", snippet="mari - diez", stance=EvidenceStance.SUPPORTS, support_score=0.85, rationale="plausible match"),
    ]
    scored = score_claim(claim, transcription_confidence=0.85, evidence=evidence, conflicts=[])
    assert scored.status == ClaimStatus.HYPOTHESIS
    assert scored.recommended_action == ACTION_KEEP_HYPOTHESIS


def test_low_confidence_routes_to_human():
    claim = _base_claim()
    evidence = [
        Evidence(source_id="dict_1", locator="L1", snippet="weak match", stance=EvidenceStance.SUPPORTS, support_score=0.2, rationale="weak"),
    ]
    scored = score_claim(claim, transcription_confidence=0.3, evidence=evidence, conflicts=[])
    assert scored.status == ClaimStatus.NEEDS_VALIDATION
    assert scored.recommended_action == ACTION_HUMAN_VALIDATION


def test_conflict_forces_human_validation_even_with_high_score():
    claim = _base_claim()
    evidence = [
        Evidence(source_id="dict_1", locator="L1", snippet="mari - diez", stance=EvidenceStance.SUPPORTS, support_score=0.95, rationale="exact match"),
        Evidence(source_id="grammar_1", locator="L2", snippet="mari used as numeral", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="grammar-consistent"),
        Evidence(source_id="corpus_1", locator="L3", snippet="mari kechu", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="corpus occurrence"),
    ]
    conflicts = [ConflictNote(source_id="corpus_2019", alternative_value="mary", likely_cause="orthographic reform")]
    scored = score_claim(claim, transcription_confidence=0.95, evidence=evidence, conflicts=conflicts)
    assert scored.status == ClaimStatus.CONFLICTED
    assert scored.recommended_action == ACTION_HUMAN_VALIDATION


def test_has_conflict_flag_alone_forces_human_validation():
    # ConflictAgent's has_conflict bool and its conflicts list are two
    # separate LLM-set fields with no schema validator forcing them to
    # agree (agents/schemas.py::ConflictCheckOutput) - a model returning
    # has_conflict=True with an empty conflicts list must still block
    # auto-accept, not just len(conflicts) > 0. Regression test for the fix
    # found 2026-08-22 auditing agents/scoring.py.
    claim = _base_claim()
    evidence = [
        Evidence(source_id="dict_1", locator="L1", snippet="mari - diez", stance=EvidenceStance.SUPPORTS, support_score=0.95, rationale="exact match"),
        Evidence(source_id="grammar_1", locator="L2", snippet="mari used as numeral", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="grammar-consistent"),
        Evidence(source_id="corpus_1", locator="L3", snippet="mari kechu", stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="corpus occurrence"),
    ]
    scored = score_claim(claim, transcription_confidence=0.95, evidence=evidence, conflicts=[], has_conflict=True)
    assert scored.status == ClaimStatus.CONFLICTED
    assert scored.recommended_action == ACTION_HUMAN_VALIDATION


def test_validated_claim_updates_graph():
    claim = score_claim(_base_claim(), transcription_confidence=0.3, evidence=[], conflicts=[])
    assert claim.status == ClaimStatus.NEEDS_VALIDATION

    validated = apply_community_validation(claim.model_copy(update={"value": "mari"}))
    assert validated.status == ClaimStatus.COMMUNITY_VALIDATED
    assert validated.combined_confidence == 0.96
    assert validated.recommended_action == "community_validated"
