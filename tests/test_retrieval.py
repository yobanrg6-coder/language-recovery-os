"""
Tests for agents/retrieval.py's two judge-audit fixes (2026-08-22):
1. An empty/whitespace query must never match every line in every source.
2. Evidence referencing a snippet that was never actually retrieved must be
   dropped, not trusted verbatim from the LLM's structured output.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.retrieval import RetrievedSnippet, filter_verified_evidence, search_all
from agents.schemas import Evidence, EvidenceStance


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_empty_query_returns_no_results(tmp_path):
    source = _write(tmp_path, "dict.txt", "river (noun): a large flowing body of water.\nvillage (noun): a small settlement.\n")
    results = search_all("", [("src_1", source, "src_1")])
    assert results == []
    results_whitespace = search_all("   ", [("src_1", source, "src_1")])
    assert results_whitespace == []


def test_nonempty_query_still_finds_matches(tmp_path):
    source = _write(tmp_path, "dict.txt", "river (noun): a large flowing body of water.\nvillage (noun): a small settlement.\n")
    results = search_all("river", [("src_1", source, "src_1")])
    assert len(results) == 1
    assert "river" in results[0].snippet.lower()


def test_phrase_query_matches_despite_trailing_punctuation(tmp_path):
    """2026-08-26 audit B1: a transcription like 'the river flows quietly.'
    (trailing period, no closing words) must still hit the corpus line even
    though it is not an exact substring of it."""
    source = _write(tmp_path, "corpus.txt", "elder says the river flows quietly through the old stone village\n")
    hits = search_all("the river flows quietly.", [("src_1", source, "src_1")])
    assert len(hits) == 1

    # a phrase with no real overlap still does not match
    assert search_all("the cat sat on the mat", [("src_1", source, "src_1")]) == []


def test_filter_verified_evidence_keeps_only_real_snippets():
    snippets = [
        RetrievedSnippet(source_id="src_1", locator="src_1#L1", snippet="river flows to the sea"),
    ]
    real = Evidence(
        source_id="src_1", locator="src_1#L1", snippet="river flows to the sea",
        stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="exact match",
    )
    hallucinated = Evidence(
        source_id="src_1", locator="src_1#L99", snippet="a locator that was never retrieved",
        stance=EvidenceStance.SUPPORTS, support_score=0.95, rationale="fabricated",
    )
    filtered = filter_verified_evidence([real, hallucinated], snippets)
    assert filtered == [real]


def test_filter_verified_evidence_empty_snippets_drops_everything():
    hallucinated = Evidence(
        source_id="src_1", locator="src_1#L1", snippet="claims a source that was never searched",
        stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="fabricated",
    )
    assert filter_verified_evidence([hallucinated], []) == []


def test_filter_verified_evidence_tolerates_reformatted_locator():
    """2026-08-26 audit M5: a model that copies the snippet faithfully but
    reformats the locator must not have its evidence silently discarded and
    reported as 'no evidence found'."""
    snippets = [
        RetrievedSnippet(source_id="src_1", locator="src_1#L14", snippet="ko means water in this passage"),
    ]
    reformatted = Evidence(
        source_id="src_1", locator="src_1 : line 14", snippet="ko means water in this passage",
        stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="same snippet, different locator style",
    )
    assert filter_verified_evidence([reformatted], snippets) == [reformatted]


def test_filter_verified_evidence_still_rejects_wrong_source():
    """The source_id boundary is not relaxed: evidence citing a source that
    was never retrieved (e.g. a governance-blocked one) is always dropped,
    even if some other source's snippet text happens to overlap."""
    snippets = [
        RetrievedSnippet(source_id="src_public", locator="src_public#L1", snippet="ko means water in this passage"),
    ]
    wrong_source = Evidence(
        source_id="src_sacred", locator="src_sacred#L1", snippet="ko means water in this passage",
        stance=EvidenceStance.SUPPORTS, support_score=0.9, rationale="cites a source that was never searched",
    )
    assert filter_verified_evidence([wrong_source], snippets) == []
