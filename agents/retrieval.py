"""
Plain-Python retrieval over the demo archive -- recorte de scope #2 in
BITACORA_PROYECTO.md: "RAG Engine gestionado -> retrieval simple en
codigo. Con 3-4 documentos no hace falta indexacion real de Vertex RAG
Engine." No embeddings, no vector store: this just loads each non-audio
source's text into memory once per job and does a bounded substring/
fuzzy search for candidate snippets around a query term. EvidenceAgent
(agents/evidence_agent.py) is the one that judges whether a candidate
snippet actually supports or contradicts a claim -- this module only
finds candidates, it never judges them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from agents.schemas import Evidence

MAX_SNIPPETS_PER_SOURCE = 4
CONTEXT_LINES = 1
_FUZZY_CUTOFF = 0.6
_TOKEN_FUZZY_CUTOFF = 0.8
_TOKEN_OVERLAP_CUTOFF = 0.6
_MIN_CONTENT_TOKEN_LEN = 3


@dataclass(frozen=True)
class RetrievedSnippet:
    source_id: str
    locator: str
    snippet: str


def _load_lines(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def _depunct(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _line_matches(line: str, query: str) -> bool:
    lowered = line.lower()
    q = query.lower()
    if q in lowered:
        return True

    # Punctuation-insensitive substring: a transcription that comes back as
    # "...village." should still match the corpus line "...village at dawn"
    # (found 2026-08-26 - a single trailing period was breaking the exact
    # substring match and reporting "no evidence").
    norm_q = _depunct(query)
    if norm_q and norm_q in _depunct(line):
        return True

    line_tokens = [t for t in re.split(r"\W+", lowered) if t]
    query_tokens = [t for t in re.split(r"\W+", q) if len(t) >= _MIN_CONTENT_TOKEN_LEN]

    if len(query_tokens) >= 2:
        # Multi-word query: match when enough of its content words appear in
        # the line. The old per-token fuzzy compared each single line token
        # against the WHOLE phrase, so its ratio was always low and a phrase
        # query effectively only ever matched by exact substring.
        hits = sum(
            1
            for qt in query_tokens
            if any(
                qt == lt or difflib.SequenceMatcher(None, lt, qt).ratio() >= _TOKEN_FUZZY_CUTOFF
                for lt in line_tokens
            )
        )
        return hits / len(query_tokens) >= _TOKEN_OVERLAP_CUTOFF

    # Single-word query: unchanged fuzzy-per-token behaviour.
    return any(difflib.SequenceMatcher(None, t, q).ratio() >= _FUZZY_CUTOFF for t in line_tokens)


def search_source(source_id: str, file_path: Path, query: str, locator_prefix: str) -> list[RetrievedSnippet]:
    """Bounded line-window search: query is a candidate transcription value or
    lexeme form. Returns at most MAX_SNIPPETS_PER_SOURCE hits, each with a
    couple of context lines so the judging agent can see alignment (e.g. the
    Mapudungun line and the Spanish translation line right after it in the
    corpus)."""
    if not file_path.exists():
        return []
    lines = _load_lines(file_path)
    hits: list[RetrievedSnippet] = []
    for i, line in enumerate(lines):
        if not line.strip() or not _line_matches(line, query):
            continue
        start = max(0, i - CONTEXT_LINES)
        end = min(len(lines), i + CONTEXT_LINES + 2)
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        hits.append(RetrievedSnippet(source_id=source_id, locator=f"{locator_prefix}#L{i + 1}", snippet=snippet))
        if len(hits) >= MAX_SNIPPETS_PER_SOURCE:
            break
    return hits


def search_all(query: str, searchable_sources: list[tuple[str, Path, str]]) -> list[RetrievedSnippet]:
    """searchable_sources: list of (source_id, file_path, locator_prefix) for
    every non-audio, governance-cleared source in the job. Sequential and
    unindexed on purpose -- at demo scale (3-4 documents) this is a few
    hundred lines per source, not a corpus that needs a real index."""
    if not query.strip():
        # TranscriptionCandidate.text has no min_length (agents/schemas.py),
        # so a segment Gemini couldn't transcribe (silence, noise, or a
        # language/audio it can't parse) can legally produce an empty claim
        # value. Without this guard, `"" in line` is True for every
        # non-blank line in every source (found 2026-08-22 auditing a
        # non-Mapudungun-shaped-audio scenario), flooding an empty claim
        # with irrelevant "supporting" snippets instead of correctly
        # reporting zero evidence.
        return []
    results: list[RetrievedSnippet] = []
    for source_id, file_path, locator_prefix in searchable_sources:
        results.extend(search_source(source_id, file_path, query, locator_prefix))
    return results


def _normalize_locator(locator: str) -> str:
    return re.sub(r"[^a-z0-9]", "", locator.lower())


def _normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


_MIN_SNIPPET_ANCHOR_CHARS = 12


def filter_verified_evidence(evidence: list[Evidence], snippets: list[RetrievedSnippet]) -> list[Evidence]:
    """Drops any Evidence entry that doesn't trace back to a snippet this
    module actually retrieved and showed to EvidenceAgent.

    EvidenceAgent's instructions tell it to copy source_id/locator verbatim
    (agents/evidence_agent.py, rule 4), but nothing enforced that
    server-side - a hallucinated or misattributed pair used to flow straight
    into scoring.py, breaking the "no knowledge without provenance"
    guarantee (found 2026-08-22).

    An exact (source_id, locator) string match was too brittle the other way
    (found 2026-08-26): a model that reformats the locator even slightly
    ("src_1#L14" -> "src_1 : line 14") had ALL of a claim's evidence dropped,
    reported as "no evidence" rather than a formatting nit. So: the
    source_id must still be one that was genuinely retrieved (this is the
    real security boundary - a never-retrieved / governance-blocked source
    can never be cited), but the locator is matched after normalizing away
    punctuation/case, with a fallback to anchoring on the snippet text the
    model was actually shown from that same source."""
    locators_by_source: dict[str, set[str]] = {}
    snippets_by_source: dict[str, list[str]] = {}
    for s in snippets:
        locators_by_source.setdefault(s.source_id, set()).add(_normalize_locator(s.locator))
        snippets_by_source.setdefault(s.source_id, []).append(_normalize_snippet(s.snippet))

    verified: list[Evidence] = []
    for e in evidence:
        known_locators = locators_by_source.get(e.source_id)
        if known_locators is None:
            continue  # source was never retrieved / not governance-cleared
        if _normalize_locator(e.locator) in known_locators:
            verified.append(e)
            continue
        norm_snippet = _normalize_snippet(e.snippet)
        if len(norm_snippet) >= _MIN_SNIPPET_ANCHOR_CHARS and any(
            norm_snippet in retrieved or retrieved in norm_snippet
            for retrieved in snippets_by_source.get(e.source_id, ())
        ):
            verified.append(e)
    return verified
