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

MAX_SNIPPETS_PER_SOURCE = 4
CONTEXT_LINES = 1
_FUZZY_CUTOFF = 0.6


@dataclass(frozen=True)
class RetrievedSnippet:
    source_id: str
    locator: str
    snippet: str


def _load_lines(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def _line_matches(line: str, query: str) -> bool:
    lowered = line.lower()
    if query.lower() in lowered:
        return True
    tokens = [t for t in re.split(r"\W+", lowered) if t]
    return any(difflib.SequenceMatcher(None, t, query.lower()).ratio() >= _FUZZY_CUTOFF for t in tokens)


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
    results: list[RetrievedSnippet] = []
    for source_id, file_path, locator_prefix in searchable_sources:
        results.extend(search_source(source_id, file_path, query, locator_prefix))
    return results
