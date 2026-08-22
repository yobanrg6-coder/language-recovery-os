"""
Tests for agents/orchestrator.py::resolve_audio_mime_type (2026-08-22 judge
audit fix). Pure function, no API key or ADK runner needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import resolve_audio_mime_type


def test_known_extensions_resolve():
    assert resolve_audio_mime_type(".wav") == "audio/wav"
    assert resolve_audio_mime_type(".WEBM") == "audio/webm"
    assert resolve_audio_mime_type(".mp3") == "audio/mpeg"


def test_unrecognized_extension_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        resolve_audio_mime_type(".flac")
