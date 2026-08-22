"""
Tests for web_app/app.py::_check_rate_limit (2026-08-22 judge audit fix):
a one-time visitor's entry must eventually be swept, not just an IP's own
entry on its own repeat visit. Pure dict/deque logic, no server or network
needed - patches time.monotonic directly.
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.app import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, _check_rate_limit, _request_log


def test_one_time_visitor_entry_is_eventually_swept(monkeypatch):
    _request_log.clear()
    clock = [1_000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    _check_rate_limit("1.2.3.4")  # a judge/crawler that hits the demo once and never returns
    assert "1.2.3.4" in _request_log

    clock[0] += RATE_LIMIT_WINDOW_SECONDS + 1
    _check_rate_limit("5.6.7.8")  # any other request should trigger the sweep

    assert "1.2.3.4" not in _request_log


def test_rate_limit_still_enforced_within_window(monkeypatch):
    _request_log.clear()
    clock = [2_000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        _check_rate_limit("9.9.9.9")

    with pytest.raises(HTTPException) as exc_info:
        _check_rate_limit("9.9.9.9")
    assert exc_info.value.status_code == 429
