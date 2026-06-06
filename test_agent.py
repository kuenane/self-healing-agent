"""
Unit tests for the self-healing agent.
Run with:  pytest tests/ -v
All tests work without Redis, Phoenix, or Gemini keys.
"""
import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.state_manager import StateManager, FailurePattern, ExecutionRecord
from src.error_handler import ErrorHandler, with_retry, fallback_response
from src.metrics import MetricsCalculator
from src.mcp_client import ArizeMCPClient, MCPError, CircuitBreaker, CircuitState


# ── Helpers ───────────────────────────────────────────────────────────────

def make_pattern(id="p1", confidence=0.8) -> FailurePattern:
    now = datetime.now()
    return FailurePattern(
        id=id,
        description="Test pattern",
        suggested_fix="Do something different",
        confidence=confidence,
        occurrences=1,
        first_seen=now,
        last_seen=now,
    )


# ── StateManager ─────────────────────────────────────────────────────────

class TestStateManager:
    def setup_method(self):
        self.sm = StateManager()  # no Redis URL → in-memory mode

    def test_initial_empty(self):
        assert asyncio.run(self.sm.get_all_patterns()) == []

    def test_save_and_retrieve_pattern(self):
        p = make_pattern()
        asyncio.run(self.sm.save_pattern(p))
        result = asyncio.run(self.sm.get_pattern("p1"))
        assert result is not None
        assert result.id == "p1"
        assert result.confidence == 0.8

    def test_high_confidence_filter(self):
        asyncio.run(self.sm.save_pattern(make_pattern("low", confidence=0.3)))
        asyncio.run(self.sm.save_pattern(make_pattern("high", confidence=0.95)))
        top = asyncio.run(self.sm.get_high_confidence_patterns(limit=1))
        assert top[0].id == "high"

    def test_save_and_retrieve_execution(self):
        rec = ExecutionRecord(
            trace_id="t1",
            task="test task",
            success=True,
            correctness=0.9,
            timestamp=datetime.now(),
        )
        asyncio.run(self.sm.save_execution(rec))
        history = asyncio.run(
            self.sm.get_execution_history(datetime.min)
        )
        assert any(r.trace_id == "t1" for r in history)


# ── Metrics ───────────────────────────────────────────────────────────────

class TestMetrics:
    def test_no_tool_calls(self):
        m = MetricsCalculator.calculate([], 0, 0)
        assert m.efficiency_score == 1.0
        assert m.completeness_score == 0.0

    def test_moderate_tool_calls(self):
        calls = [{"tool": "x"} for _ in range(4)]
        m = MetricsCalculator.calculate(calls, 200, 500)
        assert m.efficiency_score == 1.0
        assert m.correctness_score > 0.5

    def test_excessive_tool_calls_penalised(self):
        calls = [{"tool": "x"} for _ in range(15)]
        m = MetricsCalculator.calculate(calls, 1000, 2000)
        assert m.efficiency_score < 0.8

    def test_error_in_calls_lowers_correctness(self):
        calls = [{"tool": "x", "error": "oops"}]
        m = MetricsCalculator.calculate(calls, 50, 100)
        assert m.correctness_score < 0.5


# ── ErrorHandler ─────────────────────────────────────────────────────────

class TestErrorHandler:
    def setup_method(self):
        self.handler = ErrorHandler()

    def test_fallback_timeout(self):
        fb = self.handler.get_fallback_response(TimeoutError("connection timed out"))
        assert fb["status"] == "degraded"
        assert "retry_after_seconds" in fb

    def test_fallback_auth(self):
        fb = self.handler.get_fallback_response(Exception("authentication failed"))
        assert fb["requires_human_intervention"] is True

    def test_fallback_generic(self):
        fb = self.handler.get_fallback_response(ValueError("something broke"))
        assert fb["status"] == "error"


# ── Retry decorator ───────────────────────────────────────────────────────

class TestRetryDecorator:
    def test_succeeds_after_retries(self):
        call_count = {"n": 0}

        @with_retry(max_retries=3, backoff_factor=0)
        async def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        result = asyncio.run(flaky())
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_after_all_retries(self):
        @with_retry(max_retries=2, backoff_factor=0)
        async def always_fails():
            raise RuntimeError("broken")

        with pytest.raises(RuntimeError):
            asyncio.run(always_fails())


# ── Fallback decorator ───────────────────────────────────────────────────

class TestFallbackDecorator:
    def test_returns_default_on_error(self):
        @fallback_response(default={"status": "fallback"})
        async def broken():
            raise RuntimeError("kaboom")

        result = asyncio.run(broken())
        assert result == {"status": "fallback"}


# ── CircuitBreaker ────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999)

        async def failing():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                asyncio.run(cb.call(failing))
            except RuntimeError:
                pass

        assert cb.state == CircuitState.OPEN

    def test_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.now()

        async def fine():
            return "ok"

        with pytest.raises(MCPError, match="Circuit breaker OPEN"):
            asyncio.run(cb.call(fine))
