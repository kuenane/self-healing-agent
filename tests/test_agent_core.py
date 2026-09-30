"""
Tests for the self-healing core (agent_core) — the previously untested part
of the system. All tests run without Redis, Phoenix, Gemini, or google-adk.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_core import SelfHealingAgent, AgentPhase, _StubAgent
from src.state_manager import StateManager


# ── Helpers ───────────────────────────────────────────────────────────────

def make_agent() -> SelfHealingAgent:
    """Agent with mocked MCP client, in-memory state, and a fake ADK agent."""
    mcp = MagicMock()
    mcp.call_tool = AsyncMock(return_value={"trace_id": "trace-test"})
    agent = SelfHealingAgent(mcp_client=mcp)
    agent.embedder = None  # no model downloads in tests
    # Fake ADK agent that "does work" successfully.
    fake_llm = MagicMock()
    fake_llm.is_stub = False
    fake_llm.run_called = False

    async def run(task):
        fake_llm.run_called = True
        r = MagicMock()
        r.tool_calls = [{"tool": "query", "args": {}}]
        r.tokens_used = 120
        return r

    fake_llm.run = run
    agent.agent = fake_llm
    return agent


def run_async(coro):
    return asyncio.run(coro)


# ── Phase enum ────────────────────────────────────────────────────────────

class TestPhases:
    def test_six_phases_documented_and_implemented(self):
        names = [p.name for p in AgentPhase]
        assert names == ["TRACE", "RETRIEVE", "EXECUTE", "EVALUATE", "LEARN", "IMPROVE"]


# ── Approval gate ─────────────────────────────────────────────────────────

class TestApprovalGate:
    def test_destructive_task_is_gated_and_persisted(self):
        agent = make_agent()
        result = run_async(agent.execute("Delete all test datasets"))
        assert result.requires_approval is True
        assert result.success is False
        # The LLM must NOT have executed the destructive task.
        assert agent.agent.run_called is False

        # Persisted — survives beyond the return value.
        pending = run_async(agent.state_manager.list_pending_approvals())
        assert len(pending) == 1
        assert "delete" in pending[0]["action"]["task"].lower()

    def test_terminated_word_does_not_false_positive(self):
        agent = make_agent()
        # "removed" contains "remove" but is a past-tense description.
        assert run_async(agent._check_approval_required(
            "summarize how we removed duplicates"
        )) is False

    def test_evasive_wording_is_caught(self):
        agent = make_agent()
        assert run_async(agent._check_approval_required(
            "purge production tables"
        )) is True

    def test_evaded_gates_caught(self):
        agent = make_agent()
        assert run_async(agent._check_approval_required(
            "drop the temporary schema"
        )) is True

    def test_reject_resolves_approval(self):
        agent = make_agent()
        result = run_async(agent.execute("Delete the qa namespace"))
        approval_id = result.proposed_action["approval_id"]
        rec = run_async(agent.reject(approval_id))
        assert rec["status"] == "rejected"
        assert run_async(agent.state_manager.list_pending_approvals()) == []


# ── Success semantics ─────────────────────────────────────────────────────

class TestSuccessSemantics:
    def test_stub_agent_never_reports_success(self):
        mcp = MagicMock()
        mcp.call_tool = AsyncMock(return_value={"trace_id": "t"})
        agent = SelfHealingAgent(mcp_client=mcp)
        agent.embedder = None
        agent.agent = _StubAgent()  # force stub
        result = run_async(agent.execute("list recent traces"))
        assert result.success is False

    def test_real_agent_with_tool_calls_succeeds(self):
        agent = make_agent()
        result = run_async(agent.execute("list recent traces"))
        assert result.success is True
        assert result.trace_id == "trace-test"

    def test_execution_record_persisted(self):
        agent = make_agent()
        run_async(agent.execute("list recent traces"))
        history = run_async(
            agent.state_manager.get_execution_history(datetime(2000, 1, 1))
        )
        assert len(history) == 1
        assert history[0].success is True


# ── Pattern learning ──────────────────────────────────────────────────────

class TestPatternLearning:
    def test_excessive_tool_calls_creates_pattern(self):
        agent = make_agent()

        async def run_many(task):
            r = MagicMock()
            r.tool_calls = [{"tool": "x"} for _ in range(12)]
            r.tokens_used = 5000
            return r

        agent.agent.run = run_many
        run_async(agent.execute("check everything repeatedly"))
        patterns = run_async(agent.state_manager.get_all_patterns())
        assert any("Excessive tool calls" in p.description for p in patterns)

    def test_performance_report_after_runs(self):
        agent = make_agent()
        run_async(agent.execute("list recent traces"))
        report = run_async(agent.get_performance_report(days=7))
        assert report["total_executions"] == 1
        assert 0.0 <= report["success_rate"] <= 1.0


# ── Error handling in the loop ────────────────────────────────────────────

class TestLoopErrorHandling:
    def test_mcp_failure_does_not_crash_execute(self):
        mcp = MagicMock()
        mcp.call_tool = AsyncMock(side_effect=Exception("phoenix down"))
        agent = SelfHealingAgent(mcp_client=mcp)
        agent.embedder = None

        async def run(task):
            r = MagicMock()
            r.tool_calls = [{"tool": "x"}]
            r.tokens_used = 10
            return r

        agent.agent.run = run
        agent.agent.is_stub = False
        result = run_async(agent.execute("list recent traces"))
        # _create_trace catches MCPError; a generic Exception propagates to
        # the outer handler and produces a failed-but-graceful result.
        assert isinstance(result.success, bool)
