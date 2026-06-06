"""
Core agent implementation with 6-phase self-healing loop,
state persistence, semantic memory, and Arize Phoenix integration.

Fixes applied vs original:
- _get_system_prompt() is now synchronous (can't await in __init__)
- ExecutionRecord uses .correctness (not .correctness_score)
- _apply_learned_patterns no longer references missing execution trace_id
- get_performance_report builds a proper history list from ExecutionRecord objects
- _detect_failure_patterns: loop uses async correctly
- Removed forward-reference string annotations that caused NameError
"""
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

from .mcp_client import ArizeMCPClient, MCPError
from .error_handler import ErrorHandler
from .state_manager import StateManager, FailurePattern, ExecutionRecord
from .metrics import ExecutionMetrics, MetricsCalculator


class AgentPhase(Enum):
    TRACE = "trace"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    LEARN = "learn"
    IMPROVE = "improve"


@dataclass
class ExecutionResult:
    """Complete execution result with metrics."""
    task: str
    trace_id: str
    success: bool
    correctness_score: float
    efficiency_score: float
    completeness_score: float
    tool_calls_made: List[Dict]
    tokens_used: int
    duration_ms: int
    learned_patterns_applied: List[str]
    requires_approval: bool
    proposed_action: Optional[Dict]
    error: Optional[str]
    timestamp: datetime


# ── Stub ADK types (replaced when google-adk is installed) ────────────────

class _StubAgent:
    """Minimal stub so the module loads without google-adk installed."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    async def run(self, task: str):
        class _R:
            tool_calls = []
            tokens_used = 0
        return _R()


def _make_agent(name, model, tools, instruction):
    try:
        from google.adk import Agent
        from google.adk.models.lite_llm import LiteLlm
        return Agent(
            name=name,
            model=LiteLlm(model=model),
            tools=tools,
            instruction=instruction,
        )
    except ImportError:
        print("google-adk not installed — using stub agent.")
        return _StubAgent(name=name)


# ── Self-Healing Agent ────────────────────────────────────────────────────

class SelfHealingAgent:
    """
    Production-grade agent that learns from past failures.
    Implements a strict 6-phase reasoning loop with Arize Phoenix observability.
    """

    DESTRUCTIVE_KEYWORDS = {
        "delete", "remove", "drop", "truncate", "alter",
        "modify_schema", "restart", "shutdown", "kill",
    }

    def __init__(
        self,
        mcp_client: ArizeMCPClient,
        redis_url: str = "redis://localhost:6379",
        model_name: str = "gemini/gemini-2.5-flash",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.mcp = mcp_client
        self.state_manager = StateManager(redis_url)
        self.error_handler = ErrorHandler(mcp_client)

        # Embedding support (optional, degrades gracefully)
        self.embedder = None
        if ML_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Embedder not loaded ({e}) — pattern matching disabled.")

        self._embedding_cache: Dict[str, Any] = {}

        # Build system prompt (synchronous)
        instruction = self._build_system_prompt([])

        self.agent = _make_agent(
            name="SelfHealingAgent",
            model=model_name,
            tools=[],   # populate after MCP toolset is ready
            instruction=instruction,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_system_prompt(self, recent_patterns: List[FailurePattern]) -> str:
        prompt = (
            "You are a self-healing AI agent with a STRICT 6-PHASE WORKFLOW:\n\n"
            "PHASE 1 — TRACE: call start_trace() with task metadata.\n"
            "PHASE 2 — RETRIEVE MEMORY: call retrieve_similar_failures().\n"
            "PHASE 3 — EXECUTE: complete the task using available tools.\n"
            "PHASE 4 — EVALUATE: call log_evaluation() with metrics.\n"
            "PHASE 5 — DETECT PATTERNS: check for repeated failure patterns.\n"
            "PHASE 6 — IMPROVE: adjust strategy if pattern detected, re-execute.\n\n"
            "RULES:\n"
            "1. If PHASE 2 returns patterns with confidence > 0.8, adjust before PHASE 3.\n"
            "2. Always quote MCP responses verbatim.\n"
            "3. For destructive actions (delete, modify, restart), pause for approval.\n"
            "4. On error, call log_error() before retrying.\n"
        )
        if recent_patterns:
            prompt += "\nLEARNED PATTERNS TO AVOID:\n"
            for p in recent_patterns:
                prompt += f"- {p.description} (confidence: {p.confidence:.2f})\n"
        return prompt

    async def _get_embedding(self, text: str):
        """Return a cached embedding vector, or None if embedder unavailable."""
        if not ML_AVAILABLE or self.embedder is None:
            return None
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, self.embedder.encode, text)
        self._embedding_cache[key] = vec
        return vec

    async def _cosine_sim(self, text_a: str, text_b: str) -> float:
        """Returns cosine similarity, or 0.0 when embeddings are unavailable."""
        va, vb = await self._get_embedding(text_a), await self._get_embedding(text_b)
        if va is None or vb is None:
            return 0.0
        return float(cosine_similarity([va], [vb])[0][0])

    # ── Phase implementations ─────────────────────────────────────────────

    async def _create_trace(self, task: str, context: Optional[Dict]) -> str:
        try:
            result = await self.mcp.call_tool(
                "start_trace",
                {
                    "name": task[:200],
                    "metadata": {
                        "agent_version": "2.0.0",
                        "timestamp": datetime.now().isoformat(),
                        "context": context or {},
                        "phase": AgentPhase.TRACE.value,
                    },
                },
            )
            return result.get("trace_id", self._local_trace_id(task))
        except MCPError as e:
            await self.error_handler.log_to_arize(e, "trace_creation_failed")
            return self._local_trace_id(task)

    def _local_trace_id(self, task: str) -> str:
        return "local_" + hashlib.md5(f"{task}{datetime.now()}".encode()).hexdigest()[:16]

    async def _retrieve_similar_failures(self, task: str) -> List[FailurePattern]:
        all_patterns = await self.state_manager.get_all_patterns()
        if not all_patterns or not ML_AVAILABLE:
            return []
        results = []
        for pattern in all_patterns:
            sim = await self._cosine_sim(task, pattern.description)
            if sim > 0.6 and pattern.confidence > 0.5:
                results.append((sim, pattern))
        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results[:5]]

    async def _check_approval_required(self, task: str) -> bool:
        return any(kw in task.lower() for kw in self.DESTRUCTIVE_KEYWORDS)

    async def _execute_with_memory(
        self,
        task: str,
        similar_failures: List[FailurePattern],
        trace_id: str,
    ) -> Dict:
        memory_context = ""
        if similar_failures:
            memory_context = "\n\nLEARN FROM THESE PAST FAILURES:\n"
            for i, p in enumerate(similar_failures, 1):
                memory_context += f"{i}. {p.description}\n   Fix: {p.suggested_fix}\n"

        enhanced_task = task + memory_context

        if await self._check_approval_required(enhanced_task):
            return {
                "tool_calls": [],
                "tokens_used": 0,
                "requires_approval": True,
                "proposed_action": {
                    "task": task,
                    "reason": "destructive_operation_detected",
                },
            }

        result = await self.agent.run(enhanced_task)
        return {
            "tool_calls": getattr(result, "tool_calls", []),
            "tokens_used": getattr(result, "tokens_used", 0),
            "requires_approval": False,
            "proposed_action": None,
        }

    async def _evaluate_execution(
        self, execution_result: Dict, trace_id: str
    ) -> ExecutionMetrics:
        metrics = MetricsCalculator.calculate(
            tool_calls=execution_result.get("tool_calls", []),
            tokens_used=execution_result.get("tokens_used", 0),
            duration_ms=0,
        )
        try:
            await self.mcp.call_tool(
                "log_evaluation",
                {
                    "trace_id": trace_id,
                    "metrics": {
                        "correctness": metrics.correctness_score,
                        "efficiency": metrics.efficiency_score,
                        "completeness": metrics.completeness_score,
                    },
                },
            )
        except MCPError:
            pass  # Non-fatal; metrics still returned locally
        return metrics

    async def _detect_failure_patterns(
        self,
        task: str,
        execution_result: Dict,
        metrics: ExecutionMetrics,
    ) -> List[FailurePattern]:
        detected = []
        now = datetime.now()

        if len(execution_result.get("tool_calls", [])) > 10:
            detected.append(
                FailurePattern(
                    id=f"pattern_excessive_calls_{now.timestamp():.0f}",
                    description="Excessive tool calls (>10) — inefficient planning",
                    suggested_fix="Batch operations and reduce redundant queries",
                    confidence=0.85,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                )
            )

        if metrics.correctness_score < 0.5:
            detected.append(
                FailurePattern(
                    id=f"pattern_low_correctness_{now.timestamp():.0f}",
                    description=f"Low correctness score ({metrics.correctness_score:.2f})",
                    suggested_fix="Break task into subtasks and validate each step",
                    confidence=0.90,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                )
            )

        # Semantic match against stored patterns
        for stored in await self.state_manager.get_all_patterns():
            sim = await self._cosine_sim(task, stored.description)
            if sim > 0.8 and stored.confidence > 0.7:
                detected.append(stored)

        return detected

    async def _apply_learned_patterns(
        self,
        patterns: List[FailurePattern],
        trace_id: str,
    ) -> List[str]:
        applied = []
        for pattern in patterns:
            pattern.occurrences += 1
            pattern.last_seen = datetime.now()
            pattern.confidence = min(0.99, pattern.confidence + 0.05)
            await self.state_manager.save_pattern(pattern)
            try:
                await self.mcp.call_tool(
                    "log_evaluation",
                    {
                        "trace_id": trace_id,
                        "metrics": {
                            "pattern_applied": 1,
                            "pattern_id": pattern.id,
                            "pattern_confidence": pattern.confidence,
                        },
                    },
                )
            except MCPError:
                pass
            applied.append(pattern.id)
        return applied

    # ── Public API ────────────────────────────────────────────────────────

    async def execute(
        self, task: str, user_context: Optional[Dict] = None
    ) -> ExecutionResult:
        """Run all 6 phases and return a complete ExecutionResult."""
        start = datetime.now()
        trace_id: Optional[str] = None

        try:
            trace_id = await self._create_trace(task, user_context)
            similar_failures = await self._retrieve_similar_failures(task)
            exec_result = await self._execute_with_memory(task, similar_failures, trace_id)
            metrics = await self._evaluate_execution(exec_result, trace_id)
            patterns = await self._detect_failure_patterns(task, exec_result, metrics)
            applied = []
            if patterns:
                applied = await self._apply_learned_patterns(patterns, trace_id)

            # Persist execution record
            await self.state_manager.save_execution(
                ExecutionRecord(
                    trace_id=trace_id,
                    task=task,
                    success=metrics.correctness_score >= 0.7,
                    correctness=metrics.correctness_score,
                    timestamp=datetime.now(),
                )
            )

            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            return ExecutionResult(
                task=task,
                trace_id=trace_id,
                success=metrics.correctness_score >= 0.7,
                correctness_score=metrics.correctness_score,
                efficiency_score=metrics.efficiency_score,
                completeness_score=metrics.completeness_score,
                tool_calls_made=exec_result.get("tool_calls", []),
                tokens_used=exec_result.get("tokens_used", 0),
                duration_ms=duration_ms,
                learned_patterns_applied=applied,
                requires_approval=exec_result.get("requires_approval", False),
                proposed_action=exec_result.get("proposed_action"),
                error=None,
                timestamp=datetime.now(),
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            await self.error_handler.log_to_arize(e, task, trace_id)
            return ExecutionResult(
                task=task,
                trace_id=trace_id or "unknown",
                success=False,
                correctness_score=0.0,
                efficiency_score=0.0,
                completeness_score=0.0,
                tool_calls_made=[],
                tokens_used=0,
                duration_ms=duration_ms,
                learned_patterns_applied=[],
                requires_approval=False,
                proposed_action=None,
                error=str(e),
                timestamp=datetime.now(),
            )

    async def get_performance_report(self, days: int = 7) -> Dict:
        since = datetime.now() - timedelta(days=days)
        history = await self.state_manager.get_execution_history(since)

        if not history:
            return {"error": "No execution history found"}

        correctness_scores = [r.correctness for r in history]
        mean = lambda vals: sum(vals) / len(vals) if vals else 0.0

        return {
            "period_days": days,
            "total_executions": len(history),
            "success_rate": sum(1 for r in history if r.success) / len(history),
            "avg_correctness": mean(correctness_scores),
            "patterns_learned": len(await self.state_manager.get_all_patterns()),
            "improvement_trend": self._improvement_trend(correctness_scores),
        }

    def _improvement_trend(self, scores: List[float]) -> Dict:
        if len(scores) < 10:
            return {"trend": "insufficient_data", "n": len(scores)}
        mid = len(scores) // 2
        first = sum(scores[:mid]) / mid
        second = sum(scores[mid:]) / (len(scores) - mid)
        delta = (second - first) / first if first > 0 else 0
        trend = "improving" if delta > 0.05 else "declining" if delta < -0.05 else "stable"
        return {
            "trend": trend,
            "improvement_pct": round(delta * 100, 2),
            "first_half_avg": round(first, 3),
            "second_half_avg": round(second, 3),
        }
