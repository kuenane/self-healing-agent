"""
Execution metrics: correctness, efficiency, completeness.
Designed so MetricsCalculator can be extended with LLM-as-a-Judge scores.
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ExecutionMetrics:
    correctness_score: float   # 0.0–1.0
    efficiency_score: float    # 0.0–1.0
    completeness_score: float  # 0.0–1.0
    tokens_per_tool_call: float


class MetricsCalculator:
    """
    Heuristic metrics. In production, replace with LLM-as-a-Judge evals
    via the Phoenix eval API.
    """

    @staticmethod
    def calculate(
        tool_calls: List[Dict],
        tokens_used: int,
        duration_ms: int,
    ) -> ExecutionMetrics:
        n_calls = len(tool_calls)

        # Efficiency: penalise excessive tool calls (>10 is suspicious)
        if n_calls == 0:
            efficiency = 1.0
        elif n_calls <= 5:
            efficiency = 1.0
        elif n_calls <= 10:
            efficiency = 0.8
        else:
            efficiency = max(0.1, 1.0 - (n_calls - 10) * 0.05)

        # Correctness: heuristic — at least one successful tool call
        has_errors = any(c.get("error") for c in tool_calls)
        correctness = 0.4 if has_errors else (0.85 if n_calls > 0 else 0.6)

        # Completeness: did we make at least one tool call?
        completeness = min(1.0, n_calls / max(1, 3))

        tokens_per_call = tokens_used / max(1, n_calls)

        return ExecutionMetrics(
            correctness_score=correctness,
            efficiency_score=efficiency,
            completeness_score=completeness,
            tokens_per_tool_call=tokens_per_call,
        )
