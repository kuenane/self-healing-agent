"""Arize Self-Healing Agent package."""

from .agent_core import SelfHealingAgent, ExecutionResult, AgentPhase
from .mcp_client import ArizeMCPClient, MCPError
from .state_manager import StateManager, FailurePattern, ExecutionRecord
from .error_handler import ErrorHandler
from .metrics import ExecutionMetrics, MetricsCalculator

__version__ = "2.1.0"

__all__ = [
    "SelfHealingAgent",
    "ExecutionResult",
    "AgentPhase",
    "ArizeMCPClient",
    "MCPError",
    "StateManager",
    "FailurePattern",
    "ExecutionRecord",
    "ErrorHandler",
    "ExecutionMetrics",
    "MetricsCalculator",
]
