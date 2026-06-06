"""
main.py — Run the Arize Self-Healing Agent demo.

Usage:
    python main.py

Environment variables:
    PHOENIX_API_KEY     Your Phoenix Cloud API key (from app.phoenix.arize.com)
    PHOENIX_MCP_URL     Defaults to http://localhost:6006
    REDIS_URL           Defaults to redis://localhost:6379
    GEMINI_API_KEY      Your Gemini API key
"""
import asyncio
import os
from src.mcp_client import ArizeMCPClient
from src.agent_core import SelfHealingAgent


SAMPLE_TASKS = [
    "Retrieve the last 5 failed traces from Phoenix and summarise failure types.",
    "Check if our agent's correctness score has improved over the last 7 days.",
    "Identify the most common tool call pattern across recent experiments.",
    "Delete all test datasets from the Phoenix project.",          # triggers approval gate
]


async def main():
    phoenix_url = os.getenv("PHOENIX_MCP_URL", "http://localhost:6006")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    print("=" * 60)
    print("Arize Self-Healing Agent — Demo")
    print("=" * 60)
    print(f"  Phoenix URL : {phoenix_url}")
    print(f"  Redis URL   : {redis_url}")
    print()

    mcp = ArizeMCPClient(base_url=phoenix_url)
    healthy = await mcp.health_check()
    print(f"  Phoenix health check: {'✓ OK' if healthy else '✗ Unreachable (degraded mode)'}")
    print()

    agent = SelfHealingAgent(mcp_client=mcp, redis_url=redis_url)
    await agent.state_manager.connect()

    for i, task in enumerate(SAMPLE_TASKS, 1):
        print(f"[Task {i}/{len(SAMPLE_TASKS)}] {task}")
        result = await agent.execute(task)
        print(f"  trace_id       : {result.trace_id}")
        print(f"  success        : {result.success}")
        print(f"  correctness    : {result.correctness_score:.2f}")
        print(f"  efficiency     : {result.efficiency_score:.2f}")
        print(f"  duration_ms    : {result.duration_ms}")
        if result.requires_approval:
            print(f"  ⚠  Paused — requires human approval")
            print(f"     Proposed: {result.proposed_action}")
        if result.learned_patterns_applied:
            print(f"  Patterns applied: {result.learned_patterns_applied}")
        if result.error:
            print(f"  Error: {result.error}")
        print()

    print("Performance report (last 7 days):")
    report = await agent.get_performance_report(days=7)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print()
    print("MCP client metrics:")
    for k, v in mcp.get_metrics().items():
        print(f"  {k}: {v}")

    await agent.state_manager.close()
    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
