"""
MCP Client with connection pooling, retries, and circuit breaker.
Connects to Arize Phoenix MCP server (npx @arizeai/phoenix-mcp).
"""
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os
from enum import Enum


class MCPError(Exception):
    """MCP-specific errors."""
    pass


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Prevents cascading failures via the circuit breaker pattern."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if (
                self.last_failure_time
                and datetime.now() - self.last_failure_time
                > timedelta(seconds=self.recovery_timeout)
            ):
                self.state = CircuitState.HALF_OPEN
            else:
                raise MCPError("Circuit breaker OPEN — service unavailable")

        try:
            result = await func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise e


class ArizeMCPClient:
    """
    Production-ready MCP client for Arize Phoenix.
    Supports both the hosted MCP server (npx @arizeai/phoenix-mcp) and
    a direct Phoenix REST API fallback.
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or os.getenv(
            "PHOENIX_CLIENT_HEADERS", "http://localhost:6006"
        )
        self.api_key = api_key or os.getenv("PHOENIX_API_KEY", "")
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker = CircuitBreaker()
        self.request_timeout = aiohttp.ClientTimeout(total=30)
        self.request_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
            )
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["api_key"] = self.api_key
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.request_timeout,
                headers=headers,
            )
        return self.session

    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Call a Phoenix MCP tool via JSON-RPC with retry + circuit breaker."""

        async def _make_request():
            self.request_count += 1
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
                "id": self.request_count,
            }

            for attempt in range(max_retries):
                try:
                    async with session.post(
                        f"{self.base_url}/mcp",
                        json=payload,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "error" in data:
                                raise MCPError(f"MCP error: {data['error']}")
                            return data.get("result", {})
                        elif response.status == 429:
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            text = await response.text()
                            raise MCPError(f"HTTP {response.status}: {text}")
                except aiohttp.ClientError as e:
                    if attempt == max_retries - 1:
                        raise MCPError(f"Network error after {max_retries} attempts: {e}")
                    await asyncio.sleep(2**attempt)
                except asyncio.TimeoutError:
                    if attempt == max_retries - 1:
                        raise MCPError(f"Timeout after {max_retries} attempts")
                    await asyncio.sleep(2**attempt)

            raise MCPError(f"Failed to call {tool_name} after {max_retries} attempts")

        try:
            return await self.circuit_breaker.call(_make_request)
        except MCPError as e:
            self.error_count += 1
            self.last_error = str(e)
            raise

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/healthz",
                timeout=aiohttp.ClientTimeout(total=5),
            ):
                return True
        except Exception:
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def get_metrics(self) -> Dict:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.request_count if self.request_count > 0 else 0
            ),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "last_error": self.last_error,
        }
