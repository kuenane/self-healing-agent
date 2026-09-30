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
                # Half-open probe: allow one attempt, reset the failure count
                # so a single success fully closes the circuit and a single
                # failure below threshold does not leave stale state.
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
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
            if (
                self.state == CircuitState.HALF_OPEN
                or self.failure_count >= self.failure_threshold
            ):
                # A failed half-open probe reopens the circuit immediately.
                self.state = CircuitState.OPEN
            raise e


class ArizeMCPClient:
    """
    Production-ready MCP client for Arize Phoenix.
    Supports both the hosted MCP server (npx @arizeai/phoenix-mcp) and
    a direct Phoenix REST API fallback.
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        # NOTE: PHOENIX_CLIENT_HEADERS is a *header string*, not a URL.
        # The base URL comes from PHOENIX_MCP_URL (matching main.py docs).
        self.base_url = base_url or os.getenv(
            "PHOENIX_MCP_URL", "http://localhost:6006"
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
        """Call a Phoenix MCP tool via JSON-RPC.

        Each HTTP attempt goes through the circuit breaker, so retries are
        counted as individual failures (not one logical failure per call).
        """

        async def _attempt():
            self.request_count += 1
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
                "id": self.request_count,
            }
            async with session.post(
                f"{self.base_url}/mcp",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "error" in data:
                        raise MCPError(f"MCP error: {data['error']}")
                    return data.get("result", {})
                if response.status == 429:
                    raise MCPError("Rate limited (HTTP 429)")
                text = await response.text()
                raise MCPError(f"HTTP {response.status}: {text}")

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return await self.circuit_breaker.call(_attempt)
            except MCPError as e:
                # Circuit breaker open: no point retrying this call.
                if self.circuit_breaker.state == CircuitState.OPEN and "OPEN" in str(e):
                    last_error = e
                    break
                last_error = e
                await asyncio.sleep(2**attempt)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                await asyncio.sleep(2**attempt)

        self.error_count += 1
        self.last_error = str(last_error)
        if isinstance(last_error, MCPError):
            raise last_error
        raise MCPError(
            f"Failed to call {tool_name} after {max_retries} attempts: {last_error}"
        )

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/healthz",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status < 400
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
