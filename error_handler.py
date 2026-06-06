"""
Sophisticated error handling with automatic recovery, retry decorator,
fallback responses, and critical alert routing.
"""
import asyncio
import functools
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Handles errors gracefully:
    - Categorises by severity
    - Logs to Arize Phoenix via MCP
    - Emits critical alerts
    - Returns typed fallback responses
    """

    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.error_history: list = []
        self.critical_errors: list = []

    async def log_to_arize(
        self,
        error: Exception,
        context: str,
        trace_id: Optional[str] = None,
    ) -> None:
        error_info = {
            "trace_id": trace_id or "unknown",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc(),
        }
        self.error_history.append(error_info)

        if self.mcp_client:
            try:
                await self.mcp_client.call_tool("log_error", error_info)
            except Exception:
                logger.error("Failed to log error to Arize", exc_info=True)

        if self._is_critical(error):
            self.critical_errors.append(error_info)
            await self._send_critical_alert(error_info)

    def _is_critical(self, error: Exception) -> bool:
        critical_types = (ConnectionError, TimeoutError, MemoryError, SystemError)
        return isinstance(error, critical_types) or "authentication" in str(error).lower()

    async def _send_critical_alert(self, error_info: Dict) -> None:
        logger.critical(
            "CRITICAL ERROR: %s — %s | context: %s",
            error_info["error_type"],
            error_info["error_message"],
            error_info["context"],
        )

    def get_fallback_response(self, error: Exception) -> Dict:
        """Return an appropriate fallback dict based on error type."""
        s = str(error).lower()
        if "timeout" in s or "connection" in s:
            return {
                "status": "degraded",
                "message": "MCP service temporarily unavailable. Using cached responses.",
                "suggestion": "Check Phoenix server health and network connectivity.",
                "retry_after_seconds": 30,
            }
        if "authentication" in s or "api_key" in s:
            return {
                "status": "error",
                "message": "Authentication failed. Check PHOENIX_API_KEY.",
                "requires_human_intervention": True,
            }
        if "rate limit" in s or "429" in s:
            return {
                "status": "degraded",
                "message": "Rate limit exceeded. Slowing down.",
                "backoff_seconds": 5,
            }
        return {
            "status": "error",
            "message": f"Unexpected error: {error}",
            "requires_human_intervention": True,
        }


def with_retry(
    max_retries: int = 3,
    backoff_factor: int = 2,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Async decorator: automatic retry with exponential backoff.

    Usage:
        @with_retry(max_retries=3, exceptions=(MCPError, aiohttp.ClientError))
        async def my_fn(...): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    wait = backoff_factor**attempt
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %ds",
                        func.__name__,
                        attempt + 1,
                        max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def fallback_response(default: Any):
    """
    Async decorator: return `default` instead of raising on any exception.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning("%s raised %s — returning fallback", func.__name__, e)
                return default

        return wrapper

    return decorator
