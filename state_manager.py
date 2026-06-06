"""
State management with Redis persistence.
Falls back to in-memory storage when Redis is unavailable.
"""
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import redis.asyncio as redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisError = Exception


@dataclass
class FailurePattern:
    """A learned failure pattern the agent uses to avoid repeating mistakes."""
    id: str
    description: str
    suggested_fix: str
    confidence: float       # 0.0 – 1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["first_seen"] = d["first_seen"].isoformat()
        d["last_seen"] = d["last_seen"].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FailurePattern":
        data = dict(data)
        data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**data)


@dataclass
class ExecutionRecord:
    """Records a single completed execution."""
    trace_id: str
    task: str
    success: bool
    correctness: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "success": self.success,
            "correctness": self.correctness,
            "timestamp": self.timestamp.isoformat(),
        }


class StateManager:
    """
    Manages persistent agent state.
    Uses Redis when available; falls back to in-process dicts otherwise.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis: Optional[object] = None
        self._use_redis = False

        # In-memory fallback stores
        self._memory_patterns: Dict[str, FailurePattern] = {}
        self._memory_history: Dict[str, ExecutionRecord] = {}

        self._pattern_key = "agent:failure_patterns"
        self._history_key = "agent:execution_history"

    async def connect(self) -> bool:
        if not REDIS_AVAILABLE:
            print("redis package not installed — using in-memory state.")
            return False
        try:
            self._redis = await redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            await self._redis.ping()
            self._use_redis = True
            return True
        except Exception as e:
            print(f"Redis unavailable ({e}) — using in-memory state.")
            return False

    # ── Patterns ──────────────────────────────────────────────────────────

    async def save_pattern(self, pattern: FailurePattern) -> None:
        if self._use_redis:
            key = f"{self._pattern_key}:{pattern.id}"
            await self._redis.set(key, json.dumps(pattern.to_dict()), ex=60 * 60 * 24 * 30)
            await self._redis.zadd(
                f"{self._pattern_key}:by_confidence",
                {pattern.id: pattern.confidence},
            )
        else:
            self._memory_patterns[pattern.id] = pattern

    async def get_pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        if self._use_redis:
            data = await self._redis.get(f"{self._pattern_key}:{pattern_id}")
            return FailurePattern.from_dict(json.loads(data)) if data else None
        return self._memory_patterns.get(pattern_id)

    async def get_all_patterns(self) -> List[FailurePattern]:
        if self._use_redis:
            ids = await self._redis.zrevrange(
                f"{self._pattern_key}:by_confidence", 0, -1
            )
            patterns = []
            for pid in ids:
                p = await self.get_pattern(pid)
                if p:
                    patterns.append(p)
            return patterns
        return list(self._memory_patterns.values())

    async def get_high_confidence_patterns(self, limit: int = 10) -> List[FailurePattern]:
        patterns = await self.get_all_patterns()
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)[:limit]

    # ── Execution history ─────────────────────────────────────────────────

    async def save_execution(self, record: ExecutionRecord) -> None:
        if self._use_redis:
            await self._redis.zadd(
                self._history_key,
                {record.trace_id: record.timestamp.timestamp()},
            )
            await self._redis.set(
                f"{self._history_key}:{record.trace_id}",
                json.dumps(record.to_dict()),
                ex=60 * 60 * 24 * 7,
            )
        else:
            self._memory_history[record.trace_id] = record

    async def get_execution_history(self, since: datetime) -> List[ExecutionRecord]:
        if self._use_redis:
            min_score = since.timestamp()
            ids = await self._redis.zrangebyscore(self._history_key, min_score, "+inf")
            records = []
            for tid in ids:
                raw = await self._redis.get(f"{self._history_key}:{tid}")
                if raw:
                    d = json.loads(raw)
                    records.append(
                        ExecutionRecord(
                            trace_id=d["trace_id"],
                            task=d["task"],
                            success=d["success"],
                            correctness=d["correctness"],
                            timestamp=datetime.fromisoformat(d["timestamp"]),
                        )
                    )
            return sorted(records, key=lambda r: r.timestamp)

        try:
            cutoff = since.timestamp()
        except (OSError, OverflowError, ValueError):
            cutoff = 0.0
        return sorted(
            [r for r in self._memory_history.values() if r.timestamp.timestamp() >= cutoff],
            key=lambda r: r.timestamp,
        )

    async def clear_old_data(self, days_to_keep: int = 30) -> None:
        if not self._use_redis:
            return
        cutoff = datetime.now().timestamp() - days_to_keep * 86400
        old_ids = await self._redis.zrangebyscore(self._history_key, 0, cutoff)
        for tid in old_ids:
            await self._redis.delete(f"{self._history_key}:{tid}")
        await self._redis.zremrangebyscore(self._history_key, 0, cutoff)

    async def close(self) -> None:
        if self._redis and self._use_redis:
            await self._redis.close()
