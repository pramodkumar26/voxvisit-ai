"""Sanitized per-call tracing.

A trace records which tool ran, the request id, how long it took, whether it succeeded, and the
category of any error (the error code). It never carries arguments, results, identities, session
tokens, or message text, so a trace is safe to log and to keep.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

logger = logging.getLogger("app.agent_tools")


@dataclass(frozen=True)
class ToolTrace:
    tool: str  # A registered tool name, or "unknown" for anything else the model named.
    request_id: UUID
    duration_ms: float
    outcome: Literal["ok", "error"]
    error_category: str | None  # The envelope error code, for example RESOURCE_NOT_AVAILABLE.
    attempts: int


class TraceSink(Protocol):
    def record(self, trace: ToolTrace) -> None: ...


class LoggingTraceSink:
    """Writes one structured log line per call through the application's redacting formatter."""

    def record(self, trace: ToolTrace) -> None:
        logger.info(
            "tool call",
            extra={
                "tool": trace.tool,
                "request_id": str(trace.request_id),
                "duration_ms": round(trace.duration_ms, 2),
                "outcome": trace.outcome,
                "error_category": trace.error_category,
                "attempts": trace.attempts,
            },
        )


class InMemoryTraceSink:
    """Keeps traces in a list, for tests and local inspection."""

    def __init__(self) -> None:
        self.traces: list[ToolTrace] = []

    def record(self, trace: ToolTrace) -> None:
        self.traces.append(trace)
