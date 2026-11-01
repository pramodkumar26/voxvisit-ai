"""Runs one tool call: authorize, validate, execute, retry safely, trace, and wrap in the envelope.

Order matters. The role and session requirements are checked before the arguments are validated, so
an unauthorized caller learns nothing about a tool's schema. Arguments are validated against the
documented fields only. The session is then re-verified against PostgreSQL before any protected
tool runs. Every outcome, including unexpected failures, becomes an envelope with a sanitized error.
"""

import hashlib
import json
import logging
import random
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Engine

from app.agent_tools import envelope
from app.agent_tools.context import ToolContext
from app.agent_tools.registry import Access, ToolRegistry, ToolSpec, build_default_registry
from app.agent_tools.tools import ToolFailure
from app.agent_tools.tracing import LoggingTraceSink, ToolTrace, TraceSink
from app.auth.service import Clock, Principal, utc_now
from app.core.errors import (
    ApplicationError,
    AuthenticationRequiredError,
    DependencyUnavailableError,
    ForbiddenError,
)
from app.services.action_support import lock_patient_session
from app.services.pending_action_retirement import discard_pending_action
from app.services.read_access import read_only_connection

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 2  # One retry after a transient dependency failure, as the contract allows.
RETRY_DELAY_SECONDS = (0.05, 0.25)


class ToolCall:
    """One logical tool call, which owns its request id and idempotency key.

    The key is generated here, never by the model, and is reused only when the same logical call is
    retried (after an unknown outcome). A new decision by the agent is a new ToolCall with a new
    key. A call is bound to its first tool and arguments, so a retry cannot quietly change them.
    """

    def __init__(self) -> None:
        self.request_id = uuid.uuid4()
        self.idempotency_key = f"tool.{uuid.uuid4().hex}"
        self._fingerprint: str | None = None

    def bind(self, tool: object, arguments: object) -> bool:
        canonical = json.dumps([tool, arguments], sort_keys=True, default=str)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        if self._fingerprint is None:
            self._fingerprint = fingerprint
        return self._fingerprint == fingerprint


class ToolRuntime:
    def __init__(
        self,
        engine: Engine,
        *,
        registry: ToolRegistry | None = None,
        clock: Clock = utc_now,
        trace_sink: TraceSink | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._engine = engine
        self._registry = registry or build_default_registry()
        self._clock = clock
        self._sink = trace_sink or LoggingTraceSink()
        self._sleep = sleep
        self._jitter = jitter
        self._timer = timer

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def new_call(self) -> ToolCall:
        return ToolCall()

    def invoke(
        self,
        tool: object,
        arguments: Mapping[str, Any] | None,
        *,
        principal: Principal | None,
        call: ToolCall | None = None,
    ) -> envelope.ToolResponse:
        """Run a tool and return the envelope. Never raises for a tool or backend failure.

        `principal` is the verified session the surrounding application established; the model
        never supplies it. Pass the same `call` again to retry a call whose outcome is unknown.
        """
        call = call or ToolCall()
        started = self._timer()
        spec = self._registry.get(tool)
        attempts = 0
        try:
            if spec is None:
                raise _Rejected("UNKNOWN_TOOL", "That tool is not available.")
            if not isinstance(arguments, Mapping) and arguments is not None:
                raise _Rejected("INVALID_ARGUMENTS_SHAPE", "The tool arguments must be an object.")
            raw = dict(arguments or {})
            if not call.bind(spec.name, raw):
                raise _Rejected(
                    "CALL_REUSED", "A retried call cannot change its tool or arguments."
                )
            self._authorize(spec, principal)
            parsed = spec.adapter.validate_python(raw)
            while True:
                attempts += 1
                try:
                    data = self._run(spec, parsed, principal, call)
                    break
                except DependencyUnavailableError:
                    if attempts >= MAX_ATTEMPTS:
                        raise
                    self._sleep(self._jitter(*RETRY_DELAY_SECONDS))
            response = envelope.success(call.request_id, data)
        except _Rejected as rejected:
            response = envelope.failure(
                call.request_id,
                envelope.simple_error("INVALID_ARGUMENT", rejected.message, reason=rejected.reason),
            )
        except ValidationError as exc:
            response = envelope.failure(call.request_id, envelope.error_from_validation(exc, raw))
        except ToolFailure as outcome:
            response = envelope.failure(call.request_id, outcome.error)
        except ApplicationError as exc:
            response = envelope.failure(call.request_id, envelope.error_from_application(exc))
        except Exception as exc:  # A bug or an unexpected database fault: fail safe and generic.
            logger.error("tool call failed", extra={"error_type": type(exc).__name__})
            response = envelope.failure(call.request_id, envelope.INTERNAL_ERROR)

        self._trace(spec, call, response, started, attempts)
        return response

    def discard_pending_action(self, principal: Principal, pending_action_id: object) -> bool:
        """Retire one of the caller's own pending actions so its review screen can no longer commit.

        This is for trusted application code (the conversation session) when a patient drops a
        request. It is not a tool: it is not in the registry, so a model cannot name it, and it can
        only ever remove the ability to confirm. A foreign, unknown, or finished action is False.
        """
        if principal.actor_type != "PATIENT":
            raise ForbiddenError()
        return discard_pending_action(
            self._engine,
            principal=principal,
            pending_action_id=uuid.UUID(str(pending_action_id)),
            request_id=uuid.uuid4(),
            now=self._clock(),
        )

    # --- steps ----------------------------------------------------------------------------

    @staticmethod
    def _authorize(spec: ToolSpec, principal: Principal | None) -> None:
        """Role and presence checks that need no database and reveal nothing about arguments."""
        if principal is not None and principal.actor_type != "PATIENT":
            raise ForbiddenError()  # Staff use the dashboard, not the agent's tools.
        if spec.access is Access.PATIENT and principal is None:
            raise AuthenticationRequiredError()

    def _run(
        self, spec: ToolSpec, parsed: Any, principal: Principal | None, call: ToolCall
    ) -> dict[str, Any]:
        now = self._clock()
        if spec.access is Access.PATIENT:
            # Re-verify the session and account now, not when the principal was first created.
            with read_only_connection(self._engine) as connection:
                lock_patient_session(connection, principal, now, lock=False)
        context = ToolContext(
            engine=self._engine,
            principal=principal,
            now=now,
            request_id=call.request_id,
            idempotency_key=call.idempotency_key,
        )
        return spec.handler(parsed, context)

    def _trace(
        self,
        spec: ToolSpec | None,
        call: ToolCall,
        response: envelope.ToolResponse,
        started: float,
        attempts: int,
    ) -> None:
        trace = ToolTrace(
            tool=spec.name if spec else "unknown",
            request_id=call.request_id,
            duration_ms=(self._timer() - started) * 1000,
            outcome="ok" if response.ok else "error",
            error_category=response.error.code if response.error else None,
            attempts=attempts,
        )
        try:
            self._sink.record(trace)
        except Exception:  # Tracing must never change a tool's outcome.
            logger.warning("could not record a tool trace")


class _Rejected(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.message = message
