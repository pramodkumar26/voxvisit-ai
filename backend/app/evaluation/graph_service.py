"""LG-5B: resuming a LangGraph conversation, with the backend deciding who may.

`GraphConversations.turn` is what a request handler would call with the session cookie. Every turn,
first, the backend validates that cookie (`AuthService.authenticate`: it refreshes the idle
deadline and rejects an unknown, expired, logged-out or revoked session); only a verified principal
reaches the checkpoint, and the checkpoint is found from that principal's session, never from
anything the caller sends. Holding a stored conversation therefore proves nothing and restores
nothing: a revoked or expired session cannot authenticate, and a new session has a new thread.

After sign-in-state checks, a saved conversation is still only continued if it belongs to the same
patient and its review (if any) is still live in the backend; a review that was confirmed, has
lapsed or was invalidated is dropped, so a stale review is never presented again. Confirmation
itself is unchanged: only the authenticated UI endpoint can commit, and the backend refuses a
pending action unless it was issued to the very session that confirms it.
"""

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, text

from app.agent_tools import ToolRuntime
from app.auth.service import (
    AuthenticationRequiredError,
    AuthService,
    Principal,
)
from app.evaluation.conversation import AgentReply
from app.evaluation.graph_store import CheckpointStore, thread_for
from app.evaluation.llm import LlmConfig
from app.evaluation.orchestrator import OrchestratedSession

logger = logging.getLogger("voxvisit.graph_service")


class GraphConversations:
    def __init__(
        self,
        *,
        store: CheckpointStore,
        engine: Engine,
        auth: AuthService,
        config: LlmConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        session_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self._engine = engine
        self._auth = auth
        self._config = config or LlmConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._options = dict(session_options or {})  # e.g. a scripted reader, in tests

    # -- a turn -----------------------------------------------------------------------------

    def turn(self, token: str | None, utterance: str) -> AgentReply:
        return self._session(self._verified(token)).respond(utterance)

    def event(self, token: str | None, payload: Mapping[str, Any]) -> AgentReply:
        return self._session(self._verified(token)).confirmation_event(payload)

    def _verified(self, token: str | None) -> Principal:
        principal = self._auth.authenticate(token)  # Raises for anything but a live session.
        if principal.patient_id is None:
            raise AuthenticationRequiredError()  # Only patients converse here.
        return principal

    def _session(self, principal: Principal) -> OrchestratedSession:
        return OrchestratedSession(
            ToolRuntime(self._engine, clock=self._clock),
            principal,
            config=self._config,
            store=self.store,
            review_status=self._review_status(principal),
            clock=self._clock,
            **self._options,
        )

    def _review_status(self, principal: Principal) -> Callable[[str], str | None]:
        """Whether the review a saved conversation holds is still live, by the rule the
        confirmation step itself applies (the database's clock, the shorter of the two limits)."""

        def status(pending_action_id: str) -> str | None:
            with self._engine.connect() as connection:
                row = connection.execute(
                    text(
                        "SELECT status, clock_timestamp() >="
                        " LEAST(expires_at, created_at + interval '10 minutes') AS lapsed"
                        " FROM pending_actions"
                        " WHERE pending_action_id = :id AND session_id = :session"
                    ),
                    {"id": pending_action_id, "session": principal.session_id},
                ).one_or_none()
            if row is None:
                return "MISSING"  # Not this session's: nothing to continue.
            return "EXPIRED" if row.status == "ACTIVE" and row.lapsed else str(row.status)

        return status

    # -- the end of a session ---------------------------------------------------------------

    def sign_out(self, principal: Principal) -> None:
        """Delete the conversation with the session. (The backend has already invalidated the
        session's pending actions when it logged out.)"""
        self.store.delete_thread(thread_for(principal.session_id))

    def expire(self) -> list[str]:
        """Remove idle conversations and those of sessions that are over."""
        return self.store.expire(self._engine, now=self._clock())
