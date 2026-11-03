"""LG-5F voice mode, stage 1: the voice loop's `Session`, backed by the LangGraph orchestrator.

The voice loop only needs `respond(utterance) -> reply`. This module supplies that over
`GraphConversations`, so a spoken turn gets exactly what a typed LG-5F turn gets: the session token
is verified by the backend on every turn, the conversation is the PostgreSQL checkpoint of that
verified session, and the orchestrator's booking-intent and safety rules are untouched.

What this adapter does not do: it has no way to confirm anything. It never calls `event()`, so the
assistant cannot describe a booking as done (only a trusted confirmation event does that), and a
prepared change stays a review that only the authenticated confirmation endpoint can commit.

The session token lives in this object's memory and nowhere else: it is not in `repr`, not logged,
and not passed on except to `GraphConversations.turn`. The PIN is never seen here.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.evaluation.conversation import AgentReply
from app.evaluation.graph_service import GraphConversations
from app.evaluation.llm import LlmConfig

logger = logging.getLogger("voxvisit.voice")

# A spoken turn should fail rather than leave the patient in silence for two minutes (the
# evaluation's default). Every other model setting is the validated one.
VOICE_MODEL_TIMEOUT_SECONDS = 30.0
VOICE_MODEL_KEEP_ALIVE = "30m"

SESSION_ENDED_REPLY = (
    "Your session has ended, so I can't continue. Nothing was changed. Please sign in again."
)
TROUBLE_REPLY = "I'm having trouble right now. Nothing was changed. Please try again."


def voice_config() -> LlmConfig:
    return LlmConfig(
        request_timeout_s=VOICE_MODEL_TIMEOUT_SECONDS, keep_alive=VOICE_MODEL_KEEP_ALIVE
    )


class GraphVoiceSession:
    """The voice loop's `Session`: one verified patient session, one checkpointed conversation."""

    def __init__(self, conversations: GraphConversations, token: str) -> None:
        self._conversations = conversations
        self._token = token
        self.ended = False  # True once the backend has refused the session; the loop then stops.

    def __repr__(self) -> str:
        return "GraphVoiceSession()"  # Never the token.

    def respond(self, utterance: str) -> AgentReply:
        """One spoken turn. Always returns something to say, and says nothing about its cause."""
        if self.ended:
            return AgentReply(SESSION_ENDED_REPLY, False)
        try:
            return self._conversations.turn(self._token, utterance)
        except (AuthenticationRequiredError, SessionExpiredError):
            self.ended = True  # Idle limit, sign-out elsewhere, revocation: nothing resumes.
            return AgentReply(SESSION_ENDED_REPLY, False)
        except Exception as error:  # Whatever went wrong, the turn changed nothing we know of.
            logger.warning("a voice turn failed", extra={"error_type": type(error).__name__})
            return AgentReply(TROUBLE_REPLY, False)


def open_graph_voice(
    engine,
    auth,
    issued,
    *,
    clock: Callable[[], datetime] | None = None,
    session_factory: Callable[[GraphConversations, str], GraphVoiceSession] | None = None,
) -> tuple[GraphVoiceSession, Callable[[], None]]:
    """Wire LG-5F for a signed-in patient. Returns the session and a release function that is
    called after sign-out: it deletes the conversation's checkpoints and closes the store.

    `session_factory` lets a front end add behaviour around the conversation (the browser review
    does) without this adapter changing: it receives the conversations and the session token."""
    from app.evaluation.graph_store import CheckpointStore
    from app.evaluation.turn_reader import TurnReader

    config = voice_config()
    store = CheckpointStore(engine.url).open()
    try:
        conversations = GraphConversations(
            store=store, engine=engine, auth=auth, config=config, clock=clock
        )
        # Load the model and the fixed start of the prompt now, not on the patient's first words.
        TurnReader(config).warm_up((clock or (lambda: datetime.now(UTC)))().date())
    except BaseException:
        store.close()
        raise

    def release() -> None:
        try:
            conversations.sign_out(issued.principal)  # Also removes the conversation's state.
        finally:
            store.close()

    return (session_factory or GraphVoiceSession)(conversations, issued.token), release
