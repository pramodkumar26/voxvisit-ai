"""Conversation state for the text agent, and the session-scoped interface a voice agent can reuse.

The agent is text in, text out. `TextSession.respond` takes what the patient said and returns what
to say back; a speech-to-text front end and a text-to-speech back end can sit on either side of it
without the agent knowing. The session owns one conversation's state, is bound to the verified
session that created it, and talks to the backend only through the tool runtime with that
session's principal. Nothing here can confirm an appointment: the trusted UI does that, and tells
the session afterwards with `confirmation_event`.

State is memory only and never stored in the database: a draft of what the patient asked for, the
question the agent last asked, and the choices it offered. It holds no availability and no
authorization. Both are checked again by the backend on every tool call.
"""

import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.auth.service import Principal
from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.evaluation.model import Message, Model, Say, ToolAction

logger = logging.getLogger(__name__)
MAX_ACTIONS_PER_TURN = 8  # Tool calls and replies the agent may take before it must speak.
MAX_TURNS_PER_SESSION = 40  # A conversation longer than this starts over.
MAX_UTTERANCE_CHARS = 2000
MAX_SESSIONS = 500
IDLE_SECONDS = 30 * 60


class Awaiting(StrEnum):
    """What the agent last asked the patient, so the next message is read as an answer to it."""

    NOTHING = "NOTHING"
    OPERATION = "OPERATION"
    CLINIC = "CLINIC"
    CLINIC_CHOICE = "CLINIC_CHOICE"
    DOCTOR = "DOCTOR"
    DATE = "DATE"
    TIME = "TIME"
    APPOINTMENT = "APPOINTMENT"


@dataclass(frozen=True)
class Offer:
    """One choice the agent put to the patient: what was said, and what it stands for."""

    label: str
    key: str


@dataclass(frozen=True)
class Prepared:
    """The change the agent last prepared for review. Any new request makes it out of date."""

    pending_action_id: str
    summary: str


@dataclass
class ConversationState:
    draft: Any = None  # The merged PatientIntent for the request in progress; None when idle.
    awaiting: Awaiting = Awaiting.NOTHING
    last_question: str = ""
    offered: list[Offer] = field(default_factory=list)
    prepared: Prepared | None = None
    # Choices the patient picked from what was offered. Each is used only if the next fresh
    # search still returns it; otherwise it is dropped and the patient is asked again.
    pin_appointment: str | None = None
    pin_slot: str | None = None
    turns: int = 0

    def reset(self) -> None:
        self.draft = None
        self.awaiting = Awaiting.NOTHING
        self.last_question = ""
        self.offered = []
        self.prepared = None
        self.pin_appointment = self.pin_slot = None


@dataclass(frozen=True)
class AgentReply:
    text: str
    awaiting_confirmation: bool  # True when a change is prepared and needs the UI's Confirm.


class TextSession:
    """One conversation, for one verified session (or one anonymous visitor)."""

    def __init__(self, model: Model, runtime: Any, principal: Principal | None) -> None:
        self.model = model
        self.runtime = runtime
        self.principal = principal
        self.session_id = principal.session_id if principal else None
        self.patient_id = principal.patient_id if principal else None
        self.transcript: list[Message] = []
        self.prepared_action_id: str | None = None

    def respond(self, utterance: str) -> AgentReply:
        """Handle one patient message. Always returns text, even after a model or tool fault."""
        self.transcript.append(Message("user", {"request": utterance[:MAX_UTTERANCE_CHARS]}))
        return self._run()

    def confirmation_event(self, payload: Mapping[str, Any]) -> AgentReply:
        """The trusted UI reports what happened when the patient pressed Confirm (or could not)."""
        self.transcript.append(Message("trusted_event", dict(payload)))
        self.prepared_action_id = None  # The review it was about is finished, committed or not.
        return self._run()

    def _run(self) -> AgentReply:
        reply = self._turn()
        # On every way a turn can end, including a fault: a review the conversation no longer has
        # must not stay open on a screen.
        self.prepared_action_id, failed = retire_dropped_review(
            self.model, self.runtime, self.principal, self.prepared_action_id
        )
        if failed:
            reply = AgentReply(
                f"{reply.text} {REVIEW_STILL_OPEN_NOTICE}", reply.awaiting_confirmation
            )
        return reply

    def _turn(self) -> AgentReply:
        for _ in range(MAX_ACTIONS_PER_TURN):
            try:
                action = self.model.decide(self.transcript)
            except Exception:  # A provider or adapter fault must still leave the patient a reply.
                return AgentReply(FALLBACK_REPLY, False)
            if isinstance(action, Say):
                return AgentReply(action.text or FALLBACK_REPLY, self._review_pending())
            if isinstance(action, ToolAction):
                response = self.runtime.invoke(
                    action.tool, dict(action.arguments), principal=self.principal
                ).to_dict()
                self.transcript.append(Message("tool", response))
                if action.tool == "prepare_appointment_action" and response.get("ok"):
                    self.prepared_action_id = str(
                        (response.get("data") or {}).get("pending_action_id")
                    )
                continue
            break
        return AgentReply(FALLBACK_REPLY, False)

    def _review_pending(self) -> bool:
        state = getattr(self.model, "state", None)
        return bool(state is not None and state.prepared is not None)


def retire_dropped_review(
    model: Any, runtime: Any, principal: Principal | None, pending_action_id: str | None
) -> tuple[str | None, bool]:
    """If the conversation no longer has the review it prepared, retire it in the backend.

    A request that was changed or dropped must not leave a review screen that can still commit.
    A newer preparation retires the older one itself (in the backend, in its own transaction); this
    covers the turns that end with no new preparation.

    Returns (the id still being tracked, whether the review may still be open). The id stays
    tracked after a failure, so the next turn tries again; it is cleared once the backend has
    retired it, or has said it cannot be confirmed any more. Nothing is reported as retired that
    the backend did not confirm. Only the kind of error is logged, never its text or any id.
    """
    state = getattr(model, "state", None)
    if pending_action_id is None or state is None or state.prepared is not None:
        return pending_action_id, False
    discard = getattr(runtime, "discard_pending_action", None)
    if discard is None or principal is None:
        return None, False
    try:
        discard(principal, pending_action_id)
    except (AuthenticationRequiredError, SessionExpiredError):
        return None, False  # The session is over, and with it every review it held.
    except Exception as error:
        logger.warning(
            "could not retire a dropped review", extra={"error_type": type(error).__name__}
        )
        return pending_action_id, True
    return None, False


REVIEW_STILL_OPEN_NOTICE = (
    "I couldn't cancel the earlier review on my side, so please don't confirm it in the app."
)


FALLBACK_REPLY = (
    "I'm sorry, I couldn't work that out. Could you tell me whether you'd like to book, move, "
    "or cancel an appointment, and which clinic and date?"
)


class ConversationRegistry:
    """Keeps each session's conversation apart.

    A conversation is looked up by the verified session's id, so two sessions can never share
    state, and it is used only for the patient who created it: if the session now belongs to a
    different patient, or to none, the old conversation is thrown away, never reused. Idle
    conversations expire and the total is bounded.
    """

    def __init__(
        self,
        make_model: Callable[[], Model],
        *,
        max_sessions: int = MAX_SESSIONS,
        idle_seconds: float = IDLE_SECONDS,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._make_model = make_model
        self._max = max_sessions
        self._idle = idle_seconds
        self._timer = timer
        self._sessions: OrderedDict[Any, tuple[TextSession, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._sessions)

    def session_for(
        self, runtime: Any, principal: Principal | None, *, anonymous_id: str | None = None
    ) -> TextSession:
        key = ("session", principal.session_id) if principal else ("anonymous", anonymous_id)
        if key[1] is None:
            raise ValueError("an anonymous conversation needs an anonymous id")
        now = self._timer()
        self._expire(now)
        held = self._sessions.get(key)
        wanted_patient = principal.patient_id if principal else None
        if held is not None and held[0].patient_id == wanted_patient:
            session = held[0]
            session.runtime = runtime
            session.principal = principal
        else:
            session = TextSession(self._make_model(), runtime, principal)
        self._sessions[key] = (session, now)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return session

    def end(self, principal: Principal | None, *, anonymous_id: str | None = None) -> None:
        """Forget a conversation, for example at sign-out."""
        key = ("session", principal.session_id) if principal else ("anonymous", anonymous_id)
        self._sessions.pop(key, None)

    def _expire(self, now: float) -> None:
        for key in [k for k, (_, seen) in self._sessions.items() if now - seen > self._idle]:
            del self._sessions[key]


def new_anonymous_id() -> str:
    return uuid.uuid4().hex
