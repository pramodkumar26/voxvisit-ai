"""Voice mode, phase 3.3: hand a prepared review to the patient's browser, and hear what happened.

The voice session can prepare a review and nothing more. When it has, this session asks the backend
for a single-use ticket (see `app.services.review_handoffs`), shows the patient a link that carries
it in the URL *fragment*, and waits. The patient signs in in the browser with their own PIN and
presses Confirm or Decline there; the existing, authenticated confirmation endpoint does the rest.

What this module does and does not do:

- It never commits, confirms, or asks for anything to be committed. It has no route to the
  confirmation service and no way to call the confirmation endpoint.
- It learns the outcome only from the database, through `handoff_status`, which reports COMMITTED
  only when the confirmation service itself marked the browser's review as used. A patient saying
  "yes", the model, a browser claiming success: none of these count.
- Only a verified commit is announced as a booking, and only then is the conversation told (as the
  trusted event the orchestrator is built to receive). Every other outcome says nothing was changed.
- If the patient speaks again, or the session ends, the ticket and any review it opened in the
  browser are retired first: a review the conversation has moved on from must stop working.

The ticket's code is shown once on the terminal and handed to the system browser, never logged,
never stored here, and never sent to a model. The session token stays inside the parent class.
"""

import logging
import uuid
import webbrowser
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import quote

from app.auth.service import AuthService
from app.core.errors import ApplicationError, AuthenticationRequiredError, SessionExpiredError
from app.evaluation.availability_notes import short_day
from app.evaluation.conversation import AgentReply
from app.evaluation.graph_service import GraphConversations
from app.services import review_handoffs as handoffs
from app.voice.lg5f_session import SESSION_ENDED_REPLY, GraphVoiceSession

logger = logging.getLogger("voxvisit.voice")

REVIEW_PATH = "/voice-review"
DEFAULT_BASE_URL = "http://localhost:8000"

NOT_COMMITTED = "Okay, nothing was changed."
EXPIRED = "That review expired, so nothing was changed. Tell me again if you'd like to continue."
UNAVAILABLE = "That review is no longer available, so nothing was changed."
UNVERIFIED = (
    "Your session ended before I could check the result. Please look at your appointments "
    "in the app to see whether the change was made."
)
COMMITTED_FALLBACK = {
    "BOOKED": "Your appointment was booked, as confirmed in the app.",
    "RESCHEDULED": "Your appointment was rescheduled, as confirmed in the app.",
    "CANCELLED": "Your appointment was cancelled, as confirmed in the app.",
}
NOT_READY = (
    "(I could not prepare the browser link. Nothing was changed, and the review cannot be "
    "confirmed from here. Ask me again.)"
)


class BrowserReviewSession(GraphVoiceSession):
    """The LG-5F voice session, with a browser to confirm in."""

    def __init__(
        self,
        conversations: GraphConversations,
        token: str,
        *,
        engine,
        auth: AuthService,
        patient_code: str,
        base_url: str = DEFAULT_BASE_URL,
        open_browser: Callable[[str], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(conversations, token)
        self._engine = engine
        self._auth = auth
        self._patient_code = patient_code
        self._base_url = base_url.rstrip("/")
        self._open_browser = open_browser
        self._clock = clock or (lambda: datetime.now(UTC))
        self._handoff: uuid.UUID | None = None

    # -- what the voice loop asks of it ---------------------------------------------------------

    @property
    def waiting(self) -> bool:
        """Whether a review has been handed to the browser and its outcome is not yet known."""
        return self._handoff is not None and not self.ended

    def respond(self, utterance: str) -> AgentReply:
        """A new turn. The conversation has moved on, so first finish with the old review: take its
        outcome if it already has one, otherwise retire it."""
        earlier = self._settle()
        reply = super().respond(utterance)
        if earlier:
            return AgentReply(f"{earlier} {reply.text}", reply.awaiting_confirmation)
        return reply

    def review_notice(self, reply: AgentReply) -> str | None:
        """After a turn that prepared a review: the link to confirm it, or None for no review."""
        if not reply.awaiting_confirmation or self.ended:
            return None
        try:
            ticket = handoffs.create_handoff(
                self._engine,
                principal=self._auth.authenticate(self._token),
                request_id=uuid.uuid4(),
                now=self._clock(),
            )
        except (AuthenticationRequiredError, SessionExpiredError):
            self.ended = True
            return f"({SESSION_ENDED_REPLY})"
        except Exception as error:
            logger.warning("no review link", extra={"error_type": type(error).__name__})
            return NOT_READY
        self._handoff = ticket.handoff_id
        link = f"{self._base_url}{REVIEW_PATH}#h={quote(ticket.code)}&p={quote(self._patient_code)}"
        opened = self._open(link)
        minutes = handoffs.HANDOFF_MINUTES
        return (
            "(A change is ready for review. Nothing is booked until you press Confirm in your "
            f"browser.\n  {'Opened in your browser' if opened else 'Open this link'}: {link}\n"
            f"  It works once, for up to {minutes} minutes, and asks you to sign in with your PIN.)"
        )

    def poll_outcome(self) -> str | None:
        """The verified outcome of the review in the browser, as words to say; None while it waits.

        Announces a booking only when the database says the confirmation service committed it."""
        if self._handoff is None or self.ended:
            return None
        try:
            status = handoffs.handoff_status(
                self._engine,
                principal=self._auth.authenticate(self._token),
                handoff_id=self._handoff,
                now=self._clock(),
            )
        except (AuthenticationRequiredError, SessionExpiredError):
            self.ended, self._handoff = True, None
            return UNVERIFIED
        except ApplicationError:
            self._handoff = None
            return UNAVAILABLE
        except Exception as error:  # The database blinked: try again on the next poll.
            logger.warning("review status unavailable", extra={"error_type": type(error).__name__})
            return None
        if status.state in (handoffs.WAITING, handoffs.IN_BROWSER):
            return None
        self._handoff = None
        if status.state == handoffs.COMMITTED:
            return self._announce_commit(status)
        return {
            handoffs.NOT_COMMITTED: NOT_COMMITTED,
            handoffs.EXPIRED: EXPIRED,
        }.get(status.state, UNAVAILABLE)

    def end_review(self) -> None:
        """The session is ending: whatever was handed to the browser stops working."""
        if self.ended:
            return
        try:
            handoffs.supersede_open_handoffs(
                self._engine,
                principal=self._auth.authenticate(self._token),
                request_id=uuid.uuid4(),
                now=self._clock(),
            )
        except Exception as error:  # Cleanup must never stop a sign-out.
            logger.warning("review not retired", extra={"error_type": type(error).__name__})
        self._handoff = None

    # -- inside --------------------------------------------------------------------------------

    def _settle(self) -> str | None:
        """Before a new turn: an outcome already waiting is announced; otherwise the review is
        retired so the browser cannot confirm what the conversation has left behind."""
        if self._handoff is None:
            return None
        outcome = self.poll_outcome()
        if outcome is not None or self._handoff is None:
            return outcome
        self.end_review()
        return None

    def _announce_commit(self, status: handoffs.HandoffStatus) -> str:
        """A verified commit: tell the conversation (a trusted event), say what it said, and add
        what was committed from the same verified record."""
        fallback = COMMITTED_FALLBACK[status.result or "BOOKED"]
        try:
            said = self._conversations.event(
                self._token, {"type": "APPOINTMENT_COMMITTED", "result": status.result}
            ).text
        except Exception as error:
            logger.warning("event not delivered", extra={"error_type": type(error).__name__})
            said = fallback
        return f"{said} {_what_was_committed(status)}".strip()

    def _open(self, link: str) -> bool:
        if self._open_browser is None:
            return False
        try:
            return bool(self._open_browser(link))
        except Exception:
            return False


def _what_was_committed(status: handoffs.HandoffStatus) -> str:
    summary = status.summary
    shown = (summary.proposed or summary.current) if summary else None
    if shown is None:
        return ""
    day = short_day(shown.date)
    return f"That is with {shown.doctor} at {shown.clinic} on {day} at {shown.time}."


def system_browser(link: str) -> bool:
    """Hand the link to the operating system's browser; the page's own sign-in does the rest."""
    return webbrowser.open(link)
