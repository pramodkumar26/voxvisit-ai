"""Moving a prepared review from a voice session to the patient's own browser.

A voice session can prepare a review (a pending action) and nothing more; only the authenticated
UI can confirm it, and a pending action belongs to the one session that prepared it. This module
carries a review between the two without moving a credential and without changing who owns what:

1. **The voice session asks for a ticket** (`create_handoff`): a random code, shown once, of which
   only the SHA-256 is stored. It names the patient and the review, and lasts five minutes at most
   (never longer than the review itself).
2. **The browser signs in on its own** with the patient's PIN, like any other browser, and presents
   the code (`claim_handoff`). The ticket alone is worth nothing: it must be redeemed by an
   authenticated session of the *same patient*, once. The backend then prepares an equivalent
   review in the browser's own session through the ordinary prepare service (so the slot is
   checked again, and the review is owned by, expires with and can be confirmed only by that
   session) and retires the voice session's review.
3. **The browser confirms or declines** through the existing endpoints. Confirmation is unchanged.
4. **The voice session reads the outcome** (`handoff_status`), derived only from the database: a
   review counts as committed only when the confirmation service marked it CONSUMED, in the same
   transaction as the appointment change. Nothing the browser or the voice client says counts.

If the patient speaks again, or the voice session ends, the ticket and any review it opened in the
browser are retired (`supersede_open_handoffs`): a review the conversation has moved on from must
stop working, as it does within one session.

This module never imports the commit service. It prepares, retires and reads.
"""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, Engine, text

from app.auth.service import Principal
from app.auth.tokens import hash_token
from app.core.errors import HandoffInvalidError, ResourceNotAvailableError
from app.models.appointment_actions import (
    ActionSummary,
    PendingActionView,
    PrepareBook,
    PrepareCancel,
    PrepareRequest,
    PrepareReschedule,
)
from app.services.action_support import lock_patient_session, record_success
from app.services.pending_action_retirement import discard_pending_action
from app.services.pending_actions import prepare_appointment_action
from app.services.write_access import write_transaction

__all__ = [
    "HANDOFF_MINUTES",
    "CreatedHandoff",
    "HandoffStatus",
    "claim_handoff",
    "create_handoff",
    "current_review",
    "decline_review",
    "handoff_status",
    "supersede_open_handoffs",
]

HANDOFF_MINUTES = 5

# What a handoff can be, as the voice session sees it.
WAITING = "WAITING"  # Issued; the browser has not claimed it yet.
IN_BROWSER = "IN_BROWSER"  # Claimed; the review awaits Confirm or Decline in the browser.
COMMITTED = "COMMITTED"  # The confirmation service committed it.
NOT_COMMITTED = "NOT_COMMITTED"  # Declined or retired; nothing was changed.
EXPIRED = "EXPIRED"  # The ticket or the review lapsed unused; nothing was changed.
SUPERSEDED = "SUPERSEDED"  # The conversation moved on, or the browser could not open it.

RESULT_OF = {"BOOK": "BOOKED", "RESCHEDULE": "RESCHEDULED", "CANCEL": "CANCELLED"}


def _lapsed(alias: str) -> str:
    """SQL: the pending action `alias` is past its expiry, by the rule confirmation itself uses."""
    return (
        f"clock_timestamp() >= LEAST({alias}.expires_at,"
        f" {alias}.created_at + interval '10 minutes')"
    )


@dataclass(frozen=True)
class CreatedHandoff:
    handoff_id: uuid.UUID
    code: str  # Shown once; only its hash is stored.
    expires_at: datetime


@dataclass(frozen=True)
class HandoffStatus:
    state: str
    handoff_id: uuid.UUID
    action: str | None = None  # BOOK, RESCHEDULE or CANCEL
    result: str | None = None  # BOOKED, RESCHEDULED or CANCELLED: set only when COMMITTED
    summary: ActionSummary | None = None


# --- the voice side ---------------------------------------------------------------------------


def create_handoff(
    engine: Engine, *, principal: Principal, request_id: uuid.UUID, now: datetime
) -> CreatedHandoff:
    """A ticket for the one live review of the caller's own session. Replaces any earlier ticket."""
    code = secrets.token_urlsafe(32)
    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now)
        review = connection.execute(
            text(
                "SELECT p.pending_action_id, p.expires_at FROM pending_actions p"
                " WHERE p.session_id = :session AND p.status = 'ACTIVE' AND NOT ("
                + _lapsed("p")
                + ")"
                " ORDER BY p.created_at DESC LIMIT 1"
            ),
            {"session": principal.session_id},
        ).one_or_none()
        if review is None:
            raise ResourceNotAvailableError()  # Nothing is waiting for a confirmation.
        _supersede(connection, principal.session_id, now)
        created = connection.execute(
            text(
                "INSERT INTO review_handoffs (code_hash, patient_id, voice_session_id,"
                " voice_pending_action_id, expires_at)"
                " VALUES (:hash, :patient, :session, :review,"
                " LEAST(clock_timestamp() + make_interval(mins => :minutes), :review_expiry))"
                " RETURNING handoff_id, expires_at"
            ),
            {
                "hash": hash_token(code),
                "patient": principal.patient_id,
                "session": principal.session_id,
                "review": review.pending_action_id,
                "minutes": HANDOFF_MINUTES,
                "review_expiry": review.expires_at,
            },
        ).one()
        record_success(
            connection,
            "REVIEW_HANDOFF_CREATED",
            principal=principal,
            now=now,
            request_id=request_id,
            resource_type="REVIEW_HANDOFF",
            resource_id=created.handoff_id,
            details={},
        )
        return CreatedHandoff(created.handoff_id, code, created.expires_at)


def handoff_status(
    engine: Engine, *, principal: Principal, handoff_id: uuid.UUID, now: datetime
) -> HandoffStatus:
    """Where a handoff of the caller's own session stands, from the database alone."""
    with write_transaction(engine) as connection:  # Rechecks the session; reads only.
        lock_patient_session(connection, principal, now, lock=False)
        row = connection.execute(
            text(
                "SELECT h.superseded_at, h.claimed_at, h.expires_at, h.expires_at <= "
                "clock_timestamp() AS ticket_lapsed, p.pending_action_id, p.action_type, p.status,"
                " p.summary, p.session_id AS review_session, h.claimed_session_id,"
                " (p.pending_action_id IS NOT NULL AND " + _lapsed("p") + ") AS review_lapsed,"
                " v.action_type AS voice_action_type, v.summary AS voice_summary"
                " FROM review_handoffs h"
                " JOIN pending_actions v ON v.pending_action_id = h.voice_pending_action_id"
                " LEFT JOIN pending_actions p ON p.pending_action_id = h.browser_pending_action_id"
                " WHERE h.handoff_id = :id AND h.voice_session_id = :session"
                " AND h.patient_id = :patient"
            ),
            {
                "id": handoff_id,
                "session": principal.session_id,
                "patient": principal.patient_id,
            },
        ).one_or_none()
    if row is None:
        raise ResourceNotAvailableError()  # Not this session's: indistinguishable from unknown.
    action = row.action_type or row.voice_action_type
    summary = ActionSummary.model_validate(row.summary or row.voice_summary)
    if row.superseded_at is not None:
        return HandoffStatus(SUPERSEDED, handoff_id, action, None, summary)
    if row.claimed_at is None:
        state = EXPIRED if row.ticket_lapsed else WAITING
        return HandoffStatus(state, handoff_id, action, None, summary)
    if row.pending_action_id is None or row.review_session != row.claimed_session_id:
        return HandoffStatus(IN_BROWSER if not row.ticket_lapsed else EXPIRED, handoff_id, action)
    if row.status == "CONSUMED":
        # Only the confirmation service sets CONSUMED, in the transaction that changed the
        # appointment: this is the one thing that counts as a verified commit.
        return HandoffStatus(COMMITTED, handoff_id, action, RESULT_OF[row.action_type], summary)
    if row.status == "INVALIDATED":
        return HandoffStatus(NOT_COMMITTED, handoff_id, action, None, summary)
    if row.status == "EXPIRED" or row.review_lapsed:
        return HandoffStatus(EXPIRED, handoff_id, action, None, summary)
    return HandoffStatus(IN_BROWSER, handoff_id, action, None, summary)


def supersede_open_handoffs(
    engine: Engine, *, principal: Principal, request_id: uuid.UUID, now: datetime
) -> int:
    """The conversation has moved on (or ended): retire this session's open tickets and any review
    they opened in a browser. Returns how many tickets were retired."""
    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now)
        retired = _supersede(connection, principal.session_id, now)
        if retired:
            record_success(
                connection,
                "REVIEW_HANDOFF_SUPERSEDED",
                principal=principal,
                now=now,
                request_id=request_id,
                resource_type="REVIEW_HANDOFF",
                resource_id=principal.session_id,
                details={"tickets": str(retired)},
            )
        return retired


def _supersede(connection: Connection, voice_session_id: uuid.UUID, now: datetime) -> int:
    """Close every open ticket of a voice session; retire the reviews they opened in browsers.

    A review already CONSUMED, EXPIRED or INVALIDATED keeps its status: retiring only ever removes
    the ability to confirm."""
    open_tickets = connection.execute(
        text(
            "UPDATE review_handoffs SET superseded_at = :now"
            " WHERE voice_session_id = :session AND superseded_at IS NULL"
            " RETURNING browser_pending_action_id"
        ),
        {"now": now, "session": voice_session_id},
    ).all()
    browser_reviews = [
        r.browser_pending_action_id for r in open_tickets if r.browser_pending_action_id
    ]
    if browser_reviews:
        connection.execute(
            text(
                "UPDATE pending_actions SET status = 'INVALIDATED', updated_at = :now"
                " WHERE pending_action_id = ANY(:ids) AND status = 'ACTIVE'"
            ),
            {"now": now, "ids": browser_reviews},
        )
    return len(open_tickets)


# --- the browser side -------------------------------------------------------------------------


def claim_handoff(
    engine: Engine,
    *,
    principal: Principal,
    code: str,
    request_id: uuid.UUID,
    now: datetime,
) -> PendingActionView:
    """Redeem a ticket from the browser's own signed-in session. Single use.

    Every way of failing (unknown code, another patient's, already used by someone else, expired,
    replaced, the voice session gone or the review no longer live) gives the same answer. The same
    browser session presenting the same code again gets its own review back, not a second one.
    """
    digest = hash_token(code)
    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now)
        ticket = connection.execute(
            text(
                "SELECT h.handoff_id, h.patient_id, h.claimed_session_id, h.superseded_at,"
                " h.expires_at > clock_timestamp() AS ticket_live, h.voice_pending_action_id,"
                " h.browser_pending_action_id, v.action_type, v.target_id, v.expected_versions,"
                " v.status = 'ACTIVE' AND NOT (" + _lapsed("v") + ") AS review_live,"
                " s.status = 'ACTIVE' AND s.idle_expires_at > :now"
                " AND s.absolute_expires_at > :now AS voice_session_live"
                " FROM review_handoffs h"
                " JOIN pending_actions v ON v.pending_action_id = h.voice_pending_action_id"
                " JOIN auth_sessions s ON s.session_id = h.voice_session_id"
                " WHERE h.code_hash = :hash FOR UPDATE OF h"
            ),
            {"hash": digest, "now": now},
        ).one_or_none()
        if (
            ticket is None
            or ticket.patient_id != principal.patient_id
            or ticket.superseded_at is not None
            or not ticket.ticket_live
        ):
            raise HandoffInvalidError()
        if ticket.claimed_session_id is not None:
            # Already claimed: only by this same browser session, and only to get its own review
            # back while that review still waits. (Claimed but never linked means an earlier
            # attempt was interrupted; the prepare below is idempotent, so it simply finishes.)
            if ticket.claimed_session_id != principal.session_id:
                raise HandoffInvalidError()
            if ticket.browser_pending_action_id is not None:
                existing = _review_of_session(connection, principal.session_id)
                if existing is None:
                    raise HandoffInvalidError()
                return existing
        elif not ticket.review_live or not ticket.voice_session_live:
            raise HandoffInvalidError()
        else:
            connection.execute(
                text(
                    "UPDATE review_handoffs SET claimed_session_id = :browser,"
                    " claimed_at = clock_timestamp() WHERE handoff_id = :id"
                ),
                {"browser": principal.session_id, "id": ticket.handoff_id},
            )
        request = _request_of(ticket)

    # The ticket is now spent: whatever happens next, it cannot be used again. The review is
    # prepared like any other, in its own transaction, so the slot is checked afresh.
    try:
        view, _ = prepare_appointment_action(
            engine,
            principal=principal,
            request=request,
            idempotency_key=f"review-handoff:{ticket.handoff_id}",
            request_id=request_id,
            now=now,
        )
    except Exception:
        with write_transaction(engine) as connection:
            connection.execute(
                text("UPDATE review_handoffs SET superseded_at = :now WHERE handoff_id = :id"),
                {"now": now, "id": ticket.handoff_id},
            )
            _retire(connection, ticket.voice_pending_action_id, now)
        raise

    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now)
        linked = connection.execute(
            text(
                "UPDATE review_handoffs SET browser_pending_action_id = :review"
                " WHERE handoff_id = :id AND claimed_session_id = :browser"
                " AND superseded_at IS NULL RETURNING handoff_id"
            ),
            {
                "review": view.pending_action_id,
                "id": ticket.handoff_id,
                "browser": principal.session_id,
            },
        ).one_or_none()
        if linked is None:  # Replaced while it was being opened: the new review must not live.
            _retire(connection, view.pending_action_id, now)
            raise HandoffInvalidError()
        _retire(connection, ticket.voice_pending_action_id, now)  # One live review, not two.
        record_success(
            connection,
            "REVIEW_HANDOFF_CLAIMED",
            principal=principal,
            now=now,
            request_id=request_id,
            resource_type="REVIEW_HANDOFF",
            resource_id=ticket.handoff_id,
            details={"pending_action_id": str(view.pending_action_id)},
        )
    return view


def current_review(
    engine: Engine, *, principal: Principal, now: datetime
) -> PendingActionView | None:
    """The review a handoff opened in this browser session, if it is still waiting for Confirm."""
    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now, lock=False)
        return _review_of_session(connection, principal.session_id)


def decline_review(
    engine: Engine, *, principal: Principal, request_id: uuid.UUID, now: datetime
) -> bool:
    """Decline the review waiting in this browser session. Changes nothing but the review."""
    review = current_review(engine, principal=principal, now=now)
    if review is None:
        return False
    return discard_pending_action(
        engine,
        principal=principal,
        pending_action_id=review.pending_action_id,
        request_id=request_id,
        now=now,
    )


def _review_of_session(connection: Connection, session_id: uuid.UUID) -> PendingActionView | None:
    row = connection.execute(
        text(
            "SELECT p.pending_action_id, p.action_type, p.summary, p.created_at, p.expires_at"
            " FROM review_handoffs h JOIN pending_actions p"
            " ON p.pending_action_id = h.browser_pending_action_id"
            " WHERE h.claimed_session_id = :session AND p.session_id = :session"
            " AND h.superseded_at IS NULL AND p.status = 'ACTIVE' AND NOT (" + _lapsed("p") + ")"
            " ORDER BY h.claimed_at DESC LIMIT 1"
        ),
        {"session": session_id},
    ).one_or_none()
    if row is None:
        return None
    return PendingActionView(
        pending_action_id=row.pending_action_id,
        action=row.action_type,
        summary=ActionSummary.model_validate(row.summary),
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


def _retire(connection: Connection, pending_action_id: uuid.UUID, now: datetime) -> None:
    connection.execute(
        text(
            "UPDATE pending_actions SET status = 'INVALIDATED', updated_at = :now"
            " WHERE pending_action_id = :id AND status = 'ACTIVE'"
        ),
        {"now": now, "id": pending_action_id},
    )


def _request_of(ticket: Any) -> PrepareRequest:
    """The prepare request equivalent to the voice session's review, from what it stored."""
    try:
        if ticket.action_type == "BOOK":
            return PrepareBook(action="BOOK", slot_id=ticket.target_id)
        if ticket.action_type == "CANCEL":
            return PrepareCancel(action="CANCEL", appointment_id=ticket.target_id)
        if ticket.action_type == "RESCHEDULE":
            return PrepareReschedule(
                action="RESCHEDULE",
                appointment_id=ticket.target_id,
                new_slot_id=uuid.UUID(str(ticket.expected_versions["new_slot_id"])),
            )
    except (KeyError, TypeError, ValueError):
        pass
    raise HandoffInvalidError()
