"""Retiring pending actions so an outdated confirmation can never commit.

A pending action is a patient's review screen waiting for Confirm. When the patient changes their
mind, a screen they have not closed must stop working, or pressing Confirm on it would commit a
change they no longer want. Retiring marks the action INVALIDATED, the status confirmation already
refuses (`CONFIRMATION_INVALIDATED`), in the same transaction as the event that made it stale.

Two things retire pending actions:

- **A newer preparation** retires every other ACTIVE pending action of the same session, in the
  transaction that stores the new one (see `pending_actions`). A session has one live review at a
  time. Other sessions, and other patients, are never touched.
- **A discard** retires one named pending action of the caller's own session, for when the patient
  drops a request and nothing newer replaces it.

Both take the session row lock first, as confirmation does, so a retirement and a confirmation of
the same session cannot interleave: the confirmation either commits before (and the action is then
CONSUMED, which retiring leaves alone) or finds the action INVALIDATED and commits nothing. Retiring
only ever removes the ability to confirm. It creates, moves, and cancels nothing, and it is not
reachable from the agent's tools: only trusted application code calls it.
"""

import uuid
from datetime import datetime

from sqlalchemy import Connection, Engine, update

from app.auth.service import Principal
from app.db.models import PendingAction
from app.services.action_support import lock_patient_session, record_success
from app.services.write_access import write_transaction

__all__ = ["discard_pending_action", "retire_active_pending_actions"]


def retire_active_pending_actions(
    connection: Connection, session_id: uuid.UUID, now: datetime
) -> int:
    """Retire every ACTIVE pending action of one session. Returns how many were retired.

    The caller holds the session row lock. CONSUMED, EXPIRED, and already INVALIDATED actions keep
    their status.
    """
    result = connection.execute(
        update(PendingAction)
        .where(PendingAction.session_id == session_id, PendingAction.status == "ACTIVE")
        .values(status="INVALIDATED", updated_at=now)
    )
    return result.rowcount


def discard_pending_action(
    engine: Engine,
    *,
    principal: Principal,
    pending_action_id: uuid.UUID,
    request_id: uuid.UUID,
    now: datetime,
) -> bool:
    """Retire one pending action of the caller's own session. True if it was retired.

    An action of another session, an unknown one, and one that is no longer ACTIVE all give False
    and change nothing, and cannot be told apart.
    """
    with write_transaction(engine) as connection:
        lock_patient_session(connection, principal, now)
        retired = connection.execute(
            update(PendingAction)
            .where(
                PendingAction.pending_action_id == pending_action_id,
                PendingAction.session_id == principal.session_id,
                PendingAction.status == "ACTIVE",
            )
            .values(status="INVALIDATED", updated_at=now)
        ).rowcount
        if retired:
            record_success(
                connection,
                "APPOINTMENT_ACTION_DISCARDED",
                principal=principal,
                now=now,
                request_id=request_id,
                resource_type="PENDING_ACTION",
                resource_id=pending_action_id,
                details={},
            )
        return bool(retired)
