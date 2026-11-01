"""Appointment actions: prepare (agent-callable) and confirm (authenticated UI only).

The two operations are on separate routers on purpose. The conversational agent's tool layer may
use only `prepare_router`; `confirm_router` is the explicit UI confirmation boundary and is never
exposed as a tool. Both require a verified patient session with the manage-own-appointments
permission, an allowed Origin, the session-bound CSRF token, and an Idempotency-Key header.
Patient identity always comes from the session, never from the request.
"""

import logging
from collections.abc import Callable
from typing import Annotated, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request, Response

from app.api.dependencies import EngineDep, NowDep
from app.auth.dependencies import client_address, get_auth_service, request_id_of, require_actor
from app.auth.permissions import Permission
from app.auth.service import PATIENT, Principal
from app.core.errors import ApplicationError
from app.models.appointment_actions import (
    AppointmentActionResult,
    ConfirmRequest,
    PendingActionView,
    PrepareRequest,
)
from app.services.appointment_transactions import confirm_appointment_action
from app.services.pending_actions import prepare_appointment_action

logger = logging.getLogger(__name__)

prepare_router = APIRouter(tags=["appointment-actions"])
confirm_router = APIRouter(tags=["ui-confirmation"])

AppointmentManager = Annotated[
    Principal, Depends(require_actor(PATIENT, Permission.MANAGE_OWN_APPOINTMENTS))
]
IdempotencyKey = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
]

T = TypeVar("T")


def _audited(request: Request, principal: Principal, operation: str, work: Callable[[], T]) -> T:
    """Run the service call and record a sanitized audit event for a rejected attempt.

    Successful attempts are audited inside the service transaction. A rejected one rolled back
    with its transaction, so its event is written separately and never blocks the response.
    """
    try:
        return work()
    except ApplicationError as exc:
        details = {"operation": operation, "code": exc.code}
        if exc.reason:
            details["reason"] = exc.reason
        try:
            get_auth_service(request).record_event(
                "APPOINTMENT_ACTION_REJECTED",
                "DENIED",
                request_id=request_id_of(request),
                network_address=client_address(request),
                principal=principal,
                details=details,
            )
        except Exception:
            logger.warning("could not record a rejected appointment action")
        raise


@prepare_router.post("/me/appointment-actions", response_model=PendingActionView)
def prepare_action(
    body: PrepareRequest,
    principal: AppointmentManager,
    idempotency_key: IdempotencyKey,
    request: Request,
    response: Response,
    engine: EngineDep,
    now: NowDep,
) -> PendingActionView:
    """Validate and store a pending action for UI confirmation. Changes no appointment or slot."""
    response.headers["Cache-Control"] = "no-store"
    view, replayed = _audited(
        request,
        principal,
        "prepare",
        lambda: prepare_appointment_action(
            engine,
            principal=principal,
            request=body,
            idempotency_key=idempotency_key,
            request_id=request_id_of(request),
            now=now,
        ),
    )
    if replayed:
        response.headers["Idempotent-Replay"] = "true"
    return view


@confirm_router.post(
    "/me/appointment-actions/{pending_action_id}/confirm", response_model=AppointmentActionResult
)
def confirm_action(
    pending_action_id: UUID,
    body: ConfirmRequest,
    principal: AppointmentManager,
    idempotency_key: IdempotencyKey,
    request: Request,
    response: Response,
    engine: EngineDep,
    now: NowDep,
) -> AppointmentActionResult:
    """The explicit UI confirmation. Commits the prepared action in one transaction."""
    response.headers["Cache-Control"] = "no-store"
    result, replayed = _audited(
        request,
        principal,
        "confirm",
        lambda: confirm_appointment_action(
            engine,
            principal=principal,
            pending_action_id=pending_action_id,
            idempotency_key=idempotency_key,
            request_id=request_id_of(request),
            now=now,
        ),
    )
    if replayed:
        response.headers["Idempotent-Replay"] = "true"
    return result
