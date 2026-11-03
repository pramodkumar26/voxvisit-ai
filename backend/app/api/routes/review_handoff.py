"""The browser side of a voice-to-browser review handoff.

Three routes, all for a signed-in patient (session cookie, allowed Origin and the session-bound
CSRF token on every POST): claim the review a voice session prepared, read the review waiting in
this browser session, and decline it. Confirming is not here: the browser uses the one existing
confirmation endpoint, unchanged, for the review it now owns.

The ticket's code is carried in the request body, never in a URL or a header, and every failure to
redeem it gets one answer (see `HandoffInvalidError`).
"""

import logging
from collections.abc import Callable
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import EngineDep, NowDep
from app.auth.dependencies import client_address, get_auth_service, request_id_of, require_actor
from app.auth.permissions import Permission
from app.auth.service import PATIENT, Principal
from app.core.errors import ApplicationError, ResourceNotAvailableError
from app.models.appointment_actions import PendingActionView
from app.services import review_handoffs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/me/voice-handoffs", tags=["review-handoff"])

PatientReviewer = Annotated[
    Principal, Depends(require_actor(PATIENT, Permission.MANAGE_OWN_APPOINTMENTS))
]
T = TypeVar("T")


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=20, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class DeclineResult(BaseModel):
    declined: bool


def _audited(request: Request, principal: Principal, operation: str, work: Callable[[], T]) -> T:
    """Run the service call; a refused attempt leaves a sanitized audit event (never the code)."""
    try:
        return work()
    except ApplicationError as exc:
        try:
            get_auth_service(request).record_event(
                "REVIEW_HANDOFF_REJECTED",
                "DENIED",
                request_id=request_id_of(request),
                network_address=client_address(request),
                principal=principal,
                details={"operation": operation, "code": exc.code},
            )
        except Exception:
            logger.warning("could not record a rejected review handoff")
        raise


@router.post("/claim", response_model=PendingActionView)
def claim(
    body: ClaimRequest,
    principal: PatientReviewer,
    request: Request,
    response: Response,
    engine: EngineDep,
    now: NowDep,
) -> PendingActionView:
    """Redeem a voice session's ticket. Prepares the review in this browser session's name."""
    response.headers["Cache-Control"] = "no-store"
    return _audited(
        request,
        principal,
        "claim",
        lambda: review_handoffs.claim_handoff(
            engine,
            principal=principal,
            code=body.code,
            request_id=request_id_of(request),
            now=now,
        ),
    )


@router.get("/review", response_model=PendingActionView)
def review(
    principal: PatientReviewer, response: Response, engine: EngineDep, now: NowDep
) -> PendingActionView:
    """The review waiting for Confirm in this browser session (so a reload loses nothing)."""
    response.headers["Cache-Control"] = "no-store"
    found = review_handoffs.current_review(engine, principal=principal, now=now)
    if found is None:
        raise ResourceNotAvailableError()
    return found


@router.post("/decline", response_model=DeclineResult)
def decline(
    principal: PatientReviewer,
    request: Request,
    response: Response,
    engine: EngineDep,
    now: NowDep,
) -> DeclineResult:
    """Decline the waiting review. Nothing is committed; the review simply stops working."""
    response.headers["Cache-Control"] = "no-store"
    declined = _audited(
        request,
        principal,
        "decline",
        lambda: review_handoffs.decline_review(
            engine, principal=principal, request_id=request_id_of(request), now=now
        ),
    )
    return DeclineResult(declined=declined)
