"""FastAPI dependencies for public, patient, and staff access, ownership, and CSRF checks.

Mutating requests (any method other than GET, HEAD, or OPTIONS) always require an allowed Origin
and the session-bound CSRF token, so future mutation endpoints are protected by default.
"""

import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request

from app.auth.permissions import Permission, has_permission
from app.auth.service import PATIENT, STAFF, AuthService, Principal
from app.auth.tokens import tokens_match
from app.core.config import Settings
from app.core.errors import (
    ApplicationError,
    CsrfRejectedError,
    ForbiddenError,
    ResourceNotAvailableError,
)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
CSRF_HEADER = "X-CSRF-Token"


def get_settings_from(request: Request) -> Settings:
    return request.app.state.settings


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def request_id_of(request: Request) -> uuid.UUID:
    if not hasattr(request.state, "request_id"):
        request.state.request_id = uuid.uuid4()
    return request.state.request_id


def client_address(request: Request) -> str | None:
    settings = get_settings_from(request)
    if settings.auth_trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else None


def session_token_of(request: Request) -> str | None:
    return request.cookies.get(get_settings_from(request).session_cookie_name)


def _reject(
    request: Request,
    error: ApplicationError,
    event_type: str,
    principal: Principal | None,
    details: dict[str, str],
) -> ApplicationError:
    get_auth_service(request).record_event(
        event_type,
        "DENIED",
        request_id=request_id_of(request),
        network_address=client_address(request),
        principal=principal,
        details=details,
    )
    return error


def enforce_origin(request: Request, principal: Principal | None = None) -> None:
    """Require an allowed Origin. Also protects unauthenticated login against login CSRF."""
    origin = request.headers.get("origin")
    if origin is None or origin.lower() not in get_settings_from(request).allowed_origins:
        raise _reject(request, CsrfRejectedError(), "CSRF_REJECTED", principal, {"check": "origin"})


def enforce_csrf(request: Request, principal: Principal) -> None:
    if request.method in SAFE_METHODS:
        return
    enforce_origin(request, principal)
    if not tokens_match(principal.csrf_token, request.headers.get(CSRF_HEADER)):
        raise _reject(request, CsrfRejectedError(), "CSRF_REJECTED", principal, {"check": "token"})


def optional_principal(request: Request) -> Principal | None:
    """Public endpoints: a missing, invalid, or expired session simply means anonymous."""
    token = session_token_of(request)
    if not token:
        return None
    try:
        return get_auth_service(request).authenticate(token)
    except ApplicationError:
        return None


def require_actor(
    actor_type: str | None = None, permission: Permission | None = None
) -> Callable[[Request], Principal]:
    def dependency(request: Request) -> Principal:
        principal = get_auth_service(request).authenticate(session_token_of(request))
        enforce_csrf(request, principal)
        if (actor_type and principal.actor_type != actor_type) or (
            permission and not has_permission(principal.actor_type, permission)
        ):
            raise _reject(
                request,
                ForbiddenError(),
                "ACCESS_DENIED",
                principal,
                {"required": actor_type or str(permission)},
            )
        return principal

    return dependency


def require_any_permission(*permissions: Permission) -> Callable[[Request], Principal]:
    """A verified session whose role holds at least one of the permissions (any role)."""

    def dependency(request: Request) -> Principal:
        principal = get_auth_service(request).authenticate(session_token_of(request))
        enforce_csrf(request, principal)
        if not any(has_permission(principal.actor_type, permission) for permission in permissions):
            raise _reject(
                request,
                ForbiddenError(),
                "ACCESS_DENIED",
                principal,
                {"required": "|".join(str(permission) for permission in permissions)},
            )
        return principal

    return dependency


def require_permission(permission: Permission) -> Callable[[Request], Principal]:
    return require_actor(permission=permission)


require_session = require_actor()
require_patient = require_actor(PATIENT)
require_staff = require_actor(STAFF)

AnyPrincipal = Annotated[Principal, Depends(require_session)]
PatientPrincipal = Annotated[Principal, Depends(require_patient)]
StaffPrincipal = Annotated[Principal, Depends(require_staff)]
OptionalPrincipal = Annotated[Principal | None, Depends(optional_principal)]


def ensure_owns_patient(principal: Principal, patient_id: uuid.UUID) -> None:
    """Patient endpoints derive identity from the session; this guards any ID that is supplied.

    A mismatch answers exactly like a missing record so another patient's data is never confirmed.
    """
    if principal.patient_id != patient_id:
        raise ResourceNotAvailableError()
