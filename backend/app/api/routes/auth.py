from fastapi import APIRouter, Request, Response

from app.auth.dependencies import (
    AuthServiceDep,
    client_address,
    enforce_csrf,
    enforce_origin,
    get_settings_from,
    request_id_of,
    session_token_of,
)
from app.auth.service import PATIENT, STAFF, LoginResult
from app.core.errors import ApplicationError, InvalidCredentialsError, RateLimitedError
from app.models.auth import PatientLoginRequest, SessionResponse, StaffLoginRequest

router = APIRouter(prefix="/auth", tags=["authentication"])
NO_STORE = {"Cache-Control": "no-store"}


def _cookie_options(request: Request) -> dict:
    settings = get_settings_from(request)
    return {
        "key": settings.session_cookie_name,
        "path": "/",
        "domain": settings.auth_cookie_domain,
        "secure": settings.session_cookie_secure,
        "httponly": True,
        "samesite": settings.auth_cookie_samesite,
    }


def _finish_login(request: Request, response: Response, result: LoginResult) -> SessionResponse:
    if result.rate_limited:
        raise RateLimitedError(result.retry_after_seconds)
    if result.issued is None:
        raise InvalidCredentialsError()
    response.set_cookie(
        value=result.issued.token,
        max_age=result.issued.lifetime_seconds,
        **_cookie_options(request),
    )
    response.headers.update(NO_STORE)
    return SessionResponse.from_principal(result.issued.principal)


@router.post("/patient/login", response_model=SessionResponse)
def patient_login(
    body: PatientLoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDep,
) -> SessionResponse:
    enforce_origin(request)
    result = service.login(
        PATIENT,
        body.patient_code,
        body.pin.get_secret_value(),
        network_address=client_address(request),
        request_id=request_id_of(request),
    )
    return _finish_login(request, response, result)


@router.post("/staff/login", response_model=SessionResponse)
def staff_login(
    body: StaffLoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDep,
) -> SessionResponse:
    enforce_origin(request)
    result = service.login(
        STAFF,
        body.username,
        body.password.get_secret_value(),
        network_address=client_address(request),
        request_id=request_id_of(request),
    )
    return _finish_login(request, response, result)


@router.get("/session", response_model=SessionResponse)
def current_session(
    request: Request,
    service: AuthServiceDep,
) -> SessionResponse:
    principal = service.authenticate(session_token_of(request))
    return SessionResponse.from_principal(principal)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    service: AuthServiceDep,
) -> Response:
    """End the current session. Without a valid session there is nothing to protect or end."""
    response = Response(status_code=204, headers=NO_STORE)
    try:
        principal = service.authenticate(session_token_of(request))
    except ApplicationError:
        principal = None
    if principal is not None:
        enforce_csrf(request, principal)
        service.logout(principal, request_id=request_id_of(request))
    response.delete_cookie(**{k: v for k, v in _cookie_options(request).items()})
    return response
