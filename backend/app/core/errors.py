import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.headers = headers
        # A stable machine-readable refinement of the code, such as CONFIRMATION_EXPIRED.
        self.reason = reason

    def body(self) -> dict[str, str]:
        detail = {"code": self.code, "message": self.message}
        if self.reason:
            detail["reason"] = self.reason
        return detail


class DatabaseUnavailableError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="DATABASE_UNAVAILABLE",
            message="Database connectivity check failed.",
            status_code=503,
        )


class AuthenticationRequiredError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="AUTH_REQUIRED",
            message="Authentication is required.",
            status_code=401,
        )


class SessionExpiredError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="SESSION_EXPIRED",
            message="Your session has expired. Please sign in again.",
            status_code=401,
        )


class InvalidCredentialsError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="INVALID_CREDENTIALS",
            message="The credentials could not be verified.",
            status_code=401,
        )


class ForbiddenError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="FORBIDDEN",
            message="You do not have access to this resource.",
            status_code=403,
        )


class CsrfRejectedError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="CSRF_FAILED",
            message="The request could not be verified.",
            status_code=403,
        )


class RateLimitedError(ApplicationError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            code="RATE_LIMITED",
            message="Too many attempts. Please try again later.",
            status_code=429,
            headers={"Retry-After": str(max(1, retry_after_seconds))},
        )


class InvalidArgumentError(ApplicationError):
    def __init__(self, message: str) -> None:
        super().__init__(code="INVALID_ARGUMENT", message=message, status_code=422)


class DependencyUnavailableError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="DEPENDENCY_UNAVAILABLE",
            message="The service is temporarily unavailable. Please try again.",
            status_code=503,
        )


class ResourceNotAvailableError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="RESOURCE_NOT_AVAILABLE",
            message="That resource is unavailable or cannot be accessed.",
            status_code=404,
        )


class ConflictError(ApplicationError):
    """Current backend state conflicts with the request; the caller must refresh and review."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(code="CONFLICT", message=message, status_code=409, reason=reason)


class HandoffInvalidError(ApplicationError):
    """One answer for every way a confirmation link can fail, so none can be told from another."""

    def __init__(self) -> None:
        super().__init__(
            code="HANDOFF_INVALID",
            message=(
                "This confirmation link is not valid, was already used, or has expired. "
                "Ask the assistant to prepare the change again."
            ),
            status_code=404,
        )


class SlotUnavailableError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="SLOT_UNAVAILABLE",
            message="That time is no longer available. Please search again.",
            status_code=409,
        )


class SelfServiceCutoffError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="SELF_SERVICE_CUTOFF",
            message=(
                "This appointment can no longer be changed online because it has started or "
                "is about to start. Simulated staff assistance is available."
            ),
            status_code=409,
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def handle_application_error(
        request: Request,
        exc: ApplicationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.body()},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        fields = sorted(
            {".".join(str(part) for part in error["loc"][1:]) for error in exc.errors()}
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "The request was not valid.",
                    "fields": fields,
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del request
        logger.error(
            "unexpected request failure",
            extra={"error_type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "The request could not be completed.",
                }
            },
        )
