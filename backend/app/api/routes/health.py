import logging

from fastapi import APIRouter, Request

from app.core.errors import DatabaseUnavailableError
from app.models.health import HealthResponse, ReadyResponse
from app.services.health import verify_database_connection

router = APIRouter(tags=["system"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    try:
        verify_database_connection(request.app.state.db_engine)
    except Exception as exc:
        logger.warning(
            "database readiness check failed",
            extra={"error_type": type(exc).__name__},
        )
        raise DatabaseUnavailableError from None

    return ReadyResponse(status="ready", database="available")
