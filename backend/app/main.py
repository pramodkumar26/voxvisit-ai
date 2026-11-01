import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.appointment_actions import confirm_router, prepare_router
from app.api.routes.auth import router as auth_router
from app.api.routes.fhir_like import router as fhir_like_router
from app.api.routes.health import router as health_router
from app.api.routes.scheduling import router as scheduling_router
from app.auth.dependencies import CSRF_HEADER
from app.auth.passwords import create_hasher
from app.auth.service import AuthService, Clock, utc_now
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import create_database_engine

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    clock: Clock = utc_now,
    hasher: PasswordHasher | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        application.state.db_engine = create_database_engine(resolved_settings)
        application.state.auth_service = AuthService(
            application.state.db_engine,
            secret_key=_auth_secret_key(resolved_settings),
            hasher=hasher or create_hasher(),
            clock=clock,
        )
        logger.info("application started")
        try:
            yield
        finally:
            application.state.db_engine.dispose()
            logger.info("application stopped")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.clock = clock
    register_exception_handlers(application)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", CSRF_HEADER, "Idempotency-Key"],
        max_age=600,
    )
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(scheduling_router)
    application.include_router(prepare_router)
    application.include_router(confirm_router)
    application.include_router(fhir_like_router)
    return application


def _auth_secret_key(settings: Settings) -> bytes:
    if settings.auth_secret_key is not None:
        return settings.auth_secret_key.get_secret_value().encode()
    logger.warning("AUTH_SECRET_KEY is not set; using a temporary key that resets on restart")
    return secrets.token_bytes(32)


app = create_app()
