"""The trusted context the runtime attaches to every call, outside model-generated arguments."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import Engine

from app.auth.service import Principal


@dataclass(frozen=True)
class ToolContext:
    engine: Engine
    principal: Principal | None  # The verified session; never taken from the model.
    now: datetime
    request_id: UUID
    idempotency_key: str  # Runtime-generated; reused only for retries of the same logical call.
