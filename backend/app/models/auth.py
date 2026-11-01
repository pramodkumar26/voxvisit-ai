from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.auth.service import Principal


class PatientLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_code: str = Field(min_length=1, max_length=32)
    pin: SecretStr = Field(min_length=1, max_length=64)


class StaffLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: SecretStr = Field(min_length=1, max_length=256)


class SessionResponse(BaseModel):
    actor_type: Literal["PATIENT", "STAFF"]
    login: str
    display_name: str
    absolute_expires_at: datetime
    idle_expires_at: datetime
    csrf_token: str

    @classmethod
    def from_principal(cls, principal: Principal) -> "SessionResponse":
        return cls(
            actor_type=principal.actor_type,  # type: ignore[arg-type]
            login=principal.login,
            display_name=principal.display_name,
            absolute_expires_at=principal.absolute_expires_at,
            idle_expires_at=principal.idle_expires_at,
            csrf_token=principal.csrf_token,
        )
