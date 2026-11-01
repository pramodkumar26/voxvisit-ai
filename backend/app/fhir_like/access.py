"""Who may see what through the adapter, derived from the existing permission matrix.

- Practitioner and Schedule: public callers and patients see active, listed ones; staff see all
  of them, with the active flag.
- Slot: public callers and patients see bookable slots only, and patients also see the slots of
  their own appointments; staff see every slot with its mapped status.
- Patient: the patient's own record only; staff see id, code, and display name only.
- Appointment: a patient's own appointments only; staff see the clinic schedule view.

A record a caller may not see is reported exactly like one that does not exist.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from app.auth.permissions import ROLE_PERMISSIONS, Permission
from app.auth.service import Principal
from app.core.errors import AuthenticationRequiredError, ForbiddenError


@dataclass(frozen=True)
class AccessContext:
    role: Literal["PUBLIC", "PATIENT", "STAFF"]
    patient_id: uuid.UUID | None
    permissions: frozenset[Permission]

    def __post_init__(self) -> None:
        if (self.role == "PATIENT") != (self.patient_id is not None):
            raise ValueError("Exactly the patient role carries a patient id.")

    @classmethod
    def public(cls) -> "AccessContext":
        return cls("PUBLIC", None, frozenset())

    @classmethod
    def patient(cls, patient_id: uuid.UUID) -> "AccessContext":
        return cls("PATIENT", patient_id, ROLE_PERMISSIONS["PATIENT"])

    @classmethod
    def staff(cls) -> "AccessContext":
        return cls("STAFF", None, ROLE_PERMISSIONS["STAFF"])

    @classmethod
    def from_principal(cls, principal: Principal | None) -> "AccessContext":
        """Identity comes only from a verified session, never from a request value."""
        if principal is None:
            return cls.public()
        if principal.actor_type == "PATIENT" and principal.patient_id is not None:
            return cls.patient(principal.patient_id)
        if principal.actor_type == "STAFF":
            return cls.staff()
        return cls.public()

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    def require_any(self, *permissions: Permission) -> None:
        """Public callers must authenticate; authenticated callers need one of the permissions."""
        if self.role == "PUBLIC":
            raise AuthenticationRequiredError()
        if not any(self.can(permission) for permission in permissions):
            raise ForbiddenError()
