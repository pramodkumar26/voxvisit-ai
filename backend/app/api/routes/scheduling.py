"""Read-only scheduling endpoints. Each is a thin wrapper over a service the future agent tools
can call directly, so authorization and validation never depend on the transport."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response

from app.api.dependencies import EngineDep, NowDep
from app.auth.dependencies import require_actor
from app.auth.permissions import Permission
from app.auth.service import PATIENT, Principal
from app.models.scheduling import (
    AvailabilityPage,
    AvailabilityQuery,
    DirectoryPage,
    DirectoryQuery,
    MyAppointmentsPage,
    MyAppointmentsQuery,
)
from app.services import appointments, availability, directory
from app.services.read_access import read_only_connection

router = APIRouter(tags=["scheduling"])

PatientReader = Annotated[
    Principal, Depends(require_actor(PATIENT, Permission.VIEW_OWN_APPOINTMENTS))
]


@router.get("/directory", response_model=DirectoryPage)
def search_directory(query: Annotated[DirectoryQuery, Query()], engine: EngineDep) -> DirectoryPage:
    """Public. Search active clinics and doctors, or browse the whole directory."""
    with read_only_connection(engine) as connection:
        return directory.search_directory(connection, **query.model_dump())


@router.get("/availability", response_model=AvailabilityPage)
def search_availability(
    query: Annotated[AvailabilityQuery, Query()], engine: EngineDep, now: NowDep
) -> AvailabilityPage:
    """Public. Currently searchable slots; a result is never a reservation."""
    with read_only_connection(engine) as connection:
        return availability.search_availability(connection, now=now, **query.model_dump())


@router.get("/me/appointments", response_model=MyAppointmentsPage)
def my_appointments(
    query: Annotated[MyAppointmentsQuery, Query()],
    principal: PatientReader,
    engine: EngineDep,
    response: Response,
) -> MyAppointmentsPage:
    """Verified patients only. The patient is always the session's patient."""
    response.headers["Cache-Control"] = "no-store"
    with read_only_connection(engine) as connection:
        return appointments.list_my_appointments(
            connection, patient_id=principal.patient_id, **query.model_dump()
        )
