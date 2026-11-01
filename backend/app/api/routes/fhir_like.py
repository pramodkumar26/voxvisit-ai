"""Read-only endpoints for the synthetic FHIR-like adapter (GET only; no mutations exist here).

Responses are simplified, fictional, FHIR-like JSON, not HL7 FHIR: the media type is plain
application/json and every response carries X-Resource-Model: synthetic-fhir-like. Authorization is
the same as the rest of the API: identity comes only from the verified session, and a record the
caller may not see looks exactly like a missing one.
"""

from typing import Annotated
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, Request, Response

from app.api.dependencies import EngineDep, NowDep
from app.auth.dependencies import OptionalPrincipal, require_any_permission
from app.auth.service import Principal
from app.fhir_like import resources as r
from app.fhir_like.access import AccessContext
from app.fhir_like.adapter import (
    APPOINTMENT_READERS,
    PATIENT_READERS,
    Page,
    open_adapter,
    to_bundle,
)
from app.fhir_like.params import (
    AppointmentSearch,
    PractitionerSearch,
    ScheduleSearch,
    SlotSearch,
)

MODEL_HEADER = "X-Resource-Model"


def mark_synthetic(response: Response) -> None:
    response.headers[MODEL_HEADER] = r.MODEL_CODE
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(
    prefix="/integration/fhir-like",
    tags=["fhir-like (synthetic, read-only)"],
    dependencies=[Depends(mark_synthetic)],
)

ScheduleId = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]
PatientReader = Annotated[Principal, Depends(require_any_permission(*PATIENT_READERS))]
AppointmentReader = Annotated[Principal, Depends(require_any_permission(*APPOINTMENT_READERS))]
EXCLUDE_NONE = {"response_model_exclude_none": True}


def _bundle(request: Request, page: Page) -> r.Bundle:
    next_url = None
    if page.next_cursor:
        parameters = {**request.query_params, "cursor": page.next_cursor}
        next_url = f"{request.url.path}?{urlencode(parameters)}"
    return to_bundle(page, next_url=next_url)


# --- public reads (what a caller sees depends on their session) -----------------------------


@router.get("/Practitioner", response_model=r.Bundle, **EXCLUDE_NONE)
def search_practitioners(
    query: Annotated[PractitionerSearch, Query()],
    request: Request,
    principal: OptionalPrincipal,
    engine: EngineDep,
    now: NowDep,
) -> r.Bundle:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        page = adapter.search_practitioners(
            clinic_id=query.clinic,
            specialty=query.specialty,
            cursor=query.cursor,
            limit=query.limit,
        )
    return _bundle(request, page)


@router.get("/Practitioner/{doctor_id}", response_model=r.Practitioner, **EXCLUDE_NONE)
def read_practitioner(
    doctor_id: UUID, principal: OptionalPrincipal, engine: EngineDep, now: NowDep
) -> r.Practitioner:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        return adapter.practitioner(doctor_id)


@router.get("/Schedule", response_model=r.Bundle, **EXCLUDE_NONE)
def search_schedules(
    query: Annotated[ScheduleSearch, Query()],
    request: Request,
    principal: OptionalPrincipal,
    engine: EngineDep,
    now: NowDep,
) -> r.Bundle:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        page = adapter.search_schedules(
            clinic_id=query.clinic,
            doctor_id=query.practitioner,
            cursor=query.cursor,
            limit=query.limit,
        )
    return _bundle(request, page)


@router.get("/Schedule/{schedule_id}", response_model=r.Schedule, **EXCLUDE_NONE)
def read_schedule(
    schedule_id: ScheduleId, principal: OptionalPrincipal, engine: EngineDep, now: NowDep
) -> r.Schedule:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        return adapter.schedule(schedule_id)


@router.get("/Slot", response_model=r.Bundle, **EXCLUDE_NONE)
def search_slots(
    query: Annotated[SlotSearch, Query()],
    request: Request,
    principal: OptionalPrincipal,
    engine: EngineDep,
    now: NowDep,
) -> r.Bundle:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        page = adapter.search_slots(
            clinic_id=query.clinic,
            start_date=query.start_date,
            end_date=query.end_date,
            doctor_id=query.practitioner,
            specialty=query.specialty,
            cursor=query.cursor,
            limit=query.limit,
        )
    return _bundle(request, page)


@router.get("/Slot/{slot_id}", response_model=r.Slot, **EXCLUDE_NONE)
def read_slot(
    slot_id: UUID, principal: OptionalPrincipal, engine: EngineDep, now: NowDep
) -> r.Slot:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        return adapter.slot(slot_id)


# --- protected reads: a verified patient (own records) or demo staff (permitted views) ------


@router.get("/Patient/{patient_id}", response_model=r.Patient, **EXCLUDE_NONE)
def read_patient(
    patient_id: UUID, principal: PatientReader, engine: EngineDep, now: NowDep
) -> r.Patient:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        return adapter.patient(patient_id)


@router.get("/Appointment", response_model=r.Bundle, **EXCLUDE_NONE)
def search_appointments(
    query: Annotated[AppointmentSearch, Query()],
    request: Request,
    principal: AppointmentReader,
    engine: EngineDep,
    now: NowDep,
) -> r.Bundle:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        page = adapter.search_appointments(
            status=query.status,
            start_date=query.start_date,
            end_date=query.end_date,
            clinic_id=query.clinic,
            doctor_id=query.practitioner,
            cursor=query.cursor,
            limit=query.limit,
        )
    return _bundle(request, page)


@router.get("/Appointment/{appointment_id}", response_model=r.Appointment, **EXCLUDE_NONE)
def read_appointment(
    appointment_id: UUID, principal: AppointmentReader, engine: EngineDep, now: NowDep
) -> r.Appointment:
    with open_adapter(engine, AccessContext.from_principal(principal), now) as adapter:
        return adapter.appointment(appointment_id)
