"""The read-only adapter interface: authorization-aware lookups that return FHIR-like resources.

This is the internal boundary where a future, separately reviewed EHR integration would attach.
It owns no data and writes nothing. It runs inside a READ ONLY transaction, builds each resource
from the authoritative tables on request, applies the caller's AccessContext to every read, and
reuses the existing availability and own-appointment services for the lookups they already
implement. Booking, rescheduling, cancellation, and confirmation are deliberately absent.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import Connection, Engine

from app.auth.permissions import Permission
from app.core.errors import InvalidArgumentError
from app.fhir_like import mappers, queries
from app.fhir_like import resources as r
from app.fhir_like.access import AccessContext
from app.services import appointments as own_appointments
from app.services.availability import search_availability
from app.services.read_access import read_only_connection

PATIENT_READERS = (Permission.VIEW_OWN_PROFILE, Permission.VIEW_PATIENT_DIRECTORY)
APPOINTMENT_READERS = (Permission.VIEW_OWN_APPOINTMENTS, Permission.VIEW_CLINIC_SCHEDULES)


@dataclass(frozen=True)
class Page:
    resources: list
    next_cursor: str | None


class FhirLikeAdapter:
    def __init__(self, connection: Connection, access: AccessContext, now: datetime) -> None:
        self._connection = connection
        self._access = access
        self._now = now

    # --- single reads -------------------------------------------------------------------

    def practitioner(self, doctor_id: uuid.UUID) -> r.Practitioner:
        record = queries.get_practitioner(self._connection, self._access, doctor_id)
        return mappers.to_practitioner(record)

    def schedule(self, schedule_id: str) -> r.Schedule:
        try:
            doctor_id, clinic_id = mappers.parse_schedule_id(schedule_id)
        except mappers.InvalidReferenceError:
            raise InvalidArgumentError("The Schedule id is not valid.") from None
        record = queries.get_schedule(self._connection, self._access, doctor_id, clinic_id)
        return mappers.to_schedule(record)

    def slot(self, slot_id: uuid.UUID) -> r.Slot:
        record = queries.get_slot(self._connection, self._access, slot_id, self._now)
        return mappers.to_slot(record)

    def patient(self, patient_id: uuid.UUID) -> r.Patient:
        self._access.require_any(*PATIENT_READERS)
        return mappers.to_patient(queries.get_patient(self._connection, self._access, patient_id))

    def appointment(self, appointment_id: uuid.UUID) -> r.Appointment:
        self._access.require_any(*APPOINTMENT_READERS)
        record = queries.get_appointment(self._connection, self._access, appointment_id)
        return mappers.to_appointment(record)

    def resolve(self, reference: str):
        """Follow a "Type/id" reference. The caller's access rules apply exactly as for a read."""
        try:
            resource_type, resource_id = mappers.parse_reference(reference)
        except mappers.InvalidReferenceError:
            raise InvalidArgumentError("The reference is not valid.") from None
        if resource_type == "Schedule":
            return self.schedule(resource_id)
        read = {
            "Patient": self.patient,
            "Practitioner": self.practitioner,
            "Slot": self.slot,
            "Appointment": self.appointment,
        }[resource_type]
        return read(uuid.UUID(resource_id))

    # --- searches -----------------------------------------------------------------------

    def search_practitioners(
        self,
        *,
        clinic_id: uuid.UUID | None = None,
        specialty: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> Page:
        records, next_cursor = queries.search_practitioners(
            self._connection,
            self._access,
            clinic_id=clinic_id,
            specialty=specialty,
            cursor=cursor,
            limit=limit,
        )
        return Page([mappers.to_practitioner(record) for record in records], next_cursor)

    def search_schedules(
        self,
        *,
        clinic_id: uuid.UUID | None = None,
        doctor_id: uuid.UUID | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> Page:
        records, next_cursor = queries.search_schedules(
            self._connection,
            self._access,
            clinic_id=clinic_id,
            doctor_id=doctor_id,
            cursor=cursor,
            limit=limit,
        )
        return Page([mappers.to_schedule(record) for record in records], next_cursor)

    def search_slots(
        self,
        *,
        clinic_id: uuid.UUID,
        start_date: date,
        end_date: date,
        doctor_id: uuid.UUID | None = None,
        specialty: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> Page:
        """Free slots, found by the same availability search the public API uses."""
        found = search_availability(
            self._connection,
            clinic_id=clinic_id,
            start_date=start_date,
            end_date=end_date,
            doctor_id=doctor_id,
            specialty=specialty,
            time_preference="ANYTIME",
            cursor=cursor,
            limit=limit,
            now=self._now,
        )
        records = queries.slots_by_id(
            self._connection, self._access, [slot.slot_id for slot in found.slots], self._now
        )
        return Page([mappers.to_slot(record) for record in records], found.next_cursor)

    def search_appointments(
        self,
        *,
        status: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        clinic_id: uuid.UUID | None = None,
        doctor_id: uuid.UUID | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> Page:
        self._access.require_any(*APPOINTMENT_READERS)
        stored = None
        if status is not None:
            stored = mappers.STORED_APPOINTMENT_STATUS.get(status)
            if stored is None:
                raise InvalidArgumentError("The status must be booked or cancelled.")
        if self._access.can(Permission.VIEW_CLINIC_SCHEDULES):
            records, next_cursor = queries.search_staff_appointments(
                self._connection,
                self._access,
                clinic_id=clinic_id,
                doctor_id=doctor_id,
                status=stored,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor,
                limit=limit,
            )
        else:
            if clinic_id or doctor_id:
                raise InvalidArgumentError("Patients can search only their own appointments.")
            own = own_appointments.list_my_appointments(
                self._connection,
                patient_id=self._access.patient_id,
                status=stored,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor,
                limit=limit,
            )
            records = queries.appointments_by_id(
                self._connection,
                self._access,
                [item.appointment_id for item in own.appointments],
            )
            next_cursor = own.next_cursor
        return Page([mappers.to_appointment(record) for record in records], next_cursor)


@contextmanager
def open_adapter(engine: Engine, access: AccessContext, now: datetime) -> Iterator[FhirLikeAdapter]:
    """An adapter bound to a READ ONLY, time-limited transaction (the approved 8-second limit)."""
    with read_only_connection(engine) as connection:
        yield FhirLikeAdapter(connection, access, now)


def to_bundle(page: Page, *, next_url: str | None = None) -> r.Bundle:
    return r.Bundle(
        meta=mappers.bundle_meta(),
        entry=[
            r.BundleEntry(fullUrl=f"{resource.resourceType}/{resource.id}", resource=resource)
            for resource in page.resources
        ],
        link=[r.BundleLink(relation="next", url=next_url)] if next_url else [],
    )
