"""Builds every seeded row from an anchor date without touching the database.

Identifiers are UUIDv5 values derived from stable keys, so a record is recognised by its id.
Slot, leave, and holiday ids include their date: running with another anchor adds new schedule
rows but never changes existing ones.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.seed import catalog

NAMESPACE = uuid.UUID("0b0c3a4e-6d1f-5c6a-9a58-7a1d8f2e4b10")
WINDOW_DAYS = 21
SLOT_MINUTES = 30

Row = dict[str, Any]


def seed_id(*parts: object) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, ":".join(str(part) for part in parts))


def localize(zone: ZoneInfo, day: date, at: time) -> datetime:
    """Convert a clinic-local wall-clock time to UTC, rejecting times skipped by DST."""
    naive = datetime.combine(day, at)
    utc = naive.replace(tzinfo=zone).astimezone(UTC)
    if utc.astimezone(zone).replace(tzinfo=None) != naive:
        raise ValueError(f"{naive} does not exist in {zone.key}")
    return utc


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _clock(minutes: int) -> time:
    return time(minutes // 60, minutes % 60)


def fixed_holidays(year: int) -> list[tuple[date, str]]:
    first_november = date(year, 11, 1)
    first_thursday = first_november + timedelta(days=(3 - first_november.weekday()) % 7)
    return [
        (date(year, 1, 1), "New Year's Day"),
        (date(year, 7, 4), "Independence Day"),
        (first_thursday + timedelta(weeks=3), "Thanksgiving Day"),
        (date(year, 12, 25), "Christmas Day"),
    ]


@dataclass(frozen=True)
class PlannedSlot:
    doctor: str
    clinic: str
    local_date: date
    starts_at: datetime
    ends_at: datetime

    @property
    def slot_id(self) -> uuid.UUID:
        return seed_id("slot", self.doctor, self.starts_at.isoformat())


@dataclass(frozen=True)
class AppointmentPlan:
    key: str
    patient_id: uuid.UUID
    slot_id: uuid.UUID
    booked_at: datetime
    cancelled_at: datetime | None

    @property
    def appointment_id(self) -> uuid.UUID:
        return seed_id("appointment", self.key)

    @property
    def cancelled(self) -> bool:
        return self.cancelled_at is not None

    def appointment_row(self) -> Row:
        return {
            "appointment_id": self.appointment_id,
            "patient_id": self.patient_id,
            "slot_id": self.slot_id,
            "status": "CANCELLED" if self.cancelled else "CONFIRMED",
            "version": 2 if self.cancelled else 1,
            "cancelled_at": self.cancelled_at,
            "created_at": self.booked_at,
            "updated_at": self.cancelled_at or self.booked_at,
        }

    def history_rows(self) -> list[Row]:
        base = {
            "appointment_id": self.appointment_id,
            "actor_id": self.patient_id,
            "actor_type": "PATIENT",
        }
        rows = [
            {
                **base,
                "event_id": seed_id("history", self.key, "booked"),
                "action": "BOOKED",
                "new_slot_id": self.slot_id,
                "timestamp": self.booked_at,
                "request_id": seed_id("history-request", self.key, "booked"),
            }
        ]
        if self.cancelled_at:
            rows.append(
                {
                    **base,
                    "event_id": seed_id("history", self.key, "cancelled"),
                    "action": "CANCELLED",
                    "old_slot_id": self.slot_id,
                    "timestamp": self.cancelled_at,
                    "request_id": seed_id("history-request", self.key, "cancelled"),
                }
            )
        return rows


@dataclass
class SeedPlan:
    anchors: dict[str, date]
    clinics: list[Row] = field(default_factory=list)
    clinic_hours: list[Row] = field(default_factory=list)
    clinic_holidays: list[Row] = field(default_factory=list)
    doctors: list[Row] = field(default_factory=list)
    doctor_clinics: list[Row] = field(default_factory=list)
    availability_rules: list[Row] = field(default_factory=list)
    doctor_leave: list[Row] = field(default_factory=list)
    slots: list[PlannedSlot] = field(default_factory=list)
    patients: list[Row] = field(default_factory=list)
    staff_users: list[Row] = field(default_factory=list)
    appointments: list[AppointmentPlan] = field(default_factory=list)
    waitlist_entries: list[Row] = field(default_factory=list)
    handoff_tickets: list[Row] = field(default_factory=list)
    handoff_notes: list[Row] = field(default_factory=list)

    def slot_rows(self) -> list[Row]:
        return [
            {
                "slot_id": slot.slot_id,
                "doctor_id": seed_id("doctor", slot.doctor),
                "clinic_id": seed_id("clinic", slot.clinic),
                "starts_at": slot.starts_at,
                "ends_at": slot.ends_at,
            }
            for slot in self.slots
        ]


def build_plan(anchor_date: date | None = None, now: datetime | None = None) -> SeedPlan:
    now = now or datetime.now(UTC)
    clinics = {spec.key: spec for spec in catalog.CLINICS}
    zones = {key: ZoneInfo(spec.timezone) for key, spec in clinics.items()}
    anchors = {key: anchor_date or now.astimezone(zone).date() for key, zone in zones.items()}
    plan = SeedPlan(anchors=anchors)

    _plan_directory(plan)
    closures = _plan_closures(plan, clinics, anchors)
    leaves = _plan_leave(plan, zones, anchors)
    _plan_slots(plan, clinics, zones, anchors, closures, leaves)
    _plan_appointments(plan, zones, anchors)
    _plan_waitlist(plan, zones, anchors)
    _plan_handoffs(plan, anchor_date or now.astimezone(UTC).date())
    return plan


def _plan_directory(plan: SeedPlan) -> None:
    for clinic in catalog.CLINICS:
        clinic_id = seed_id("clinic", clinic.key)
        plan.clinics.append(
            {
                "clinic_id": clinic_id,
                "name": clinic.name,
                "timezone": clinic.timezone,
                "address_line": clinic.address_line,
                "city": clinic.city,
                "region": clinic.region,
                "postal_code": clinic.postal_code,
            }
        )
        for weekday, (opens_at, closes_at) in clinic.hours.items():
            plan.clinic_hours.append(
                {
                    "hours_id": seed_id("hours", clinic.key, weekday),
                    "clinic_id": clinic_id,
                    "day_of_week": weekday,
                    "opens_at": opens_at,
                    "closes_at": closes_at,
                }
            )
    plan.doctors = [
        {
            "doctor_id": seed_id("doctor", doctor.key),
            "display_name": doctor.display_name,
            "specialty": doctor.specialty,
        }
        for doctor in catalog.DOCTORS
    ]
    plan.patients = [
        {
            "patient_id": seed_id("patient", patient.key),
            "patient_code": patient.login,
            "display_name": patient.display_name,
        }
        for patient in catalog.PATIENTS
    ]
    plan.staff_users = [
        {
            "staff_id": seed_id("staff", staff.key),
            "username": staff.login,
            "display_name": staff.display_name,
        }
        for staff in catalog.STAFF
    ]


def _plan_closures(
    plan: SeedPlan, clinics: dict[str, catalog.ClinicSpec], anchors: dict[str, date]
) -> dict[str, dict[date, list[tuple[time, time] | None]]]:
    """Return closed periods by clinic and date; None means closed for the whole day."""
    closed: dict[str, dict[date, list[tuple[time, time] | None]]] = {
        key: defaultdict(list) for key in clinics
    }
    declared: set[tuple[str, date, time | None, time | None]] = set()

    def add(clinic: str, day: date, label: str, start: time | None, end: time | None) -> None:
        if (clinic, day, start, end) in declared:
            return
        declared.add((clinic, day, start, end))
        closed[clinic][day].append(None if start is None else (start, end))
        plan.clinic_holidays.append(
            {
                "holiday_id": seed_id("holiday", clinic, day, start, end),
                "clinic_id": seed_id("clinic", clinic),
                "local_date": day,
                "start_time": start,
                "end_time": end,
                "label": label,
            }
        )

    for key in clinics:
        first, last = anchors[key] + timedelta(days=1), anchors[key] + timedelta(days=WINDOW_DAYS)
        for year in range(first.year, last.year + 1):
            for day, label in fixed_holidays(year):
                if first <= day <= last:
                    add(key, day, label, None, None)

    for closure in catalog.CLOSURES:
        day = anchors[closure.clinic] + timedelta(days=closure.offset)
        while day.weekday() not in clinics[closure.clinic].hours:
            day += timedelta(days=1)
        add(closure.clinic, day, closure.label, closure.closed_from, closure.closed_until)
    return closed


def _plan_leave(
    plan: SeedPlan, zones: dict[str, ZoneInfo], anchors: dict[str, date]
) -> dict[str, list[tuple[datetime, datetime]]]:
    leaves: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for leave in catalog.LEAVES:
        zone, anchor = zones[leave.clinic], anchors[leave.clinic]
        first = anchor + timedelta(days=leave.start_offset)
        starts_at = localize(zone, first, time(0))
        ends_at = localize(zone, anchor + timedelta(days=leave.end_offset + 1), time(0))
        leaves[leave.doctor].append((starts_at, ends_at))
        plan.doctor_leave.append(
            {
                "leave_id": seed_id("leave", leave.doctor, leave.clinic, first),
                "doctor_id": seed_id("doctor", leave.doctor),
                "clinic_id": seed_id("clinic", leave.clinic),
                "starts_at": starts_at,
                "ends_at": ends_at,
                "reason_code": leave.reason_code,
            }
        )
    return leaves


def _plan_slots(
    plan: SeedPlan,
    clinics: dict[str, catalog.ClinicSpec],
    zones: dict[str, ZoneInfo],
    anchors: dict[str, date],
    closures: dict[str, dict[date, list[tuple[time, time] | None]]],
    leaves: dict[str, list[tuple[datetime, datetime]]],
) -> None:
    assigned: set[tuple[str, str]] = set()
    for assignment in catalog.ASSIGNMENTS:
        hours = clinics[assignment.clinic].hours
        if (assignment.doctor, assignment.clinic) not in assigned:
            assigned.add((assignment.doctor, assignment.clinic))
            plan.doctor_clinics.append(
                {
                    "doctor_id": seed_id("doctor", assignment.doctor),
                    "clinic_id": seed_id("clinic", assignment.clinic),
                }
            )
        for weekday in assignment.weekdays:
            for start, end in assignment.windows:
                opens_at, closes_at = hours.get(weekday, (None, None))
                if opens_at is None or start < opens_at or end > closes_at:
                    raise ValueError(
                        f"{assignment.doctor} at {assignment.clinic} is scheduled "
                        "outside clinic hours"
                    )
                plan.availability_rules.append(
                    {
                        "availability_id": seed_id(
                            "availability",
                            assignment.doctor,
                            assignment.clinic,
                            weekday,
                            start,
                            end,
                        ),
                        "doctor_id": seed_id("doctor", assignment.doctor),
                        "clinic_id": seed_id("clinic", assignment.clinic),
                        "day_of_week": weekday,
                        "start_time": start,
                        "end_time": end,
                        "effective_start_date": date(anchors[assignment.clinic].year, 1, 1),
                        "effective_end_date": None,
                    }
                )

        anchor, zone = anchors[assignment.clinic], zones[assignment.clinic]
        for offset in range(1, WINDOW_DAYS + 1):
            day = anchor + timedelta(days=offset)
            if day.weekday() not in assignment.weekdays:
                continue
            day_closures = closures[assignment.clinic].get(day, [])
            if None in day_closures:
                continue
            for start, end in assignment.windows:
                for begin in range(_minutes(start), _minutes(end) - SLOT_MINUTES + 1, SLOT_MINUTES):
                    local_start, local_end = _clock(begin), _clock(begin + SLOT_MINUTES)
                    if any(
                        window and local_start < window[1] and local_end > window[0]
                        for window in day_closures
                    ):
                        continue
                    starts_at = localize(zone, day, local_start)
                    ends_at = starts_at + timedelta(minutes=SLOT_MINUTES)
                    if any(
                        starts_at < leave_end and ends_at > leave_start
                        for leave_start, leave_end in leaves[assignment.doctor]
                    ):
                        continue
                    plan.slots.append(
                        PlannedSlot(assignment.doctor, assignment.clinic, day, starts_at, ends_at)
                    )

    plan.slots.sort(key=lambda slot: (slot.doctor, slot.starts_at))
    for earlier, later in zip(plan.slots, plan.slots[1:], strict=False):
        if earlier.doctor == later.doctor and later.starts_at < earlier.ends_at:
            raise ValueError(f"{earlier.doctor} has overlapping planned slots at {later.starts_at}")


def _plan_appointments(
    plan: SeedPlan, zones: dict[str, ZoneInfo], anchors: dict[str, date]
) -> None:
    used: set[uuid.UUID] = set()
    for spec in catalog.APPOINTMENTS:
        anchor, zone = anchors[spec.clinic], zones[spec.clinic]
        earliest = anchor + timedelta(days=spec.min_offset)
        by_day: dict[date, list[PlannedSlot]] = defaultdict(list)
        for slot in plan.slots:
            if (
                slot.doctor == spec.doctor
                and slot.clinic == spec.clinic
                and slot.local_date >= earliest
            ):
                by_day[slot.local_date].append(slot)
        candidates = [
            day_slots for _, day_slots in sorted(by_day.items()) if len(day_slots) > spec.slot_index
        ]
        if not candidates:
            raise ValueError(f"No slot available for scenario {spec.key}")
        slot = candidates[0][spec.slot_index]
        if slot.slot_id in used:
            raise ValueError(f"Scenario {spec.key} reuses a slot")
        used.add(slot.slot_id)

        cancelled_at = None
        if spec.cancelled_days_ago is not None:
            if spec.cancelled_days_ago >= spec.booked_days_ago:
                raise ValueError(f"Scenario {spec.key} cancels before it is booked")
            cancelled_at = localize(
                zone, anchor - timedelta(days=spec.cancelled_days_ago), time(11)
            )
        plan.appointments.append(
            AppointmentPlan(
                key=spec.key,
                patient_id=seed_id("patient", spec.patient),
                slot_id=slot.slot_id,
                booked_at=localize(zone, anchor - timedelta(days=spec.booked_days_ago), time(10)),
                cancelled_at=cancelled_at,
            )
        )


def _plan_waitlist(plan: SeedPlan, zones: dict[str, ZoneInfo], anchors: dict[str, date]) -> None:
    specialties = {doctor.key: doctor.specialty for doctor in catalog.DOCTORS}
    served = {(a.clinic, specialties[a.doctor]) for a in catalog.ASSIGNMENTS}
    for spec in catalog.WAITLIST:
        if (spec.clinic, spec.specialty) not in served:
            raise ValueError(f"{spec.key} names a specialty the clinic does not offer")
        anchor = anchors[spec.clinic]
        created_at = localize(
            zones[spec.clinic], anchor - timedelta(days=spec.created_days_ago), time(9)
        )
        plan.waitlist_entries.append(
            {
                "waitlist_id": seed_id("waitlist", spec.key),
                "patient_id": seed_id("patient", spec.patient),
                "clinic_id": seed_id("clinic", spec.clinic),
                "specialty": spec.specialty,
                "start_date": anchor + timedelta(days=spec.start_offset),
                "end_date": anchor + timedelta(days=spec.end_offset),
                "time_preference": spec.time_preference,
                "status": "ACTIVE",
                "created_at": created_at,
                "updated_at": created_at,
            }
        )


def _plan_handoffs(plan: SeedPlan, reference: date) -> None:
    versions = {"OPEN": 1, "IN_PROGRESS": 2, "RESOLVED": 3}
    for spec in catalog.HANDOFFS:
        created_at = datetime.combine(
            reference - timedelta(days=spec.created_days_ago), time(15), UTC
        )
        resolved_at = None
        if spec.resolved_days_ago is not None:
            resolved_at = datetime.combine(
                reference - timedelta(days=spec.resolved_days_ago), time(16), UTC
            )
        ticket_id = seed_id("handoff", spec.key)
        plan.handoff_tickets.append(
            {
                "ticket_id": ticket_id,
                "patient_id": seed_id("patient", spec.patient) if spec.patient else None,
                "category": spec.category,
                "status": spec.status,
                "redacted_summary": spec.summary,
                "version": versions[spec.status],
                "created_at": created_at,
                "updated_at": resolved_at or created_at,
                "resolved_at": resolved_at,
            }
        )
        if spec.note:
            plan.handoff_notes.append(
                {
                    "note_id": seed_id("handoff-note", spec.key),
                    "ticket_id": ticket_id,
                    "staff_id": seed_id("staff", catalog.STAFF_AUTHOR_FOR_NOTES),
                    "note_text": spec.note,
                    "created_at": created_at + timedelta(hours=1),
                    "request_id": seed_id("handoff-note-request", spec.key),
                }
            )
