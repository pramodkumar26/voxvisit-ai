from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo, available_timezones

import pytest

from app.seed import catalog
from app.seed.credentials import (
    SeedCredentialError,
    create_hasher,
    validate_patient_pin,
    validate_staff_password,
)
from app.seed.plan import SLOT_MINUTES, WINDOW_DAYS, build_plan, fixed_holidays, localize, seed_id

ANCHOR = date(2026, 10, 28)
APPROVED_CATEGORIES = {
    "SCHEDULING",
    "CLINIC_INFORMATION",
    "ACCESSIBILITY",
    "TECHNICAL_SUPPORT",
    "OTHER_ADMINISTRATIVE",
}
CLINICS = {clinic.key: clinic for clinic in catalog.CLINICS}


def local(slot, field="starts_at"):
    return getattr(slot, field).astimezone(ZoneInfo(CLINICS[slot.clinic].timezone))


def test_catalog_sizes_and_timezones() -> None:
    assert len(catalog.CLINICS) == 3
    assert 6 <= len(catalog.DOCTORS) <= 9
    assert len(catalog.PATIENTS) >= 12
    assert len(catalog.STAFF) == 2
    for clinic in catalog.CLINICS:
        assert clinic.timezone in available_timezones()
    assert len({clinic.timezone for clinic in catalog.CLINICS}) == 3


def test_plan_is_reproducible_for_an_anchor_date() -> None:
    first = build_plan(ANCHOR, now=datetime(2030, 1, 1, tzinfo=UTC))
    second = build_plan(ANCHOR, now=datetime(2040, 6, 1, tzinfo=UTC))

    assert first.slot_rows() == second.slot_rows()
    assert first.appointments == second.appointments
    assert first.waitlist_entries == second.waitlist_entries
    assert first.clinic_holidays == second.clinic_holidays


def test_default_anchor_uses_each_clinics_local_date() -> None:
    plan = build_plan(None, now=datetime(2026, 6, 15, 6, 30, tzinfo=UTC))

    assert plan.anchors == {
        "riverbend": date(2026, 6, 15),
        "lakeside": date(2026, 6, 15),
        "harborview": date(2026, 6, 14),
    }


def test_slots_fall_inside_the_three_week_window_and_search_limit() -> None:
    plan = build_plan(ANCHOR)

    assert WINDOW_DAYS == 21
    for slot in plan.slots:
        offset = (slot.local_date - ANCHOR).days
        assert 1 <= offset <= WINDOW_DAYS
        assert local(slot).date() == slot.local_date
        assert slot.ends_at - slot.starts_at == timedelta(minutes=SLOT_MINUTES)


def test_slots_respect_availability_hours_holidays_and_leave() -> None:
    plan = build_plan(ANCHOR)
    assignments = defaultdict(list)
    for assignment in catalog.ASSIGNMENTS:
        assignments[(assignment.doctor, assignment.clinic)].append(assignment)
    full_closures = {
        (row["clinic_id"], row["local_date"])
        for row in plan.clinic_holidays
        if row["start_time"] is None
    }
    partial = defaultdict(list)
    for row in plan.clinic_holidays:
        if row["start_time"] is not None:
            partial[(row["clinic_id"], row["local_date"])].append(
                (row["start_time"], row["end_time"])
            )
    leave_by_doctor = defaultdict(list)
    for leave in plan.doctor_leave:
        leave_by_doctor[leave["doctor_id"]].append((leave["starts_at"], leave["ends_at"]))
    assert plan.slots
    for slot in plan.slots:
        start, end = local(slot).time(), local(slot, "ends_at").time()
        weekday = slot.local_date.weekday()
        opens_at, closes_at = CLINICS[slot.clinic].hours[weekday]
        assert opens_at <= start and end <= closes_at
        assert any(
            weekday in assignment.weekdays
            and any(
                window_start <= start and end <= window_end
                for window_start, window_end in assignment.windows
            )
            for assignment in assignments[(slot.doctor, slot.clinic)]
        )
        clinic_id = seed_id("clinic", slot.clinic)
        assert (clinic_id, slot.local_date) not in full_closures
        for closed_from, closed_until in partial[(clinic_id, slot.local_date)]:
            assert end <= closed_from or start >= closed_until
        for leave_start, leave_end in leave_by_doctor[seed_id("doctor", slot.doctor)]:
            assert slot.ends_at <= leave_start or slot.starts_at >= leave_end


def test_each_doctor_has_no_overlapping_slots() -> None:
    slots = build_plan(ANCHOR).slots
    by_doctor = defaultdict(list)
    for slot in slots:
        by_doctor[slot.doctor].append(slot)

    for doctor_slots in by_doctor.values():
        ordered = sorted(doctor_slots, key=lambda slot: slot.starts_at)
        for earlier, later in pairwise(ordered):
            assert earlier.ends_at <= later.starts_at


def test_leave_and_closures_actually_remove_slots() -> None:
    plan = build_plan(ANCHOR)
    lindqvist = [slot for slot in plan.slots if slot.doctor == "lindqvist"]
    leave_days = {ANCHOR + timedelta(days=offset) for offset in range(10, 17)}

    assert lindqvist
    assert not any(slot.local_date in leave_days for slot in lindqvist)

    closure = next(row for row in plan.clinic_holidays if row["label"] == "Staff training day")
    riverbend = [slot for slot in plan.slots if slot.clinic == "riverbend"]
    assert not any(slot.local_date == closure["local_date"] for slot in riverbend)
    assert any(
        slot.local_date != closure["local_date"]
        and slot.local_date.weekday() == closure["local_date"].weekday()
        for slot in riverbend
    )


@pytest.mark.parametrize(
    "anchor",
    [date(2026, 10, 28), date(2027, 3, 10)],
    ids=["us-fall-back", "us-spring-forward"],
)
def test_local_times_stay_fixed_across_daylight_saving_changes(anchor: date) -> None:
    plan = build_plan(anchor)
    morning_starts = [
        slot for slot in plan.slots if slot.doctor == "hartwell" and local(slot).time() == time(9)
    ]

    assert {local(slot).time() for slot in morning_starts} == {time(9)}
    assert {slot.starts_at.hour for slot in morning_starts} == {15, 16}


def test_localize_rejects_times_skipped_by_daylight_saving() -> None:
    zone = ZoneInfo("America/Denver")

    assert localize(zone, date(2027, 3, 15), time(2, 30)).hour == 8
    with pytest.raises(ValueError, match="does not exist"):
        localize(zone, date(2027, 3, 14), time(2, 30))


def test_fixed_holidays_include_computed_thanksgiving() -> None:
    dates = dict((label, day) for day, label in fixed_holidays(2026))

    assert dates["Thanksgiving Day"] == date(2026, 11, 26)
    assert dates["Christmas Day"] == date(2026, 12, 25)


def test_appointment_scenarios_use_distinct_valid_slots() -> None:
    plan = build_plan(ANCHOR)
    slot_ids = {slot.slot_id: slot for slot in plan.slots}

    assert len(plan.appointments) == 11
    assert sum(item.cancelled for item in plan.appointments) == 3
    assert len({item.slot_id for item in plan.appointments}) == 11
    for item in plan.appointments:
        slot = slot_ids[item.slot_id]
        assert item.booked_at < slot.starts_at
        if item.cancelled_at:
            assert item.booked_at < item.cancelled_at < slot.starts_at


def test_waitlist_entries_do_not_overlap_and_match_offered_specialties() -> None:
    plan = build_plan(ANCHOR)
    seen = []

    for entry in plan.waitlist_entries:
        assert entry["start_date"] <= entry["end_date"]
        assert (entry["end_date"] - entry["start_date"]).days < 31
        for other in seen:
            same_scope = all(
                entry[name] == other[name] for name in ("patient_id", "clinic_id", "specialty")
            )
            assert not same_scope or (
                entry["end_date"] < other["start_date"] or entry["start_date"] > other["end_date"]
            )
        seen.append(entry)
    assert len(plan.waitlist_entries) == 3


def test_handoff_tickets_are_sanitized_and_categorized() -> None:
    plan = build_plan(ANCHOR)

    assert plan.handoff_tickets
    for ticket in plan.handoff_tickets:
        assert ticket["category"] in APPROVED_CATEGORIES
        assert len(ticket["redacted_summary"]) <= 500
        assert (ticket["status"] == "RESOLVED") == (ticket["resolved_at"] is not None)


@pytest.mark.parametrize("pin", ["123456", "000000", "905173"])
def test_valid_patient_pins_are_accepted(pin: str) -> None:
    validate_patient_pin(pin)


@pytest.mark.parametrize(
    "pin", ["12345", "1234567", "12345a", "", " 12345", "\uff11\uff12\uff13\uff14\uff15\uff16"]
)
def test_invalid_patient_pins_are_rejected_without_echoing_them(pin: str) -> None:
    with pytest.raises(SeedCredentialError) as exc_info:
        validate_patient_pin(pin)

    if pin:
        assert pin not in str(exc_info.value)


@pytest.mark.parametrize("password", ["short", "elevenchars", "password1234", "aaaaaaaaaaaaaaaa"])
def test_weak_staff_passwords_are_rejected_without_echoing_them(password: str) -> None:
    with pytest.raises(SeedCredentialError) as exc_info:
        validate_staff_password(password)

    assert password not in str(exc_info.value)


def test_strong_staff_password_is_accepted() -> None:
    validate_staff_password("Tw3lve-chars-ok")


def test_default_hasher_produces_verifiable_argon2id_hashes() -> None:
    hasher = create_hasher()
    digest = hasher.hash("905173")

    assert digest.startswith("$argon2id$v=19$")
    assert hasher.verify(digest, "905173")
    assert not hasher.check_needs_rehash(digest)
