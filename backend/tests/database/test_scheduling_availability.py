import uuid
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from scheduling_support import (
    ALL_DAY,
    ALL_WEEK,
    MONDAY,
    add_holiday,
    add_leave,
    add_slots,
    assign_doctor,
    half_hours,
    new_clinic,
    new_doctor,
)
from sqlalchemy import text

from app.seed.plan import localize

TUESDAY = MONDAY + timedelta(days=1)
SEEDED_CLINICS = (
    "Riverbend Family Health",
    "Lakeside Community Clinic",
    "Harborview Medical Associates",
)
SLOT_KEYS = {
    "slot_id",
    "clinic",
    "doctor",
    "starts_at_utc",
    "ends_at_utc",
    "starts_at_local",
    "ends_at_local",
    "clinic_timezone",
}


def search(env, clinic, start: date, end: date | None = None, **params):
    query = {
        "clinic_id": str(clinic.clinic_id),
        "start_date": start.isoformat(),
        "end_date": (end or start).isoformat(),
        **{key: str(value) for key, value in params.items()},
    }
    return env.client.get("/availability", params=query)


def starts(response) -> list[str]:
    """Clinic-local 'YYYY-MM-DD HH:MM' for each returned slot, in response order."""
    assert response.status_code == 200, response.text
    return [slot["starts_at_local"][:16].replace("T", " ") for slot in response.json()["slots"]]


def local(day: date, at: time) -> str:
    return f"{day.isoformat()} {at.strftime('%H:%M')}"


# --- what counts as bookable -----------------------------------------------------------------


def test_only_available_non_removed_future_slots_are_returned(auth_env) -> None:
    engine = auth_env.engine
    clinic = new_clinic(engine)
    doctor = new_doctor(engine, clinic)
    at = lambda hour, minute=0: (MONDAY, time(hour, minute))  # noqa: E731
    add_slots(engine, clinic, doctor, [at(9)])
    add_slots(engine, clinic, doctor, [at(9, 30)], status="BOOKED")
    add_slots(engine, clinic, doctor, [at(10)], status="HELD")
    add_slots(engine, clinic, doctor, [at(10, 30)], removed_at=auth_env.clock.now)
    add_slots(engine, clinic, doctor, [at(11)])

    response = search(auth_env, clinic, MONDAY)

    assert starts(response) == [local(MONDAY, time(9)), local(MONDAY, time(11))]


def test_slots_that_have_already_started_are_not_searchable(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic)
    add_slots(auth_env.engine, clinic, doctor, half_hours(MONDAY, time(9), 3))
    auth_env.clock.now = localize(clinic.zone, MONDAY, time(9, 30))

    assert starts(search(auth_env, clinic, MONDAY)) == [local(MONDAY, time(10))]


def test_searching_does_not_reserve_or_change_anything(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic)
    ids = add_slots(auth_env.engine, clinic, doctor, half_hours(MONDAY, time(9), 3))

    def state():
        with auth_env.engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT slot_id, status, version, updated_at FROM slots"
                    " WHERE slot_id = ANY(:ids) ORDER BY 1"
                ),
                {"ids": ids},
            ).all()

    before = state()
    search(auth_env, clinic, MONDAY)
    search(auth_env, clinic, MONDAY)

    assert state() == before
    with auth_env.engine.connect() as connection:
        held = connection.execute(
            text("SELECT count(*) FROM slots WHERE status = 'HELD' AND slot_id = ANY(:ids)"),
            {"ids": ids},
        ).scalar_one()
    assert held == 0


def test_slot_fields_expose_only_approved_information(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic, specialty="Cardiology")
    add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9))])

    response = search(auth_env, clinic, MONDAY)

    body = response.json()
    assert set(body) == {"window", "slots", "next_cursor"}
    [slot] = body["slots"]
    assert set(slot) == SLOT_KEYS
    assert set(slot["clinic"]) == {"clinic_id", "name", "timezone"}
    assert set(slot["doctor"]) == {"doctor_id", "display_name", "specialty"}
    assert slot["doctor"]["specialty"] == "Cardiology"
    assert body["window"] == {
        "start_date": MONDAY.isoformat(),
        "end_date": MONDAY.isoformat(),
        "timezone": "America/Denver",
    }
    for forbidden in ("version", "status", "removed_at", "patient", "appointment", "token"):
        assert forbidden not in response.text


# --- clinic closures, leave, hours, and rules ------------------------------------------------


def test_full_day_holidays_remove_that_days_slots(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic)
    add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9)), (TUESDAY, time(9))])
    add_holiday(auth_env.engine, clinic, TUESDAY)

    assert starts(search(auth_env, clinic, MONDAY, TUESDAY)) == [local(MONDAY, time(9))]


def test_partial_holidays_remove_only_overlapping_slots(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic, rule_windows=((time(9), time(16)),))
    add_slots(auth_env.engine, clinic, doctor, half_hours(MONDAY, time(12, 30), 4))
    add_holiday(auth_env.engine, clinic, MONDAY, time(13), time(14))

    # 12:30-13:00 ends exactly when the closure starts; 14:00 starts exactly when it ends.
    assert starts(search(auth_env, clinic, MONDAY)) == [
        local(MONDAY, time(12, 30)),
        local(MONDAY, time(14)),
    ]


def test_doctor_leave_removes_overlapping_slots_only(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic)
    colleague = new_doctor(auth_env.engine, clinic)
    moments = half_hours(MONDAY, time(9), 4)  # 09:00, 09:30, 10:00, 10:30
    add_slots(auth_env.engine, clinic, doctor, moments)
    add_slots(auth_env.engine, clinic, colleague, moments)
    add_leave(auth_env.engine, clinic, doctor, (MONDAY, time(9, 30)), (MONDAY, time(10, 30)))

    response = search(auth_env, clinic, MONDAY, doctor_id=doctor.doctor_id)
    assert starts(response) == [local(MONDAY, time(9)), local(MONDAY, time(10, 30))]
    assert len(starts(search(auth_env, clinic, MONDAY, doctor_id=colleague.doctor_id))) == 4


def test_leave_recorded_at_one_clinic_does_not_hide_the_doctor_at_another_clinic(auth_env) -> None:
    clinic_a = new_clinic(auth_env.engine, hours=ALL_DAY, open_days=ALL_WEEK)
    clinic_b = new_clinic(auth_env.engine, hours=ALL_DAY, open_days=ALL_WEEK)
    doctor = new_doctor(auth_env.engine, clinic_a, rule_days=ALL_WEEK, rule_windows=(ALL_DAY,))
    assign_doctor(auth_env.engine, doctor, clinic_b)
    # One doctor cannot hold overlapping slots, so each clinic gets different times. Monday leave
    # is recorded at clinic A only and Tuesday leave at clinic B only; both cover 08:00-12:00.
    add_slots(auth_env.engine, clinic_a, doctor, [(MONDAY, time(9)), (TUESDAY, time(10))])
    add_slots(auth_env.engine, clinic_b, doctor, [(MONDAY, time(10)), (TUESDAY, time(9))])
    add_leave(auth_env.engine, clinic_a, doctor, (MONDAY, time(8)), (MONDAY, time(12)))
    add_leave(auth_env.engine, clinic_b, doctor, (TUESDAY, time(8)), (TUESDAY, time(12)))

    at_a = search(auth_env, clinic_a, MONDAY, TUESDAY)
    at_b = search(auth_env, clinic_b, MONDAY, TUESDAY)

    # Clinic A: Monday 09:00 is on leave there; Tuesday 10:00 is not.
    assert starts(at_a) == [local(TUESDAY, time(10))]
    # Clinic B: Monday 10:00 falls inside A's leave window but B has no leave then; Tuesday 09:00
    # is on leave at B.
    assert starts(at_b) == [local(MONDAY, time(10))]


def test_slots_outside_clinic_operating_hours_are_excluded(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)  # open Monday-Friday 08:00-17:00
    doctor = new_doctor(
        auth_env.engine, clinic, rule_days=ALL_WEEK, rule_windows=((time(7), time(19)),)
    )
    saturday = MONDAY + timedelta(days=5)
    add_slots(
        auth_env.engine,
        clinic,
        doctor,
        [
            (MONDAY, time(7, 30)),
            (MONDAY, time(8)),
            (MONDAY, time(16, 30)),
            (MONDAY, time(17)),
            (saturday, time(9)),
        ],
    )

    assert starts(search(auth_env, clinic, MONDAY, saturday)) == [
        local(MONDAY, time(8)),
        local(MONDAY, time(16, 30)),
    ]


def test_slots_outside_doctor_availability_rules_are_excluded(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic)  # 09:00-12:00 and 13:00-16:00 weekdays
    add_slots(
        auth_env.engine,
        clinic,
        doctor,
        [
            (MONDAY, time(8, 30)),
            (MONDAY, time(11, 30)),
            (MONDAY, time(12)),
            (MONDAY, time(12, 30)),
            (MONDAY, time(15, 30)),
            (MONDAY, time(16)),
        ],
    )

    assert starts(search(auth_env, clinic, MONDAY)) == [
        local(MONDAY, time(11, 30)),
        local(MONDAY, time(15, 30)),
    ]


@pytest.mark.parametrize(
    "rule",
    [
        {"rule_active": False},
        {"rule_days": [1]},  # Tuesday only; the slot is on Monday
        {"effective_end": MONDAY - timedelta(days=1)},
        {"effective_start": MONDAY + timedelta(days=1)},
    ],
    ids=["inactive-rule", "other-weekday", "rule-ended", "rule-not-started"],
)
def test_unusable_availability_rules_remove_the_slots(auth_env, rule) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic, **rule)
    add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9))])

    assert starts(search(auth_env, clinic, MONDAY)) == []


def test_rules_effective_on_the_search_date_are_honored_inclusively(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic, effective_start=MONDAY, effective_end=MONDAY)
    add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9)), (TUESDAY, time(9))])

    assert starts(search(auth_env, clinic, MONDAY, TUESDAY)) == [local(MONDAY, time(9))]


# --- active status and filters ---------------------------------------------------------------


def test_inactive_and_unknown_clinics_are_not_available(auth_env) -> None:
    inactive = new_clinic(auth_env.engine, active=False)
    doctor = new_doctor(auth_env.engine, inactive)
    add_slots(auth_env.engine, inactive, doctor, [(MONDAY, time(9))])
    unknown = new_clinic(auth_env.engine)
    unknown.clinic_id = uuid.uuid4()

    for clinic in (inactive, unknown):
        response = search(auth_env, clinic, MONDAY)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_AVAILABLE"


def test_inactive_doctors_and_inactive_assignments_hide_their_slots(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    visible = new_doctor(auth_env.engine, clinic)
    inactive_doctor = new_doctor(auth_env.engine, clinic, active=False)
    inactive_assignment = new_doctor(auth_env.engine, clinic, assignment_active=False)
    for doctor in (visible, inactive_doctor, inactive_assignment):
        add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9))])

    response = search(auth_env, clinic, MONDAY)

    assert [slot["doctor"]["doctor_id"] for slot in response.json()["slots"]] == [
        str(visible.doctor_id)
    ]


def test_doctor_filter_requires_an_active_assignment_at_the_clinic(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    elsewhere = new_clinic(auth_env.engine)
    stranger = new_doctor(auth_env.engine, elsewhere)
    retired = new_doctor(auth_env.engine, clinic, assignment_active=False)
    inactive = new_doctor(auth_env.engine, clinic, active=False)

    for doctor in (stranger, retired, inactive, uuid.uuid4()):
        doctor_id = getattr(doctor, "doctor_id", doctor)
        response = search(auth_env, clinic, MONDAY, doctor_id=doctor_id)
        assert response.status_code == 404, doctor_id


def test_specialty_and_doctor_filters_narrow_results(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    heart = new_doctor(auth_env.engine, clinic, specialty="Cardiology")
    skin = new_doctor(auth_env.engine, clinic, specialty="Dermatology")
    for doctor in (heart, skin):
        add_slots(auth_env.engine, clinic, doctor, [(MONDAY, time(9))])

    by_specialty = search(auth_env, clinic, MONDAY, specialty="cardiology")
    by_doctor = search(auth_env, clinic, MONDAY, doctor_id=skin.doctor_id)

    assert [s["doctor"]["doctor_id"] for s in by_specialty.json()["slots"]] == [
        str(heart.doctor_id)
    ]
    assert [s["doctor"]["doctor_id"] for s in by_doctor.json()["slots"]] == [str(skin.doctor_id)]
    assert starts(search(auth_env, clinic, MONDAY, specialty="Neurology")) == []


def test_time_preferences_use_clinic_local_boundaries(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)
    doctor = new_doctor(auth_env.engine, clinic, rule_windows=((time(8), time(17)),))
    moments = [(MONDAY, time(h, m)) for h, m in ((8, 0), (11, 30), (12, 0), (16, 30))]
    add_slots(auth_env.engine, clinic, doctor, moments)

    assert starts(search(auth_env, clinic, MONDAY, time_preference="MORNING")) == [
        local(MONDAY, time(8)),
        local(MONDAY, time(11, 30)),
    ]
    assert starts(search(auth_env, clinic, MONDAY, time_preference="AFTERNOON")) == [
        local(MONDAY, time(12)),
        local(MONDAY, time(16, 30)),
    ]
    assert len(starts(search(auth_env, clinic, MONDAY, time_preference="ANYTIME"))) == 4
    assert len(starts(search(auth_env, clinic, MONDAY))) == 4


# --- request validation ----------------------------------------------------------------------


def test_search_windows_are_limited_to_thirty_one_days(auth_env) -> None:
    clinic = new_clinic(auth_env.engine)

    assert search(auth_env, clinic, MONDAY, MONDAY + timedelta(days=30)).status_code == 200
    too_long = search(auth_env, clinic, MONDAY, MONDAY + timedelta(days=31))
    backwards = search(auth_env, clinic, MONDAY, MONDAY - timedelta(days=1))

    assert too_long.status_code == backwards.status_code == 422
    assert too_long.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert "31 days" in too_long.json()["error"]["message"]
    assert backwards.json()["error"]["code"] == "INVALID_ARGUMENT"


DAY = {"start_date": "2026-12-07", "end_date": "2026-12-07"}


@pytest.mark.parametrize(
    ("params", "echoed"),
    [
        ({}, None),
        ({"clinic_id": "not-a-uuid", **DAY}, "not-a-uuid"),
        (DAY, None),
        (
            {"clinic_id": "CLINIC", "start_date": "07/12/2026", "end_date": "2026-12-07"},
            "07/12/2026",
        ),
        ({"clinic_id": "CLINIC", "start_date": "2026-12-07"}, None),
        ({"clinic_id": "CLINIC", **DAY, "time_preference": "EVENING"}, "EVENING"),
        ({"clinic_id": "CLINIC", **DAY, "limit": "21"}, None),
        ({"clinic_id": "CLINIC", **DAY, "limit": "0"}, None),
        ({"clinic_id": "CLINIC", **DAY, "patient_id": "sentinel-patient"}, "sentinel-patient"),
        ({"clinic_id": "CLINIC", **DAY, "doctor_id": "sentinel-doctor"}, "sentinel-doctor"),
    ],
)
def test_malformed_availability_requests_are_rejected_without_echoing_input(
    auth_env, params, echoed
) -> None:
    clinic = new_clinic(auth_env.engine)
    query = {k: (str(clinic.clinic_id) if v == "CLINIC" else v) for k, v in params.items()}

    response = auth_env.client.get("/availability", params=query)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    if echoed:
        assert echoed not in response.text


# --- pagination ------------------------------------------------------------------------------


def three_days_of_slots(env, doctors=1):
    clinic = new_clinic(env.engine)
    made = [
        new_doctor(env.engine, clinic, rule_windows=((time(8), time(17)),)) for _ in range(doctors)
    ]
    moments = [
        m for offset in range(3) for m in half_hours(MONDAY + timedelta(days=offset), time(8), 15)
    ]
    for doctor in made:
        add_slots(env.engine, clinic, doctor, moments)
    return clinic, made


def collect(env, clinic, limit, **params):
    pages, cursor = [], None
    while True:
        extra = {"limit": limit, **params, **({"cursor": cursor} if cursor else {})}
        body = search(env, clinic, MONDAY, MONDAY + timedelta(days=2), **extra).json()
        pages.append(body["slots"])
        cursor = body["next_cursor"]
        if not cursor:
            return pages


def test_default_page_size_is_twenty_with_stable_ordering(auth_env) -> None:
    clinic, _ = three_days_of_slots(auth_env)

    pages = collect(auth_env, clinic, 20)

    assert [len(page) for page in pages] == [20, 20, 5]
    flat = [slot for page in pages for slot in page]
    assert len({slot["slot_id"] for slot in flat}) == 45
    assert [s["starts_at_utc"] for s in flat] == sorted(s["starts_at_utc"] for s in flat)


def test_pages_never_skip_or_repeat_slots_that_share_a_start_time(auth_env) -> None:
    clinic, doctors = three_days_of_slots(auth_env, doctors=2)

    pages = collect(auth_env, clinic, 1)
    flat = [slot for page in pages for slot in page]

    assert len(flat) == 90 and len({s["slot_id"] for s in flat}) == 90
    keys = [(s["starts_at_utc"], s["slot_id"]) for s in flat]
    assert keys == sorted(keys)
    assert {s["doctor"]["doctor_id"] for s in flat} == {str(d.doctor_id) for d in doctors}


def test_cursors_are_bound_to_their_search_and_reject_tampering(auth_env) -> None:
    clinic, _ = three_days_of_slots(auth_env)
    first = search(auth_env, clinic, MONDAY, MONDAY + timedelta(days=2), limit=5).json()
    cursor = first["next_cursor"]

    different_filter = search(
        auth_env,
        clinic,
        MONDAY,
        MONDAY + timedelta(days=2),
        limit=5,
        cursor=cursor,
        time_preference="MORNING",
    )
    different_window = search(
        auth_env, clinic, MONDAY, MONDAY + timedelta(days=3), limit=5, cursor=cursor
    )
    garbage = search(auth_env, clinic, MONDAY, limit=5, cursor="not-a-cursor")
    oversized = search(auth_env, clinic, MONDAY, limit=5, cursor="A" * 600)

    for rejected in (different_filter, different_window, garbage, oversized):
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "INVALID_ARGUMENT"


# --- daylight saving and time zones ----------------------------------------------------------


def all_day_clinic(env, timezone):
    clinic = new_clinic(env.engine, timezone=timezone, hours=ALL_DAY, open_days=ALL_WEEK)
    doctor = new_doctor(env.engine, clinic, rule_days=ALL_WEEK, rule_windows=(ALL_DAY,))
    env.clock.now = datetime(2026, 1, 1, tzinfo=UTC)
    return clinic, doctor


def slot_view(response) -> list[tuple[str, str]]:
    return [(s["starts_at_local"], s["starts_at_utc"]) for s in response.json()["slots"]]


def test_slots_show_the_correct_offset_before_and_after_the_fall_back_change(auth_env) -> None:
    clinic, doctor = all_day_clinic(auth_env, "America/Denver")
    before, after = date(2026, 10, 30), date(2026, 11, 2)
    add_slots(auth_env.engine, clinic, doctor, [(before, time(9)), (after, time(9))])

    response = search(auth_env, clinic, before, after)

    assert slot_view(response) == [
        ("2026-10-30T09:00:00-06:00", "2026-10-30T15:00:00Z"),
        ("2026-11-02T09:00:00-07:00", "2026-11-02T16:00:00Z"),
    ]
    assert response.json()["slots"][0]["clinic_timezone"] == "America/Denver"


def test_the_25_hour_fall_back_day_contains_exactly_its_own_slots(auth_env) -> None:
    clinic, doctor = all_day_clinic(auth_env, "America/Denver")
    change_day = date(2026, 11, 1)  # 01:00-02:00 happens twice
    add_slots(auth_env.engine, clinic, doctor, [(date(2026, 10, 31), time(23))])
    add_slots(
        auth_env.engine, clinic, doctor, [(change_day, time(0, 30)), (change_day, time(1, 30))]
    )
    second_half_hour = localize(clinic.zone, change_day, time(1, 30)) + timedelta(hours=1)
    add_slots(auth_env.engine, clinic, doctor, [(change_day, time(23))])
    with auth_env.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO slots (doctor_id, clinic_id, starts_at, ends_at)"
                " VALUES (:d, :c, :s, :e)"
            ),
            {
                "d": doctor.doctor_id,
                "c": clinic.clinic_id,
                "s": second_half_hour,
                "e": second_half_hour + timedelta(minutes=30),
            },
        )

    on_change_day = slot_view(search(auth_env, clinic, change_day))

    assert on_change_day == [
        ("2026-11-01T00:30:00-06:00", "2026-11-01T06:30:00Z"),
        ("2026-11-01T01:30:00-06:00", "2026-11-01T07:30:00Z"),
        ("2026-11-01T01:30:00-07:00", "2026-11-01T08:30:00Z"),
        ("2026-11-01T23:00:00-07:00", "2026-11-02T06:00:00Z"),
    ]
    assert slot_view(search(auth_env, clinic, date(2026, 10, 31))) == [
        ("2026-10-31T23:00:00-06:00", "2026-11-01T05:00:00Z")
    ]
    assert slot_view(search(auth_env, clinic, date(2026, 11, 2))) == []


def test_the_23_hour_spring_forward_day_skips_the_missing_hour(auth_env) -> None:
    clinic, doctor = all_day_clinic(auth_env, "America/Denver")
    change_day = date(2027, 3, 14)  # 02:00-03:00 does not exist
    add_slots(auth_env.engine, clinic, doctor, [(date(2027, 3, 13), time(23))])
    add_slots(
        auth_env.engine,
        clinic,
        doctor,
        [(change_day, time(1, 30)), (change_day, time(3, 0)), (change_day, time(23))],
    )

    assert slot_view(search(auth_env, clinic, change_day)) == [
        ("2027-03-14T01:30:00-07:00", "2027-03-14T08:30:00Z"),
        ("2027-03-14T03:00:00-06:00", "2027-03-14T09:00:00Z"),
        ("2027-03-14T23:00:00-06:00", "2027-03-15T05:00:00Z"),
    ]


@pytest.mark.parametrize(
    ("timezone", "day"),
    [
        ("America/Los_Angeles", date(2026, 11, 2)),
        ("America/Chicago", date(2027, 3, 15)),
        ("Europe/London", date(2026, 10, 26)),
        ("Europe/London", date(2026, 7, 1)),
        ("Asia/Kolkata", date(2026, 12, 7)),
        ("Australia/Lord_Howe", date(2026, 10, 5)),
        ("Pacific/Auckland", date(2026, 12, 7)),
    ],
)
def test_local_times_convert_exactly_for_other_time_zones(auth_env, timezone, day) -> None:
    clinic, doctor = all_day_clinic(auth_env, timezone)
    add_slots(auth_env.engine, clinic, doctor, [(day, time(9)), (day, time(14, 30))])
    zone = ZoneInfo(timezone)

    response = search(auth_env, clinic, day)

    expected = [datetime.combine(day, time(9), zone), datetime.combine(day, time(14, 30), zone)]
    slots = response.json()["slots"]
    for slot, moment in zip(slots, expected, strict=True):
        local_value = datetime.fromisoformat(slot["starts_at_local"])
        assert local_value == moment and local_value.utcoffset() == moment.utcoffset()
        assert slot["starts_at_utc"] == moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert slot["clinic_timezone"] == timezone


# --- seeded synthetic data -------------------------------------------------------------------


def test_seeded_availability_excludes_booked_slots_and_respects_closures_and_leave(
    auth_env,
) -> None:
    with auth_env.engine.connect() as connection:
        clinics = {
            row.name: row
            for row in connection.execute(
                text("SELECT clinic_id, name, timezone FROM clinics WHERE name = ANY(:names)"),
                {"names": list(SEEDED_CLINICS)},
            ).all()
        }
        booked = {
            row.slot_id
            for row in connection.execute(
                text(
                    "SELECT s.slot_id FROM slots s JOIN clinics c USING (clinic_id)"
                    " WHERE s.status = 'BOOKED' AND c.name = ANY(:names)"
                ),
                {"names": list(SEEDED_CLINICS)},
            )
        }
    assert len(booked) == 8
    window = (date(2026, 11, 2), date(2026, 11, 20))

    returned_ids, riverbend_days, lindqvist_days = set(), set(), set()
    for clinic in clinics.values():
        cursor = None
        while True:
            params = {"limit": 20, **({"cursor": cursor} if cursor else {})}
            response = search(auth_env, clinic, *window, **params)
            assert response.status_code == 200, response.text
            body = response.json()
            for slot in body["slots"]:
                returned_ids.add(uuid.UUID(slot["slot_id"]))
                day = slot["starts_at_local"][:10]
                if clinic.name == "Riverbend Family Health":
                    riverbend_days.add(day)
                if slot["doctor"]["display_name"] == "Dr. Sofia Lindqvist":
                    lindqvist_days.add(day)
                assert datetime.fromisoformat(slot["starts_at_utc"]) > auth_env.clock.now
            cursor = body["next_cursor"]
            if not cursor:
                break

    assert returned_ids and not returned_ids & booked
    assert "2026-11-09" not in riverbend_days  # Riverbend staff training day
    assert riverbend_days  # ...but other days are open
    assert not {f"2026-11-{day}" for day in range(7, 14)} & lindqvist_days  # a week of leave
    assert lindqvist_days
