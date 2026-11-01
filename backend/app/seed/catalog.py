"""Static fictional definitions. Weekdays use Monday=0 through Sunday=6 (date.weekday())."""

from dataclasses import dataclass
from datetime import time

MON, TUE, WED, THU, FRI, SAT = range(6)


@dataclass(frozen=True)
class ClinicSpec:
    key: str
    name: str
    timezone: str
    address_line: str
    city: str
    region: str
    postal_code: str
    hours: dict[int, tuple[time, time]]


@dataclass(frozen=True)
class DoctorSpec:
    key: str
    display_name: str
    specialty: str


@dataclass(frozen=True)
class AssignmentSpec:
    doctor: str
    clinic: str
    weekdays: tuple[int, ...]
    windows: tuple[tuple[time, time], ...]


@dataclass(frozen=True)
class LeaveSpec:
    doctor: str
    clinic: str
    start_offset: int
    end_offset: int
    reason_code: str


@dataclass(frozen=True)
class ClosureSpec:
    """A demonstration closure on the first open day on or after anchor + offset."""

    clinic: str
    offset: int
    label: str
    closed_from: time | None = None
    closed_until: time | None = None


@dataclass(frozen=True)
class PersonSpec:
    key: str
    login: str
    display_name: str


@dataclass(frozen=True)
class AppointmentSpec:
    key: str
    patient: str
    doctor: str
    clinic: str
    min_offset: int
    slot_index: int
    booked_days_ago: int
    cancelled_days_ago: int | None = None


@dataclass(frozen=True)
class WaitlistSpec:
    key: str
    patient: str
    clinic: str
    specialty: str
    start_offset: int
    end_offset: int
    time_preference: str
    created_days_ago: int


@dataclass(frozen=True)
class HandoffSpec:
    key: str
    patient: str | None
    category: str
    status: str
    summary: str
    created_days_ago: int
    resolved_days_ago: int | None = None
    note: str | None = None


WEEKDAY_HOURS = (time(8), time(17))
SPLIT_DAY = ((time(9), time(12)), (time(13), time(16)))

CLINICS = (
    ClinicSpec(
        key="riverbend",
        name="Riverbend Family Health",
        timezone="America/Denver",
        address_line="412 Aspen Mill Road",
        city="Aspenfield",
        region="CO",
        postal_code="80901",
        hours={day: WEEKDAY_HOURS for day in (MON, TUE, WED, THU, FRI)},
    ),
    ClinicSpec(
        key="lakeside",
        name="Lakeside Community Clinic",
        timezone="America/Chicago",
        address_line="88 Harbor Light Avenue",
        city="Port Linden",
        region="IL",
        postal_code="60601",
        hours={
            **{day: (time(7, 30), time(16, 30)) for day in (MON, TUE, WED, THU, FRI)},
            SAT: (time(9), time(13)),
        },
    ),
    ClinicSpec(
        key="harborview",
        name="Harborview Medical Associates",
        timezone="America/Los_Angeles",
        address_line="2150 Cypress Bluff Drive",
        city="Marlow Bay",
        region="CA",
        postal_code="90401",
        hours={
            **{day: (time(8), time(18)) for day in (MON, TUE, WED, THU)},
            FRI: (time(8), time(15)),
        },
    ),
)

DOCTORS = (
    DoctorSpec("hartwell", "Dr. Amelia Hartwell", "Family Medicine"),
    DoctorSpec("desai", "Dr. Rohan Desai", "Internal Medicine"),
    DoctorSpec("nambiar", "Dr. Priya Nambiar", "Pediatrics"),
    DoctorSpec("ellery", "Dr. Marcus Ellery", "Cardiology"),
    DoctorSpec("lindqvist", "Dr. Sofia Lindqvist", "Dermatology"),
    DoctorSpec("okafor", "Dr. Tomas Okafor", "Orthopedics"),
    DoctorSpec("whitlock", "Dr. Grace Whitlock", "Family Medicine"),
    DoctorSpec("vasquez", "Dr. Elena Vasquez", "Pediatrics"),
)

# A doctor's weekdays never repeat across clinics, so no two assignments can overlap in time.
ASSIGNMENTS = (
    AssignmentSpec("hartwell", "riverbend", (MON, TUE, WED, THU), SPLIT_DAY),
    AssignmentSpec("desai", "riverbend", (WED, FRI), ((time(8), time(12)), (time(13), time(16)))),
    AssignmentSpec("desai", "lakeside", (MON, TUE), ((time(8), time(12)), (time(13), time(16)))),
    AssignmentSpec(
        "nambiar", "lakeside", (MON, WED, THU, FRI), ((time(8), time(12)), (time(13), time(16)))
    ),
    AssignmentSpec(
        "vasquez",
        "riverbend",
        (TUE, THU, FRI),
        ((time(8, 30), time(12)), (time(13), time(16, 30))),
    ),
    AssignmentSpec("ellery", "lakeside", (TUE, THU), SPLIT_DAY),
    AssignmentSpec("ellery", "harborview", (MON, WED), ((time(9), time(12)), (time(13), time(17)))),
    AssignmentSpec(
        "lindqvist", "harborview", (TUE, THU, FRI), ((time(9), time(12)), (time(13), time(15)))
    ),
    AssignmentSpec(
        "okafor",
        "harborview",
        (MON, TUE, WED, THU),
        ((time(8, 30), time(12)), (time(13), time(17))),
    ),
    AssignmentSpec(
        "whitlock", "harborview", (MON, TUE, WED, THU), ((time(8), time(12)), (time(13), time(17)))
    ),
)

LEAVES = (
    LeaveSpec("lindqvist", "harborview", 10, 16, "VACATION"),
    LeaveSpec("hartwell", "riverbend", 8, 9, "CONFERENCE"),
    LeaveSpec("nambiar", "lakeside", 3, 3, "PERSONAL"),
)

CLOSURES = (
    ClosureSpec("riverbend", 10, "Staff training day"),
    ClosureSpec(
        "lakeside", 12, "Community health fair", closed_from=time(13), closed_until=time(16, 30)
    ),
    ClosureSpec("harborview", 14, "Facilities maintenance day"),
)

PATIENTS = (
    PersonSpec("p01", "PT-1001", "Avery Calloway"),
    PersonSpec("p02", "PT-1002", "Jordan Fairweather"),
    PersonSpec("p03", "PT-1003", "Maya Thornbury"),
    PersonSpec("p04", "PT-1004", "Lucas Brightwater"),
    PersonSpec("p05", "PT-1005", "Noor Aldana"),
    PersonSpec("p06", "PT-1006", "Eli Marchetti"),
    PersonSpec("p07", "PT-1007", "Sienna Okoro"),
    PersonSpec("p08", "PT-1008", "Caleb Hawthorne"),
    PersonSpec("p09", "PT-1009", "Isabel Quintero"),
    PersonSpec("p10", "PT-1010", "Theo Lindgren"),
    PersonSpec("p11", "PT-1011", "Harper Nakamura"),
    PersonSpec("p12", "PT-1012", "Wesley Abernathy"),
    PersonSpec("p13", "PT-1013", "Zara Whitfield"),
    PersonSpec("p14", "PT-1014", "Mateo Santoro"),
)

STAFF = (
    PersonSpec("s01", "staff.morgan", "Jamie Morgan"),
    PersonSpec("s02", "staff.ellis", "Riley Ellis"),
)

APPOINTMENTS = (
    AppointmentSpec("confirmed-01", "p01", "hartwell", "riverbend", 2, 1, 4),
    AppointmentSpec("confirmed-02", "p02", "hartwell", "riverbend", 2, 2, 3),
    AppointmentSpec("confirmed-03", "p03", "nambiar", "lakeside", 1, 0, 5),
    AppointmentSpec("confirmed-04", "p04", "ellery", "lakeside", 3, 4, 2),
    AppointmentSpec("confirmed-05", "p05", "okafor", "harborview", 4, 0, 6),
    AppointmentSpec("confirmed-06", "p06", "whitlock", "harborview", 5, 3, 1),
    AppointmentSpec("confirmed-07", "p07", "vasquez", "riverbend", 6, 2, 2),
    AppointmentSpec("confirmed-08", "p01", "desai", "lakeside", 7, 5, 3),
    AppointmentSpec("cancelled-01", "p08", "nambiar", "lakeside", 9, 1, 7, 5),
    AppointmentSpec("cancelled-02", "p09", "lindqvist", "harborview", 4, 2, 8, 6),
    AppointmentSpec("cancelled-03", "p10", "desai", "riverbend", 8, 0, 6, 4),
)

WAITLIST = (
    WaitlistSpec("waitlist-01", "p11", "lakeside", "Cardiology", 1, 14, "MORNING", 3),
    WaitlistSpec("waitlist-02", "p12", "harborview", "Orthopedics", 3, 21, "ANYTIME", 2),
    WaitlistSpec("waitlist-03", "p13", "riverbend", "Family Medicine", 1, 10, "AFTERNOON", 1),
)

HANDOFFS = (
    HandoffSpec(
        "handoff-01",
        None,
        "SCHEDULING",
        "OPEN",
        "Visitor asked how to arrange a follow-up visit after a recent appointment.",
        2,
    ),
    HandoffSpec(
        "handoff-02",
        "p03",
        "TECHNICAL_SUPPORT",
        "OPEN",
        "Patient reported difficulty viewing upcoming appointment details.",
        1,
    ),
    HandoffSpec(
        "handoff-03",
        None,
        "ACCESSIBILITY",
        "IN_PROGRESS",
        "Visitor requested information about wheelchair-accessible entrances.",
        4,
        note="Reviewing accessibility information with the front desk.",
    ),
    HandoffSpec(
        "handoff-04",
        None,
        "CLINIC_INFORMATION",
        "RESOLVED",
        "Visitor asked about weekend hours at the Lakeside clinic.",
        9,
        resolved_days_ago=7,
        note="Provided the published Saturday hours.",
    ),
)

STAFF_AUTHOR_FOR_NOTES = "s01"
