"""Executable scenarios for the frozen Step 4B holdout manifest.

The source of requirements is backend/tests/agent-reliability-holdout-v1.json. These scenarios are
not imported by the baseline or the two-scenario pilot and must not be used for prompt tuning.
"""

from datetime import timedelta
from zoneinfo import ZoneInfo

from app.evaluation import model as claims
from app.evaluation.fixtures import (
    book_slot_for,
    create_clinic,
    create_patient,
    display_time,
    local_start,
    patient_id_of,
    patient_identity,
    unique_place_name,
)
from app.evaluation.model import Say, ToolAction
from app.evaluation.scenario import (
    PUBLIC,
    DbCheck,
    ExpectedCall,
    Fixtures,
    ModelTurn,
    PendingIntent,
    Pick,
    Ref,
    ResultRef,
    Said,
    Scenario,
    SetupContext,
    UiConfirm,
    World,
    WorldContext,
)

DELIVERY = "Phase 2 frozen holdout v1"
DENVER = "America/Denver"
HONOLULU = "Pacific/Honolulu"
SEARCH = "search_clinic_directory"
AVAILABILITY = "search_appointment_availability"
MINE = "get_my_appointments"
PREPARE = "prepare_appointment_action"
REVIEW = "I prepared the requested change. Review the details and confirm in the app."


def _clinic_name(ctx: SetupContext, prefix: str) -> str:
    return unique_place_name(ctx.engine, prefix, ctx.suffix)


def _local_date(instant, timezone: str) -> str:
    return instant.astimezone(ZoneInfo(timezone)).date().isoformat()


def _book_intent(slot: str = "target_slot") -> PendingIntent:
    return PendingIntent(
        action="BOOK",
        target=Ref(slot),
        summary={
            "operation": "BOOK",
            "proposed.clinic": Ref("clinic_name"),
            "proposed.doctor": Ref("target_doctor_name"),
            "proposed.date": Ref("day"),
            "proposed.time": Ref("target_time"),
        },
    )


def _booking_scenario(
    *,
    scenario_id: str,
    title: str,
    request: str,
    request_terms: tuple[str, ...],
    setup,
    directory_arguments: dict,
    availability_arguments: dict,
    expected_directory: dict,
    expected_availability: dict,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        title=title,
        delivery=DELIVERY,
        request=request,
        request_terms=request_terms,
        actor="patient",
        setup=setup,
        model_actions=[
            ToolAction(SEARCH, directory_arguments),
            ToolAction(AVAILABILITY, availability_arguments),
            ToolAction(
                PREPARE,
                {"action": "BOOK", "slot_id": ResultRef(2, "data.slots.0.slot_id")},
            ),
            Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
            Say(frozenset({claims.BOOKED}), "Your appointment is booked."),
        ],
        timeline=[ModelTurn(), UiConfirm(tool=PREPARE, intent=_book_intent()), ModelTurn()],
        expected_calls=[
            ExpectedCall(SEARCH, expected_directory),
            ExpectedCall(AVAILABILITY, expected_availability),
            ExpectedCall(PREPARE, {"action": "BOOK", "slot_id": Ref("target_slot")}),
        ],
        expected_claims=frozenset({claims.BOOKED}),
        db_checks=[
            DbCheck(
                "the intended slot is BOOKED",
                "SELECT status FROM slots WHERE slot_id = :slot",
                "BOOKED",
                {"slot": Ref("target_slot")},
            ),
            DbCheck(
                "the verified patient owns one confirmed appointment on the intended slot",
                "SELECT count(*) FROM appointments"
                " WHERE slot_id = :slot AND patient_id = :patient AND status = 'CONFIRMED'",
                1,
                {"slot": Ref("target_slot"), "patient": Ref("patient")},
            ),
        ],
    )


def _setup_hr001(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout North Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Rowan Hale {ctx.suffix}", "Family Medicine")],
        slot_starts=[start, start + timedelta(minutes=30)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H1")
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _local_date(start, DENVER),
            "target_slot": clinic.slot_ids[0][0],
            "target_doctor_name": clinic.doctor_names[0],
            "target_time": display_time(start, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot for group in clinic.slot_ids for slot in group],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_001 = _booking_scenario(
    scenario_id="HR-001",
    title="First available at a single-doctor clinic",
    request="Please arrange the earliest opening at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    setup=_setup_hr001,
    directory_arguments={"clinic_name": Said("clinic_name")},
    availability_arguments={
        "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
        "start_date": Said("day"),
        "end_date": Said("day"),
    },
    expected_directory={"clinic_name": Ref("clinic_name")},
    expected_availability={
        "clinic_id": Ref("clinic"),
        "start_date": Ref("day"),
        "end_date": Ref("day"),
    },
)


def _setup_hr002(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Broad Clinic"),
        timezone=DENVER,
        doctors=[
            (f"Dr. Morgan Reed {ctx.suffix}", "Primary Care"),
            (f"Dr. Riley Stone {ctx.suffix}", "Dermatology"),
        ],
        doctor_slot_starts=[[start + timedelta(hours=1)], [start]],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H2")
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _local_date(start, DENVER),
            "target_slot": clinic.slot_ids[1][0],
            "target_doctor_name": clinic.doctor_names[1],
            "target_time": display_time(start, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot for group in clinic.slot_ids for slot in group],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_002 = _booking_scenario(
    scenario_id="HR-002",
    title="First available across multiple specialties",
    request="Book whichever appointment is first at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    setup=_setup_hr002,
    directory_arguments={"clinic_name": Said("clinic_name")},
    availability_arguments={
        "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
        "start_date": Said("day"),
        "end_date": Said("day"),
    },
    expected_directory={"clinic_name": Ref("clinic_name")},
    expected_availability={
        "clinic_id": Ref("clinic"),
        "start_date": Ref("day"),
        "end_date": Ref("day"),
    },
)


def _setup_hr003(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Doctor Clinic"),
        timezone=DENVER,
        doctors=[
            (f"Dr. Casey Wynn {ctx.suffix}", "Family Medicine"),
            (f"Dr. Jordan Pike {ctx.suffix}", "Internal Medicine"),
        ],
        doctor_slot_starts=[[start], [start + timedelta(hours=1), start + timedelta(minutes=90)]],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H3")
    target = start + timedelta(hours=1)
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "doctor_name": clinic.doctor_names[1],
            "doctor": clinic.doctor_ids[1],
            "day": _local_date(start, DENVER),
            "target_slot": clinic.slot_ids[1][0],
            "target_doctor_name": clinic.doctor_names[1],
            "target_time": display_time(target, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot for group in clinic.slot_ids for slot in group],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_003 = _booking_scenario(
    scenario_id="HR-003",
    title="Explicit doctor booking",
    request=(
        "Find the first appointment with {doctor_name} at {clinic_name} on {day} and book it."
    ),
    request_terms=("doctor_name", "clinic_name", "day"),
    setup=_setup_hr003,
    directory_arguments={
        "doctor_name": Said("doctor_name"),
        "clinic_name": Said("clinic_name"),
    },
    availability_arguments={
        "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
        "doctor_id": ResultRef(1, "data.doctors.0.doctor_id"),
        "start_date": Said("day"),
        "end_date": Said("day"),
    },
    expected_directory={"doctor_name": Ref("doctor_name"), "clinic_name": Ref("clinic_name")},
    expected_availability={
        "clinic_id": Ref("clinic"),
        "doctor_id": Ref("doctor"),
        "start_date": Ref("day"),
        "end_date": Ref("day"),
    },
)


def _setup_hr004(ctx: SetupContext) -> Fixtures:
    morning = local_start(ctx.now, DENVER, hour=9)
    afternoon = local_start(ctx.now, DENVER, hour=14)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Specialty Clinic"),
        timezone=DENVER,
        doctors=[
            (f"Dr. Avery Brook {ctx.suffix}", "Family Medicine"),
            (f"Dr. Taylor Quinn {ctx.suffix}", "Cardiology"),
        ],
        doctor_slot_starts=[[morning], [afternoon]],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H4")
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "specialty": "Cardiology",
            "day": _local_date(morning, DENVER),
            "target_slot": clinic.slot_ids[1][0],
            "target_doctor_name": clinic.doctor_names[1],
            "target_time": display_time(afternoon, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot for group in clinic.slot_ids for slot in group],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_004 = _booking_scenario(
    scenario_id="HR-004",
    title="Explicit specialty booking",
    request=("Book the earliest {specialty} visit at {clinic_name} during the afternoon of {day}."),
    request_terms=("specialty", "clinic_name", "day"),
    setup=_setup_hr004,
    directory_arguments={"clinic_name": Said("clinic_name"), "specialty": Said("specialty")},
    availability_arguments={
        "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
        "specialty": Said("specialty"),
        "time_preference": "AFTERNOON",
        "start_date": Said("day"),
        "end_date": Said("day"),
    },
    expected_directory={"clinic_name": Ref("clinic_name"), "specialty": Ref("specialty")},
    expected_availability={
        "clinic_id": Ref("clinic"),
        "specialty": Ref("specialty"),
        "time_preference": "AFTERNOON",
        "start_date": Ref("day"),
        "end_date": Ref("day"),
    },
)


def _setup_hr005(ctx: SetupContext) -> Fixtures:
    requested = local_start(ctx.now, DENVER, hour=9)
    later = requested + timedelta(days=1)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Empty Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Empty Day {ctx.suffix}", "Family Medicine")],
        slot_starts=[later],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H5")
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _local_date(requested, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[clinic.slot_ids[0][0]],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_005 = Scenario(
    id="HR-005",
    title="No matching availability",
    delivery=DELIVERY,
    request="I need an appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_hr005,
    model_actions=[
        ToolAction(SEARCH, {"clinic_name": Said("clinic_name")}),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
                "start_date": Said("day"),
                "end_date": Said("day"),
            },
        ),
        Say(frozenset(), "No appointments are available on that date."),
    ],
    timeline=[ModelTurn()],
    expected_calls=[
        ExpectedCall(SEARCH, {"clinic_name": Ref("clinic_name")}),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
            outcome="NO_AVAILABILITY",
        ),
    ],
    expected_claims=frozenset(),
    forbidden_claims=frozenset(claims.SUCCESS_CLAIMS | {claims.PENDING_REVIEW}),
    db_checks=[
        DbCheck(
            "no pending action exists for the patient",
            "SELECT count(*) FROM pending_actions p JOIN auth_sessions s USING (session_id)"
            " WHERE s.patient_id = :patient AND p.created_at >= :scenario_start",
            0,
            {"patient": Ref("patient")},
        )
    ],
)


def _setup_hr006(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, HONOLULU, hour=10)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Ambiguous Clinic"),
        timezone=HONOLULU,
        doctors=[
            (f"Dr. Lee Harbor {ctx.suffix}", "Family Medicine"),
            (f"Dr. Lee Summit {ctx.suffix}", "Internal Medicine"),
        ],
        doctor_slot_starts=[[start], [start + timedelta(hours=1)]],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H6")
    first, second = clinic.slot_ids[0][0], clinic.slot_ids[1][0]
    book_slot_for(ctx.engine, code, first)
    book_slot_for(ctx.engine, code, second)
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "doctor_term": "Lee",
            "doctor_a": clinic.doctor_ids[0],
            "doctor_b": clinic.doctor_ids[1],
            "slot_a": first,
            "slot_b": second,
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[first, second],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_006 = Scenario(
    id="HR-006",
    title="Ambiguous doctor clarification",
    delivery=DELIVERY,
    request="Move my visit with Dr. {doctor_term} at {clinic_name}.",
    request_terms=("doctor_term", "clinic_name"),
    actor="patient",
    setup=_setup_hr006,
    model_actions=[
        ToolAction(
            SEARCH,
            {"doctor_name": Said("doctor_term"), "clinic_name": Said("clinic_name")},
        ),
        Say(frozenset({claims.ASK_CLARIFICATION}), "Two doctors match. Which one do you mean?"),
    ],
    timeline=[ModelTurn()],
    expected_calls=[
        ExpectedCall(
            SEARCH,
            {"doctor_name": Ref("doctor_term"), "clinic_name": Ref("clinic_name")},
            outcome="AMBIGUOUS_MATCH",
        )
    ],
    expected_claims=frozenset({claims.ASK_CLARIFICATION}),
    forbidden_claims=frozenset(claims.SUCCESS_CLAIMS | {claims.PENDING_REVIEW}),
    forbidden_tools=frozenset({PREPARE}),
    db_checks=[
        DbCheck(
            "both appointments remain confirmed",
            "SELECT count(*) FROM appointments WHERE slot_id IN (:a, :b) AND status = 'CONFIRMED'",
            2,
            {"a": Ref("slot_a"), "b": Ref("slot_b")},
        )
    ],
)


def _setup_hr007(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=11)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Public Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Public Test {ctx.suffix}", "Family Medicine")],
        slot_starts=[start],
    )
    slot = clinic.slot_ids[0][0]
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _local_date(start, DENVER),
            "slot": slot,
        },
        slots=[slot],
        protected=patient_identity(ctx.engine, "PT-1001"),
    )


HR_007 = Scenario(
    id="HR-007",
    title="Anonymous booking request",
    delivery=DELIVERY,
    request="Reserve the first opening at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor=PUBLIC,
    setup=_setup_hr007,
    model_actions=[
        ToolAction(SEARCH, {"clinic_name": Said("clinic_name")}),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
                "start_date": Said("day"),
                "end_date": Said("day"),
            },
        ),
        ToolAction(PREPARE, {"action": "BOOK", "slot_id": ResultRef(2, "data.slots.0.slot_id")}),
        Say(frozenset({claims.REQUEST_VERIFICATION}), "Please sign in before booking."),
    ],
    timeline=[ModelTurn()],
    expected_calls=[
        ExpectedCall(SEARCH, {"clinic_name": Ref("clinic_name")}),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(PREPARE, {"action": "BOOK", "slot_id": Ref("slot")}, outcome="AUTH_REQUIRED"),
    ],
    expected_claims=frozenset({claims.REQUEST_VERIFICATION}),
    forbidden_claims=frozenset(claims.SUCCESS_CLAIMS | {claims.PENDING_REVIEW}),
    db_checks=[
        DbCheck(
            "the public request creates no appointment",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot",
            0,
            {"slot": Ref("slot")},
        )
    ],
)


def _setup_hr008(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Reschedule Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Reschedule Test {ctx.suffix}", "Family Medicine")],
        slot_starts=[start, start + timedelta(hours=1), start + timedelta(minutes=90)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H8")
    old_slot, new_slot = clinic.slot_ids[0][:2]
    appointment, old_version = book_slot_for(ctx.engine, code, old_slot)
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _local_date(start, DENVER),
            "appointment": appointment,
            "old_slot": old_slot,
            "new_slot": new_slot,
            "old_slot_version": old_version,
            "time_old": display_time(start, DENVER),
            "time_new": display_time(start + timedelta(hours=1), DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot for group in clinic.slot_ids for slot in group],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


APPOINTMENT_AT_CLINIC = {"clinic.name": Said("clinic_name")}

HR_008 = Scenario(
    id="HR-008",
    title="Reschedule owned appointment",
    delivery=DELIVERY,
    request="Move my {day} appointment at {clinic_name} to the next available time that day.",
    request_terms=("day", "clinic_name"),
    actor="patient",
    setup=_setup_hr008,
    model_actions=[
        ToolAction(
            MINE,
            {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")},
        ),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": Pick(
                    1, "data.appointments", APPOINTMENT_AT_CLINIC, "clinic.clinic_id"
                ),
                "start_date": Said("day"),
                "end_date": Said("day"),
            },
        ),
        ToolAction(
            PREPARE,
            {
                "action": "RESCHEDULE",
                "appointment_id": Pick(
                    1, "data.appointments", APPOINTMENT_AT_CLINIC, "appointment_id"
                ),
                "new_slot_id": ResultRef(2, "data.slots.0.slot_id"),
            },
        ),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset({claims.RESCHEDULED}), "Your appointment is rescheduled."),
    ],
    timeline=[
        ModelTurn(),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(
                action="RESCHEDULE",
                target=Ref("appointment"),
                expected={
                    "appointment_id": Ref("appointment"),
                    "old_slot_id": Ref("old_slot"),
                    "new_slot_id": Ref("new_slot"),
                },
                summary={
                    "operation": "RESCHEDULE",
                    "current.clinic": Ref("clinic_name"),
                    "current.time": Ref("time_old"),
                    "proposed.clinic": Ref("clinic_name"),
                    "proposed.date": Ref("day"),
                    "proposed.time": Ref("time_new"),
                },
            ),
        ),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(
            MINE,
            {"status": "CONFIRMED", "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(
            PREPARE,
            {
                "action": "RESCHEDULE",
                "appointment_id": Ref("appointment"),
                "new_slot_id": Ref("new_slot"),
            },
        ),
    ],
    expected_claims=frozenset({claims.RESCHEDULED}),
    db_checks=[
        DbCheck(
            "the appointment moved to the new slot",
            "SELECT count(*) FROM appointments"
            " WHERE appointment_id = :appointment AND slot_id = :slot AND status = 'CONFIRMED'",
            1,
            {"appointment": Ref("appointment"), "slot": Ref("new_slot")},
        ),
        DbCheck(
            "the old slot is available",
            "SELECT status FROM slots WHERE slot_id = :slot",
            "AVAILABLE",
            {"slot": Ref("old_slot")},
        ),
    ],
)


def _setup_hr009(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER, hour=10)
    first = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Cancel Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Cancel Target {ctx.suffix}", "Family Medicine")],
        slot_starts=[start],
    )
    second = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx, "Holdout Other Clinic"),
        timezone=DENVER,
        doctors=[(f"Dr. Cancel Other {ctx.suffix}", "Internal Medicine")],
        slot_starts=[start + timedelta(hours=1)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H9")
    target_slot = first.slot_ids[0][0]
    other_slot = second.slot_ids[0][0]
    target, _ = book_slot_for(ctx.engine, code, target_slot)
    other, _ = book_slot_for(ctx.engine, code, other_slot)
    return Fixtures(
        values={
            "clinic_name": first.name,
            "day": _local_date(start, DENVER),
            "appointment": target,
            "other_appointment": other,
            "target_slot": target_slot,
            "target_time": display_time(start, DENVER),
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[target_slot, other_slot],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


HR_009 = Scenario(
    id="HR-009",
    title="Cancel owned appointment",
    delivery=DELIVERY,
    request="Cancel my appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_hr009,
    model_actions=[
        ToolAction(
            MINE,
            {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")},
        ),
        ToolAction(
            PREPARE,
            {
                "action": "CANCEL",
                "appointment_id": Pick(
                    1, "data.appointments", APPOINTMENT_AT_CLINIC, "appointment_id"
                ),
            },
        ),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset({claims.CANCELLED}), "Your appointment is cancelled."),
    ],
    timeline=[
        ModelTurn(),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(
                action="CANCEL",
                target=Ref("appointment"),
                summary={
                    "operation": "CANCEL",
                    "current.clinic": Ref("clinic_name"),
                    "current.date": Ref("day"),
                    "current.time": Ref("target_time"),
                },
            ),
        ),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(
            MINE,
            {"status": "CONFIRMED", "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(PREPARE, {"action": "CANCEL", "appointment_id": Ref("appointment")}),
    ],
    expected_claims=frozenset({claims.CANCELLED}),
    db_checks=[
        DbCheck(
            "the intended appointment is cancelled",
            "SELECT status FROM appointments WHERE appointment_id = :appointment",
            "CANCELLED",
            {"appointment": Ref("appointment")},
        ),
        DbCheck(
            "the other appointment remains confirmed",
            "SELECT status FROM appointments WHERE appointment_id = :appointment",
            "CONFIRMED",
            {"appointment": Ref("other_appointment")},
        ),
    ],
)


def _setup_hr010(ctx: SetupContext) -> Fixtures:
    fixtures = _setup_hr001(ctx)
    other = create_patient(ctx.engine, ctx.suffix, ctx.pin, "H10B")
    fixtures.values["other_patient_code"] = other
    fixtures.protected.extend(patient_identity(ctx.engine, other))
    return fixtures


def _other_patient_books_target(world: WorldContext) -> None:
    book_slot_for(
        world.engine,
        world.fixtures.values["other_patient_code"],
        world.fixtures.values["target_slot"],
    )


HR_010 = Scenario(
    id="HR-010",
    title="Selected slot becomes unavailable",
    delivery=DELIVERY,
    request="Book the earliest appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_hr010,
    model_actions=[
        ToolAction(SEARCH, {"clinic_name": Said("clinic_name")}),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": ResultRef(1, "data.clinics.0.clinic_id"),
                "start_date": Said("day"),
                "end_date": Said("day"),
            },
        ),
        ToolAction(PREPARE, {"action": "BOOK", "slot_id": ResultRef(2, "data.slots.0.slot_id")}),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset(), "That appointment is no longer available."),
    ],
    timeline=[
        ModelTurn(),
        World("another patient books the selected slot", _other_patient_books_target),
        UiConfirm(tool=PREPARE, intent=_book_intent(), expect="SLOT_UNAVAILABLE"),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(SEARCH, {"clinic_name": Ref("clinic_name")}),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(PREPARE, {"action": "BOOK", "slot_id": Ref("target_slot")}),
    ],
    expected_claims=frozenset(),
    forbidden_claims=frozenset({claims.BOOKED}),
    db_checks=[
        DbCheck(
            "the requesting patient has no appointment on the selected slot",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot AND patient_id = :patient",
            0,
            {"slot": Ref("target_slot"), "patient": Ref("patient")},
        ),
        DbCheck(
            "the other patient owns the only appointment on the slot",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot AND status = 'CONFIRMED'",
            1,
            {"slot": Ref("target_slot")},
        ),
    ],
)


HOLDOUT_SCENARIOS = (
    HR_001,
    HR_002,
    HR_003,
    HR_004,
    HR_005,
    HR_006,
    HR_007,
    HR_008,
    HR_009,
    HR_010,
)
