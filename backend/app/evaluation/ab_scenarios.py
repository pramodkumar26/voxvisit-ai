"""Fresh development scenarios used only for the free-form versus hybrid comparison."""

from datetime import timedelta
from zoneinfo import ZoneInfo

from app.evaluation import model as claims
from app.evaluation.fixtures import (
    book_slot_for,
    create_clinic,
    create_patient,
    local_start,
    patient_id_of,
    patient_identity,
    unique_place_name,
)
from app.evaluation.model import Say, ToolAction
from app.evaluation.scenario import (
    DbCheck,
    ExpectedCall,
    Fixtures,
    ModelTurn,
    PendingIntent,
    Ref,
    ResultRef,
    Said,
    Scenario,
    SetupContext,
    UiConfirm,
    World,
    WorldContext,
)

SEARCH = "search_clinic_directory"
AVAILABILITY = "search_appointment_availability"
MINE = "get_my_appointments"
PREPARE = "prepare_appointment_action"
ZONE = "America/Denver"
DELIVERY = "Phase 2 architecture comparison v1"
REVIEW = "I prepared the requested change. Review the details and confirm in the app."


def _name(ctx: SetupContext, prefix: str) -> str:
    return unique_place_name(ctx.engine, prefix, ctx.suffix)


def _day(start) -> str:
    return start.astimezone(ZoneInfo(ZONE)).date().isoformat()


BOOK_FILTERS = {
    SEARCH: frozenset({"clinic_name"}),
    AVAILABILITY: frozenset({"clinic_id", "start_date", "end_date"}),
}


def _setup_book(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, ZONE, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_name(ctx, "Comparison Booking Clinic"),
        timezone=ZONE,
        doctors=[(f"Dr. Devon Park {ctx.suffix}", "Primary Care")],
        slot_starts=[start, start + timedelta(minutes=45)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "AB1")
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _day(start),
            "slot": clinic.slot_ids[0][0],
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=list(clinic.slot_ids[0]),
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


AB_BOOK = Scenario(
    id="AB-BOOK",
    title="Clear earliest booking",
    delivery=DELIVERY,
    request="Schedule the soonest visit at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_book,
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
        Say(frozenset({claims.BOOKED}), "Your appointment is booked."),
    ],
    timeline=[
        ModelTurn(follow_ups=("Please use the soonest available time.",)),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(action="BOOK", target=Ref("slot")),
        ),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(SEARCH, {"clinic_name": Ref("clinic_name")}),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
        ),
        ExpectedCall(PREPARE, {"action": "BOOK", "slot_id": Ref("slot")}),
    ],
    expected_claims=frozenset({claims.BOOKED}),
    db_checks=[
        DbCheck(
            "the requested slot is booked for the patient",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot"
            " AND patient_id = :patient AND status = 'CONFIRMED'",
            1,
            {"slot": Ref("slot"), "patient": Ref("patient")},
        )
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_ambiguity(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, ZONE, hour=10)
    clinic = create_clinic(
        ctx.engine,
        name=_name(ctx, "Comparison Ambiguity Clinic"),
        timezone=ZONE,
        doctors=[
            (f"Dr. Lee North {ctx.suffix}", "Primary Care"),
            (f"Dr. Lee South {ctx.suffix}", "Cardiology"),
        ],
        doctor_slot_starts=[[start], [start + timedelta(hours=1)]],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "AB2")
    for slots in clinic.slot_ids:
        book_slot_for(ctx.engine, code, slots[0])
    return Fixtures(
        values={"clinic_name": clinic.name, "doctor_term": "Lee"},
        slots=[slots[0] for slots in clinic.slot_ids],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


AB_AMBIGUITY = Scenario(
    id="AB-AMBIGUITY",
    title="Ambiguous doctor requires clarification",
    delivery=DELIVERY,
    request="I need to move my appointment with Dr. {doctor_term} at {clinic_name}.",
    request_terms=("doctor_term", "clinic_name"),
    actor="patient",
    setup=_setup_ambiguity,
    model_actions=[
        ToolAction(
            SEARCH,
            {"doctor_name": Said("doctor_term"), "clinic_name": Said("clinic_name")},
        ),
        Say(frozenset(), "I found multiple matching doctors. Which one do you mean?"),
    ],
    timeline=[ModelTurn(question_is_unnecessary=False)],
    expected_calls=[
        ExpectedCall(
            SEARCH,
            {"doctor_name": Ref("doctor_term"), "clinic_name": Ref("clinic_name")},
            outcome="AMBIGUOUS_MATCH",
        )
    ],
    expected_claims=frozenset(),
    forbidden_tools=frozenset({PREPARE}),
    db_checks=[
        DbCheck(
            "both ambiguous appointments remain confirmed",
            "SELECT count(*) FROM appointments WHERE status = 'CONFIRMED' AND slot_id IN (:a, :b)",
            2,
            {"a": Ref("slot_a"), "b": Ref("slot_b")},
        )
    ],
    authorized_filters={SEARCH: frozenset({"clinic_name", "doctor_name"})},
)


def _setup_ambiguity_with_slots(ctx: SetupContext) -> Fixtures:
    fixtures = _setup_ambiguity(ctx)
    fixtures.values["slot_a"], fixtures.values["slot_b"] = fixtures.slots
    return fixtures


AB_AMBIGUITY = Scenario(**{**vars(AB_AMBIGUITY), "setup": _setup_ambiguity_with_slots})


def _setup_cancel(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, ZONE, hour=11)
    clinic = create_clinic(
        ctx.engine,
        name=_name(ctx, "Comparison Cancel Clinic"),
        timezone=ZONE,
        doctors=[(f"Dr. Robin Vale {ctx.suffix}", "Internal Medicine")],
        slot_starts=[start],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "AB3")
    slot = clinic.slot_ids[0][0]
    appointment, _ = book_slot_for(ctx.engine, code, slot)
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "day": _day(start),
            "appointment": appointment,
            "slot": slot,
        },
        slots=[slot],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


AB_CANCEL = Scenario(
    id="AB-CANCEL",
    title="Cancel a uniquely identified appointment",
    delivery=DELIVERY,
    request="Cancel my visit at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_cancel,
    model_actions=[
        ToolAction(
            MINE, {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")}
        ),
        ToolAction(
            PREPARE,
            {
                "action": "CANCEL",
                "appointment_id": ResultRef(1, "data.appointments.0.appointment_id"),
            },
        ),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset({claims.CANCELLED}), "Your appointment is cancelled."),
    ],
    timeline=[
        ModelTurn(follow_ups=("Yes, cancel that visit.",)),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="CANCEL", target=Ref("appointment"))),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(
            MINE, {"status": "CONFIRMED", "start_date": Ref("day"), "end_date": Ref("day")}
        ),
        ExpectedCall(PREPARE, {"action": "CANCEL", "appointment_id": Ref("appointment")}),
    ],
    expected_claims=frozenset({claims.CANCELLED}),
    db_checks=[
        DbCheck(
            "the selected appointment is cancelled",
            "SELECT status FROM appointments WHERE appointment_id = :appointment",
            "CANCELLED",
            {"appointment": Ref("appointment")},
        )
    ],
    authorized_filters={
        SEARCH: frozenset({"clinic_name"}),
        MINE: frozenset({"status", "start_date", "end_date"}),
    },
)


def _setup_reschedule(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, ZONE, hour=9)
    clinic = create_clinic(
        ctx.engine,
        name=_name(ctx, "Comparison Move Clinic"),
        timezone=ZONE,
        doctors=[(f"Dr. Morgan Lake {ctx.suffix}", "Family Medicine")],
        slot_starts=[start, start + timedelta(hours=1)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, "AB4")
    old_slot, new_slot = clinic.slot_ids[0]
    appointment, _ = book_slot_for(ctx.engine, code, old_slot)
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "day": _day(start),
            "appointment": appointment,
            "old_slot": old_slot,
            "new_slot": new_slot,
        },
        slots=[old_slot, new_slot],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


AB_RESCHEDULE = Scenario(
    id="AB-RESCHEDULE",
    title="Reschedule to the next available time",
    delivery=DELIVERY,
    request="Move my {day} visit at {clinic_name} to the next available time that day.",
    request_terms=("day", "clinic_name"),
    actor="patient",
    setup=_setup_reschedule,
    model_actions=[
        ToolAction(
            MINE, {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")}
        ),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": ResultRef(1, "data.appointments.0.clinic.clinic_id"),
                "start_date": Said("day"),
                "end_date": Said("day"),
            },
        ),
        ToolAction(
            PREPARE,
            {
                "action": "RESCHEDULE",
                "appointment_id": ResultRef(1, "data.appointments.0.appointment_id"),
                "new_slot_id": ResultRef(2, "data.slots.0.slot_id"),
            },
        ),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset({claims.RESCHEDULED}), "Your appointment is rescheduled."),
    ],
    timeline=[
        ModelTurn(follow_ups=("Use the next available time that day.",)),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(
                action="RESCHEDULE",
                target=Ref("appointment"),
                expected={"new_slot_id": Ref("new_slot")},
            ),
        ),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(
            MINE, {"status": "CONFIRMED", "start_date": Ref("day"), "end_date": Ref("day")}
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
            "the appointment moved to the next slot",
            "SELECT count(*) FROM appointments WHERE appointment_id = :appointment"
            " AND slot_id = :slot AND status = 'CONFIRMED'",
            1,
            {"appointment": Ref("appointment"), "slot": Ref("new_slot")},
        )
    ],
    authorized_filters={
        SEARCH: frozenset({"clinic_name"}),
        MINE: frozenset({"status", "start_date", "end_date"}),
        AVAILABILITY: frozenset({"clinic_id", "start_date", "end_date"}),
    },
)


def _setup_stale(ctx: SetupContext) -> Fixtures:
    fixtures = _setup_book(ctx)
    other = create_patient(ctx.engine, ctx.suffix, ctx.pin, "AB5B")
    fixtures.values["other_code"] = other
    fixtures.protected.extend(patient_identity(ctx.engine, other))
    return fixtures


def _take_slot(world: WorldContext) -> None:
    book_slot_for(world.engine, world.fixtures.values["other_code"], world.fixtures.values["slot"])


AB_STALE = Scenario(
    **{
        **vars(AB_BOOK),
        "id": "AB-STALE",
        "title": "Selected slot becomes unavailable before confirmation",
        "setup": _setup_stale,
        "model_actions": [
            *AB_BOOK.model_actions[:-1],
            Say(frozenset(), "That appointment is no longer available."),
        ],
        "timeline": [
            ModelTurn(follow_ups=("Please use the soonest available time.",)),
            World("another patient books the selected slot", _take_slot),
            UiConfirm(
                tool=PREPARE,
                intent=PendingIntent(action="BOOK", target=Ref("slot")),
                expect="SLOT_UNAVAILABLE",
            ),
            ModelTurn(),
        ],
        "expected_claims": frozenset(),
        "forbidden_claims": frozenset({claims.BOOKED}),
        "db_checks": [
            DbCheck(
                "the requesting patient did not receive the stale slot",
                "SELECT count(*) FROM appointments WHERE slot_id = :slot AND patient_id = :patient",
                0,
                {"slot": Ref("slot"), "patient": Ref("patient")},
            )
        ],
    }
)


AB_SCENARIOS = (AB_BOOK, AB_AMBIGUITY, AB_CANCEL, AB_RESCHEDULE, AB_STALE)
