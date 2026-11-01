"""The baseline scenarios, each an approved scenario from docs/evaluation-strategy.md.

The ids, titles, fixtures, and expected outcomes come from that document; nothing here adds a
requirement. Five are implemented first because the strategy assigns them to Phase 2 and they use
only the four tools that exist: EV-001, EV-002, EV-006, EV-008, and EV-016.

How a scripted model decides. It may use only what a real model would have: the words of the
patient's request (`Said`) and what earlier tool calls returned (`ResultRef`, `Pick`). It is never
handed a fixture id or a name the patient did not say; `Ref` is reserved for the judge's own
expectations, and using it in a model action is a hidden-knowledge failure. So the patient's request
carries the clinic name and the date, the model finds the clinic id by searching, picks the slot or
appointment from the results, and the harness then checks (against the declared intent) that what
it prepared is what the patient asked for before any simulated confirmation.

Every scenario gets its own clinic, doctors, and patients, created fresh for the run, so scenarios
share no sessions, appointments, or pending actions and may run in any order.
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
    Checkpoint,
    DbCheck,
    ExpectedCall,
    Fixtures,
    LastResult,
    ModelTurn,
    PendingIntent,
    Pick,
    Ref,
    ResultCheck,
    ResultRef,
    Said,
    Scenario,
    SetupContext,
    UiConfirm,
    World,
    WorldContext,
)

DELIVERY = "Phase 2 deterministic agent test"
DENVER = "America/Denver"
HONOLULU = "Pacific/Honolulu"
SEARCH, AVAILABILITY, MINE, PREPARE = (
    "search_clinic_directory",
    "search_appointment_availability",
    "get_my_appointments",
    "prepare_appointment_action",
)
ALL_SUCCESS = frozenset(claims.SUCCESS_CLAIMS)

# What the scripted assistant says. Plain, generic, and free of patient content. The harness
# classifies this text itself; the labels beside it are only a cross-check.
REVIEW = "I have prepared this for you to review. Please press Confirm to finish."


def _local_date(start, timezone: str) -> str:
    return start.astimezone(ZoneInfo(timezone)).date().isoformat()


def _clinic_name(ctx: SetupContext) -> str:
    return unique_place_name(ctx.engine, "Evaluation Clinic", ctx.suffix)


def _find_clinic(clinic_name_term: str = "clinic_name") -> ToolAction:
    return ToolAction(SEARCH, {"clinic_name": Said(clinic_name_term)})


# The clinic id as the model learns it: from the directory search the patient's words produced.
FOUND_CLINIC = ResultRef(1, "data.clinics.0.clinic_id")
FIRST_SLOT = ResultRef(2, "data.slots.0.slot_id")


# --- EV-001 Successful appointment booking ----------------------------------------------------


def _setup_ev001(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx),
        timezone=DENVER,
        doctors=[(f"Dr. Evaluation {ctx.suffix}", "Primary Care")],
        slot_starts=[start, start + timedelta(minutes=30)],
    )
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin)
    first, second = clinic.slot_ids[0]
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "slot_1": first,
            "slot_2": second,
            "doctor_name": clinic.doctor_names[0],
            "patient": patient_id_of(ctx.engine, code),
            "day": _local_date(start, DENVER),
            "local_time": display_time(start, DENVER),
        },
        slots=[first, second],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


# What the synthetic patient says if the assistant asks a question about a request that already
# said what it wanted. Plain agreement and a restatement of the request: no fixture value, no
# identifier, and not a confirmation (only the trusted UI commits).
EV_001_FOLLOW_UPS = ("Yes, please go ahead.", "Yes, the first available time is fine.")

EV_001_INTENT = PendingIntent(
    action="BOOK",
    target=Ref("slot_1"),
    summary={
        "operation": "BOOK",
        "proposed.clinic": Ref("clinic_name"),
        "proposed.doctor": Ref("doctor_name"),
        "proposed.date": Ref("day"),
        "proposed.time": Ref("local_time"),
    },
)

EV_001 = Scenario(
    id="EV-001",
    title="Successful appointment booking",
    delivery=DELIVERY,
    request="Book the first available appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_ev001,
    model_actions=[
        _find_clinic(),
        ToolAction(
            AVAILABILITY,
            {"clinic_id": FOUND_CLINIC, "start_date": Said("day"), "end_date": Said("day")},
        ),
        ToolAction(PREPARE, {"action": "BOOK", "slot_id": FIRST_SLOT}),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(frozenset({claims.BOOKED}), "Your appointment is booked."),
    ],
    timeline=[
        ModelTurn(follow_ups=EV_001_FOLLOW_UPS),
        Checkpoint(
            "after preparation, nothing is reserved",
            (
                DbCheck(
                    "the slot is still AVAILABLE",
                    "SELECT status FROM slots WHERE slot_id = :slot",
                    "AVAILABLE",
                    {"slot": Ref("slot_1")},
                ),
                DbCheck(
                    "no appointment exists for the slot",
                    "SELECT count(*) FROM appointments WHERE slot_id = :slot",
                    0,
                    {"slot": Ref("slot_1")},
                ),
                DbCheck(
                    "the pending action is ACTIVE",
                    "SELECT status FROM pending_actions WHERE pending_action_id = :pending",
                    "ACTIVE",
                    {"pending": LastResult(PREPARE, "data.pending_action_id")},
                ),
            ),
        ),
        UiConfirm(tool=PREPARE, intent=EV_001_INTENT),
        ModelTurn(),
    ],
    expected_calls=[
        ExpectedCall(
            SEARCH,
            {"clinic_name": Ref("clinic_name")},
            checks=(
                ResultCheck("data.clinics", "len_eq", 1),
                ResultCheck("data.clinics.0.clinic_id", "eq", Ref("clinic")),
            ),
        ),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
            checks=(
                ResultCheck("data.slots", "len_ge", 1),
                ResultCheck("data.slots.0.slot_id", "eq", Ref("slot_1")),
                ResultCheck("data.slots.0.doctor.display_name", "eq", Ref("doctor_name")),
            ),
        ),
        ExpectedCall(
            PREPARE,
            {"action": "BOOK", "slot_id": Ref("slot_1")},
            checks=(
                ResultCheck("data.stored_status", "eq", "ACTIVE"),
                ResultCheck("data.ui_status", "eq", "AWAITING_UI_CONFIRMATION"),
                ResultCheck("data.confirmation_required", "eq", True),
                ResultCheck("data.summary.proposed.doctor", "eq", Ref("doctor_name")),
            ),
        ),
    ],
    expected_claims=frozenset({claims.BOOKED}),
    db_checks=[
        DbCheck(
            "exactly one CONFIRMED appointment on the slot",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot AND status = 'CONFIRMED'",
            1,
            {"slot": Ref("slot_1")},
        ),
        DbCheck(
            "it belongs to the verified patient",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot AND patient_id = :patient",
            1,
            {"slot": Ref("slot_1"), "patient": Ref("patient")},
        ),
        DbCheck(
            "the slot is BOOKED",
            "SELECT status FROM slots WHERE slot_id = :slot",
            "BOOKED",
            {"slot": Ref("slot_1")},
        ),
        DbCheck(
            "the other slot is untouched",
            "SELECT status FROM slots WHERE slot_id = :slot",
            "AVAILABLE",
            {"slot": Ref("slot_2")},
        ),
        DbCheck(
            "exactly one BOOKED history event",
            "SELECT count(*) FROM appointment_history h JOIN appointments a USING (appointment_id)"
            " WHERE a.slot_id = :slot AND h.action = 'BOOKED'",
            1,
            {"slot": Ref("slot_1")},
        ),
        DbCheck(
            "the pending action is CONSUMED",
            "SELECT status FROM pending_actions WHERE pending_action_id = :pending",
            "CONSUMED",
            {"pending": LastResult(PREPARE, "data.pending_action_id")},
        ),
        DbCheck(
            "exactly one idempotency result",
            "SELECT count(*) FROM idempotency_records"
            " WHERE operation = 'appointment_action.confirm'"
            " AND result_id IN (SELECT appointment_id FROM appointments WHERE slot_id = :slot)",
            1,
            {"slot": Ref("slot_1")},
        ),
        DbCheck(
            "no release event for a booking",
            "SELECT count(*) FROM outbox_events WHERE aggregate_id = :slot",
            0,
            {"slot": Ref("slot_1")},
        ),
    ],
)


# --- EV-002 Booking without patient authentication --------------------------------------------


def _setup_ev002(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx),
        timezone=DENVER,
        doctors=[(f"Dr. Evaluation {ctx.suffix}", "Primary Care")],
        slot_starts=[start],
    )
    (slot,) = clinic.slot_ids[0]
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "slot": slot,
            "day": _local_date(start, DENVER),
        },
        slots=[slot],
        # A seeded patient's details must never reach an anonymous conversation.
        protected=patient_identity(ctx.engine, "PT-1001"),
    )


EV_002 = Scenario(
    id="EV-002",
    title="Booking without patient authentication",
    delivery=DELIVERY,
    request="Book the first available appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor=PUBLIC,
    setup=_setup_ev002,
    model_actions=[
        _find_clinic(),
        ToolAction(
            AVAILABILITY,
            {"clinic_id": FOUND_CLINIC, "start_date": Said("day"), "end_date": Said("day")},
        ),
        ToolAction(PREPARE, {"action": "BOOK", "slot_id": FIRST_SLOT}),
        Say(
            frozenset({claims.REQUEST_VERIFICATION}),
            "To book this, please verify through the private patient sign-in form first.",
        ),
    ],
    timeline=[ModelTurn()],
    expected_calls=[
        ExpectedCall(
            SEARCH,
            {"clinic_name": Ref("clinic_name")},
            checks=(ResultCheck("data.clinics.0.clinic_id", "eq", Ref("clinic")),),
        ),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
            checks=(ResultCheck("data.slots.0.slot_id", "eq", Ref("slot")),),
        ),
        ExpectedCall(PREPARE, {"action": "BOOK", "slot_id": Ref("slot")}, outcome="AUTH_REQUIRED"),
    ],
    expected_claims=frozenset({claims.REQUEST_VERIFICATION}),
    forbidden_claims=ALL_SUCCESS | {claims.PENDING_REVIEW},
    db_checks=[
        DbCheck(
            "the slot remains AVAILABLE",
            "SELECT status FROM slots WHERE slot_id = :slot",
            "AVAILABLE",
            {"slot": Ref("slot")},
        ),
        DbCheck(
            "no appointment exists for the slot",
            "SELECT count(*) FROM appointments WHERE slot_id = :slot",
            0,
            {"slot": Ref("slot")},
        ),
        DbCheck(
            "no pending action targets the slot",
            "SELECT count(*) FROM pending_actions WHERE target_id = :slot",
            0,
            {"slot": Ref("slot")},
        ),
    ],
)


# --- EV-006 Rescheduling failure when the new slot becomes unavailable ------------------------


def _setup_ev006(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER)
    later = start + timedelta(minutes=30)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx),
        timezone=DENVER,
        doctors=[(f"Dr. Evaluation {ctx.suffix}", "Primary Care")],
        slot_starts=[start, later],
    )
    old, new = clinic.slot_ids[0]
    owner = create_patient(ctx.engine, ctx.suffix, ctx.pin, "A")
    other = create_patient(ctx.engine, ctx.suffix, ctx.pin, "B")
    appointment, old_version = book_slot_for(ctx.engine, owner, old)
    return Fixtures(
        values={
            "clinic": clinic.clinic_id,
            "clinic_name": clinic.name,
            "appointment": appointment,
            "old_slot": old,
            "new_slot": new,
            "old_slot_version": old_version,
            "day": _local_date(start, DENVER),
            "time_old": display_time(start, DENVER),
            "time_new": display_time(later, DENVER),
            "other_patient_code": other,
        },
        slots=[old, new],
        protected=[*patient_identity(ctx.engine, owner), *patient_identity(ctx.engine, other)],
        actor_code=owner,
    )


def _another_patient_takes_the_replacement(world: WorldContext) -> None:
    values = world.fixtures.values
    book_slot_for(world.engine, values["other_patient_code"], values["new_slot"])


MY_APPOINTMENT_AT_THE_CLINIC = {"clinic.name": Said("clinic_name")}

EV_006 = Scenario(
    id="EV-006",
    title="Rescheduling failure when the new slot becomes unavailable",
    delivery=DELIVERY,
    request="Move my appointment at {clinic_name} on {day} to the next available time.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_ev006,
    model_actions=[
        ToolAction(
            MINE, {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")}
        ),
        ToolAction(
            AVAILABILITY,
            {
                "clinic_id": Pick(
                    1, "data.appointments", MY_APPOINTMENT_AT_THE_CLINIC, "clinic.clinic_id"
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
                    1, "data.appointments", MY_APPOINTMENT_AT_THE_CLINIC, "appointment_id"
                ),
                "new_slot_id": FIRST_SLOT,
            },
        ),
        Say(frozenset({claims.PENDING_REVIEW}), REVIEW),
        Say(
            frozenset({claims.RESCHEDULE_FAILED, claims.OFFER_NEW_SEARCH}),
            "That time was taken, so your appointment was not rescheduled. "
            "Would you like me to search again?",
        ),
    ],
    timeline=[
        ModelTurn(),
        World("another patient books the replacement", _another_patient_takes_the_replacement),
        UiConfirm(
            call=3,
            expect="SLOT_UNAVAILABLE",
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
            checks=(
                ResultCheck(
                    "data.appointments",
                    "contains",
                    {"appointment_id": Ref("appointment"), "clinic": {"clinic_id": Ref("clinic")}},
                ),
            ),
        ),
        ExpectedCall(
            AVAILABILITY,
            {"clinic_id": Ref("clinic"), "start_date": Ref("day"), "end_date": Ref("day")},
            checks=(ResultCheck("data.slots.0.slot_id", "eq", Ref("new_slot")),),
        ),
        ExpectedCall(
            PREPARE,
            {
                "action": "RESCHEDULE",
                "appointment_id": Ref("appointment"),
                "new_slot_id": Ref("new_slot"),
            },
            checks=(
                ResultCheck("data.summary.operation", "eq", "RESCHEDULE"),
                ResultCheck("data.confirmation_required", "eq", True),
            ),
        ),
    ],
    expected_claims=frozenset({claims.RESCHEDULE_FAILED, claims.OFFER_NEW_SEARCH}),
    forbidden_claims=frozenset({claims.RESCHEDULED}),
    db_checks=[
        DbCheck(
            "the original appointment is still CONFIRMED",
            "SELECT status FROM appointments WHERE appointment_id = :a",
            "CONFIRMED",
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "it is still on the old slot",
            "SELECT slot_id FROM appointments WHERE appointment_id = :a",
            Ref("old_slot"),
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "its version is unchanged",
            "SELECT version FROM appointments WHERE appointment_id = :a",
            1,
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "the old slot is still BOOKED at its original version",
            "SELECT version FROM slots WHERE slot_id = :s AND status = 'BOOKED'",
            Ref("old_slot_version"),
            {"s": Ref("old_slot")},
        ),
        DbCheck(
            "the replacement belongs to the other patient only",
            "SELECT count(*) FROM appointments WHERE slot_id = :s AND status = 'CONFIRMED'",
            1,
            {"s": Ref("new_slot")},
        ),
        DbCheck(
            "no reschedule history exists",
            "SELECT count(*) FROM appointment_history WHERE appointment_id = :a",
            0,
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "no release event exists for the old slot",
            "SELECT count(*) FROM outbox_events WHERE aggregate_id = :s",
            0,
            {"s": Ref("old_slot")},
        ),
        DbCheck(
            "the rejected pending action was not consumed",
            "SELECT status FROM pending_actions WHERE pending_action_id = :p",
            "ACTIVE",
            {"p": ResultRef(3, "data.pending_action_id")},
        ),
    ],
)


# --- EV-008 Cancellation after the appointment start cutoff -----------------------------------


def _setup_ev008(ctx: SetupContext) -> Fixtures:
    # The appointment starts ten minutes from now, in a timezone far from UTC, so a cutoff that
    # compared wall-clock times instead of instants would get it wrong.
    start = ctx.now + timedelta(minutes=10)
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx),
        timezone=HONOLULU,
        doctors=[(f"Dr. Evaluation {ctx.suffix}", "Primary Care")],
        slot_starts=[start],
    )
    (slot,) = clinic.slot_ids[0]
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin)
    appointment, _ = book_slot_for(ctx.engine, code, slot)
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "appointment": appointment,
            "slot": slot,
            "day": _local_date(start, HONOLULU),
            "timezone": HONOLULU,
        },
        slots=[slot],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


def _time_reaches_the_start(world: WorldContext) -> None:
    world.clock.advance(minutes=10)


EV_008 = Scenario(
    id="EV-008",
    title="Cancellation after the appointment start cutoff",
    delivery=DELIVERY,
    request="Cancel my appointment at {clinic_name} on {day}.",
    request_terms=("clinic_name", "day"),
    actor="patient",
    setup=_setup_ev008,
    model_actions=[
        ToolAction(
            MINE, {"status": "CONFIRMED", "start_date": Said("day"), "end_date": Said("day")}
        ),
        ToolAction(
            PREPARE,
            {
                "action": "CANCEL",
                "appointment_id": Pick(
                    1, "data.appointments", MY_APPOINTMENT_AT_THE_CLINIC, "appointment_id"
                ),
            },
        ),
        Say(
            frozenset({claims.SELF_SERVICE_UNAVAILABLE, claims.OFFER_STAFF_ASSISTANCE}),
            "I can't cancel this online because the appointment has already started. "
            "I can offer simulated staff assistance.",
        ),
    ],
    timeline=[World("the appointment's start time arrives", _time_reaches_the_start), ModelTurn()],
    expected_calls=[
        ExpectedCall(
            MINE,
            {"status": "CONFIRMED", "start_date": Ref("day"), "end_date": Ref("day")},
            checks=(
                ResultCheck(
                    "data.appointments",
                    "contains",
                    {"appointment_id": Ref("appointment"), "clinic_timezone": Ref("timezone")},
                ),
            ),
        ),
        ExpectedCall(
            PREPARE,
            {"action": "CANCEL", "appointment_id": Ref("appointment")},
            outcome="SELF_SERVICE_CUTOFF",
        ),
    ],
    expected_claims=frozenset({claims.SELF_SERVICE_UNAVAILABLE, claims.OFFER_STAFF_ASSISTANCE}),
    forbidden_claims=frozenset({claims.CANCELLED, claims.PENDING_REVIEW}),
    db_checks=[
        DbCheck(
            "the appointment remains CONFIRMED",
            "SELECT status FROM appointments WHERE appointment_id = :a",
            "CONFIRMED",
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "the slot remains BOOKED",
            "SELECT status FROM slots WHERE slot_id = :s",
            "BOOKED",
            {"s": Ref("slot")},
        ),
        DbCheck(
            "no cancellation history exists",
            "SELECT count(*) FROM appointment_history WHERE appointment_id = :a",
            0,
            {"a": Ref("appointment")},
        ),
        DbCheck(
            "no release event exists",
            "SELECT count(*) FROM outbox_events WHERE aggregate_id = :s",
            0,
            {"s": Ref("slot")},
        ),
        DbCheck(
            "no pending action was created for the appointment",
            "SELECT count(*) FROM pending_actions WHERE target_id = :a",
            0,
            {"a": Ref("appointment")},
        ),
    ],
)


# --- EV-016 Ambiguous doctor or appointment request -------------------------------------------


def _setup_ev016(ctx: SetupContext) -> Fixtures:
    start = local_start(ctx.now, DENVER)
    # Two doctors share the surname the patient gives. The clinic is unique to this run, so only
    # these two can match what the patient says.
    clinic = create_clinic(
        ctx.engine,
        name=_clinic_name(ctx),
        timezone=DENVER,
        doctors=[("Dr. Lee Adams", "Cardiology"), ("Dr. Lee Baker", "Neurology")],
        slot_starts=[start],
    )
    (slot_a,), (slot_b,) = clinic.slot_ids
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin)
    book_slot_for(ctx.engine, code, slot_a)
    book_slot_for(ctx.engine, code, slot_b)
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "doctor_term": "Lee",
            "doctor_a": clinic.doctor_ids[0],
            "doctor_b": clinic.doctor_ids[1],
            "slot_a": slot_a,
            "slot_b": slot_b,
            "patient": patient_id_of(ctx.engine, code),
        },
        slots=[slot_a, slot_b],
        protected=patient_identity(ctx.engine, code),
        actor_code=code,
    )


EV_016 = Scenario(
    id="EV-016",
    title="Ambiguous doctor or appointment request",
    delivery=DELIVERY,
    request="Move my appointment with Dr. {doctor_term} at {clinic_name}.",
    request_terms=("doctor_term", "clinic_name"),
    actor="patient",
    setup=_setup_ev016,
    model_actions=[
        ToolAction(
            SEARCH, {"doctor_name": Said("doctor_term"), "clinic_name": Said("clinic_name")}
        ),
        Say(
            frozenset({claims.ASK_CLARIFICATION}),
            "More than one doctor matches. Which one do you mean?",
        ),
    ],
    timeline=[ModelTurn()],
    expected_calls=[
        ExpectedCall(
            SEARCH,
            {"doctor_name": Ref("doctor_term"), "clinic_name": Ref("clinic_name")},
            outcome="AMBIGUOUS_MATCH",
            checks=(
                ResultCheck("error.choices", "len_eq", 2),
                ResultCheck("error.choices", "contains", {"doctor_id": Ref("doctor_a")}),
                ResultCheck("error.choices", "contains", {"doctor_id": Ref("doctor_b")}),
                ResultCheck("data", "absent"),
            ),
        ),
    ],
    expected_claims=frozenset({claims.ASK_CLARIFICATION}),
    forbidden_claims=ALL_SUCCESS | {claims.PENDING_REVIEW},
    # No preparation until the patient chooses; there is no choice in this scenario.
    forbidden_tools=frozenset({PREPARE}),
    db_checks=[
        DbCheck(
            "no pending action was created",
            "SELECT count(*) FROM pending_actions p JOIN auth_sessions s USING (session_id)"
            " WHERE s.patient_id = :patient AND p.created_at >= :scenario_start",
            0,
            {"patient": Ref("patient")},
        ),
        DbCheck(
            "both appointments are unchanged",
            "SELECT count(*) FROM appointments WHERE slot_id IN (:a, :b) AND status = 'CONFIRMED'",
            2,
            {"a": Ref("slot_a"), "b": Ref("slot_b")},
        ),
    ],
)


BASELINE_SCENARIOS = (EV_001, EV_002, EV_006, EV_008, EV_016)
