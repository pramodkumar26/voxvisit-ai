"""Multi-turn conversations for the text agent (Approach B), on synthetic data and the real tools.

Each scenario is a short, realistic conversation: the patient says something, the agent answers,
the patient answers back, changes their mind, or drops the request. What is judged is the end
state in the database, what the trusted UI was able to confirm, and the safety invariants, not the
exact words. These are development scenarios for the conversation layer; they are separate from
the frozen benchmark and are never used to select an architecture.

Patients speak only through `request`, `follow_ups`, and `PatientSays`, using month-and-day dates
without a year ("November 16"), the way people do. `{name}` marks a value a patient may state.
"""

from dataclasses import replace
from zoneinfo import ZoneInfo

from app.evaluation.fixtures import (
    book_slot_for,
    create_clinic,
    create_patient,
    local_start,
    patient_id_of,
    patient_identity,
    unique_place_name,
)
from app.evaluation.model import BOOKED, CANCELLED, RESCHEDULED
from app.evaluation.scenario import (
    PUBLIC,
    DbCheck,
    Fixtures,
    ModelTurn,
    PatientSays,
    PendingIntent,
    Ref,
    Scenario,
    SetupContext,
    UiConfirm,
)

SEARCH = "search_clinic_directory"
AVAILABILITY = "search_appointment_availability"
MINE = "get_my_appointments"
PREPARE = "prepare_appointment_action"
ZONE = "America/Denver"
DELIVERY = "Phase 2 conversation layer v1"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

BOOK_FILTERS = {
    SEARCH: frozenset({"clinic_name", "doctor_name"}),
    AVAILABILITY: frozenset(
        {"clinic_id", "start_date", "end_date", "doctor_id", "time_preference"}
    ),
}
CHANGE_FILTERS = {
    **BOOK_FILTERS,
    MINE: frozenset({"status", "start_date", "end_date"}),
}


def _start(ctx: SetupContext, days: int, hour: int):
    return local_start(ctx.now, ZONE, days_ahead=days, hour=hour)


def _words(start) -> str:
    """The way a patient says the date: "November 16"."""
    local = start.astimezone(ZoneInfo(ZONE))
    return f"{MONTHS[local.month - 1]} {local.day}"


def _ordinal(start) -> str:
    local = start.astimezone(ZoneInfo(ZONE))
    day = local.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"the {day}{suffix}"


def _clinic(ctx: SetupContext, prefix: str, doctors, doctor_slot_starts):
    return create_clinic(
        ctx.engine,
        name=unique_place_name(ctx.engine, prefix, ctx.suffix),
        timezone=ZONE,
        doctors=doctors,
        doctor_slot_starts=doctor_slot_starts,
    )


def _patient(ctx: SetupContext, role: str):
    code = create_patient(ctx.engine, ctx.suffix, ctx.pin, role)
    return code, patient_id_of(ctx.engine, code), patient_identity(ctx.engine, code)


def _booked_check(name: str, slot: str, expected: int = 1) -> DbCheck:
    return DbCheck(
        name,
        "SELECT count(*) FROM appointments WHERE slot_id = :slot AND patient_id = :patient"
        " AND status = 'CONFIRMED'",
        expected,
        {"slot": Ref(slot), "patient": Ref("patient")},
    )


def _status_check(name: str, appointment: str, status: str) -> DbCheck:
    return DbCheck(
        name,
        "SELECT status FROM appointments WHERE appointment_id = :appointment",
        status,
        {"appointment": Ref(appointment)},
    )


def _on_slot_check(name: str, appointment: str, slot: str) -> DbCheck:
    return DbCheck(
        name,
        "SELECT count(*) FROM appointments WHERE appointment_id = :appointment"
        " AND slot_id = :slot AND status = 'CONFIRMED'",
        1,
        {"appointment": Ref(appointment), "slot": Ref(slot)},
    )


NO_LIVE_REVIEW = DbCheck(
    "no review is left that a screen could still confirm",
    "SELECT count(*) FROM pending_actions p JOIN auth_sessions s USING (session_id)"
    " WHERE s.patient_id = :patient AND p.status = 'ACTIVE'",
    0,
    {"patient": Ref("patient")},
)


def _conversation(*, live_review_check: bool = True, **fields) -> Scenario:
    """A conversation ends with every review it prepared either committed or retired: a request
    the patient changed or dropped must not leave a pending action that Confirm could commit."""
    if live_review_check:
        fields["db_checks"] = [*fields["db_checks"], NO_LIVE_REVIEW]
    return Scenario(
        delivery=DELIVERY,
        actor="patient",
        model_actions=[],  # The model is real (or a stand-in extractor); nothing is scripted.
        expected_calls=[],
        check_call_order=False,
        version="1",
        **fields,
    )


# --- CV-01: the example from the design -----------------------------------------------------------


def _setup_clarify_then_change_date(ctx: SetupContext) -> Fixtures:
    first, second = _start(ctx, 14, 11), _start(ctx, 15, 10)
    clinic = _clinic(
        ctx,
        "Conversation Heart",
        [(f"Dr. Anita Rao {ctx.suffix}", "Cardiology")],
        [[first, second]],
    )
    code, patient, identity = _patient(ctx, "CV1")
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(first),
            "d2": _words(second),
            "slot1": clinic.slot_ids[0][0],
            "slot2": clinic.slot_ids[0][1],
            "patient": patient,
        },
        slots=list(clinic.slot_ids[0]),
        protected=identity,
        actor_code=code,
    )


CV_01 = _conversation(
    id="CV-01",
    title="Clarification, then the patient changes the date after a review",
    request="Book an appointment for {d1}.",
    request_terms=("d1", "d2", "clinic_name"),
    setup=_setup_clarify_then_change_date,
    timeline=[
        ModelTurn(follow_ups=("{clinic_name}.",), question_is_unnecessary=False),
        PatientSays("Actually, make it {d2}."),
        ModelTurn(),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the patient holds the second day's slot", "slot2"),
        _booked_check("nothing is booked on the first day", "slot1", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-02: the patient changes the doctor ------------------------------------------------------


def _setup_change_doctor(ctx: SetupContext) -> Fixtures:
    nine, ten = _start(ctx, 14, 9), _start(ctx, 14, 10)
    clinic = _clinic(
        ctx,
        "Conversation Family",
        [
            (f"Dr. Hana Okafor {ctx.suffix}", "Family Medicine"),
            (f"Dr. Sol Brandt {ctx.suffix}", "Family Medicine"),
        ],
        [[nine], [ten]],
    )
    code, patient, identity = _patient(ctx, "CV2")
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(nine),
            "doctor_a": "Okafor",
            "doctor_b": "Brandt",
            "slot_a": clinic.slot_ids[0][0],
            "slot_b": clinic.slot_ids[1][0],
            "patient": patient,
        },
        slots=[clinic.slot_ids[0][0], clinic.slot_ids[1][0]],
        protected=identity,
        actor_code=code,
    )


CV_02 = _conversation(
    id="CV-02",
    title="The patient asks for a different doctor after a review",
    request="Book me with Dr. {doctor_a} at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1", "doctor_a", "doctor_b"),
    setup=_setup_change_doctor,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("Actually, I'd rather see Dr. {doctor_b}."),
        ModelTurn(),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot_b"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the patient holds the second doctor's slot", "slot_b"),
        _booked_check("the first doctor's slot is not booked", "slot_a", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-03: the patient changes the clinic ------------------------------------------------------


def _setup_change_clinic(ctx: SetupContext) -> Fixtures:
    start = _start(ctx, 14, 9)
    one = _clinic(
        ctx, "Conversation North", [(f"Dr. Ida Voss {ctx.suffix}", "Family Medicine")], [[start]]
    )
    two = _clinic(
        ctx, "Conversation South", [(f"Dr. Ravi Meyer {ctx.suffix}", "Family Medicine")], [[start]]
    )
    code, patient, identity = _patient(ctx, "CV3")
    return Fixtures(
        values={
            "clinic_a": one.name,
            "clinic_b": two.name,
            "d1": _words(start),
            "slot_a": one.slot_ids[0][0],
            "slot_b": two.slot_ids[0][0],
            "patient": patient,
        },
        slots=[one.slot_ids[0][0], two.slot_ids[0][0]],
        protected=identity,
        actor_code=code,
    )


CV_03 = _conversation(
    id="CV-03",
    title="The patient changes the clinic after a review",
    request="Book the first appointment at {clinic_a} on {d1}.",
    request_terms=("clinic_a", "clinic_b", "d1"),
    setup=_setup_change_clinic,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("Sorry, I meant {clinic_b}."),
        ModelTurn(),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot_b"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the patient holds the second clinic's slot", "slot_b"),
        _booked_check("the first clinic's slot is not booked", "slot_a", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-04: a partial doctor name matches two doctors -----------------------------------------


def _setup_ambiguous_doctor(ctx: SetupContext) -> Fixtures:
    nine, ten = _start(ctx, 14, 9), _start(ctx, 14, 10)
    clinic = _clinic(
        ctx,
        "Conversation Lee",
        [
            (f"Dr. Lee North {ctx.suffix}", "Family Medicine"),
            (f"Dr. Lee South {ctx.suffix}", "Family Medicine"),
        ],
        [[nine], [ten]],
    )
    code, patient, identity = _patient(ctx, "CV4")
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(nine),
            "partial": "Lee",
            "full": "Lee South",
            "slot_north": clinic.slot_ids[0][0],
            "slot_south": clinic.slot_ids[1][0],
            "patient": patient,
        },
        slots=[clinic.slot_ids[0][0], clinic.slot_ids[1][0]],
        protected=identity,
        actor_code=code,
    )


CV_04 = _conversation(
    id="CV-04",
    title="A partial doctor name matches two doctors and the patient chooses",
    request="Book me with Dr. {partial} at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1", "partial", "full"),
    setup=_setup_ambiguous_doctor,
    timeline=[
        ModelTurn(follow_ups=("Dr. {full}, please.",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot_south"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the patient holds the chosen doctor's slot", "slot_south"),
        _booked_check("the other doctor's slot is not booked", "slot_north", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-05, CV-06, CV-07: several appointments, rescheduling, cancelling ---------------------


def _setup_two_appointments(ctx: SetupContext) -> Fixtures:
    morning, afternoon = _start(ctx, 14, 9), _start(ctx, 14, 14)
    target, later = _start(ctx, 15, 10), _start(ctx, 16, 13)
    clinic = _clinic(
        ctx,
        "Conversation Move",
        [(f"Dr. Marta Quill {ctx.suffix}", "Family Medicine")],
        [[morning, afternoon, target, later]],
    )
    code, patient, identity = _patient(ctx, "CV5")
    slots = clinic.slot_ids[0]
    appt_morning, _ = book_slot_for(ctx.engine, code, slots[0])
    appt_afternoon, _ = book_slot_for(ctx.engine, code, slots[1])
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(morning),
            "d2": _words(target),
            "d3": _words(later),
            "appt_morning": appt_morning,
            "appt_afternoon": appt_afternoon,
            "slot_morning": slots[0],
            "slot_afternoon": slots[1],
            "slot_d2": slots[2],
            "slot_d3": slots[3],
            "patient": patient,
        },
        slots=list(slots),
        protected=identity,
        actor_code=code,
    )


CV_05 = _conversation(
    id="CV-05",
    title="Two appointments match, the patient says which by its time, then it is moved",
    request="I need to move my appointment at {clinic_name} to {d2}.",
    request_terms=("clinic_name", "d1", "d2"),
    setup=_setup_two_appointments,
    timeline=[
        ModelTurn(follow_ups=("The one at 2 PM.",), question_is_unnecessary=False),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(
                action="RESCHEDULE",
                target=Ref("appt_afternoon"),
                expected={"new_slot_id": Ref("slot_d2")},
            ),
        ),
        ModelTurn(),
    ],
    expected_claims=frozenset({RESCHEDULED}),
    db_checks=[
        _on_slot_check("the 2 PM appointment moved", "appt_afternoon", "slot_d2"),
        _on_slot_check("the 9 AM appointment did not move", "appt_morning", "slot_morning"),
    ],
    authorized_filters=CHANGE_FILTERS,
)

CV_06 = _conversation(
    id="CV-06",
    title="After a reschedule review the patient picks another new day",
    request="Please move my {d1} appointment at {clinic_name} to {d2}, the 9 AM one.",
    request_terms=("clinic_name", "d1", "d2", "d3"),
    setup=_setup_two_appointments,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("Actually, {d3} works better for me."),
        ModelTurn(),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(
                action="RESCHEDULE",
                target=Ref("appt_morning"),
                expected={"new_slot_id": Ref("slot_d3")},
            ),
        ),
        ModelTurn(),
    ],
    expected_claims=frozenset({RESCHEDULED}),
    db_checks=[
        _on_slot_check("the 9 AM appointment moved to the later day", "appt_morning", "slot_d3"),
        _on_slot_check("the 2 PM appointment did not move", "appt_afternoon", "slot_afternoon"),
    ],
    authorized_filters=CHANGE_FILTERS,
)

CV_07 = _conversation(
    id="CV-07",
    title="Two appointments match and the patient picks the second by position",
    request="Please cancel my appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_two_appointments,
    timeline=[
        ModelTurn(follow_ups=("The second one.",), question_is_unnecessary=False),
        UiConfirm(
            tool=PREPARE,
            intent=PendingIntent(action="CANCEL", target=Ref("appt_afternoon")),
        ),
        ModelTurn(),
    ],
    expected_claims=frozenset({CANCELLED}),
    db_checks=[
        _status_check("the second appointment is cancelled", "appt_afternoon", "CANCELLED"),
        _status_check("the first appointment is still confirmed", "appt_morning", "CONFIRMED"),
    ],
    authorized_filters=CHANGE_FILTERS,
)


# --- CV-08: the patient drops a prepared request ---------------------------------------------


def _setup_one_appointment(ctx: SetupContext) -> Fixtures:
    start = _start(ctx, 14, 11)
    clinic = _clinic(
        ctx,
        "Conversation Cancel",
        [(f"Dr. Robin Vale {ctx.suffix}", "Internal Medicine")],
        [[start]],
    )
    code, patient, identity = _patient(ctx, "CV8")
    appointment, _ = book_slot_for(ctx.engine, code, clinic.slot_ids[0][0])
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(start),
            "appointment": appointment,
            "slot": clinic.slot_ids[0][0],
            "patient": patient,
        },
        slots=[clinic.slot_ids[0][0]],
        protected=identity,
        actor_code=code,
    )


CV_08 = _conversation(
    id="CV-08",
    title="The patient drops a prepared cancellation and nothing is changed",
    request="Please cancel my appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_one_appointment,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("Never mind, forget it."),
        ModelTurn(),
    ],
    expected_claims=frozenset(),
    db_checks=[_status_check("the appointment is still confirmed", "appointment", "CONFIRMED")],
    authorized_filters=CHANGE_FILTERS,
)


# --- CV-09: missing information, asked for step by step ----------------------------------------


def _setup_single_slot(ctx: SetupContext) -> Fixtures:
    start, other = _start(ctx, 14, 9), _start(ctx, 15, 10)
    clinic = _clinic(
        ctx,
        "Conversation Single",
        [(f"Dr. Mina Castro {ctx.suffix}", "Family Medicine")],
        [[start, other]],
    )
    code, patient, identity = _patient(ctx, "CV9")
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(start),
            "d2": _words(other),
            "d2_day": _ordinal(other),
            "slot1": clinic.slot_ids[0][0],
            "slot2": clinic.slot_ids[0][1],
            "patient": patient,
        },
        slots=list(clinic.slot_ids[0]),
        protected=identity,
        actor_code=code,
    )


CV_09 = _conversation(
    id="CV-09",
    title="A vague request is completed by answering the clinic and date questions",
    request="I need an appointment.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_single_slot,
    timeline=[
        ModelTurn(follow_ups=("{clinic_name}", "{d1}"), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[_booked_check("the patient holds the slot", "slot1")],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-10: an unrelated question, then something that must not continue the old request ---------


def _setup_two_times(ctx: SetupContext) -> Fixtures:
    nine, ten = _start(ctx, 14, 9), _start(ctx, 14, 10)
    clinic = _clinic(
        ctx,
        "Conversation Times",
        [(f"Dr. Aiko Lund {ctx.suffix}", "Family Medicine")],
        [[nine, ten]],
    )
    code, patient, identity = _patient(ctx, "CV10")
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(nine),
            "slot_nine": clinic.slot_ids[0][0],
            "slot_ten": clinic.slot_ids[0][1],
            "patient": patient,
        },
        slots=list(clinic.slot_ids[0]),
        protected=identity,
        actor_code=code,
    )


CV_10 = _conversation(
    id="CV-10",
    title="An unrelated question ends the old request instead of continuing it",
    request="Book an appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_two_times,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("Quick question, what time does the pharmacy close?"),
        ModelTurn(),
        PatientSays("The 9 AM one, please."),
        ModelTurn(),
    ],
    expected_claims=frozenset(),
    forbidden_tools=frozenset({PREPARE}),
    db_checks=[
        _booked_check("nothing was booked at 9", "slot_nine", 0),
        _booked_check("nothing was booked at 10", "slot_ten", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


# --- CV-11: not signed in ---------------------------------------------------------------------


def _setup_unauthenticated(ctx: SetupContext) -> Fixtures:
    start = _start(ctx, 14, 11)
    clinic = _clinic(
        ctx, "Conversation Guest", [(f"Dr. Noor Pike {ctx.suffix}", "Internal Medicine")], [[start]]
    )
    code, _, identity = _patient(ctx, "CV11")
    appointment, _ = book_slot_for(ctx.engine, code, clinic.slot_ids[0][0])
    return Fixtures(
        values={
            "clinic_name": clinic.name,
            "d1": _words(start),
            "appointment": appointment,
        },
        slots=[clinic.slot_ids[0][0]],
        protected=identity,
        actor_code=None,
    )


CV_11 = _conversation(
    live_review_check=False,  # Not signed in: there is no patient and nothing to prepare.
    id="CV-11",
    title="Not signed in: the agent sends the patient to sign-in and never asks for a PIN",
    request="Please cancel my appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_unauthenticated,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("My PIN is 0000, so can you cancel it now?"),
        ModelTurn(),
    ],
    expected_claims=frozenset(),
    forbidden_tools=frozenset({PREPARE}),
    db_checks=[_status_check("the appointment is still confirmed", "appointment", "CONFIRMED")],
    authorized_filters=CHANGE_FILTERS,
)
CV_11 = Scenario(**{**vars(CV_11), "actor": PUBLIC})


# --- CV-12: no openings, then another day ------------------------------------------------------


CV_12 = _conversation(
    id="CV-12",
    title="No openings on the first day, so the patient tries another",
    request="Book an appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1", "d2"),
    setup=lambda ctx: _setup_only_second_day(ctx),
    timeline=[
        ModelTurn(follow_ups=("How about {d2}?",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[_booked_check("the patient holds the second day's slot", "slot2")],
    authorized_filters=BOOK_FILTERS,
)


def _setup_only_second_day(ctx: SetupContext) -> Fixtures:
    fixtures = _setup_single_slot(ctx)
    # The first day has no open slot: another patient already holds it.
    other = create_patient(ctx.engine, ctx.suffix, ctx.pin, "CV12B")
    book_slot_for(ctx.engine, other, fixtures.values["slot1"])
    fixtures.protected.extend(patient_identity(ctx.engine, other))
    return fixtures


# --- CV-13 and CV-14: the model's reading fails ----------------------------------------------


CV_13 = _conversation(
    id="CV-13",
    title="The first reading of a message is invalid; the agent recovers and books",
    request="Book an appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_single_slot,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[_booked_check("the patient holds the slot", "slot1")],
    authorized_filters=BOOK_FILTERS,
)


def _setup_two_clinics_one_day(ctx: SetupContext) -> Fixtures:
    morning, afternoon = _start(ctx, 14, 9), _start(ctx, 14, 14)
    one = _clinic(
        ctx, "Conversation East", [(f"Dr. Kai Adler {ctx.suffix}", "Family Medicine")], [[morning]]
    )
    two = _clinic(
        ctx, "Conversation West", [(f"Dr. Zoe Hart {ctx.suffix}", "Family Medicine")], [[afternoon]]
    )
    code, patient, identity = _patient(ctx, "CV14")
    first, _ = book_slot_for(ctx.engine, code, one.slot_ids[0][0])
    second, _ = book_slot_for(ctx.engine, code, two.slot_ids[0][0])
    return Fixtures(
        values={
            "clinic_name": one.name,
            "d1": _words(morning),
            "appt_first": first,
            "appt_second": second,
            "patient": patient,
        },
        slots=[one.slot_ids[0][0], two.slot_ids[0][0]],
        protected=identity,
        actor_code=code,
    )


CV_14 = _conversation(
    id="CV-14",
    title="The model is unavailable: the agent does not guess which appointment to cancel",
    request="I need to cancel my appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_two_clinics_one_day,
    timeline=[
        ModelTurn(follow_ups=("The first one.",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="CANCEL", target=Ref("appt_first"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({CANCELLED}),
    db_checks=[
        _status_check("the chosen appointment is cancelled", "appt_first", "CANCELLED"),
        _status_check("the other appointment is still confirmed", "appt_second", "CONFIRMED"),
    ],
    authorized_filters=CHANGE_FILTERS,
)


# --- CV-15: the patient corrects a date ------------------------------------------------------

CV_15 = _conversation(
    id="CV-15",
    title="The patient corrects the day with only its number",
    request="Book an appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1", "d2_day"),
    setup=_setup_single_slot,
    timeline=[
        ModelTurn(question_is_unnecessary=False),
        PatientSays("No, sorry, I meant {d2_day}."),
        ModelTurn(),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the patient holds the corrected day's slot", "slot2"),
        _booked_check("the first day's slot is not booked", "slot1", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


CONVERSATION_SCENARIOS = (
    CV_01,
    CV_02,
    CV_03,
    CV_04,
    CV_05,
    CV_06,
    CV_07,
    CV_08,
    CV_09,
    CV_10,
    CV_11,
    CV_12,
    CV_13,
    CV_14,
    CV_15,
)

# Faults a test injects into the model's reading, so recovery is exercised with the real tools.
# "invalid_once" makes the first reading of the first message invalid; "outage" makes every
# reading fail as if the model server were unreachable.
FAULTS = {"CV-13": "invalid_once", "CV-14": "outage"}


# --- the same situations in other words ---------------------------------------------------------
#
# Written once, before the real model was run on them, and not edited afterwards. They exist to
# show how the conversation layer copes with wording it was not developed against. The stand-in
# reader of the deterministic tests does not understand them; only a real model is asked.


def _variant(base: Scenario, suffix: str, **changes) -> Scenario:
    return replace(base, id=f"{base.id}{suffix}", **changes)


CONVERSATION_VARIANTS = (
    _variant(
        CV_01,
        "V",
        title="Wording variant: asks to be seen, then prefers another day",
        request="Hi, could I get in to see someone on {d1}?",
        timeline=[
            ModelTurn(follow_ups=("It's {clinic_name}.",), question_is_unnecessary=False),
            PatientSays("Hmm, {d2} would suit me better actually."),
            ModelTurn(),
            UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
            ModelTurn(),
        ],
    ),
    _variant(
        CV_05,
        "V",
        title="Wording variant: shift a visit, say which by its time",
        request="My schedule changed. Can I shift my visit at {clinic_name} over to {d2}?",
        timeline=[
            ModelTurn(follow_ups=("The 2:00 PM appointment.",), question_is_unnecessary=False),
            *CV_05.timeline[1:],
        ],
    ),
    _variant(
        CV_06,
        "V",
        title="Wording variant: change a visit by position, then another new day",
        request="Please change my {d1} visit at {clinic_name} to {d2}.",
        timeline=[
            ModelTurn(follow_ups=("The first one.",), question_is_unnecessary=False),
            PatientSays("Sorry, could we do {d3} instead?"),
            ModelTurn(),
            *CV_06.timeline[3:],
        ],
    ),
    _variant(
        CV_07,
        "V",
        title="Wording variant: can't make it, then picks the second",
        request="I won't be able to make my appointment at {clinic_name} on {d1}, please cancel.",
        timeline=[
            ModelTurn(follow_ups=("The second one, please.",), question_is_unnecessary=False),
            *CV_07.timeline[1:],
        ],
    ),
    _variant(
        CV_12,
        "V",
        title="Wording variant: asks what is available, then another day",
        request="Do you have anything at {clinic_name} on {d1}? I'd like to book.",
        timeline=[
            ModelTurn(follow_ups=("Okay, what about {d2}?",), question_is_unnecessary=False),
            *CV_12.timeline[1:],
        ],
    ),
    _variant(
        CV_15,
        "V",
        title="Wording variant: booked the wrong day, corrects it by number",
        request="Book me in at {clinic_name} for {d1}.",
        timeline=[
            ModelTurn(question_is_unnecessary=False),
            PatientSays("Oops, wrong day, make that {d2_day}."),
            *CV_15.timeline[2:],
        ],
    ),
    _variant(
        CV_08,
        "V",
        title="Wording variant: keeps the appointment after a cancellation review",
        request="I want to cancel my {d1} appointment at {clinic_name}.",
        timeline=[
            ModelTurn(question_is_unnecessary=False),
            PatientSays("Actually I'll keep it, never mind."),
            ModelTurn(),
        ],
    ),
    _variant(
        CV_10,
        "V",
        title="Wording variant: an unrelated question mid-request",
        request="I'd like to book at {clinic_name} on {d1}.",
        timeline=[
            ModelTurn(question_is_unnecessary=False),
            PatientSays("Do you take walk-ins?"),
            ModelTurn(),
            PatientSays("OK, 9 AM then."),
            ModelTurn(),
        ],
    ),
)


# --- spoken dates and times ---------------------------------------------------------------------
#
# The patient says dates and times the way people say them aloud. Each scenario states its dates
# and times in words, so what is judged is whether the right appointment ends up booked, with the
# right questions asked on the way, and nothing guessed.

_DAY_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
    8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty-first", 22: "twenty-second",
    23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth", 26: "twenty-sixth",
    27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
    31: "thirty-first",
}  # fmt: skip


def _spoken(start) -> str:
    """The date as said aloud: "November seventeenth"."""
    local = start.astimezone(ZoneInfo(ZONE))
    return f"{MONTHS[local.month - 1]} {_DAY_WORDS[local.day]}"


def _spoken_of(start) -> str:
    """ "the twenty-fourth of November"."""
    local = start.astimezone(ZoneInfo(ZONE))
    return f"the {_DAY_WORDS[local.day]} of {MONTHS[local.month - 1]}"


def _spoken_clinic(ctx: SetupContext, prefix: str, role: str, slots: list[tuple[int, int, int]]):
    """A clinic with one doctor and slots at (days ahead, hour, minute) clinic-local times."""
    starts = [
        local_start(ctx.now, ZONE, days_ahead=days, hour=hour, minute=minute)
        for days, hour, minute in slots
    ]
    clinic = _clinic(ctx, prefix, [(f"Dr. Wren Alder {ctx.suffix}", "Family Medicine")], [starts])
    code, patient, identity = _patient(ctx, role)
    return clinic, starts, code, patient, identity


def _spoken_fixtures(clinic, starts, code, patient, identity, **extra) -> Fixtures:
    values = {"clinic_name": clinic.name, "patient": patient, **extra}
    for number, slot_id in enumerate(clinic.slot_ids[0]):
        values[f"slot{number}"] = slot_id
    return Fixtures(
        values=values,
        slots=list(clinic.slot_ids[0]),
        protected=identity,
        actor_code=code,
    )


def _setup_sp_time(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(
        ctx, "Spoken Clock", "SP1", [(14, 9, 0), (14, 11, 30), (14, 14, 0)]
    )
    return _spoken_fixtures(clinic, starts, *rest, d1=_spoken(starts[0]))


SP_01 = _conversation(
    id="SP-01",
    title="A date and a time said in words: November sixteenth, eleven thirty in the morning",
    request="Book an appointment at {clinic_name} on {d1} at eleven thirty in the morning.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_sp_time,
    timeline=[
        ModelTurn(question_is_unnecessary=True),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the 11:30 slot is booked", "slot1"),
        _booked_check("the 9:00 slot is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_quarter(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(
        ctx, "Spoken Quarter", "SP2", [(14, 2, 15), (14, 9, 0), (14, 14, 15)]
    )
    return _spoken_fixtures(clinic, starts, *rest, d1=_spoken(starts[0]))


SP_02 = _conversation(
    id="SP-02",
    title="Quarter past two has no am or pm: asked, then booked as answered",
    request="Book an appointment at {clinic_name} on {d1} at quarter past two.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_sp_quarter,
    timeline=[
        ModelTurn(follow_ups=("In the afternoon.",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the afternoon slot is booked", "slot2"),
        _booked_check("the 2:15 AM slot is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_after(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(
        ctx, "Spoken After", "SP3", [(14, 9, 0), (14, 10, 30), (14, 14, 0)]
    )
    return _spoken_fixtures(clinic, starts, *rest, d1=_spoken(starts[0]))


SP_03 = _conversation(
    id="SP-03",
    title="The first available appointment after ten: asked which ten, then the earliest after it",
    request="I need the first available appointment after ten at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1"),
    setup=_setup_sp_after,
    timeline=[
        ModelTurn(follow_ups=("In the morning.",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the 10:30 slot is booked", "slot1"),
        _booked_check("the 9:00 slot is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_tomorrow(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(
        ctx, "Spoken Tomorrow", "SP4", [(0, 15, 0), (1, 10, 0), (1, 15, 0), (2, 15, 0)]
    )
    return _spoken_fixtures(clinic, starts, *rest)


SP_04 = _conversation(
    id="SP-04",
    title="Tomorrow afternoon: the next day, in the afternoon only",
    request="Book an appointment at {clinic_name} tomorrow afternoon.",
    request_terms=("clinic_name",),
    setup=_setup_sp_tomorrow,
    timeline=[
        ModelTurn(question_is_unnecessary=True),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot2"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("tomorrow's afternoon slot is booked", "slot2"),
        _booked_check("tomorrow's morning slot is not", "slot1", 0),
        _booked_check("today's slot is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_correction(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(ctx, "Spoken Fix", "SP5", [(14, 11, 0), (22, 10, 0)])
    return _spoken_fixtures(clinic, starts, *rest, d1=_spoken(starts[0]), d2=_spoken_of(starts[1]))


SP_05 = _conversation(
    id="SP-05",
    title="A spoken correction: November sixteenth, then the twenty-fourth of November",
    request="Book an appointment at {clinic_name} on {d1}.",
    request_terms=("clinic_name", "d1", "d2"),
    setup=_setup_sp_correction,
    timeline=[
        ModelTurn(question_is_unnecessary=True),
        PatientSays("Actually, {d2}, please."),
        ModelTurn(),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the corrected day's slot is booked", "slot1"),
        _booked_check("the first day's slot is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_next_tuesday(ctx: SetupContext) -> Fixtures:
    clinic, starts, *rest = _spoken_clinic(ctx, "Spoken Tuesday", "SP6", [(1, 9, 0), (8, 9, 0)])
    return _spoken_fixtures(clinic, starts, *rest, d2=_spoken(starts[1]))


SP_06 = _conversation(
    id="SP-06",
    title="Next Tuesday, said on a Monday, has two meanings: asked, never guessed",
    request="Book an appointment at {clinic_name} next Tuesday morning.",
    request_terms=("clinic_name", "d2"),
    setup=_setup_sp_next_tuesday,
    timeline=[
        ModelTurn(follow_ups=("I mean {d2}.",), question_is_unnecessary=False),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the later Tuesday is booked", "slot1"),
        _booked_check("the nearer Tuesday is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


def _setup_sp_auckland(ctx: SetupContext) -> Fixtures:
    zone = "Pacific/Auckland"  # 04:00 on the 3rd when it is 15:00 UTC on the 2nd.
    starts = [
        local_start(ctx.now, zone, days_ahead=0, hour=14),
        local_start(ctx.now, zone, days_ahead=1, hour=14),
    ]
    clinic = create_clinic(
        ctx.engine,
        name=unique_place_name(ctx.engine, "Spoken Auckland", ctx.suffix),
        timezone=zone,
        doctors=[(f"Dr. Tui Marsh {ctx.suffix}", "Family Medicine")],
        doctor_slot_starts=[starts],
    )
    code, patient, identity = _patient(ctx, "SP7")
    return _spoken_fixtures(clinic, starts, code, patient, identity)


SP_07 = _conversation(
    id="SP-07",
    title="Tomorrow means the clinic's tomorrow: an Auckland clinic is already a day ahead of UTC",
    request="Book an appointment at {clinic_name} tomorrow.",
    request_terms=("clinic_name",),
    setup=_setup_sp_auckland,
    timeline=[
        ModelTurn(question_is_unnecessary=True),
        UiConfirm(tool=PREPARE, intent=PendingIntent(action="BOOK", target=Ref("slot1"))),
        ModelTurn(),
    ],
    expected_claims=frozenset({BOOKED}),
    db_checks=[
        _booked_check("the clinic's tomorrow is booked", "slot1"),
        _booked_check("the clinic's today is not", "slot0", 0),
    ],
    authorized_filters=BOOK_FILTERS,
)


SPOKEN_SCENARIOS = (SP_01, SP_02, SP_03, SP_04, SP_05, SP_06, SP_07)

# Spoken phrasings that were not used while the layer was developed; see CONVERSATION_VARIANTS.
SPOKEN_VARIANTS = (
    _variant(
        SP_01,
        "V",
        title="Wording variant: a booking for a date and time said in words",
        request="Could I get in at {clinic_name} on {d1}, say half past eleven in the morning?",
    ),
    _variant(
        SP_02,
        "V",
        title="Wording variant: a quarter after two, then am or pm",
        request="I'd like a slot at {clinic_name} on {d1} around, hmm, a quarter after two.",
        timeline=[
            ModelTurn(follow_ups=("P.M.",), question_is_unnecessary=False),
            *SP_02.timeline[1:],
        ],
    ),
    _variant(
        SP_04,
        "V",
        title="Wording variant: tomorrow in the afternoon",
        request="Do you have anything at {clinic_name} tomorrow in the afternoon? Book it.",
    ),
    _variant(
        SP_05,
        "V",
        title="Wording variant: the correction said as a day of the month",
        request="Please book {clinic_name} for {d1}.",
        timeline=[
            ModelTurn(question_is_unnecessary=True),
            PatientSays("Sorry, I meant {d2}."),
            *SP_05.timeline[2:],
        ],
    ),
)

ALL_SCENARIOS = CONVERSATION_SCENARIOS + CONVERSATION_VARIANTS + SPOKEN_SCENARIOS + SPOKEN_VARIANTS
