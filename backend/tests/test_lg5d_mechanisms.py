"""LG-5D: the general mechanisms behind three conversational fixes, tested on their own with
scripted model readings. (How well the real model does is measured by the LG-5D evaluation.)

Each test states a rule, not a phrase:

- a correction reaches a request in progress however the first reading missed it, and the review it
  replaces is dropped, never reused;
- a lookup that turns into a booking carries what was asked and shown; ambiguity is asked about;
- a review is prepared only for a request that was clearly asked for, whatever was proposed."""

from datetime import date

import pytest
from test_conversation_unit import NOW, RetiringBackend
from test_orchestrator import MONDAY, THURSDAY, ask, book, days_backend, sched, session

from app.evaluation.hybrid import operation_stated, withdrawal_adds
from app.evaluation.orchestrator import ASK_TO_BOOK_REPLY
from app.evaluation.turn_reader import Revision, TurnReader

NEXT_THURSDAY = THURSDAY.isoformat()


def change(**fields) -> dict:
    return {"kind": "CHANGE", **fields}


def question(**fields) -> dict:
    return {"kind": "QUESTION", **fields}


# --- what counts as asking for a booking ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Book Pine Clinic on Monday",
        "I need an appointment Monday",
        "I'd like to see someone at Pine Clinic",
        "Can you get me in Thursday",
        "Please schedule me for Friday",
        "We'd like an appointment",
    ],
)
def test_an_act_of_booking_or_a_wish_for_an_appointment_is_a_request_by_its_wording(
    text: str,
) -> None:
    assert operation_stated("BOOK", text)


@pytest.mark.parametrize(
    "text",
    [
        "Any availability at Pine Clinic tomorrow afternoon?",
        "Are there openings on Monday?",
        "Is there an appointment available Friday?",
        "I'd like to know what's open",
        "I want to see what's available",
        "Is Dr. Kim around on Thursday?",
        "and Thursday?",
    ],
)
def test_words_that_only_ask_what_exists_are_not_a_request_by_their_wording(text: str) -> None:
    assert not operation_stated("BOOK", text)


def test_what_is_left_of_a_withdrawal_tells_a_plain_one_from_a_replacement() -> None:
    assert not withdrawal_adds("never mind")
    assert not withdrawal_adds("never mind, thanks")
    assert withdrawal_adds("forget it, the heart center instead")
    assert withdrawal_adds("scratch that, tomorrow at 11")


# --- no review from a question ---------------------------------------------------------------


def test_an_open_times_question_read_as_a_booking_prepares_no_review() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        {"asks_for": "NONE"},
    )

    reply = made.respond("Any availability at Pine Clinic on November 9?")

    assert "9:00 AM" in reply.text and "9:30 AM" in reply.text
    assert not reply.awaiting_confirmation and backend.prepared == []
    assert made.snapshot()["request"] == {} and made.snapshot()["review"] is None


def test_a_booking_the_second_reading_cannot_confirm_is_not_started() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        None,  # The second reading failed (not valid JSON): nothing is assumed.
    )

    reply = made.respond("Pine Clinic on November 9, anything open?")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_the_last_line_of_defence_refuses_to_prepare_for_an_unvouched_request() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    made.respond("Book Pine Clinic Monday morning")
    request = dict(made.snapshot()["request"], vouched=False)  # What a lookup would leave.
    made.graph.update_state(made.config, {"request": request})

    reply = made.respond("9:30 AM")

    assert reply.text == ASK_TO_BOOK_REPLY
    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_a_request_asked_for_by_an_act_of_booking_is_vouched_in_the_checkpoint() -> None:
    backend = days_backend()
    made, _ = session(
        backend, book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09")
    )

    made.respond("Book Pine Clinic on November 9.")

    assert made.snapshot()["request"]["vouched"] is True


# --- corrections after a review --------------------------------------------------------------


def reviewed(backend, *then):
    made, wire = session(
        backend,
        book(
            clinic_name="Pine Clinic", start_date=NEXT_THURSDAY, end_date=NEXT_THURSDAY, time="2 PM"
        ),
        *then,
    )
    first = made.respond("Book Pine Clinic on November 12 at 2 PM.")
    assert first.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s3"}]
    return made, wire


def test_a_correction_the_first_reading_missed_is_read_again_and_replaces_the_review() -> None:
    backend = days_backend()
    made, wire = reviewed(backend, sched(same_request=True), change(time="3 PM"))

    reply = made.respond("make it 3 in the afternoon")

    assert reply.awaiting_confirmation
    assert backend.prepared[-1] == {"action": "BOOK", "slot_id": "s4"}
    assert "3:00 PM" in reply.text and "2:00 PM" not in reply.text  # Nothing outdated is reused.
    assert made.snapshot()["review"]["pending_action_id"] == "pending-2"
    assert len(wire.payloads) == 3  # The reading, then one second reading with the request in view.
    assert "2 PM" in wire.payloads[2]["messages"][0]["content"]


def test_a_correction_that_changes_nothing_the_request_has_keeps_the_review() -> None:
    backend = days_backend()
    made, _ = reviewed(backend, sched(same_request=True), {"kind": "NONE"})

    reply = made.respond("hmm, let me think")

    assert "already prepared" in reply.text
    assert made.snapshot()["review"]["pending_action_id"] == "pending-1"


def test_a_question_asked_while_a_review_waits_is_answered_and_leaves_the_review_alone() -> None:
    backend = days_backend()
    made, _ = reviewed(
        backend,
        sched(),
        question(start_date="2026-11-09", end_date="2026-11-09"),
    )

    reply = made.respond("is anything open on Monday?")

    assert "9:00 AM" in reply.text
    assert made.snapshot()["review"]["pending_action_id"] == "pending-1"
    assert len(backend.prepared) == 1


def test_a_doctor_who_is_only_described_is_named_from_the_directorys_own_list() -> None:
    reading = Revision(kind="CHANGE", doctor_name="Dr. Aiden Kim")
    doctors = {"Dr. Anita Rao": "Family Medicine", "Dr. Aiden Kim": "Cardiology"}

    read = TurnReader._grounded_revision(
        reading, "actually the heart doctor", date(2026, 11, 2), None, doctors
    )

    assert read.intent.doctor_name == "Dr. Aiden Kim"


def test_a_doctor_the_directory_does_not_list_is_not_accepted() -> None:
    reading = Revision(kind="CHANGE", doctor_name="Dr. Nobody")
    doctors = {"Dr. Anita Rao": "Family Medicine"}

    read = TurnReader._grounded_revision(
        reading, "actually the heart doctor", date(2026, 11, 2), None, doctors
    )

    assert read.intent.doctor_name is None


def test_a_specialty_that_fits_exactly_one_listed_doctor_names_that_doctor() -> None:
    reading = Revision(kind="CHANGE", specialty="Cardiology")
    doctors = {"Dr. Anita Rao": "Family Medicine", "Dr. Aiden Kim": "Cardiology"}

    read = TurnReader._grounded_revision(
        reading, "actually the heart doctor", date(2026, 11, 2), None, doctors
    )

    assert read.intent.doctor_name == "Dr. Aiden Kim" and read.intent.specialty is None


def test_a_specialty_that_fits_two_listed_doctors_names_neither() -> None:
    reading = Revision(kind="CHANGE", specialty="Cardiology")
    doctors = {"Dr. A": "Cardiology", "Dr. B": "Cardiology"}

    read = TurnReader._grounded_revision(
        reading, "actually the heart doctor", date(2026, 11, 2), None, doctors
    )

    assert read.intent.doctor_name is None


def test_a_day_or_time_the_second_reading_invents_is_dropped() -> None:
    reading = Revision(kind="CHANGE", start_date=date(2026, 11, 20), time="7 PM")

    read = TurnReader._grounded_revision(reading, "make it later", date(2026, 11, 2), None, {})

    assert read.intent.start_date is None and read.intent.time is None


def test_a_withdrawal_with_a_replacement_is_a_correction_not_a_dropped_request() -> None:
    backend = days_backend()
    made, _ = reviewed(
        backend,
        sched(message_type="ABANDON"),
        change(time="3 PM"),
    )

    reply = made.respond("scratch that, 3 PM")

    assert reply.awaiting_confirmation and "3:00 PM" in reply.text
    assert backend.prepared[-1] == {"action": "BOOK", "slot_id": "s4"}


def test_a_plain_withdrawal_still_drops_the_request_and_its_review() -> None:
    backend = days_backend()
    made, _ = reviewed(backend)  # "never mind" is settled without a model.
    made.quick_paths = True

    reply = made.respond("never mind")

    assert "dropped" in reply.text and made.snapshot()["review"] is None
    assert [p for _, p in backend.discarded] == ["pending-1"]


def test_a_withdrawal_that_says_more_but_changes_nothing_still_drops_the_request() -> None:
    backend = days_backend()
    made, _ = reviewed(backend, sched(message_type="ABANDON"), {"kind": "NONE"})

    reply = made.respond("never mind, I'll call them")

    assert "dropped" in reply.text and made.snapshot()["review"] is None
    assert [p for _, p in backend.discarded] == ["pending-1"]


# --- a lookup that turns into a booking -------------------------------------------------------


def test_the_day_asked_about_in_a_lookup_is_the_day_of_the_booking_that_follows() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("DOCTORS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        book(doctor_name="Dr. Anita Rao", clinic_name="Pine Clinic", time="2 PM"),
    )
    made.respond("Who is at Pine Clinic on November 12?")
    assert made.snapshot()["context"]["asked"]["start_date"] == "2026-11-12"

    reply = made.respond("book Dr. Anita Rao at 2 PM")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s3"}]


def test_a_booking_asked_for_after_a_lookup_carries_its_clinic_and_day() -> None:
    backend = days_backend()
    backend.slots = [s for s in backend.slots if s[0] != "s3"]  # Only 3 PM on the 12th.
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        book(),
    )
    made.respond("What does Pine Clinic have on November 12?")

    reply = made.respond("great, book that")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]


def test_a_doctor_asked_about_with_no_clinic_is_looked_up_at_the_one_clinic_they_work_at() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "OPENINGS", doctor_name="Dr. Anita Rao", start_date="2026-11-09", end_date="2026-11-09"
        ),
    )

    reply = made.respond("Is Dr. Anita Rao around on Monday?")

    assert "Which clinic" not in reply.text and "9:00 AM" in reply.text
    assert backend.prepared == []


def test_a_question_that_names_no_place_is_about_the_place_just_asked_about() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(starts_booking=True, start_date="2026-11-12", end_date="2026-11-12"),
        {"asks_for": "NONE"},
    )
    made.respond("Anything at Pine Clinic on Monday?")

    reply = made.respond("and the 12th?")

    assert "2:00 PM" in reply.text and backend.prepared == []
    assert not reply.awaiting_confirmation


def test_agreeing_without_choosing_picks_none_of_several_offers() -> None:
    backend, made = listed_two()

    reply = made.respond("ok that's fine")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_a_pick_only_the_model_proposes_picks_nothing() -> None:
    backend, made = listed_two(sched(operation="BOOK", option_number=2))

    reply = made.respond("one of those")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_a_reference_the_words_settle_prepares_that_opening_without_a_second_model_call() -> None:
    backend, made = listed_two(sched(operation="BOOK"))

    reply = made.respond("the later one")

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]


def listed_two(*readings):
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        *[r for r in readings if r is not None],
    )
    listed = made.respond("What does Pine Clinic have on November 12?")
    assert "2:00 PM" in listed.text and "3:00 PM" in listed.text
    return backend, made


# --- times and days as patients correct them -------------------------------------------------


def test_a_time_said_where_an_appointment_is_moved_to_is_the_new_time() -> None:
    backend = RetiringBackend()
    backend.appointment("a1", "c-pine", "d-rao", MONDAY, 10)
    backend.slot("s1", "c-pine", "d-rao", THURSDAY, 14)
    backend.slot("s2", "c-pine", "d-kim", THURSDAY, 15)
    made, _ = session(
        backend,
        {"route": "SCHEDULING", "operation": "RESCHEDULE", "start_date": NEXT_THURSDAY},
    )

    reply = made.respond("move my appointment to November 12 at 2")

    assert "Which time" not in reply.text
    assert backend.prepared and backend.prepared[-1]["new_slot_id"] == "s1"


def test_an_hour_with_a_part_of_the_day_is_a_clock_time_but_a_day_of_the_month_is_not() -> None:
    from app.evaluation.spoken import normalize_spoken, stated_clocks

    assert stated_clocks(normalize_spoken("nope, 3 in the afternoon")) == {15 * 60}
    assert stated_clocks(normalize_spoken("November 12 in the morning")) == set()
    assert NOW  # The module's clock is imported for the sessions above.


def test_a_clinic_named_with_its_article_is_searched_for_without_it() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="The Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
        ),
    )

    reply = made.respond("What does the Pine Clinic have on Monday?")

    assert "9:00 AM" in reply.text


def test_a_doctor_the_reading_left_out_is_still_the_doctor_asked_about() -> None:
    backend = days_backend()
    backend.slot("s5", "c-pine", "d-kim", THURSDAY, 16)
    made, _ = session(
        backend,
        book(clinic_name="Pine Clinic", doctor_name="Dr. Anita Rao", start_date=NEXT_THURSDAY),
        ask("OPENINGS", start_date=NEXT_THURSDAY, end_date=NEXT_THURSDAY),  # No doctor named.
    )
    made.respond("Book Dr. Anita Rao at Pine Clinic on November 12")

    reply = made.respond("wait, is Dr. Aiden Kim free that day?")

    assert "November 12, 2026: 4:00 PM." in reply.text  # Dr. Kim's opening, not Dr. Rao's.
