"""LG-5E: the four design decisions, tested on their own with scripted model readings.

- A possessive ("Dr. Kim's Monday") names the doctor.
- Selection is deterministic: the model never chooses among offers, several matches are a question,
  and a delegation counts only when it resolves to exactly one opening.
- Right after the assistant asked whether to book, a short affirmative reply starts or continues a
  booking; it chooses no unspecified opening and commits nothing.
- An operation change must come from the patient's words and the state of the conversation."""

from datetime import date

import pytest
from test_conversation_unit import Backend, RetiringBackend
from test_orchestrator import MONDAY, THURSDAY, ask, book, days_backend, sched, session

from app.evaluation.hybrid import affirmative_stated, refers_to_existing
from app.evaluation.planner import _TITLED_NAME

NEXT_THURSDAY = THURSDAY.isoformat()


# --- possessive names -------------------------------------------------------------------------


def test_a_possessive_is_not_part_of_the_doctors_name() -> None:
    found = [m[0] for m in _TITLED_NAME.finditer("Compare Dr. Kim's Monday with Dr. Rao's Monday")]
    assert found == ["Dr. Kim", "Dr. Rao"]
    assert [m[0] for m in _TITLED_NAME.finditer("Is Dr. O'Brien in?")] == ["Dr. O'Brien"]


def test_a_doctors_day_asked_about_with_a_possessive_is_answered_for_that_doctor() -> None:
    backend = days_backend()
    made, _ = session(backend, ask("DOCTORS", clinic_name="Pine Clinic"))

    reply = made.respond("Is Dr. Anita Rao's Monday open at Pine Clinic?")

    assert "9:00 AM" in reply.text and "couldn't find" not in reply.text
    assert backend.prepared == []


def test_a_possessive_in_a_name_the_reading_wrote_is_still_the_doctor() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="Pine Clinic",
            doctor_name="Dr. Rao's",
            start_date="2026-11-09",
            end_date="2026-11-09",
        ),
    )

    reply = made.respond("What's Dr. Rao's Monday like at Pine Clinic?")

    assert "9:00 AM" in reply.text and "couldn't find" not in reply.text


# --- booking intent from context --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "yes"),
    [
        ("yes", True),
        ("okay, Wednesday", True),
        ("sure, the 3 PM", True),
        ("that works", True),
        ("yes?", False),
        ("Dr. Kim then", False),
        ("and Thursday?", False),
        ("not yet", False),
    ],
)
def test_only_a_plain_yes_that_is_not_a_question_counts_as_one(text: str, yes: bool) -> None:
    assert affirmative_stated(text) is yes


def one_opening_listed(*then):
    backend = days_backend()
    backend.slots = [s for s in backend.slots if s[0] != "s3"]  # Only the 3 PM on the 12th.
    made, wire = session(
        backend,
        ask(
            "OPENINGS", clinic_name="Pine Clinic", start_date=NEXT_THURSDAY, end_date=NEXT_THURSDAY
        ),
        *then,
    )
    listed = made.respond("What does Pine Clinic have on November 12?")
    assert "3:00 PM" in listed.text and "book, move, or cancel" in listed.text
    return backend, made, wire


def test_yes_to_the_question_whether_to_book_books_the_one_opening_listed() -> None:
    backend, made, _ = one_opening_listed(sched(message_type="AGREEMENT"))

    reply = made.respond("yes")

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]


def test_okay_and_a_day_continues_the_booking_from_the_lookup() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("DOCTORS", clinic_name="Pine Clinic"),
        sched(start_date=NEXT_THURSDAY, end_date=NEXT_THURSDAY),
    )
    made.respond("Who works at Pine Clinic?")

    reply = made.respond("okay, the 12th")

    assert "Which" in reply.text or reply.awaiting_confirmation
    assert made.snapshot()["request"]["operation"] == "BOOK"
    assert made.snapshot()["request"]["vouched"] is True


def test_yes_with_several_openings_chooses_none_of_them() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(message_type="AGREEMENT"),
    )
    made.respond("What does Pine Clinic have on Monday?")

    reply = made.respond("yes")

    assert backend.prepared == [] and not reply.awaiting_confirmation
    assert "Which" in reply.text


def test_a_reply_without_a_yes_is_not_a_booking_even_right_after_the_question() -> None:
    backend, made, _ = one_opening_listed(sched(), {"asks_for": "NONE"})

    reply = made.respond("and the 5th?")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_the_offer_to_book_lasts_one_turn() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(),
        sched(message_type="AGREEMENT"),
    )
    made.respond("What does Pine Clinic have on Monday?")
    assert made.snapshot()["context"]["booking_offered"] is True
    made.respond("hmm, let me think")
    assert made.snapshot()["context"]["booking_offered"] is False

    reply = made.respond("yes")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_a_yes_never_commits_anything_the_review_still_waits_for_the_app() -> None:
    backend, made, _ = one_opening_listed(sched(message_type="AGREEMENT"))

    reply = made.respond("yes")

    assert "confirm" in reply.text.lower() and made.snapshot()["review"] is not None
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]  # A review, nothing booked.


# --- operations --------------------------------------------------------------------------------


def with_a_booking_in_hand(*then):
    backend = RetiringBackend()
    for number, (day, hour) in enumerate([(MONDAY, 9), (MONDAY, 11), (THURSDAY, 14)]):
        backend.slot(f"s{number}", "c-pine", "d-rao", day, hour)
    backend.appointment("a1", "c-pine", "d-rao", date(2026, 11, 10), 10)
    made, wire = session(
        backend,
        book(
            clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09", time="9 AM"
        ),
        *then,
    )
    assert made.respond("Book Pine Clinic on Monday at 9 AM").awaiting_confirmation
    return backend, made, wire


def test_changing_a_booking_in_hand_stays_a_booking_whatever_the_model_proposes() -> None:
    backend, made, _ = with_a_booking_in_hand(
        sched(operation="RESCHEDULE", start_date=NEXT_THURSDAY, end_date=NEXT_THURSDAY),
    )

    reply = made.respond("change it to the 12th")

    assert backend.prepared[-1]["action"] == "BOOK" and "reschedul" not in reply.text.lower()
    assert all(call["action"] == "BOOK" for call in backend.prepared)


def test_moving_an_appointment_the_patient_already_has_is_a_move_when_the_words_say_so() -> None:
    backend, made, _ = with_a_booking_in_hand(
        sched(operation="RESCHEDULE", start_date="2026-11-10", new_date=NEXT_THURSDAY),
    )

    made.respond("move my appointment on the 10th to the 12th")

    assert backend.prepared[-1]["action"] == "RESCHEDULE"


def test_a_cancel_the_words_do_not_state_is_not_a_cancel_even_if_proposed() -> None:
    backend, made, _ = with_a_booking_in_hand(sched(operation="CANCEL", start_date="2026-11-10"))

    made.respond("make it Thursday")

    assert all(call["action"] == "BOOK" for call in backend.prepared)


def test_the_reference_to_an_appointment_already_held_is_what_tells_a_move_from_a_change() -> None:
    assert refers_to_existing("move my appointment on the 10th")
    assert refers_to_existing("cancel my booking")
    assert not refers_to_existing("change it to Thursday")
    assert not refers_to_existing("make it the 12th with Dr. Rao")


# --- delegation --------------------------------------------------------------------------------


def test_a_later_one_with_no_offers_to_compare_is_not_the_earliest_opening() -> None:
    backend = days_backend()
    made, _ = session(
        backend, book(clinic_name="Pine Clinic", start_date="2026-11-09", selection="LATER")
    )

    reply = made.respond("Book Pine Clinic on Monday, the later one")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_unused_imports_stay_used() -> None:
    assert Backend and MONDAY


def test_a_doctor_and_a_day_asked_alongside_another_question_keeps_both_answers() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "DOCTORS",
            "OPENINGS",
            clinic_name="Pine Clinic",
            doctor_name="Dr. Anita Rao",
            start_date="2026-11-09",
            end_date="2026-11-09",
        ),
    )

    reply = made.respond("Is Dr. Anita Rao at Pine Clinic, and is she free Monday?")

    assert reply.text.count("Openings at") == 1 and "Matching doctors" in reply.text


def test_a_doctor_asked_about_on_a_day_alone_is_answered_with_that_days_openings() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "DOCTORS",
            clinic_name="Pine Clinic",
            doctor_name="Dr. Anita Rao",
            start_date="2026-11-09",
            end_date="2026-11-09",
        ),
    )

    reply = made.respond("Is Dr. Anita Rao around on Monday at Pine Clinic?")

    assert "9:00 AM" in reply.text


def test_the_day_in_a_two_part_question_belongs_to_the_part_about_open_times() -> None:
    backend = days_backend()
    first = ask(
        "DOCTORS",
        clinic_name="Pine Clinic",
        start_date="2026-11-09",
        end_date="2026-11-09",
        more=[{"subject": "OPENINGS", "clinic_name": "Pine Clinic"}],
    )
    made, _ = session(backend, first)

    reply = made.respond("Who's at Pine Clinic, and what's open Monday?")

    assert "Matching doctors" in reply.text and "9:00 AM" in reply.text
    assert "Which date" not in reply.text


# --- LG-5F: a bare phrase is a question; a clear request is a booking ---------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the heart center, Monday",
        "Dr. Kim on Thursday",
        "Pine Clinic Thursday at 3 PM",
        "Whatever is soonest at Pine Clinic",
        "Any availability at Pine Clinic tomorrow?",
    ],
)
def test_a_bare_phrase_or_a_question_is_not_a_booking_even_if_a_reading_says_book(
    text: str,
) -> None:
    from app.evaluation.hybrid import operation_stated

    assert not operation_stated("BOOK", text)


@pytest.mark.parametrize(
    "text",
    [
        "Can you fit me in Thursday?",
        "I'd like an appointment at Pine Clinic",
        "Could I get an appointment with Dr. Reed on Monday?",
        "Is it possible to get an appointment on Friday?",
        "I'm looking for an appointment at Two Rivers",
        "Put me down for Thursday",
        "An appointment at Linden Court on Monday",
        "I'm hoping to see Dr. Kim",
    ],
)
def test_indirect_but_clear_requests_for_an_appointment_are_booking_requests(text: str) -> None:
    from app.evaluation.hybrid import operation_stated

    assert operation_stated("BOOK", text)


def test_a_clinic_and_day_after_the_clinic_list_is_a_lookup_not_a_review() -> None:
    """The EH10 shape: the reading proposes BOOK, a second reading would agree, and nothing in the
    patient's words, or in a yes to the assistant's question, asks for a booking."""
    backend = Backend()
    backend.slot("h1", "c-cedar", "d-cedar", MONDAY, 13)
    made, wire = session(
        backend,
        ask("CLINICS"),
        book(clinic_name="Heart Center", start_date="2026-11-09", end_date="2026-11-09"),
        {"asks_for": "BOOK"},
    )
    made.respond("Which clinics do you have?")

    reply = made.respond("the heart center, Monday")

    assert backend.prepared == [] and not reply.awaiting_confirmation
    assert "1:00 PM" in reply.text and made.snapshot()["review"] is None
    assert len(wire.payloads) == 2  # Two readings, one per message: no second look.


def test_a_doctor_and_day_after_a_doctor_list_is_a_lookup_not_a_review() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("DOCTORS", clinic_name="Pine Clinic"),
        book(doctor_name="Dr. Anita Rao", start_date="2026-11-09", end_date="2026-11-09"),
    )
    made.respond("Who works at Pine Clinic?")

    reply = made.respond("Dr. Anita Rao on Monday")

    assert backend.prepared == [] and not reply.awaiting_confirmation


def test_picking_a_listed_opening_or_saying_yes_still_starts_the_booking() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        book(time="9:30 AM"),
    )
    made.respond("What does Pine Clinic have on Monday?")

    reply = made.respond("Monday at 9:30")

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]
