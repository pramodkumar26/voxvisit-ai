"""LG-5A: the general mechanisms behind the conversational fixes, tested on their own with scripted
model readings. (How well the real model does is measured by the LG-5 evaluation, not here.)

Each test states a rule, not a phrase: a pick from offers needs agreeing evidence; an operation the
wording does not state needs an independent second reading; a day said relative to today is read on
the clinic's calendar; a question about what exists never prepares anything."""

from datetime import date

import pytest
from test_conversation_unit import NOW, Backend, RetiringBackend
from test_orchestrator import MONDAY, THURSDAY, Wire, ask, book, days_backend, sched, session

from app.evaluation.hybrid import _specialty_stated
from app.evaluation.offers import match_offer, numbered, options_from, resolve_offer
from app.evaluation.planner import PlannerDecision, ground, split_questions, week_in
from app.evaluation.spoken import date_mentions, without_dates

TODAY = date(2026, 11, 2)  # A Monday.


# --- offers -----------------------------------------------------------------------------------

CLINICS = ["Linden Court Heart Center", "Linden Court Family Clinic"]
TIMES = ["9:00 AM with Dr. Anita Rao", "9:30 AM with Dr. Anita Rao", "10:30 AM with Dr. Aiden Kim"]


def test_the_words_single_out_the_one_offer_they_describe() -> None:
    assert match_offer("the family one", CLINICS) == 2
    assert match_offer("the heart center please", CLINICS) == 1
    assert match_offer("Dr. Kim then", TIMES) == 3
    assert match_offer("the 9:30", TIMES) == 2


def test_what_every_offer_shares_or_what_fits_none_settles_nothing() -> None:
    assert match_offer("Linden Court", CLINICS) is None  # Both are Linden Court.
    assert match_offer("the one on Monday", CLINICS) is None
    assert match_offer("Dr. Rao", TIMES) is None  # Two of the three are Dr. Rao's.
    assert match_offer("anything", ["only one"]) is None


def test_nothing_but_the_words_and_the_offers_picks_an_offer() -> None:
    options = options_from(TIMES)
    assert resolve_offer("the last one", options).number == 3
    assert resolve_offer("the second one", options).number == 2
    assert resolve_offer("the ninth one", options).number is None  # There is no ninth.
    assert resolve_offer("anything", options).number is None
    assert resolve_offer("the heart one", options_from(CLINICS)).number == 1
    assert resolve_offer("anything", []).number is None


def test_offers_are_numbered_as_the_patient_heard_them() -> None:
    assert numbered(["A", "B"]) == "1) A; 2) B"


# --- dates and weeks as patients say them -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("mon morning", "2026-11-09"),
        ("thurs afternoon", "2026-11-05"),
        ("Wed", "2026-11-04"),
        ("fri please", "2026-11-06"),
        ("two days from now", "2026-11-04"),
        ("in 3 days", "2026-11-05"),
        ("in a week", "2026-11-09"),
        ("in two weeks", "2026-11-16"),
        ("10 days from today", "2026-11-12"),
    ],
)
def test_weekday_abbreviations_and_offsets_mean_what_they_say(text: str, iso: str) -> None:
    (found,) = date_mentions(text, TODAY)
    assert found.iso == iso


def test_words_that_only_look_like_days_are_not_days() -> None:
    assert date_mentions("this month, a monitor, wedding season", TODAY) == []


def test_numbers_left_after_dates_are_removed_are_not_days() -> None:
    assert "12" not in without_dates("November 12 at 3")
    assert "3" in without_dates("November 12 at 3")


@pytest.mark.parametrize(
    ("text", "week"),
    [
        ("what's open next week", "NEXT"),
        ("anything this week?", "THIS"),
        ("the week after this one", "NEXT"),
        ("later this week", "THIS"),
        ("anything on Friday", "NONE"),
    ],
)
def test_the_week_comes_from_the_patients_words(text: str, week: str) -> None:
    assert week_in(text) == week


def test_a_week_the_model_missed_is_still_the_week_asked_about() -> None:
    decision = PlannerDecision(route="LOOKUP", subject="OPENINGS", clinic_name="Pine Clinic")

    grounded = ground(decision, "What does Pine Clinic have next week?", TODAY)

    assert grounded.week == "NEXT"


def test_a_week_the_model_invented_is_dropped() -> None:
    decision = PlannerDecision(route="LOOKUP", subject="OPENINGS", clinic_name="Pine Clinic")
    decision = decision.model_copy(update={"week": "NEXT"})

    assert ground(decision, "What does Pine Clinic have on Friday?", TODAY).week == "NONE"


def test_a_phrase_with_two_meanings_is_looked_up_for_both() -> None:
    decision = PlannerDecision(route="LOOKUP", subject="OPENINGS", clinic_name="Pine Clinic")

    grounded = ground(decision, "Anything at Pine Clinic next Wednesday?", TODAY)

    assert (grounded.start_date, grounded.end_date) == (date(2026, 11, 4), date(2026, 11, 11))


@pytest.mark.parametrize(
    ("specialty", "text", "stated"),
    [
        ("Cardiology", "I'd like a cardiologist", True),
        ("Dermatology", "any dermatologists around?", True),
        ("Pediatrics", "I need a pediatrician", True),
        ("Family Medicine", "family medicine please", True),
        ("Cardiology", "I'd like a dermatologist", False),
        ("Cardiology", "my heart hurts", False),
        ("Oncology", "I'd like to book", False),
    ],
)
def test_a_specialty_is_stated_in_any_form_of_its_name(
    specialty: str, text: str, stated: bool
) -> None:
    assert _specialty_stated(specialty, text) is stated


# --- an operation the wording does not state ---------------------------------------------------


def cancel_backend() -> RetiringBackend:
    backend = RetiringBackend()
    backend.appointment("a1", "c-pine", "d-rao", date(2026, 11, 10), 10)
    return backend


CANCEL_READING = sched(
    operation="CANCEL", start_date="2026-11-10", end_date="2026-11-10", message_type="REQUEST"
)


def test_cancelling_in_other_words_is_still_cancelling_by_the_patients_own_language() -> None:
    backend = cancel_backend()
    made, wire = session(backend, CANCEL_READING)

    reply = made.respond("Scrap my appointment on the tenth")

    assert reply.awaiting_confirmation
    assert backend.prepared and backend.prepared[-1]["action"] == "CANCEL"
    assert len(wire.payloads) == 1  # The wording states it: no second look is needed.


def test_a_move_or_cancel_not_in_the_words_is_dropped_whatever_a_second_reading_says() -> None:
    backend = cancel_backend()
    made, wire = session(backend, CANCEL_READING, {"asks_for": "CANCEL"})

    reply = made.respond("Please get my Tuesday visit off the books")

    assert not reply.awaiting_confirmation and backend.prepared == []
    assert len(wire.payloads) == 1  # A second model reading is never consulted for this.


def test_a_plainly_worded_operation_needs_no_second_reading() -> None:
    backend = cancel_backend()
    made, wire = session(backend, CANCEL_READING)

    reply = made.respond("Cancel my appointment on the tenth")

    assert reply.awaiting_confirmation and len(wire.payloads) == 1


def test_a_correction_taken_for_a_withdrawal_keeps_the_request() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(message_type="ABANDON"),
        {"asks_for": "NONE"},
    )
    made.respond("Book Pine Clinic on Monday")

    made.respond("hmm, actually, could we try something else there?")

    assert made.snapshot()["request"]["clinic_name"] == "Pine Clinic"


def test_a_withdrawal_the_second_reading_agrees_with_drops_the_request() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(message_type="ABANDON"),
        {"asks_for": "WITHDRAW"},
    )
    made.respond("Book Pine Clinic on Monday")

    made.respond("I've decided against it")

    assert not made.snapshot()["request"]


def test_a_withdrawal_in_plain_words_needs_no_second_reading() -> None:
    backend = days_backend()
    made, wire = session(
        backend,
        book(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
        sched(message_type="ABANDON"),
    )
    made.respond("Book Pine Clinic on Monday")

    made.respond("never mind")

    assert len(wire.payloads) == 2 and not made.snapshot()["request"]


# --- a question about what exists is not a request to schedule ---------------------------------


def test_a_scheduling_label_with_no_operation_and_nothing_in_progress_is_a_lookup() -> None:
    backend = days_backend()
    made, _ = session(
        backend, sched(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09")
    )

    reply = made.respond("Any openings Monday at Pine Clinic?")

    assert "9:00 AM" in reply.text and not reply.awaiting_confirmation
    assert backend.prepared == []


def test_a_scheduling_label_with_nothing_to_look_up_goes_to_the_scheduler() -> None:
    backend = days_backend()
    made, _ = session(backend, sched())

    reply = made.respond("hmm")

    assert "book, move, or cancel" in reply.text


# --- picking from what was offered ------------------------------------------------------------


def morning_offers():
    backend = days_backend()
    made, wire = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        sched(option_number=1),  # The model's number is ignored: the words say "the last".
    )
    first = made.respond("Book Pine Clinic Monday morning")
    assert "Which time" in first.text
    return backend, made, wire


def test_the_last_one_is_the_last_time_offered() -> None:
    backend, made, wire = morning_offers()

    reply = made.respond("the last one")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s2"}]
    assert "Numbered choices" in wire.payloads[1]["messages"][0]["content"]


def test_a_number_outside_the_offers_picks_nothing() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        sched(option_number=9),
    )
    made.respond("Book Pine Clinic Monday morning")

    reply = made.respond("the ninth one")

    assert not reply.awaiting_confirmation and backend.prepared == []


def listed_openings(*readings, backend=None):
    backend = backend or days_backend()
    made, wire = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        *readings,
    )
    listed = made.respond("What does Pine Clinic have on November 12?")
    assert "2:00 PM" in listed.text and "3:00 PM" in listed.text
    return backend, made, wire


def test_the_later_one_after_a_listing_is_the_later_opening_listed() -> None:
    backend, made, _ = listed_openings(sched(operation="BOOK", option_number=1))

    reply = made.respond("the later one")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]


def test_a_pick_after_a_listing_prepares_for_review_and_confirms_nothing() -> None:
    backend, made, _ = listed_openings(sched(operation="BOOK", option_number=2))

    reply = made.respond("the earlier one")

    assert reply.awaiting_confirmation and "confirm" in reply.text.lower()
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s3"}]


def test_a_pick_the_model_cannot_tie_to_an_offer_prepares_nothing() -> None:
    backend, made, _ = listed_openings(sched(operation="BOOK"))

    reply = made.respond("that one")

    assert not reply.awaiting_confirmation and backend.prepared == []


def test_yes_to_the_one_opening_listed_books_that_opening() -> None:
    backend = days_backend()
    backend.slots = [s for s in backend.slots if s[0] != "s3"]  # Only the 3 PM is left.
    made, _ = session(
        backend,
        ask("OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-12", end_date="2026-11-12"),
        sched(operation="BOOK", message_type="AGREEMENT"),
    )
    made.respond("What does Pine Clinic have on November 12?")

    reply = made.respond("yes, book that")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s4"}]


def test_yes_to_two_listed_openings_does_not_choose_one() -> None:
    backend, made, _ = listed_openings(sched(operation="BOOK", message_type="AGREEMENT"))

    reply = made.respond("yes, book that")

    assert not reply.awaiting_confirmation and backend.prepared == []


def test_a_clinic_picked_from_a_list_of_clinics_is_that_clinic() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask("CLINICS"),
        sched(clinic_name="Pine Clinic", start_date="2026-11-09", end_date="2026-11-09"),
    )
    made.respond("Which clinics do you have?")

    reply = made.respond("the Pine one, Monday")

    assert "9:00 AM" in reply.text


def test_ambiguous_clinics_are_resolved_by_the_words_that_tell_them_apart() -> None:
    backend = Backend()
    backend.slot("h1", "c-cedar", "d-cedar", MONDAY, 13)
    made, _ = session(
        backend,
        book(clinic_name="Linden Court", start_date="2026-11-09", end_date="2026-11-09"),
        {"asks_for": "BOOK"},  # No act of booking is worded: an independent reading confirms it.
        sched(),
    )
    asked = made.respond("An appointment at Linden Court on Monday")
    assert "Which clinic" in asked.text

    reply = made.respond("the heart center")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "h1"}]


def test_the_models_number_is_ignored_when_the_words_decide() -> None:
    backend = Backend()
    backend.slot("h1", "c-cedar", "d-cedar", MONDAY, 13)
    made, _ = session(
        backend,
        book(clinic_name="Linden Court", start_date="2026-11-09", end_date="2026-11-09"),
        {"asks_for": "BOOK"},
        sched(option_number=2),  # The model says the second; the words say the heart center.
    )
    made.respond("An appointment at Linden Court on Monday")

    reply = made.respond("the heart center")

    assert reply.awaiting_confirmation  # The words decide; the model's number is ignored.
    assert backend.prepared == [{"action": "BOOK", "slot_id": "h1"}]


def test_a_number_said_inside_a_sentence_answers_a_time_question() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        ask("CLINICS"),
        book(clinic_name="Pine Clinic"),
    )
    made.respond("Book Pine Clinic Monday morning")
    made.respond("which clinics are there?")

    reply = made.respond("back to Pine Clinic, 11")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s2"}]


# --- finding what the patient did not say ------------------------------------------------------


def test_a_doctor_who_works_at_one_clinic_settles_the_clinic() -> None:
    backend = Backend()
    backend.slot("r1", "c-rivers", "d-rivers", THURSDAY, 10)
    made, _ = session(
        backend,
        book(doctor_name="Wren", start_date="2026-11-12", end_date="2026-11-12"),
        {"asks_for": "BOOK"},
    )

    reply = made.respond("Find me a time with Dr. Wren on November 12")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "r1"}]


def test_a_doctor_who_works_at_two_clinics_leaves_the_question_to_the_scheduler() -> None:
    backend = Backend()
    backend.doctors["d-rao2"] = {
        "doctor_id": "d-rao2",
        "display_name": "Dr. Anita Rao",
        "clinic": "c-rivers",
    }
    made, _ = session(
        backend, book(doctor_name="Rao", start_date="2026-11-12", end_date="2026-11-12")
    )

    reply = made.respond("Find me a time with Dr. Rao on November 12")

    assert not reply.awaiting_confirmation and backend.prepared == []


def test_whatever_is_soonest_looks_two_weeks_ahead_and_says_so() -> None:
    backend = days_backend()
    made, _ = session(backend, book(clinic_name="Pine Clinic", selection="EARLIEST"))

    reply = made.respond("Book whatever is soonest at Pine Clinic")

    assert reply.awaiting_confirmation and "earliest" in reply.text
    assert backend.prepared == [{"action": "BOOK", "slot_id": "s0"}]
    availability = [a for t, a, _ in backend.calls if t == "search_appointment_availability"]
    assert availability[-1]["start_date"] == "2026-11-02"
    assert availability[-1]["end_date"] == "2026-11-15"


# --- days on the clinic's own calendar --------------------------------------------------------


def kauri_backend() -> Backend:
    backend = Backend()
    # 04:00 on Tuesday 3 November in Auckland, 15:00 on Monday 2 November in UTC and Denver.
    backend.slot("k1", "c-kiwi", "d-kiwi", date(2026, 11, 3), 9, offset="+13:00")
    backend.slot("k2", "c-kiwi", "d-kiwi", date(2026, 11, 4), 10, offset="+13:00")
    return backend


def test_tomorrow_is_the_clinics_tomorrow_not_utcs() -> None:
    made, _ = session(
        kauri_backend(), ask("OPENINGS", clinic_name="Kauri Coast Clinic", start_date="2026-11-03")
    )

    reply = made.respond("What does Kauri Coast Clinic have tomorrow?")

    assert "Wednesday, November 4" in reply.text and "10:00 AM" in reply.text
    assert "Tuesday, November 3" not in reply.text


def test_next_week_is_the_clinics_next_week() -> None:
    backend = days_backend()
    made, _ = session(backend, ask("OPENINGS", clinic_name="Pine Clinic"))

    reply = made.respond("What does Pine Clinic have next week?")

    assert "Monday, November 9" in reply.text


def test_next_wednesday_said_on_a_monday_shows_both_wednesdays() -> None:
    backend = Backend()
    backend.slot("p1", "c-pine", "d-rao", date(2026, 11, 4), 9)
    backend.slot("p2", "c-pine", "d-rao", date(2026, 11, 11), 9)
    backend.slot("p3", "c-pine", "d-rao", date(2026, 11, 6), 9)  # A Friday: not asked about.
    made, _ = session(backend, ask("OPENINGS", clinic_name="Pine Clinic"))

    reply = made.respond("Anything at Pine Clinic next Wednesday?")

    assert "November 4" in reply.text and "November 11" in reply.text
    assert "November 6" not in reply.text


# --- more than one question --------------------------------------------------------------------


def test_a_second_question_about_another_clinic_is_answered_too() -> None:
    backend = days_backend()
    backend.slot("r1", "c-rivers", "d-rivers", THURSDAY, 10)
    made, _ = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            more=[
                {
                    "subject": "OPENINGS",
                    "clinic_name": "Two Rivers Clinic",
                    "start_date": "2026-11-12",
                    "end_date": "2026-11-12",
                }
            ],
        ),
    )

    reply = made.respond(
        "Anything at Pine Clinic on Monday, and what about Two Rivers Clinic on 12 Nov?"
    )

    assert "9:00 AM" in reply.text and "10:00 AM" in reply.text
    assert "Pine Clinic" in reply.text and "Two Rivers Clinic" in reply.text


def test_a_second_question_whose_name_is_not_in_the_words_is_not_answered() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            more=[{"subject": "OPENINGS", "clinic_name": "Atlantis Clinic"}],
        ),
    )

    reply = made.respond("Anything at Pine Clinic on Monday, and anything else?")

    assert "Atlantis" not in reply.text
    assert NOW  # The fixed clock of these tests.
    assert Wire  # The scripted wire is shared with the other orchestrator tests.


# --- the model never chooses among offers ---------------------------------------------------------


def test_a_number_only_the_model_proposes_picks_nothing_and_asks_no_second_question() -> None:
    backend, made, wire = listed_openings(sched(operation="BOOK", option_number=2))

    reply = made.respond("hmm, whichever")

    assert not reply.awaiting_confirmation and backend.prepared == []
    assert not any("has replied" in p["messages"][0]["content"] for p in wire.payloads)


def test_a_reference_that_fits_several_offers_picks_none() -> None:
    options = options_from(TIMES)
    for words in ("the morning one", "book Dr. Rao", "the later one", "the earlier one"):
        found = resolve_offer(words, options)
        assert found.number is None and len(found.candidates) >= 2, words


def test_a_delegation_to_the_earliest_resolves_only_to_one_opening() -> None:
    assert resolve_offer("the earliest available", options_from(TIMES)).number == 1
    tied = options_from(["9:00 AM with Dr. Anita Rao", "9:00 AM with Dr. Aiden Kim"])
    assert resolve_offer("the earliest available", tied).number is None


def test_a_description_is_matched_to_the_offers_own_data_not_to_a_guess() -> None:
    options = options_from(
        TIMES,
        [
            {"specialty": "Family Medicine"},
            {"specialty": "Family Medicine"},
            {"doctor": "Dr. Aiden Kim", "specialty": "Cardiology"},
        ],
    )
    assert resolve_offer("the heart doctor", options).number == 3
    assert resolve_offer("the skin doctor", options).number is None


def test_the_closed_question_is_not_asked_of_a_contrast_a_question_or_a_withdrawal() -> None:
    for words in ("not the later one", "which is the later one?", "never mind"):
        _backend, made, wire = listed_openings(sched(operation="BOOK"), {"asks_for": "NONE"})
        made.respond(words)
        closed = [p for p in wire.payloads if p["format"].get("title") == "Pick"]
        assert closed == [], words


# --- short replies settled without the model ----------------------------------------------------


def quick_morning():
    backend = days_backend()
    made, wire = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time_preference="MORNING",
        ),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        quick=True,
    )
    assert "Which time" in made.respond("Book Pine Clinic Monday morning").text
    return backend, made, wire


def test_a_lone_time_answering_a_time_question_needs_no_model() -> None:
    backend, made, wire = quick_morning()

    reply = made.respond("9:30")

    assert reply.awaiting_confirmation and backend.prepared == [{"action": "BOOK", "slot_id": "s1"}]
    assert len(wire.payloads) == 1 and made.quick_reads == 1
    assert made.trace[-1]["model_skipped"] is True


def test_a_time_with_a_contrast_in_it_goes_to_the_model() -> None:
    _backend, made, wire = quick_morning()

    made.respond("not 9:30")

    assert len(wire.payloads) == 2 and made.quick_reads == 0


def test_a_plain_yes_with_a_review_waiting_needs_no_model() -> None:
    backend, made, wire = quick_morning()
    made.respond("9:30")

    reply = made.respond("yes")

    assert "confirm" in reply.text.lower() and len(wire.payloads) == 1
    assert backend.discarded == []  # Agreement changes nothing.


def test_a_plain_yes_with_nothing_to_continue_goes_to_the_model() -> None:
    made, wire = session(days_backend(), sched(message_type="AGREEMENT"), quick=True)

    made.respond("yes")

    assert len(wire.payloads) == 1 and made.quick_reads == 0


def test_a_plain_never_mind_with_a_request_in_progress_needs_no_model() -> None:
    _backend, made, wire = quick_morning()

    made.respond("never mind")

    assert len(wire.payloads) == 1 and not made.snapshot()["request"]


def test_a_long_reply_is_always_read_by_the_model() -> None:
    _backend, made, wire = quick_morning()

    made.respond("9:30 sounds good to me, thank you so much for your help today")

    assert len(wire.payloads) == 2


# --- several questions in one message, split by one more reading --------------------------------


def two_clinics_backend():
    backend = days_backend()
    backend.slot("r1", "c-rivers", "d-rivers", THURSDAY, 10)
    return backend


COMPOUND = ask("OPENINGS", "OPENINGS", clinic_name="Pine Clinic")


def test_a_message_asking_about_two_places_is_split_and_each_is_answered_on_its_own_day() -> None:
    made, _ = session(
        two_clinics_backend(),
        COMPOUND,
        {
            "questions": [
                {"subject": "OPENINGS", "clinic_name": "Pine Clinic"},
                {"subject": "OPENINGS", "clinic_name": "Two Rivers Clinic"},
            ]
        },
    )

    reply = made.respond(
        "Anything at Pine Clinic on Monday, and what about Two Rivers Clinic on 12 November?"
    )

    assert "Monday, November 9" in reply.text and "9:00 AM" in reply.text
    assert "Thursday, November 12" in reply.text and "10:00 AM" in reply.text
    assert "2:00 PM" not in reply.text  # Pine's Thursday is not Two Rivers' day.


def test_two_doctors_of_one_clinic_are_each_looked_up_at_that_clinic() -> None:
    backend = days_backend()
    backend.slot("k1", "c-pine", "d-kim", MONDAY, 10, 30)
    made, _ = session(
        backend,
        ask("OPENINGS", "OPENINGS", clinic_name="Pine Clinic", doctor_name="Kim"),
        {
            "questions": [
                {"subject": "OPENINGS", "clinic_name": "Pine Clinic", "doctor_name": "Dr. Kim"},
                {"subject": "OPENINGS", "doctor_name": "Dr. Rao"},
            ]
        },
    )

    reply = made.respond("Tell me about Dr. Kim's Monday and Dr. Rao's Monday at Pine Clinic")

    assert "10:30 AM" in reply.text and "9:00 AM" in reply.text


def test_one_clinic_asked_two_ways_is_not_split() -> None:
    backend = days_backend()
    made, wire = session(
        backend,
        ask("DOCTORS", "OPENINGS", clinic_name="Pine Clinic", start_date="2026-11-09"),
        {
            "questions": [
                {"subject": "DOCTORS", "clinic_name": "Pine Clinic"},
                {"subject": "OPENINGS", "clinic_name": "Pine Clinic"},
            ]
        },
    )

    reply = made.respond("Who's at Pine Clinic, and what's open Monday?")

    assert "Dr. Anita Rao" in reply.text and "9:00 AM" in reply.text
    assert len(wire.payloads) == 2  # The reading, and the look at whether it was several.


def test_a_split_whose_names_are_not_in_the_words_is_ignored() -> None:
    made, _ = session(
        two_clinics_backend(),
        COMPOUND,
        {
            "questions": [
                {"subject": "OPENINGS", "clinic_name": "Pine Clinic"},
                {"subject": "OPENINGS", "clinic_name": "Atlantis Clinic"},
            ]
        },
    )

    reply = made.respond("Anything at Pine Clinic on Monday, and anything else?")

    assert "Atlantis" not in reply.text


# --- what a checkpoint keeps of a review --------------------------------------------------------


def test_a_saved_review_keeps_what_the_report_back_needs_and_no_identifiers() -> None:
    backend = days_backend()
    made, _ = session(
        backend,
        book(
            clinic_name="Pine Clinic",
            start_date="2026-11-09",
            end_date="2026-11-09",
            time="9:30 AM",
            selection="SPECIFIC_TIME",
        ),
    )
    made.respond("Book Pine Clinic on Monday at 9:30 AM")

    saved = made.snapshot()["review"]["chosen"]

    assert set(saved) <= {"starts_at_local", "doctor", "clinic", "slot_id", "appointment_id"}
    assert set(saved["doctor"]) == {"display_name"} and set(saved["clinic"]) == {"name"}
    assert "doctor_id" not in str(saved) and "clinic_id" not in str(saved)


# --- names as far as the patient said them ----------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "said", "kept"),
    [
        ("Two Rivers Clinic", "the Two Rivers one on Thursday", "Two Rivers"),
        ("Pine Clinic Medical Center", "Pine please", "Pine"),
        ("Linden Court Heart Center", "heart center", "heart center"),
        ("Linden Court Heart Center", "the center please", None),  # Only generic words were said.
        ("Atlantis Clinic", "the Pine one", None),
        ("Dr. Aiden Kim", "with Kim", "Kim"),
    ],
)
def test_a_name_the_model_made_longer_is_kept_as_far_as_it_was_said(
    model_name: str, said: str, kept: str | None
) -> None:
    from app.evaluation.hybrid import _stated_part

    assert _stated_part(model_name, said) == kept


def test_the_clinic_a_doctor_works_at_picks_it_from_the_clinics_offered() -> None:
    backend = Backend()
    backend.slot("f1", "c-cedar2", "d-moss", MONDAY, 14)
    backend.doctors["d-moss"] = {
        "doctor_id": "d-moss",
        "display_name": "Dr. Mara Moss",
        "clinic": "c-cedar2",
    }
    made, _ = session(
        backend,
        book(clinic_name="Linden Court", start_date="2026-11-09", end_date="2026-11-09"),
        book(doctor_name="Moss"),
    )
    assert "Which clinic" in made.respond("An appointment at Linden Court on Monday").text

    reply = made.respond("the one with Dr. Moss")

    assert reply.awaiting_confirmation
    assert backend.prepared == [{"action": "BOOK", "slot_id": "f1"}]


# --- the prompt, laid out so that the part that never changes comes first -----------------------


def test_the_prefix_layout_puts_everything_that_changes_last() -> None:
    from app.evaluation.turn_reader import system_prompt

    first = system_prompt(
        today=date(2026, 11, 2), context="Waiting for a time.", layout="prefix", compact=False
    )
    second = system_prompt(
        today=date(2026, 11, 3), context="Waiting for a clinic.", layout="prefix", compact=False
    )

    shared = 0
    for a, b in zip(first, second, strict=False):
        if a != b:
            break
        shared += 1
    assert shared / len(first) > 0.97  # Only the closing lines differ from day to day.
    assert "Today is 2026-11-02" in first and "Waiting for a time." in first


def test_both_layouts_carry_the_same_instructions() -> None:
    from app.evaluation.turn_reader import system_prompt

    a = system_prompt(today=TODAY, context="x", layout="legacy", compact=False)
    b = system_prompt(today=TODAY, context="x", layout="prefix", compact=False)

    def lines(text: str) -> set[str]:
        return {
            line.replace(" Today is 2026-11-02 (Monday).", "")
            for line in text.split("\n")
            if line.strip() and not line.startswith("Today is")
        }

    assert lines(a) == lines(b)


def test_the_compact_wording_asks_only_for_the_fields_that_apply() -> None:
    from app.evaluation.turn_reader import system_prompt

    assert "do not write null" in system_prompt(
        today=TODAY, context="", layout="prefix", compact=True
    )
    assert "do not write null" not in system_prompt(
        today=TODAY, context="", layout="prefix", compact=False
    )


def test_warming_up_loads_the_model_without_reading_a_patient_message() -> None:
    from app.evaluation.llm import LlmConfig
    from app.evaluation.turn_reader import TurnReader

    wire = Wire({"message": {"content": "{}"}})
    reader = TurnReader(LlmConfig(keep_alive="30m"), transport=wire)

    reader.warm_up(TODAY)

    (payload,) = wire.payloads
    assert payload["keep_alive"] == "30m" and payload["options"]["num_predict"] == 1
    assert reader.usage.calls == []  # A warm-up is not a model call of any conversation.


def test_a_split_keeps_the_subjects_of_the_first_reading() -> None:
    made, _ = session(
        two_clinics_backend(),
        COMPOUND,
        {
            "questions": [
                {"subject": "CLINICS", "clinic_name": "Pine Clinic"},
                {"subject": "CLINICS", "clinic_name": "Two Rivers Clinic"},
            ]
        },
    )

    reply = made.respond(
        "Anything at Pine Clinic on Monday, and what about Two Rivers Clinic on 12 November?"
    )

    assert "9:00 AM" in reply.text and "10:00 AM" in reply.text
    assert "clinics are available" not in reply.text


def test_a_day_belongs_to_the_place_it_follows_even_when_the_name_is_long() -> None:
    questions = [
        PlannerDecision(
            route="LOOKUP", subject="OPENINGS", clinic_name="Pine Ridge Clinic Willow Square"
        ),
        PlannerDecision(
            route="LOOKUP", subject="OPENINGS", clinic_name="Two Rivers Clinic Willow Square"
        ),
    ]

    pine, rivers = split_questions(
        questions,
        "Anything at Pine Ridge Clinic Willow Square Monday, and what about "
        "Two Rivers Clinic Willow Square on Friday?",
        TODAY,
    )

    assert pine.start_date == date(2026, 11, 9) and rivers.start_date == date(2026, 11, 6)


def test_the_tail_of_a_clinics_own_name_is_not_a_doctor_or_a_specialty() -> None:
    backend = Backend()
    backend.clinics["c-pine"]["name"] = "Pine Clinic Juniper Ridge"
    backend.slot("p1", "c-pine", "d-rao", MONDAY, 9)
    made, _ = session(
        backend,
        ask(
            "OPENINGS",
            clinic_name="Pine Clinic",
            specialty="Juniper Ridge",
            start_date="2026-11-09",
            end_date="2026-11-09",
        ),
    )

    reply = made.respond("What does Pine Clinic Juniper Ridge have on Monday?")

    assert "9:00 AM" in reply.text and "doctor" not in reply.text


def test_the_lean_reading_asks_for_no_dates_and_still_reads_them_from_the_words() -> None:
    from app.evaluation.llm import LlmConfig
    from app.evaluation.turn_reader import TurnReader, reading_schema, system_prompt

    schema = reading_schema(lean=True)
    assert not {"start_date", "end_date", "new_date"} & set(schema["properties"])
    assert "start_date" in reading_schema(lean=False)["properties"]
    lean = system_prompt(today=TODAY, context="", layout="prefix", compact=False, lean=True)
    assert "start_date" not in lean and "end_date" not in lean

    wire = Wire(ask("OPENINGS", clinic_name="Pine Clinic"))
    reader = TurnReader(LlmConfig(), transport=wire, layout="prefix", lean=True)
    read = reader.read(
        "Anything at Pine Clinic next Wednesday?", context="", today=TODAY, anchor=None
    )

    assert read is not None and (read.decision.start_date, read.decision.end_date) == (
        date(2026, 11, 4),
        date(2026, 11, 11),
    )
    assert "start_date" not in wire.payloads[0]["format"]["properties"]
