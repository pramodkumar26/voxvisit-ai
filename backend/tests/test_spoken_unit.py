"""Spoken and written dates and times: what is understood, what is asked, what is left alone."""

from datetime import date

import pytest

from app.evaluation.spoken import (
    clock_mentions,
    date_mentions,
    minutes_of,
    minutes_of_mention,
    normalize_spoken,
    parts_of_day,
    resolve_spec,
    stated_clocks,
    stated_dates,
)

MONDAY = date(2026, 11, 2)


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("November seventeenth", "November 17"),
        ("the twenty-fourth of November", "the 24 of November"),
        ("the twenty fourth of november", "the 24 of november"),
        ("march first", "march 1"),
        ("the first of March", "the 1 of March"),
        ("December thirty-first", "December 31"),
        ("november seventeen", "november 17"),
        ("November seventeenth, twenty twenty-six", "November 17 2026"),
        ("November seventeenth two thousand and twenty seven", "November 17 2027"),
        ("quarter past two", "2:15"),
        ("a quarter after two", "2:15"),
        ("half past three", "3:30"),
        ("quarter to three", "2:45"),
        ("quarter to one", "12:45"),
        ("ten past two", "2:10"),
        ("twenty to four", "3:40"),
        ("five to ten", "9:55"),
        ("eleven thirty in the morning", "11:30 am"),
        ("two forty-five in the afternoon", "2:45 pm"),
        ("ten o'clock", "10:00"),
        ("nine oh five", "9:05"),
        ("ten a.m.", "10 am"),
        ("two in the afternoon", "2 pm"),
        ("seven in the evening", "7 pm"),
        ("after ten", "after 10"),
        ("the first available appointment after ten", "the first available appointment after 10"),
        ("before eleven please", "before 11 please"),
        ("after ten at Pine Clinic", "after 10 at Pine Clinic"),
        ("midday", "noon"),
        ("tomorrow afternoon", "tomorrow afternoon"),
    ],
)
def test_spoken_dates_and_times_become_digits(spoken, written) -> None:
    assert normalize_spoken(spoken) == written


@pytest.mark.parametrize(
    "text",
    [
        "Book at Two Rivers Clinic",
        "Book at Seven Oaks Clinic on Friday",
        "Twenty Pines Family Clinic",
        "I want the second one",
        "the first one please",
        "the third appointment",
        "Dr. Fifth is fine",
        "Ten Mile Medical",
        "at two Rivers",
        "after ten Rivers Clinic",
        "one of the doctors",
        "My PIN is one two three four",  # The credential screen runs before any rewriting.
    ],
)
def test_number_words_that_are_not_dates_or_times_are_left_alone(text) -> None:
    assert normalize_spoken(text) == text


def test_a_day_word_alone_is_a_date_only_from_the_sixth_on() -> None:
    assert normalize_spoken("make it the twenty-second") == "make it the 22nd"
    assert normalize_spoken("the sixth") == "the 6th"
    assert normalize_spoken("the fifth one") == "the fifth one"


# --- dates ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("November seventeenth", {"2026-11-17"}),
        ("the twenty-fourth of November", {"2026-11-24"}),
        ("on Nov 17th", {"2026-11-17"}),
        ("17 November 2027", {"2027-11-17"}),
        ("2026-11-17", {"2026-11-17"}),
        ("tomorrow", {"2026-11-03"}),
        ("the day after tomorrow", {"2026-11-04"}),
        ("today", {"2026-11-02"}),
        ("this Friday", {"2026-11-06"}),
        ("on Monday", {"2026-11-09"}),
        ("next Monday", {"2026-11-09"}),
        ("January fifth", {"2027-01-05"}),
        ("February thirtieth", set()),
        ("next Tuesday", set()),  # Two reasonable meanings: asked, never guessed.
    ],
)
def test_the_dates_a_message_settles(text, expected) -> None:
    assert stated_dates(normalize_spoken(text), MONDAY) == expected


def test_next_weekday_is_ambiguous_only_when_that_day_is_still_to_come_this_week() -> None:
    [monday_next_tuesday] = date_mentions("next Tuesday", MONDAY)
    assert monday_next_tuesday.iso == ""
    assert monday_next_tuesday.options == ("2026-11-03", "2026-11-10")
    wednesday = date(2026, 11, 4)
    [settled] = date_mentions("next Tuesday", wednesday)  # This week's Tuesday has passed.
    assert settled.iso == "2026-11-10" and settled.options == ()
    sunday = date(2026, 11, 8)
    [from_sunday] = date_mentions("next Tuesday", sunday)
    assert from_sunday.iso == "2026-11-10"


def test_a_date_that_depends_on_today_can_be_read_again_for_another_day() -> None:
    [tomorrow] = date_mentions("tomorrow", MONDAY)
    [november] = date_mentions("November 1", MONDAY)
    assert tomorrow.spec == "rel:tomorrow" and november.spec == "md:11:1"
    late_sunday = date(2026, 11, 1)  # In a clinic whose calendar is a day behind.
    assert resolve_spec(tomorrow.spec, late_sunday) == ("2026-11-02",)
    assert resolve_spec(november.spec, late_sunday) == ("2026-11-01",)  # Not 2027.
    assert resolve_spec(november.spec, MONDAY) == ("2027-11-01",)
    [written_in_full] = date_mentions("2026-11-17", MONDAY)
    assert written_in_full.spec == ""  # The same everywhere: never re-read.


def test_a_bare_day_number_means_that_day_of_the_month_under_discussion() -> None:
    [mention] = date_mentions("no, the 18th", MONDAY, anchor=date(2026, 11, 16))
    assert mention.iso == "2026-11-18"


def test_dates_are_reported_in_the_order_said() -> None:
    found = date_mentions("move my November 16 visit to November 20", MONDAY)
    assert [m.iso for m in found] == ["2026-11-16", "2026-11-20"]
    assert found[0].position < found[1].position


# --- times ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("10:30 AM", 630),
        ("2 pm", 840),
        ("09 AM", 540),
        ("12 PM", 720),
        ("12 AM", 0),
        ("noon", 720),
        ("14:30", 870),
        ("09:30", 570),
        ("eleven thirty in the morning", 690),
        ("two in the afternoon", 840),
        ("half past three in the afternoon", 930),
        ("quarter to nine in the evening", 20 * 60 + 45),
        ("13 pm", None),
        ("later", None),
        (None, None),
        ("quarter past two", None),  # No am or pm, and nothing that settles it.
        ("10:30", None),
        ("at 10", None),
        ("2:15", None),
    ],
)
def test_a_clock_time_is_settled_or_reported_ambiguous(text, minutes) -> None:
    assert minutes_of(text) == minutes


def test_a_part_of_the_day_settles_a_clock_without_am_or_pm() -> None:
    text = normalize_spoken("tomorrow afternoon at two thirty")
    [clock] = clock_mentions(text)
    assert minutes_of_mention(clock) is None
    assert minutes_of_mention(clock, parts_of_day(text)) == 14 * 60 + 30
    morning = normalize_spoken("next Tuesday morning at ten")
    [ten] = clock_mentions(morning)
    assert minutes_of_mention(ten, parts_of_day(morning)) == 600


def test_twelve_in_the_morning_stays_ambiguous_and_two_parts_settle_nothing() -> None:
    [twelve] = clock_mentions("12:30")
    assert minutes_of_mention(twelve, {"morning"}) is None
    [two] = clock_mentions("at 2")
    assert minutes_of_mention(two, {"morning", "afternoon"}) is None


def test_after_and_before_are_kinds_of_clock_mention() -> None:
    [after] = clock_mentions(normalize_spoken("the first available appointment after ten"))
    [before] = clock_mentions("anything before 11 am")
    [exact] = clock_mentions("book 11:30 am")
    assert (after.kind, after.hour, after.meridiem) == ("after", 10, "")
    assert (before.kind, before.hour, before.meridiem) == ("before", 11, "am")
    assert exact.kind == "at"


def test_a_time_is_not_taken_from_a_date_or_a_name() -> None:
    assert clock_mentions("November 17th") == []
    assert clock_mentions("at 17th Avenue") == []
    assert clock_mentions("Pine Clinic at 5th and Main") == []
    assert stated_clocks("book 2 pm and 3:30 pm") == {840, 930}
