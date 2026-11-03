"""LG-5G: empty searches explained (found in the first microphone trial).

A DEVELOPMENT set only, written after the failure it covers was seen: it is evidence that the real
model and the repaired code behave on these shapes, not a frozen holdout and not a reliability
claim. The trial: "a general physician" became `specialty: "General Practice"`; the clinic had no
such doctor; the assistant said "no openings" three times without saying why.

The world is `lg5d_scenarios`' (see its docstring). Two Rivers has one doctor, Dr. Wren
(Dermatology), open Thursday the 5th 10:00 and Friday the 6th 1:00 PM. Pine Ridge's Thursday
openings are in the afternoon (Dr. Rao 2:00 PM, Dr. Kim, cardiology, 3:00 PM) and it has nothing on
Friday; its next cardiology openings are Thursday the 5th, Monday the 9th and Thursday the 12th.
"""

from app.evaluation.lg4_scenarios import book
from app.evaluation.lg5d_scenarios import NONE, Spec, build_world

__all__ = ["DEVELOPMENT", "build_world"]

DEVELOPMENT: tuple[Spec, ...] = (
    # The trial itself: a specialty the clinic lacks, explained; then one it has; then a review.
    Spec(
        "GV01",
        "mismatch",
        (
            "I'd like an appointment with a general physician on Thursday.",
            "At {rivers}",
            "Dermatology",
        ),
        book("rivers_thu_1000"),
        contains={1: ("doesn't have a", "Dermatology")},
        avoids={0: ("is booked",), 1: ("is booked", "any openings")},
        quiet=(0, 1),
        retention=True,
    ),
    # Nothing at all that day for the specialty: the days named are real ones.
    Spec(
        "GV02",
        "no-openings",
        ("Any cardiology openings at {pine} on Friday?",),
        NONE,
        contains={0: ("Cardiology openings are on", "Thursday, November 5")},
        quiet=(0,),
    ),
    # Openings exist that day but not in the morning: a filter, not an empty day.
    Spec(
        "GV03",
        "filtered",
        ("Any openings at {pine} on Thursday morning?",),
        NONE,
        contains={0: ("morning openings", "other openings then")},
        quiet=(0,),
    ),
    # The second microphone trial: Whisper heard "gender physician", and a role was read as a name.
    Spec(
        "GV05",
        "mismatch",
        (
            "I want to book an appointment with a gender physician on Thursday.",
            "General Physician at {rivers}",
            "Dermatology",
        ),
        book("rivers_thu_1000"),
        contains={1: ("doesn't have a",)},
        avoids={1: ("couldn't find a doctor", "is booked")},
        quiet=(0, 1),
        retention=True,
    ),
    # Changing the day keeps the specialty and finds the review.
    Spec(
        "GV04",
        "change",
        ("Book a cardiologist at {pine} on Friday", "Thursday instead"),
        book("pine_thu_1500"),
        quiet=(0,),
        retention=True,
    ),
)
