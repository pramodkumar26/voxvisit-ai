"""LG-5D: fresh synthetic conversations for three conversational-correctness mechanisms.

1. Corrections after a review is prepared (clinic, doctor, date, time, operation, withdrawal).
2. Lookup-to-booking continuity (a question about what exists turns into a request).
3. No review from a pure information request (availability is not a booking).

The LG-4 and LG-5 sets are regression data only: their holdouts were spent when they were run.
This module adds:

- DEVELOPMENT: used to find and fix general mechanisms (up to three improvement cycles), with
  adversarial variations of all three problems.
- HOLDOUT: written and fingerprinted before any LG-5D implementation change, run once on the
  locked implementation, and never edited or tuned against afterwards. `HOLDOUT_FINGERPRINT` is the
  SHA-256 of its canonical form; a test fails if the set changes.

One holdout case (LH16) was replaced before any holdout run, after the development baseline
showed that the world also holds the seeded clinics (two dermatologists, two doctors named
Okafor): its first draft assumed one dermatologist, so its expected outcome was wrong about the
world, not about any implementation. The replacement was fixed from the same world facts, and the
set was fingerprinted again.

Expected outcomes were fixed from the patient's intent and the world below, not from what any
implementation did: each says which slot a reasonable patient who read the assistant's replies would
end up with. Where a patient's words honestly allow two readings the conversation avoids the
ambiguity (or expects the assistant to ask rather than act).

The world is the LG-5 world (see `lg5_scenarios`). Today is Monday 2 November 2026, 08:00 in
America/Denver. What is open (slot keys in the `World`):

  Pine Ridge ({pine}): Dr. Rao (Family Medicine) Mon 9th 9:00 and 9:30, Thu 5th 2:00 PM, Thu 12th
      2:00 PM; Dr. Kim (Cardiology) Mon 9th 10:30 (11:00 is held by someone else), Thu 5th 3:00 PM,
      Thu 12th 3:00 PM. The patient's own appointment is Tue 10th 10:00 AM with Dr. Rao.
  Two Rivers ({rivers}): Dr. Wren (Dermatology) Thu 5th 10:00 AM, Fri 6th 1:00 PM.
  Linden Court Heart Center ({heart}): Dr. Reed (Cardiology) Mon 9th 1:00 PM.
  Linden Court Family Clinic ({family}): Dr. Moss (Family Medicine) Mon 9th 2:00 PM.
  The seeded clinics (Riverbend, Lakeside, Harborview Medical Associates) also exist, with doctors
      such as Dr. Marcus Ellery (Cardiology), Dr. Sofia Lindqvist (Dermatology) and Dr. Tomas
      Okafor.
  Harbor View ({harbor}): Dr. Ortiz (Internal Medicine) tomorrow Tue 11:00 AM and 3:00 PM, Wed 4th
      9:00 AM, Wed 11th 9:00 AM; Dr. Okafor (Pediatrics) Wed 4th 4:00 PM, Fri 13th 10:00 AM.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

from app.evaluation import lg5_scenarios as lg5
from app.evaluation.lg4_scenarios import Expect, book

NONE = Expect()
# A review was prepared, then dropped: nothing left, nothing changed.
WITHDRAWN = Expect("withdrawn")
MOVE_TO = "reschedule"

build_world = lg5.build_world


@dataclass(frozen=True)
class Spec(lg5.Spec):
    # Turns (0-based) after which no review may have been prepared yet: they only asked to know.
    quiet: tuple[int, ...] = ()


def move(slot: str) -> Expect:
    return Expect(MOVE_TO, slot, appointment="mine")


# --- development set --------------------------------------------------------------------------

DEVELOPMENT: tuple[Spec, ...] = (
    # Corrections after a review: the doctor, by name or by description.
    Spec(
        "LD01",
        "correction",
        ("Book Dr. Rao at {pine} for Thursday", "actually the heart doctor"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "LD02",
        "correction",
        ("Book {pine} Thursday with Dr. Kim", "I'd rather see the family doctor"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "LD03",
        "correction",
        ("Book Dr. Kim at {pine} Thursday", "swap that to Dr. Rao"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "LD04",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "actually, Dr. Kim", "no wait, back to Dr. Rao"),
        book("pine_thu_1400"),
        retention=True,
    ),
    # The date.
    Spec(
        "LD05",
        "correction",
        ("Book {rivers} on Friday", "hold on, Thursday"),
        book("rivers_thu_1000"),
        retention=True,
    ),
    Spec(
        "LD06",
        "correction",
        ("Book Dr. Kim at {pine} on Thursday", "can you make it the 12th instead"),
        book("pine_nthu_1500"),
        retention=True,
    ),
    # The time.
    Spec(
        "LD07",
        "correction",
        ("Book Dr. Rao at {pine} on Monday at 9 AM", "make it 9:30"),
        book("pine_mon_0930"),
        retention=True,
    ),
    Spec(
        "LD08",
        "correction",
        ("Book {harbor} tomorrow at 3 PM", "scratch that, tomorrow at 11"),
        book("harbor_tue_1100"),
        retention=True,
    ),
    Spec(
        "LD09",
        "correction",
        ("Book {harbor} tomorrow at 11", "nope, 3 in the afternoon"),
        book("harbor_tue_1500"),
        retention=True,
    ),
    # The clinic.
    Spec(
        "LD10",
        "correction",
        ("Book {family} on Monday", "actually {harbor} tomorrow at 3"),
        book("harbor_tue_1500"),
        retention=True,
    ),
    # The operation.
    Spec(
        "LD11",
        "correction",
        ("Cancel my appointment on the 10th", "wait, I'd rather move it to the 12th at 2"),
        move("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "LD12",
        "correction",
        ("Move my appointment on the 10th to the 12th at 3 PM", "no, cancel it instead"),
        Expect("cancel", appointment="mine"),
        retention=True,
    ),
    # Withdrawing, and withdrawing with a replacement.
    Spec(
        "LD13",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "never mind"),
        WITHDRAWN,
        contains={1: ("nothing was changed",)},
        retention=True,
    ),
    Spec(
        "LD14",
        "correction",
        ("Book {family} Monday", "forget it, the heart center instead"),
        book("heart_mon_1300"),
        retention=True,
    ),
    # A superseded review must never be confirmable, and the new one must be.
    Spec(
        "LD15",
        "correction",
        ("Book {rivers} on Friday", "actually Thursday"),
        book("rivers_thu_1000", commit=True),
        retention=True,
    ),
    # Not a correction: the review stands.
    Spec(
        "LD16",
        "correction",
        ("Book {rivers} Thursday", "yes"),
        book("rivers_thu_1000"),
        contains={1: ("Confirm",)},
    ),
    Spec(
        "LD17",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "thanks!"),
        book("pine_thu_1400"),
    ),
    # Lookup, then a request that rests on what was found.
    Spec(
        "LD18",
        "lookup-to-booking",
        ("Is Dr. Moss available on Monday?", "great, book that"),
        book("family_mon_1400"),
        contains={0: ("2:00 PM",)},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD19",
        "lookup-to-booking",
        ("Do any cardiologists have time on Thursday?", "Dr. Kim at {pine} then"),
        book("pine_thu_1500"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD20",
        "lookup-to-booking",
        ("What does {rivers} have on Friday?", "I'll take it"),
        book("rivers_fri_1300"),
        contains={0: ("1:00 PM",)},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD21",
        "lookup-to-booking",
        ("Anything at {pine} on Monday morning?", "the 10:30"),
        book("pine_mon_1030"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD22",
        "lookup-to-booking",
        ("Is there a dermatologist at {rivers}?", "ok, Thursday with them"),
        book("rivers_thu_1000"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD23",
        "lookup-to-booking",
        ("What's open at {harbor} on Wednesday?", "the afternoon one"),
        book("harbor_wed_1600"),
        contains={0: ("9:00 AM", "4:00 PM")},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD24",
        "lookup-to-booking",
        ("Which doctors are at {pine}?", "Dr. Kim, Thursday"),
        book("pine_thu_1500"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD25",
        "lookup-to-booking",
        ("Who works at {harbor}?", "Dr. Ortiz tomorrow at 3"),
        book("harbor_tue_1500"),
        quiet=(0,),
        retention=True,
    ),
    # Clarify, never guess.
    Spec(
        "LD26",
        "lookup-to-booking",
        ("What's free at {pine} on Thursday?", "book the 9 AM"),
        NONE,
        contains={0: ("2:00 PM", "3:00 PM")},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "LD27",
        "lookup-to-booking",
        ("What's available at {pine} on Monday?", "book one of the morning ones"),
        NONE,
        quiet=(0, 1),
        avoids={1: ("I prepared",)},
        retention=True,
    ),
    Spec(
        "LD28",
        "lookup-to-booking",
        ("Is there anything with {heart} on Monday?", "book it"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    # Information only: nothing may be prepared.
    Spec(
        "LD29",
        "pure-lookup",
        ("Do you happen to have anything at {pine} on Monday afternoon?",),
        NONE,
        quiet=(0,),
    ),
    Spec(
        "LD30",
        "pure-lookup",
        ("I'm just curious whether {rivers} has time on Thursday",),
        NONE,
        contains={0: ("10:00 AM",)},
        quiet=(0,),
    ),
    Spec(
        "LD31",
        "pure-lookup",
        ("When is Dr. Reed free on Monday?",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "LD32",
        "pure-lookup",
        ("Could you check what {harbor} has tomorrow afternoon?",),
        NONE,
        contains={0: ("3:00 PM",)},
        avoids={0: ("11:00 AM",)},
        quiet=(0,),
    ),
    Spec(
        "LD33",
        "pure-lookup",
        ("I may come in on Thursday, what is open at {pine}?",),
        NONE,
        contains={0: ("2:00 PM", "3:00 PM")},
        quiet=(0,),
    ),
    Spec(
        "LD34",
        "pure-lookup",
        ("Which doctors work at {harbor}?", "and is Dr. Ortiz free on Wednesday?"),
        NONE,
        contains={0: ("Dr. Lena Ortiz",), 1: ("9:00 AM",)},
        quiet=(0, 1),
    ),
    Spec(
        "LD35",
        "pure-lookup",
        ("Is there a cardiologist at {pine} on Thursday?", "what time?"),
        NONE,
        contains={0: ("Aiden Kim",)},
        quiet=(0, 1),
    ),
    Spec(
        "LD36",
        "pure-lookup",
        ("Anything at {rivers} on Friday?", "and Thursday?"),
        NONE,
        contains={0: ("1:00 PM",), 1: ("10:00 AM",)},
        quiet=(0, 1),
    ),
    # Anonymous and interrupted.
    Spec(
        "LD37",
        "mixed",
        ("Anything at {harbor} tomorrow?", "ok book the 3 PM"),
        NONE,
        anonymous=True,
        contains={0: ("11:00 AM", "3:00 PM")},
        contains_any={1: ("sign in", "Sign in", "signed in")},
    ),
    Spec(
        "LD38",
        "mixed",
        ("Book {pine} Monday at 9:30 AM", "is Dr. Kim in on Thursday?", "ok that's fine"),
        book("pine_mon_0930"),
        contains={1: ("3:00 PM",)},
        retention=True,
    ),
    # Added after the first cycles, as adversarial variations (other wording, other shapes).
    Spec(
        "LD39",
        "correction",
        ("Book {pine} Thursday with Dr. Kim", "no, the other doctor"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "LD40",
        "correction",
        ("Book Dr. Rao at {pine} on Thursday", "can we do the 12th instead?"),
        book("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "LD41",
        "correction",
        ("Book Dr. Kim at {pine} on Monday", "what about Thursday instead?"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "LD42",
        "correction",
        (
            "Book {rivers} Thursday",
            "actually hold off, I'd like to cancel my appointment on the 10th",
        ),
        Expect("cancel", appointment="mine"),
        retention=True,
    ),
    Spec(
        "LD43",
        "correction",
        (
            "Book Dr. Rao at {pine} for Thursday",
            "wait, is Dr. Kim free that day?",
            "yes, switch to Dr. Kim",
        ),
        book("pine_thu_1500"),
        contains={1: ("3:00 PM",)},
        retention=True,
    ),
    Spec(
        "LD44",
        "lookup-to-booking",
        ("What time is Dr. Reed in on Monday?", "I'll take it"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD45",
        "lookup-to-booking",
        ("Do you have anything with cardiology at {pine} on Thursday?", "ok, book it"),
        book("pine_thu_1500"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD46",
        "lookup-to-booking",
        ("Who's free at {rivers} on Thursday and Friday?", "Friday"),
        book("rivers_fri_1300"),
        contains={0: ("10:00 AM", "1:00 PM")},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD47",
        "lookup-to-booking",
        ("tell me what's open at {pine} on Monday", "I'll go with the 10:30"),
        book("pine_mon_1030"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LD48",
        "pure-lookup",
        ("I'm trying to see if there's any availability for Thursday at {pine}",),
        NONE,
        contains={0: ("2:00 PM", "3:00 PM")},
        quiet=(0,),
    ),
    Spec(
        "LD49",
        "pure-lookup",
        ("Can you check whether Dr. Okafor has time on Wednesday?",),
        NONE,
        quiet=(0,),
    ),
    Spec(
        "LD50",
        "pure-lookup",
        ("Do you have any appointments open at {harbor} tomorrow?", "and the day after?"),
        NONE,
        contains={0: ("11:00 AM", "3:00 PM")},
        quiet=(0, 1),
    ),
    Spec(
        "LD51",
        "pure-lookup",
        ("Is there an appointment available with Dr. Kim on Thursday?",),
        NONE,
        contains={0: ("3:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "LD52",
        "pure-lookup",
        ("I need to know whether {rivers} has anything on Friday",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "LD53",
        "mixed",
        ("Book {rivers} Thursday", "please confirm it for me"),
        book("rivers_thu_1000"),
        contains={1: ("Confirm",)},
    ),
)

# --- holdout set -----------------------------------------------------------------------------
# Written and frozen before any LG-5D implementation change. Never edit.

HOLDOUT: tuple[Spec, ...] = (
    # Corrections after a review is prepared.
    Spec(
        "LH01",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "make that Dr. Kim"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "LH02",
        "correction",
        ("Book {rivers} Friday", "no, Thursday instead"),
        book("rivers_thu_1000"),
        retention=True,
    ),
    Spec(
        "LH03",
        "correction",
        ("I'd like Dr. Rao at {pine} on Monday at 9:30 AM", "sorry, 9 AM"),
        book("pine_mon_0900"),
        retention=True,
    ),
    Spec(
        "LH04",
        "correction",
        (
            "Book Dr. Kim at {pine} on Monday",
            "can I see a family doctor instead",
            "the earlier one",
        ),
        book("pine_mon_0900"),
        retention=True,
    ),
    Spec(
        "LH05",
        "correction",
        ("Cancel my appointment on the 10th", "actually don't cancel, move it to the 12th at 3"),
        move("pine_nthu_1500"),
        retention=True,
    ),
    Spec(
        "LH06",
        "correction",
        ("Book {heart} on Monday", "actually {harbor} tomorrow at 11"),
        book("harbor_tue_1100"),
        retention=True,
    ),
    Spec(
        "LH07",
        "correction",
        ("Book {family} Monday", "forget that, the heart center instead"),
        book("heart_mon_1300"),
        retention=True,
    ),
    Spec(
        "LH08",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "never mind, I'll call them"),
        WITHDRAWN,
        contains={1: ("nothing was changed",)},
        retention=True,
    ),
    Spec(
        "LH09",
        "correction",
        ("Book {harbor} tomorrow afternoon", "make it the day after tomorrow instead"),
        book("harbor_wed_1600"),
        retention=True,
    ),
    Spec(
        "LH10",
        "correction",
        ("Book Dr. Rao at {pine} on Thursday", "make it the 12th with Dr. Kim"),
        book("pine_nthu_1500"),
        retention=True,
    ),
    Spec(
        "LH11",
        "correction",
        ("Book Dr. Kim at {pine} Thursday", "sorry I meant Monday"),
        book("pine_mon_1030", commit=True),
        retention=True,
    ),
    Spec(
        "LH12",
        "correction",
        ("Move my appointment on the 10th to the 5th at 2 PM", "no, the 12th instead"),
        move("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "LH13",
        "correction",
        ("Book Dr. Rao at {pine} Thursday", "actually Dr. Kim", "no wait, Dr. Rao again"),
        book("pine_thu_1400"),
        retention=True,
    ),
    # Lookup, then a request that rests on what was found.
    Spec(
        "LH14",
        "lookup-to-booking",
        ("Does Dr. Reed have anything on Monday?", "book it"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH15",
        "lookup-to-booking",
        ("What's open at {pine} Thursday afternoon?", "the one with the cardiologist"),
        book("pine_thu_1500"),
        contains={0: ("2:00 PM", "3:00 PM")},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH16",
        "lookup-to-booking",
        ("Is there a pediatrician at {harbor}?", "ok, Wednesday"),
        book("harbor_wed_1600"),
        contains={0: ("Okafor",)},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH17",
        "lookup-to-booking",
        (
            "Which clinics do you have?",
            "Linden Court Heart Center then",
            "Monday please",
        ),
        book("heart_mon_1300"),
        contains={0: ("These clinics",)},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH18",
        "lookup-to-booking",
        ("Is there anything at {rivers} on Friday?", "yes please"),
        book("rivers_fri_1300"),
        contains={0: ("1:00 PM",)},
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH19",
        "lookup-to-booking",
        ("Who are the doctors at {harbor}?", "Dr. Okafor on Wednesday"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "LH20",
        "lookup-to-booking",
        ("What's available at {pine} on Thursday?", "book me the 11 AM"),
        NONE,
        contains={0: ("2:00 PM", "3:00 PM")},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "LH21",
        "lookup-to-booking",
        ("What's open at {pine} on Monday?", "book the morning one"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    # Information only: nothing may be prepared.
    Spec(
        "LH22",
        "pure-lookup",
        ("Do you have anything open at {harbor} tomorrow?",),
        NONE,
        contains={0: ("11:00 AM", "3:00 PM")},
        quiet=(0,),
    ),
    Spec(
        "LH23",
        "pure-lookup",
        ("I was wondering whether Dr. Moss has any time on Monday",),
        NONE,
        contains={0: ("2:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "LH24",
        "pure-lookup",
        ("Could you tell me when {rivers} is free on Friday?",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "LH25",
        "pure-lookup",
        ("Just checking, is there a cardiologist at {pine} on Thursday?",),
        NONE,
        contains={0: ("Aiden Kim",)},
        quiet=(0,),
    ),
    Spec(
        "LH26",
        "pure-lookup",
        ("I might want to come in Monday at {pine}. What's available?",),
        NONE,
        contains={0: ("9:00 AM",)},
        quiet=(0,),
    ),
    Spec(
        "LH27",
        "pure-lookup",
        ("Which doctors work at {harbor}?", "and what is Dr. Ortiz's availability tomorrow?"),
        NONE,
        contains={0: ("Dr. Lena Ortiz",), 1: ("11:00 AM", "3:00 PM")},
        quiet=(0, 1),
    ),
    # Boundaries: confirmation stays in the app, sign-in is required, asides keep the review.
    Spec(
        "LH28",
        "mixed",
        ("Book {rivers} Thursday", "yes, confirm it"),
        book("rivers_thu_1000"),
        contains={1: ("Confirm",)},
    ),
    Spec(
        "LH29",
        "mixed",
        ("Anything at {harbor} tomorrow?", "ok, book the 3 PM"),
        NONE,
        anonymous=True,
        contains={0: ("11:00 AM", "3:00 PM")},
        contains_any={1: ("sign in", "Sign in", "signed in")},
        quiet=(0, 1),
    ),
    Spec(
        "LH30",
        "mixed",
        (
            "Book {pine} Monday at 9:30 AM",
            "is Dr. Kim in on Thursday?",
            "ok, the Monday one is good",
        ),
        book("pine_mon_0930"),
        contains={1: ("3:00 PM",)},
        retention=True,
    ),
)

CATEGORIES = ("correction", "lookup-to-booking", "pure-lookup", "mixed")


def fingerprint(specs: tuple[Spec, ...]) -> str:
    canonical = json.dumps([asdict(s) for s in specs], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


HOLDOUT_FINGERPRINT = "b7372c62a0bd2649b504ea42851f6f7602f1fdfcad50f09a46b87a6b21e80bda"
