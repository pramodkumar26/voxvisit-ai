"""LG-5E: fresh synthetic conversations for four design decisions.

1. Names with a possessive ("Dr. Kim's Monday") are the doctor.
2. Appointment selection is deterministic: a reference to what was offered picks an opening only
   when the words and the offered options identify exactly one; otherwise the patient is asked. A
   delegation ("the earliest available") counts only when it resolves to exactly one opening.
3. Booking intent may come from context: right after the assistant asked whether the patient wants
   to book, a short affirmative reply ("yes", "okay, Wednesday") starts or continues a booking. It
   never chooses an unspecified opening and never commits anything.
4. An operation change is checked against the patient's words and the workflow: a correction to a
   pending booking stays a booking; moving or cancelling needs wording about an existing
   appointment.

The earlier sets (LG-4, LG-5, LG-5D) are regression data only: their holdouts were spent when they
were run. This module adds a DEVELOPMENT set (focused cases and adversarial variations) and a
HOLDOUT, written and fingerprinted before any LG-5E implementation change, run once on the locked
implementation and never edited or tuned against afterwards.

One expectation (EH30, and its development twin EV36) was corrected before any holdout run: it
first used a clinic with two openings, where a plain "yes" cannot be turned into a review (which
time?) for a signed-out visitor or anyone else, so asking which time is right and the sign-in
prompt comes only once a single opening is chosen. The corrected case uses one opening. The set was
fingerprinted again; no holdout result existed when this was done.

Expected outcomes come from the patient's intent and the world's published openings (see
`lg5d_scenarios` for the world), not from what any implementation did. Where the patient's words
honestly fit more than one opening, the expected outcome is a clarifying question and no review.
Today is Monday 2 November 2026. Pine Ridge: Monday 9th 9:00 and 9:30 (Dr. Rao), 10:30 (Dr. Kim,
cardiology); Thursday 5th 2:00 PM (Rao) and 3:00 PM (Kim); Thursday 12th the same.
"""

import hashlib
import json
from dataclasses import asdict

from app.evaluation.lg4_scenarios import Expect, book
from app.evaluation.lg5d_scenarios import NONE, WITHDRAWN, Spec, build_world, move

__all__ = ["DEVELOPMENT", "HOLDOUT", "HOLDOUT_FINGERPRINT", "build_world", "fingerprint"]

CATEGORIES = ("selection", "intent", "operation", "names", "mixed")

DEVELOPMENT: tuple[Spec, ...] = (
    # Selection among what was listed: unique means pick, several means ask.
    Spec(
        "EV01",
        "selection",
        ("Anything at {pine} on Monday?", "book one of the morning slots"),
        NONE,
        avoids={1: ("I prepared",)},
        contains={1: ("Which",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EV02",
        "selection",
        ("What does {pine} have on Monday?", "the later one please"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EV03",
        "selection",
        ("What does {pine} have on Monday?", "I'll take the last one"),
        book("pine_mon_1030"),
        retention=True,
    ),
    Spec(
        "EV04",
        "selection",
        ("What's free at {pine} on Thursday?", "book the later one"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "EV05",
        "selection",
        ("What's free at {pine} on Thursday?", "the earlier one"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "EV06",
        "selection",
        ("Any openings at {pine} on Monday?", "just give me the earliest available"),
        book("pine_mon_0900"),
        retention=True,
    ),
    Spec(
        "EV07",
        "selection",
        ("Any openings at {pine} on Monday?", "book Dr. Anita Rao"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EV08",
        "selection",
        ("Any openings at {pine} on Monday?", "I'd like the cardiologist"),
        book("pine_mon_1030"),
        retention=True,
    ),
    Spec(
        "EV09",
        "selection",
        ("Which clinics do you have?", "Linden Court, Monday"),
        NONE,
        contains={1: ("Heart Center", "Family Clinic")},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EV10",
        "selection",
        ("Book {pine} on Monday", "whichever works"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(1,),
        retention=True,
    ),
    Spec(
        "EV11",
        "selection",
        ("Book {pine} on Monday", "either is fine"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(1,),
        retention=True,
    ),
    Spec(
        "EV12",
        "selection",
        ("What's open at {pine} on Monday morning?", "the 10:30"),
        book("pine_mon_1030"),
        retention=True,
    ),
    # Booking intent from context.
    Spec(
        "EV13",
        "intent",
        ("Is there anything at {rivers} on Friday?", "yes"),
        book("rivers_fri_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EV14",
        "intent",
        ("Is there a pediatrician at {harbor}?", "okay, Wednesday"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EV15",
        "intent",
        ("What's open at {pine} on Monday?", "yes please"),
        NONE,
        avoids={1: ("I prepared",)},
        contains={1: ("Which",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EV16",
        "intent",
        ("What's open at {pine} on Thursday?", "sure, the 2 PM"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "EV17",
        "intent",
        ("Does Dr. Reed have anything on Monday?", "ok"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EV18",
        "intent",
        ("What's available at {rivers} on Thursday?", "and Friday?"),
        NONE,
        contains={0: ("10:00 AM",), 1: ("1:00 PM",)},
        quiet=(0, 1),
    ),
    Spec(
        "EV19",
        "intent",
        ("Who are the doctors at {harbor}?", "yes, Dr. Okafor on Wednesday"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EV20",
        "intent",
        ("Which doctors work at {pine}?", "Dr. Kim then"),
        NONE,
        quiet=(0, 1),
        avoids={1: ("I prepared",)},
    ),
    Spec(
        "EV21",
        "intent",
        ("Anything at {rivers} on Friday?", "yeah that works"),
        book("rivers_fri_1300"),
        quiet=(0,),
        retention=True,
    ),
    # Operation changes.
    Spec(
        "EV22",
        "operation",
        ("Book Dr. Rao at {pine} on Thursday", "change it to Dr. Kim"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "EV23",
        "operation",
        ("Book {rivers} on Thursday", "switch it to Friday"),
        book("rivers_fri_1300"),
        retention=True,
    ),
    Spec(
        "EV24",
        "operation",
        ("Book Dr. Kim at {pine} on Thursday", "make it the 12th with Dr. Rao"),
        book("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "EV25",
        "operation",
        ("Book {pine} on Monday morning", "actually cancel my appointment on the 10th"),
        Expect("cancel", appointment="mine"),
        retention=True,
    ),
    Spec(
        "EV26",
        "operation",
        ("Book {rivers} on Thursday", "move my appointment on the 10th to the 12th at 3 PM"),
        move("pine_nthu_1500"),
        retention=True,
    ),
    Spec(
        "EV27",
        "operation",
        ("Scrap my appointment on the 10th",),
        Expect("cancel", appointment="mine"),
    ),
    Spec(
        "EV28",
        "operation",
        ("Cancel my appointment on the 10th", "no wait, never mind"),
        WITHDRAWN,
        retention=True,
    ),
    Spec(
        "EV29",
        "operation",
        ("Book {harbor} tomorrow at 3 PM", "could we do 11 instead?"),
        book("harbor_tue_1100"),
        retention=True,
    ),
    Spec(
        "EV30",
        "operation",
        ("Move my appointment on the 10th to the 12th at 2 PM", "no, cancel it instead"),
        Expect("cancel", appointment="mine"),
        retention=True,
    ),
    # Possessive names.
    Spec(
        "EV31",
        "names",
        ("Tell me about Dr. Rao's Monday and Dr. Kim's Monday at {pine}",),
        NONE,
        contains={0: ("9:00 AM", "10:30 AM")},
        quiet=(0,),
    ),
    Spec(
        "EV32",
        "names",
        ("Is Dr. Kim's Thursday open at {pine}?",),
        NONE,
        contains={0: ("3:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "EV33",
        "names",
        ("What does Dr. Reed's Monday look like?",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "EV34",
        "names",
        ("Book Dr. Moss's Monday appointment at {family}",),
        book("family_mon_1400"),
    ),
    Spec(
        "EV35",
        "names",
        ("Can I see Dr. Wren's availability on Friday?",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    # Boundaries.
    Spec(
        "EV36",
        "mixed",
        ("What's available at {rivers} on Thursday?", "yes"),
        NONE,
        anonymous=True,
        contains_any={1: ("sign in", "Sign in", "signed in")},
    ),
    Spec(
        "EV37",
        "mixed",
        ("Book {rivers} on Thursday", "yes"),
        book("rivers_thu_1000"),
        contains={1: ("Confirm",)},
    ),
    Spec(
        "EV38",
        "mixed",
        ("Anything at {rivers} on Thursday?", "yes", "ok"),
        book("rivers_thu_1000"),
        contains={2: ("Confirm",)},
        quiet=(0,),
        retention=True,
    ),
)

# --- holdout set -----------------------------------------------------------------------------
# Written and frozen before any LG-5E implementation change. Never edit.

HOLDOUT: tuple[Spec, ...] = (
    # Selection: one match is a pick, several are a question.
    Spec(
        "EH01",
        "selection",
        ("Can you list what's open at {pine} on Monday?", "book the morning one"),
        NONE,
        contains={1: ("Which",)},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EH02",
        "selection",
        ("Anything at {pine} on Monday?", "the later one"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EH03",
        "selection",
        ("Anything at {pine} on Monday?", "book the last one"),
        book("pine_mon_1030"),
        retention=True,
    ),
    Spec(
        "EH04",
        "selection",
        ("Any openings at {pine} on Thursday?", "book the later one"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "EH05",
        "selection",
        ("Any openings at {pine} on Thursday?", "I want the earlier one"),
        book("pine_thu_1400"),
        retention=True,
    ),
    Spec(
        "EH06",
        "selection",
        ("What's open at {pine} on Monday?", "book the earliest available"),
        book("pine_mon_0900"),
        retention=True,
    ),
    Spec(
        "EH07",
        "selection",
        ("What's open at {pine} on Monday?", "book Dr. Rao"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EH08",
        "selection",
        ("What's open at {pine} on Monday?", "book the heart doctor"),
        book("pine_mon_1030"),
        retention=True,
    ),
    Spec(
        "EH09",
        "selection",
        ("What clinics are there?", "Linden Court, on Monday"),
        NONE,
        contains={1: ("Heart Center", "Family Clinic")},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EH10",
        "selection",
        ("Which clinics do you have?", "the heart center, Monday"),
        NONE,
        contains={1: ("1:00 PM",)},
        quiet=(0, 1),
        retention=True,
    ),
    # Booking intent from context.
    Spec(
        "EH11",
        "intent",
        ("Is there anything open at {rivers} on Friday?", "yes please"),
        book("rivers_fri_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EH12",
        "intent",
        ("Do you have a pediatrician at {harbor}?", "okay, Wednesday works"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EH13",
        "intent",
        ("Who works at {harbor}?", "yes, Dr. Okafor on Wednesday"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EH14",
        "intent",
        ("What's available at {pine} on Monday?", "yes"),
        NONE,
        contains={1: ("Which",)},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "EH15",
        "intent",
        ("What's open at {pine} on Thursday?", "sure, the 3 PM"),
        book("pine_thu_1500"),
        retention=True,
    ),
    Spec(
        "EH16",
        "intent",
        ("Is Dr. Reed free on Monday?", "okay"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EH17",
        "intent",
        ("Anything at {rivers} on Thursday?", "that works"),
        book("rivers_thu_1000"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "EH18",
        "intent",
        ("Is there anything at {rivers} on Friday?", "and Thursday?"),
        NONE,
        contains={0: ("1:00 PM",), 1: ("10:00 AM",)},
        quiet=(0, 1),
    ),
    # Operation changes.
    Spec(
        "EH19",
        "operation",
        ("Book Dr. Kim at {pine} on Thursday afternoon", "make it the 12th with Dr. Rao"),
        book("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "EH20",
        "operation",
        ("Book {rivers} on Friday", "change it to Thursday"),
        book("rivers_thu_1000"),
        retention=True,
    ),
    Spec(
        "EH21",
        "operation",
        ("Book {pine} on Monday morning", "hold on, I need to cancel my appointment on the 10th"),
        Expect("cancel", appointment="mine"),
        retention=True,
    ),
    Spec(
        "EH22",
        "operation",
        ("Book {rivers} on Thursday", "move my appointment on the 10th to the 12th at 2 PM"),
        move("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "EH23",
        "operation",
        ("Drop my appointment on the 10th",),
        Expect("cancel", appointment="mine"),
    ),
    Spec(
        "EH24",
        "operation",
        ("Cancel my appointment on the 10th", "never mind, keep it"),
        WITHDRAWN,
        retention=True,
    ),
    Spec(
        "EH25",
        "operation",
        ("Book {harbor} tomorrow at 3 PM", "can we do 11 instead?"),
        book("harbor_tue_1100"),
        retention=True,
    ),
    # Possessive names.
    Spec(
        "EH26",
        "names",
        ("Compare Dr. Kim's Monday with Dr. Rao's Monday at {pine}",),
        NONE,
        contains={0: ("10:30 AM", "9:00 AM")},
        quiet=(0,),
    ),
    Spec(
        "EH27",
        "names",
        ("Is Dr. Kim's Thursday slot open at {pine}?",),
        NONE,
        contains={0: ("3:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "EH28",
        "names",
        ("What is Dr. Ortiz's availability tomorrow at {harbor}?",),
        NONE,
        contains={0: ("11:00 AM", "3:00 PM")},
        quiet=(0,),
    ),
    Spec(
        "EH29",
        "names",
        ("Book Dr. Moss's Monday slot at {family}",),
        book("family_mon_1400"),
    ),
    # Boundary.
    Spec(
        "EH30",
        "mixed",
        ("Anything at {rivers} on Friday?", "yes"),
        NONE,
        anonymous=True,
        contains_any={1: ("sign in", "Sign in", "signed in")},
        quiet=(0, 1),
    ),
)


def fingerprint(specs: tuple[Spec, ...]) -> str:
    canonical = json.dumps([asdict(s) for s in specs], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


HOLDOUT_FINGERPRINT = "3bf72cfc9d350f96222ddf42b68dbf76fd04bb807eb289f368a4b1202e592368"
