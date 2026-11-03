"""LG-5F: the final booking-intent boundary.

A booking begins only on evidence that does not come from a model reading: an act of booking in the
patient's words (including indirect but clear requests: "Can you fit me in?", "I'd like an
appointment at Pine Clinic"), a plain yes right after the assistant asked whether to book, the
patient's pick of an opening just listed, or a request already being arranged. A bare clinic, day
or time ("the heart center, Monday", "Pine Ridge Thursday at 3") is a question about what exists
until one of those makes it a booking.

The earlier sets (LG-4, LG-5, LG-5D, LG-5E) are regression data only: their holdouts are
spent. This module adds a small DEVELOPMENT set and a HOLDOUT of 30 conversations, written and
fingerprinted before any LG-5F implementation change, run once on the locked implementation
and never edited afterwards. Expected outcomes come from the patient's intent and the world's
published openings (see `lg5d_scenarios`); where a message honestly fits several openings, or
states no booking, the expected outcome is a lookup or a question and no review.

Pine Ridge: Monday 9th 9:00 and 9:30 (Dr. Rao), 10:30 (Dr. Kim, cardiology); Thursday 5th and
12th 2:00 PM (Rao) and 3:00 PM (Kim). Two Rivers: Thu 5th 10:00, Fri 6th 1:00 PM. Heart Center:
Monday 1:00 PM (Dr. Reed). Family Clinic: Monday 2:00 PM (Dr. Moss). Harbor View: tomorrow 11:00 AM
and 3:00 PM (Dr. Ortiz), Wed 4th 9:00 AM (Ortiz) and 4:00 PM (Dr. Okafor, pediatrics).
"""

import hashlib
import json
from dataclasses import asdict

from app.evaluation.lg4_scenarios import book
from app.evaluation.lg5d_scenarios import NONE, Spec, build_world

__all__ = ["DEVELOPMENT", "HOLDOUT", "HOLDOUT_FINGERPRINT", "build_world", "fingerprint"]

CATEGORIES = ("bare", "indirect", "affirmative", "correction", "selection")

DEVELOPMENT: tuple[Spec, ...] = (
    Spec(
        "FV01",
        "bare",
        ("Which clinics are available?", "Linden Court Heart Center, Monday"),
        NONE,
        contains={1: ("1:00 PM",)},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FV02",
        "bare",
        ("{rivers} on Friday at 1",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "FV03",
        "bare",
        ("Does {pine} have a cardiologist?", "Dr. Kim, Thursday afternoon"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FV04",
        "indirect",
        ("Could you squeeze me in at {rivers} on Thursday?",),
        book("rivers_thu_1000"),
    ),
    Spec(
        "FV05",
        "indirect",
        ("I'm looking for an appointment at {family} on Monday",),
        book("family_mon_1400"),
    ),
    Spec(
        "FV06",
        "indirect",
        ("Is it possible to get an appointment with Dr. Reed on Monday?",),
        book("heart_mon_1300"),
    ),
    Spec(
        "FV07",
        "affirmative",
        ("Are there openings at {rivers} on Friday?", "yes"),
        book("rivers_fri_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FV08",
        "selection",
        ("What's available at {harbor} tomorrow?", "the 11 AM"),
        book("harbor_tue_1100"),
        quiet=(0,),
        retention=True,
    ),
)

# --- holdout set -----------------------------------------------------------------------------
# Written and frozen before any LG-5F implementation change. Never edit.

HOLDOUT: tuple[Spec, ...] = (
    # Bare phrases and ambiguous follow-ups: still questions about what exists.
    Spec(
        "FH01",
        "bare",
        ("What clinics are there?", "Pine Ridge on Thursday"),
        NONE,
        contains={1: ("2:00 PM", "3:00 PM")},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH02",
        "bare",
        ("What doctors does {pine} have?", "Dr. Kim on Thursday"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH03",
        "bare",
        ("Is there anything at {rivers} on Thursday?", "what about Friday?"),
        NONE,
        contains={1: ("1:00 PM",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH04",
        "bare",
        ("{pine} Thursday at 3 PM",),
        NONE,
        contains={0: ("3:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "FH05",
        "bare",
        ("{rivers} Friday at 1 PM",),
        NONE,
        contains={0: ("1:00 PM",)},
        quiet=(0,),
    ),
    Spec(
        "FH06",
        "bare",
        ("Which clinics do you have?", "the family clinic on Monday"),
        NONE,
        contains={1: ("2:00 PM",)},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH07",
        "bare",
        ("Does Dr. Wren work at {rivers}?", "Friday"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH08",
        "bare",
        ("Who works at {harbor}?", "Dr. Ortiz tomorrow"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    # Indirect but clear requests for an appointment.
    Spec(
        "FH09",
        "indirect",
        ("Can you fit me in Thursday at {pine}?",),
        NONE,
        contains={0: ("Which",)},
        avoids={0: ("I prepared",)},
        quiet=(0,),
    ),
    Spec(
        "FH10",
        "indirect",
        ("Can you fit me in at {rivers} on Friday?",),
        book("rivers_fri_1300"),
    ),
    Spec(
        "FH11",
        "indirect",
        ("I'd like an appointment at {rivers} on Thursday.",),
        book("rivers_thu_1000"),
    ),
    Spec(
        "FH12",
        "indirect",
        ("Could I get an appointment with Dr. Reed on Monday?",),
        book("heart_mon_1300"),
    ),
    Spec(
        "FH13",
        "indirect",
        ("I need to see Dr. Moss on Monday at {family}",),
        book("family_mon_1400"),
    ),
    Spec(
        "FH14",
        "indirect",
        ("Squeeze me in at {harbor} tomorrow at 11",),
        book("harbor_tue_1100"),
    ),
    Spec(
        "FH15",
        "indirect",
        ("Put me down for {rivers} on Thursday",),
        book("rivers_thu_1000"),
    ),
    # A plain yes to the assistant's own question.
    Spec(
        "FH16",
        "affirmative",
        ("Anything at {rivers} on Thursday?", "sure"),
        book("rivers_thu_1000"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH17",
        "affirmative",
        ("Is Dr. Reed in on Monday?", "yes please"),
        book("heart_mon_1300"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH18",
        "affirmative",
        ("What's open at {pine} on Thursday?", "okay, the 2 PM"),
        book("pine_thu_1400"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH19",
        "affirmative",
        ("What's open at {pine} on Monday?", "yes"),
        NONE,
        contains={1: ("Which",)},
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH20",
        "affirmative",
        ("Do you have a pediatrician at {harbor}?", "yeah, Wednesday"),
        book("harbor_wed_1600"),
        quiet=(0,),
        retention=True,
    ),
    # Corrections to a booking in hand.
    Spec(
        "FH21",
        "correction",
        ("Book Dr. Rao at {pine} on Thursday afternoon", "actually make it the 12th"),
        book("pine_nthu_1400"),
        retention=True,
    ),
    Spec(
        "FH22",
        "correction",
        ("Book {rivers} on Friday", "no, Thursday"),
        book("rivers_thu_1000"),
        retention=True,
    ),
    Spec(
        "FH23",
        "correction",
        ("Book {harbor} tomorrow at 11", "3 in the afternoon instead"),
        book("harbor_tue_1500"),
        retention=True,
    ),
    Spec(
        "FH24",
        "correction",
        ("I'd like an appointment at {pine} on Monday at 9 AM", "make it 9:30"),
        book("pine_mon_0930"),
        retention=True,
    ),
    Spec(
        "FH25",
        "correction",
        ("Can you fit me in at {rivers} on Thursday?", "actually Friday"),
        book("rivers_fri_1300"),
        retention=True,
    ),
    # Choosing among what was listed.
    Spec(
        "FH26",
        "selection",
        ("Any openings at {harbor} tomorrow?", "the 3 PM"),
        book("harbor_tue_1500"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH27",
        "selection",
        ("What's open at {pine} on Monday?", "Monday at 9:30"),
        book("pine_mon_0930"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH28",
        "selection",
        ("What's open at {pine} on Monday?", "the afternoon one"),
        NONE,
        avoids={1: ("I prepared",)},
        quiet=(0, 1),
        retention=True,
    ),
    Spec(
        "FH29",
        "selection",
        ("Anything at {pine} on Thursday?", "book Dr. Kim"),
        book("pine_thu_1500"),
        quiet=(0,),
        retention=True,
    ),
    Spec(
        "FH30",
        "selection",
        ("What's open at {pine} on Monday?", "I'll take the cardiologist"),
        book("pine_mon_1030"),
        quiet=(0,),
        retention=True,
    ),
)


def fingerprint(specs: tuple[Spec, ...]) -> str:
    canonical = json.dumps([asdict(s) for s in specs], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


HOLDOUT_FINGERPRINT = "c2d0781e7d528c0da48c13d4dc8c8835d66f2566461f6e6c2992007579737e30"
