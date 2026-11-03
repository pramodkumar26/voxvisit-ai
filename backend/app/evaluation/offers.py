"""Resolving what a patient picks from the choices the assistant just offered, deterministically.

A reference is resolved against the offers themselves (their times, doctors, specialties and
names) and against nothing else. Every constraint the words state narrows the offers:

- a part of the day (morning, afternoon, evening), a clock time, a doctor, a kind of doctor
  ("the heart doctor"), a word that tells the offers apart (a clinic's name);
- a position among what is left: "the last one", "the second", "the earliest", and "the earlier"
  or "the later" only between exactly two.

One offer left is the pick. Several left is an ambiguity, and the answer is a question: nothing is
chosen. Nothing left means the words fit none of them. No model proposal, and no second model call,
takes part: a language model reads the message elsewhere, but it never chooses among offers.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.evaluation.spoken import (
    clock_mentions,
    minutes_of_mention,
    normalize_spoken,
    parts_of_day,
    without_dates,
)

_TOKEN = re.compile(r"\d{1,2}:\d{2}|[a-z][a-z'.-]*[a-z]|[a-z]")
_FILLER = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "at",
        "book",
        "booking",
        "by",
        "can",
        "could",
        "do",
        "for",
        "from",
        "go",
        "going",
        "have",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "in",
        "is",
        "it",
        "its",
        "just",
        "like",
        "me",
        "my",
        "of",
        "ok",
        "okay",
        "on",
        "one",
        "ones",
        "please",
        "pls",
        "so",
        "sure",
        "take",
        "that",
        "the",
        "then",
        "this",
        "to",
        "want",
        "wanna",
        "with",
        "would",
        "yes",
        "yeah",
        "yep",
        "you",
        "dr",
        "dr.",
        "doctor",
        "clinic",
        "center",
        "centre",
        "medical",
        "health",
        "care",
        "am",
        "pm",
        "appointment",
    ]
)


def _tokens(text: str) -> set[str]:
    return {t.strip(".'-") for t in _TOKEN.findall(text.casefold())} - _FILLER - {""}


def match_offer(text: str, labels: Sequence[str]) -> int | None:
    """The 1-based number of the one offer the words single out, or None."""
    if len(labels) < 2:
        return None
    parts = [_tokens(label) for label in labels]
    common = set.intersection(*parts)
    said = _tokens(text)
    hits = [i for i, p in enumerate(parts, 1) if (p - common) & said]
    return hits[0] if len(hits) == 1 else None


_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "5th": 5,
}
_EARLIEST = re.compile(r"\b(earliest|soonest|asap|as soon as|first available|first opening)\b")
_LAST = re.compile(r"\b(last|latest|final)\b")
_EARLIER = re.compile(r"\b(earlier|sooner)\b")
_LATER = re.compile(r"\b(later)\b")
_NUMBER = re.compile(r"(?:number|option|choice|#)\s*(\d)")
_PART_BOUNDS = {"morning": (0, 720), "afternoon": (720, 1020), "evening": (1020, 1440)}
# Kinds of doctor by the part of the body or the patient they care for: the stem of the specialty.
_SPECIALTY_WORDS = {
    "heart": "cardio",
    "cardiac": "cardio",
    "cardiologist": "cardio",
    "cardiology": "cardio",
    "skin": "dermat",
    "dermatologist": "dermat",
    "dermatology": "dermat",
    "child": "pediat",
    "children": "pediat",
    "kids": "pediat",
    "pediatrician": "pediat",
    "pediatrics": "pediat",
    "bone": "orthop",
    "bones": "orthop",
    "orthopedic": "orthop",
    "orthopedist": "orthop",
    "family": "family",
    "internal": "internal",
    "internist": "internal",
}
_OPENING = re.compile(r"^(\d{1,2}):(\d{2}) (AM|PM)")
_WITH = re.compile(r"\bwith (Dr\. [^,;]+?)(?: on |$)")
_DOCTOR_LABEL = re.compile(r"^(Dr\. [^()]+?)(?: \(([^)]+)\))?$")


@dataclass(frozen=True)
class Option:
    """One thing that was offered, as far as it can be told apart."""

    label: str
    minutes: int | None = None  # The clock time of an opening.
    day: str | None = None  # Its date, when the listing spanned several days.
    doctor: str = ""
    specialty: str = ""


@dataclass(frozen=True)
class Resolution:
    number: int | None  # The 1-based pick, only when exactly one offer fits.
    candidates: tuple[int, ...]  # What the words leave, when more than one (ask about these).
    stated: bool  # Whether the words said anything about which offer.


def options_from(labels: Sequence[str], details: Sequence[Mapping[str, str]] | None = None):
    """The offers as the patient heard them, with what is known about each: from the listing's own
    data when there is any, otherwise from the label."""
    options = []
    for index, label in enumerate(labels):
        data = details[index] if details and index < len(details) else {}
        minutes, doctor, specialty = None, str(data.get("doctor") or ""), ""
        opening = _OPENING.match(label)
        if opening:
            minutes = (int(opening[1]) % 12 + (12 if opening[3] == "PM" else 0)) * 60 + int(
                opening[2]
            )
            if not doctor and (with_doctor := _WITH.search(label)):
                doctor = with_doctor[1]
        elif named := _DOCTOR_LABEL.match(label):
            doctor, specialty = named[1], named[2] or ""
        options.append(
            Option(
                label,
                minutes,
                str(data["date"]) if data.get("date") else None,
                doctor,
                str(data.get("specialty") or specialty),
            )
        )
    return options


def _name_tokens(value: str) -> set[str]:
    return {t for t in _tokens(value)}


def _keep(fits: list[int], keep: set[int]) -> list[int]:
    return [i for i in fits if i in keep]


def resolve_offer(text: str, options: Sequence[Option]) -> Resolution:
    """Which offer the words pick, if exactly one fits."""
    if not options:
        return Resolution(None, (), False)
    low = normalize_spoken(text).casefold()
    fits = list(range(len(options)))
    stated = False

    if given := _NUMBER.search(low):
        number = int(given[1])
        return Resolution(number if 1 <= number <= len(options) else None, (), True)

    parts = parts_of_day(low) & set(_PART_BOUNDS)
    if parts and any(o.minutes is not None for o in options):
        stated = True
        fits = _keep(
            fits,
            {
                i
                for i, o in enumerate(options)
                if o.minutes is not None
                and any(_PART_BOUNDS[p][0] <= o.minutes < _PART_BOUNDS[p][1] for p in parts)
            },
        )

    clocks = clock_mentions(without_dates(low))
    if clocks and any(o.minutes is not None for o in options):
        stated = True
        wanted: set[int] = set()
        for mention in clocks:
            exact = minutes_of_mention(mention, parts_of_day(low))
            for i, o in enumerate(options):
                if o.minutes is not None and (
                    o.minutes == exact
                    if exact is not None
                    else o.minutes % 720 == (mention.hour % 12) * 60 + mention.minute
                ):
                    wanted.add(i)
        fits = _keep(fits, wanted)

    said = _tokens(text)
    liked = [w for w in said if w in _SPECIALTY_WORDS]
    kinds = bool(liked) and any(o.specialty for o in options)
    if kinds:
        stated = True
        stems = {_SPECIALTY_WORDS[w] for w in liked}
        fits = _keep(
            fits,
            {
                i
                for i, o in enumerate(options)
                if any(o.specialty.casefold().startswith(stem) for stem in stems)
            },
        )

    # What tells the offers apart in their own words: a doctor's name, a clinic's name.
    tokens = [_name_tokens(" ".join((o.label, o.doctor))) for o in options]
    common = set.intersection(*tokens) if len(tokens) > 1 else set()
    distinctive = said - set(_SPECIALTY_WORDS) if kinds else said
    hits = {i for i, t in enumerate(tokens) if (t - common) & distinctive}
    if hits and len(hits) < len(options):
        stated = True
        fits = _keep(fits, hits)

    # A position among what is left. "Earlier" and "later" compare two things, not three.
    def order(i: int) -> tuple[str, int]:
        return (options[i].day or "", options[i].minutes if options[i].minutes is not None else -1)

    timed = bool(fits) and all(options[i].minutes is not None for i in fits)
    if _EARLIEST.search(low) and timed:
        stated = True
        best = min(order(i) for i in fits)
        fits = _keep(fits, {i for i in fits if order(i) == best})
    elif _LAST.search(low):
        stated = True
        if timed:
            best = max(order(i) for i in fits)
            fits = _keep(fits, {i for i in fits if order(i) == best})
        elif fits:
            fits = _keep(fits, {fits[-1]})
    elif _EARLIER.search(low) or _LATER.search(low):
        stated = True
        if timed and len(fits) == 2:
            pick = min(fits, key=order) if _EARLIER.search(low) else max(fits, key=order)
            fits = _keep(fits, {pick})
    else:
        for word, position in _ORDINALS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                stated = True
                fits = _keep(fits, {fits[position - 1]} if position <= len(fits) else set())
                break

    if len(fits) == 1 and stated:
        return Resolution(fits[0] + 1, (), True)
    return Resolution(None, tuple(i + 1 for i in fits) if len(fits) > 1 and stated else (), stated)


def numbered(labels: Sequence[str]) -> str:
    """The offers as the model is shown them."""
    return "; ".join(f"{i}) {label}" for i, label in enumerate(labels, 1))
