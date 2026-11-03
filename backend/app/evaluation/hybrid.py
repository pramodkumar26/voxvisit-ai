"""Approach B: Qwen reads a message into a structured intent; deterministic code does the rest.

The model's only job is to turn one patient message into a small, schema-bound `PatientIntent`.
Everything that matters is decided by code: which tool to call and with what, which doctor,
appointment or time is meant (from trusted tool results, never from the model), what to ask when
something is missing, and what to say (every detail in a reply is read back from a tool result).
The model never sees a tool result, never chooses a tool, and has no way to confirm a change: the
trusted UI does that, and it tells the agent afterwards.

The controller keeps a small `ConversationState` between messages so a conversation can continue:
it answers a question, changes a date or a doctor, drops a request, or moves on. Each message
merges into the draft request, and the request is then planned again from the start with fresh tool
calls, so availability and authorization are never reused from an earlier turn.
"""

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.evaluation.availability_notes import (
    FILTER_ARGUMENTS,
    LOOK_AHEAD_DAYS,
    filtered_out,
    later_days,
    next_days,
    no_such_specialty,
    nothing_ahead,
    same_specialty,
    specialties_listed,
    specialties_open,
)
from app.evaluation.conversation import (
    MAX_TURNS_PER_SESSION,
    Awaiting,
    ConversationState,
    Offer,
    Prepared,
)
from app.evaluation.llm import LlmConfig, LlmUsage, Transport, http_transport, record_of
from app.evaluation.model import MalformedModelResponse, Message, Say, ToolAction
from app.evaluation.spoken import (
    MONTHS,
    WEEKDAYS,
    ClockMention,
    DateMention,
    clock_mentions,
    date_mentions,
    label,
    minutes_of,
    minutes_of_mention,
    normalize_spoken,
    parts_of_day,
    resolve_spec,
    stated_clocks,
    stated_dates,
)

MAX_SEARCH_DAYS = 31  # The backend refuses a longer availability window.
SPOKEN_OPTION_LIMIT = 3
HYBRID_PROMPT_VERSION = "ab-b6"
_INTENT_INSTRUCTIONS = """\
Read the patient's LATEST message about a scheduling request and return its intent as JSON
matching the schema. Today is {today}.

- operation: BOOK, RESCHEDULE, CANCEL, or UNKNOWN. UNKNOWN if the message does not say (for
  example when it only answers a question or changes a detail).
- clinic_name: the clinic's name exactly as the patient wrote it, or null if none is named.
- doctor_name: the doctor's name as the patient wrote it, with or without "Dr." and including
  every word of the name ("Dr. Lee Marsh" gives "Lee Marsh"); null if no doctor is named.
- specialty: only if the patient named a medical specialty; otherwise null.
- start_date, end_date: the day the patient means, written YYYY-MM-DD, even if they wrote it
  another way ("November 17", "17 November 2026", "tomorrow", "next Monday"). A date without a
  year is its next occurrence. For one day, both are that day.
- time_preference: MORNING or AFTERNOON only if the patient said so; otherwise ANYTIME.
- new_date: only when the patient moves an existing appointment to a different day: that new day
  (YYYY-MM-DD). Then start_date is the day of the existing appointment, if they gave it.
- time: a clock time the patient wants, exactly as written ("11 AM"); otherwise null.
- appointment_time: the clock time of an EXISTING appointment the patient points to ("the 10 AM
  one", "my 2 PM appointment"); otherwise null.
- option_number: 1 to 5, only when the patient picks from a list by position ("the second one");
  otherwise null.
- selection: EARLIEST for the first, earliest, soonest, or next available time (including "move
  it to the next available time"); LATER only for words like "later" or "afterwards"; EARLIER
  only for words like "earlier" or "before"; ANY if any time that fits is fine ("any morning
  time"); SPECIFIC_TIME if they name a clock time (put it in "time"); otherwise UNSPECIFIED.
- message_type: REQUEST for anything that states, answers, or changes a scheduling request;
  ABANDON when the patient drops the request ("never mind", "forget it"); AGREEMENT when the
  message is only "yes", "okay", or "sounds good" with nothing new; OFF_TOPIC when it is not about
  booking, moving, or cancelling an appointment.
Return only what the LATEST message says; use null for anything it does not say, because earlier
details are remembered separately. Never invent an identifier, credential, tool name,
confirmation, or preference. Copy names from the message.
{context}
Examples (made up):
{examples}
"""

_EXAMPLES = (
    (
        "Book the first appointment at Quartz Ridge Clinic on 2031-03-04.",
        {
            "operation": "BOOK",
            "clinic_name": "Quartz Ridge Clinic",
            "start_date": "2031-03-04",
            "end_date": "2031-03-04",
            "selection": "EARLIEST",
        },
    ),
    (
        "I need a Cardiology visit at Sapphire Medical on March 9, 2031, afternoon only.",
        {
            "operation": "BOOK",
            "clinic_name": "Sapphire Medical",
            "specialty": "Cardiology",
            "start_date": "2031-03-09",
            "end_date": "2031-03-09",
            "time_preference": "AFTERNOON",
        },
    ),
    (
        "Could you move my visit at Amber Clinic on 12 May 2031 to a later time?",
        {
            "operation": "RESCHEDULE",
            "clinic_name": "Amber Clinic",
            "start_date": "2031-05-12",
            "end_date": "2031-05-12",
            "selection": "LATER",
        },
    ),
    (
        "Find the first opening with Dr. Jonas Ferrell at Garnet Clinic on 2031-06-02 and book it.",
        {
            "operation": "BOOK",
            "clinic_name": "Garnet Clinic",
            "doctor_name": "Jonas Ferrell",
            "start_date": "2031-06-02",
            "end_date": "2031-06-02",
            "selection": "EARLIEST",
        },
    ),
    (
        "Move my 2031-07-08 appointment at Opal Health to the next available time that day.",
        {
            "operation": "RESCHEDULE",
            "clinic_name": "Opal Health",
            "start_date": "2031-07-08",
            "end_date": "2031-07-08",
            "selection": "EARLIEST",
        },
    ),
    (
        "Cancel my appointment at Topaz Health on 5 April 2031.",
        {
            "operation": "CANCEL",
            "clinic_name": "Topaz Health",
            "start_date": "2031-04-05",
            "end_date": "2031-04-05",
        },
    ),
    (
        "Please move my Opal Health appointment to 2031-07-10 instead.",
        {
            "operation": "RESCHEDULE",
            "clinic_name": "Opal Health",
            "new_date": "2031-07-10",
        },
    ),
    (
        "Actually, make it the 11th.",
        {"operation": "UNKNOWN", "start_date": "2031-07-11", "end_date": "2031-07-11"},
    ),
    ("Never mind, forget it.", {"operation": "UNKNOWN", "message_type": "ABANDON"}),
    ("Yes, that sounds good.", {"operation": "UNKNOWN", "message_type": "AGREEMENT"}),
    ("What time does the pharmacy close?", {"operation": "UNKNOWN", "message_type": "OFF_TOPIC"}),
)
_EXAMPLE_TEXT = "\n".join(f"Message: {text}\n{json.dumps(answer)}" for text, answer in _EXAMPLES)
# The prompt as the model sees it on a first message, with no date and no earlier conversation.
INTENT_PROMPT = _INTENT_INSTRUCTIONS.format(
    today="(given with each message)", context="", examples=_EXAMPLE_TEXT
)
REPAIR_NOTE = (
    "\nYour previous reply could not be used. Reply with only one JSON object that follows the "
    "schema exactly: no explanation, no extra fields."
)


_POSSESSIVE = re.compile(r"['\u2019]s$", re.IGNORECASE)


def without_possessive(name: str | None) -> str | None:
    """ "Dr. Kim's" (said of Dr. Kim's Monday) is Dr. Kim."""
    return _POSSESSIVE.sub("", name.strip()) if name else name


class PatientIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("clinic_name", "doctor_name", mode="after")
    @classmethod
    def _name_without_possessive(cls, value: str | None) -> str | None:
        return without_possessive(value)

    operation: Literal["BOOK", "RESCHEDULE", "CANCEL", "UNKNOWN"]
    clinic_name: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    new_date: date | None = None
    time: str | None = None
    appointment_time: str | None = None
    option_number: int | None = None
    selection: Literal["EARLIEST", "LATER", "EARLIER", "ANY", "SPECIFIC_TIME", "UNSPECIFIED"] = (
        "UNSPECIFIED"
    )
    message_type: Literal["REQUEST", "ABANDON", "AGREEMENT", "OFF_TOPIC"] = "REQUEST"


@dataclass
class PlannerHints:
    """What the graph planner has understood about this one message, passed to the controller
    and used for that message only. A hint can keep the request in progress from being replaced,
    or carry what a lookup just showed; it never supplies a time, a doctor, or a decision."""

    same_request: bool = False  # The message answers or corrects the request in progress.
    keep_goal: bool = False  # An off-topic reading must not end the request.
    carry_clinic: str | None = None  # A clinic the patient was just shown openings for.
    carry_date: date | None = None  # The one day they were shown.
    carry_window: tuple[date, date] | None = None  # Or the days: "Thursday" is one of these.
    carry_doctor: str | None = None  # The one doctor they were just told about.
    start_booking: bool = False  # The message asks for an appointment without a booking word.


class Draft(PatientIntent):
    """The request so far, merged across messages. Only the model-facing fields above are read
    from the model; the rest is kept by the controller from the words of the messages."""

    not_before: int | None = None  # Minutes after midnight: "after ten".
    not_after: int | None = None  # "before eleven".
    pending_clock: str | None = None  # "time:2:15": said without am or pm, still to be asked.
    start_spec: str = (
        ""  # How start_date was said ("rel:tomorrow"), to read it on the clinic's day.
    )
    new_spec: str = ""
    # Whether the patient clearly asked for this operation (by an act of booking, by agreeing to
    # a listed opening, or by a reading an independent second reading confirmed). A request that
    # only asked what exists is never turned into a review: see `_continue`.
    vouched: bool = False


# --- reading dates and times: see app/evaluation/spoken.py ---------------------------------------

_ORDINALS = re.compile(
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|(?:number|option|#)\s*[1-5])\b",
    re.IGNORECASE,
)
_SELECTION_WORDS = {
    "EARLIEST": re.compile(r"\b(first|earliest|soonest|next|asap|as soon as)\b", re.IGNORECASE),
    "LATER": re.compile(r"\b(later|afterwards?|after|latest)\b", re.IGNORECASE),
    "EARLIER": re.compile(r"\b(earlier|before|sooner)\b", re.IGNORECASE),
    "ANY": re.compile(
        r"\b(any|anytime|whenever|whatever|morning|afternoon|doesn'?t matter|don'?t care)\b",
        re.IGNORECASE,
    ),
}
_ORDINAL_VALUE = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _name_tokens(value: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s']", " ", re.sub(r"['\u2019]s\b", "", value.casefold()))
    return [token for token in cleaned.split() if token not in {"dr", "doctor"}]


def _name_supported(value: str, request: str) -> bool:
    """Whether the message really contains this name (every word of it, in any case)."""
    if value.casefold() in request.casefold():
        return True
    wanted, stated = _name_tokens(value), set(_name_tokens(request))
    return bool(wanted) and all(token in stated for token in wanted)


_GENERIC_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "at",
        "care",
        "center",
        "centre",
        "clinic",
        "clinics",
        "dr",
        "doctor",
        "group",
        "health",
        "hospital",
        "in",
        "medical",
        "of",
        "office",
        "practice",
        "the",
    ]
)


def _stated_part(value: str, request: str) -> str | None:
    """The longest run of words of a name that the message really contains, as written there, if
    it holds a word that tells places apart. "Two Rivers Clinic" for a patient who said "Two
    Rivers" is "Two Rivers"; a lone "Clinic" is nothing."""
    words = value.split()
    lowered = request.casefold()
    for size in range(len(words) - 1, 0, -1):
        for start in range(len(words) - size + 1):
            part = " ".join(words[start : start + size])
            at = lowered.find(part.casefold())
            if at >= 0 and any(
                w.casefold().strip(".,") not in _GENERIC_WORDS for w in part.split()
            ):
                return request[at : at + len(part)]
    return None


def _specialty_stated(specialty: str, request: str) -> bool:
    """Whether the message names this specialty: "cardiologist" names Cardiology. Compared by
    stem, so any specialty and any form of its name works, with no list of them."""
    if _name_supported(specialty, request):
        return True
    stem = specialty.casefold().strip()[:5]
    return len(stem) == 5 and any(
        word[:5] == stem for word in re.findall(r"[a-z]{5,}", request.casefold())
    )


class StructuredIntentExtractor:
    def __init__(
        self,
        config: LlmConfig,
        *,
        transport: Transport | None = None,
        timer=time.perf_counter,
    ) -> None:
        self.config = config
        self.real_transport = transport is None
        self._post = transport or http_transport(config.base_url)
        self._timer = timer
        self.usage = LlmUsage()
        self.served_replies = 0
        self.last_content: str | None = None  # The latest reply, kept in memory for recovery only.

    def prompt(self, today: date | None, context: str) -> str:
        return _INTENT_INSTRUCTIONS.format(
            today=(today or datetime.now(UTC).date()).strftime("%Y-%m-%d (%A)"),
            context=f"\nThe conversation so far: {context}\n" if context else "",
            examples=_EXAMPLE_TEXT,
        )

    def extract(
        self,
        request: str,
        *,
        context: str = "",
        today: date | None = None,
        anchor: date | None = None,
        repair: bool = False,
    ) -> PatientIntent:
        """One strict reading: any reply that is not valid, or states something the message does
        not, is rejected whole (see `salvage` for what the controller does next)."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.prompt(today, context) + REPAIR_NOTE * repair},
                {"role": "user", "content": request[: self.config.max_message_chars]},
            ],
            "format": PatientIntent.model_json_schema(),
            "stream": False,
            **({"think": self.config.think} if self.config.think is not None else {}),
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
                "num_ctx": self.config.context_tokens,
                **({"seed": self.config.seed} if self.config.seed is not None else {}),
            },
        }
        started = self._timer()
        self.last_content = None
        try:
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
        except Exception as error:
            self._record(started, type(error).__name__)
            raise
        self.served_replies += 1
        try:
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise MalformedModelResponse("the structured intent reply is empty")
            self.last_content = content
            intent = PatientIntent.model_validate_json(content)
            if self.unsupported(intent, request, today, anchor):
                raise ValueError("an extracted value was not stated")
        except MalformedModelResponse:
            self._record(started, "invalid_structured_intent", body)
            raise
        except (ValidationError, ValueError, json.JSONDecodeError):
            self._record(started, "invalid_structured_intent", body)
            raise MalformedModelResponse("the structured intent was not valid") from None
        self._record(started, "structured_intent", body)
        return intent

    @staticmethod
    def unsupported(
        intent: PatientIntent,
        request: str,
        today: date | None = None,
        anchor: date | None = None,
    ) -> dict[str, Any]:
        """Fields the message does not support, with the value that replaces each."""
        stated = request.casefold()
        dates = stated_dates(request, today or datetime.now(UTC).date(), anchor)
        clocks = stated_clocks(request)
        replaced: dict[str, Any] = {}
        for field in ("clinic_name", "doctor_name"):
            value = getattr(intent, field)
            if value and not _name_supported(value, request):
                replaced[field] = _stated_part(value, request)  # What was said, or nothing.
        if intent.specialty and not _specialty_stated(intent.specialty, request):
            replaced["specialty"] = None
        for field in ("start_date", "end_date", "new_date"):
            value = getattr(intent, field)
            if value and value.isoformat() not in dates:
                replaced[field] = None
        for field in ("time", "appointment_time"):
            value = getattr(intent, field)
            if value and minutes_of(value) not in clocks:
                replaced[field] = None
        if intent.time_preference != "ANYTIME" and intent.time_preference.casefold() not in stated:
            replaced["time_preference"] = "ANYTIME"
        word = _SELECTION_WORDS.get(intent.selection)
        if word is not None and not word.search(request):
            replaced["selection"] = "UNSPECIFIED"
        if intent.selection == "SPECIFIC_TIME":
            # A named clock time is what makes a selection specific; without one it is nothing.
            clock_kept = [f for f in ("time", "appointment_time") if getattr(intent, f)]
            if not [f for f in clock_kept if f not in replaced]:
                replaced["selection"] = "UNSPECIFIED"
        if intent.option_number is not None and not (
            _ORDINALS.search(request) and 1 <= intent.option_number <= 5
        ):
            replaced["option_number"] = None
        return replaced

    def salvage(
        self,
        request: str,
        today: date | None = None,
        anchor: date | None = None,
    ) -> PatientIntent | None:
        """Use the reply the last strict reading rejected, keeping only what the message supports.

        No new model call. A reply that was not JSON, or not the schema's shape, is not salvaged:
        the answer is None. Fields that fail on their own are dropped, and a value the message
        does not contain becomes null, so the controller asks about it instead of trusting it.
        """
        if not self.last_content:
            return None
        try:
            raw = json.loads(self.last_content)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict) or raw.get("operation") not in {
            "BOOK",
            "RESCHEDULE",
            "CANCEL",
            "UNKNOWN",
        }:
            return None
        raw = {key: value for key, value in raw.items() if key in PatientIntent.model_fields}
        for _ in range(len(raw)):
            try:
                intent = PatientIntent.model_validate(raw)
                break
            except ValidationError as error:
                bad = {item["loc"][0] for item in error.errors() if item["loc"]} - {"operation"}
                if not bad:
                    return None
                raw = {key: value for key, value in raw.items() if key not in bad}
        else:
            return None
        return intent.model_copy(update=self.unsupported(intent, request, today, anchor))

    def _record(self, started: float, outcome: str, body: Mapping[str, Any] | None = None) -> None:
        self.usage.calls.append(record_of((self._timer() - started) * 1000, outcome, body))

    @property
    def server_attested_replies(self) -> int:
        return sum(
            call.server_ms is not None and call.output_tokens is not None
            for call in self.usage.calls
        )


# --- helpers for wording ----------------------------------------------------------------------


def _text(value: object) -> str:
    return str(value or "").casefold().strip()


def _clock(local: str) -> str:
    """ "2026-11-16T09:00:00-07:00" -> "9:00 AM"."""
    hour, minute = int(local[11:13]), local[14:16]
    return f"{hour % 12 or 12}:{minute} {'AM' if hour < 12 else 'PM'}"


def _minutes(local: str) -> int:
    return int(local[11:13]) * 60 + int(local[14:16])


def _day_label(local: str) -> str:
    """ "2026-11-16T..." -> "Monday, November 16, 2026"."""
    day = date.fromisoformat(local[:10])
    return f"{day.strftime('%A')}, {MONTHS[day.month - 1].title()} {day.day}, {day.year}"


def _relative_date_words(spec: str) -> str | None:
    """Patient-facing wording for a date that still depends on the clinic's calendar."""
    kind, _, rest = spec.partition(":")
    if kind == "rel":
        parts = rest.split(":")
        if parts[0] == "dayafter":
            return "the day after tomorrow"
        if parts[0] == "plus" and len(parts) == 2:
            return f"{parts[1]} days from today"
        if len(parts) == 2 and parts[0] == "next":
            return f"next {parts[1].title()}"
        return parts[0].replace("_", " ")
    if kind == "week":
        return f"{rest} week"
    if kind == "wkday":
        scope, _, weekday = rest.partition(":")
        return f"{weekday.title()} {scope} week"
    return None


def _when(first: date | None, last: date | None = None) -> str:
    """ " on Monday, ..." for one day, " from ... to ..." for a stretch, empty for none."""
    if first is None:
        return ""
    if last is None or last == first:
        return f" on {_day_label(first.isoformat())}"
    return f" from {_day_label(first.isoformat())} to {_day_label(last.isoformat())}"


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def numbered_labels(labels: Sequence[str]) -> str:
    return "; ".join(f"option {index}, {label}" for index, label in enumerate(labels, 1))


def numbered_options(offers: Sequence[Offer]) -> str:
    return numbered_labels([offer.label for offer in offers])


def compact_numbered_labels(labels: Sequence[str]) -> str:
    split = [label.partition(" on ") for label in labels]
    if split and all(separator and details == split[0][2] for _, separator, details in split):
        return f"on {split[0][2]}: {numbered_labels([clock for clock, _, _ in split])}"
    return numbered_labels(labels)


def compact_numbered_options(offers: Sequence[Offer]) -> str:
    return compact_numbered_labels([offer.label for offer in offers])


def _window_words(draft: Draft) -> str:
    after = f"after {label(draft.not_before)}" if draft.not_before is not None else ""
    before = f"before {label(draft.not_after)}" if draft.not_after is not None else ""
    return " and ".join(part for part in (after, before) if part)


def _before(first: Any, second: Any) -> bool:
    """Whether the first local time is earlier than the second. Either may be a slot or a string."""
    left = first["starts_at_local"] if isinstance(first, Mapping) else first
    right = second["starts_at_local"] if isinstance(second, Mapping) else second
    return str(left) < str(right)


# --- reading a message without the model ------------------------------------------------------

_CREDENTIAL = re.compile(
    r"\b(pin|passcode|password|passphrase|login code|log-in code|access code|security code|"
    r"verification code|one[- ]time code|otp|session token|access token|patient code)\b",
    re.IGNORECASE,
)
_DIGITS_ONLY = re.compile(r"^\W*\d{4,10}\W*$")
# Digits said aloud, one at a time ("four eight two one"): a spoken PIN or code, not a date or time.
_SPOKEN_DIGITS = re.compile(
    r"^\W*(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|\d)[\s,.-]*){4,10}\W*$",
    re.IGNORECASE,
)
_AGREEMENT = re.compile(
    r"^\W*(yes|yeah|yep|yup|ok|okay|sure|alright|fine|perfect|great|sounds good|that works|"
    r"that sounds good|that's right|that is right|correct|go ahead|do it|please do|"
    r"confirm|confirmed|i confirm|book it|yes please|ok(ay)? (great|thanks|thank you))"
    r"(\s*(please|thanks|thank you|then))?\W*$",
    re.IGNORECASE,
)
_ABANDON = re.compile(
    r"\b(never\s*mind|forget (?:it|that|about it)|scratch that|drop (?:it|that)|"
    r"don'?t bother|leave it|i changed my mind|no thanks|no thank you|stop)\b",
    re.IGNORECASE,
)
_OPERATION_WORDS = {
    "BOOK": re.compile(
        r"\b(book(?:ing)?|schedule|reserve|appointments?|visits?|see|slots?|openings?|available|"
        r"availabilit(?:y|ies)|"
        r"need|want|make|set up|get me|find|check|looking|"
        r"get (?:me |us )?in(?! touch)|(?:fit|squeeze|work) (?:me|us) in|sign (?:me|us) up)\b",
        re.IGNORECASE,
    ),
    "CANCEL": re.compile(
        r"\b(cancel|call off|scrap|drop|remove|delete|get rid of|can'?t make|cannot make|"
        r"can'?t come|won'?t be able|unable to (?:make|attend))\b",
        re.IGNORECASE,
    ),
    "RESCHEDULE": re.compile(
        r"\b(reschedul\w*|mov(?:e|ing)|chang(?:e|ing)|push|switch|shift|postpone|"
        r"another time|different time|later time|earlier time)\b",
        re.IGNORECASE,
    ),
}


# What plainly asks for a booking. The wider list above is only a necessary condition (the words an
# appointment request may contain); "any availability?", "are there openings?" and "can I see what
# is free?" use those words to ask what exists. Only an act of booking is a request by wording
# alone; anything else needs the reading to be vouched for (see `OrchestratedSession`).
_BOOKING_ACT = re.compile(
    r"\b(book(?:ing)?|schedule|reserve|set up|"
    r"make (?:me |us )?(?:an?|the) (?:appointment|booking)|"
    r"get (?:me |us )?(?:in|an appointment|a slot)(?! touch)|"
    r"(?:fit|squeeze|work|pencil) (?:me|us) in|sign (?:me|us) up|put (?:me|us) down|"
    r"(?:find|get|give) (?:me|us) (?:an?|something|someone)\b|"
    # "Can I get an appointment…?", "Is it possible to get an appointment…?"
    r"(?:can|could|may|might) (?:i|we) (?:please )?(?:get|have|make|book|schedule) (?:an?|the) "
    r"(?:\w+ )?(?:appointment|visit|slot|time)|"
    r"(?:is it|would it be) possible (?:to|for me to|for us to) (?:get|have|make|book|schedule)|"
    r"(?:i'?m|we'?re|i am|we are) (?:looking|hoping|trying) (?:for (?:an? )?(?:appointment|visit)"
    r"|to (?:book|schedule|get|come in|be seen|make|see (?:dr\b|doctor|someone|a doctor))))\b"
    # A wish for an appointment is a request; a wish to know what exists is not.
    r"|\b(?:i|we)(?:'d| would|'ll| will| need| want| wanna| have to)\b\s+"
    r"(?!(?:like\s+)?(?:to\s+)?(?:know\b|ask\b|find out\b|check\b|hear\b|learn\b|confirm\b|"
    r"see (?:what|if|whether|when|who|which|how|about|the (?:times|openings|slots|availability))))",
    re.IGNORECASE,
)


# A request that is only the thing asked for: "An appointment at Linden Court on Monday". Not
# when it asks whether one exists ("an appointment available Friday?").
_NOUN_REQUEST = re.compile(
    r"^\W*an?\s+(?:appointment|visit)\b(?!.*\b(?:available|availability|open|free|slots?|openings?)\b)"
    r"[^?]*$",
    re.IGNORECASE,
)


def operation_stated(operation: str, text: str) -> bool:
    """Whether the wording itself plainly asks for the operation (no judgement of meaning)."""
    if operation == "BOOK":
        return bool(_BOOKING_ACT.search(text) or _NOUN_REQUEST.match(text))
    pattern = _OPERATION_WORDS.get(operation)
    return bool(pattern and pattern.search(text))


def agreement_stated(text: str) -> bool:
    """Whether the whole message is a plain "yes" (and nothing else)."""
    return bool(_AGREEMENT.match(text))


_AFFIRMATIVE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|ok|okay|sure|alright|all right|absolutely|definitely|please|fine|"
    r"great|perfect|sounds good|that works|that would be great|let'?s do it|go ahead|do it)\b",
    re.IGNORECASE,
)
_EXISTING = re.compile(
    r"\bmy\s+(?:\w+\s+){0,2}(?:appointment|booking|visit)s?\b|"
    r"\b(?:existing|current|already (?:booked|scheduled))\s*(?:appointment|booking)?\b",
    re.IGNORECASE,
)


def affirmative_stated(text: str) -> bool:
    """Whether the message opens with a plain yes and is not a question: the reply to "Would you
    like to book…?" that says so ("yes", "okay, Wednesday")."""
    return bool(_AFFIRMATIVE.match(text)) and "?" not in text


def refers_to_existing(text: str) -> bool:
    """Whether the wording is about an appointment the patient already has ("my appointment on
    the 10th"). Moving or cancelling needs that; a correction to a booking being arranged does
    not."""
    return bool(_EXISTING.search(text))


def withdrawal_stated(text: str) -> bool:
    """Whether the wording itself plainly drops the request ("never mind")."""
    return bool(_ABANDON.search(text))


def withdrawal_remainder(text: str) -> str:
    """What the message says besides the withdrawal itself: "forget it, the heart center instead"
    leaves "the heart center instead". A bare "never mind" leaves nothing."""
    return re.sub(r"^[\W_]+|[\W_]+$", "", _ABANDON.sub(" ", text)).strip()


# Words around a withdrawal that add nothing: "never mind, thanks" drops the request, "never mind,
# the heart center instead" drops it for another.
_POLITE = frozenset(
    [
        "the",
        "at",
        "ok",
        "okay",
        "please",
        "one",
        "works",
        "is",
        "fine",
        "good",
        "perfect",
        "great",
        "thanks",
        "then",
        "i'll",
        "ill",
        "take",
        "go",
        "for",
        "with",
        "make",
        "it",
        "that",
        "sounds",
        "let's",
        "lets",
        "do",
        "a",
        "an",
        "um",
        "uh",
        "yes",
        "yeah",
        "sure",
        "sorry",
        "thank",
        "you",
        "actually",
        "now",
        "just",
        "i",
        "we",
        "hmm",
        "well",
        "so",
        "all",
    ]
)


def withdrawal_adds(text: str) -> bool:
    """Whether a message that withdraws the request also says something else (a replacement)."""
    words = re.findall(r"[a-z0-9']+", withdrawal_remainder(text).casefold())
    return any(w not in _POLITE for w in words)


_OFFERED_CLOCK = re.compile(r"(\d{1,2}):(\d{2}) (AM|PM)")
_TO_TIME = re.compile(r"\b(to|for|at)\s*$", re.IGNORECASE)
_MERIDIEM = re.compile(r"\b([ap])\.?m\b", re.IGNORECASE)


def _mask_names(text: str, *names: str | None) -> str:
    """The text with each known name replaced by spaces of the same length (positions are kept)."""
    for name in names:
        if name and name.strip():
            text = re.sub(
                re.escape(name.strip()), lambda m: " " * len(m.group(0)), text, flags=re.IGNORECASE
            )
    return text


def _meridiem_in(text: str, parts: set[str]) -> str | None:
    """ "am" or "pm" if the message says which, by am or pm or by morning or afternoon."""
    found = {m.group(1).lower() + "m" for m in _MERIDIEM.finditer(text)}
    if "morning" in parts:
        found.add("am")
    if parts & {"afternoon", "evening"}:
        found.add("pm")
    return found.pop() if len(found) == 1 else None


_DESTINATION = re.compile(r"\b(to|into|onto|for)\s*(the\s*)?$", re.IGNORECASE)
_SOURCE = re.compile(r"\b(from|my)\s*(the\s*)?$", re.IGNORECASE)
_MOVE_TO = re.compile(r"\b(?:to|into|onto)\b", re.IGNORECASE)  # Anywhere before: where it goes.

# Words that hand the choice of time to the agent. A time of day ("morning") or a date phrase
# ("next Monday") is a preference, not a choice: the patient is asked which exact time.
_DELEGATION = {
    "EARLIEST": re.compile(
        r"\b(first|earliest|soonest|asap|as soon as|next (?:available|open|opening|appointment|"
        r"slot|time))\b",
        re.IGNORECASE,
    ),
    "ANY": re.compile(
        r"\b(any ?time|any (?:available |open |free )?(?:time|day|slot|opening|appointment|"
        r"morning|afternoon)s?|whenever|whatever (?:time|works|is)|doesn'?t matter|don'?t care|"
        r"no preference|you pick|you choose|up to you|surprise me)\b",
        re.IGNORECASE,
    ),
}
_ANY_DOCTOR = re.compile(
    r"\b(?:any|whichever|whoever|no preference (?:on|for|about) the)\s+"
    r"(?:doctor|physician|provider|available doctor)s?\b|\bany ?one\b|"
    r"\b(?:doesn'?t|does not) matter (?:which|who)\b",
    re.IGNORECASE,
)
_NO_PREFERENCE = re.compile(
    r"\b(any|anyone|whichever|whoever|doesn'?t matter|no preference|don'?t care|you pick|"
    r"you choose|up to you)\b",
    re.IGNORECASE,
)
_GENERIC_NAME = {
    "any", "anyone", "anybody", "whoever", "whichever", "one", "available", "a", "the", "some",
    "provider", "physician", "clinic", "clinics", "location", "office", "that", "this", "same",
}  # fmt: skip
_ROLE_WORDS = {
    "general",
    "practitioner",
    "gp",
    "family",
    "primary",
    "care",
    "internal",
    "specialist",
}


def _role_phrase(value: str | None) -> bool:
    """ "General Physician", "family doctor": a kind of doctor, which a reading sometimes writes
    in the place of a doctor's name. With "Dr." or any word beyond these, it is a name."""
    words = re.findall(r"[a-z]+", (value or "").casefold())
    allowed = _GENERIC_NAME | _ROLE_WORDS | {"doctor"}
    return (
        bool(words)
        and "dr" not in words
        and all(word in allowed for word in words)
        and any(word in _ROLE_WORDS for word in words)
    )


_CLINIC_LIST = re.compile(
    r"\b(?:which|what|list|show|tell me|name|any other|other|different|where are)\b[^.?!]{0,40}?"
    r"\b(?:clinics|locations|offices|hospitals|centers|centres)\b"
    r"|\b(?:which|what) clinic\b[^.?!]{0,30}"
    r"\b(?:available|options?|choices?|do you have|are there)\b"
    r"|\b(?:any|whichever|whatever|either) (?:clinic|location|office)\b",
    re.IGNORECASE,
)
_DOCTOR_LIST = re.compile(
    r"\b(?:which|what|list|show|who|tell me|name)\b[^.?!]{0,40}?"
    r"\b(?:doctors|physicians|providers|specialists)\b"
    r"|\b(?:which|what|who) (?:doctor|physician|provider)s? (?:is|are|do you have|can i)\b"
    r"|\bwho can i see\b",
    re.IGNORECASE,
)
_WHAT_IS_THERE = re.compile(
    r"\bwhat (?:do you have|are my options|options (?:do i have|are there)|is available)\b",
    re.IGNORECASE,
)
_MORE_OPTIONS = re.compile(
    r"\b(?:more|next|other|additional)\s+(?:options?|times?|openings?|slots?)\b|"
    r"\b(?:show|give|tell)\s+me\s+more\b",
    re.IGNORECASE,
)
_WEEK = re.compile(r"\b(next|this|the coming|the following)\s+week\b", re.IGNORECASE)
LIST_LIMIT = 6
SOONEST_DAYS = 14  # How far ahead "whatever is soonest" looks.


def more_options_requested(text: str) -> bool:
    return bool(_MORE_OPTIONS.search(text))


def _week_span(spec: str, today: date) -> tuple[date, date]:
    """The days a week phrase means on `today`'s calendar: "week:next", "week:this", or one named
    day of such a week ("wkday:next:tuesday")."""
    kind, _, rest = spec.partition(":")
    scope, _, name = rest.partition(":")
    monday = today - timedelta(days=today.weekday())
    if scope == "next":
        monday += timedelta(days=7)
    if kind == "wkday":
        day = monday + timedelta(days=WEEKDAYS.index(name))
        return day, day
    return (today if scope == "this" else monday), monday + timedelta(days=6)


_ARTICLE = re.compile(r"^(?:the|a|an|our|my)\s+", re.IGNORECASE)


def search_name(name: str) -> str:
    """A name as the directory is searched for it: "the heart center" looks for Heart Center."""
    return re.sub(r"['\u2019]s\b", "", _ARTICLE.sub("", name.strip())).strip() or name


def _generic_name(value: str | None) -> bool:
    """ "any doctor", "whichever clinic": words that stand for no name at all."""
    tokens = _name_tokens(value or "")
    return not tokens or all(token in _GENERIC_NAME for token in tokens)


CREDENTIAL_NOTICE = (
    "For your security, please don't share a PIN, password, or login code here, and I will never "
    "ask for one. Use the secure sign-in screen to sign in, then tell me what you'd like to do."
)
SIGN_IN_NOTICE = (
    "You'll need to sign in first. Please use the secure sign-in screen (I can't take a PIN or "
    "login code in this chat), and then ask me again."
)
SCOPE_NOTICE = (
    "I can only help with booking, moving, or cancelling appointments. "
    "Tell me if you'd like to do one of those."
)
TROUBLE_NOTICE = (
    "I'm having trouble reaching the scheduling system right now. Nothing was changed. "
    "Please try again in a moment."
)
GENERIC_NOTICE = (
    "I could not complete that request with the available details. Nothing was changed."
)

_PLACEHOLDER = re.compile(r"\b(None|null|undefined|NaN)\b|\{|\}|Traceback")


class HybridControllerModel:
    """Qwen reads each message into an intent; this code keeps the conversation and does the rest.

    One instance is one conversation. It plans in a fixed order: read the message, update the
    draft request, ask for anything essential that is missing, then run the tools (directory,
    appointments, availability, prepare) in order, choosing by rule among what they return. A
    reply is written from the tool results, so it can state real details and never wrong ones, and
    it never claims more than the trusted interface has reported.
    """

    name = "hybrid-controller"

    def __init__(
        self,
        extractor: StructuredIntentExtractor,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.extractor = extractor
        self.clock = clock or (lambda: datetime.now(UTC))
        self.state = ConversationState()
        self.issued: list[str] = []
        self.recoveries: list[str] = []  # Codes only, never text: how the agent got past a fault.
        self._users_seen = 0
        self._events_seen = 0
        self._awaiting_tool = False
        self._pending_pin: tuple[str, str] | None = None
        self._degraded = False  # True when the latest message was read by rules, not the model.
        self._ambiguous_date: tuple[str, DateMention] | None = None
        self._chosen_date: tuple[str, str] | None = None
        self._chosen_meridiem: str | None = None
        self._ask_part = False
        self.hints = PlannerHints()
        self.operation_vouched = False  # The graph's reading vouches for an unspoken BOOK.
        self.replacing = False  # The message drops the request and states its replacement.
        self._listing: tuple[bool, bool, str] | None = None  # (clinics, doctors, text)
        self.local_today: date | None = None
        self._reset_plan()

    def _reset_plan(self) -> None:
        self.clinic: Mapping[str, Any] | None = None
        self.doctor_id: str | None = None
        self.current: Mapping[str, Any] | None = None
        self.chosen: Mapping[str, Any] | None = None
        self.auto_pick = ""  # Why a time was chosen without being named: "earliest" or "only".
        self._probing = False
        self._replans = 0
        self._searched: dict[str, Any] = {}  # The arguments of the last availability search.
        self._diagnosis: str | None = None  # "unfiltered" or "ahead": why a search came back empty.

    @property
    def intent(self) -> PatientIntent | None:
        return self.state.draft

    # -- what a planner outside the controller may ask of it, without touching its state ----------

    def context(self) -> str:
        """The question being waited on and the request so far, as the extractor is told them."""
        return self._context()

    def pin_day(self, iso: str) -> None:
        """The day a pick named: the next merge takes it as the day, whatever else the words say
        ("I'll take Friday" after being shown next week's Friday is that Friday)."""
        self._chosen_date = ("start", iso)

    def resume_prompt(self) -> str:
        """The unanswered question again, or the prompt that fits the state."""
        return self._resume()

    @staticmethod
    def screens_secret(text: str) -> bool:
        """Whether the message holds a credential, or digits that may be one: it is never sent
        to a model."""
        return bool(
            _CREDENTIAL.search(text) or _DIGITS_ONLY.match(text) or _SPOKEN_DIGITS.match(text)
        )

    # -- the Model protocol -------------------------------------------------------------------

    def decide(self, transcript: Sequence[Message]):
        events = [m.content for m in transcript if m.role == "trusted_event"]
        users = [m.content for m in transcript if m.role == "user"]
        results = [m.content for m in transcript if m.role == "tool"]
        if len(events) > self._events_seen:
            self._events_seen = len(events)
            return self._say(self._after_event(events[-1]))
        if len(users) > self._users_seen:
            self._users_seen = len(users)
            self._awaiting_tool = False
            return self._begin_turn(str(users[-1].get("request", "")))
        if not self._awaiting_tool:
            return self._say(self.state.last_question or "How can I help with your appointments?")
        if len(results) != len(self.issued):
            raise MalformedModelResponse("a controller tool result was missing")
        self._awaiting_tool = False
        return self._consume(results[-1])

    def _emit(self, tool: str, arguments: dict[str, Any]) -> ToolAction:
        self.issued.append(tool)
        self._awaiting_tool = True
        return ToolAction(tool, arguments)

    @staticmethod
    def _say(text: str) -> Say:
        """Every reply passes here: an internal placeholder never reaches the patient."""
        if not text.strip() or _PLACEHOLDER.search(text):
            text = GENERIC_NOTICE
        return Say(frozenset(), text)

    def _today(self) -> date:
        return self.clock().astimezone(UTC).date()

    # -- a new message ------------------------------------------------------------------------

    def _begin_turn(self, text: str):
        state = self.state
        text = text.strip()
        self._listing = None
        state.turns += 1
        if state.turns > MAX_TURNS_PER_SESSION:
            state.reset()
            state.turns = 1
            self._reset_plan()
        if not text:
            return self._ask_again("Sorry, I didn't get a message.")
        if _CREDENTIAL.search(text) or _DIGITS_ONLY.match(text) or _SPOKEN_DIGITS.match(text):
            return self._say(CREDENTIAL_NOTICE)  # Never read, stored, or sent to the model.
        # A draft the graph already settled from what was offered ("book it", said of the one
        # opening listed) is a request to act on, not a bare "yes".
        settled = (
            state.draft is not None
            and state.prepared is None
            and state.awaiting == Awaiting.NOTHING
            and self.operation_vouched
        )
        if _AGREEMENT.match(text) and not settled:
            return self._agreement()
        if (
            more_options_requested(text)
            and state.draft is not None
            and state.awaiting in (Awaiting.DATE, Awaiting.TIME)
            and state.offered
        ):
            state.offer_offset += len(state.offered)
            state.prepared = None
            self._reset_plan()
            return self._plan()
        if (
            _ABANDON.search(text)
            and len(text.split()) <= 8
            and self._in_progress()
            and not self.replacing
            and not withdrawal_adds(text)  # "scratch that, tomorrow at 11" drops nothing.
        ):
            return self._abandon()
        if listing := self._directory_question(text):
            return self._list_directory(*listing, text)
        intent = self._read(text)  # The model reads the patient's own words, names included.
        if not self._has_content(intent) and not self._schedule_words(text):
            if intent.message_type == "ABANDON":
                return self._abandon()
            if intent.message_type == "AGREEMENT":
                return self._agreement()
            if intent.message_type == "OFF_TOPIC":
                return self._off_topic()
            if self._degraded:
                return self._ask_again("Sorry, I didn't quite catch that.")
        return self._continue(intent, text)

    def _directory_question(self, text: str) -> tuple[bool, bool] | None:
        """(clinics, doctors) when the patient asks what the directory holds: "which clinics are
        available?", "who are the doctors?", "any clinic is fine". Read from the words, so it is
        the same with or without the model, and it never ends the request in progress."""
        clinics = bool(_CLINIC_LIST.search(text))
        doctors = bool(_DOCTOR_LIST.search(text))
        awaiting = self.state.awaiting
        asked_what = bool(_WHAT_IS_THERE.search(text))
        shrug = bool(_NO_PREFERENCE.search(text)) and len(text.split()) <= 6 and "?" not in text
        if asked_what or shrug:  # Asked of a clinic question, it is about clinics.
            clinics = clinics or awaiting in (Awaiting.CLINIC, Awaiting.CLINIC_CHOICE)
        if asked_what:
            doctors = doctors or awaiting == Awaiting.DOCTOR
        return (clinics, doctors) if clinics or doctors else None

    def _list_directory(self, clinics: bool, doctors: bool, text: str):
        """Ask the directory what it holds. The request in progress is kept as it was."""
        state = self.state
        draft = state.draft
        if self._schedule_words(text) and state.prepared is None:
            # "Which clinics have mornings on Monday?" also says when: keep that for the booking.
            said = self._read(text)
            if (draft is None and said.operation == "BOOK") or (
                draft is not None and draft.operation == "BOOK"
            ):
                self._merge(
                    said.model_copy(update={"clinic_name": None, "doctor_name": None}), text
                )
        self._listing = (clinics, doctors, text)
        return self._emit("search_clinic_directory", {"browse": True})

    def _after_listing(self, result: Mapping[str, Any]):
        assert self._listing is not None
        clinics, doctors, text = self._listing
        self._listing = None
        data = result.get("data") or {}
        shown = []
        if not result.get("ok"):
            shown.append("I couldn't look up the list just now.")
        else:
            found = list(data.get("clinics") or [])
            people = list(data.get("doctors") or [])
            draft = self.state.draft
            lowered = text.casefold()
            wanted = _text(draft.clinic_name) if draft is not None else ""
            scope = [c for c in found if _text(c.get("name")) in lowered] or [
                c for c in found if wanted and wanted == _text(c.get("name"))
            ]
            if scope and doctors:
                ids = {c.get("clinic_id") for c in scope}
                people = [
                    d
                    for d in people
                    if any(c.get("clinic_id") in ids for c in d.get("clinics", []))
                ]
            more = (
                " There are more; ask me about a particular one." if data.get("next_cursor") else ""
            )
            if clinics or not doctors:
                shown.append(self._clinic_list(found) + (more if not doctors else ""))
            if doctors:
                at = f" at {_names([str(c['name']) for c in scope])}" if scope else ""
                shown.append(self._doctor_list(people, at) + more)
        shown.append(self._resume())
        return self._say(" ".join(shown))

    @staticmethod
    def _clinic_list(clinics: list[Mapping[str, Any]]) -> str:
        if not clinics:
            return "I don't see any clinics listed right now."
        items = [
            f"{c['name']} in {c['city']}" if c.get("city") else str(c["name"])
            for c in clinics[:LIST_LIMIT]
        ]
        extra = f", and {len(clinics) - LIST_LIMIT} more" if len(clinics) > LIST_LIMIT else ""
        return f"These clinics are available: {', '.join(items)}{extra}."

    @staticmethod
    def _doctor_list(doctors: list[Mapping[str, Any]], where: str) -> str:
        if not doctors:
            return f"I don't see any doctors listed{where} right now."
        items = []
        for d in doctors[:LIST_LIMIT]:
            places = _names([str(c["name"]) for c in d.get("clinics", [])])
            entry = (
                f"{d['display_name']} ({d['specialty']})"
                if d.get("specialty")
                else str(d["display_name"])
            )
            items.append(entry if where or not places else f"{entry} at {places}")
        extra = f", and {len(doctors) - LIST_LIMIT} more" if len(doctors) > LIST_LIMIT else ""
        return f"These doctors are available{where}: {', '.join(items)}{extra}."

    def _resume(self) -> str:
        """Back to what was being asked, unchanged: the list was an aside."""
        state = self.state
        if state.prepared is not None:
            return (
                "Your earlier review is still waiting: press Confirm in the app, or tell me what "
                "you'd like to change."
            )
        if state.awaiting != Awaiting.NOTHING and state.last_question:
            question = state.last_question
            for ending in ("Which clinic would you like?", "Which date would you like?"):
                if question.endswith(ending):
                    return ending  # Without the hint and the summary the patient has just heard.
            return question
        if state.draft is not None and (missing := self._missing(state.draft)):
            return self._ask(*missing).text
        return "Would you like to book, move, or cancel an appointment?"

    @staticmethod
    def _schedule_words(text: str) -> bool:
        """Whether the message states a date, a clock time, or morning or afternoon."""
        parts = parts_of_day(text) & {"morning", "afternoon", "evening", "night"}
        spoken = normalize_spoken(text)
        return bool(
            date_mentions(spoken, datetime.now(UTC).date()) or clock_mentions(spoken) or parts
        )

    def _in_progress(self) -> bool:
        return self.state.draft is not None or self.state.prepared is not None

    @staticmethod
    def _has_content(intent: PatientIntent) -> bool:
        return bool(
            intent.operation != "UNKNOWN"
            or any(
                getattr(intent, field) is not None
                for field in (
                    "clinic_name",
                    "doctor_name",
                    "specialty",
                    "start_date",
                    "end_date",
                    "new_date",
                    "time",
                    "appointment_time",
                    "option_number",
                )
            )
            or intent.time_preference != "ANYTIME"
            or intent.selection != "UNSPECIFIED"
        )

    # How the intent is read, in order, never more than two model calls for one message:
    # 1) the strict reading; 2) if that was rejected, what the message supports of the same reply;
    # 3) if there was nothing to salvage, one more strict reading with a repair note;
    # 4) rules over the message itself. A provider fault skips straight to the rules.
    def _read(self, text: str) -> PatientIntent:
        context = self._context()
        today = self._today()
        anchor = self.state.draft.start_date if self.state.draft else None
        self._degraded = False
        for repair in (False, True):
            try:
                intent = self.extractor.extract(
                    text, context=context, today=today, anchor=anchor, repair=repair
                )
                if repair:
                    self.recoveries.append("repaired_reply")
                return intent
            except MalformedModelResponse:
                salvaged = self.extractor.salvage(text, today, anchor)
                if salvaged is not None:
                    self.recoveries.append("salvaged_reply")
                    return salvaged
            except Exception:  # A timeout or an unreachable model server: do not wait or retry.
                break
        self.recoveries.append("rule_based_reading")
        self._degraded = True
        return self._rule_based(text)

    def _rule_based(self, text: str) -> PatientIntent:
        """What the message plainly says, read without the model: a safe, partial intent."""
        found = {name for name, pattern in _OPERATION_WORDS.items() if pattern.search(text)}
        if "CANCEL" in found:
            operation = "CANCEL"
        elif "RESCHEDULE" in found:
            operation = "RESCHEDULE"
        elif "BOOK" in found:
            operation = "BOOK"
        else:
            operation = "UNKNOWN"
        update: dict[str, Any] = {"operation": operation}
        awaiting = self.state.awaiting
        short_answer = len(text.split()) <= 8 and "?" not in text
        if awaiting in (Awaiting.CLINIC, Awaiting.CLINIC_CHOICE) and short_answer:
            update["clinic_name"] = text.strip(" .!?")  # The directory search checks it.
        if awaiting == Awaiting.DOCTOR and short_answer:
            update["doctor_name"] = text.strip(" .!?")
        if awaiting == Awaiting.SPECIALTY and short_answer and not _NO_PREFERENCE.search(text):
            update["specialty"] = text.strip(" .!?")
        if ordinal := _ORDINALS.search(text):
            word = ordinal.group(1).casefold()
            digits = re.search(r"\d", word)
            update["option_number"] = int(digits[0]) if digits else _ORDINAL_VALUE.get(word)
        return PatientIntent(**update)

    def _context(self) -> str:
        state = self.state
        parts = []
        if state.last_question and state.awaiting != Awaiting.NOTHING:
            options = (
                f" (options: {'; '.join(o.label for o in state.offered)})" if state.offered else ""
            )
            parts.append(f'the assistant just asked: "{state.last_question}"{options}.')
        elif state.prepared:
            parts.append("the assistant prepared a change and asked the patient to confirm it.")
        draft = state.draft
        if draft is not None:
            known = {
                key: value
                for key, value in draft.model_dump(mode="json").items()
                if value not in (None, "", "ANYTIME", "UNSPECIFIED", "REQUEST")
            }
            parts.append(f"The request so far: {json.dumps(known)}.")
        return " ".join(parts)

    # -- what a message does to the conversation -------------------------------------------------

    def _agreement(self):
        state = self.state
        if state.prepared:
            return self._say(
                "I can't confirm for you. Please review the details and press Confirm in the "
                "app, or tell me what you'd like to change."
            )
        return self._ask_again("I haven't prepared anything yet.")

    def _abandon(self):
        had_prepared = self.state.prepared is not None
        self.state.reset()
        self._reset_plan()
        note = " Please don't confirm the earlier review in the app." if had_prepared else ""
        return self._say(f"Okay, I've dropped that request and nothing was changed.{note}")

    def _off_topic(self):
        if self.hints.keep_goal:
            return self._say(f"{SCOPE_NOTICE} {self._resume()}")
        dropped = self._in_progress()
        self.state.reset()
        self._reset_plan()
        extra = (
            " I've set aside your earlier request; tell me again if you still want it."
            if dropped
            else ""
        )
        return self._say(SCOPE_NOTICE + extra)

    def _ask_again(self, lead: str):
        question = (
            self.state.last_question or "Would you like to book, move, or cancel an appointment?"
        )
        return self._say(f"{lead} {question}")

    def _continue(self, intent: PatientIntent, text: str):
        state = self.state
        before = state.draft.model_copy() if state.draft is not None else None
        had_prepared = state.prepared
        self._merge(intent, text)
        state.offer_offset = 0
        if had_prepared and state.draft == before:
            state.prepared = had_prepared
            return self._say(
                "I've already prepared that. Please review the details and press Confirm in the "
                "app, or tell me what you'd like to change."
            )
        state.prepared = None  # Anything new makes the earlier preparation out of date.
        if state.draft is not None:
            acts = self.operation_vouched or operation_stated(state.draft.operation, text)
            if acts and not state.draft.vouched:
                state.draft = state.draft.model_copy(update={"vouched": True})
        self._reset_plan()  # Fresh tool calls: nothing found earlier is reused.
        if self._pending_pin is not None:
            kind, key = self._pending_pin
            setattr(state, kind, key)
        self._pending_pin = None
        return self._plan()

    def _merge(self, delta: PatientIntent, text: str) -> None:
        """Fold what the latest message says into the draft request. The latest message wins."""
        state = self.state
        draft: PatientIntent | None = state.draft
        # An operation needs words in the message that mean it; otherwise it is not a change.
        operation = delta.operation
        if (
            operation != "UNKNOWN"
            and not self.operation_vouched
            and not _OPERATION_WORDS[operation].search(text)
        ):
            self.recoveries.append("operation_without_words")
            operation = "UNKNOWN"
        hints = self.hints
        if (
            hints.same_request
            and draft is not None
            and draft.operation == "BOOK"
            and operation == "RESCHEDULE"
        ):
            operation = (
                "UNKNOWN"  # "Make it Thursday instead" corrects the booking; it moves nothing.
            )
        carrying = bool(
            draft is None
            and (hints.carry_clinic or hints.carry_date or hints.carry_window or hints.carry_doctor)
        )
        if carrying and operation == "UNKNOWN" and self._schedule_words(text):
            operation = "BOOK"  # "I'll take 2 PM", said right after being shown the openings.
        if (
            hints.start_booking
            and operation == "UNKNOWN"
            and (draft is None or draft.operation == "UNKNOWN")
            and (self._schedule_words(text) or delta.clinic_name)
        ):
            operation = "BOOK"  # "Pine Clinic Monday 9:30": an appointment asked for, unsaid.
        fresh = draft is None or (
            operation != "UNKNOWN" and draft.operation not in ("UNKNOWN", operation)
        )
        if fresh:
            draft = Draft(operation=operation)
            if carrying and operation == "BOOK":
                draft = draft.model_copy(
                    update={
                        "clinic_name": hints.carry_clinic,
                        "doctor_name": hints.carry_doctor,
                        "start_date": hints.carry_date,
                        "end_date": hints.carry_date,
                    }
                )
            state.offered, state.awaiting = [], Awaiting.NOTHING
            state.pin_appointment = state.pin_slot = None
        elif draft.operation == "UNKNOWN" and operation != "UNKNOWN":
            draft = draft.model_copy(update={"operation": operation})
        assert draft is not None
        update: dict[str, Any] = {}
        self._pending_pin = None
        self._resolve_choice(delta, draft, update)
        role = delta.doctor_name if _role_phrase(delta.doctor_name) else None
        for field in ("clinic_name", "doctor_name", "specialty", "time", "appointment_time"):
            value = getattr(delta, field)
            if field == "doctor_name" and role:
                continue  # A kind of doctor, not a doctor: handled as the specialty below.
            if value and not (field in ("clinic_name", "doctor_name") and _generic_name(value)):
                update[field] = value  # "any doctor" names nobody.
        if role and not update.get("specialty"):
            update["specialty"] = role
        if not update.get("doctor_name") and (
            _ANY_DOCTOR.search(text)
            or (
                state.awaiting == Awaiting.DOCTOR
                and _NO_PREFERENCE.search(text)
                and len(text.split()) <= 6
            )
        ):
            update["doctor_name"] = None  # Any doctor will do: stop looking for the named one.
        if (
            state.awaiting == Awaiting.SPECIALTY
            and not update.get("specialty")
            and (
                _ANY_DOCTOR.search(text) or (_NO_PREFERENCE.search(text) and len(text.split()) <= 6)
            )
        ):
            update["specialty"] = (
                None  # The patient chose to drop the filter they were asked about.
            )
        # A clock time answers the question that was asked: "the 10 AM one" to "which time?" is a
        # time, and to "which appointment?" it names an appointment.
        if state.awaiting == Awaiting.TIME and delta.appointment_time and not delta.time:
            update["time"], update["appointment_time"] = delta.appointment_time, None
            update["selection"] = "SPECIFIC_TIME"
        elif state.awaiting == Awaiting.APPOINTMENT and delta.time and not delta.appointment_time:
            update["appointment_time"], update["time"] = delta.time, draft.time
        changed_clinic = update.get("clinic_name") and _text(update["clinic_name"]) != _text(
            draft.clinic_name
        )
        if changed_clinic and not delta.doctor_name:
            update["doctor_name"] = None  # A doctor belongs to one clinic.
        if (
            update.get("specialty")
            and not delta.doctor_name
            and _text(update["specialty"]) != _text(draft.specialty)
        ):
            update["doctor_name"] = None  # "Actually the cardiologist" replaces the doctor named.
        self._ambiguous_date = None
        self._ask_part = False
        # Dates and times are read from the words with the names removed (a clinic called "Tomorrow
        # Clinic" is not a date) and with spoken numbers written as digits.
        scan = normalize_spoken(
            _mask_names(
                text, delta.clinic_name, delta.doctor_name, draft.clinic_name, draft.doctor_name
            )
        )
        self._merge_dates(delta, scan, draft, update, fresh)
        self._merge_times(delta, scan, draft, update)
        if delta.time_preference != "ANYTIME":
            update["time_preference"] = delta.time_preference
        selection = delta.selection
        if selection in _DELEGATION and not _DELEGATION[selection].search(scan):
            selection = "UNSPECIFIED"  # "Monday morning" leaves the time to the patient.
        if selection == "SPECIFIC_TIME" and not delta.time and delta.appointment_time:
            pass  # The clock time names an existing appointment, not the time wanted.
        elif selection != "UNSPECIFIED":
            update["selection"] = selection
        elif update.get("time"):
            update["selection"] = "SPECIFIC_TIME"
        elif _SELECTION_WORDS["EARLIEST"].search(scan) and (
            update.get("not_before") is not None
            or update.get("not_after") is not None
            or str(update.get("pending_clock") or "").startswith(("after", "before"))
        ):
            # "The first available appointment after ten": kept while "which ten?" is asked.
            update["selection"] = "EARLIEST"
        state.draft = draft.model_copy(update=update)

    def _date_role(self, text: str, position: int, draft: Draft, fresh: bool) -> str:
        """Whether a date in the message is the day of the appointment or the day to move it to."""
        if draft.operation != "RESCHEDULE":
            return "start"
        before = text[:position]
        if _DESTINATION.search(before):
            return "new"  # "move it to November 20"
        if (
            not fresh
            and draft.start_date is not None
            and self.state.awaiting != Awaiting.APPOINTMENT
            and not _SOURCE.search(before)
        ):
            return "new"  # The existing day is known: this says where to.
        return "start"

    def _merge_dates(
        self, delta: PatientIntent, text: str, draft: Draft, update: dict, fresh: bool
    ) -> None:
        """The dates the message states. Its own words decide which date is which; a date that is
        said relative to today keeps how it was said, so it can be read again on the clinic's
        calendar day; a phrase with two reasonable meanings is asked about, not resolved."""
        found = date_mentions(text, self._today(), draft.start_date)
        spec_of = {m.iso: m.spec for m in found if m.iso}
        start, end, new = delta.start_date, delta.end_date, delta.new_date
        chosen, self._chosen_date = self._chosen_date, None
        chosen = chosen or self._offered_day(found)
        week = _WEEK.search(text) if draft.operation == "BOOK" and not chosen else None
        if week:
            named = [m.spec.rpartition(":")[2] for m in found if m.spec.startswith("rel:")]
            if not found or (len(found) == 1 and named and named[0] in WEEKDAYS):
                scope = "this" if week.group(1).casefold() == "this" else "next"
                spec = f"wkday:{scope}:{named[0]}" if named else f"week:{scope}"
                first, last = _week_span(spec, self._today())
                update.update(start_date=first, end_date=last, start_spec=spec)
                return
        mention: dict[str, DateMention | None] = {"start": None, "new": None}
        if chosen:
            mention[chosen[0]] = DateMention(0, chosen[1])
            start = end = new = None
        elif draft.operation == "RESCHEDULE" and found:
            end = start = new = None
            if len(found) >= 2:
                first, last = found[0], found[-1]
                if _SOURCE.search(text[: last.position]):
                    first, last = last, first  # "to the 18th from the 16th"
                mention["start"], mention["new"] = first, last
            else:
                mention[self._date_role(text, found[0].position, draft, fresh)] = found[0]
        else:
            if new is not None:  # Only a move has a second date; for anything else it is the date.
                start, new = start or new, None
            if start is None and len(found) == 1:
                mention["start"] = found[0]  # One date stated: it is the one meant.
        for field, item in mention.items():
            if item is None:
                continue
            if not item.iso:
                update[f"{field}_date"] = None
                if field == "start":
                    update["end_date"] = None
                self._ambiguous_date = (field, item)
                continue
            if field == "start":
                start = date.fromisoformat(item.iso)
                spec_of[item.iso] = item.spec
            else:
                new = date.fromisoformat(item.iso)
                spec_of[item.iso] = item.spec
        if start is not None:
            update["start_date"] = start
            update["end_date"] = end if end and end >= start else start
            update["start_spec"] = spec_of.get(start.isoformat(), "")
        if new is not None:
            update["new_date"] = new
            update["new_spec"] = spec_of.get(new.isoformat(), "")

    def _offered_day(self, found: list[DateMention]) -> tuple[str, str] | None:
        """A weekday said to "which day?" (or after being shown a week) means the day in what was
        offered or shown that falls on it."""
        if len(found) != 1:
            return None
        name = found[0].spec.rpartition(":")[2]
        if not found[0].spec.startswith("rel:") or name not in WEEKDAYS:
            return None
        if self.state.awaiting == Awaiting.DATE:
            days = [
                date.fromisoformat(o.key.partition(":")[2])
                for o in self.state.offered
                if o.key.startswith("start:")
            ]
        elif self.hints.carry_window and self.state.draft is None:
            first, last = self.hints.carry_window
            days = [first + timedelta(days=n) for n in range((last - first).days + 1)]
        else:
            return None
        same = [day.isoformat() for day in days if day.weekday() == WEEKDAYS.index(name)]
        return ("start", same[0]) if len(same) == 1 else None

    def _merge_times(self, delta: PatientIntent, text: str, draft: Draft, update: dict) -> None:
        """The clock times and parts of the day the message states, read from its own words.

        A time with no am or pm is settled by a morning or afternoon word in the same message, or
        else asked about. The model's own reading of a time is used only where it is supported."""
        parts = parts_of_day(text)
        mentions = clock_mentions(text)
        if draft.pending_clock and not mentions:
            meridiem = self._chosen_meridiem or _meridiem_in(text, parts)
            self._chosen_meridiem = None
            if meridiem:
                role, hour, minute = draft.pending_clock.split(":")
                self._apply_clock(
                    role,
                    (int(hour) % 12 + (12 if meridiem == "pm" else 0)) * 60 + int(minute),
                    update,
                )
        modelled = bool(delta.time or delta.appointment_time)
        at_mentions = [m for m in mentions if m.kind == "at"]
        for mention in mentions:
            if mention.kind == "at" and (modelled or len(at_mentions) != 1):
                continue
            role = self._clock_role(text, mention, draft)
            if role is None:
                continue
            minutes = minutes_of_mention(mention, parts)
            if minutes is None and role == "time":
                minutes = self._offered_clock(mention)
            if minutes is None:
                update["pending_clock"] = f"{role}:{mention.hour}:{mention.minute:02d}"
                break
            self._apply_clock(role, minutes, update)
        applied_time = bool(update.get("time"))
        morning_afternoon = parts & {"morning", "afternoon"}
        if (
            delta.time_preference == "ANYTIME"
            and len(morning_afternoon) == 1
            and not applied_time
            and not update.get("not_before")
            and not update.get("not_after")
        ):
            update["time_preference"] = morning_afternoon.pop().upper()
        if parts & {"evening", "night"} and not mentions:
            self._ask_part = True

    def _offered_clock(self, mention: ClockMention) -> int | None:
        """A bare "9:30" said to "which time?" is the one offered time that reads that way."""
        if self.state.awaiting != Awaiting.TIME:
            return None
        wanted = (mention.hour % 12) * 60 + mention.minute
        offered = set()
        for offer in self.state.offered:
            said = _OFFERED_CLOCK.match(offer.label)
            if said:
                hour = int(said[1]) % 12 + (12 if said[3] == "PM" else 0)
                offered.add(hour * 60 + int(said[2]))
        fits = {m for m in offered if m % 720 == wanted}
        return fits.pop() if len(fits) == 1 else None

    def _clock_role(self, text: str, mention: ClockMention, draft: Draft) -> str | None:
        """What a clock time in the message is for, or None when the words do not say."""
        if mention.kind in ("after", "before"):
            return mention.kind
        if draft.operation == "BOOK" or self.state.awaiting == Awaiting.TIME:
            return "time"
        if draft.operation == "CANCEL":
            return "appt"
        if draft.operation == "RESCHEDULE" and (
            _TO_TIME.search(text[: mention.start]) or _MOVE_TO.search(text[: mention.start])
        ):
            return "time"  # "move it to the 12th at 2": where it goes, and when.
        return None

    @staticmethod
    def _apply_clock(role: str, minutes: int, update: dict) -> None:
        update["pending_clock"] = None
        if role == "time":
            update["time"] = label(minutes)
        elif role == "appt":
            update["appointment_time"] = label(minutes)
        elif role == "after":
            update["not_before"] = minutes
        else:
            update["not_after"] = minutes

    def _resolve_choice(self, delta: PatientIntent, draft: PatientIntent, update: dict) -> None:
        """ "The second one" means the second thing the agent offered, if it offered anything."""
        state = self.state
        number = delta.option_number
        if number is None or not 1 <= number <= len(state.offered):
            return
        offer = state.offered[number - 1]
        if state.awaiting == Awaiting.DATE and offer.key.partition(":")[0] in ("start", "new"):
            field, _, iso = offer.key.partition(":")
            self._chosen_date = (field, iso)
        elif state.awaiting == Awaiting.TIME and offer.key in ("am", "pm"):
            self._chosen_meridiem = offer.key
        elif state.awaiting == Awaiting.CLINIC_CHOICE:
            update["clinic_name"] = offer.key
        elif state.awaiting == Awaiting.DOCTOR:
            update["doctor_name"] = offer.key
        elif state.awaiting == Awaiting.APPOINTMENT:
            self._pending_pin = ("pin_appointment", offer.key)
        elif state.awaiting == Awaiting.TIME:
            self._pending_pin = ("pin_slot", offer.key)

    # -- planning ------------------------------------------------------------------------------

    def _ask(self, awaiting: Awaiting, question: str, offers: Sequence[Offer] = ()):
        state = self.state
        state.awaiting, state.last_question, state.offered = awaiting, question, list(offers)
        return self._say(question)

    def _plan(self):
        draft = self.state.draft
        assert draft is not None
        missing = self._missing(draft)
        if missing:
            return self._ask(*missing)
        self.state.awaiting = Awaiting.NOTHING
        if draft.operation == "BOOK":
            arguments: dict[str, Any] = {"clinic_name": search_name(draft.clinic_name)}
            if draft.doctor_name:
                arguments["doctor_name"] = draft.doctor_name
            return self._emit("search_clinic_directory", arguments)
        return self._appointments()

    def _missing(
        self, draft: Draft
    ) -> tuple[Awaiting, str] | tuple[Awaiting, str, list[Offer]] | None:
        """The first thing that must be known before any tool is called, as a question."""
        if draft.operation == "UNKNOWN":
            return (
                Awaiting.OPERATION,
                "Would you like to book, move, or cancel an appointment?",
            )
        if self._ambiguous_date is not None:
            return self._date_question(*self._ambiguous_date)
        if draft.operation == "BOOK":
            if not draft.clinic_name:
                return (
                    Awaiting.CLINIC,
                    f"{self._heard(draft)}You can ask me which clinics are available. "
                    "Which clinic would you like?",
                )
            if not draft.start_date and draft.selection == "EARLIEST":
                # "Whatever is soonest": the first opening in the next two weeks, said so in the
                # review. The days are read again on the clinic's calendar once it is known.
                today = self._today()
                self.state.draft = draft = draft.model_copy(
                    update={
                        "start_date": today,
                        "end_date": today + timedelta(days=SOONEST_DAYS - 1),
                        "start_spec": f"soon:{SOONEST_DAYS}",
                    }
                )
            if not draft.start_date:
                return (
                    Awaiting.DATE,
                    f"{self._heard(draft)}A day like 'tomorrow' or 'Tuesday', or 'next week', "
                    "works. Which date would you like?",
                )
        if draft.pending_clock and not self._clock_checkable(draft):
            return self._clock_question(draft.pending_clock)
        if self._ask_part:
            return (
                Awaiting.TIME,
                "I can look for mornings, afternoons, or a particular time. Which would you like?",
            )
        if draft.selection == "SPECIFIC_TIME" and minutes_of(draft.time) is None:
            return Awaiting.TIME, "What time would you like? For example, 10:30 AM."
        if (
            draft.operation == "RESCHEDULE"
            and not (draft.start_date or draft.new_date or draft.clinic_name or draft.doctor_name)
            and not draft.appointment_time
            and draft.selection == "UNSPECIFIED"
            and draft.time_preference == "ANYTIME"
        ):
            return Awaiting.APPOINTMENT, "Which appointment would you like to move, and to when?"
        return None

    @staticmethod
    def _heard(draft: Draft) -> str:
        """What the request holds so far, as a sentence to put before the next question."""
        known = []
        if draft.clinic_name:
            known.append(f"at {draft.clinic_name}")
        if draft.doctor_name:
            known.append(f"with {draft.doctor_name}")
        elif draft.specialty and draft.operation == "BOOK":
            known.append(f"for {draft.specialty}")
        relative_words = _relative_date_words(draft.start_spec)
        if relative_words:
            known.append(f"for {relative_words}")
        elif draft.start_date:
            last = draft.end_date or draft.start_date
            first = _day_label(draft.start_date.isoformat())
            known.append(
                f"on {first}"
                if last == draft.start_date
                else f"from {first} to {_day_label(last.isoformat())}"
            )
        if draft.time:
            known.append(f"at {draft.time}")
        elif draft.time_preference != "ANYTIME":
            known.append(f"in the {draft.time_preference.lower()}")
        # What the patient is asking for, never a statement that an appointment exists.
        kind = {
            "BOOK": "a request to book an appointment",
            "RESCHEDULE": "a request to move an appointment",
            "CANCEL": "a request to cancel an appointment",
        }.get(str(draft.operation), "a request for an appointment")
        return f"So far I have {kind} {' '.join(known)}. " if known else ""

    @staticmethod
    def _clock_checkable(draft: Draft) -> bool:
        """A time said without am or pm can be settled by the schedule itself, once there is one to
        look at: if only one of the two readings is open, that is the one meant."""
        role = str(draft.pending_clock).partition(":")[0]
        return role == "time" and (
            draft.operation == "RESCHEDULE" or bool(draft.clinic_name and draft.start_date)
        )

    def _settle_pending_clock(self, draft: Draft, slots: list[Mapping[str, Any]]):
        """Use the open slots to read a bare clock time: None if settled, else the question."""
        _, hour, minute = str(draft.pending_clock).split(":")
        wanted = (int(hour) % 12) * 60 + int(minute)
        fits = sorted({_minutes(str(s["starts_at_local"])) for s in slots} & {wanted, wanted + 720})
        if len(fits) == 1:
            self.auto_pick = "clock"
            self.state.draft = draft.model_copy(
                update={"time": label(fits[0]), "selection": "SPECIFIC_TIME", "pending_clock": None}
            )
            return None
        return self._ask(*self._clock_question(str(draft.pending_clock)))

    @staticmethod
    def _date_question(field: str, mention: DateMention) -> tuple[Awaiting, str, list[Offer]]:
        offers = [Offer(_day_label(iso), f"{field}:{iso}") for iso in mention.options]
        return (
            Awaiting.DATE,
            f"Do you mean {_names([o.label for o in offers])}?",
            offers,
        )

    @staticmethod
    def _clock_question(pending: str) -> tuple[Awaiting, str, list[Offer]]:
        role, hour, minute = pending.split(":")
        clock = f"{int(hour)}:{minute}"
        offers = [Offer(f"{clock} AM", "am"), Offer(f"{clock} PM", "pm")]
        lead = {"after": "after ", "before": "before "}.get(role, "")
        return (
            Awaiting.TIME,
            f"Do you mean {lead}{offers[0].label} or {lead}{offers[1].label}?",
            offers,
        )

    def _appointments(self):
        draft = self.state.draft
        assert draft is not None
        arguments: dict[str, Any] = {"status": "CONFIRMED"}
        if draft.start_date:
            first, last = draft.start_date, draft.end_date or draft.start_date
            if draft.start_spec:
                # Said relative to today, with the clinic not yet known: look a day or two either
                # side, then keep what falls on the day it means at each appointment's clinic.
                first, last = first - timedelta(days=2), last + timedelta(days=2)
            arguments["start_date"] = first.isoformat()
            arguments["end_date"] = last.isoformat()
        return self._emit("get_my_appointments", arguments)

    def _today_in(self, timezone: object) -> date:
        """Today on a clinic's own calendar. Without a usable time zone, the provisional day."""
        try:
            return self.clock().astimezone(ZoneInfo(str(timezone))).date()
        except Exception:
            return self._today()

    def _localize(self, timezone: object):
        """Read the dates the patient said relative to today again on the clinic's calendar day.

        "Tomorrow" and "November 1" mean the clinic's tomorrow and the clinic's next November 1,
        not UTC's. If that makes a phrase ambiguous, the patient is asked. Returns a question to
        say, or None."""
        draft = self.state.draft
        assert draft is not None
        today = self._today_in(timezone)
        self.local_today = today
        update: dict[str, Any] = {}
        for spec_field, date_field in (("start_spec", "start_date"), ("new_spec", "new_date")):
            spec, current = getattr(draft, spec_field), getattr(draft, date_field)
            if not spec:
                continue
            if spec.startswith("soon:"):
                first, last = today, today + timedelta(days=int(spec[5:]) - 1)
                if date_field == "start_date" and (current, draft.end_date) != (first, last):
                    update.update(start_date=first, end_date=last)
                continue
            if spec.startswith(("week:", "wkday:")):
                first, last = _week_span(spec, today)
                if date_field == "start_date" and (current, draft.end_date) != (first, last):
                    update.update(start_date=first, end_date=last)
                continue
            resolved = resolve_spec(spec, today)
            if len(resolved) == 2:
                field = date_field.removesuffix("_date")
                update.update({date_field: None, spec_field: ""})
                if field == "start":
                    update["end_date"] = None
                self.state.draft = draft.model_copy(update=update)
                return self._ask(*self._date_question(field, DateMention(0, "", spec, resolved)))
            if len(resolved) == 1 and (current is None or resolved[0] != current.isoformat()):
                moved = date.fromisoformat(resolved[0])
                update[date_field] = moved
                if date_field == "start_date" and draft.end_date in (None, current):
                    update["end_date"] = moved
        if update:
            self.state.draft = draft.model_copy(update=update)
        return None

    # -- after each tool ----------------------------------------------------------------------

    def _consume(self, result: Mapping[str, Any]):
        tool = self.issued[-1]
        if self._listing is not None and tool == "search_clinic_directory":
            return self._after_listing(result)
        if self._diagnosis is not None and tool == "search_appointment_availability":
            return self._diagnose(result)
        if not result.get("ok"):
            return self._failed(tool, result)
        data = result.get("data") or {}
        if tool == "search_clinic_directory":
            return self._after_directory(data)
        if tool == "get_my_appointments":
            return self._after_appointments(data)
        if tool == "search_appointment_availability":
            return self._after_availability(data)
        if tool == "prepare_appointment_action":
            return self._after_prepare(data)
        raise MalformedModelResponse("the controller reached an unknown state")

    def _after_directory(self, data: Mapping[str, Any]):
        draft = self.state.draft
        assert draft is not None
        if self._probing:
            stated = _name_tokens(draft.doctor_name or "")
            owner = [
                c
                for c in data.get("clinics") or []
                if stated and all(word in _name_tokens(str(c.get("name"))) for word in stated)
            ]
            if len(owner) == 1:
                # The "doctor" is the rest of the clinic's own name, split off by mistake.
                self._probing = False
                self.state.draft = draft = draft.model_copy(
                    update={"clinic_name": str(owner[0]["name"]), "doctor_name": None}
                )
                data = {"clinics": owner, "doctors": []}
        if self._probing:
            return self._ask(
                Awaiting.DOCTOR,
                f"I couldn't find a doctor called {draft.doctor_name} at {draft.clinic_name}. "
                "Which doctor would you like?",
            )
        clinics = list(data.get("clinics") or [])
        wanted = _text(draft.clinic_name)
        exact = [c for c in clinics if _text(c.get("name")) == wanted]
        pool = exact if len(exact) == 1 else clinics
        if len(pool) != 1:
            offers = [Offer(str(c["name"]), str(c["name"])) for c in clinics[:5]]
            return self._ask(
                Awaiting.CLINIC_CHOICE,
                f"Which clinic do you mean: {_names([o.label for o in offers])}?",
                offers,
            )
        self.clinic = pool[0]
        if draft.doctor_name:
            doctors = self._matching_doctors(list(data.get("doctors") or []), draft.doctor_name)
            if len(doctors) != 1:
                offers = [
                    Offer(str(d["display_name"]), str(d["display_name"])) for d in doctors[:5]
                ]
                if not offers:
                    return self._ask(
                        Awaiting.DOCTOR,
                        f"I couldn't find a doctor called {draft.doctor_name} at "
                        f"{self.clinic['name']}. Which doctor would you like?",
                    )
                return self._ask(
                    Awaiting.DOCTOR,
                    f"Which doctor do you mean: {_names([o.label for o in offers])}?",
                    offers,
                )
            self.doctor_id = str(doctors[0]["doctor_id"])
            if draft.specialty:
                # A real doctor has been chosen by name. A specialty asked for earlier no longer
                # narrows the search, and the doctor is named in everything said back. (A name that
                # matches nobody chooses no one, so it never removes the specialty.)
                self.state.draft = draft = draft.model_copy(update={"specialty": None})
        asked = self._localize(self.clinic.get("timezone"))
        if asked is not None:
            return asked
        draft = self.state.draft
        assert draft is not None and draft.start_date is not None
        return self._availability(str(self.clinic["clinic_id"]), draft.start_date, draft.end_date)

    @staticmethod
    def _matching_doctors(doctors: list[Mapping[str, Any]], stated: str) -> list[Mapping[str, Any]]:
        """Doctors whose name contains every word the patient said ("Rao", "Dr. Anita Rao"),
        judged from the directory's own records. An exact full name settles several matches."""
        wanted = _name_tokens(stated)
        found = [
            d for d in doctors if all(w in _name_tokens(str(d.get("display_name"))) for w in wanted)
        ]
        exact = [d for d in found if _name_tokens(str(d.get("display_name"))) == wanted]
        return exact if len(exact) == 1 else found

    def _availability(self, clinic_id: str, day: date, end: date | None = None):
        draft = self.state.draft
        assert draft is not None
        today = self.local_today or self._today()
        moving = draft.operation == "RESCHEDULE" and draft.new_date == day
        if day < today:
            cleared = (
                {"new_date": None, "new_spec": ""}
                if moving
                else {"start_date": None, "end_date": None, "start_spec": ""}
            )
            self.state.draft = draft.model_copy(update=cleared)
            return self._ask(
                Awaiting.DATE,
                f"{_day_label(day.isoformat())} has already passed on the clinic's calendar. "
                "Which date would you like?",
            )
        if ((end or day) - day).days + 1 > MAX_SEARCH_DAYS:
            self.state.draft = draft.model_copy(
                update={"start_date": None, "end_date": None, "start_spec": ""}
            )
            return self._ask(
                Awaiting.DATE,
                f"I can search up to {MAX_SEARCH_DAYS} days at a time. Which dates would you like?",
            )
        arguments: dict[str, Any] = {
            "clinic_id": clinic_id,
            "start_date": day.isoformat(),
            "end_date": (end or day).isoformat(),
        }
        if self.doctor_id:
            arguments["doctor_id"] = self.doctor_id
        if draft.specialty and draft.operation == "BOOK":
            arguments["specialty"] = draft.specialty
        if draft.time_preference != "ANYTIME":
            arguments["time_preference"] = draft.time_preference
        self._searched = dict(arguments)
        return self._emit("search_appointment_availability", arguments)

    def _matching_appointments(self, data: Mapping[str, Any], draft: Draft, identifying):
        """The patient's appointments that fit what was said, or the question to ask when a date
        phrase has two readings."""
        found = []
        for item in data.get("appointments", []):
            if draft.start_spec:
                targets = resolve_spec(
                    draft.start_spec, self._today_in((item.get("clinic") or {}).get("timezone"))
                )
                if len(targets) == 2:
                    mention = DateMention(0, "", draft.start_spec, targets)
                    return self._ask(*self._date_question("start", mention))
                if str(item["starts_at_local"])[:10] not in targets:
                    continue
            clinic_tokens = _name_tokens(str((item.get("clinic") or {}).get("name")))
            doctor_tokens = _name_tokens(str((item.get("doctor") or {}).get("display_name")))
            if draft.clinic_name and not all(
                word in clinic_tokens for word in _name_tokens(draft.clinic_name)
            ):
                continue
            if draft.doctor_name and not all(
                w in doctor_tokens for w in _name_tokens(draft.doctor_name)
            ):
                continue
            if identifying is not None and _minutes(str(item["starts_at_local"])) != identifying:
                continue
            found.append(item)
        return found

    def _after_appointments(self, data: Mapping[str, Any]):
        draft = self.state.draft
        assert draft is not None
        identifying = minutes_of(draft.appointment_time) if draft.appointment_time else None
        if identifying is None and draft.operation == "CANCEL":
            identifying = minutes_of(draft.time) if draft.time else None
        found = self._matching_appointments(data, draft, identifying)
        if not found and identifying is not None and draft.operation == "RESCHEDULE":
            # The clock time names none of their appointments, so it is the time they want.
            found = self._matching_appointments(data, draft, None)
            if found:
                moved = {"appointment_time": None}
                if not draft.time:
                    moved |= {"time": draft.appointment_time, "selection": "SPECIFIC_TIME"}
                self.state.draft = draft = draft.model_copy(update=moved)
        if isinstance(found, Say):
            return found
        if self.state.pin_appointment is not None:
            same = [i for i in found if str(i.get("appointment_id")) == self.state.pin_appointment]
            found = same or found
            if not same:
                self.state.pin_appointment = None
        if self._degraded and self.state.pin_appointment is None and found:
            # The message was read by rules, not by the model, so the target is never guessed:
            # even one match is put to the patient, who names it.
            offers = [Offer(self._describe(i), str(i["appointment_id"])) for i in found[:5]]
            return self._ask(
                Awaiting.APPOINTMENT,
                "I'm not certain I understood, so I won't guess. You can say the time or 'the "
                f"first one'. Which appointment do you mean: {_names([o.label for o in offers])}?",
                offers,
            )
        if not found:
            where = f" at {draft.clinic_name}" if draft.clinic_name else ""
            when = f" on {_day_label(draft.start_date.isoformat())}" if draft.start_date else ""
            return self._ask(
                Awaiting.APPOINTMENT,
                f"I couldn't find an active appointment{where}{when}. "
                "Could you tell me the clinic or date of the appointment?",
            )
        if len(found) > 1:
            offers = [
                Offer(self._describe(item), str(item["appointment_id"])) for item in found[:5]
            ]
            return self._ask(
                Awaiting.APPOINTMENT,
                f"Which appointment do you mean: {_names([o.label for o in offers])}?",
                offers,
            )
        self.current = found[0]
        appointment_id = str(self.current["appointment_id"])
        if draft.operation == "CANCEL":
            self.chosen = self.current
            return self._emit(
                "prepare_appointment_action",
                {"action": "CANCEL", "appointment_id": appointment_id},
            )
        held = date.fromisoformat(str(self.current["starts_at_local"])[:10])
        if draft.start_date is None or draft.start_spec:
            # The appointment found fixes its own day; a phrase said relative to today is done.
            self.state.draft = draft.model_copy(
                update={"start_date": held, "end_date": held, "start_spec": ""}
            )
        clinic = self.current.get("clinic") or {}
        asked = self._localize(clinic.get("timezone"))
        if asked is not None:
            return asked
        draft = self.state.draft
        assert draft is not None
        return self._availability(
            str(clinic["clinic_id"]), draft.new_date or draft.start_date or held
        )

    def _after_availability(self, data: Mapping[str, Any]):
        draft = self.state.draft
        assert draft is not None
        slots = sorted(data.get("slots") or [], key=lambda slot: str(slot["starts_at_utc"]))
        if self.current is not None:
            now = str(self.current["starts_at_local"])
            if draft.selection == "LATER":
                slots = [s for s in slots if _before(now, s)]
            elif draft.selection == "EARLIER":
                slots = [s for s in slots if _before(s, now)]
        if draft.pending_clock and str(draft.pending_clock).startswith("time:"):
            question = self._settle_pending_clock(draft, slots)
            if question is not None:
                return question
            draft = self.state.draft
            assert draft is not None
        before_window = slots
        if draft.not_before is not None:
            slots = [s for s in slots if _minutes(str(s["starts_at_local"])) >= draft.not_before]
        if draft.not_after is not None:
            slots = [s for s in slots if _minutes(str(s["starts_at_local"])) <= draft.not_after]
        if not slots and before_window:
            offers = self._slot_offers(before_window)
            return self._ask(
                Awaiting.TIME,
                f"I don't see an opening {_window_words(draft)}. These are open: "
                f"{_names([o.label for o in offers])}. Which would you like?",
                offers,
            )
        pick = self._pick(slots)
        if isinstance(pick, Say):
            return pick
        self.chosen = pick
        if draft.operation == "BOOK":
            arguments = {"action": "BOOK", "slot_id": str(pick["slot_id"])}
        else:
            assert self.current is not None
            arguments = {
                "action": "RESCHEDULE",
                "appointment_id": str(self.current["appointment_id"]),
                "new_slot_id": str(pick["slot_id"]),
            }
        return self._emit("prepare_appointment_action", arguments)

    def _slot_offers(self, slots: Sequence[Mapping[str, Any]]) -> list[Offer]:
        offset = self.state.offer_offset
        shown = slots[offset : offset + SPOKEN_OPTION_LIMIT]
        offers = []
        for slot in shown:
            local = str(slot["starts_at_local"])
            label = f"{_clock(local)} on {_day_label(local)}"
            doctor = str((slot.get("doctor") or {}).get("display_name", ""))
            if doctor:
                label += f" with {doctor}"
            offers.append(Offer(label, str(slot["slot_id"])))
        if len({o.label for o in offers}) < len(offers):
            # The same clock time twice in one day: the hour the clocks go back. Tell them apart.
            offers = [
                Offer(
                    o.label.replace(
                        _clock(str(slot["starts_at_local"])),
                        f"{_clock(str(slot['starts_at_local']))} "
                        f"(UTC{str(slot['starts_at_local'])[-6:]})",
                        1,
                    ),
                    o.key,
                )
                for o, slot in zip(offers, shown, strict=False)
            ]
        return offers

    def _pick(self, slots: list[Mapping[str, Any]]) -> Mapping[str, Any] | Say:
        """The slot the draft selects, or the question or sentence to say instead."""
        draft = self.state.draft
        assert draft is not None
        if not slots:
            where = (self.clinic or (self.current or {}).get("clinic") or {}).get(
                "name", "that clinic"
            )
            when = _when(draft.start_date, draft.end_date)
            if draft.operation == "RESCHEDULE" and draft.new_date:
                when = _when(draft.new_date)
            return self._ask(
                Awaiting.DATE,
                f"I don't see any matching openings at {where}{when}. "
                "Would another day or time of day work?",
            )
        if self.state.pin_slot is not None:
            chosen = [s for s in slots if str(s["slot_id"]) == self.state.pin_slot]
            if chosen:
                return chosen[0]
            self.state.pin_slot = None
        selection = draft.selection
        offers = self._slot_offers(slots)
        if selection == "SPECIFIC_TIME":
            wanted = minutes_of(draft.time)
            matches = [s for s in slots if _minutes(str(s["starts_at_local"])) == wanted]
            if len(matches) == 1:
                return matches[0]
            shown = _names([o.label for o in offers])
            if matches:
                offers = self._slot_offers(matches)
                return self._ask(
                    Awaiting.TIME,
                    f"Which would you like: {_names([o.label for o in offers])}?",
                    offers,
                )
            return self._ask(
                Awaiting.TIME,
                f"I don't see an opening at {label(wanted) if wanted is not None else draft.time}. "
                f"These are open: {shown}. "
                "Which would you like?",
                offers,
            )
        if selection == "EARLIEST":
            # An explicit delegation counts only when it resolves to exactly one opening: two
            # doctors free at the same earliest time is a question, not a choice made for them.
            first = min(str(s["starts_at_utc"]) for s in slots)
            earliest = [s for s in slots if str(s["starts_at_utc"]) == first]
            if len(earliest) == 1:
                self.auto_pick = "earliest"  # The patient left the time to us, and is told so.
                return earliest[0]
            offers = self._slot_offers(earliest)
            return self._ask(
                Awaiting.TIME,
                f"{_names([o.label for o in offers])} are all the earliest. Which would you like?",
                offers,
            )
        if len(slots) == 1:
            self.auto_pick = "only"
            return slots[0]
        # A time of day or a stretch of days is a preference, not a time: show what is open.
        days = sorted({str(s["starts_at_local"])[:10] for s in slots})
        if len(days) > 1 and draft.operation == "BOOK":
            offset = self.state.offer_offset
            shown_days = days[offset : offset + SPOKEN_OPTION_LIMIT]
            day_offers = [Offer(_day_label(day), f"start:{day}") for day in shown_days]
            if not day_offers:
                self.state.offer_offset = max(0, offset - SPOKEN_OPTION_LIMIT)
                return self._ask(
                    Awaiting.DATE,
                    "Those are all the matching days. You can name another day or change the time.",
                    self.state.offered,
                )
            more = (
                " Ask for more options or name another day."
                if offset + len(day_offers) < len(days)
                else ""
            )
            return self._ask(
                Awaiting.DATE,
                f"I have {self._preference(draft)}openings at {self._clinic_name()} on "
                f"{numbered_options(day_offers)}.{more} Which day would you like?",
                day_offers,
            )
        if not offers:
            self.state.offer_offset = max(0, self.state.offer_offset - SPOKEN_OPTION_LIMIT)
            return self._ask(
                Awaiting.TIME,
                "Those are all the matching times. You can name a different time or another day.",
                self.state.offered,
            )
        more = (
            " Ask for more options, name a different time, or try another day."
            if self.state.offer_offset + len(offers) < len(slots)
            else ""
        )
        spoken = compact_numbered_options(offers)
        separator = " " if spoken.startswith("on ") else ": "
        return self._ask(
            Awaiting.TIME,
            f"I have {self._preference(draft)}openings at {self._clinic_name()}"
            f"{separator}{spoken}."
            f"{more} Which time would you like?",
            offers,
        )

    def _clinic_name(self) -> str:
        clinic = self.clinic or (self.current or {}).get("clinic") or {}
        return str(clinic.get("name", "that clinic"))

    @staticmethod
    def _preference(draft: Draft) -> str:
        """ "morning " for a morning preference; empty when none was stated."""
        return "" if draft.time_preference == "ANYTIME" else f"{draft.time_preference.lower()} "

    def _after_prepare(self, data: Mapping[str, Any]):
        self.state.awaiting = Awaiting.NOTHING
        text = self._review_text()
        self.state.last_question = ""
        self.state.offered = []
        self.state.prepared = Prepared(str(data.get("pending_action_id", "")), text)
        return self._say(text)

    # -- what it says -------------------------------------------------------------------------

    @staticmethod
    def _describe(item: Mapping[str, Any]) -> str:
        local = str(item["starts_at_local"])
        doctor = (item.get("doctor") or {}).get("display_name", "")
        clinic = (item.get("clinic") or {}).get("name", "")
        return f"{_clock(local)} on {_day_label(local)} with {doctor} at {clinic}"

    def _review_text(self) -> str:
        draft = self.state.draft
        assert draft is not None and self.chosen is not None
        target = self._describe(self.chosen)
        if draft.operation == "BOOK":
            what = f"booking {target}"
        elif draft.operation == "CANCEL":
            what = f"cancelling your appointment, {target}"
        else:
            assert self.current is not None
            local = str(self.current["starts_at_local"])
            what = (
                f"moving your appointment from {_clock(local)} on {_day_label(local)} to {target}"
            )
        why = {
            "earliest": " That is the earliest opening that fits what you asked for.",
            "only": " That is the only opening that fits what you asked for.",
            "clock": " That is the only opening at that time, morning or evening.",
        }.get(self.auto_pick, "")
        return f"I prepared {what}.{why} Please review the details and confirm in the app."

    def _after_event(self, event: Mapping[str, Any]) -> str:
        """The trusted UI's report. Only a reported commit is ever described as done."""
        state, draft, chosen = self.state, self.state.draft, self.chosen
        prepared = state.prepared
        reported = str(event.get("pending_action_id") or "")
        matches = prepared is not None and (not reported or reported == prepared.pending_action_id)
        committed = event.get("type") == "APPOINTMENT_COMMITTED"
        state.reset()
        self._reset_plan()
        if not committed:
            return "Sorry, that is no longer possible, so nothing was changed."
        if draft is None or chosen is None or not matches:
            word = {"BOOKED": "booked", "CANCELLED": "cancelled", "RESCHEDULED": "rescheduled"}.get(
                str(event.get("result")), "changed"
            )
            return f"Your appointment was {word}, as confirmed in the app."
        target = self._describe(chosen)
        word = {"BOOK": "booked", "CANCEL": "cancelled", "RESCHEDULE": "rescheduled"}[
            draft.operation
        ]
        if draft.operation == "RESCHEDULE":
            return f"Your appointment is rescheduled to {target}."
        return f"Your appointment, {target}, is {word}."

    # -- an empty search, explained --------------------------------------------------------------

    def _no_openings_facts(self, draft: Draft):
        """Where, when, and which of the patient's filters narrowed the empty search."""
        searched = self._searched
        where = (self.clinic or (self.current or {}).get("clinic") or {}).get("name", "that clinic")
        when = _when(draft.new_date) if draft.new_date else _when(draft.start_date, draft.end_date)
        specialty = draft.specialty if "specialty" in searched else None
        part_of_day = draft.time_preference.lower() if "time_preference" in searched else None
        doctor = (draft.doctor_name or "that doctor") if "doctor_id" in searched else None
        return (
            where,
            when,
            {
                "specialty": specialty,
                "part_of_day": part_of_day,
                "doctor": doctor,
            },
        )

    def _explain_no_openings(self, draft: Draft):
        """No openings: say which of three things that means. A filter is never dropped to make
        the search succeed; the patient is told what excluded the openings and chooses."""
        if "start_date" not in self._searched:
            where, when, _ = self._no_openings_facts(draft)
            return self._ask(
                Awaiting.DATE, f"I don't see any openings at {where}{when}. Would another day work?"
            )
        where, when, filters = self._no_openings_facts(draft)
        specialty = filters["specialty"]
        if specialty and self.clinic is not None:
            listed = specialties_listed(self.clinic)
            if listed and not any(same_specialty(specialty, name) for name in listed):
                return self._ask(Awaiting.SPECIALTY, no_such_specialty(where, specialty, listed))
        if any(name in self._searched for name in FILTER_ARGUMENTS):
            self._diagnosis = "unfiltered"  # What does the clinic have open, whatever was asked?
            everything = {k: v for k, v in self._searched.items() if k not in FILTER_ARGUMENTS}
            return self._emit("search_appointment_availability", everything)
        return self._look_ahead()

    def _look_ahead(self):
        """Nothing is open then: search the next days from the clinic's today (or from the day
        asked for, if that is further off than one search can span), with the filters kept."""
        self._diagnosis = "ahead"
        wanted = date.fromisoformat(str(self._searched["start_date"]))
        today = self.local_today or self._today()
        first = today if 0 <= (wanted - today).days < LOOK_AHEAD_DAYS else wanted
        last = first + timedelta(days=LOOK_AHEAD_DAYS - 1)
        return self._emit(
            "search_appointment_availability",
            {**self._searched, "start_date": first.isoformat(), "end_date": last.isoformat()},
        )

    def _diagnose(self, result: Mapping[str, Any]):
        stage, self._diagnosis = self._diagnosis, None
        draft = self.state.draft
        assert draft is not None
        code = (result.get("error") or {}).get("code")
        if not result.get("ok") and code != "NO_AVAILABILITY":
            return self._failed("search_appointment_availability", result)
        slots = list((result.get("data") or {}).get("slots") or []) if result.get("ok") else []
        where, when, filters = self._no_openings_facts(draft)
        if stage == "unfiltered":
            if not slots:
                return self._look_ahead()  # Nothing open at all that day.
            wanted = filters["specialty"]
            awaiting = (
                Awaiting.SPECIALTY
                if wanted
                else Awaiting.DOCTOR
                if filters["doctor"] and not filters["part_of_day"]
                else Awaiting.DATE
            )
            return self._ask(
                awaiting,
                filtered_out(where, when, **filters, open_specialties=specialties_open(slots)),
            )
        if slots:
            return self._ask(Awaiting.DATE, later_days(where, when, next_days(slots), **filters))
        return self._ask(
            Awaiting.SPECIALTY if filters["specialty"] else Awaiting.CLINIC,
            nothing_ahead(where, when, **filters),
        )

    def _failed(self, tool: str, result: Mapping[str, Any]):
        error = result.get("error") or {}
        code = error.get("code")
        draft = self.state.draft
        if code == "AMBIGUOUS_MATCH":
            choices = list(error.get("choices") or [])[:5]
            is_clinic = bool(choices) and "clinic_id" in choices[0]
            offers = [
                Offer(
                    str(c.get("name") or c.get("display_name")),
                    str(c.get("name") or c.get("display_name")),
                )
                for c in choices
            ]
            return self._ask(
                Awaiting.CLINIC_CHOICE if is_clinic else Awaiting.DOCTOR,
                f"Which {'clinic' if is_clinic else 'doctor'} do you mean: "
                f"{_names([o.label for o in offers])}?",
                offers,
            )
        if code == "NO_RESULTS" and tool == "search_clinic_directory" and draft is not None:
            if draft.doctor_name and not self._probing:
                self._probing = True
                return self._emit(
                    "search_clinic_directory", {"clinic_name": search_name(draft.clinic_name)}
                )
            self._probing = False
            named = f" called {draft.clinic_name}" if draft.clinic_name else ""
            self.state.draft = draft.model_copy(update={"clinic_name": None, "doctor_name": None})
            return self._ask(
                Awaiting.CLINIC, f"I couldn't find a clinic{named}. Which clinic would you like?"
            )
        if code == "NO_AVAILABILITY" and draft is not None:
            return self._explain_no_openings(draft)
        if code in ("AUTH_REQUIRED", "SESSION_EXPIRED", "FORBIDDEN"):
            self.state.prepared = None
            return self._say(SIGN_IN_NOTICE)
        if code in ("SLOT_UNAVAILABLE", "CONFLICT") and tool == "prepare_appointment_action":
            if self._replans < 1 and self.state.draft is not None:
                self._replans += 1  # Look again once, with fresh availability.
                self.chosen = None
                self.state.pin_slot = None
                return self._plan()
            return self._say(
                "That time was just taken, so nothing was prepared. Want me to look again?"
            )
        if code == "SELF_SERVICE_CUTOFF":
            return self._say(
                "That appointment is too close to its start time to change here, so nothing was "
                "changed. Please contact the clinic directly."
            )
        if code == "RESOURCE_NOT_AVAILABLE":
            return self._say(
                "I couldn't find that appointment or time any more, so nothing was changed."
            )
        if code in ("DEPENDENCY_UNAVAILABLE", "RATE_LIMITED", "DATABASE_UNAVAILABLE"):
            return self._say(TROUBLE_NOTICE)
        return self._say(GENERIC_NOTICE)
