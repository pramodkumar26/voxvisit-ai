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
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, ValidationError

from app.evaluation.conversation import (
    MAX_TURNS_PER_SESSION,
    Awaiting,
    ConversationState,
    Offer,
    Prepared,
)
from app.evaluation.llm import CallRecord, LlmConfig, LlmUsage, Transport, http_transport
from app.evaluation.model import MalformedModelResponse, Message, Say, ToolAction
from app.evaluation.spoken import (
    MONTHS,
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


class PatientIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    cleaned = re.sub(r"[^\w\s']", " ", value.casefold())
    return [token for token in cleaned.split() if token not in {"dr", "doctor"}]


def _name_supported(value: str, request: str) -> bool:
    """Whether the message really contains this name (every word of it, in any case)."""
    if value.casefold() in request.casefold():
        return True
    wanted, stated = _name_tokens(value), set(_name_tokens(request))
    return bool(wanted) and all(token in stated for token in wanted)


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
        for field in ("clinic_name", "doctor_name", "specialty"):
            value = getattr(intent, field)
            if value and not _name_supported(value, request):
                replaced[field] = None
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
        body = body or {}
        duration = (self._timer() - started) * 1000
        server_ns = body.get("total_duration")
        self.usage.calls.append(
            CallRecord(
                duration_ms=duration,
                outcome=outcome,
                prompt_tokens=body.get("prompt_eval_count")
                if isinstance(body.get("prompt_eval_count"), int)
                else None,
                output_tokens=body.get("eval_count")
                if isinstance(body.get("eval_count"), int)
                else None,
                server_ms=server_ns / 1e6 if isinstance(server_ns, int) else None,
            )
        )

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


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + f", or {items[-1]}"


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
        r"\b(cancel|call off|can'?t make|cannot make|won'?t be able|unable to (?:make|attend))\b",
        re.IGNORECASE,
    ),
    "RESCHEDULE": re.compile(
        r"\b(reschedul\w*|mov(?:e|ing)|chang(?:e|ing)|push|switch|shift|postpone|"
        r"another time|different time|later time|earlier time|instead)\b",
        re.IGNORECASE,
    ),
}
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
        self.local_today: date | None = None
        self._reset_plan()

    def _reset_plan(self) -> None:
        self.clinic: Mapping[str, Any] | None = None
        self.doctor_id: str | None = None
        self.current: Mapping[str, Any] | None = None
        self.chosen: Mapping[str, Any] | None = None
        self._probing = False
        self._replans = 0

    @property
    def intent(self) -> PatientIntent | None:
        return self.state.draft

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
        state.turns += 1
        if state.turns > MAX_TURNS_PER_SESSION:
            state.reset()
            state.turns = 1
            self._reset_plan()
        if not text:
            return self._ask_again("Sorry, I didn't get a message.")
        if _CREDENTIAL.search(text) or _DIGITS_ONLY.match(text) or _SPOKEN_DIGITS.match(text):
            return self._say(CREDENTIAL_NOTICE)  # Never read, stored, or sent to the model.
        if _AGREEMENT.match(text):
            return self._agreement()
        if _ABANDON.search(text) and len(text.split()) <= 8 and self._in_progress():
            return self._abandon()
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
        if had_prepared and state.draft == before:
            state.prepared = had_prepared
            return self._say(
                "I've already prepared that. Please review the details and press Confirm in the "
                "app, or tell me what you'd like to change."
            )
        state.prepared = None  # Anything new makes the earlier preparation out of date.
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
        if operation != "UNKNOWN" and not _OPERATION_WORDS[operation].search(text):
            self.recoveries.append("operation_without_words")
            operation = "UNKNOWN"
        fresh = draft is None or (
            operation != "UNKNOWN" and draft.operation not in ("UNKNOWN", operation)
        )
        if fresh:
            draft = Draft(operation=operation)
            state.offered, state.awaiting = [], Awaiting.NOTHING
            state.pin_appointment = state.pin_slot = None
        elif draft.operation == "UNKNOWN" and operation != "UNKNOWN":
            draft = draft.model_copy(update={"operation": operation})
        assert draft is not None
        update: dict[str, Any] = {}
        self._pending_pin = None
        self._resolve_choice(delta, draft, update)
        for field in ("clinic_name", "doctor_name", "specialty", "time", "appointment_time"):
            if getattr(delta, field):
                update[field] = getattr(delta, field)
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
        if delta.selection == "SPECIFIC_TIME" and not delta.time and delta.appointment_time:
            pass  # The clock time names an existing appointment, not the time wanted.
        elif delta.selection != "UNSPECIFIED":
            update["selection"] = delta.selection
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

    def _clock_role(self, text: str, mention: ClockMention, draft: Draft) -> str | None:
        """What a clock time in the message is for, or None when the words do not say."""
        if mention.kind in ("after", "before"):
            return mention.kind
        if draft.operation == "BOOK" or self.state.awaiting == Awaiting.TIME:
            return "time"
        if draft.operation == "CANCEL":
            return "appt"
        if draft.operation == "RESCHEDULE" and _TO_TIME.search(text[: mention.start]):
            return "time"
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
            arguments: dict[str, Any] = {"clinic_name": draft.clinic_name}
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
                return Awaiting.CLINIC, "Which clinic would you like?"
            if not draft.start_date:
                return Awaiting.DATE, "Which date would you like?"
        if draft.pending_clock:
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
        return self._emit("search_appointment_availability", arguments)

    def _after_appointments(self, data: Mapping[str, Any]):
        draft = self.state.draft
        assert draft is not None
        identifying = minutes_of(draft.appointment_time) if draft.appointment_time else None
        if identifying is None and draft.operation == "CANCEL":
            identifying = minutes_of(draft.time) if draft.time else None
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
        days = {str(s["starts_at_local"])[:10] for s in slots}
        doctors = {str((s.get("doctor") or {}).get("doctor_id")) for s in slots}
        offers = []
        for slot in slots[:5]:
            label = _clock(str(slot["starts_at_local"]))
            if len(days) > 1:
                label = f"{label} on {_day_label(str(slot['starts_at_local']))}"
            if len(doctors) > 1:
                label += f" with {(slot.get('doctor') or {}).get('display_name', '')}"
            offers.append(Offer(label, str(slot["slot_id"])))
        if len({o.label for o in offers}) < len(offers):
            # The same clock time twice in one day: the hour the clocks go back. Tell them apart.
            offers = [
                Offer(f"{o.label} (UTC{str(slot['starts_at_local'])[-6:]})", o.key)
                for o, slot in zip(offers, slots, strict=False)
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
            when = f" on {_day_label(draft.start_date.isoformat())}" if draft.start_date else ""
            if draft.operation == "RESCHEDULE" and draft.new_date:
                when = f" on {_day_label(draft.new_date.isoformat())}"
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
        if selection == "UNSPECIFIED" and draft.time_preference != "ANYTIME":
            selection = "ANY"
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
        if selection in ("EARLIEST", "ANY", "LATER", "EARLIER") or len(slots) == 1:
            return slots[0]
        return self._ask(
            Awaiting.TIME,
            f"Which time would you like: {_names([o.label for o in offers])}?",
            offers,
        )

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
        return f"I prepared {what}. Please review the details and confirm in the app."

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
                return self._emit("search_clinic_directory", {"clinic_name": draft.clinic_name})
            self._probing = False
            named = f" called {draft.clinic_name}" if draft.clinic_name else ""
            self.state.draft = draft.model_copy(update={"clinic_name": None, "doctor_name": None})
            return self._ask(
                Awaiting.CLINIC, f"I couldn't find a clinic{named}. Which clinic would you like?"
            )
        if code == "NO_AVAILABILITY" and draft is not None:
            where = (self.clinic or (self.current or {}).get("clinic") or {}).get(
                "name", "that clinic"
            )
            when = draft.new_date or draft.start_date
            on = f" on {_day_label(when.isoformat())}" if when else ""
            return self._ask(
                Awaiting.DATE,
                f"I don't see any openings at {where}{on}. Would another day work?",
            )
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
