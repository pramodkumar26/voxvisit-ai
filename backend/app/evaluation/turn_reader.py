"""LG-3: one model call per message that both routes it and reads the request out of it.

LG-2 spent two calls on a scheduling message: the planner decided what the message was, then the
intent extractor read it. `TurnReader` asks for both in one structured reply (`TurnReading`), so a
scheduling turn costs one call. Whether that is also as accurate is measured (see the LG-4 report),
not assumed: the split path stays available beside it.

The reading is a proposal. Every value in it is checked against the patient's own words exactly as
before (`StructuredIntentExtractor.unsupported` for the request, `planner.ground` for lookups), the
model names no tool and no identifier, and any failure returns None so the caller falls back to the
two-call path.
"""

import contextlib
import time
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.evaluation.hybrid import PatientIntent, StructuredIntentExtractor
from app.evaluation.llm import LlmConfig, LlmUsage, Transport, http_transport, record_of
from app.evaluation.offers import options_from, resolve_offer
from app.evaluation.planner import PlannerDecision, ground

TURN_PROMPT_VERSION = "lg5d-t1"

_COMPACT = """
Write only the fields that apply to this message. Leave out every field that is null, empty or
at its default: do not write null, [] or "NONE"."""

_INSTRUCTIONS = """\
You read ONE message from a patient of a clinic appointment assistant. Return JSON matching the
schema. Today is {today}.

route (what the message is):
- SCHEDULING: starts, continues, answers a question about, corrects or drops a request to book,
  move or cancel an appointment: also a short answer to the assistant's question ("Pine Clinic",
  "the second one", "Thursday"), a changed detail, "never mind", "yes". If the message asks a
  question AND asks to book, choose SCHEDULING. When the assistant just asked something, a message
  that answers it is SCHEDULING even with extra words ("Dr. Kim, the earliest").
- LOOKUP: asks to be told what the clinic directory or the schedule holds, with no request to book
  or to be seen: which clinics, which doctors, whether a doctor works at a clinic, whether there is
  anything open, which times are free. "Any openings Friday?" and "Is there something with a
  cardiologist on Monday?" are LOOKUP: they ask what exists. "I want something Friday" is not.
- CLARIFY: about care or appointments, but too unclear to act on, or could mean two things.
- OUT_OF_SCOPE: nothing to do with clinics, doctors or appointments.

For LOOKUP, subjects lists what is asked (CLINICS, DOCTORS, OPENINGS), one entry per question in the
message, in order. week is THIS or NEXT for "this week" or "next week". When the message asks about
a second place, doctor or day, the first goes in the fields below and each further one is an entry
of "more" with its own subject, clinic_name, doctor_name, specialty, start_date and end_date.
For CLARIFY set unclear_about (GOAL, CLINIC, DOCTOR, DATE or TIME).
same_request: true only when a request is already in progress (see the conversation so far) and
this message answers, corrects ("sorry, I meant Thursday instead") or adds to it.
urgent: true when the patient describes symptoms that may be an emergency right now (chest pain,
trouble breathing, severe bleeding, thoughts of self-harm); false otherwise.
starts_booking: true when the message asks for an appointment without saying "book" (it gives only a
clinic, a day or a time they can come) and nothing is in progress.

The request fields (fill them for SCHEDULING and LOOKUP, from this message only; null for anything
it does not say; never invent a name, date, time or preference; copy names as written):{compact}
- operation: BOOK, RESCHEDULE, CANCEL or UNKNOWN (UNKNOWN when the message only answers a question
  or changes a detail).
- clinic_name, doctor_name (without "Dr." only if written that way; every word of the name),
  specialty (a medical specialty the message names or asks for: "a cardiologist" is Cardiology).
- message_type ABANDON is only for dropping the whole request ("never mind"). Changing the doctor,
  clinic, day or time ("actually the cardiologist", "make it Thursday") is a correction: REQUEST.
- start_date, end_date: the day meant, YYYY-MM-DD ("November 17", "tomorrow", "next Monday"); both
  the same for one day. new_date: only when moving an existing appointment to a different day.
- time_preference: MORNING or AFTERNOON only if said; otherwise ANYTIME.
- time: a clock time they want, as written ("11 AM"). appointment_time: the clock time of an
  EXISTING appointment they point to ("the 10 AM one").
- option_number: when the conversation lists numbered choices the assistant offered, the number of
  the ONE choice this message picks: by position ("the last one", "the second"), by comparison
  ("the later one", "the earlier"), by description ("the afternoon one", "the heart doctor") or by
  name. null when no choices were offered, or when the message does not clearly pick exactly one.
- selection: EARLIEST for first, earliest, soonest or next available; LATER or EARLIER only for
  those words; ANY if they say any time is fine; SPECIFIC_TIME if they name a clock time; otherwise
  UNSPECIFIED.
- message_type: REQUEST for anything that states, answers or changes a request; ABANDON when they
  drop it; AGREEMENT when it is only "yes" or "okay"; OFF_TOPIC when unrelated to appointments.
{context}
Examples (made up):
{examples}
"""

_EXAMPLES = (
    (
        "Book the first appointment at Quartz Ridge Clinic on 2031-03-04.",
        {
            "route": "SCHEDULING",
            "operation": "BOOK",
            "clinic_name": "Quartz Ridge Clinic",
            "start_date": "2031-03-04",
            "end_date": "2031-03-04",
            "selection": "EARLIEST",
        },
    ),
    (
        "Do you have anything open at Birch Hill Clinic on 2031-03-04?",
        {
            "route": "LOOKUP",
            "subjects": ["OPENINGS"],
            "clinic_name": "Birch Hill Clinic",
            "start_date": "2031-03-04",
            "end_date": "2031-03-04",
        },
    ),
    (
        "Are there any openings at Birch Hill Clinic on 2031-03-05?",
        {
            "route": "LOOKUP",
            "subjects": ["OPENINGS"],
            "clinic_name": "Birch Hill Clinic",
            "start_date": "2031-03-05",
            "end_date": "2031-03-05",
        },
    ),
    (
        "Does Opal Health have anything with a dermatologist next week?",
        {
            "route": "LOOKUP",
            "subjects": ["OPENINGS"],
            "clinic_name": "Opal Health",
            "specialty": "Dermatology",
            "week": "NEXT",
        },
    ),
    (
        "Anything at Birch Hill Clinic on 2031-03-04, and what about Opal Health on 2031-03-06?",
        {
            "route": "LOOKUP",
            "subjects": ["OPENINGS"],
            "clinic_name": "Birch Hill Clinic",
            "start_date": "2031-03-04",
            "end_date": "2031-03-04",
            "more": [
                {
                    "subject": "OPENINGS",
                    "clinic_name": "Opal Health",
                    "start_date": "2031-03-06",
                    "end_date": "2031-03-06",
                }
            ],
        },
    ),
    (
        "Which doctors are at Cobalt Medical and what's free next week?",
        {
            "route": "LOOKUP",
            "subjects": ["DOCTORS", "OPENINGS"],
            "clinic_name": "Cobalt Medical",
            "week": "NEXT",
        },
    ),
    (
        "Actually, can I have the pediatrician instead?",
        {
            "route": "SCHEDULING",
            "same_request": True,
            "specialty": "Pediatrics",
            "message_type": "REQUEST",
        },
    ),
    (
        "Actually, make it the 11th instead.",
        {
            "route": "SCHEDULING",
            "same_request": True,
            "start_date": "2031-07-11",
            "end_date": "2031-07-11",
        },
    ),
    (
        "I can only do Fridays at Birch Hill Clinic.",
        {"route": "SCHEDULING", "starts_booking": True, "clinic_name": "Birch Hill Clinic"},
    ),
    (
        "Move my Opal Health appointment to 2031-07-10.",
        {
            "route": "SCHEDULING",
            "operation": "RESCHEDULE",
            "clinic_name": "Opal Health",
            "new_date": "2031-07-10",
        },
    ),
    ("Something is wrong with my arm.", {"route": "CLARIFY", "unclear_about": "GOAL"}),
    ("Never mind, forget it.", {"route": "SCHEDULING", "message_type": "ABANDON"}),
    ("I can't breathe and my chest hurts.", {"route": "OUT_OF_SCOPE", "urgent": True}),
    (
        "How much does parking cost downtown?",
        {"route": "OUT_OF_SCOPE", "message_type": "OFF_TOPIC"},
    ),
)
_DATE_KEYS = ("start_date", "end_date", "new_date")


def _example_text(*, lean: bool) -> str:
    def shown(answer: dict) -> dict:
        if not lean:
            return answer
        trimmed = {k: v for k, v in answer.items() if k not in _DATE_KEYS}
        if "more" in trimmed:
            trimmed["more"] = [
                {k: v for k, v in q.items() if k not in _DATE_KEYS} for q in trimmed["more"]
            ]
        return trimmed

    return (
        "\n".join(f"Message: {text}\n{shown(answer)}" for text, answer in _EXAMPLES)
        .replace("'", '"')
        .replace("True", "true")
    )


_EXAMPLE_TEXT = _example_text(lean=False)
_LEAN_EXAMPLE_TEXT = _example_text(lean=True)
_DATE_LINES = (
    '- start_date, end_date: the day meant, YYYY-MM-DD ("November 17", "tomorrow", "next '
    'Monday"); both\n  the same for one day. new_date: only when moving an existing appointment '
    "to a different day.\n"
)


class Question(BaseModel):
    """A further question in the message, about another place, doctor or day."""

    model_config = ConfigDict(extra="forbid")

    subject: Literal["CLINICS", "DOCTORS", "OPENINGS"] = "OPENINGS"
    clinic_name: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    start_date: date | None = None
    end_date: date | None = None


def system_prompt(
    *, today: date, context: str, layout: str, compact: bool, lean: bool = False
) -> str:
    """The instructions for one reading. In the "prefix" layout everything that never changes
    comes first and what changes with the day and the conversation comes last, so a server that
    keeps the start of the last prompt in memory only has to read the end again."""
    asked = f"\nThe conversation so far: {context}\n" if context else ""
    when = today.strftime("%Y-%m-%d (%A)")
    note = _COMPACT if compact else ""
    instructions = _INSTRUCTIONS
    if lean:
        instructions = instructions.replace(_DATE_LINES, "").replace(
            "doctor_name, specialty, start_date and end_date.", "doctor_name and specialty."
        )
    examples = _LEAN_EXAMPLE_TEXT if lean else _EXAMPLE_TEXT
    if layout == "prefix":
        fixed = instructions.replace(" Today is {today}.", "").replace("{context}\n", "")
        fixed = fixed.format(examples=examples, compact=note)
        return f"{fixed}\nToday is {when}.{asked}"
    return instructions.format(today=when, context=asked, examples=examples, compact=note)


def reading_schema(*, lean: bool) -> dict:
    """The shape the model must write. Lean leaves out the dates: the code reads every date from
    the patient's own words and uses a date from the model only if it equals one of those, so a
    date the model writes never changes an outcome and only costs time to write."""
    schema = TurnReading.model_json_schema()
    if lean:
        for key in _DATE_KEYS:
            schema["properties"].pop(key, None)
        for definition in schema.get("$defs", {}).values():
            for key in _DATE_KEYS:
                definition.get("properties", {}).pop(key, None)
    return schema


class TurnReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["SCHEDULING", "LOOKUP", "CLARIFY", "OUT_OF_SCOPE"]
    subjects: list[Literal["CLINICS", "DOCTORS", "OPENINGS"]] = []
    more: list[Question] = []
    week: Literal["NONE", "THIS", "NEXT"] = "NONE"
    unclear_about: Literal["NONE", "GOAL", "CLINIC", "DOCTOR", "DATE", "TIME"] = "NONE"
    same_request: bool = False
    starts_booking: bool = False
    urgent: bool = False
    operation: Literal["BOOK", "RESCHEDULE", "CANCEL", "UNKNOWN"] = "UNKNOWN"
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


class Read:
    """A grounded reading: what to do with the message, and the request it states.
    `option_number` is the model's proposal for which offered choice the message picks; it is not
    used until it has been checked against the choices that were actually offered."""

    def __init__(
        self,
        decision: PlannerDecision,
        intent: PatientIntent,
        option_number: int | None = None,
    ) -> None:
        self.decision, self.intent, self.option_number = decision, intent, option_number


_CHECK = """\
You decide what ONE message from a patient of a clinic appointment assistant asks the assistant to
do. Return JSON matching the schema.
asks_for:
- BOOK: wants a new appointment, however it is worded ("get seen", "pencil me in", "the soonest
  slot at ...").
- RESCHEDULE: wants an existing appointment moved to another time.
- CANCEL: wants an existing appointment cancelled or dropped ("scrap it", "I can't come").
- WITHDRAW: drops the request being arranged right now and asks for nothing else ("never mind").
- NONE: anything else: a question about clinics, doctors or open times; an answer to the
  assistant's question; a correction or a detail ("actually the cardiologist", "make it
  Thursday"); small talk.
A message that asks what is available, open or free ("any availability tomorrow?", "is Dr. Lee
around Monday?", "what's open at Pine Clinic?") asks to be told, not to be booked: NONE, even when
it mentions an appointment. "I need something Friday" and "can I get in Monday morning?" ask to be
seen: BOOK.
{context}"""


_QUESTIONS = """\
A patient of a clinic appointment assistant sent ONE message that asks several separate questions
about clinics, doctors or open times. Today is {today}. Return JSON matching the schema.
questions: one entry per separate question, in the order asked, at most three. For each: subject
(CLINICS for which clinics exist, DOCTORS for who works somewhere or whether a doctor works there,
OPENINGS for open times), clinic_name and doctor_name (copied exactly as written, null if the
question does not name one), specialty (if one is named), start_date and end_date (YYYY-MM-DD, only
if that question names a day; null otherwise). Every question carries only what it names itself.
"""


class Questions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[Question] = Field(default_factory=list, max_length=3)


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asks_for: Literal["BOOK", "RESCHEDULE", "CANCEL", "WITHDRAW", "NONE"]


_REVISE = """\
A patient of a clinic appointment assistant is in the middle of arranging an appointment and has
sent a new message. Today is {today}. Return JSON matching the schema.
The request so far: {request}
{doctors}
kind says what the new message does to that request:
- CHANGE: changes or adds a detail (another doctor, clinic, day, time or kind of doctor), or
  switches the action (move it instead of cancel it), however short or loosely worded
  ("actually Dr. Lee", "make it Friday", "3 in the afternoon", "the heart doctor", "not that one").
- QUESTION: asks what exists (who works somewhere, what is open, whether someone is in) and leaves
  the request as it is.
- NONE: neither ("thanks", small talk, "ok").
For CHANGE fill only what the message changes; for QUESTION fill what it asks about; leave
everything it does not say null. operation: BOOK, RESCHEDULE or CANCEL only when the message
switches the action. clinic_name, doctor_name and specialty: copy names as written; when the
patient describes a doctor ("the heart doctor") and the list above holds exactly one doctor that
fits, give that doctor's name from the list. start_date and end_date: the day meant, YYYY-MM-DD;
new_date only when moving an existing appointment. time_preference: MORNING or AFTERNOON only if
said. time: a clock time as written ("3 PM").
"""


class Revision(BaseModel):
    """What a message does to a request already in progress."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["CHANGE", "QUESTION", "NONE"]
    operation: Literal["BOOK", "RESCHEDULE", "CANCEL", "UNKNOWN"] = "UNKNOWN"
    clinic_name: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    new_date: date | None = None
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    time: str | None = None


class TurnReader:
    """One bounded model call per message; None when the reply cannot be used."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        transport: Transport | None = None,
        timer: Callable[[], float] = time.perf_counter,
        extra_options: Mapping[str, Any] | None = None,
        layout: str = "prefix",
        compact: bool = False,
        lean: bool = False,
    ) -> None:
        self.lean = lean
        self.layout = layout
        self.compact = compact
        self.config = config
        self._post = transport or http_transport(config.base_url)
        self._timer = timer
        self._extra = dict(extra_options or {})
        self.usage = LlmUsage()
        self.fallbacks = 0

    def read(self, text: str, *, context: str, today: date, anchor: date | None) -> Read | None:
        started = self._timer()
        body: Mapping[str, Any] = {}
        outcome = "reader_fault"
        try:
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt(
                            today=today,
                            context=context,
                            layout=self.layout,
                            compact=self.compact,
                            lean=self.lean,
                        ),
                    },
                    {"role": "user", "content": text[: self.config.max_message_chars]},
                ],
                "format": reading_schema(lean=self.lean),
                "stream": False,
                **({"think": self.config.think} if self.config.think is not None else {}),
                **({"keep_alive": self.config.keep_alive} if self.config.keep_alive else {}),
                **self._extra,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_output_tokens,
                    "num_ctx": self.config.context_tokens,
                    **({"seed": self.config.seed} if self.config.seed is not None else {}),
                },
            }
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty reply")
            reading = TurnReading.model_validate_json(content)
            outcome = "turn_reading"
            return self._grounded(reading, text, today, anchor)
        except (ValidationError, ValueError):
            outcome = "invalid_turn_reading"
            self.fallbacks += 1
            return None
        except Exception:
            self.fallbacks += 1
            return None
        finally:
            self._record(started, outcome, body)

    @staticmethod
    def _grounded(reading: TurnReading, text: str, today: date, anchor: date | None) -> Read:
        decision = PlannerDecision(
            route=reading.route,
            subject=reading.subjects[0] if reading.subjects else "NONE",
            also=reading.subjects[1:3],
            clinic_name=reading.clinic_name,
            doctor_name=reading.doctor_name,
            specialty=reading.specialty,
            start_date=reading.start_date,
            end_date=reading.end_date,
            week=reading.week,
            time_preference=reading.time_preference,
            unclear_about=reading.unclear_about,
            same_request=reading.same_request,
            starts_booking=reading.starts_booking,
            urgent=reading.urgent,
            more=[
                PlannerDecision(route="LOOKUP", **q.model_dump(exclude_none=True))
                for q in reading.more[:2]
            ],
        )
        decision = ground(decision, text, today, anchor)
        intent = PatientIntent(
            **reading.model_dump(
                include=set(PatientIntent.model_fields), exclude_none=True, mode="python"
            )
        )
        intent = intent.model_copy(
            update=StructuredIntentExtractor.unsupported(intent, text, today, anchor)
        )
        return Read(decision, intent, reading.option_number)

    def warm_up(self, today: date) -> float:
        """Load the model and read the fixed start of the prompt once, before the first patient
        message: the first real turn then pays for neither. Returns the seconds it took; a
        server that cannot be reached is not an error here (the turn will report it)."""
        started = self._timer()
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt(
                        today=today, context="", layout="prefix", compact=self.compact
                    ),
                },
                {"role": "user", "content": "hello"},
            ],
            "format": TurnReading.model_json_schema(),
            "stream": False,
            "keep_alive": self.config.keep_alive or "30m",
            "options": {"num_predict": 1, "num_ctx": self.config.context_tokens},
        }
        with contextlib.suppress(Exception):
            self._post("/api/chat", payload, self.config.request_timeout_s)
        return self._timer() - started

    def questions(self, text: str, today: date) -> list[Question] | None:
        """A message that asks several things, split into the separate questions it asks."""
        started = self._timer()
        body: Mapping[str, Any] = {}
        outcome = "questions_fault"
        try:
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": _QUESTIONS.format(today=today.strftime("%Y-%m-%d (%A)")),
                    },
                    {"role": "user", "content": text[: self.config.max_message_chars]},
                ],
                "format": Questions.model_json_schema(),
                "stream": False,
                **({"think": self.config.think} if self.config.think is not None else {}),
                **({"keep_alive": self.config.keep_alive} if self.config.keep_alive else {}),
                **self._extra,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": 260,
                    "num_ctx": self.config.context_tokens,
                    **({"seed": self.config.seed} if self.config.seed is not None else {}),
                },
            }
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty reply")
            found = Questions.model_validate_json(content).questions
            outcome = "questions"
            return found
        except Exception:
            self.fallbacks += 1
            return None
        finally:
            self._record(started, outcome, body)

    def check(self, text: str, *, in_progress: bool) -> str | None:
        """An independent second reading of what the message asks for, used only when the first
        reading named an operation (or a withdrawal) that the wording does not plainly show.
        Returns BOOK, RESCHEDULE, CANCEL, WITHDRAW or NONE, or None if the reply was unusable."""
        started = self._timer()
        body: Mapping[str, Any] = {}
        outcome = "check_fault"
        try:
            note = (
                "\nA request to book, move or cancel is being arranged with the patient now.\n"
                if in_progress
                else "\nNothing is being arranged with the patient yet.\n"
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": _CHECK.format(context=note)},
                    {"role": "user", "content": text[: self.config.max_message_chars]},
                ],
                "format": Check.model_json_schema(),
                "stream": False,
                **({"think": self.config.think} if self.config.think is not None else {}),
                **({"keep_alive": self.config.keep_alive} if self.config.keep_alive else {}),
                **self._extra,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": 40,
                    "num_ctx": self.config.context_tokens,
                    **({"seed": self.config.seed} if self.config.seed is not None else {}),
                },
            }
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty reply")
            verdict = Check.model_validate_json(content).asks_for
            outcome = "operation_check"
            return verdict
        except Exception:
            self.fallbacks += 1
            return None
        finally:
            self._record(started, outcome, body)

    def revise(
        self,
        text: str,
        *,
        request: str,
        doctors: str,
        today: date,
        anchor: date | None,
        known_doctors: Mapping[str, str] | None = None,
    ) -> tuple[str, Read] | None:
        """What a message does to the request in progress: a change, a question about what
        exists, or nothing. Used when the first reading found nothing to apply to a request that is
        being arranged. The values are held to the same evidence as any reading: names, days and
        times must be in the words, except that a doctor the patient describes may be named from
        `known_doctors` (name -> specialty), which came from the directory just now. Returns
        (kind, the grounded reading), or None when the reply was unusable."""
        started = self._timer()
        body: Mapping[str, Any] = {}
        outcome = "revise_fault"
        try:
            known = (
                f"Doctors at the clinic: {doctors}." if doctors else "The clinic is not known yet."
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": _REVISE.format(
                            today=today.strftime("%Y-%m-%d (%A)"), request=request, doctors=known
                        ),
                    },
                    {"role": "user", "content": text[: self.config.max_message_chars]},
                ],
                "format": Revision.model_json_schema(),
                "stream": False,
                **({"think": self.config.think} if self.config.think is not None else {}),
                **({"keep_alive": self.config.keep_alive} if self.config.keep_alive else {}),
                **self._extra,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": 160,
                    "num_ctx": self.config.context_tokens,
                    **({"seed": self.config.seed} if self.config.seed is not None else {}),
                },
            }
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty reply")
            revision = Revision.model_validate_json(content)
            outcome = "revision"
            return revision.kind, self._grounded_revision(
                revision, text, today, anchor, known_doctors or {}
            )
        except Exception:
            self.fallbacks += 1
            return None
        finally:
            self._record(started, outcome, body)

    @staticmethod
    def _grounded_revision(
        revision: Revision,
        text: str,
        today: date,
        anchor: date | None,
        known_doctors: Mapping[str, str],
    ) -> Read:
        intent = PatientIntent(
            **revision.model_dump(
                include=set(PatientIntent.model_fields), exclude_none=True, mode="python"
            )
        )
        replaced = StructuredIntentExtractor.unsupported(intent, text, today, anchor)
        # A doctor the patient only described ("the heart doctor") is found among the doctors the
        # directory lists, by the same deterministic rule as any reference to offers: exactly one
        # must fit. The model's own choice of name is not taken.
        if replaced.get("doctor_name", intent.doctor_name) is None and known_doctors:
            doctors = list(known_doctors)
            found = resolve_offer(
                text, options_from([f"{n} ({known_doctors[n]})" for n in doctors])
            ).number
            if found is not None:
                replaced["doctor_name"] = doctors[found - 1]
                replaced["specialty"] = None  # The doctor decides, not the kind of doctor.
        intent = intent.model_copy(update=replaced)
        decision = ground(
            PlannerDecision(
                route="SCHEDULING",
                same_request=True,
                clinic_name=intent.clinic_name,
                doctor_name=intent.doctor_name,
                specialty=intent.specialty,
                start_date=intent.start_date,
                end_date=intent.end_date,
                time_preference=intent.time_preference,
            ),
            text,
            today,
            anchor,
        )
        return Read(decision, intent)

    def _record(self, started: float, outcome: str, body: Mapping[str, Any]) -> None:
        self.usage.calls.append(record_of((self._timer() - started) * 1000, outcome, body))


class PrecomputedExtractor:
    """Stands in for the intent extractor when the reading was made in the combined call: the
    controller asks for the intent and is given the grounded one. It has no model of its own."""

    usage = None

    def __init__(self, intent: PatientIntent) -> None:
        self.intent = intent

    def extract(self, request: str, **_: Any) -> PatientIntent:
        return self.intent

    def salvage(self, *_args: Any, **_kwargs: Any) -> None:
        return None
