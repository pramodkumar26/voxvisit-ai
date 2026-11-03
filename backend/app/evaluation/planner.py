"""LG-2: goal-aware planning for the experimental graph path.

Qwen gets a bigger role than reading slots out of a message, but a bounded one. For each message
it decides, from the meaning, what the patient is doing:

- SCHEDULING: starting, continuing, answering, correcting or dropping a booking, move or
  cancellation. Handed to the existing controller, which alone plans the scheduling tools, holds
  the request, and prepares a review.
- LOOKUP: asking what the clinic directory or the schedule can answer (which clinics, which
  doctors, does a doctor work somewhere, what is open). Answered here from read-only tools.
- CLARIFY: about appointments but too unclear to act on. Asked about, from a fixed question.
- OUT_OF_SCOPE: nothing to do with clinics or appointments.

What the model may not do: write anything the patient reads (every sentence is built here from tool
results or fixed text), name a tool (the tool follows from the subject: the directory, then
availability), supply an identifier (a clinic id comes from a directory result), touch the
scheduling request, or prepare or confirm anything. A value it returns is kept only if the patient's
own message supports it, as in the intent extractor. If the planner fails for any reason, the
message goes to the controller exactly as before.

Nothing here changes the controller's state: the request in progress, the question it asked, the
choices it offered and any review awaiting Confirm are as they were after a lookup. The only state
the planner owns is a lookup left open for want of one detail ("which clinic?"), so the answer can
complete it.
"""

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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
from app.evaluation.hybrid import (
    _WEEK,
    SPOKEN_OPTION_LIMIT,
    HybridControllerModel,
    PatientIntent,
    StructuredIntentExtractor,
    _clock,
    _day_label,
    _names,
    _text,
    _week_span,
    compact_numbered_labels,
    date_mentions,
    normalize_spoken,
    resolve_spec,
    search_name,
    without_possessive,
)
from app.evaluation.llm import LlmConfig, LlmUsage, Transport, http_transport, record_of

_TITLED_NAME = re.compile(r"\bDr\.?\s+[A-Z][\w-]*(?:['\u2019](?!s\b)[\w-]+)*")

PLANNER_PROMPT_VERSION = "lg2-p1"
MAX_LOOKUP_CALLS = 7  # Read-only tool calls for one message, across all its questions.
MAX_SEARCH_DAYS = 31
LIST_LIMIT = 6

_INSTRUCTIONS = """\
You route ONE message from a patient of a clinic appointment assistant. Return JSON matching the
schema. Today is {today}.

route:
- SCHEDULING: the message starts, continues, answers a question about, corrects or drops a request
  to book, move or cancel an appointment. This includes a short answer to the assistant's question
  ("Pine Clinic", "the second one", "afternoon"), a changed detail ("actually Thursday instead"),
  "never mind", and "yes".
- LOOKUP: the patient asks to be told something the clinic directory or the schedule can answer,
  without asking to book: which clinics or locations exist, which doctors or specialties there are,
  whether a doctor works at a clinic, or which appointment times are open.
- CLARIFY: it is about care or appointments, but is too unclear to tell what the patient wants, or
  could reasonably mean two different things.
- OUT_OF_SCOPE: nothing to do with clinics, doctors or appointments (billing, medical advice,
  weather, small talk).
If the patient asks a question AND asks to book in the same message, choose SCHEDULING. If you are
unsure between SCHEDULING and LOOKUP, choose SCHEDULING. When the assistant has just asked the
patient something (see the conversation so far), a message that answers it is SCHEDULING even with
extra words ("Dr. Kim, the earliest", "the second one", "Thursday"). When the assistant has just
listed open times, a message that picks a day or a time is SCHEDULING.

For LOOKUP set subject: CLINICS (which clinics exist), DOCTORS (which doctors, or where a doctor
works), OPENINGS (which times are open). Then fill only what the message says: clinic_name,
doctor_name, specialty (names copied as written), start_date and end_date (YYYY-MM-DD), week (THIS
or NEXT for "this week" or "next week"), time_preference (MORNING or AFTERNOON). Use null, NONE or
ANYTIME for anything the message does not say. Never invent a name, date or preference.
For CLARIFY set unclear_about: GOAL (what they want), CLINIC, DOCTOR, DATE or TIME.
Set same_request true only when a request to book, move or cancel is already in progress (see the
conversation so far) and this message answers it, corrects it ("sorry, I meant Thursday instead"),
or adds a detail to it; false when it starts something different.
Set starts_booking true when a message asks for an appointment without saying "book" (for example
it only gives a clinic, a day or a time they can come) and nothing is in progress.
{context}
Examples (made up):
{examples}
"""

_EXAMPLES = (
    (
        "Do you have anything open at Birch Hill Clinic on 2031-03-04?",
        {
            "route": "LOOKUP",
            "subject": "OPENINGS",
            "clinic_name": "Birch Hill Clinic",
            "start_date": "2031-03-04",
            "end_date": "2031-03-04",
        },
    ),
    ("Where can I go in this area?", {"route": "LOOKUP", "subject": "CLINICS"}),
    (
        "Is Dr. Wells at Cobalt Medical?",
        {
            "route": "LOOKUP",
            "subject": "DOCTORS",
            "clinic_name": "Cobalt Medical",
            "doctor_name": "Wells",
        },
    ),
    ("Book me at Birch Hill Clinic on 2031-03-04.", {"route": "SCHEDULING"}),
    ("Actually, make it the afternoon.", {"route": "SCHEDULING"}),
    ("Something is wrong with my arm.", {"route": "CLARIFY", "unclear_about": "GOAL"}),
    (
        "I might want to move my visit, or maybe book a new one, not sure.",
        {"route": "CLARIFY", "unclear_about": "GOAL"},
    ),
    (
        "I can only do Fridays at Birch Hill Clinic.",
        {"route": "SCHEDULING", "starts_booking": True},
    ),
    ("No, I meant Friday instead.", {"route": "SCHEDULING", "same_request": True}),
    ("Dr. Wells, whichever is earliest.", {"route": "SCHEDULING", "same_request": True}),
    ("How much does parking cost downtown?", {"route": "OUT_OF_SCOPE"}),
)
_EXAMPLE_TEXT = "\n".join(f"Message: {text}\n{answer}" for text, answer in _EXAMPLES).replace(
    "'", '"'
)


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("clinic_name", "doctor_name", mode="after")
    @classmethod
    def _name_without_possessive(cls, value: str | None) -> str | None:
        return without_possessive(value)

    route: Literal["SCHEDULING", "LOOKUP", "CLARIFY", "OUT_OF_SCOPE"]
    subject: Literal["CLINICS", "DOCTORS", "OPENINGS", "NONE"] = "NONE"
    clinic_name: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    # A day said relative to today ("tomorrow", "Friday"), kept as the phrase's meaning so it can
    # be read again on the clinic's own calendar once the clinic is known.
    start_spec: str | None = None
    end_spec: str | None = None
    week: Literal["NONE", "THIS", "NEXT"] = "NONE"
    time_preference: Literal["ANYTIME", "MORNING", "AFTERNOON"] = "ANYTIME"
    unclear_about: Literal["NONE", "GOAL", "CLINIC", "DOCTOR", "DATE", "TIME"] = "NONE"
    same_request: bool = False
    starts_booking: bool = False
    also: list[Literal["CLINICS", "DOCTORS", "OPENINGS"]] = []  # More questions in the message.
    # More questions about another place, doctor or day, each complete in itself.
    more: list["PlannerDecision"] = []
    urgent: bool = False


PlannerDecision.model_rebuild()


HANDOFF = PlannerDecision(route="SCHEDULING")  # What happens when the planner cannot decide.
QUESTIONS = {
    "GOAL": "I'm not sure what you'd like to do. Would you like to book, move, or cancel an "
    "appointment, or are you asking about clinics, doctors, or open times?",
    "CLINIC": "Which clinic do you mean?",
    "DOCTOR": "Which doctor do you mean?",
    "DATE": "Which date do you mean?",
    "TIME": "Which time do you mean?",
    "NONE": "Could you tell me a little more about what you need?",
}
EMERGENCY_NOTICE = (
    "If this is an emergency, please call your local emergency number or go to the nearest "
    "emergency room now. I can't give medical advice; I can only help with appointments."
)
OUT_OF_SCOPE_NOTICE = (
    "I can only help with appointments, clinics, doctors, and open times, so I can't help with "
    "that."
)
TROUBLE = "I'm having trouble looking that up right now. Nothing was changed."


class GoalPlanner:
    """One bounded model call per message: which route, and for a lookup, what to look up."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        transport: Transport | None = None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._post = transport or http_transport(config.base_url)
        self._timer = timer
        self.usage = LlmUsage()
        self.fallbacks = 0  # Messages the planner could not decide; the controller took them.

    def plan(self, text: str, *, context: str, today: date, anchor: date | None) -> PlannerDecision:
        """The decision for this message, or the hand-off to the controller if anything is wrong."""
        started = self._timer()
        try:
            return self._plan(text, context, today, anchor, started)
        except Exception:  # A fault, a timeout or an unusable reply must never reach the patient.
            self.fallbacks += 1
            return HANDOFF

    def _plan(
        self, text: str, context: str, today: date, anchor: date | None, started: float
    ) -> PlannerDecision:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": _INSTRUCTIONS.format(
                        today=today.strftime("%Y-%m-%d (%A)"),
                        context=f"\nThe conversation so far: {context}\n" if context else "",
                        examples=_EXAMPLE_TEXT,
                    ),
                },
                {"role": "user", "content": text[: self.config.max_message_chars]},
            ],
            "format": PlannerDecision.model_json_schema(),
            "stream": False,
            **({"think": self.config.think} if self.config.think is not None else {}),
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
                "num_ctx": self.config.context_tokens,
                **({"seed": self.config.seed} if self.config.seed is not None else {}),
            },
        }
        body: Mapping[str, Any] = {}
        outcome = "planner_fault"
        try:
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty planner reply")
            decision = ground(PlannerDecision.model_validate_json(content), text, today, anchor)
            outcome = "planner_decision"
            return decision
        except (ValidationError, ValueError):
            outcome = "invalid_planner_reply"
            raise
        finally:
            self._record(started, outcome, body)

    def _record(self, started: float, outcome: str, body: Mapping[str, Any]) -> None:
        self.usage.calls.append(record_of((self._timer() - started) * 1000, outcome, body))


_WEEK_AFTER = re.compile(r"\bthe week (?:after|following) (?:this|this one|this week)\b", re.I)


_same_specialty = same_specialty


def week_in(text: str) -> str:
    """THIS or NEXT when the patient's own words name a week, else NONE. The words decide, not
    the model: a week the model failed to notice is still the week that was asked about."""
    if _WEEK_AFTER.search(text):
        return "NEXT"
    found = _WEEK.search(text)
    if not found:
        return "NONE"
    return "THIS" if found.group(1).casefold() == "this" else "NEXT"


def ground(
    decision: PlannerDecision, text: str, today: date, anchor: date | None = None
) -> PlannerDecision:
    """Keep only what the patient's message supports, and route to the controller when a lookup
    has nothing to look up. The same evidence rules as the intent extractor."""
    shaped = PatientIntent(
        operation="UNKNOWN",
        clinic_name=decision.clinic_name,
        doctor_name=decision.doctor_name,
        specialty=decision.specialty,
        start_date=decision.start_date,
        end_date=decision.end_date,
        time_preference=decision.time_preference,
    )
    replaced = StructuredIntentExtractor.unsupported(shaped, text, today, anchor)
    fields = {
        name: replaced.get(name, getattr(decision, name))
        for name in (
            "clinic_name",
            "doctor_name",
            "specialty",
            "start_date",
            "end_date",
            "time_preference",
        )
    }
    # Dates are read from the patient's own words, as the controller does, never from the model:
    # "Thursday" and "the 12th" mean the same thing here as they do in a booking.
    mentions = date_mentions(normalize_spoken(text), today, anchor)
    said = [m for m in mentions if m.iso]
    unsettled = [m for m in mentions if not m.iso and len(m.options) == 2]
    if decision.more and decision.route == "LOOKUP":
        # Each question names its own day: the model's day counts only if the words state it,
        # and the first question may fall back on the first day said.
        if fields["start_date"] is None and said:
            fields["start_date"] = fields["end_date"] = date.fromisoformat(said[0].iso)
    elif said and decision.route == "LOOKUP":
        fields["start_date"] = date.fromisoformat(said[0].iso)
        fields["end_date"] = date.fromisoformat(max(m.iso for m in said))
        last = max(said, key=lambda m: m.iso)
        fields["start_spec"] = said[0].spec if said[0].spec.startswith("rel:") else None
        fields["end_spec"] = last.spec if last.spec.startswith("rel:") else None
    elif unsettled and decision.route == "LOOKUP":
        # "Next Wednesday" said on a Monday means two days. Looking is harmless, so both are
        # looked at and shown, rather than asking which was meant.
        first, second = (date.fromisoformat(o) for o in unsettled[0].options)
        fields["start_date"], fields["end_date"] = first, second
        fields["start_spec"], fields["end_spec"] = unsettled[0].spec, None
    elif decision.route == "LOOKUP":
        fields["start_date"] = fields["end_date"] = None
    grounded = decision.model_copy(update={**fields, "week": week_in(text)})
    if grounded.more:
        grounded = grounded.model_copy(
            update={"more": [_ground_extra(q, text, today, anchor) for q in grounded.more[:2]]}
        )
    if grounded.route == "LOOKUP" and grounded.subject == "NONE":
        return HANDOFF
    if grounded.route != "LOOKUP":
        return grounded.model_copy(update={"subject": "NONE"})
    return grounded


def _ground_extra(
    question: PlannerDecision, text: str, today: date, anchor: date | None
) -> PlannerDecision:
    """A further question in the message, held to the same evidence: names must be in the words,
    and its day only counts if the words state it."""
    shaped = PatientIntent(
        operation="UNKNOWN",
        clinic_name=question.clinic_name,
        doctor_name=question.doctor_name,
        specialty=question.specialty,
        start_date=question.start_date,
        end_date=question.end_date,
        time_preference=question.time_preference,
    )
    replaced = StructuredIntentExtractor.unsupported(shaped, text, today, anchor)
    fields = {
        name: replaced.get(name, getattr(question, name))
        for name in (
            "clinic_name",
            "doctor_name",
            "specialty",
            "start_date",
            "end_date",
            "time_preference",
        )
    }
    if fields["start_date"] and not fields["end_date"]:
        fields["end_date"] = fields["start_date"]
    week = question.week if week_in(text) != "NONE" else "NONE"
    return question.model_copy(update={**fields, "week": week, "more": [], "route": "LOOKUP"})


def split_questions(
    questions: Sequence[PlannerDecision], text: str, today: date
) -> list[PlannerDecision]:
    """Several questions asked in one message, each held to the same evidence as a lone one.

    A question's names must be in the words. Each day the words state belongs to the question whose
    named clinic or doctor stands nearest it in the message, so "Pine on Monday, and Two Rivers on
    Friday" gives each its own day. A question with no name of its own takes the first one's
    clinic ("Dr. Kim's Monday and Dr. Rao's Monday at Pine")."""
    spoken = normalize_spoken(text)
    lowered = spoken.casefold()
    grounded = [_ground_extra(q, text, today, None) for q in questions]
    first_clinic = next((q.clinic_name for q in grounded if q.clinic_name), None)

    def anchor(q: PlannerDecision) -> tuple[int, int] | None:
        for name in (q.doctor_name, q.clinic_name):  # The more particular name anchors a day.
            where = lowered.find(name.casefold()) if name else -1
            if where >= 0:
                return where, where + len(name)
        return None

    def distance(span: tuple[int, int], position: int) -> int:
        return max(span[0] - position, position - span[1], 0)

    anchors = [anchor(q) for q in grounded]
    given = [i for i, a in enumerate(anchors) if a is not None]
    mentions = [m for m in date_mentions(spoken, today) if m.iso]
    assigned: dict[int, list[Any]] = {i: [] for i in range(len(grounded))}
    for mention in mentions:
        if given:
            nearest = min(given, key=lambda i: (distance(anchors[i], mention.position), i))
            assigned[nearest].append(mention)
    result = []
    for i, q in enumerate(grounded):
        update: dict[str, Any] = {"route": "LOOKUP"}
        if q.subject != "CLINICS" and not q.clinic_name and first_clinic and i > 0:
            update["clinic_name"] = first_clinic
        if assigned[i]:
            days = sorted(assigned[i], key=lambda m: m.iso)
            update.update(
                start_date=date.fromisoformat(days[0].iso),
                end_date=date.fromisoformat(days[-1].iso),
                start_spec=days[0].spec if days[0].spec.startswith("rel:") else None,
                end_spec=days[-1].spec if days[-1].spec.startswith("rel:") else None,
            )
        elif q.subject != "OPENINGS":
            update.update(start_date=None, end_date=None)
        result.append(q.model_copy(update=update))
    return result


def merge_open(decision: PlannerDecision, open_lookup: PlannerDecision | None) -> PlannerDecision:
    """A lookup left open for one detail is completed by the answer to it."""
    if open_lookup is None or decision.route != "LOOKUP":
        return decision
    if decision.subject not in ("NONE", open_lookup.subject):
        return decision  # A different question: the old one is dropped.
    update = {
        name: getattr(open_lookup, name)
        for name in (
            "clinic_name",
            "doctor_name",
            "specialty",
            "start_date",
            "end_date",
            "start_spec",
            "end_spec",
        )
        if getattr(decision, name) is None and getattr(open_lookup, name) is not None
    }
    if decision.week == "NONE" and decision.start_date is None:
        update["week"] = open_lookup.week
    if decision.time_preference == "ANYTIME":
        update["time_preference"] = open_lookup.time_preference
    return decision.model_copy(update={"subject": open_lookup.subject, **update})


# --- answering a lookup from read-only tools ------------------------------------------------------


@dataclass
class Answer:
    text: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    open_lookup: PlannerDecision | None = None  # Set when one detail is still needed.
    shown_clinic: str | None = None  # The clinic whose openings were just listed.
    shown_date: date | None = None  # The one day listed, if it was one day.
    shown_window: tuple[date, date] | None = None  # Or the days listed.
    shown_doctor: str | None = None  # The one doctor named, when the answer was about one.
    offers: list[dict[str, str]] = field(default_factory=list)  # The openings listed, in order.
    offer_offset: int = 0
    has_more: bool = False


class LookupAnswerer:
    """Runs the read-only tools a lookup needs, in a fixed order, and words the result."""

    def __init__(
        self,
        runtime: Any,
        principal: Any,
        clock: Callable[[], datetime] | None = None,
        offer_offset: int = 0,
    ) -> None:
        self.runtime = runtime
        self.principal = principal
        self.clock = clock or (lambda: datetime.now(UTC))
        self.offer_offset = offer_offset
        self._calls: list[tuple[str, dict[str, Any]]] = []

    def _call(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any] | None:
        if len(self._calls) >= MAX_LOOKUP_CALLS or tool not in (
            "search_clinic_directory",
            "search_appointment_availability",
        ):
            return None
        self._calls.append((tool, dict(arguments)))
        return self.runtime.invoke(tool, dict(arguments), principal=self.principal).to_dict()

    def answer(self, decision: PlannerDecision, draft: Any, text: str = "") -> Answer:
        self._calls = []
        subjects = [decision.subject, *[s for s in decision.also if s != decision.subject]]
        try:
            decision = self._unsplit_name(decision, text)
            decision = self._with_named_doctor(decision, text)
            if not decision.clinic_name and any(s in ("OPENINGS", "DOCTORS") for s in subjects):
                decision = self._with_named_clinic(decision, text)
            if self._asks_a_doctors_day(decision, subjects):
                subjects = ["OPENINGS"]
            answers = [self._one(decision, draft, subject) for subject in subjects]
            for extra in decision.more:
                extra_subject = extra.subject if extra.subject != "NONE" else "OPENINGS"
                if self._asks_a_doctors_day(extra, [extra_subject]):
                    extra_subject = "OPENINGS"
                if (
                    extra_subject == "OPENINGS"
                    and extra.start_date is None
                    and extra.week == "NONE"
                    and subjects[0] != "OPENINGS"
                ):
                    # "Who's at Pine Clinic, and what's open Monday?": the day was grounded on the
                    # first question, but it belongs to the one that asks about open times.
                    extra = extra.model_copy(
                        update={
                            "start_date": decision.start_date,
                            "end_date": decision.end_date,
                            "start_spec": decision.start_spec,
                            "end_spec": decision.end_spec,
                            "week": decision.week,
                        }
                    )
                if not extra.clinic_name and extra_subject in ("OPENINGS", "DOCTORS"):
                    extra = self._with_named_clinic(extra, text)
                answers.append(self._one(extra, draft, extra_subject))
        except Exception:  # Whatever went wrong, nothing was changed and nothing is made up.
            answers = [Answer(TROUBLE)]
        answer = next((a for a in reversed(answers) if a.shown_clinic), answers[-1])
        answer = Answer(
            " ".join(a.text for a in answers),
            shown_clinic=answer.shown_clinic,
            shown_date=answer.shown_date,
            shown_window=answer.shown_window,
            shown_doctor=answer.shown_doctor if len(answers) == 1 else None,
            offers=answer.offers,
            offer_offset=answer.offer_offset,
            has_more=answer.has_more,
            open_lookup=next((a.open_lookup for a in answers if a.open_lookup), None),
        )
        answer.calls = list(self._calls)
        return answer

    @staticmethod
    def _asks_a_doctors_day(decision: PlannerDecision, subjects: Sequence[str]) -> bool:
        """A question about a doctor and a day, with nothing else asked ("Is Dr. Moss around on
        Monday?"), is a question about that doctor's openings that day."""
        return list(subjects) == ["DOCTORS"] and bool(decision.doctor_name and decision.start_date)

    def _unsplit_name(self, decision: PlannerDecision, text: str) -> PlannerDecision:
        """The tail of a clinic's own name that the reading took for a doctor or a specialty
        ("Harbor View Clinic" and "Juniper Ridge"): when the clinic and that text together, as
        the patient wrote them, are the name of a clinic in the directory, they are one name."""
        lowered = text.casefold()
        for which in ("specialty", "doctor_name"):
            value = getattr(decision, which)
            joined = f"{decision.clinic_name} {value}" if decision.clinic_name and value else ""
            if not joined or joined.casefold() not in lowered:
                continue
            listing = self._call("search_clinic_directory", {"browse": True})
            clinics = ((listing or {}).get("data") or {}).get("clinics") or []
            named = [c for c in clinics if _text(c.get("name")) == _text(joined)]
            if len(named) == 1:
                return decision.model_copy(
                    update={"clinic_name": str(named[0]["name"]), which: None}
                )
        return decision

    def _one(self, decision: PlannerDecision, draft: Any, subject: str) -> "Answer":
        if subject == "OPENINGS":
            return self._openings(decision, draft)
        return self._directory(decision.model_copy(update={"subject": subject}), draft)

    @staticmethod
    def _with_named_doctor(decision: PlannerDecision, text: str) -> PlannerDecision:
        """A doctor the message names ("Dr. Kim") that the reading left out. Without it the
        question would be answered about whoever the request in hand names; the directory still
        checks the name when it is searched for."""
        if decision.doctor_name or decision.specialty:
            return decision
        named = {m[0].casefold(): m[0] for m in _TITLED_NAME.finditer(text)}
        if len(named) != 1:  # Two doctors in one message are two questions, read as such.
            return decision
        return decision.model_copy(update={"doctor_name": next(iter(named.values()))})

    def _with_named_clinic(self, decision: PlannerDecision, text: str) -> PlannerDecision:
        """A clinic the message names that the model left out, found in the directory itself."""
        listing = self._call("search_clinic_directory", {"browse": True})
        clinics = ((listing or {}).get("data") or {}).get("clinics") or []
        lowered = text.casefold()
        named = [c for c in clinics if _text(c.get("name")) in lowered]
        if len(named) == 1:
            return decision.model_copy(update={"clinic_name": str(named[0]["name"])})
        return decision

    # The directory: clinics and doctors.

    def _directory(self, decision: PlannerDecision, draft: Any) -> Answer:
        arguments: dict[str, Any] = {}
        clinic = decision.clinic_name
        if decision.subject == "DOCTORS" and not clinic and not decision.doctor_name:
            clinic = getattr(
                draft, "clinic_name", None
            )  # "Who are the doctors?" while at a clinic.
        for name, value in (
            (
                "clinic_name",
                search_name(clinic or "")
                if decision.subject == "DOCTORS"
                else search_name(decision.clinic_name or ""),
            ),
            ("doctor_name", decision.doctor_name),
            ("specialty", decision.specialty),
        ):
            if value:
                arguments[name] = value
        if not arguments:
            arguments["browse"] = True
        result = self._call("search_clinic_directory", arguments)
        if result is None:
            return Answer(TROUBLE)
        if not result.get("ok"):
            return self._directory_problem(decision, arguments, result)
        data = result.get("data") or {}
        clinics, doctors = list(data.get("clinics") or []), list(data.get("doctors") or [])
        more = bool(data.get("next_cursor"))
        if decision.subject == "CLINICS":
            listed = [{"kind": "clinic", "clinic": str(c["name"])} for c in clinics[:LIST_LIMIT]]
            return Answer(self._clinic_list(clinics, more), offers=listed)
        answer = Answer(self._doctor_list(doctors, clinics, decision, more))
        scoped = clinics[0]["name"] if len(clinics) == 1 else decision.clinic_name
        answer.offers = [
            {
                "kind": "doctor",
                "doctor": str(d["display_name"]),
                "specialty": str(d.get("specialty") or ""),
                "clinic": str(scoped) if scoped else "",
            }
            for d in doctors[:LIST_LIMIT]
        ]
        if len(doctors) == 1 and len(doctors[0].get("clinics") or []) == 1:
            answer.shown_clinic = str(doctors[0]["clinics"][0]["name"])
            answer.shown_doctor = str(doctors[0]["display_name"])
        return answer

    def _directory_problem(
        self, decision: PlannerDecision, arguments: dict[str, Any], result: Mapping[str, Any]
    ) -> Answer:
        error = result.get("error") or {}
        code = error.get("code")
        if code == "NO_RESULTS" and "doctor_name" in arguments and "clinic_name" in arguments:
            elsewhere = self._call(
                "search_clinic_directory", {"doctor_name": arguments["doctor_name"]}
            )
            found = ((elsewhere or {}).get("data") or {}).get("doctors") or []
            if found:
                places = _names(
                    sorted({str(c["name"]) for d in found for c in d.get("clinics", [])})
                )
                return Answer(
                    f"{found[0]['display_name']} isn't listed at {arguments['clinic_name']}. "
                    f"The directory lists {found[0]['display_name']} at {places}."
                )
        if code == "NO_RESULTS":
            wanted = arguments.get("doctor_name") or arguments.get("clinic_name") or "that"
            return Answer(f"I couldn't find {wanted} in the directory.")
        if code == "AMBIGUOUS_MATCH":
            choices = list(error.get("choices") or [])[:5]
            names = [str(c.get("name") or c.get("display_name")) for c in choices]
            return Answer(f"That could be {_names(names)}. Which one do you mean?")
        return Answer(TROUBLE)

    @staticmethod
    def _clinic_list(clinics: Sequence[Mapping[str, Any]], more: bool) -> str:
        if not clinics:
            return "I don't see any clinics listed right now."
        items = [
            f"{c['name']} in {c['city']}" if c.get("city") else str(c["name"])
            for c in clinics[:LIST_LIMIT]
        ]
        extra = f", and {len(clinics) - LIST_LIMIT} more" if len(clinics) > LIST_LIMIT else ""
        tail = " There are more; ask me about a particular one." if more else ""
        return f"These clinics are available: {', '.join(items)}{extra}.{tail}"

    @staticmethod
    def _doctor_list(
        doctors: Sequence[Mapping[str, Any]],
        clinics: Sequence[Mapping[str, Any]],
        decision: PlannerDecision,
        more: bool,
    ) -> str:
        if not doctors:
            return "I don't see any matching doctors listed."
        scoped = bool(decision.clinic_name) or (len(clinics) == 1 and not decision.doctor_name)
        items = []
        for doctor in doctors[:LIST_LIMIT]:
            entry = str(doctor["display_name"])
            if doctor.get("specialty"):
                entry += f" ({doctor['specialty']})"
            places = _names([str(c["name"]) for c in doctor.get("clinics", [])])
            items.append(entry if scoped or not places else f"{entry} at {places}")
        where = f" at {clinics[0]['name']}" if scoped and len(clinics) == 1 else ""
        extra = f", and {len(doctors) - LIST_LIMIT} more" if len(doctors) > LIST_LIMIT else ""
        tail = " There are more; ask me about a particular one." if more else ""
        return f"Matching doctors{where}: {', '.join(items)}{extra}.{tail}"

    # Open times: find the clinic in the directory, then ask for its availability.

    def _openings(self, decision: PlannerDecision, draft: Any) -> Answer:
        clinic_name = decision.clinic_name or getattr(draft, "clinic_name", None)
        if not clinic_name and decision.doctor_name:
            clinic_name, asked = self._clinic_of_doctor(decision)
            if asked is not None:
                return asked
            if clinic_name:
                decision = decision.model_copy(update={"clinic_name": clinic_name})
        if not clinic_name:
            return Answer(
                "Which clinic should I check?",
                open_lookup=decision,
            )
        if self._window(decision, draft, self.clock().astimezone(UTC).date())[0] is None:
            return Answer(
                f"Which date or week should I check at {clinic_name}?",
                open_lookup=decision.model_copy(update={"clinic_name": clinic_name}),
            )
        doctor_name = decision.doctor_name or getattr(draft, "doctor_name", None)
        lookup: dict[str, Any] = {"clinic_name": search_name(clinic_name)}
        if doctor_name:
            lookup["doctor_name"] = doctor_name
        found = self._call("search_clinic_directory", lookup)
        if found is None:
            return Answer(TROUBLE)
        if not found.get("ok"):
            return self._directory_problem(decision, lookup, found)
        data = found.get("data") or {}
        clinics = list(data.get("clinics") or [])
        exact = [c for c in clinics if _text(c.get("name")) == _text(clinic_name)]
        pool = exact if len(exact) == 1 else clinics
        if len(pool) != 1:
            names = _names([str(c["name"]) for c in clinics[:5]])
            return Answer(
                f"Which clinic do you mean: {names}?",
                open_lookup=decision.model_copy(update={"clinic_name": None}),
            )
        clinic = pool[0]
        arguments: dict[str, Any] = {"clinic_id": str(clinic["clinic_id"])}
        today = self._today(clinic.get("timezone"))
        first, last, days = self._window(decision, draft, today)
        assert first is not None and last is not None
        wanted_doctors: set[str] = set()
        if decision.specialty and not doctor_name:
            wanted_doctors = {
                str(d["doctor_id"])
                for d in data.get("doctors") or []
                if _same_specialty(decision.specialty, str(d.get("specialty") or ""))
            }
            if not wanted_doctors:
                listed = specialties_listed(clinic)
                if not listed:
                    return Answer(
                        f"I don't see a {decision.specialty} doctor at {clinic['name']}.",
                        shown_clinic=None,
                    )
                return Answer(  # The specialty stays asked for until the patient chooses.
                    no_such_specialty(str(clinic["name"]), decision.specialty, listed),
                    open_lookup=decision.model_copy(update={"clinic_name": str(clinic["name"])}),
                )
        if doctor_name:
            doctors = HybridControllerModel._matching_doctors(
                list(data.get("doctors") or []), doctor_name
            )
            if len(doctors) != 1:
                names = _names([str(d["display_name"]) for d in doctors[:5]])
                if not names:
                    return Answer(f"I couldn't find {doctor_name} at {clinic['name']}.")
                return Answer(
                    f"Which doctor do you mean: {names}?",
                    open_lookup=decision.model_copy(
                        update={"clinic_name": str(clinic["name"]), "doctor_name": None}
                    ),
                )
            arguments["doctor_id"] = str(doctors[0]["doctor_id"])
        if decision.time_preference != "ANYTIME":
            arguments["time_preference"] = decision.time_preference
        if last < today:
            return Answer(
                f"{_day_label(last.isoformat())} has already passed at {clinic['name']}. "
                "Which date should I check?",
                open_lookup=decision.model_copy(
                    update={"start_date": None, "end_date": None, "week": "NONE"}
                ),
            )
        first = max(first, today)
        if (last - first).days + 1 > MAX_SEARCH_DAYS:
            return Answer(
                f"I can look up to {MAX_SEARCH_DAYS} days at a time. Which dates should I check?",
                open_lookup=decision.model_copy(
                    update={"start_date": None, "end_date": None, "week": "NONE"}
                ),
            )
        arguments |= {"start_date": first.isoformat(), "end_date": last.isoformat()}
        result = self._call("search_appointment_availability", arguments)
        if result is None:
            return Answer(TROUBLE)
        if not result.get("ok"):
            code = (result.get("error") or {}).get("code")
            if code == "NO_AVAILABILITY":
                return self._nothing_open(
                    decision, str(clinic["name"]), arguments, (first, last), today, wanted_doctors
                )
            return Answer(TROUBLE)
        slots = list((result.get("data") or {}).get("slots") or [])
        if days:
            slots = [s for s in slots if date.fromisoformat(str(s["starts_at_local"])[:10]) in days]
        if wanted_doctors:
            before = slots
            slots = [
                s for s in slots if str((s.get("doctor") or {}).get("doctor_id")) in wanted_doctors
            ]
            if not slots:
                return Answer(
                    filtered_out(
                        str(clinic["name"]),
                        self._span(first, last),
                        specialty=decision.specialty,
                        part_of_day=self._part_of_day(decision),
                        open_specialties=specialties_open(before),
                    ),
                    open_lookup=decision.model_copy(update={"clinic_name": str(clinic["name"])}),
                )
        if days and not slots:
            return Answer(
                f"I don't see any openings at {clinic['name']} on "
                f"{_names([_day_label(d.isoformat()) for d in days])}"
                f"{self._preference(decision)}."
            )
        more = bool((result.get("data") or {}).get("next_cursor"))
        page = slots[self.offer_offset : self.offer_offset + SPOKEN_OPTION_LIMIT]
        return Answer(
            self._openings_text(str(clinic["name"]), slots, decision, more),
            shown_clinic=str(clinic["name"]) if slots else None,
            shown_date=first if slots and first == last else None,
            shown_window=(first, last) if slots and first != last else None,
            offers=self._offers(page, data.get("doctors") or []),
            offer_offset=self.offer_offset,
            has_more=self.offer_offset + len(page) < len(slots) or more,
        )

    def _clinic_of_doctor(self, decision: PlannerDecision) -> tuple[str | None, "Answer | None"]:
        """The clinic of a doctor asked about with no clinic, from the directory: the one clinic
        they work at, or a question when there are several (or the doctor is not found)."""
        found = self._call("search_clinic_directory", {"doctor_name": decision.doctor_name})
        if found is None:
            return None, Answer(TROUBLE)
        if not found.get("ok"):
            return None, self._directory_problem(
                decision, {"doctor_name": decision.doctor_name}, found
            )
        doctors = list((found.get("data") or {}).get("doctors") or [])
        places = sorted({str(c["name"]) for d in doctors for c in d.get("clinics", [])})
        if len(places) == 1:
            return places[0], None
        if not places:
            return None, Answer(f"I couldn't find {decision.doctor_name} in the directory.")
        who = doctors[0]["display_name"] if len(doctors) == 1 else decision.doctor_name
        return None, Answer(
            f"{who} works at {_names(places)}. Which clinic should I check?",
            open_lookup=decision,
        )

    def _today(self, timezone: object) -> date:
        try:
            return self.clock().astimezone(ZoneInfo(str(timezone))).date()
        except Exception:
            return self.clock().astimezone(UTC).date()

    def _window(
        self, decision: PlannerDecision, draft: Any, today: date
    ) -> tuple[date | None, date | None, list[date]]:
        """The days to look at, on `today`'s calendar (the clinic's own, once it is known), and the
        particular days when a phrase has two meanings."""
        if decision.start_date:
            first, last = decision.start_date, decision.end_date or decision.start_date
            days: list[date] = []
            for spec, which in ((decision.start_spec, "first"), (decision.end_spec, "last")):
                resolved = resolve_spec(spec, today) if spec else ()
                if len(resolved) == 1:
                    day = date.fromisoformat(resolved[0])
                    first, last = (day, last) if which == "first" else (first, day)
                elif len(resolved) == 2 and which == "first":
                    days = [date.fromisoformat(r) for r in resolved]
                    first, last = min(days), max(days)
            return min(first, last), max(first, last), days
        if decision.week != "NONE":
            first, last = _week_span("week:next" if decision.week == "NEXT" else "week:this", today)
            return first, last, []
        start = getattr(draft, "start_date", None)
        if start:
            return start, getattr(draft, "end_date", None) or start, []
        return None, None, []

    @staticmethod
    def _offers(
        slots: Sequence[Mapping[str, Any]], doctors: Sequence[Mapping[str, Any]] = ()
    ) -> list[dict[str, str]]:
        """The openings a listing showed, in the order shown, as the patient may refer to them,
        with each doctor's specialty from the directory ("the heart doctor" can then be told)."""
        kinds = {str(d.get("doctor_id")): str(d.get("specialty") or "") for d in doctors}
        ordered = sorted(slots, key=lambda s: str(s["starts_at_utc"]))
        return [
            {
                "kind": "opening",
                "time": _clock(str(s["starts_at_local"])),
                "date": str(s["starts_at_local"])[:10],
                "doctor": str((s.get("doctor") or {}).get("display_name", "")),
                "specialty": kinds.get(str((s.get("doctor") or {}).get("doctor_id")), ""),
            }
            for s in ordered
        ]

    @staticmethod
    def _span(first: date, last: date) -> str:
        if first == last:
            return f" on {_day_label(first.isoformat())}"
        return f" from {_day_label(first.isoformat())} to {_day_label(last.isoformat())}"

    @staticmethod
    def _part_of_day(decision: PlannerDecision) -> str | None:
        return None if decision.time_preference == "ANYTIME" else decision.time_preference.lower()

    def _nothing_open(
        self,
        decision: PlannerDecision,
        clinic: str,
        arguments: dict[str, Any],
        window: tuple[date, date],
        today: date,
        wanted_doctors: set[str],
    ) -> Answer:
        """The search came back empty. Say whether that is "nothing at all" or "nothing that
        matches", and when the clinic does have openings. Every day named was listed by the
        backend; no filter is dropped on the patient's behalf."""
        first, last = window
        span = self._span(first, last)
        part = self._part_of_day(decision)
        doctor = (decision.doctor_name or "that doctor") if "doctor_id" in arguments else None
        filters = {"specialty": decision.specialty if wanted_doctors else None}
        filters |= {"part_of_day": part, "doctor": doctor}
        plain = Answer(f"I don't see any openings at {clinic}{span}{self._preference(decision)}.")
        narrowed = {k: v for k, v in arguments.items() if k not in FILTER_ARGUMENTS}
        if narrowed != arguments:  # Was the search itself narrowed? Then look at everything.
            everything = self._call("search_appointment_availability", narrowed)
            if everything is None:
                return plain
            others = list((everything.get("data") or {}).get("slots") or [])
            if everything.get("ok") and others:
                return Answer(
                    filtered_out(
                        clinic, span, **filters, open_specialties=specialties_open(others)
                    ),
                    open_lookup=decision.model_copy(update={"clinic_name": clinic}),
                )
        start = first if not 0 <= (first - today).days < LOOK_AHEAD_DAYS else today
        ahead = {
            **arguments,
            "start_date": start.isoformat(),
            "end_date": date.fromordinal(start.toordinal() + LOOK_AHEAD_DAYS - 1).isoformat(),
        }
        result = self._call("search_appointment_availability", ahead)
        if result is None:
            return plain
        slots = list((result.get("data") or {}).get("slots") or []) if result.get("ok") else []
        if wanted_doctors:
            slots = [
                s for s in slots if str((s.get("doctor") or {}).get("doctor_id")) in wanted_doctors
            ]
        open_lookup = decision.model_copy(update={"clinic_name": clinic})
        if slots:
            return Answer(
                later_days(clinic, span, next_days(slots), **filters), open_lookup=open_lookup
            )
        return Answer(nothing_ahead(clinic, span, **filters), open_lookup=open_lookup)

    @staticmethod
    def _preference(decision: PlannerDecision) -> str:
        return (
            ""
            if decision.time_preference == "ANYTIME"
            else f" in the {decision.time_preference.lower()}"
        )

    def _openings_text(
        self,
        clinic: str,
        slots: Sequence[Mapping[str, Any]],
        decision: PlannerDecision,
        more: bool,
    ) -> str:
        if not slots:
            return f"I don't see any openings at {clinic}."
        ordered = sorted(slots, key=lambda s: str(s["starts_at_utc"]))
        shown = ordered[self.offer_offset : self.offer_offset + SPOKEN_OPTION_LIMIT]
        if not shown:
            return (
                f"Those are all the matching openings at {clinic}. "
                "You can ask about a different time or another day."
            )
        labels = []
        for slot in shown:
            local = str(slot["starts_at_local"])
            doctor = str((slot.get("doctor") or {}).get("display_name", ""))
            label = f"{_clock(local)} on {_day_label(local)}"
            if doctor:
                label += f" with {doctor}"
            labels.append(label)
        if len(set(labels)) < len(labels):
            labels = [
                label.replace(
                    _clock(str(slot["starts_at_local"])),
                    f"{_clock(str(slot['starts_at_local']))} "
                    f"(UTC{str(slot['starts_at_local'])[-6:]})",
                    1,
                )
                for label, slot in zip(labels, shown, strict=True)
            ]
        options = compact_numbered_labels(labels)
        separator = " " if options.startswith("on ") else ": "
        tail = (
            " Ask for more options, a different time, or another day."
            if self.offer_offset + len(shown) < len(ordered) or more
            else ""
        )
        pref = self._preference(decision)
        return f"Openings{pref} at {clinic}{separator}{options}.{tail}"


def planner_context(
    model: HybridControllerModel,
    open_lookup: PlannerDecision | None,
    shown: tuple[str | None, Any, Any, str | None] = (None, None, None, None),
) -> str:
    """What the planner is told about the conversation: no secrets, no earlier patient text."""
    parts = []
    base = model.context()
    if base:
        parts.append(base)
    if shown[0]:
        parts.append(f"The assistant just listed open times at {shown[0]}.")
    if open_lookup is not None:
        parts.append(
            f"The assistant is waiting for a detail to finish looking up {open_lookup.subject}."
        )
    return " ".join(parts)
