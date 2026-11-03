"""LG-3: the conversation as a LangGraph graph whose checkpointed state is the conversation.

In LG-1 and LG-2 the controller owned the request and the graph kept a copy of it. Here the graph
state is the only record. Each turn the deterministic scheduler (`HybridControllerModel`, the same
code that plans the scheduling tools, matches doctors and appointments from tool results and words
the review) is built from that state, runs the turn, and its state is written back. It keeps nothing
between turns, so there is nothing for the graph to disagree with, and a checkpoint is enough to
resume a conversation.

    intake -> understand -> lookup ------------------------------> finish
                         \\-> (clarify / out of scope) ----------> finish
                         \\-> schedule: decide <-> tool ----------> finish

`understand` is one model call per message (`TurnReader`): what the message is, what it asks to be
looked up, and what request it states. Everything the model returns is proposed, then checked
against the patient's words and the conversation state here. `lookup` answers questions from the
read-only tools (directory, availability), several in one message if asked, in a fixed order, with
every sentence built from tool results. `schedule` is the existing deterministic path: it alone
plans and prepares booking, rescheduling and cancellation reviews. Confirmation is still only the
authenticated UI's: nothing in this module can confirm.

State, all of it in the checkpoint:
- request: the draft the patient is building (clinic, doctor, dates, preferences, selection).
- dialogue: the question being waited on, the choices offered, picks pinned for the next search.
- review: the change awaiting Confirm (its id and the details the report-back needs).
- context: what the last lookup showed (so "I'll take 2 PM" has a referent) and any lookup left
  open for one missing detail.
- interruptions: asides taken without leaving the request.
What is not in it: the principal, the session id (the thread is a hash of it), tokens, the patient's
words, the reply text, the tool runtime and model.
"""

import contextlib
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.auth.service import Principal
from app.evaluation.conversation import (
    FALLBACK_REPLY,
    MAX_ACTIONS_PER_TURN,
    AgentReply,
    Awaiting,
    Offer,
    Prepared,
    TextSession,
)
from app.evaluation.graph_store import (
    STORE_FAULTS,
    CheckpointStore,
    CheckpointStoreError,
    ConversationBusy,
    owner_of,
    thread_for,
)
from app.evaluation.hybrid import (
    Draft,
    HybridControllerModel,
    PatientIntent,
    StructuredIntentExtractor,
    _day_label,
    _name_tokens,
    affirmative_stated,
    agreement_stated,
    more_options_requested,
    operation_stated,
    refers_to_existing,
    withdrawal_adds,
    withdrawal_stated,
)
from app.evaluation.llm import LlmConfig
from app.evaluation.model import ToolAction
from app.evaluation.offers import match_offer, numbered, options_from, resolve_offer
from app.evaluation.planner import (
    EMERGENCY_NOTICE,
    OUT_OF_SCOPE_NOTICE,
    QUESTIONS,
    GoalPlanner,
    LookupAnswerer,
    PlannerDecision,
    ground,
    merge_open,
    split_questions,
)
from app.evaluation.spoken import normalize_spoken, without_dates
from app.evaluation.turn_reader import PrecomputedExtractor, Read, TurnReader

MAX_INTERRUPTIONS = 5
QUICK_WORDS = 6  # A reply longer than this is always read by the model.
# A short reply that says "not", "instead", "other" is a contrast: only a reading can tell what to
# keep, so these are never settled by a rule.
_CONTRAST = frozenset(
    [
        "not",
        "no",
        "don't",
        "dont",
        "except",
        "other",
        "another",
        "instead",
        "rather",
        "neither",
        "nor",
        "without",
        "but",
        "however",
        "different",
    ]
)
_FILLER_AROUND_A_CLOCK = frozenset(
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
    ]
)
ASK_TO_BOOK_REPLY = (
    "I haven't prepared anything. Would you like me to prepare a booking for you to review? "
    "Tell me which opening you'd like."
)
BUSY_REPLY = "I'm still working on your last message. Please try again in a moment."
TROUBLE_REPLY = "I'm having trouble right now. Nothing was changed. Please try again."
FINISHED_REVIEW = frozenset({"CONSUMED", "EXPIRED", "INVALIDATED", "MISSING"})
EMPTY_CONVERSATION: dict[str, Any] = {
    "request": {},
    "dialogue": {},
    "review": None,
    "retiring": None,
    "context": {},
    "interruptions": [],
    "turn": 0,
}
Mode = Literal["combined", "split"]
_BARE_NUMBER = re.compile(
    r"(?:the |number |option )?(\d{1,2})(?::(\d{2}))?(?: please| thanks| thank you)?",
    re.IGNORECASE,
)
_OFFERED_CLOCK = re.compile(r"^(\d{1,2}):(\d{2}) (AM|PM)")
_NUMBER_IN = re.compile(r"(?<![\d:./-])(\d{1,2})(?::(\d{2}))?(?![\d:/])(?!\s?(?:st|nd|rd|th)\b)")


class AgentState(TypedDict, total=False):
    # this turn
    kind: str  # "utterance" or "event"
    route: str  # SCHEDULING, LOOKUP, CLARIFY or OUT_OF_SCOPE
    steps: int
    needs_tool: bool
    # the conversation
    request: dict[str, Any]
    dialogue: dict[str, Any]
    review: dict[str, Any] | None
    retiring: str | None
    context: dict[str, Any]
    interruptions: list[dict[str, Any]]
    turn: int
    owner: str  # An opaque digest of the patient the conversation belongs to.


def thread_id_for(principal: Principal | None, fallback: str) -> str:
    if principal is not None:
        return thread_for(principal.session_id)
    return hashlib.sha256(f"anonymous:{fallback}".encode()).hexdigest()[:32]


# --- the state, to and from the scheduler ------------------------------------------------


def hydrate(state: Mapping[str, Any], controller: HybridControllerModel) -> None:
    """Load the conversation into a scheduler for one turn."""
    request, dialogue = state.get("request") or {}, state.get("dialogue") or {}
    review = state.get("review")
    held = controller.state
    held.draft = Draft.model_validate(request) if request else None
    held.awaiting = Awaiting(dialogue.get("awaiting", "NOTHING"))
    held.last_question = dialogue.get("question", "")
    held.offered = [Offer(o["label"], o["key"]) for o in dialogue.get("offered", [])]
    held.pin_appointment = dialogue.get("pin_appointment")
    held.pin_slot = dialogue.get("pin_slot")
    held.offer_offset = int(dialogue.get("offer_offset", 0))
    held.turns = int(state.get("turn", 0))
    held.prepared = (
        Prepared(review["pending_action_id"], review.get("summary", "")) if review else None
    )
    controller.chosen = (review or {}).get("chosen")
    controller.current = (review or {}).get("current")


def _essential(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Of a slot or appointment the review is about, only what the report-back words need: when,
    with whom, where. Identifiers of clinic and doctor, times in UTC and the rest of the tool's
    payload are not kept."""
    if not item:
        return None
    kept: dict[str, Any] = {
        "starts_at_local": item.get("starts_at_local"),
        "doctor": {"display_name": (item.get("doctor") or {}).get("display_name", "")},
        "clinic": {"name": (item.get("clinic") or {}).get("name", "")},
    }
    for name in ("appointment_id", "slot_id"):
        if item.get(name):
            kept[name] = item[name]
    return kept


def dehydrate(controller: HybridControllerModel, retiring: str | None) -> dict[str, Any]:
    """The conversation as the scheduler left it, as plain values."""
    held = controller.state
    review = None
    if held.prepared is not None:
        review = {
            "pending_action_id": held.prepared.pending_action_id,
            "summary": held.prepared.summary,
            "chosen": _essential(controller.chosen),
            "current": _essential(controller.current),
        }
    return {
        "request": held.draft.model_dump(mode="json") if held.draft is not None else {},
        "dialogue": {
            "awaiting": str(held.awaiting),
            "question": held.last_question,
            "offered": [{"label": o.label, "key": o.key} for o in held.offered],
            "pin_appointment": held.pin_appointment,
            "pin_slot": held.pin_slot,
            "offer_offset": held.offer_offset,
        },
        "review": review,
        "retiring": retiring,
        "turn": held.turns,
    }


def _opening_label(offer: Mapping[str, Any], listed: list[Mapping[str, Any]]) -> str:
    """An offer as the patient heard it. An opening: the time, with the doctor when more than one,
    and the day when the listing spans several. A doctor: the name and specialty. A clinic: its
    name."""
    if offer.get("kind") == "clinic":
        return str(offer["clinic"])
    if offer.get("kind") == "doctor":
        specialty = f" ({offer['specialty']})" if offer.get("specialty") else ""
        return f"{offer['doctor']}{specialty}"
    label = str(offer["time"])
    if len({o.get("doctor") for o in listed}) > 1 and offer.get("doctor"):
        label += f" with {offer['doctor']}"
    if len({o.get("date") for o in listed}) > 1:
        label += f" on {_day_label(str(offer['date']))}"
    return label


def _same_name(stated: str, listed: str | None) -> bool:
    return bool(listed) and stated.casefold().strip() == str(listed).casefold().strip()


def bare_time_choice(text: str, dialogue: Mapping[str, Any]) -> str | None:
    """A lone "11" or "9:30" said to "which time?" is the one offered time that reads that way,
    whether it is the whole message or part of a sentence ("back to the first one, 11")."""
    if dialogue.get("awaiting") != "TIME" or not dialogue.get("offered"):
        return None
    spoken = without_dates(normalize_spoken(text))
    said = list(_NUMBER_IN.finditer(spoken))
    if len(said) != 1:
        return None
    hour, minute = int(said[0][1]) % 12, int(said[0][2] or 0)
    fits = []
    for offer in dialogue["offered"]:
        clock = _OFFERED_CLOCK.match(offer["label"])
        if clock and int(clock[1]) % 12 == hour and int(clock[2]) == minute:
            fits.append(f"{clock[1]}:{clock[2]} {clock[3]}")
    return fits[0] if len(set(fits)) == 1 else None


# --- the session ----------------------------------------------------------------------------------


class OrchestratedSession(TextSession):
    """A `TextSession` whose conversation lives in a LangGraph checkpoint."""

    def __init__(
        self,
        runtime: Any,
        principal: Principal | None,
        *,
        config: LlmConfig | None = None,
        mode: Mode = "combined",
        reader: TurnReader | None = None,
        planner: GoalPlanner | None = None,
        extractor: StructuredIntentExtractor | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: CheckpointStore | None = None,
        review_status: Callable[[str], str | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        quick_paths: bool = True,
        reader_options: Mapping[str, Any] | None = None,
    ) -> None:
        config = config or LlmConfig()
        self.mode: Mode = mode
        self.extractor = extractor or StructuredIntentExtractor(config)
        self.reader = reader or TurnReader(config, **(reader_options or {}))
        self.planner = planner or GoalPlanner(config)
        self._clock = clock or (lambda: datetime.now(UTC))
        super().__init__(
            HybridControllerModel(self.extractor, clock=self._clock), runtime, principal
        )
        if store is not None:
            checkpointer = store.saver
        elif checkpointer is None:
            if principal is not None:
                raise ValueError("a signed-in conversation needs a durable checkpoint store")
            checkpointer = InMemorySaver()  # A visitor's lookups hold nothing worth keeping.
        self._saver = checkpointer
        self._store = store
        self._review_status = review_status
        self.quick_paths = quick_paths
        self.quick_reads = 0  # Turns settled without a model call.
        self._owner = owner_of(principal.patient_id) if principal and principal.patient_id else ""
        self._review_at_intake: str | None = None
        self.trace: list[dict[str, Any]] = []  # What each turn read and decided, for diagnosis.
        self._thread = thread_id_for(principal, uuid.uuid4().hex)
        self._pending_input: tuple[str, Any] | None = None
        self._text = ""
        self._action: ToolAction | None = None
        self._reply: AgentReply | None = None
        self._final: AgentReply | None = None
        self._decision = PlannerDecision(route="SCHEDULING")
        self._quick_turn = False
        self._intent: PatientIntent | None = None
        self._aside: str | None = None
        self._shown_now: dict[str, Any] | None = None
        self._asked_now: dict[str, Any] | None = None
        self._open_now: dict[str, Any] | None = None
        self.graph = self._build()

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self._thread}}

    @property
    def model_calls(self) -> int:
        counted = [self.extractor.usage, self.reader.usage, self.planner.usage]
        return sum(len(u.calls) for u in counted if u is not None)

    # -- the graph ------------------------------------------------------------------------------

    def _build(self):
        graph = StateGraph(AgentState)
        for name, node in (
            ("intake", self._intake),
            ("understand", self._understand),
            ("lookup", self._lookup),
            ("schedule", self._schedule),
            ("decide", self._decide),
            ("tool", self._tool),
            ("finish", self._finish),
        ):
            graph.add_node(name, node)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "understand")
        graph.add_conditional_edges(
            "understand", self._after_understand, ["schedule", "lookup", "finish"]
        )
        graph.add_edge("lookup", "finish")
        graph.add_edge("schedule", "decide")
        graph.add_conditional_edges("decide", self._after_decide, ["tool", "finish"])
        graph.add_edge("tool", "decide")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self._saver)

    def _intake(self, state: AgentState) -> dict[str, Any]:
        assert self._pending_input is not None
        kind, payload = self._pending_input
        self._pending_input = None
        self._reply = self._action = self._aside = None
        self._shown_now = self._open_now = self._asked_now = None
        self._intent = None
        self.transcript = []
        reset = self._unusable(state)
        if reset:
            state = {**state, **EMPTY_CONVERSATION, "turn": int(state.get("turn", 0))}
        review = state.get("review") or {}
        self.prepared_action_id = state.get("retiring") or review.get("pending_action_id")
        self._review_at_intake = self.prepared_action_id
        controller = HybridControllerModel(self.extractor, clock=self._clock)
        hydrate(state, controller)
        self.model = controller
        if kind == "event":
            self.add_event(payload)
        else:
            self._text = payload
        update: dict[str, Any] = {"kind": kind, "steps": 0, "needs_tool": False}
        if reset:
            update.update(EMPTY_CONVERSATION)
            update["turn"] = state["turn"]
        return update

    def _unusable(self, state: AgentState) -> bool:
        """Whether the saved conversation must not be continued: it belongs to someone else, or
        its review has already been decided or has lapsed (so the screen it describes is gone)."""
        saved_owner = state.get("owner")
        if saved_owner and saved_owner != self._owner:
            return True
        review = state.get("review") or {}
        pending = review.get("pending_action_id")
        if pending and self._review_status is not None:
            try:
                return self._review_status(str(pending)) in FINISHED_REVIEW
            except Exception:  # Unknown is not finished: the app's Confirm decides, not this.
                return False
        return False

    def _understand(self, state: AgentState) -> dict[str, Any]:
        controller = self.model
        assert isinstance(controller, HybridControllerModel)
        if state.get("kind") == "event" or controller.screens_secret(self._text):
            return {"route": "SCHEDULING"}  # The scheduler answers both; no model reads a secret.
        context = self._context_for_model(state, controller)
        today = self._clock().astimezone(UTC).date()
        draft = controller.state.draft
        anchor = getattr(draft, "start_date", None)
        context_state = state.get("context") or {}
        open_lookup = (
            PlannerDecision.model_validate(context_state["open_lookup"])
            if context_state.get("open_lookup")
            else None
        )
        decision = PlannerDecision(route="SCHEDULING")
        intent: PatientIntent | None = None
        proposed: int | None = None
        quick = self._quick_read(state, controller) if self.quick_paths else None
        self._quick_turn = quick is not None
        if quick is not None:
            self.quick_reads += 1
            decision, intent, proposed = quick.decision, quick.intent, quick.option_number
        elif self.mode == "combined":
            read = self.reader.read(self._text, context=context, today=today, anchor=anchor)
            if read is not None:
                decision, intent, proposed = read.decision, read.intent, read.option_number
        else:
            decision = self.planner.plan(self._text, context=context, today=today, anchor=anchor)
        decision = merge_open(decision, open_lookup)
        decision = self._compound(decision, today)
        decision, intent = self._reconcile(decision, intent, state, controller, proposed)
        self._decision = decision
        self.trace.append(
            {
                "text": self._text,
                "route": decision.route,
                "decision": decision.model_dump(mode="json", exclude_defaults=True),
                "intent": intent.model_dump(mode="json", exclude_defaults=True) if intent else None,
                "proposed_option": proposed,
                "model_skipped": quick is not None,
            }
        )
        return {"route": decision.route}

    def _compound(self, decision: PlannerDecision, today: date) -> PlannerDecision:
        """A message that asks several things about different places, doctors or days is split by
        one more focused reading into the questions it asks, each answered from its own words."""
        if (
            decision.route != "LOOKUP"
            or not decision.also
            or self.mode != "combined"
            or decision.more
        ):
            return decision
        asked = self.reader.questions(self._text, today)
        if not asked or len(asked) < 2:
            return decision
        questions = split_questions(
            [PlannerDecision(route="LOOKUP", **q.model_dump(exclude_none=True)) for q in asked],
            self._text,
            today,
        )
        if len({(q.clinic_name, q.doctor_name) for q in questions}) < 2:
            return decision  # About one place and one doctor: the first reading stands.
        subjects = [decision.subject, *decision.also]
        if len(subjects) == len(questions):  # The first reading's labels, in the order asked.
            questions = [
                q.model_copy(update={"subject": s})
                for q, s in zip(questions, subjects, strict=True)
            ]
        first, rest = questions[0], questions[1:]
        return decision.model_copy(
            update={
                "subject": first.subject,
                "clinic_name": first.clinic_name,
                "doctor_name": first.doctor_name,
                "specialty": first.specialty,
                "start_date": first.start_date,
                "end_date": first.end_date,
                "start_spec": first.start_spec,
                "end_spec": first.end_spec,
                "week": "NONE",
                "time_preference": "ANYTIME",
                "also": [],
                "more": rest,
            }
        )

    def _quick_read(self, state: AgentState, controller: HybridControllerModel) -> Read | None:
        """The reading of a short reply that cannot mean anything else, made without the model:
        a plain "yes", a plain "never mind", a lone clock time that answers a time question, a
        name that singles out one of the offers. Anything longer, or with a contrast in it, is
        left to the model; so is every reply whose meaning depends on more than the reply."""
        text = self._text.strip()
        words = text.split()
        tokens = {w.strip(".,!?;:").casefold() for w in words}
        wants_more = more_options_requested(text)
        if not words or len(words) > QUICK_WORDS or (tokens & _CONTRAST and not wants_more):
            return None
        held = controller.state
        context = state.get("context") or {}
        shown_state = context.get("shown") or {}
        shown = shown_state.get("offers")
        nothing_to_continue = held.prepared is None and held.draft is None and not shown
        asked = held.awaiting != Awaiting.NOTHING

        def read(**fields: Any) -> Read:
            return Read(
                PlannerDecision(route="SCHEDULING", same_request=True),
                PatientIntent(operation="UNKNOWN", **fields),
            )

        if asked and held.offered and wants_more:
            return read()
        if shown and shown[0].get("kind") == "opening" and wants_more:
            criteria = context.get("asked") or {}
            window = shown_state.get("window") or []
            first = criteria.get("start_date") or shown_state.get("date")
            last = criteria.get("end_date") or shown_state.get("date")
            if len(window) == 2:
                first, last = first or window[0], last or window[1]
            decision = PlannerDecision(
                route="LOOKUP",
                subject="OPENINGS",
                clinic_name=criteria.get("clinic_name") or shown_state.get("clinic"),
                doctor_name=criteria.get("doctor_name"),
                specialty=criteria.get("specialty"),
                start_date=first,
                end_date=last,
                time_preference=criteria.get("time_preference", "ANYTIME"),
            )
            return Read(decision, PatientIntent(operation="UNKNOWN"))
        if agreement_stated(text) and not nothing_to_continue:
            return read(message_type="AGREEMENT")
        if (
            withdrawal_stated(text)
            and (held.draft is not None or held.prepared is not None)
            and not withdrawal_adds(text)
        ):
            return read(message_type="ABANDON")
        if asked and held.offered:
            dialogue = state.get("dialogue") or {}
            clock = bare_time_choice(text, dialogue)
            leftover = {
                t for t in tokens if t.isalpha() and t not in _FILLER_AROUND_A_CLOCK and t != "am"
            }
            if clock and not leftover - {"pm", "o'clock", "oclock"}:
                return read(time=clock, selection="SPECIFIC_TIME")
            if held.awaiting in (Awaiting.CLINIC_CHOICE, Awaiting.DOCTOR):
                number = match_offer(text, [o.label for o in held.offered])
                if number is not None:
                    return read(option_number=number)
        return None

    def _context_for_model(self, state: AgentState, controller: HybridControllerModel) -> str:
        parts = [controller.context()]
        context = state.get("context") or {}
        shown = context.get("shown")
        if (
            shown
            and shown.get("clinic")
            and (shown.get("offers") or [{}])[0].get("kind")
            in (
                None,
                "opening",
            )
        ):
            parts.append(f"The assistant just listed open times at {shown['clinic']}.")
        if context.get("open_lookup"):
            parts.append("The assistant is waiting for a detail to finish a lookup.")
        labels = self._offers(state, controller)
        if labels:
            # The patient heard the choices in an order; the model is given that order, numbered.
            plain = f" (options: {'; '.join(labels)})"
            parts[0] = parts[0].replace(plain, "")
            parts.append(f"Numbered choices the assistant offered: {numbered(labels)}.")
        return " ".join(p for p in parts if p)

    @staticmethod
    def _offers(state: AgentState, controller: HybridControllerModel) -> list[str]:
        """What the patient was last offered to pick from, as they heard it: the scheduler's own
        question first, otherwise the openings a lookup just listed."""
        if controller.state.offered:
            return [o.label for o in controller.state.offered]
        shown = (state.get("context") or {}).get("shown") or {}
        return [_opening_label(o, shown.get("offers") or []) for o in shown.get("offers") or []]

    def _reconcile(
        self,
        decision: PlannerDecision,
        intent: PatientIntent | None,
        state: AgentState,
        controller: HybridControllerModel,
        proposed: int | None = None,
    ) -> tuple[PlannerDecision, PatientIntent | None]:
        """Settle the reading against the conversation: what the model proposed is applied only
        where the state makes it sensible, and checked where it is about to change something.

        In order: a withdrawal is told apart from a correction; a pick from what was offered is
        resolved on evidence; a message that says nothing a request in progress can use is read
        again for what it changes; an operation the wording does not state is confirmed by an
        independent reading or dropped; and only then does a request carry on from a lookup."""
        held = controller.state
        if (
            intent is not None
            and decision.route == "LOOKUP"
            and held.draft is None
            and held.prepared is None
            and self._affirmed(state, self._text)
        ):
            # "Okay, Wednesday", said right after the assistant asked whether to book: an answer
            # to that question, not a new question about what exists.
            decision = decision.model_copy(update={"route": "SCHEDULING"})
        if intent is not None and decision.route == "SCHEDULING":
            intent, decision = self._withdrawal(intent, decision, controller)
            # "Make it Thursday instead" corrects the request in hand; it starts no other one.
            if (
                decision.same_request
                and held.draft is not None
                and held.draft.operation in ("BOOK", "RESCHEDULE")
                and intent.operation in ("BOOK", "RESCHEDULE")
                and intent.operation != held.draft.operation
            ):
                intent = intent.model_copy(update={"operation": "UNKNOWN"})
            # An appointment asked for without a booking word is a booking to be confirmed below.
            if (
                decision.starts_booking
                and held.draft is None
                and intent.operation in ("UNKNOWN", "BOOK")
                and (controller._schedule_words(self._text) or intent.clinic_name)
            ):
                intent = intent.model_copy(update={"operation": "BOOK"})
            intent = self._option_applied(intent, state, controller)
            intent, decision = self._revised(intent, decision, controller)
            if (
                held.draft is None
                and held.prepared is None
                and intent.operation in ("UNKNOWN", "BOOK")
                and intent.message_type in ("REQUEST", "AGREEMENT")
                and self._affirmed(state, self._text)
            ):
                # The assistant just asked whether to book and the patient said yes: that is the
                # request. It chooses nothing: what is still unsaid is asked for.
                intent = intent.model_copy(update={"operation": "BOOK"})
                controller.operation_vouched = True
            asked_to_book = intent.operation == "BOOK"
            if decision.route == "SCHEDULING":
                intent = self._operation_verified(intent, controller)
            demoted = asked_to_book and intent.operation == "UNKNOWN"
            intent = self._carried(intent, state, controller)
            bare = bare_time_choice(self._text, state.get("dialogue") or {})
            if bare and not intent.time:
                intent = intent.model_copy(update={"time": bare, "selection": "SPECIFIC_TIME"})
            decision = self._lookup_if_only_asking(decision, intent, held, demoted)
            if decision.route == "SCHEDULING":
                intent = self._clinic_of_doctor(intent, controller)
            has_content = intent.operation != "UNKNOWN" or any(
                getattr(intent, f) is not None
                for f in ("clinic_name", "doctor_name", "start_date", "time", "option_number")
            )
            if intent.message_type == "OFF_TOPIC" and not has_content:
                decision = decision.model_copy(update={"route": "OUT_OF_SCOPE"})
            if self.mode == "combined":
                controller.extractor = PrecomputedExtractor(intent)
        if decision.route == "LOOKUP":
            decision = self._inherit_lookup(decision, state)
        if decision.urgent and (
            decision.route != "SCHEDULING" or intent is None or intent.operation == "UNKNOWN"
        ):
            decision = decision.model_copy(update={"route": "OUT_OF_SCOPE"})  # Not a booking.
        return decision, intent

    # -- the checks ------------------------------------------------------------------------

    def _withdrawal(
        self, intent: PatientIntent, decision: PlannerDecision, controller: HybridControllerModel
    ) -> tuple[PatientIntent, PlannerDecision]:
        """Dropping a request retires its review, so the wording or a second reading must agree.
        A correction the model took for a withdrawal ("actually the cardiologist") is only that.
        A withdrawal that goes on to say what is wanted instead ("forget it, the heart center
        instead") drops nothing: it is a correction, read for what it changes."""
        held = controller.state
        if (
            intent.message_type != "ABANDON"
            or (held.draft is None and held.prepared is None)
            or self._quick_turn
        ):
            return intent, decision
        if not withdrawal_stated(self._text) and (
            self.reader.check(self._text, in_progress=True) != "WITHDRAW"
        ):
            return intent.model_copy(update={"message_type": "REQUEST"}), decision
        if not withdrawal_adds(self._text):
            return intent, decision
        got = self._revision_of(controller)
        if got is None or got[0] != "CHANGE" or not self._adds(got[1].intent, held.draft):
            return intent, decision
        controller.replacing = True
        return self._with_changes(intent, decision, got[1].intent)

    def _revised(
        self, intent: PatientIntent, decision: PlannerDecision, controller: HybridControllerModel
    ) -> tuple[PatientIntent, PlannerDecision]:
        """A message that reaches a request in progress but says nothing the request can use is
        read once more, with the request in front of the reader: it may change a detail the first
        reading missed, ask what exists (which leaves the request alone), or be small talk."""
        held = controller.state
        if (
            self.mode != "combined"
            or self._quick_turn
            or decision.route != "SCHEDULING"
            or (held.draft is None and held.prepared is None)
            or intent.message_type not in ("REQUEST", "AGREEMENT")
            or intent.option_number is not None
            or agreement_stated(self._text)
            or withdrawal_stated(self._text)
            or controller.replacing
            or controller.operation_vouched
            or not (not self._adds(intent, held.draft) or "?" in self._text)
        ):
            return intent, decision
        got = self._revision_of(controller)
        if got is None:
            return intent, decision
        kind, read = got
        if kind == "CHANGE" and self._adds(read.intent, held.draft):
            return self._with_changes(intent, decision, read.intent)
        if kind == "QUESTION":
            asked = self._question_of(read)
            if asked is not None:
                return intent.model_copy(update={"message_type": "REQUEST"}), asked
        return intent, decision

    @staticmethod
    def _adds(intent: PatientIntent, draft: Draft | None) -> bool:
        """Whether a reading carries anything the request in progress could take."""
        if any(
            getattr(intent, f) is not None
            for f in (
                "clinic_name",
                "doctor_name",
                "specialty",
                "start_date",
                "end_date",
                "new_date",
                "time",
                "appointment_time",
            )
        ):
            return True
        if intent.time_preference != "ANYTIME" or intent.selection != "UNSPECIFIED":
            return True
        return intent.operation not in ("UNKNOWN", draft.operation if draft else "UNKNOWN")

    @staticmethod
    def _with_changes(
        intent: PatientIntent, decision: PlannerDecision, changes: PatientIntent
    ) -> tuple[PatientIntent, PlannerDecision]:
        update: dict[str, Any] = {
            field: getattr(changes, field)
            for field in (
                "clinic_name",
                "doctor_name",
                "specialty",
                "start_date",
                "end_date",
                "new_date",
                "time",
                "appointment_time",
            )
            if getattr(changes, field) is not None
        }
        if changes.time_preference != "ANYTIME":
            update["time_preference"] = changes.time_preference
        if changes.time:
            update["selection"] = "SPECIFIC_TIME"
        if changes.operation != "UNKNOWN":
            update["operation"] = changes.operation
        update["message_type"] = "REQUEST"
        return intent.model_copy(update=update), decision.model_copy(update={"same_request": True})

    def _question_of(self, read: Read) -> PlannerDecision | None:
        """A question asked while a request is in progress, as a lookup (the request waits)."""
        found = read.intent
        asked = PlannerDecision(
            route="LOOKUP",
            subject="OPENINGS",
            clinic_name=found.clinic_name,
            doctor_name=found.doctor_name,
            specialty=found.specialty,
            start_date=found.start_date,
            end_date=found.end_date,
            time_preference=found.time_preference,
        )
        today = self._clock().astimezone(UTC).date()
        grounded = ground(asked, self._text, today)  # The days come from the patient's words.
        if grounded.route != "LOOKUP":
            return None
        timed = bool(
            grounded.start_date or grounded.week != "NONE" or grounded.time_preference != "ANYTIME"
        )
        named = bool(grounded.clinic_name or grounded.doctor_name or grounded.specialty)
        if not (named or timed):
            return None
        return grounded.model_copy(update={"subject": "OPENINGS" if timed else "DOCTORS"})

    def _revision_of(self, controller: HybridControllerModel) -> tuple[str, Read] | None:
        """The second reading of a message that arrives while a request is in progress."""
        held = controller.state
        draft = held.draft
        shown = {k: v for k, v in (draft.model_dump(mode="json") if draft else {}).items()}
        known = {
            k: v
            for k, v in shown.items()
            if k
            in (
                "operation",
                "clinic_name",
                "doctor_name",
                "specialty",
                "start_date",
                "end_date",
                "new_date",
                "time",
                "appointment_time",
                "time_preference",
            )
            and v not in (None, "", "ANYTIME", "UNSPECIFIED")
        }
        parts = [json.dumps(known)] if known else []
        if held.prepared is not None:
            parts.append(
                f'A review is waiting for the patient to confirm: "{held.prepared.summary}"'
            )
        doctors = self._doctors_at(draft.clinic_name if draft is not None else None)
        listed = ", ".join(f"{name} ({kind})" if kind else name for name, kind in doctors.items())
        today = self._clock().astimezone(UTC).date()
        return self.reader.revise(
            self._text,
            request=" ".join(parts) or "nothing is settled yet",
            doctors=listed,
            today=today,
            anchor=getattr(draft, "start_date", None),
            known_doctors=doctors,
        )

    def _doctors_at(self, clinic: str | None) -> dict[str, str]:
        """The doctors the directory lists at a clinic, name to specialty."""
        if not clinic:
            return {}
        result = LookupAnswerer(self.runtime, self.principal, self._clock)._call(
            "search_clinic_directory", {"clinic_name": clinic}
        )
        found = ((result or {}).get("data") or {}).get("doctors") or []
        return {str(d["display_name"]): str(d.get("specialty") or "") for d in found}

    @staticmethod
    def _booking_offered(state: AgentState) -> bool:
        """Whether the assistant's last reply asked if the patient wants to book."""
        return bool((state.get("context") or {}).get("booking_offered"))

    def _affirmed(self, state: AgentState, text: str) -> bool:
        return self._booking_offered(state) and affirmative_stated(text)

    def _operation_verified(
        self, intent: PatientIntent, controller: HybridControllerModel
    ) -> PatientIntent:
        """An operation is accepted from what the patient said and where the conversation stands,
        never because the model proposed it.

        Moving or cancelling needs wording that says so; and while a booking is being arranged a
        correction to it ("change it to Thursday", "make it the 12th with Dr. Rao") stays a
        booking unless the wording is about an appointment the patient already has ("my
        appointment on the 10th"). A booking is accepted on an act of booking in the patient's
        words, on a request being arranged already, on an answer to the assistant's own question, or
        on the patient's pick of an opening that was listed (settled before this point). No model
        reading, the first or a second, ever starts a booking by itself: a message with none of that
        asks for nothing to be done and is answered as a question about what exists. The backend
        still validates the rest (the appointment must exist; the review must be confirmed)."""
        held = controller.state
        operation = intent.operation
        if operation == "UNKNOWN" or controller.operation_vouched:
            return intent
        arranging = held.draft.operation if held.draft is not None else None
        if operation in ("RESCHEDULE", "CANCEL"):
            if not operation_stated(operation, self._text) or (
                arranging == "BOOK" and not refers_to_existing(self._text)
            ):
                return intent.model_copy(update={"operation": "UNKNOWN"})
            controller.operation_vouched = True
            return intent
        if operation_stated("BOOK", self._text) or (
            held.draft is not None and arranging == "BOOK" and held.draft.vouched
        ):
            controller.operation_vouched = True
            return intent
        return intent.model_copy(update={"operation": "UNKNOWN"})

    def _carried(
        self, intent: PatientIntent, state: AgentState, controller: HybridControllerModel
    ) -> PatientIntent:
        """A request that goes on from a lookup: what was asked about and shown (the clinic, the
        doctor, the day) is what the booking is about, except where the message says otherwise.
        Only an operation that was clearly asked for starts one."""
        held = controller.state
        context = state.get("context") or {}
        shown, asked = context.get("shown") or {}, context.get("asked") or {}
        if (
            held.draft is not None
            or held.prepared is not None
            or held.awaiting != Awaiting.NOTHING
            or not (shown or asked)
            or intent.operation != "BOOK"
            or intent.message_type not in ("REQUEST", "AGREEMENT")
            or not controller.operation_vouched
        ):
            return intent
        names_doctor = bool(intent.doctor_name or intent.specialty)
        clinic = intent.clinic_name or shown.get("clinic") or asked.get("clinic_name")
        same_place = not intent.clinic_name or _same_name(intent.clinic_name, clinic)
        doctor = None
        if not names_doctor and same_place:
            doctor = shown.get("doctor") or asked.get("doctor_name")
        specialty = None if names_doctor else asked.get("specialty")
        first = shown.get("date") or asked.get("start_date")
        last = shown.get("date") or asked.get("end_date") or first
        held.draft = Draft(
            operation="BOOK",
            clinic_name=clinic if not intent.clinic_name else None,
            doctor_name=doctor,
            specialty=specialty,
            start_date=date.fromisoformat(first) if first else None,
            end_date=date.fromisoformat(last) if last else None,
            time_preference=asked.get("time_preference") or "ANYTIME",
        )
        return intent

    def _option_applied(
        self,
        intent: PatientIntent,
        state: AgentState,
        controller: HybridControllerModel,
    ) -> PatientIntent:
        """A pick from the choices on offer, resolved deterministically against the offers
        themselves (see `offers.resolve_offer`). The model's own number is never used: a reference
        that fits several offers picks none, and the patient is asked."""
        held = controller.state
        intent = intent.model_copy(update={"option_number": None})
        labels = self._offers(state, controller)
        if not labels:
            return intent
        dialogue = state.get("dialogue") or {}
        if held.offered:
            # The scheduler's own question: an explicit clock time answers a time question itself.
            if held.awaiting == Awaiting.TIME and intent.time:
                return intent
            number = resolve_offer(self._text, options_from(labels)).number
            if number is None and held.awaiting == Awaiting.CLINIC_CHOICE and intent.doctor_name:
                number = self._clinic_of(intent.doctor_name, labels)  # A fact from the directory.
            return intent if number is None else intent.model_copy(update={"option_number": number})
        shown = (state.get("context") or {}).get("shown") or {}
        offers = shown.get("offers") or []
        if (
            held.draft is not None
            or not offers
            or dialogue.get("awaiting") not in (None, "NOTHING")
        ):
            return intent
        number = resolve_offer(self._text, options_from(labels, offers)).number
        alone = (
            len(offers) == 1
            and offers[0].get("kind") == "opening"
            and not any(
                getattr(intent, f) for f in ("clinic_name", "doctor_name", "start_date", "time")
            )
            and (
                agreement_stated(self._text)
                or operation_stated("BOOK", self._text)
                or self._affirmed(state, self._text)
            )
        )
        if number is None and alone:
            number = 1  # "Yes, book that", said of the one opening that was listed.
        if number is None:
            return intent
        offer = offers[number - 1]
        kind = offer.get("kind", "opening")
        if kind == "clinic":
            return intent.model_copy(update={"clinic_name": offer["clinic"]})
        if kind == "doctor":
            update: dict[str, Any] = {"doctor_name": offer["doctor"]}
            if offer.get("clinic") and not intent.clinic_name:
                update["clinic_name"] = offer["clinic"]
            return intent.model_copy(update=update)
        held.draft = Draft(
            operation="BOOK",
            clinic_name=shown.get("clinic"),
            doctor_name=offer.get("doctor") or None,
            start_date=date.fromisoformat(offer["date"]),
            end_date=date.fromisoformat(offer["date"]),
        )
        controller.operation_vouched = True
        controller.pin_day(offer["date"])
        return intent.model_copy(
            update={"operation": "BOOK", "time": offer["time"], "selection": "SPECIFIC_TIME"}
        )

    def _clinic_of(self, doctor: str, labels: list[str]) -> int | None:
        """Which of the clinics on offer the doctor works at, from the directory: "the one with
        Dr. Moss" names a clinic by who is there. One match settles it; none or several do not."""
        result = LookupAnswerer(self.runtime, self.principal, self._clock)._call(
            "search_clinic_directory", {"doctor_name": doctor}
        )
        doctors = ((result or {}).get("data") or {}).get("doctors") or []
        places = {str(c["name"]).casefold() for d in doctors for c in d.get("clinics", [])}
        fits = [i for i, label in enumerate(labels, 1) if label.casefold() in places]
        return fits[0] if len(fits) == 1 else None

    def _lookup_if_only_asking(
        self,
        decision: PlannerDecision,
        intent: PatientIntent,
        held: Any,
        demoted: bool = False,
    ) -> PlannerDecision:
        """A message labelled as scheduling that states no operation, continues no request and
        picks no offer is a question about what exists, or too little to act on. (The reading's
        own say-so that it starts a booking does not count once an independent reading has said
        the message asks for no booking.)"""
        if (
            decision.route != "SCHEDULING"
            or intent.operation != "UNKNOWN"
            or intent.message_type != "REQUEST"
            or intent.option_number is not None
            or held.draft is not None
            or held.prepared is not None
            or held.awaiting != Awaiting.NOTHING
            or (decision.starts_booking and not demoted)
            or decision.same_request
        ):
            return decision
        today = self._clock().astimezone(UTC).date()
        asked = ground(
            decision.model_copy(update={"route": "LOOKUP", "subject": "OPENINGS"}),
            self._text,
            today,
        )
        named = bool(asked.clinic_name or asked.doctor_name or asked.specialty)
        timed = bool(asked.start_date or asked.week != "NONE")
        if asked.route != "LOOKUP" or not (named or timed):
            return decision  # Too little to look anything up: the scheduler asks what is wanted.
        if not timed and not asked.clinic_name:
            return asked.model_copy(update={"subject": "DOCTORS"})
        return asked

    @staticmethod
    def _inherit_lookup(decision: PlannerDecision, state: AgentState) -> PlannerDecision:
        """A question that names no place ("and Thursday?") is about the place just asked about."""
        context = state.get("context") or {}
        asked, shown = context.get("asked") or {}, context.get("shown") or {}
        if decision.subject not in ("OPENINGS", "NONE") or decision.clinic_name:
            return decision
        if decision.doctor_name:
            return decision  # The doctor's own clinics are looked up (see `_lookup`).
        clinic = asked.get("clinic_name") or shown.get("clinic")
        return decision.model_copy(update={"clinic_name": clinic}) if clinic else decision

    def _clinic_of_doctor(
        self, intent: PatientIntent, controller: HybridControllerModel
    ) -> PatientIntent:
        """A doctor named with no clinic: the directory says where they work. One clinic settles
        it; several, or none, leave the scheduler to ask."""
        held = controller.state
        draft = held.draft
        wants_booking = intent.operation == "BOOK" or (
            draft is not None and draft.operation == "BOOK"
        )
        doctor = intent.doctor_name or (draft.doctor_name if draft is not None else None)
        if (
            not wants_booking
            or not doctor
            or intent.clinic_name
            or (draft is not None and draft.clinic_name)
        ):
            return intent
        result = LookupAnswerer(self.runtime, self.principal, self._clock)._call(
            "search_clinic_directory", {"doctor_name": doctor}
        )
        found = ((result or {}).get("data") or {}).get("doctors") or []
        places = {str(c["name"]) for d in found for c in d.get("clinics", [])}
        if len(found) == 1 and len(places) == 1:
            return intent.model_copy(update={"clinic_name": places.pop()})
        return intent

    @staticmethod
    def _after_understand(state: AgentState) -> str:
        return {"LOOKUP": "lookup", "CLARIFY": "finish", "OUT_OF_SCOPE": "finish"}.get(
            state.get("route", ""), "schedule"
        )

    def _lookup(self, state: AgentState) -> dict[str, Any]:
        controller = self.model
        assert isinstance(controller, HybridControllerModel)
        previous = (state.get("context") or {}).get("shown") or {}
        offset = int(previous.get("next_offset", 0)) if more_options_requested(self._text) else 0
        answerer = LookupAnswerer(self.runtime, self.principal, self._clock, offer_offset=offset)
        decision = self._place_of_doctor(self._decision, state)
        answer = answerer.answer(decision, controller.state.draft, self._text)
        self.turn_calls = list(answer.calls)
        self._aside = "lookup"
        self._asked_now = self._criteria(decision, answer)
        self._open_now = answer.open_lookup.model_dump(mode="json") if answer.open_lookup else None
        if (answer.shown_clinic or answer.offers) and controller.state.draft is None:
            self._shown_now = {
                "clinic": answer.shown_clinic,
                "date": answer.shown_date.isoformat() if answer.shown_date else None,
                "window": [d.isoformat() for d in answer.shown_window]
                if answer.shown_window
                else None,
                "doctor": answer.shown_doctor,
                "offers": answer.offers,
                "offer_offset": answer.offer_offset,
                "next_offset": answer.offer_offset + len(answer.offers),
                "has_more": answer.has_more,
            }
        self._reply = AgentReply(
            f"{answer.text} {controller.resume_prompt()}", self._review_pending()
        )
        return {}

    @staticmethod
    def _place_of_doctor(decision: PlannerDecision, state: AgentState) -> PlannerDecision:
        """A doctor asked about with no clinic is the one the patient was just told about: the
        doctors a listing named carry the clinic they were listed at."""
        if decision.clinic_name or not decision.doctor_name:
            return decision
        offers = ((state.get("context") or {}).get("shown") or {}).get("offers") or []
        said = set(_name_tokens(decision.doctor_name))
        places = {
            o["clinic"]
            for o in offers
            if o.get("kind") == "doctor"
            and o.get("clinic")
            and said
            and said <= set(_name_tokens(o["doctor"]))
        }
        return (
            decision.model_copy(update={"clinic_name": places.pop()})
            if len(places) == 1
            else (decision)
        )

    @staticmethod
    def _criteria(decision: PlannerDecision, answer: Any) -> dict[str, Any] | None:
        """What a lookup was about, as the patient asked it: the next message may carry on from
        it ("book that", "Dr. Kim then"). Names and days only, as already grounded."""
        if decision.subject == "CLINICS" and not decision.clinic_name:
            return None
        criteria: dict[str, Any] = {
            "clinic_name": answer.shown_clinic or decision.clinic_name,
            "doctor_name": answer.shown_doctor or decision.doctor_name,
            "specialty": decision.specialty,
            "start_date": decision.start_date.isoformat() if decision.start_date else None,
            "end_date": decision.end_date.isoformat() if decision.end_date else None,
            "time_preference": decision.time_preference,
        }
        return {k: v for k, v in criteria.items() if v not in (None, "", "ANYTIME")} or None

    def _schedule(self, state: AgentState) -> dict[str, Any]:
        if state.get("kind") != "event":
            self.add_utterance(self._text)
        return {}

    def _decide(self, state: AgentState) -> dict[str, Any]:
        steps = state.get("steps", 0) + 1
        if steps > MAX_ACTIONS_PER_TURN:
            self._reply, self._action = AgentReply(FALLBACK_REPLY, False), None
            return {"steps": steps, "needs_tool": False}
        step = self.decide()
        if isinstance(step, ToolAction) and not self._may_act(step):
            step = AgentReply(ASK_TO_BOOK_REPLY, False)  # A review needs a clear request first.
        if isinstance(step, AgentReply):
            self._reply, self._action = step, None
        else:
            self._reply, self._action = None, step
        return {"steps": steps, "needs_tool": self._action is not None}

    def _may_act(self, action: ToolAction) -> bool:
        """Whether a review may be prepared now. The last line of defence against a lookup that was
        taken for a booking: whatever the readings said, a review is prepared only for a request
        that was clearly asked for (see `Draft.vouched`)."""
        if action.tool != "prepare_appointment_action":
            return True
        draft = self.model.state.draft if isinstance(self.model, HybridControllerModel) else None
        return bool(draft is not None and draft.vouched)

    @staticmethod
    def _after_decide(state: AgentState) -> str:
        return "tool" if state.get("needs_tool") else "finish"

    def _tool(self, state: AgentState) -> dict[str, Any]:
        assert self._action is not None
        action, self._action = self._action, None
        self.run_tool(action)
        return {"needs_tool": False}

    def _finish(self, state: AgentState) -> dict[str, Any]:
        controller = self.model
        assert isinstance(controller, HybridControllerModel)
        route = state.get("route", "SCHEDULING")
        if self._reply is None and route in ("CLARIFY", "OUT_OF_SCOPE"):
            self._reply = self._say_aside(route, controller)
        assert self._reply is not None
        self._final = self.conclude(self._reply)
        self._reply = None
        retiring = self.prepared_action_id if controller.state.prepared is None else None
        update = dehydrate(controller, retiring)
        aside = self._aside is not None
        context = dict(state.get("context") or {})
        if route == "SCHEDULING":
            context["shown"] = None  # Used once, or not at all.
            context["open_lookup"] = None
            context["asked"] = None
            context["booking_offered"] = False
        elif aside:
            context["shown"] = self._shown_now or context.get("shown")
            context["open_lookup"] = self._open_now
            context["asked"] = self._asked_now or context.get("asked")
            context["booking_offered"] = (
                self._aside == "lookup"
                and controller.state.draft is None
                and controller.state.prepared is None
            )
        update["context"] = context
        update["owner"] = self._owner
        update["interruptions"] = self._interruptions(state, controller, aside)
        update["needs_tool"] = False
        return update

    def _say_aside(self, route: str, controller: HybridControllerModel) -> AgentReply:
        self._aside = route.lower()
        if route == "OUT_OF_SCOPE":
            notice = EMERGENCY_NOTICE if self._decision.urgent else OUT_OF_SCOPE_NOTICE
            text = f"{notice} {controller.resume_prompt()}"
        else:
            about = self._decision.unclear_about
            text = QUESTIONS.get(about, QUESTIONS["NONE"])
            if about == "GOAL" and controller.state.draft is not None:
                text = f"{text} {controller.resume_prompt()}"
        return AgentReply(text, self._review_pending())

    @staticmethod
    def _interruptions(
        state: AgentState, controller: HybridControllerModel, aside: bool
    ) -> list[dict[str, Any]]:
        history = [dict(item) for item in state.get("interruptions", [])]
        if aside and controller.state.draft is not None:
            history.append(
                {
                    "turn": controller.state.turns,
                    "awaiting": str(controller.state.awaiting),
                    "resumed": False,
                }
            )
        elif not aside:
            for item in history:
                item["resumed"] = True
        return history[-MAX_INTERRUPTIONS:]

    # -- the TextSession interface ---------------------------------------------------------------

    def _run_graph(self, kind: str, payload: Any) -> AgentReply:
        self._pending_input = (kind, payload)
        self._final = None
        try:
            with self._exclusive():
                self.graph.invoke({"kind": kind}, self.config)
                self._prune()
        except ConversationBusy:
            self._pending_input = None
            return AgentReply(BUSY_REPLY, False)  # Nothing ran, so nothing changed.
        except (CheckpointStoreError, *STORE_FAULTS):
            self._pending_input = None
            self._retire_unsaved_review()
            return AgentReply(TROUBLE_REPLY, False)  # The saved state is as it was.
        except Exception:  # A fault in the graph must still leave the patient a reply.
            self._pending_input = None
            self._retire_unsaved_review()
            return self.conclude(AgentReply(FALLBACK_REPLY, False))
        return self._final or AgentReply(FALLBACK_REPLY, False)

    @contextlib.contextmanager
    def _exclusive(self):
        if self._store is None:
            yield
        else:
            with self._store.turn(self._thread):
                yield

    def _prune(self) -> None:
        if self._store is not None:
            with contextlib.suppress(Exception):  # Housekeeping never fails a finished turn.
                self._store.prune_history(self._thread)

    def _retire_unsaved_review(self) -> None:
        """A turn that failed after preparing a review, before its state was saved: the patient
        was never shown it, so retire it rather than leave it open."""
        fresh = self.prepared_action_id
        if not fresh or fresh == self._review_at_intake:
            return
        discard = getattr(self.runtime, "discard_pending_action", None)
        if discard is not None and self.principal is not None:
            with contextlib.suppress(Exception):
                discard(self.principal, fresh)
        self.prepared_action_id = self._review_at_intake

    def respond(self, utterance: str) -> AgentReply:
        return self._run_graph("utterance", utterance[:2000])

    def confirmation_event(self, payload: Mapping[str, Any]) -> AgentReply:
        return self._run_graph("event", dict(payload))

    def snapshot(self) -> dict[str, Any]:
        saved = self._saver.get_tuple(self.config)
        return dict(saved.checkpoint["channel_values"]) if saved else {}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._saver.delete_thread(self._thread)
