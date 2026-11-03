"""LG-1: the text conversation as a LangGraph `StateGraph`, beside the existing controller.

This is a separate integration path. Nothing live uses it yet: the voice agent and the chat route
still build a `TextSession`. `GraphSession` is a `TextSession` (same `respond`,
`confirmation_event`, `close`, same registry), so it can replace one later without a change to its
callers.

What the graph does in LG-1 is only to run a turn as graph steps, with a checkpoint after it:

    intake -> decide -> (tool -> decide)* -> finish

Each step calls the same `TextSession` method the old loop calls, so scheduling behaviour,
authorization, tool routing and review retirement are unchanged. The model is not given anything
new to plan with.

One owner of conversation state. The controller's `ConversationState` stays the only authoritative
record of the request. The graph state holds a *projection* of it (`project`), written once per turn
by `finish` and never read back to drive the controller, so the two cannot disagree: a lost or
stale checkpoint changes nothing the agent does. The projection is what LG-2 will make
authoritative, one field at a time.

What is never in a checkpoint: the principal, the session id (the thread id is a one-way hash of
it), CSRF or any token, a PIN, the tool runtime, the model, the transcript, tool results, and the
words the patient said. Those stay on the session object, in memory, as before. A checkpoint holds
the request's own fields (clinic, doctor, dates, preferences), the question being waited on and the
choices offered, the open interruptions, and the id of the review awaiting the UI's Confirm.

The words of a reply go to the caller from the session object, not through a checkpoint.

Checkpoints are in memory (`InMemorySaver`), for local development. They are deleted when the
session is closed: at sign-out, expiry, eviction or when the session id changes patient.
"""

import contextlib
import hashlib
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.auth.service import Principal
from app.evaluation.conversation import (
    FALLBACK_REPLY,
    MAX_ACTIONS_PER_TURN,
    AgentReply,
    ConversationState,
    TextSession,
)
from app.evaluation.hybrid import HybridControllerModel, PlannerHints
from app.evaluation.model import Model, ToolAction
from app.evaluation.planner import (
    OUT_OF_SCOPE_NOTICE,
    QUESTIONS,
    GoalPlanner,
    LookupAnswerer,
    PlannerDecision,
    merge_open,
    planner_context,
)

MAX_INTERRUPTIONS = 5  # The most recent are kept.


class GraphState(TypedDict, total=False):
    # -- this turn: set by `intake`, gone from meaning once `finish` has run -------------------
    kind: str  # "utterance" or "event"
    route: str  # SCHEDULING (the controller), LOOKUP, CLARIFY or OUT_OF_SCOPE
    steps: int
    needs_tool: bool
    # -- the projection of the conversation, kept between turns --------------------------------
    turn: int
    goal: dict[str, Any]  # What the patient wants done, and to which appointment, where, when.
    preferences: dict[str, Any]  # How they want it: time of day, a time, "earliest", a window.
    clarification: dict[str, Any]  # The question being waited on.
    offered: list[dict[str, str]]  # The choices put to the patient, in the order offered.
    interruptions: list[dict[str, Any]]  # Asides taken without leaving the request.
    review: dict[str, Any] | None  # The change awaiting the UI's Confirm, by id only.
    planner: dict[str, Any]  # What the planner owns: its last route and any lookup left open.


GOAL_FIELDS = (
    "operation",
    "clinic_name",
    "doctor_name",
    "specialty",
    "start_date",
    "end_date",
    "new_date",
    "appointment_time",
)
PREFERENCE_FIELDS = ("time_preference", "time", "selection", "not_before", "not_after")


def project(state: ConversationState, *, review_id: str | None) -> dict[str, Any]:
    """The checkpointed view of the controller's state. Plain JSON values only."""
    draft = state.draft.model_dump(mode="json") if state.draft is not None else {}
    return {
        "goal": {name: draft.get(name) for name in GOAL_FIELDS if draft.get(name) is not None},
        "preferences": {
            name: draft.get(name)
            for name in PREFERENCE_FIELDS
            if draft.get(name) not in (None, "", "ANYTIME", "UNSPECIFIED")
        },
        "clarification": {"awaiting": str(state.awaiting), "question": state.last_question}
        if state.last_question or str(state.awaiting) != "NOTHING"
        else {},
        "offered": [{"label": o.label, "key": o.key} for o in state.offered],
        "review": {"pending_action_id": review_id} if review_id and state.prepared else None,
        "turn": state.turns,
    }


def thread_id_for(principal: Principal | None, fallback: str) -> str:
    """The checkpoint thread of a session: a hash, so the id of the session is not stored."""
    source = f"session:{principal.session_id}" if principal else f"anonymous:{fallback}"
    return hashlib.sha256(source.encode()).hexdigest()[:32]


class GraphSession(TextSession):
    """A `TextSession` whose turns run as a LangGraph graph with a session-scoped checkpoint."""

    def __init__(
        self,
        model: Model,
        runtime: Any,
        principal: Principal | None,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        planner: GoalPlanner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(model, runtime, principal)
        # The planner works only beside the hybrid controller, whose state it must not disturb.
        self._planner = planner if isinstance(model, HybridControllerModel) else None
        self._clock = clock or (lambda: datetime.now(UTC))
        self._open_lookup: PlannerDecision | None = None
        self._text = ""
        self._shown: tuple[str | None, Any, Any, str | None] = (None, None, None, None)
        self._decision = PlannerDecision(route="SCHEDULING")
        self._aside: str | None = None
        self._saver = checkpointer or InMemorySaver()
        self._thread = thread_id_for(principal, uuid.uuid4().hex)
        self._action: ToolAction | None = None  # The tool to run next; never checkpointed.
        self._reply: AgentReply | None = None
        self._final: AgentReply | None = None
        self._pending_input: tuple[str, Any] | None = None
        self.graph = self._build()

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self._thread}}

    # -- the graph ------------------------------------------------------------------------------

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("intake", self._intake)
        graph.add_node("plan", self._plan)
        graph.add_node("handoff", self._handoff)
        graph.add_node("lookup", self._lookup)
        graph.add_node("decide", self._decide)
        graph.add_node("tool", self._tool)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "plan")
        graph.add_conditional_edges("plan", self._after_plan, ["handoff", "lookup", "finish"])
        graph.add_edge("handoff", "decide")
        graph.add_edge("lookup", "finish")
        graph.add_conditional_edges("decide", self._after_decide, ["tool", "finish"])
        graph.add_edge("tool", "decide")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self._saver)

    def _intake(self, state: GraphState) -> dict[str, Any]:
        assert self._pending_input is not None
        kind, payload = self._pending_input
        self._pending_input = None
        self._reply, self._action, self._aside = None, None, None
        if kind == "event":
            self.add_event(payload)
        else:
            self._text = payload  # Handed to the controller only if the plan says it is its own.
        return {"kind": kind, "steps": 0, "needs_tool": False}

    def _plan(self, state: GraphState) -> dict[str, Any]:
        """Decide whose message this is. An event from the UI is always the controller's."""
        if state.get("kind") == "event" or self._planner is None:
            return {"route": "SCHEDULING"}
        model = self.model
        assert isinstance(model, HybridControllerModel)
        if model.screens_secret(self._text):
            return {"route": "SCHEDULING"}  # The controller refuses it; no model reads it.
        draft = model.state.draft
        decision = self._planner.plan(
            self._text,
            context=planner_context(model, self._open_lookup, self._shown),
            today=self._clock().astimezone(UTC).date(),
            anchor=getattr(draft, "start_date", None),
        )
        self._decision = merge_open(decision, self._open_lookup)
        if self._decision.route == "SCHEDULING":
            self._open_lookup = None  # The patient has moved on to the request itself.
        return {"route": self._decision.route}

    @staticmethod
    def _after_plan(state: GraphState) -> str:
        return {"LOOKUP": "lookup", "CLARIFY": "finish", "OUT_OF_SCOPE": "finish"}.get(
            state.get("route", ""), "handoff"
        )

    def _handoff(self, state: GraphState) -> dict[str, Any]:
        """Pass the message to the controller, with what the planner understood of it."""
        model = self.model
        if isinstance(model, HybridControllerModel):
            clinic, day, window, doctor = self._shown
            model.hints = PlannerHints(
                same_request=self._decision.same_request and model.state.draft is not None,
                keep_goal=self._planner is not None,
                carry_clinic=clinic if model.state.draft is None else None,
                carry_date=day if model.state.draft is None else None,
                carry_window=window if model.state.draft is None else None,
                carry_doctor=doctor if model.state.draft is None else None,
                start_booking=self._decision.starts_booking and model.state.draft is None,
            )
            self._shown = (None, None, None, None)  # Used once, or not at all.
        self.add_utterance(self._text)
        return {}

    def _lookup(self, state: GraphState) -> dict[str, Any]:
        """Read-only tools, chosen by the subject the planner found; the words are built here."""
        model = self.model
        assert isinstance(model, HybridControllerModel)
        answerer = LookupAnswerer(self.runtime, self.principal, self._clock)
        answer = answerer.answer(self._decision, model.state.draft, self._text)
        self.turn_calls = list(answer.calls)
        self._open_lookup = answer.open_lookup
        if answer.shown_clinic and model.state.draft is None:
            self._shown = (
                answer.shown_clinic,
                answer.shown_date,
                answer.shown_window,
                answer.shown_doctor,
            )
        self._aside = "lookup"
        self._reply = AgentReply(f"{answer.text} {model.resume_prompt()}", self._review_pending())
        return {}

    def _decide(self, state: GraphState) -> dict[str, Any]:
        steps = state.get("steps", 0) + 1
        if steps > MAX_ACTIONS_PER_TURN:
            self._reply, self._action = AgentReply(FALLBACK_REPLY, False), None
            return {"steps": steps, "needs_tool": False}
        step = self.decide()
        if isinstance(step, AgentReply):
            self._reply, self._action = step, None
        else:
            self._reply, self._action = None, step
        return {"steps": steps, "needs_tool": self._action is not None}

    @staticmethod
    def _after_decide(state: GraphState) -> str:
        return "tool" if state.get("needs_tool") else "finish"

    def _tool(self, state: GraphState) -> dict[str, Any]:
        assert self._action is not None
        action, self._action = self._action, None
        self.run_tool(action)
        return {"needs_tool": False}

    def _finish(self, state: GraphState) -> dict[str, Any]:
        route = state.get("route", "SCHEDULING")
        if self._reply is None and route in ("CLARIFY", "OUT_OF_SCOPE"):
            self._reply = self._say_aside(route)
        assert self._reply is not None
        self._final = self.conclude(
            self._reply
        )  # The words stay on the session, not in a checkpoint.
        self._reply = None
        controller = getattr(self.model, "state", None)
        update: dict[str, Any] = {"needs_tool": False}
        if controller is not None:
            update |= project(controller, review_id=self.prepared_action_id)
            update["interruptions"] = self._interruptions(state, controller)
        if self._planner is not None:
            update["planner"] = {
                "route": route,
                "open_lookup": self._open_lookup.model_dump(mode="json", exclude_defaults=True)
                if self._open_lookup
                else None,
            }
        return update

    def _say_aside(self, route: str) -> AgentReply:
        """A question or a notice that does not touch the request in progress."""
        model = self.model
        assert isinstance(model, HybridControllerModel)
        self._aside = route.lower()
        if route == "OUT_OF_SCOPE":
            text = f"{OUT_OF_SCOPE_NOTICE} {model.resume_prompt()}"
        else:
            about = self._decision.unclear_about
            text = QUESTIONS.get(about, QUESTIONS["NONE"])
            if about == "GOAL" and model.state.draft is not None:
                text = f"{text} {model.resume_prompt()}"
        return AgentReply(text, self._review_pending())

    def _interruptions(self, state: GraphState, controller: ConversationState) -> list[dict]:
        """An aside is a directory lookup made while a request was open. It is resumed when a later
        turn goes back to the request without another aside."""
        history = [dict(item) for item in state.get("interruptions", [])]
        asked = self._aside is not None or any(
            tool == "search_clinic_directory" and arguments.get("browse")
            for tool, arguments in self.turn_calls
        )
        if asked and controller.draft is not None:
            history.append(
                {
                    "kind": self._aside or "directory_question",
                    "turn": controller.turns,
                    "awaiting": str(controller.awaiting),
                    "resumed": False,
                }
            )
        elif not asked:
            for item in history:
                item["resumed"] = True
        return history[-MAX_INTERRUPTIONS:]

    # -- the TextSession interface --------------------------------------------------------------

    def _run_graph(self, kind: str, payload: Any) -> AgentReply:
        self._pending_input = (kind, payload)
        self._final = None
        try:
            self.graph.invoke({"kind": kind}, self.config)
        except Exception:  # A fault in the graph itself must still leave the patient a reply.
            self._pending_input = None
            return self.conclude(AgentReply(FALLBACK_REPLY, False))
        return self._final or AgentReply(FALLBACK_REPLY, False)

    def respond(self, utterance: str) -> AgentReply:
        return self._run_graph("utterance", utterance)

    def confirmation_event(self, payload: Mapping[str, Any]) -> AgentReply:
        return self._run_graph("event", dict(payload))

    def snapshot(self) -> dict[str, Any]:
        """What is checkpointed for this session (the projection only), for tests and tools."""
        saved = self._saver.get_tuple(self.config)
        return dict(saved.checkpoint["channel_values"]) if saved else {}

    def close(self) -> None:
        """Forget this session's checkpoints. Called when the session ends or is thrown away."""
        with contextlib.suppress(Exception):  # Cleanup must never stop a sign-out.
            self._saver.delete_thread(self._thread)
