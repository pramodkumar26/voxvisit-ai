"""What a model is, to the harness: something that looks at the conversation and decides.

Today's implementation replays a fixed script, so every run is identical. A real LLM adapter will
implement the same `decide` method. The runner trusts nothing a model returns: every action goes
through the tool runtime's allowlist, authorization, and validation.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# Claims an assistant message can make. A success claim is allowed only after the trusted UI
# reported the matching committed result (docs/agent-tool-contracts.md, "Confirmation Boundary").
BOOKED = "BOOKED"
RESCHEDULED = "RESCHEDULED"
CANCELLED = "CANCELLED"
SUCCESS_CLAIMS = frozenset({BOOKED, RESCHEDULED, CANCELLED})
PENDING_REVIEW = "PENDING_REVIEW"
REQUEST_VERIFICATION = "REQUEST_VERIFICATION"
ASK_CLARIFICATION = "ASK_CLARIFICATION"
RESCHEDULE_FAILED = "RESCHEDULE_FAILED"
OFFER_NEW_SEARCH = "OFFER_NEW_SEARCH"
SELF_SERVICE_UNAVAILABLE = "SELF_SERVICE_UNAVAILABLE"
OFFER_STAFF_ASSISTANCE = "OFFER_STAFF_ASSISTANCE"


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Say:
    """The end of the model's turn: a message to the patient.

    `text` is the message itself. A real model supplies only text. The harness classifies it
    independently (see app.evaluation.claims) and never records it, so a report stays free of
    patient content. `claims` are labels the *model side declared*. They are not trusted: with text
    present they are only cross-checked, and without text they are the scripted stub's stand-in for
    a message, reported as unverified.
    """

    claims: frozenset[str] = frozenset()
    text: str | None = None


Action = ToolAction | Say


@dataclass(frozen=True)
class Message:
    role: Literal["user", "tool", "trusted_event"]
    content: Mapping[str, Any]


class MalformedModelResponse(Exception):
    """An adapter could not turn a provider's reply into a decision.

    `problem` is one of a small set of fixed descriptions chosen by the adapter. It never contains
    any of the provider's text, so reporting it cannot leak model output or patient content.
    """

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem


class ScriptExhaustedError(Exception):
    """The script ran out of actions while the scenario still expected the model to act."""


class ModelGroundingError(Exception):
    """A decision could not be made from what the patient said and the tools returned.

    A scripted stub raises this when an argument refers to something it was never told (no matching
    item in an earlier result, an unknown call). A real model adapter may raise it for the same
    reason. The runner records it as an evaluation failure.
    """


class HiddenKnowledgeError(ModelGroundingError):
    """A decision used a fixture value the patient never stated and no tool ever returned."""


class Model(Protocol):
    name: str

    def decide(self, transcript: Sequence[Message]) -> Action: ...


class ScriptedModel:
    """Replays a fixed list of actions. Argument placeholders are resolved when decided, from
    information the model would legitimately have by then (fixture ids and earlier results)."""

    name = "scripted-stub"

    def __init__(
        self,
        actions: Sequence[Action],
        resolve: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> None:
        self._actions = list(actions)
        self._resolve = resolve
        self._next = 0

    def decide(self, transcript: Sequence[Message]) -> Action:
        if self._next >= len(self._actions):
            raise ScriptExhaustedError
        action = self._actions[self._next]
        self._next += 1
        if isinstance(action, ToolAction):
            return ToolAction(action.tool, self._resolve(action.arguments))
        return action
