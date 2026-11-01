"""Scenario specifications: declarative, versioned, and tied to an approved EV id."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Engine

from app.evaluation.model import Action

PUBLIC = "public"
ANY = object()  # "Any value is acceptable here" in an expected argument.


@dataclass(frozen=True)
class Ref:
    """A fixture value, by label (an id, a name the scenario created)."""

    name: str


@dataclass(frozen=True)
class ResultRef:
    """A value from an earlier tool result, as the model would read it. Calls count from 1."""

    call: int
    path: str


@dataclass(frozen=True)
class LastResult:
    """The latest successful result of one tool, whatever its position in the conversation.

    A real model may search again before it acts, so "call 3" is not always the call that
    prepared the action. This is a judge-side reference like ResultRef: a model never uses it.
    """

    tool: str
    path: str


@dataclass(frozen=True)
class Said:
    """A value the patient stated in their request, named in the scenario's `request_terms`.

    This and ResultRef are the only ways a scripted model may obtain a value. Using Ref (a fixture
    the patient never said and no tool returned) in a model action is a hidden-knowledge failure.
    """

    name: str


@dataclass(frozen=True)
class Pick:
    """The one item of a list in an earlier result whose fields match what the patient said.

    This is how a model chooses among several results by patient-visible attributes, such as an
    appointment at the clinic the patient named. No match, or more than one, is a grounding
    failure: a model should ask, not guess.
    """

    call: int
    items: str  # Dotted path to the list inside the result envelope.
    where: Mapping[str, Any]  # Dotted path within an item -> a Said value or constant.
    field: str  # Dotted path within the chosen item to return.


@dataclass(frozen=True)
class ResultCheck:
    """An assertion about one value inside a tool result envelope, found by dotted path."""

    path: str
    op: str  # eq, ge, len_eq, len_ge, contains, endswith, absent
    value: Any = None


@dataclass(frozen=True)
class ExpectedCall:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)  # A subset; Ref values allowed.
    outcome: str = "ok"  # "ok", or the expected error code.
    checks: tuple[ResultCheck, ...] = ()


@dataclass(frozen=True)
class DbCheck:
    """An expectation about authoritative database state, as a scalar SQL query.

    Parameters may be Ref or ResultRef. `:scenario_start` (the database clock when the scenario
    began) is always available.
    """

    name: str
    sql: str
    expected: Any
    params: Mapping[str, Any] = field(default_factory=dict)


# --- the timeline: who acts, in order ---------------------------------------------------------


@dataclass(frozen=True)
class ModelTurn:
    """Run the model until it speaks.

    If it ends on a question before any action has been prepared, the synthetic patient answers
    with the next of `follow_ups` and the model continues the same turn. A reply is plain
    conversation, fixed by the scenario in advance, and never carries a fixture value, an
    identifier, or a secret. It is not a confirmation: only the trusted UI commits. When the
    request already answers the question, `question_is_unnecessary` records the question as a
    failure, whether or not the conversation then completes.
    """

    follow_ups: tuple[str, ...] = ()
    question_is_unnecessary: bool = True


@dataclass(frozen=True)
class PatientSays:
    """The synthetic patient says something more, whatever the agent last did.

    This is how a scenario changes its mind after a review is offered, corrects the agent, or
    drops the request. It is conversation only, fixed in advance: nothing is prepared, confirmed or
    booked by it. `{name}` marks a fixture value named in the scenario's `request_terms`, such as
    a clinic the patient is naming; the text never carries an identifier or a secret.
    """

    text: str


@dataclass(frozen=True)
class PendingIntent:
    """What the patient actually wanted, written from the scenario, not from the model's output.

    Before pressing Confirm the UI driver reads the stored pending action and checks it against
    this. A prepared action that differs is never confirmed.
    """

    action: str  # BOOK, RESCHEDULE, or CANCEL.
    target: Any  # The slot (BOOK) or appointment (RESCHEDULE, CANCEL) the action concerns.
    expected: Mapping[str, Any] = field(default_factory=dict)  # Entries of expected_versions.
    summary: Mapping[str, Any] = field(default_factory=dict)  # Dotted paths in the summary shown.


@dataclass(frozen=True, kw_only=True)
class UiConfirm:
    """The trusted UI confirms a pending action, if it matches the patient's intent.

    The action is the one prepared by tool call number `call`, or the latest successful result of
    the tool named by `tool` (for a real model, whose calls do not fall in a fixed order).
    `expect` is "COMMITTED" or the error code the backend must answer with once confirmed.
    """

    intent: PendingIntent
    call: int | None = None
    tool: str | None = None
    expect: str = "COMMITTED"
    # With `tool`: which of the successful preparations, in order (0 is the first). The default,
    # -1, is the latest. A patient pressing Confirm on an earlier, out-of-date review screen is
    # how a scenario tests that a retired review cannot commit.
    nth: int = -1
    # The screen being pressed is out of date and does not know it: the simulated patient presses
    # Confirm on it although the stored action is no longer ACTIVE. Used only to show that a
    # retired review is refused by the backend.
    stale: bool = False

    def __post_init__(self) -> None:
        if (self.call is None) == (self.tool is None):
            raise ValueError("a UI confirmation names exactly one of call and tool")


@dataclass(frozen=True)
class World:
    """Something outside the conversation: time passing, another patient acting."""

    name: str
    action: Callable[["WorldContext"], None]


@dataclass(frozen=True)
class Checkpoint:
    """Database expectations to hold at this moment, for example after preparation."""

    name: str
    checks: tuple[DbCheck, ...]


Turn = ModelTurn | PatientSays | UiConfirm | World | Checkpoint


@dataclass
class Fixtures:
    """The synthetic data a scenario created, by label. Values never reach a report; labels do."""

    values: dict[str, Any] = field(default_factory=dict)
    slots: list[Any] = field(default_factory=list)  # Every slot the scenario owns.
    protected: list[str] = field(default_factory=list)  # Text that must never reach the model.
    actor_code: str | None = None  # The patient the conversation is with; None for the public.


@dataclass(frozen=True)
class SetupContext:
    engine: Engine
    now: datetime
    suffix: str  # A per-run unique suffix for names, so repeated runs never collide.
    pin: str  # For creating this run's own synthetic patients. Never given to a model.


@dataclass(frozen=True)
class WorldContext:
    engine: Engine
    clock: Any
    fixtures: Fixtures


@dataclass(frozen=True)
class Scenario:
    id: str  # The approved EV id, for example "EV-001".
    title: str  # The approved title.
    delivery: str  # Which approved delivery phase assigns it here.
    request: str  # What the synthetic patient says: a template with no personal data.
    actor: str  # PUBLIC, or "patient" (the patient is named by the fixtures).
    setup: Callable[[SetupContext], Fixtures]
    model_actions: Sequence[Action]
    timeline: Sequence[Turn]
    expected_calls: Sequence[ExpectedCall]
    expected_claims: frozenset[str]
    db_checks: Sequence[DbCheck]
    forbidden_claims: frozenset[str] = frozenset()
    forbidden_tools: frozenset[str] = frozenset()
    request_terms: Sequence[str] = ()  # Fixture values the patient states, filled into `request`.
    # Narrowing fields the patient authorized, grouped by tool. This is independent of the
    # reference call order so an incorrect or repeated call is still judged against patient intent.
    # When omitted, the runner derives a compatibility policy from expected calls and request terms.
    authorized_filters: Mapping[str, frozenset[str]] | None = None
    version: str = "1"
    # A multi-turn conversation has no fixed order of calls to compare against. When False, the
    # calls are judged by what they did (filters, arguments, results, end state), not by position.
    check_call_order: bool = True
