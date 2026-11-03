"""A real text-based LLM behind the evaluation `Model` protocol (a local Ollama server).

The model sees exactly three things: the patient's request, the tool results the runtime returned,
and short notices from the trusted interface about what the patient pressed. It is offered the four
approved tools by name, description and parameter schema (`registry.definitions()`), none of which
takes an identity, session, idempotency key or confirmation. It never receives a credential, a
fixture identifier, or an expected answer, and it has no way to confirm anything: the runner sends
every tool call through the same allowlist, authentication and validation a scripted model faces.

Everything a provider can get wrong is bounded here: one wall-clock timeout per call, an output
token limit, a cap on decisions per conversation, a cap on message and result sizes, and a reply
that cannot be read becomes `MalformedModelResponse` (reported as a failed decision, never
executed). Provider text and error bodies are never copied into errors or reports.

The default transport talks to a loopback address only. A reply counts as coming from a real model
only when the default transport received it; an injected transport (used by unit tests) is reported
as such and never as a real-model result.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from app.evaluation.model import Action, MalformedModelResponse, Message, Say, ToolAction

PROMPT_VERSION = "5"  # Earlier versions and their measured results are listed in the docs.
APPROACH_A_PROMPT_VERSION = "ab-a2"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """\
You are VoxVisit, a text assistant that helps patients of fictional clinics find, book, move and \
cancel their own appointments. You can act only through the tools you are given.

Rules:
- Use only what the patient has told you and what tool results contain. Dates are YYYY-MM-DD.
- Clinic, doctor, appointment and slot identifiers are long opaque values (UUIDs) that appear only \
in tool results. The patient does not know them and they are not part of any name. Never invent \
one. When the patient names a clinic or doctor, your first step is search_clinic_directory with \
that name; copy the identifiers you need for other tools from its result.
- Look things up with the tools instead of asking the patient for details a tool can find.
- Search filters represent patient preferences. Use a doctor, specialty, clinic, date or time \
filter only when the patient requested it. Seeing a value in a tool result does not make it a \
patient preference. You may copy an identifier from a result when it is required to represent a \
clinic, doctor, appointment or slot the patient already selected.
- If the request is ambiguous, or a tool reports several possible matches, ask the patient one \
short clarifying question that lists the options. Do not guess, and do not prepare an action until \
the patient has clarified.
- When the request already identifies what to do (for example, "the first available" time means \
the earliest time a search returns), prepare it rather than asking the patient to choose again: \
preparing is safe because it changes nothing until the patient reviews and confirms it.
- A clear request to book, reschedule or cancel is enough to prepare the requested action. Do not \
ask for another verbal confirmation before preparation. The separate interface confirmation is \
what authorizes execution.
- prepare_appointment_action only prepares a change. Nothing is booked, moved or cancelled until \
the patient presses Confirm in the app. After preparing, tell the patient to review the details \
and confirm in the app. You cannot confirm anything yourself.
- For prepare_appointment_action, `action` is one exact uppercase string: `BOOK`, `RESCHEDULE`, or \
`CANCEL`. Never wrap the action in an object or list. `BOOK` uses `slot_id`; `RESCHEDULE` uses \
`appointment_id` and `new_slot_id`; `CANCEL` uses `appointment_id`.
- Do not say an appointment is booked, rescheduled or cancelled unless a notice from the app says \
the patient's confirmation succeeded.
- Never ask for, repeat or guess a PIN, password, token or other credential.
- Keep every reply short and plain.
"""

APPROACH_A_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "When the patient names a clinic or doctor, your first step is search_clinic_directory with "
    "that name; copy the identifiers you need for other tools from its result.",
    "For booking, resolve a named clinic or doctor with search_clinic_directory before searching "
    "availability. For cancellation or rescheduling, first use get_my_appointments to locate the "
    "patient's own appointment; do not search the public directory merely because the request "
    "names its clinic. get_my_appointments accepts only status, date range, and cursor fields. "
    "Copy only "
    "the identifiers required by the next appropriate tool.",
).replace(
    'When the request already identifies what to do (for example, "the first available" time '
    "means the earliest time a search returns), prepare it rather than asking the patient to "
    "choose again: preparing is safe because it changes nothing until the patient reviews and "
    "confirms it.",
    "Availability results are ordered earliest first. When the request clearly asks for the first, "
    "earliest, soonest, or next available time, use the first eligible result and prepare it "
    "without asking the patient to choose again. Preparing is safe because it changes nothing "
    "until the "
    "patient reviews and confirms it.",
)

APPROACH_A_WORKFLOW = """
Workflow (follow it exactly):
1. Use the tools to find what the request needs.
2. The moment the request and the results point to one appointment or one time, call \
prepare_appointment_action in that same turn. The patient's request is already their instruction, \
and the app's review screen is where they confirm. So never end a turn with a question such as \
"Would you like me to book this?", "Is this the appointment you mean?" or "Shall I cancel it?".
3. After prepare_appointment_action succeeds, reply in one or two sentences: say what you \
prepared (the doctor, the time, the date, and the clinic) and ask the patient to review and \
confirm it in the app.
4. Ask a question only when the request cannot be carried out without it: the date or clinic is \
missing, or several different doctors or appointments match. List the options in the question.
5. "Later" or "earlier" than an appointment means the closest such time that is open. A part of \
the day ("morning", "afternoon") means only times in it. Pick the earliest open time that fits, \
and do not ask the patient to choose among times that all fit.
"""
APPROACH_A_SYSTEM_PROMPT = APPROACH_A_SYSTEM_PROMPT + APPROACH_A_WORKFLOW

# Notices from the trusted interface, worded for the model. They never contain patient content.
COMMITTED_NOTICE = {
    "BOOKED": "The patient pressed Confirm in the app and the appointment was booked.",
    "RESCHEDULED": "The patient pressed Confirm in the app and the appointment was rescheduled.",
    "CANCELLED": "The patient pressed Confirm in the app and the appointment was cancelled.",
}
WITHHELD_NOTICE = "The patient did not confirm the prepared action. Nothing was changed."


class ModelConfigurationError(ValueError):
    """The adapter was configured unsafely (for example, a non-loopback server)."""


class ModelTimeoutError(Exception):
    """The provider did not answer within the configured time."""


class ModelUnavailableError(Exception):
    """The provider could not be reached or answered with an error."""


class ModelCallLimitError(Exception):
    """The conversation used its whole budget of model calls."""


@dataclass(frozen=True)
class LlmConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    request_timeout_s: float = 120.0  # One call, wall clock; a local 7B model may need to load.
    max_output_tokens: int = 512
    context_tokens: int = 8192
    temperature: float = 0.0
    seed: int | None = 7
    max_decisions: int = 10  # Model calls per conversation: the provider-cost bound.
    max_message_chars: int = 2000  # A longer reply is cut before it is shown or classified.
    max_result_chars: int = 12000  # A longer tool result is cut before it reaches the model.
    allow_remote: bool = False
    think: bool | None = None  # Reasoning mode for models that have one; None leaves the default.
    keep_alive: str | None = None  # How long the server keeps the model loaded, e.g. "30m".

    def __post_init__(self) -> None:
        host = urlparse(self.base_url).hostname
        if not self.allow_remote and host not in LOOPBACK_HOSTS:
            raise ModelConfigurationError("the model server must be on this machine")
        if self.max_decisions < 1 or self.request_timeout_s <= 0 or self.max_output_tokens < 1:
            raise ModelConfigurationError("limits must be positive")


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def http_transport(base_url: str) -> Transport:
    """The real transport: one non-streaming POST, errors reduced to their category."""

    def post(path: str, payload: Mapping[str, Any], timeout_s: float) -> Mapping[str, Any]:
        timeout = httpx.Timeout(timeout_s, connect=min(5.0, timeout_s))
        try:
            response = httpx.post(base_url.rstrip("/") + path, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise ModelTimeoutError from None
        except httpx.HTTPError:
            raise ModelUnavailableError from None
        if response.status_code != 200:
            raise ModelUnavailableError
        try:
            body = response.json()
        except ValueError:
            raise MalformedModelResponse("the reply is not JSON") from None
        if not isinstance(body, dict):
            raise MalformedModelResponse("the reply is not an object")
        return body

    return post


def tool_schemas(definitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": definition["name"],
                "description": definition["description"],
                "parameters": definition["parameters"],
            },
        }
        for definition in definitions
    ]


@dataclass
class CallRecord:
    duration_ms: float
    outcome: str  # "tool_call", "message", or the problem
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    server_ms: float | None = None  # The server's own measurement of the whole call.
    load_ms: float | None = None  # Loading the model into memory (non-zero only when it was cold).
    prefill_ms: float | None = None  # Reading the prompt.
    decode_ms: float | None = None  # Writing the reply.


def record_of(duration_ms: float, outcome: str, body: Mapping[str, Any] | None) -> CallRecord:
    """A call record from the server's reply, with its own timings when it gave them."""
    body = body or {}

    def count(name: str) -> int | None:
        return body[name] if isinstance(body.get(name), int) else None

    def millis(name: str) -> float | None:
        return body[name] / 1e6 if isinstance(body.get(name), int) else None

    return CallRecord(
        duration_ms=duration_ms,
        outcome=outcome,
        prompt_tokens=count("prompt_eval_count"),
        output_tokens=count("eval_count"),
        server_ms=millis("total_duration"),
        load_ms=millis("load_duration"),
        prefill_ms=millis("prompt_eval_duration"),
        decode_ms=millis("eval_duration"),
    )


@dataclass
class LlmUsage:
    calls: list[CallRecord] = field(default_factory=list)
    extra_tool_calls_dropped: int = 0
    messages_truncated: int = 0
    results_truncated: int = 0

    def as_dict(self) -> dict[str, Any]:
        durations = [call.duration_ms for call in self.calls]
        return {
            "model_calls": len(self.calls),
            "prompt_tokens": sum(call.prompt_tokens or 0 for call in self.calls),
            "output_tokens": sum(call.output_tokens or 0 for call in self.calls),
            "latency_ms": [round(value, 1) for value in durations],
            "server_reported_ms": [
                None if call.server_ms is None else round(call.server_ms, 1) for call in self.calls
            ],
            "extra_tool_calls_dropped": self.extra_tool_calls_dropped,
            "messages_truncated": self.messages_truncated,
            "results_truncated": self.results_truncated,
        }


class OllamaModel:
    """One conversation with a local model. Create a new instance per scenario run."""

    def __init__(
        self,
        definitions: Sequence[Mapping[str, Any]],
        config: LlmConfig | None = None,
        *,
        transport: Transport | None = None,
        timer: Callable[[], float] = time.perf_counter,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.config = config or LlmConfig()
        self.name = f"ollama:{self.config.model}"
        self.real_transport = transport is None
        self._post = transport or http_transport(self.config.base_url)
        self._tools = tool_schemas(definitions)
        self._timer = timer
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._seen = 0  # How much of the runner's transcript has been absorbed.
        self.usage = LlmUsage()
        self.served_replies = 0  # Replies actually received from a provider.

    # -- the Model protocol -------------------------------------------------------------------

    def decide(self, transcript: Sequence[Message]) -> Action:
        if len(self.usage.calls) >= self.config.max_decisions:
            raise ModelCallLimitError
        self._absorb(transcript)
        payload = {
            "model": self.config.model,
            "messages": self._messages,
            "tools": self._tools,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
                "num_ctx": self.config.context_tokens,
                **({"seed": self.config.seed} if self.config.seed is not None else {}),
            },
        }
        started = self._timer()
        try:
            body = self._post("/api/chat", payload, self.config.request_timeout_s)
        except Exception as error:
            self._record(started, type(error).__name__)
            raise
        self.served_replies += 1
        try:
            action, assistant = self._parse(body)
        except MalformedModelResponse as error:
            self._record(started, error.problem, body)
            raise
        self._record(started, "tool_call" if isinstance(action, ToolAction) else "message", body)
        self._messages.append(assistant)
        return action

    # -- building the conversation ------------------------------------------------------------

    def _absorb(self, transcript: Sequence[Message]) -> None:
        for message in transcript[self._seen :]:
            if message.role == "user":
                text = str(message.content.get("request", ""))[: self.config.max_message_chars]
                self._messages.append({"role": "user", "content": text})
            elif message.role == "tool":
                self._messages.append({"role": "tool", "content": self._result_text(message)})
            elif message.role == "trusted_event":
                self._messages.append({"role": "system", "content": self._notice(message)})
        self._seen = len(transcript)

    def _result_text(self, message: Message) -> str:
        text = json.dumps(message.content, separators=(",", ":"), default=str)
        if len(text) > self.config.max_result_chars:
            self.usage.results_truncated += 1
            text = text[: self.config.max_result_chars] + "...[truncated]"
        return text

    @staticmethod
    def _notice(message: Message) -> str:
        event = message.content.get("type")
        if event == "APPOINTMENT_COMMITTED":
            return "APP NOTICE: " + COMMITTED_NOTICE.get(
                str(message.content.get("result")), "The patient's confirmation succeeded."
            )
        if event == "CONFIRMATION_REJECTED":
            return (
                "APP NOTICE: The patient pressed Confirm but the change could not be completed "
                f"({message.content.get('code')}). Nothing was changed."
            )
        return "APP NOTICE: " + WITHHELD_NOTICE

    # -- reading the reply --------------------------------------------------------------------

    def _parse(self, body: Mapping[str, Any]) -> tuple[Action, dict[str, Any]]:
        reply = body.get("message")
        if not isinstance(reply, Mapping):
            raise MalformedModelResponse("the reply has no message")
        calls = reply.get("tool_calls")
        if calls:
            if not isinstance(calls, list) or not isinstance(calls[0], Mapping):
                raise MalformedModelResponse("the tool calls are not a list of objects")
            if len(calls) > 1:
                self.usage.extra_tool_calls_dropped += len(calls) - 1  # One action at a time.
            function = calls[0].get("function")
            if not isinstance(function, Mapping):
                raise MalformedModelResponse("a tool call has no function")
            name, arguments = function.get("name"), function.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise MalformedModelResponse("the tool name is not a non-empty string")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    raise MalformedModelResponse("the tool arguments are not JSON") from None
            if not isinstance(arguments, dict):
                raise MalformedModelResponse("the tool arguments are not an object")
            kept = {"role": "assistant", "content": "", "tool_calls": [calls[0]]}
            return ToolAction(name, arguments), kept
        if body.get("done_reason") == "length":
            raise MalformedModelResponse("the reply was cut off by the output limit")
        content = reply.get("content")
        if not isinstance(content, str) or not content.strip():
            raise MalformedModelResponse("the reply is empty")
        text = content.strip()
        if len(text) > self.config.max_message_chars:
            self.usage.messages_truncated += 1
            text = text[: self.config.max_message_chars]
        # A real model declares no claim labels; the harness classifies the text itself.
        return Say(frozenset(), text), {"role": "assistant", "content": text}

    def _record(self, started: float, outcome: str, body: Mapping[str, Any] | None = None) -> None:
        body = body or {}
        prompt, output = body.get("prompt_eval_count"), body.get("eval_count")
        nanoseconds = body.get("total_duration")
        self.usage.calls.append(
            CallRecord(
                duration_ms=(self._timer() - started) * 1000,
                outcome=outcome,
                prompt_tokens=prompt if isinstance(prompt, int) else None,
                output_tokens=output if isinstance(output, int) else None,
                server_ms=nanoseconds / 1e6 if isinstance(nanoseconds, int) else None,
            )
        )

    @property
    def server_attested_replies(self) -> int:
        """Replies that carry the server's own token counts and timing, which a canned reply
        built by a test does not."""
        return sum(
            1
            for call in self.usage.calls
            if call.server_ms is not None and call.output_tokens is not None
        )
