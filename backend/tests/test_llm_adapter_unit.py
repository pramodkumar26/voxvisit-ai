"""The LLM adapter in isolation. Every reply here is canned by the test: these tests check how
the adapter builds a conversation and reads replies, and say nothing about any real model."""

import json

import pytest

from app.agent_tools import build_default_registry
from app.evaluation import llm
from app.evaluation.llm import (
    LlmConfig,
    ModelCallLimitError,
    ModelConfigurationError,
    ModelTimeoutError,
    OllamaModel,
)
from app.evaluation.model import MalformedModelResponse, Message, Say, ToolAction

DEFINITIONS = build_default_registry().definitions()


class CannedProvider:
    """Stands in for the model server: returns the next canned body and records each request."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.requests: list[dict] = []

    def __call__(self, path, payload, timeout_s):
        self.requests.append(
            {"path": path, "payload": json.loads(json.dumps(payload)), "timeout_s": timeout_s}
        )
        body = self.bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        return body


def tool_reply(name="search_clinic_directory", arguments=None, **extra):
    call = {"function": {"name": name, "arguments": {} if arguments is None else arguments}}
    return {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, **extra}


def text_reply(content, **extra):
    return {"message": {"role": "assistant", "content": content}, **extra}


def model_with(*bodies, config=None):
    provider = CannedProvider(*bodies)
    return OllamaModel(DEFINITIONS, config, transport=provider), provider


REQUEST = [Message("user", {"request": "Book the first appointment at Some Clinic on 2030-01-01."})]


# -- configuration and bounds ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://example.com:11434", "http://10.0.0.5:11434", "https://api.example.com"]
)
def test_a_model_server_that_is_not_on_this_machine_is_refused(url) -> None:
    with pytest.raises(ModelConfigurationError):
        LlmConfig(base_url=url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434"])
def test_loopback_servers_are_accepted(url) -> None:
    assert LlmConfig(base_url=url).base_url == url


@pytest.mark.parametrize(
    "overrides", [{"max_decisions": 0}, {"request_timeout_s": 0}, {"max_output_tokens": 0}]
)
def test_limits_must_be_positive(overrides) -> None:
    with pytest.raises(ModelConfigurationError):
        LlmConfig(**overrides)


def test_every_call_carries_the_timeout_and_output_bounds() -> None:
    config = LlmConfig(request_timeout_s=7.5, max_output_tokens=33, context_tokens=2048)
    model, provider = model_with(text_reply("Hello."), config=config)

    model.decide(REQUEST)

    sent = provider.requests[0]
    assert sent["path"] == "/api/chat" and sent["timeout_s"] == 7.5
    assert sent["payload"]["stream"] is False
    assert sent["payload"]["options"]["num_predict"] == 33
    assert sent["payload"]["options"]["num_ctx"] == 2048


def test_the_conversation_has_a_call_budget() -> None:
    model, provider = model_with(
        *[tool_reply() for _ in range(3)], config=LlmConfig(max_decisions=3)
    )
    for _ in range(3):
        model.decide(REQUEST)

    with pytest.raises(ModelCallLimitError):
        model.decide(REQUEST)
    assert len(provider.requests) == 3  # The fourth call was never made.


def test_a_timeout_propagates_as_a_timeout_and_is_still_counted() -> None:
    model, _ = model_with(ModelTimeoutError())

    with pytest.raises(ModelTimeoutError):
        model.decide(REQUEST)

    assert model.usage.calls[0].outcome == "ModelTimeoutError"
    assert model.served_replies == 0


# -- what the model is shown ----------------------------------------------------------------


def test_the_model_is_offered_exactly_the_four_approved_tools_and_nothing_to_confirm_with() -> None:
    model, provider = model_with(text_reply("Hello."))

    model.decide(REQUEST)

    tools = provider.requests[0]["payload"]["tools"]
    names = {tool["function"]["name"] for tool in tools}
    assert names == {
        "search_clinic_directory",
        "search_appointment_availability",
        "get_my_appointments",
        "prepare_appointment_action",
    }
    parameter_names = set()

    def collect(node):
        if isinstance(node, dict):
            parameter_names.update(node.get("properties", {}))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect([tool["function"]["parameters"] for tool in tools])
    assert parameter_names, "the parameters were not collected"
    for fragment in ("confirm", "idempotency", "session", "csrf", "patient", "token", "pin"):
        assert not any(fragment in name for name in parameter_names), fragment


def test_the_prompt_names_no_fixture_value_credential_or_answer() -> None:
    prompt = llm.SYSTEM_PROMPT.lower()

    for fragment in ("evaluation clinic", "dr. lee", "adams", "baker", "ambiguous_match"):
        assert fragment not in prompt
    assert "you cannot confirm anything yourself" in prompt


def test_the_prompt_gives_general_preparation_and_filter_rules() -> None:
    prompt = llm.SYSTEM_PROMPT

    assert "Do not ask for another verbal confirmation before preparation" in prompt
    assert "Seeing a value in a tool result does not make it a patient preference" in prompt
    assert "`BOOK`, `RESCHEDULE`, or `CANCEL`" in prompt
    assert "Never wrap the action in an object or list" in prompt


def test_only_the_request_tool_results_and_app_notices_reach_the_model() -> None:
    model, provider = model_with(
        tool_reply("search_clinic_directory", {"clinic_name": "Some Clinic"}),
        text_reply("Which doctor?"),
        text_reply("Done."),
    )
    transcript = list(REQUEST)
    first = model.decide(transcript)
    assert first == ToolAction("search_clinic_directory", {"clinic_name": "Some Clinic"})
    transcript.append(Message("tool", {"ok": True, "data": {"clinics": []}, "error": None}))
    model.decide(transcript)
    transcript.append(
        Message("trusted_event", {"type": "APPOINTMENT_COMMITTED", "result": "BOOKED"})
    )
    model.decide(transcript)

    roles = [m["role"] for m in provider.requests[2]["payload"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "system"]
    last = provider.requests[2]["payload"]["messages"][-1]["content"]
    assert last.startswith("APP NOTICE:") and "was booked" in last
    # The earlier assistant tool call is replayed so the model remembers what it asked for.
    assert provider.requests[2]["payload"]["messages"][2]["tool_calls"][0]["function"]["name"] == (
        "search_clinic_directory"
    )


def test_a_withheld_or_rejected_confirmation_is_described_as_changing_nothing() -> None:
    model, provider = model_with(text_reply("Ok."), text_reply("Ok."))
    transcript = [*REQUEST, Message("trusted_event", {"type": "CONFIRMATION_WITHHELD"})]
    model.decide(transcript)
    transcript.append(
        Message("trusted_event", {"type": "CONFIRMATION_REJECTED", "code": "SLOT_UNAVAILABLE"})
    )
    model.decide(transcript)

    notices = [
        m["content"] for m in provider.requests[1]["payload"]["messages"] if m["role"] == "system"
    ][1:]
    assert all("Nothing was changed" in notice for notice in notices) and len(notices) == 2
    assert "SLOT_UNAVAILABLE" in notices[1]


def test_an_oversized_tool_result_is_cut_before_it_reaches_the_model() -> None:
    model, provider = model_with(text_reply("Ok."), config=LlmConfig(max_result_chars=100))
    transcript = [*REQUEST, Message("tool", {"ok": True, "data": {"blob": "x" * 5000}})]

    model.decide(transcript)

    sent = provider.requests[0]["payload"]["messages"][-1]["content"]
    assert len(sent) < 130 and sent.endswith("[truncated]")
    assert model.usage.results_truncated == 1


# -- reading replies ------------------------------------------------------------------------


def test_a_tool_call_becomes_a_tool_action() -> None:
    model, _ = model_with(tool_reply("get_my_appointments", {"status": "CONFIRMED"}))

    assert model.decide(REQUEST) == ToolAction("get_my_appointments", {"status": "CONFIRMED"})


def test_text_becomes_a_message_with_no_declared_claims() -> None:
    model, _ = model_with(text_reply("  Which doctor do you mean?  "))

    action = model.decide(REQUEST)

    assert action == Say(frozenset(), "Which doctor do you mean?")


def test_arguments_sent_as_a_json_string_are_parsed() -> None:
    model, _ = model_with(tool_reply("get_my_appointments", '{"status": "CONFIRMED"}'))

    assert model.decide(REQUEST).arguments == {"status": "CONFIRMED"}


def test_only_the_first_of_several_tool_calls_is_taken() -> None:
    body = tool_reply("get_my_appointments")
    body["message"]["tool_calls"].append({"function": {"name": "prepare_appointment_action"}})
    model, _ = model_with(body)

    action = model.decide(REQUEST)

    assert action.tool == "get_my_appointments"
    assert model.usage.extra_tool_calls_dropped == 1


def test_a_tool_the_model_invents_is_passed_on_for_the_runtime_to_refuse() -> None:
    model, _ = model_with(tool_reply("confirm_appointment_action", {"confirmed": True}))

    assert model.decide(REQUEST) == ToolAction("confirm_appointment_action", {"confirmed": True})


def test_a_long_message_is_cut() -> None:
    model, _ = model_with(text_reply("a" * 5000), config=LlmConfig(max_message_chars=50))

    action = model.decide(REQUEST)

    assert len(action.text) == 50 and model.usage.messages_truncated == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": "text"},
        {"message": {"content": ""}},
        {"message": {"content": "   "}},
        {"message": {"content": None}},
        {"message": {"content": 42}},
        {"message": {"tool_calls": "get_my_appointments"}},
        {"message": {"tool_calls": [5]}},
        {"message": {"tool_calls": [{"function": "x"}]}},
        {"message": {"tool_calls": [{"function": {"arguments": {}}}]}},
        {"message": {"tool_calls": [{"function": {"name": "", "arguments": {}}}]}},
        {"message": {"tool_calls": [{"function": {"name": 7, "arguments": {}}}]}},
        {"message": {"tool_calls": [{"function": {"name": "x", "arguments": ["a"]}}]}},
        {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]}},
        {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "[1]"}}]}},
        {"message": {"content": "partial answ"}, "done_reason": "length"},
    ],
)
def test_an_unreadable_reply_is_a_malformed_response_with_a_fixed_description(body) -> None:
    model, _ = model_with(body)

    with pytest.raises(MalformedModelResponse) as caught:
        model.decide(REQUEST)

    assert caught.value.problem and "partial answ" not in caught.value.problem
    assert model.usage.calls[0].outcome == caught.value.problem


def test_usage_records_server_reported_counts_and_does_not_invent_them() -> None:
    attested, _ = model_with(
        text_reply("Ok.", prompt_eval_count=11, eval_count=3, total_duration=2_500_000_000)
    )
    canned, _ = model_with(text_reply("Ok."))
    attested.decide(REQUEST)
    canned.decide(REQUEST)

    assert attested.server_attested_replies == 1 and attested.usage.calls[0].server_ms == 2500.0
    assert canned.server_attested_replies == 0
    assert not canned.real_transport  # A canned transport is never a real-model run.
