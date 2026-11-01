"""The agent tool layer without a database: envelope, allowlist, argument rules, authorization
order, retries, and tracing."""

import ast
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from app.agent_tools import ALLOWED_TOOLS, ToolRuntime, build_default_registry, envelope
from app.agent_tools.registry import Access, ToolRegistry, ToolRegistryError, ToolSpec
from app.agent_tools.schemas import AvailabilityArgs, DirectoryArgs, MyAppointmentsArgs
from app.agent_tools.tools import ToolFailure
from app.agent_tools.tracing import InMemoryTraceSink, LoggingTraceSink
from app.auth.service import Principal
from app.core.errors import (
    DependencyUnavailableError,
    ResourceNotAvailableError,
    SessionExpiredError,
)
from app.core.logging import JsonFormatter
from app.models.appointment_actions import PrepareRequest

APP_DIR = Path(__file__).parents[1] / "app"
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
FOUR = {
    "search_clinic_directory",
    "search_appointment_availability",
    "get_my_appointments",
    "prepare_appointment_action",
}
SLOT = "00000000-0000-4000-8000-000000000001"


def principal(actor_type: str = "PATIENT") -> Principal:
    patient = uuid.uuid4() if actor_type == "PATIENT" else None
    return Principal(
        session_id=uuid.uuid4(),
        actor_type=actor_type,
        login="x",
        display_name="Test",
        patient_id=patient,
        staff_id=None if patient else uuid.uuid4(),
        absolute_expires_at=NOW + timedelta(minutes=30),
        idle_expires_at=NOW + timedelta(minutes=15),
        csrf_token="not-a-real-token",
    )


def runtime(
    registry: ToolRegistry | None = None, **options
) -> tuple[ToolRuntime, InMemoryTraceSink]:
    sink = InMemoryTraceSink()
    return (
        ToolRuntime(
            None,  # type: ignore[arg-type] - these tests must never reach the database
            registry=registry,
            clock=lambda: NOW,
            trace_sink=sink,
            sleep=options.pop("sleep", lambda seconds: None),
            jitter=lambda low, high: low,
            **options,
        ),
        sink,
    )


def fake_registry(handler, *, access=Access.PUBLIC_OR_PATIENT) -> ToolRegistry:
    handler.__module__ = "app.agent_tools.tools"  # Looks like a package handler to the guard.
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_clinic_directory",
            description="test",
            access=access,
            arguments=DirectoryArgs,
            handler=handler,
        )
    )
    return registry


# --- the envelope -----------------------------------------------------------------------------


def test_every_response_has_exactly_the_four_envelope_fields() -> None:
    request_id = uuid.uuid4()
    ok = envelope.success(request_id, {"slots": []}).to_dict()
    bad = envelope.failure(
        request_id, envelope.simple_error("RESOURCE_NOT_AVAILABLE", "That item is unavailable.")
    ).to_dict()

    assert set(ok) == set(bad) == {"ok", "request_id", "data", "error"}
    assert ok == {"ok": True, "request_id": str(request_id), "data": {"slots": []}, "error": None}
    assert bad["ok"] is False and bad["data"] is None
    assert bad["error"] == {
        "code": "RESOURCE_NOT_AVAILABLE",
        "message": "That item is unavailable.",
        "retryable": False,
    }
    json.dumps(ok)
    json.dumps(bad)


def test_only_transient_codes_are_retryable() -> None:
    codes = {
        "DEPENDENCY_UNAVAILABLE": True,
        "RATE_LIMITED": True,
        "INVALID_ARGUMENT": False,
        "AUTH_REQUIRED": False,
        "SLOT_UNAVAILABLE": False,
        "CONFLICT": False,
        "INTERNAL_ERROR": False,
    }
    for code, retryable in codes.items():
        assert envelope.simple_error(code, "m").retryable is retryable, code


def test_backend_errors_keep_their_code_message_and_reason_only() -> None:
    error = envelope.error_from_application(SessionExpiredError())
    dependency = envelope.error_from_application(DependencyUnavailableError())

    assert (error.code, error.retryable) == ("SESSION_EXPIRED", False)
    assert (dependency.code, dependency.retryable) == ("DEPENDENCY_UNAVAILABLE", True)
    assert set(error.model_dump(exclude_none=True)) == {"code", "message", "retryable"}


def test_validation_errors_name_fields_but_never_echo_values() -> None:
    raw = {"clinic_id": "SECRET-VALUE-123", "patient_id": "p-1"}
    with pytest.raises(ValidationError) as exc_info:
        DirectoryArgs.model_validate(raw)

    error = envelope.error_from_validation(exc_info.value, raw)

    assert error.code == "INVALID_ARGUMENT" and error.fields == ["clinic_id", "patient_id"]
    assert {item.value_type for item in error.validation} == {"string"}
    assert "SECRET-VALUE-123" not in error.model_dump_json()


@pytest.mark.parametrize(
    ("action", "value_type"),
    [({"kind": "BOOK"}, "object"), (["BOOK"], "array"), (None, "null"), (7, "integer")],
)
def test_invalid_action_diagnostics_reveal_shape_but_never_value(action, value_type) -> None:
    raw = {"action": action, "slot_id": SLOT}
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(PrepareRequest).validate_python(raw)

    error = envelope.error_from_validation(exc_info.value, raw)

    assert error.fields == ["action"]
    assert [item.model_dump() for item in error.validation] == [
        {"field": "action", "value_type": value_type, "category": "union_tag_invalid"}
    ]
    body = error.model_dump_json()
    assert "kind" not in body and "BOOK" not in body and str(action) not in body


def test_ordinary_validation_diagnostics_name_the_field_type_and_category() -> None:
    raw = {"action": "BOOK", "slot_id": "SECRET-BAD-UUID"}
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(PrepareRequest).validate_python(raw)

    error = envelope.error_from_validation(exc_info.value, raw)

    assert [item.model_dump() for item in error.validation] == [
        {"field": "slot_id", "value_type": "string", "category": "uuid_parsing"}
    ]
    assert "SECRET-BAD-UUID" not in error.model_dump_json()


def test_runtime_rejects_a_non_string_action_with_sanitized_diagnostics() -> None:
    tool_runtime, _ = runtime()

    response = tool_runtime.invoke(
        "prepare_appointment_action",
        {"action": {"kind": "BOOK"}, "slot_id": SLOT},
        principal=principal(),
    ).to_dict()

    assert response["error"]["validation"] == [
        {"field": "action", "value_type": "object", "category": "union_tag_invalid"}
    ]
    assert "kind" not in json.dumps(response) and "BOOK" not in json.dumps(response)


def test_hostile_field_names_are_neutralised_and_bounded() -> None:
    secret_field = "MODEL_SUPPLIED_SECRET_VALUE"
    hostile = {f"x{i}\n<script>": 1 for i in range(30)} | {"a" * 500: 1, secret_field: 1}
    with pytest.raises(ValidationError) as exc_info:
        DirectoryArgs.model_validate(hostile)

    error = envelope.error_from_validation(exc_info.value)

    assert len(error.fields) <= envelope.MAX_REPORTED_FIELDS
    assert len(error.validation) <= envelope.MAX_REPORTED_FIELDS
    assert all(len(name) <= envelope.MAX_FIELD_NAME_LENGTH for name in error.fields)
    assert all("\n" not in name and "<" not in name for name in error.fields)
    assert error.fields == ["unknown_field"]
    assert secret_field not in error.model_dump_json()


def test_dynamic_nested_paths_cannot_echo_model_supplied_field_names() -> None:
    class Nested(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Container(BaseModel):
        action: Nested

    secret_field = "NESTED_MODEL_VALUE_ABC123"
    raw = {"action": {secret_field: "hidden"}}
    with pytest.raises(ValidationError) as exc_info:
        Container.model_validate(raw)

    error = envelope.error_from_validation(exc_info.value, raw)
    serialized = error.model_dump_json()

    assert error.fields == ["action.unknown_field"]
    assert error.validation[0].value_type == "string"
    assert secret_field not in serialized and "hidden" not in serialized


# --- the allowlist ----------------------------------------------------------------------------


def test_the_allowlist_is_exactly_the_four_implemented_tools() -> None:
    assert ALLOWED_TOOLS == FOUR
    assert build_default_registry().names() == FOUR


@pytest.mark.parametrize(
    "name",
    [
        "confirm_appointment_action",
        "confirm",
        "commit_appointment",
        "execute_action",
        "prepare_waitlist_action",
        "search_clinic_policies",
        "request_staff_handoff",
        "get_my_waitlist",
        "",
        "SEARCH_CLINIC_DIRECTORY",
    ],
)
def test_nothing_else_can_be_registered(name: str) -> None:
    def handler(args, context):
        return {}

    handler.__module__ = "app.agent_tools.tools"
    with pytest.raises(ToolRegistryError):
        ToolRegistry().register(
            ToolSpec(
                name=name,
                description="x",
                access=Access.PATIENT,
                arguments=DirectoryArgs,
                handler=handler,
            )
        )


def test_a_confirmation_name_is_refused_even_if_added_to_the_allowlist(monkeypatch) -> None:
    from app.agent_tools import registry as registry_module

    monkeypatch.setattr(
        registry_module, "ALLOWED_TOOLS", frozenset({*FOUR, "confirm_appointment_action"})
    )

    def handler(args, context):
        return {}

    handler.__module__ = "app.agent_tools.tools"
    with pytest.raises(ToolRegistryError, match="commit or confirmation"):
        ToolRegistry().register(
            ToolSpec(
                name="confirm_appointment_action",
                description="x",
                access=Access.PATIENT,
                arguments=DirectoryArgs,
                handler=handler,
            )
        )


def test_handlers_from_outside_the_tool_package_are_refused() -> None:
    from app.services.appointment_transactions import confirm_appointment_action
    from app.services.pending_actions import prepare_appointment_action

    for outsider in (confirm_appointment_action, prepare_appointment_action, print, json.dumps):
        with pytest.raises(ToolRegistryError, match="outside the tool package"):
            ToolRegistry().register(
                ToolSpec(
                    name="get_my_appointments",
                    description="x",
                    access=Access.PATIENT,
                    arguments=MyAppointmentsArgs,
                    handler=outsider,
                )
            )


def test_a_tool_cannot_be_registered_twice() -> None:
    registry = build_default_registry()
    spec = registry.get("get_my_appointments")

    with pytest.raises(ToolRegistryError, match="already registered"):
        registry.register(spec)


def test_the_package_never_imports_the_commit_or_confirmation_code() -> None:
    forbidden_modules = {
        "app.services.appointment_transactions",
        "app.api.routes.appointment_actions",
    }
    forbidden_names = {
        "appointment_transactions",
        "confirm_appointment_action",
        "confirm_router",
        "confirm_action",
    }
    offenders = []
    for path in (APP_DIR / "agent_tools").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                if node.module in forbidden_modules or names & forbidden_names:
                    offenders.append(path.name)
                if node.module in {"app.api.routes"} and "appointment_actions" in names:
                    offenders.append(path.name)
            elif isinstance(node, ast.Import):
                if any(alias.name in forbidden_modules for alias in node.names):
                    offenders.append(path.name)

    assert offenders == []


def _parameter_names(schema) -> set[str]:
    """Every parameter name in a JSON schema, not enum values or descriptions."""
    names: set[str] = set()
    if isinstance(schema, dict):
        names.update(schema.get("properties", {}))
        for value in schema.values():
            names |= _parameter_names(value)
    elif isinstance(schema, list):
        for item in schema:
            names |= _parameter_names(item)
    return names


def test_tool_definitions_expose_no_identity_session_or_idempotency_parameters() -> None:
    definitions = build_default_registry().definitions()

    assert {d["name"] for d in definitions} == FOUR
    names = set().union(*(_parameter_names(d["parameters"]) for d in definitions))
    assert names, "the schemas should declare parameters"
    for forbidden in (
        "patient_id",
        "session_id",
        "session_token",
        "token",
        "csrf_token",
        "idempotency_key",
        "role",
        "actor_type",
        "pending_action_id",
        "confirmed",
        "confirm",
        "limit",
        "request_id",
    ):
        assert forbidden not in names, forbidden
    prepare = next(d for d in definitions if d["name"] == "prepare_appointment_action")
    assert "cannot confirm" in prepare["description"]


# --- model-visible arguments ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (DirectoryArgs, {"browse": True, "patient_id": SLOT}),
        (DirectoryArgs, {"browse": True, "limit": 500}),
        (DirectoryArgs, {"browse": True, "idempotency_key": "x" * 20}),
        (
            AvailabilityArgs,
            {
                "clinic_id": SLOT,
                "start_date": "2026-11-02",
                "end_date": "2026-11-03",
                "role": "STAFF",
            },
        ),
        (
            AvailabilityArgs,
            {"clinic_id": "nope", "start_date": "2026-11-02", "end_date": "2026-11-03"},
        ),
        (AvailabilityArgs, {"clinic_id": SLOT, "start_date": "tomorrow", "end_date": "2026-11-03"}),
        (
            AvailabilityArgs,
            {
                "clinic_id": SLOT,
                "start_date": "2026-11-02",
                "end_date": "2026-11-03",
                "time_preference": "NIGHT",
            },
        ),
        (MyAppointmentsArgs, {"patient_id": SLOT}),
        (MyAppointmentsArgs, {"status": "PENDING"}),
        (MyAppointmentsArgs, {"session_token": "abc"}),
        (DirectoryArgs, {"clinic_name": "x" * 81}),
        (DirectoryArgs, {"cursor": "c" * 513}),
    ],
)
def test_only_documented_and_well_formed_arguments_are_accepted(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


# --- authorization happens before validation, and needs no database ---------------------------


def test_patient_tools_require_a_verified_patient() -> None:
    rt, sink = runtime()
    anonymous = rt.invoke("get_my_appointments", {}, principal=None).to_dict()
    staff_read = rt.invoke("get_my_appointments", {}, principal=principal("STAFF")).to_dict()
    staff_prepare = rt.invoke(
        "prepare_appointment_action",
        {"action": "BOOK", "slot_id": SLOT},
        principal=principal("STAFF"),
    ).to_dict()

    assert anonymous["error"]["code"] == "AUTH_REQUIRED"
    assert staff_read["error"]["code"] == staff_prepare["error"]["code"] == "FORBIDDEN"
    assert [t.error_category for t in sink.traces] == ["AUTH_REQUIRED", "FORBIDDEN", "FORBIDDEN"]


@pytest.mark.parametrize("tool", sorted(FOUR))
def test_staff_sessions_cannot_use_any_agent_tool(tool: str) -> None:
    rt, _ = runtime()

    response = rt.invoke(tool, {}, principal=principal("STAFF"))

    assert response.error.code == "FORBIDDEN" and response.error.retryable is False


def test_unauthorized_callers_learn_nothing_about_argument_schemas() -> None:
    rt, _ = runtime()

    bad_args = {"nonsense": 1, "patient_id": "x"}
    anonymous = rt.invoke("prepare_appointment_action", bad_args, principal=None).to_dict()
    staff = rt.invoke("get_my_appointments", bad_args, principal=principal("STAFF")).to_dict()

    assert anonymous["error"] == {
        "code": "AUTH_REQUIRED",
        "message": "Authentication is required.",
        "retryable": False,
    }
    assert "fields" not in staff["error"] and staff["error"]["code"] == "FORBIDDEN"


@pytest.mark.parametrize(
    "tool",
    [
        "confirm_appointment_action",
        "confirm",
        "commit_appointment",
        "prepare_waitlist_action",
        None,
        7,
        ["x"],
        "",
    ],
)
def test_unknown_or_forbidden_tools_are_all_reported_identically(tool) -> None:
    rt, sink = runtime()

    response = rt.invoke(
        tool, {"pending_action_id": SLOT, "confirmed": True}, principal=principal()
    )

    assert response.error.code == "INVALID_ARGUMENT" and response.error.reason == "UNKNOWN_TOOL"
    assert response.error.message == "That tool is not available."
    assert sink.traces[-1].tool == "unknown"  # An arbitrary name from the model is never logged.


def test_arguments_must_be_an_object() -> None:
    rt, _ = runtime()

    for arguments in ([], "x", 5):
        response = rt.invoke("get_my_appointments", arguments, principal=principal())
        assert response.error.code == "INVALID_ARGUMENT"


# --- every outcome is an envelope; failures are sanitized -------------------------------------


def test_unexpected_exceptions_become_a_generic_internal_error() -> None:
    def handler(args, context):
        raise RuntimeError("password=hunter2 SELECT * FROM patient_credentials")

    rt, sink = runtime(fake_registry(handler))

    response = rt.invoke("search_clinic_directory", {"browse": True}, principal=None)
    text = json.dumps(response.to_dict()) + repr(sink.traces)

    assert response.error.code == "INTERNAL_ERROR" and response.error.retryable is False
    for leaked in ("hunter2", "SELECT", "patient_credentials", "RuntimeError"):
        assert leaked not in text


def test_contract_outcomes_raised_by_a_handler_pass_through_unchanged() -> None:
    def handler(args, context):
        raise ToolFailure(envelope.simple_error("NO_RESULTS", "Nothing matched."))

    rt, _ = runtime(fake_registry(handler))

    assert rt.invoke("search_clinic_directory", {"browse": True}, principal=None).error.code == (
        "NO_RESULTS"
    )


# --- retries and idempotency keys -------------------------------------------------------------


def test_a_transient_failure_is_retried_once_with_the_same_key() -> None:
    seen = []

    def handler(args, context):
        seen.append((context.request_id, context.idempotency_key))
        if len(seen) == 1:
            raise DependencyUnavailableError()
        return {"ok": True}

    sleeps = []
    rt, sink = runtime(fake_registry(handler), sleep=sleeps.append)

    response = rt.invoke("search_clinic_directory", {"browse": True}, principal=None)

    assert response.ok and len(seen) == 2 and seen[0] == seen[1]
    assert sleeps == [0.05]  # Bounded jitter.
    assert sink.traces[-1].attempts == 2


def test_retries_are_bounded() -> None:
    calls = []

    def handler(args, context):
        calls.append(1)
        raise DependencyUnavailableError()

    rt, sink = runtime(fake_registry(handler))

    response = rt.invoke("search_clinic_directory", {"browse": True}, principal=None)

    assert len(calls) == 2
    assert response.error.code == "DEPENDENCY_UNAVAILABLE" and response.error.retryable is True
    assert sink.traces[-1].attempts == 2


def test_other_errors_are_never_retried() -> None:
    calls = []

    def handler(args, context):
        calls.append(1)
        raise ResourceNotAvailableError()

    rt, _ = runtime(fake_registry(handler))
    rt.invoke("search_clinic_directory", {"browse": True}, principal=None)

    assert len(calls) == 1


def test_each_logical_call_gets_its_own_unguessable_key() -> None:
    keys = []

    def handler(args, context):
        keys.append(context.idempotency_key)
        return {}

    rt, _ = runtime(fake_registry(handler))
    for _ in range(3):
        rt.invoke("search_clinic_directory", {"browse": True}, principal=None)

    assert len(set(keys)) == 3
    for key in keys:  # Valid for the idempotency header rules and carrying no argument content.
        assert 16 <= len(key) <= 128 and key.startswith("tool.")
        assert all(c.isalnum() or c in "._:-" for c in key)


def test_a_retried_call_reuses_its_key_and_cannot_change_its_arguments() -> None:
    keys = []

    def handler(args, context):
        keys.append((context.request_id, context.idempotency_key))
        return {}

    rt, _ = runtime(fake_registry(handler))
    call = rt.new_call()

    first = rt.invoke("search_clinic_directory", {"browse": True}, principal=None, call=call)
    again = rt.invoke("search_clinic_directory", {"browse": True}, principal=None, call=call)
    changed = rt.invoke("search_clinic_directory", {"clinic_name": "X"}, principal=None, call=call)

    assert first.ok and again.ok and keys[0] == keys[1]
    assert changed.error.code == "INVALID_ARGUMENT" and changed.error.reason == "CALL_REUSED"
    assert len(keys) == 2  # The changed call never reached the handler.


# --- tracing ----------------------------------------------------------------------------------


def test_traces_record_only_name_id_duration_outcome_category_and_attempts() -> None:
    ticks = iter([10.0, 10.25])
    rt, sink = runtime(fake_registry(lambda args, context: {}), timer=lambda: next(ticks))
    rt.invoke("search_clinic_directory", {"browse": True}, principal=None)

    [trace] = sink.traces

    assert trace.tool == "search_clinic_directory" and trace.outcome == "ok"
    assert trace.error_category is None and trace.attempts == 1
    assert trace.duration_ms == pytest.approx(250.0)
    assert set(vars(trace)) == {
        "tool",
        "request_id",
        "duration_ms",
        "outcome",
        "error_category",
        "attempts",
    }


def test_a_failing_trace_sink_never_changes_the_outcome() -> None:
    class Broken:
        def record(self, trace):
            raise OSError("disk full")

    rt = ToolRuntime(
        None,  # type: ignore[arg-type]
        registry=fake_registry(lambda args, context: {"fine": True}),
        clock=lambda: NOW,
        trace_sink=Broken(),
    )

    assert rt.invoke("search_clinic_directory", {"browse": True}, principal=None).ok


def test_logged_traces_carry_structured_fields_and_no_content(caplog) -> None:
    secret_name = "Zelda-Sentinel-Name"
    rt = ToolRuntime(
        None,  # type: ignore[arg-type]
        registry=fake_registry(lambda args, context: {"clinics": [secret_name]}),
        clock=lambda: NOW,
        trace_sink=LoggingTraceSink(),
    )
    with caplog.at_level(logging.INFO, logger="app.agent_tools"):
        rt.invoke(
            "search_clinic_directory",
            {"clinic_name": secret_name, "browse": False},
            principal=principal(),
        )

    [record] = [r for r in caplog.records if r.name == "app.agent_tools"]
    line = json.loads(JsonFormatter().format(record))

    assert line["tool"] == "search_clinic_directory" and line["outcome"] == "ok"
    assert set(line) == {
        "timestamp",
        "level",
        "logger",
        "message",
        "request_id",
        "tool",
        "duration_ms",
        "outcome",
        "error_category",
        "attempts",
    }
    assert secret_name not in json.dumps(line)


def test_one_request_id_ties_the_envelope_the_handler_and_the_trace_together() -> None:
    seen = []

    def handler(args, context):
        seen.append(context.request_id)
        return {}

    rt, sink = runtime(fake_registry(handler))

    ok = rt.invoke("search_clinic_directory", {"browse": True}, principal=None)
    rejected = rt.invoke("nonexistent_tool", {}, principal=None)

    assert ok.request_id == seen[0] == sink.traces[0].request_id
    assert rejected.request_id == sink.traces[1].request_id
    assert ok.request_id != rejected.request_id
