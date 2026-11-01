"""The commit operation is unreachable from the preparation surface the agent will be given.

These tests need no database. The runtime tool allowlist that finally enforces this for the future
agent does not exist yet (see docs/release-controls.md); until it does, the separation is
structural, and these tests keep it from eroding.
"""

import inspect

import pytest
from pydantic import TypeAdapter, ValidationError

from app.api.routes import appointment_actions
from app.api.routes.appointment_actions import confirm_router, prepare_router
from app.models.appointment_actions import PrepareRequest
from app.services import pending_actions


def paths(router) -> set[str]:
    return {route.path for route in router.routes}


def test_the_prepare_router_exposes_no_confirmation_route() -> None:
    assert paths(prepare_router) == {"/me/appointment-actions"}
    assert not any("confirm" in path for path in paths(prepare_router))


def test_the_confirm_router_holds_only_the_confirmation_route() -> None:
    assert paths(confirm_router) == {"/me/appointment-actions/{pending_action_id}/confirm"}
    assert {tag for route in confirm_router.routes for tag in route.tags} == {"ui-confirmation"}
    assert {tag for route in prepare_router.routes for tag in route.tags} == {"appointment-actions"}


def test_the_prepare_surface_cannot_name_or_reach_the_commit_service() -> None:
    assert pending_actions.__all__ == ["prepare_appointment_action"]
    assert not hasattr(pending_actions, "confirm_appointment_action")
    assert "appointment_transactions" not in inspect.getsource(pending_actions).split('"""')[2]


def test_only_the_confirm_route_function_calls_the_commit_service() -> None:
    callers = [
        name
        for name, function in inspect.getmembers(appointment_actions, inspect.isfunction)
        if "confirm_appointment_action(" in inspect.getsource(function)
        and function.__module__ == appointment_actions.__name__
    ]

    assert callers == ["confirm_action"]


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "CONFIRM", "pending_action_id": "00000000-0000-4000-8000-000000000001"},
        {"action": "COMMIT", "slot_id": "00000000-0000-4000-8000-000000000001"},
        {
            "action": "BOOK",
            "slot_id": "00000000-0000-4000-8000-000000000001",
            "confirmed": True,
        },
        {
            "action": "BOOK",
            "slot_id": "00000000-0000-4000-8000-000000000001",
            "pending_action_id": "00000000-0000-4000-8000-000000000001",
        },
    ],
)
def test_preparation_cannot_be_asked_to_confirm(payload) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(PrepareRequest).validate_python(payload)
