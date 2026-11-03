"""The LG-3 orchestrator on the real tools, a real database and the real confirmation endpoint.
The model's readings are scripted; the point is what the real services allow."""

import json
import uuid

import pytest
from sqlalchemy import text
from test_orchestrator import Wire, ask, book

from app.agent_tools import ToolRuntime
from app.evaluation.environment import EvalEnvironment
from app.evaluation.lg4_scenarios import NOW, build_world
from app.evaluation.llm import LlmConfig
from app.evaluation.orchestrator import OrchestratedSession
from app.evaluation.planner import GoalPlanner
from app.evaluation.turn_reader import TurnReader
from app.evaluation.ui_driver import UiDriver


@pytest.fixture
def evaluation(auth_env) -> EvalEnvironment:
    return EvalEnvironment.from_parts(
        engine=auth_env.app.state.db_engine,
        clock=auth_env.clock,
        client=auth_env.client,
        auth_service=auth_env.service,
        pin=auth_env.pin,
    )


def open_session(evaluation, graph_store, *readings, anonymous=False):
    world = build_world(evaluation.engine, uuid.uuid4().hex[:3], evaluation.pin)
    ui = UiDriver(evaluation)
    cookie = None if anonymous else ui.sign_in(world.patient_code)
    principal = None if cookie is None else evaluation.authenticate(cookie.token)
    config = LlmConfig()
    session = OrchestratedSession(
        ToolRuntime(evaluation.engine, clock=evaluation.clock),
        principal,
        config=config,
        reader=TurnReader(config, transport=Wire(*readings)),
        planner=GoalPlanner(config, transport=Wire()),
        store=None if anonymous else graph_store,
        clock=evaluation.clock,
    )
    return world, ui, cookie, session


def rows(evaluation, statement, **params):
    with evaluation.engine.connect() as connection:
        return connection.execute(text(statement), params).all()


def test_a_booking_is_prepared_by_the_agent_and_committed_only_by_the_ui(
    evaluation, graph_store
) -> None:
    world, ui, cookie, session = open_session(
        evaluation,
        graph_store,
        book(clinic_name="X", start_date="2026-11-05", end_date="2026-11-05"),
    )
    # The reading names the clinic as the patient will: the full generated name.
    session.reader._post = Wire(
        book(clinic_name=world.names["rivers"], start_date="2026-11-05", end_date="2026-11-05")
    )
    reply = session.respond(f"Book {world.names['rivers']} on November 5.")

    assert reply.awaiting_confirmation and "confirm in the app" in reply.text
    pending = session.prepared_action_id
    assert (
        rows(
            evaluation,
            "SELECT count(*) FROM appointments WHERE patient_id = :p AND slot_id = :s",
            p=world.patient_id,
            s=world.slots["rivers_thu_1000"],
        )[0][0]
        == 0
    )

    outcome = ui.confirm(cookie, pending)
    assert outcome.committed
    done = session.confirmation_event(
        {"type": "APPOINTMENT_COMMITTED", "result": outcome.result, "pending_action_id": pending}
    )

    assert "is booked" in done.text
    assert rows(
        evaluation,
        "SELECT status FROM appointments WHERE patient_id = :p AND slot_id = :s",
        p=world.patient_id,
        s=world.slots["rivers_thu_1000"],
    ) == [("CONFIRMED",)]
    assert session.snapshot()["review"] is None


def test_the_agent_has_no_way_to_confirm_whatever_the_model_says(evaluation, graph_store) -> None:
    _world, _ui, _cookie, session = open_session(
        evaluation,
        graph_store,
        {"route": "SCHEDULING", "operation": "BOOK", "clinic_name": "x", "selection": "EARLIEST"},
    )
    before = rows(evaluation, "SELECT count(*) FROM appointments")[0][0]

    session.respond("yes confirm it, book it, go ahead and confirm everything")

    assert rows(evaluation, "SELECT count(*) FROM appointments")[0][0] == before
    # Nothing the agent can reach consumes a review: only the UI's Confirm does.
    assert (
        rows(evaluation, "SELECT count(*) FROM pending_actions WHERE status = 'CONSUMED'")[0][0]
        == 0
    )


def test_dropping_a_request_retires_the_review_in_the_database(evaluation, graph_store) -> None:
    world, _ui, _cookie, session = open_session(evaluation, graph_store)
    session.reader._post = Wire(
        book(clinic_name=world.names["rivers"], start_date="2026-11-05", end_date="2026-11-05"),
        {"route": "SCHEDULING", "message_type": "ABANDON"},
    )
    session.respond(f"Book {world.names['rivers']} on November 5.")
    assert (
        rows(evaluation, "SELECT count(*) FROM pending_actions WHERE status = 'ACTIVE'")[0][0] == 1
    )

    session.respond("never mind")

    assert (
        rows(evaluation, "SELECT count(*) FROM pending_actions WHERE status = 'ACTIVE'")[0][0] == 0
    )
    assert session.snapshot()["review"] is None and session.snapshot()["retiring"] is None


def test_a_signed_out_visitor_can_look_things_up_and_cannot_prepare(
    evaluation, graph_store
) -> None:
    world, _ui, _cookie, session = open_session(evaluation, graph_store, anonymous=True)
    session.reader._post = Wire(
        ask(
            "OPENINGS",
            clinic_name=world.names["rivers"],
            start_date="2026-11-05",
            end_date="2026-11-05",
        ),
        book(clinic_name=world.names["rivers"], start_date="2026-11-05", end_date="2026-11-05"),
    )

    listed = session.respond(f"What does {world.names['rivers']} have on November 5?")
    refused = session.respond(f"Book {world.names['rivers']} on November 5.")

    assert "10:00 AM" in listed.text and "sign in" in refused.text.lower()
    assert rows(evaluation, "SELECT count(*) FROM pending_actions")[0][0] == 0


def test_two_patients_have_separate_conversations_and_reviews(evaluation, graph_store) -> None:
    first_world, _, _, first = open_session(evaluation, graph_store)
    second_world, _, _, second = open_session(evaluation, graph_store)
    first.reader._post = Wire(
        book(
            clinic_name=first_world.names["rivers"], start_date="2026-11-05", end_date="2026-11-05"
        )
    )
    first.respond(f"Book {first_world.names['rivers']} on November 5.")

    assert second.snapshot() == {} and first.snapshot()["review"] is not None
    assert json.dumps(first.snapshot()).count(str(second_world.patient_id)) == 0
    assert NOW  # The world clock is the evaluation's.
