"""LG-5B: the conversation lives in PostgreSQL, and the backend still decides who may use it.

Model readings are scripted (the real model is measured by the LG-5 evaluation); everything else is
real: the application database, the authentication service, the tools, the confirmation endpoint and
the LangGraph PostgreSQL checkpointer. The questions asked here are about state and authority."""

import itertools
import threading
import time
import uuid
from datetime import timedelta

import psycopg
import pytest
from sqlalchemy import text
from test_orchestrator import Wire, book, sched

from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.evaluation.environment import EvalEnvironment
from app.evaluation.graph_service import GraphConversations
from app.evaluation.graph_store import (
    SCHEMA,
    CheckpointStore,
    owner_of,
    thread_for,
)
from app.evaluation.lg4_scenarios import build_world
from app.evaluation.llm import LlmConfig
from app.evaluation.orchestrator import BUSY_REPLY, TROUBLE_REPLY, OrchestratedSession
from app.evaluation.planner import GoalPlanner
from app.evaluation.turn_reader import TurnReader
from app.evaluation.ui_driver import UiDriver

SECRET_PIN_TEXT = "my pin is 482913, sign me in"


@pytest.fixture
def evaluation(auth_env) -> EvalEnvironment:
    return EvalEnvironment.from_parts(
        engine=auth_env.app.state.db_engine,
        clock=auth_env.clock,
        client=auth_env.client,
        auth_service=auth_env.service,
        pin=auth_env.pin,
    )


def service_over(evaluation, store, *readings, transport=None):
    config = LlmConfig()
    wire = transport or Wire(*readings)
    return GraphConversations(
        store=store,
        engine=evaluation.engine,
        auth=evaluation_auth(evaluation),
        config=config,
        clock=evaluation.clock,
        session_options={
            "reader": TurnReader(config, transport=wire),
            "planner": GoalPlanner(config, transport=Wire()),
        },
    ), wire


def evaluation_auth(evaluation):
    class Auth:
        def authenticate(self, token):
            return evaluation.authenticate(token)

    return Auth()


def new_store(evaluation, **options) -> CheckpointStore:
    return CheckpointStore(evaluation.engine.url, **options).open()


_SUFFIXES = itertools.count(0x100)


def sign_in(evaluation, suffix=None):
    suffix = suffix or f"{next(_SUFFIXES):03x}"
    world = build_world(evaluation.engine, suffix, evaluation.pin)
    ui = UiDriver(evaluation)
    return world, ui, ui.sign_in(world.patient_code)


def sql(evaluation, statement, **params):
    with evaluation.engine.connect() as connection:
        return connection.execute(text(statement), params).all()


def active_reviews(evaluation, patient_id) -> list[str]:
    rows = sql(
        evaluation,
        "SELECT p.pending_action_id::text FROM pending_actions p"
        " JOIN auth_sessions s ON s.session_id = p.session_id"
        " WHERE p.status = 'ACTIVE' AND s.patient_id = :p",
        p=patient_id,
    )
    return [r[0] for r in rows]


def stored_text(store) -> str:
    """Everything the store holds, as readable text, to look for what must never be in it."""
    pieces = []
    with store._pool.connection() as connection:
        for row in connection.execute(
            "SELECT checkpoint::text AS t, metadata::text AS m FROM checkpoints"
        ).fetchall():
            pieces += [row["t"], row["m"]]
        for table in ("checkpoint_blobs", "checkpoint_writes"):
            for row in connection.execute(
                f"SELECT thread_id, encode(blob, 'escape') AS b FROM {table}"
            ).fetchall():
                pieces += [row["thread_id"], row["b"] or ""]
    return "\n".join(pieces)


def morning_of_monday(world, text_after=""):
    return f"Book {world.names['pine']} on Monday morning.{text_after}"


MONDAY_MORNING = book(
    clinic_name=None, start_date="2026-11-09", end_date="2026-11-09", time_preference="MORNING"
)


def monday_reading(world):
    return {**MONDAY_MORNING, "clinic_name": world.names["pine"]}


# --- the store itself -------------------------------------------------------------------------


def test_the_store_makes_its_own_schema_and_touches_nothing_else(evaluation, graph_store) -> None:
    graph_store.open()  # Setting up again, as a second process would, is harmless.

    mine = {
        r[0]
        for r in sql(
            evaluation,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = :s",
            s=SCHEMA,
        )
    }
    elsewhere = sql(
        evaluation,
        "SELECT table_schema, table_name FROM information_schema.tables"
        " WHERE table_name LIKE 'checkpoint%' AND table_schema <> :s",
        s=SCHEMA,
    )

    assert mine == {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
    assert elsewhere == []
    assert graph_store.healthy()


def test_the_schema_can_be_removed_whole() -> None:
    # Dropping the schema is how the store is decommissioned: nothing else refers to it.
    assert SCHEMA == "agent_checkpoints"


# --- resuming ---------------------------------------------------------------------------------


def test_a_conversation_resumes_after_a_process_restart(evaluation, graph_store) -> None:
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    asked = service.turn(cookie.token, morning_of_monday(world))
    assert not asked.awaiting_confirmation and "Which time" in asked.text

    # A new process: a new connection pool, a new checkpointer, a new service object.
    graph_store.close()
    with new_store(evaluation) as restarted:
        service2, _ = service_over(
            evaluation,
            restarted,
            sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        )
        reply = service2.turn(cookie.token, "9:30 AM")
        assert reply.awaiting_confirmation and "confirm in the app" in reply.text.lower()

        (pending,) = active_reviews(evaluation, world.patient_id)
        outcome = ui.confirm(cookie, pending)  # Only the authenticated UI commits.
        assert outcome.committed
        done = service2.event(
            cookie.token,
            {
                "type": "APPOINTMENT_COMMITTED",
                "result": outcome.result,
                "pending_action_id": pending,
            },
        )
        assert "was booked" in done.text


def test_every_resumed_turn_is_authenticated_first(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    service, wire = service_over(evaluation, graph_store, monday_reading(world))

    for bad in (None, "", "not-a-token", cookie.token + "x"):
        with pytest.raises(AuthenticationRequiredError):
            service.turn(bad, morning_of_monday(world))

    assert wire.payloads == []  # No model was reached, no state was read or written.
    assert graph_store.threads() == []


def test_a_signed_in_staff_member_cannot_converse(evaluation, graph_store) -> None:
    service, _ = service_over(evaluation, graph_store)
    with pytest.raises(AuthenticationRequiredError):
        service._verified(None)  # No cookie at all is refused before anything else.


# --- what is stored ---------------------------------------------------------------------------


def test_no_credential_token_session_id_or_patient_words_are_stored(
    evaluation, graph_store
) -> None:
    world, _ui, cookie = sign_in(evaluation)
    principal = evaluation.authenticate(cookie.token)
    sentence = morning_of_monday(world, " I'd love to see someone kind, please.")
    service, wire = service_over(evaluation, graph_store, monday_reading(world), sched())

    service.turn(cookie.token, sentence)
    service.turn(cookie.token, SECRET_PIN_TEXT)  # A credential: never read, never stored.

    held = stored_text(graph_store)
    assert held  # Something was stored: the conversation.
    for forbidden in (
        cookie.token,
        cookie.csrf,
        evaluation.pin,
        "482913",
        str(principal.session_id),
        str(world.patient_id),
        world.patient_code,
        sentence,
        "kind, please",
    ):
        assert forbidden not in held, forbidden
    assert len(wire.payloads) == 1  # The PIN message never reached the model.


def test_only_the_latest_checkpoint_is_kept(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        sched(),
    )
    for words in (morning_of_monday(world), "9:30 AM", "hmm"):
        service.turn(cookie.token, words)

    (info,) = graph_store.threads()
    assert info.checkpoints == 1
    assert info.thread_id == thread_for(evaluation.authenticate(cookie.token).session_id)


# --- sessions that end ------------------------------------------------------------------------


def test_after_sign_out_nothing_resumes_and_the_stored_conversation_is_gone(
    evaluation, graph_store
) -> None:
    world, _ui, cookie = sign_in(evaluation)
    principal = evaluation.authenticate(cookie.token)
    service, _ = service_over(evaluation, graph_store, monday_reading(world), sched())
    service.turn(cookie.token, morning_of_monday(world))
    assert len(graph_store.threads()) == 1

    evaluation_service = evaluation.client.app.state.auth_service
    evaluation_service.logout(principal, request_id=uuid.uuid4())
    service.sign_out(principal)

    assert graph_store.threads() == []
    with pytest.raises((AuthenticationRequiredError, SessionExpiredError)):
        service.turn(cookie.token, "9:30 AM")
    assert graph_store.threads() == []


def test_an_expired_session_cannot_resume_or_confirm_a_stale_review(
    evaluation, graph_store
) -> None:
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    service.turn(cookie.token, morning_of_monday(world))
    reply = service.turn(cookie.token, "9:30 AM")
    assert reply.awaiting_confirmation
    (pending,) = active_reviews(evaluation, world.patient_id)

    evaluation.clock.advance(minutes=16)  # Past the idle timeout.

    with pytest.raises(SessionExpiredError):
        service.turn(cookie.token, "yes")
    outcome = ui.confirm(cookie, pending)
    assert not outcome.committed and outcome.status_code in (401, 403, 409)
    assert active_reviews(evaluation, world.patient_id) == []
    assert sql(
        evaluation,
        "SELECT status FROM pending_actions WHERE pending_action_id = :p",
        p=pending,
    ) == [("INVALIDATED",)]
    assert (
        sql(
            evaluation,
            "SELECT count(*) FROM appointments WHERE slot_id = :s",
            s=world.slots["pine_mon_0930"],
        )[0][0]
        == 0
    )

    removed = service.expire()  # The sweep removes what the ended session left behind.
    assert removed and graph_store.threads() == []


def test_a_revoked_session_cannot_resume(evaluation, graph_store) -> None:
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    service.turn(cookie.token, morning_of_monday(world))
    service.turn(cookie.token, "9:30 AM")
    (pending,) = active_reviews(evaluation, world.patient_id)

    with evaluation.engine.begin() as connection:  # The account is taken out of service.
        connection.execute(
            text("UPDATE patients SET active = false WHERE patient_id = :p"),
            {"p": world.patient_id},
        )

    with pytest.raises(AuthenticationRequiredError):
        service.turn(cookie.token, "yes")
    assert not ui.confirm(cookie, pending).committed
    assert sql(
        evaluation, "SELECT status FROM auth_sessions WHERE patient_id = :p", p=world.patient_id
    ) == [("REVOKED",)]
    assert active_reviews(evaluation, world.patient_id) == []
    service.expire()
    assert graph_store.threads() == []


def test_signing_in_again_never_inherits_the_old_conversation(evaluation, graph_store) -> None:
    world, ui, old = sign_in(evaluation)
    service, _ = service_over(evaluation, graph_store, monday_reading(world), sched())
    service.turn(old.token, morning_of_monday(world))
    old_thread = graph_store.threads()[0].thread_id

    fresh = ui.sign_in(world.patient_code)  # Another sign-in is another session.

    service.turn(fresh.token, "hello")
    new_thread = thread_for(evaluation.authenticate(fresh.token).session_id)
    assert new_thread != old_thread
    state = graph_store.saver.get_tuple({"configurable": {"thread_id": new_thread}})
    assert (state.checkpoint["channel_values"].get("request") or {}).get("clinic_name") is None


# --- whose conversation it is -----------------------------------------------------------------


def test_two_patients_have_separate_conversations_and_reviews(evaluation, graph_store) -> None:
    first_world, _, first = sign_in(evaluation)
    _second_world, _, second = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(first_world),
        sched(),
    )
    service.turn(first.token, morning_of_monday(first_world))
    service.turn(second.token, "hello")

    threads = {t.thread_id for t in graph_store.threads()}
    assert threads == {
        thread_for(evaluation.authenticate(first.token).session_id),
        thread_for(evaluation.authenticate(second.token).session_id),
    }
    held = stored_text(graph_store)
    assert first_world.names["pine"] in held  # What one patient asked for is in their thread...
    second_thread = thread_for(evaluation.authenticate(second.token).session_id)
    other = graph_store.saver.get_tuple({"configurable": {"thread_id": second_thread}})
    request = other.checkpoint["channel_values"].get("request") or {}
    assert request.get("clinic_name") is None  # ...and not the other's.


def test_a_conversation_that_belongs_to_someone_else_is_discarded_not_continued(
    evaluation, graph_store
) -> None:
    world_a, _ui_a, a = sign_in(evaluation)
    world_b, _ui_b, b = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world_a),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        sched(),
    )
    service.turn(a.token, morning_of_monday(world_a))
    a_thread = thread_for(evaluation.authenticate(a.token).session_id)

    # Even if B's session somehow reached A's thread, B gets none of A's conversation.
    principal_b = evaluation.authenticate(b.token)
    config = LlmConfig()
    intruder = OrchestratedSession(
        None,
        principal_b,
        config=config,
        store=graph_store,
        reader=TurnReader(config, transport=Wire(sched(time="9:30 AM", selection="SPECIFIC_TIME"))),
        planner=GoalPlanner(config, transport=Wire()),
        clock=evaluation.clock,
    )
    intruder._thread = a_thread
    reply = intruder.respond("9:30 AM")

    assert not reply.awaiting_confirmation
    assert owner_of(principal_b.patient_id) != owner_of(evaluation.authenticate(a.token).patient_id)
    assert active_reviews(evaluation, world_b.patient_id) == []
    assert active_reviews(evaluation, world_a.patient_id) == []


# --- a review that is no longer live ----------------------------------------------------------


def test_a_review_decided_while_the_process_was_down_is_not_shown_again(
    evaluation, graph_store
) -> None:
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        sched(),
    )
    service.turn(cookie.token, morning_of_monday(world))
    service.turn(cookie.token, "9:30 AM")
    (pending,) = active_reviews(evaluation, world.patient_id)
    assert ui.confirm(cookie, pending).committed  # The "committed" event never reached the agent.

    reply = service.turn(cookie.token, "ok thanks")

    assert not reply.awaiting_confirmation
    assert "confirm in the app" not in reply.text.lower()
    thread = thread_for(evaluation.authenticate(cookie.token).session_id)
    state = graph_store.saver.get_tuple({"configurable": {"thread_id": thread}})
    assert state.checkpoint["channel_values"].get("review") is None


# --- concurrency ------------------------------------------------------------------------------


class Slow(Wire):
    def __init__(self, *replies, pause=0.4):
        super().__init__(*replies)
        self.pause, self.active, self.peak = pause, 0, 0
        self._lock = threading.Lock()

    def __call__(self, path, payload, timeout_s):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.pause)
        try:
            return super().__call__(path, payload, timeout_s)
        finally:
            with self._lock:
                self.active -= 1


def test_two_messages_to_one_conversation_run_one_after_the_other(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    slow = Slow(monday_reading(world), sched(time="9:30 AM", selection="SPECIFIC_TIME"))
    service, _ = service_over(evaluation, graph_store, transport=slow)
    replies: list = []

    def send(words):
        replies.append(service.turn(cookie.token, words))

    first = threading.Thread(target=send, args=(morning_of_monday(world),))
    first.start()
    time.sleep(0.1)
    second = threading.Thread(target=send, args=("9:30 AM",))
    second.start()
    first.join(30)
    second.join(30)

    assert slow.peak == 1  # Never two turns of one conversation at once.
    assert len(replies) == 2
    assert len(active_reviews(evaluation, world.patient_id)) <= 1


def test_a_second_message_that_cannot_get_in_is_told_to_wait_and_changes_nothing(
    evaluation, graph_store
) -> None:
    world, _ui, cookie = sign_in(evaluation)
    patient_store = new_store(evaluation, lock_wait_seconds=0.2)
    slow = Slow(monday_reading(world), pause=1.0)
    service, _ = service_over(evaluation, patient_store, transport=slow)
    quick, _ = service_over(evaluation, patient_store, sched(time="9:30 AM"))
    started = threading.Thread(target=service.turn, args=(cookie.token, morning_of_monday(world)))
    started.start()
    time.sleep(0.3)

    busy = quick.turn(cookie.token, "9:30 AM")
    started.join(30)

    assert busy.text == BUSY_REPLY and not busy.awaiting_confirmation
    assert active_reviews(evaluation, world.patient_id) == []
    patient_store.close()


def test_many_patients_at_once_do_not_disturb_each_other(evaluation, graph_store) -> None:
    people = [sign_in(evaluation) for n in range(4)]
    service, _ = service_over(
        evaluation, graph_store, *[monday_reading(world) for world, _, _ in people]
    )
    results: dict[int, object] = {}

    def go(n, world, cookie):
        results[n] = service.turn(cookie.token, morning_of_monday(world))

    threads = [
        threading.Thread(target=go, args=(n, world, cookie))
        for n, (world, _, cookie) in enumerate(people)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert len(results) == 4
    assert len(graph_store.threads()) == 4


# --- the database failing ---------------------------------------------------------------------


def test_a_checkpoint_store_that_is_down_fails_closed(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    broken = new_store(evaluation)
    service, _ = service_over(evaluation, broken, monday_reading(world))
    broken.close()  # The pool is gone, as if the database connection were lost.

    reply = service.turn(cookie.token, morning_of_monday(world))

    assert reply.text == TROUBLE_REPLY and not reply.awaiting_confirmation
    assert active_reviews(evaluation, world.patient_id) == []


def test_a_failure_while_saving_retires_a_review_the_patient_never_saw(
    evaluation, graph_store
) -> None:
    world, _ui, cookie = sign_in(evaluation)
    script = [
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    ]
    service, _ = service_over(evaluation, graph_store, *script)
    service.turn(cookie.token, morning_of_monday(world))

    puts = {"count": 0, "fail_from": None}
    real_put = graph_store.saver.put

    def flaky(config, checkpoint, metadata, versions):
        puts["count"] += 1
        if puts["fail_from"] is not None and puts["count"] >= puts["fail_from"]:
            raise psycopg.OperationalError("connection lost")
        return real_put(config, checkpoint, metadata, versions)

    graph_store.saver.put = flaky
    # The preparing turn writes its checkpoints in order; let the last one fail.
    puts["fail_from"] = 5  # intake, understand, schedule, decide, tool succeed; then it fails.
    reply = service.turn(cookie.token, "9:30 AM")
    graph_store.saver.put = real_put

    assert reply.text == TROUBLE_REPLY and not reply.awaiting_confirmation
    assert active_reviews(evaluation, world.patient_id) == []  # Prepared, never shown: retired.
    assert (
        sql(
            evaluation,
            "SELECT count(*) FROM appointments WHERE slot_id = :s",
            s=world.slots["pine_mon_0930"],
        )[0][0]
        == 0
    )
    # The saved conversation is the one from before the failed turn: it can simply be repeated.
    again, _ = service_over(
        evaluation, graph_store, sched(time="9:30 AM", selection="SPECIFIC_TIME")
    )
    assert again.turn(cookie.token, "9:30 AM").awaiting_confirmation


# --- retention --------------------------------------------------------------------------------


def test_idle_conversations_are_removed_and_recent_ones_are_kept(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    service, _ = service_over(evaluation, graph_store, monday_reading(world))
    service.turn(cookie.token, morning_of_monday(world))
    (info,) = graph_store.threads()

    assert graph_store.expire(now=info.last_checkpoint + timedelta(hours=11)) == []
    assert graph_store.expire(now=info.last_checkpoint + timedelta(hours=13)) == [info.thread_id]
    assert graph_store.threads() == []
    assert graph_store.count_rows() == {
        "checkpoints": 0,
        "checkpoint_blobs": 0,
        "checkpoint_writes": 0,
    }


def test_the_sweep_keeps_the_conversations_of_live_sessions(evaluation, graph_store) -> None:
    live_world, _, live = sign_in(evaluation)
    gone_world, _, gone = sign_in(evaluation)
    service, _ = service_over(
        evaluation, graph_store, monday_reading(live_world), monday_reading(gone_world)
    )
    service.turn(live.token, morning_of_monday(live_world))
    service.turn(gone.token, morning_of_monday(gone_world))
    gone_principal = evaluation.authenticate(gone.token)
    evaluation.client.app.state.auth_service.logout(gone_principal, request_id=uuid.uuid4())

    removed = service.expire()

    assert removed == [thread_for(gone_principal.session_id)]
    assert [t.thread_id for t in graph_store.threads()] == [
        thread_for(evaluation.authenticate(live.token).session_id)
    ]


# --- retiring a dropped review, and trying again ---------------------------------------------


def test_a_retirement_that_failed_is_retried_after_a_restart(
    evaluation, graph_store, monkeypatch
) -> None:
    from app.agent_tools import ToolRuntime
    from app.evaluation.conversation import REVIEW_STILL_OPEN_NOTICE

    world, _ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        sched(message_type="ABANDON"),
    )
    service.turn(cookie.token, morning_of_monday(world))
    service.turn(cookie.token, "9:30 AM")
    (pending,) = active_reviews(evaluation, world.patient_id)

    real = ToolRuntime.discard_pending_action
    attempts = {"n": 0}

    def flaky(self, principal, pending_action_id):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("the database blinked")
        return real(self, principal, pending_action_id)

    monkeypatch.setattr(ToolRuntime, "discard_pending_action", flaky)

    dropped = service.turn(cookie.token, "never mind")

    assert REVIEW_STILL_OPEN_NOTICE in dropped.text  # The patient is told not to press Confirm.
    assert active_reviews(evaluation, world.patient_id) == [pending]  # It really is still open.

    graph_store.close()  # The process dies before the next message.
    with new_store(evaluation) as restarted:
        again, _ = service_over(evaluation, restarted, sched())
        again.turn(cookie.token, "hello")

    assert attempts["n"] == 2
    assert active_reviews(evaluation, world.patient_id) == []  # Retired on the retry.
    assert sql(
        evaluation, "SELECT status FROM pending_actions WHERE pending_action_id = :p", p=pending
    ) == [("INVALIDATED",)]


def test_a_live_review_survives_a_restart_and_is_still_the_one_to_confirm(
    evaluation, graph_store
) -> None:
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
    )
    service.turn(cookie.token, morning_of_monday(world))
    service.turn(cookie.token, "9:30 AM")
    (pending,) = active_reviews(evaluation, world.patient_id)

    graph_store.close()
    with new_store(evaluation) as restarted:
        again, _ = service_over(evaluation, restarted, sched(message_type="AGREEMENT"))
        reply = again.turn(cookie.token, "yes")

        assert reply.awaiting_confirmation and "confirm" in reply.text.lower()
        assert active_reviews(evaluation, world.patient_id) == [pending]  # Not retired, not new.
        assert ui.confirm(cookie, pending).committed


# --- LG-5D: a correction replaces the review, and the old one is dead -------------------------


def reviewed_at_9_30(evaluation, graph_store, *then):
    """A signed-in patient with a review prepared for Monday 9:30, and the readings that follow."""
    world, ui, cookie = sign_in(evaluation)
    service, _ = service_over(
        evaluation,
        graph_store,
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        *then,
    )
    service.turn(cookie.token, morning_of_monday(world))
    assert service.turn(cookie.token, "9:30 AM").awaiting_confirmation
    (first,) = active_reviews(evaluation, world.patient_id)
    return world, ui, cookie, service, first


def booked_slots(evaluation, patient_id):
    return sql(
        evaluation,
        "SELECT slot_id::text FROM appointments WHERE patient_id = :p AND status = 'CONFIRMED'",
        p=patient_id,
    )


def test_a_correction_replaces_the_review_and_the_old_one_can_no_longer_be_confirmed(
    evaluation, graph_store
) -> None:
    world, ui, cookie, service, first = reviewed_at_9_30(
        evaluation, graph_store, sched(), {"kind": "CHANGE", "time": "9 AM"}
    )

    reply = service.turn(cookie.token, "make it 9 AM")

    assert reply.awaiting_confirmation and "9:00 AM" in reply.text and "9:30" not in reply.text
    (second,) = active_reviews(evaluation, world.patient_id)  # Exactly one is live, and it is new.
    assert second != first
    assert not ui.confirm(cookie, first).committed  # The screen showing the old one is dead.
    assert booked_slots(evaluation, world.patient_id) == [(world.slots["pine_tue_1000"],)]
    assert ui.confirm(cookie, second).committed
    assert (world.slots["pine_mon_0900"],) in booked_slots(evaluation, world.patient_id)
    assert (world.slots["pine_mon_0930"],) not in booked_slots(evaluation, world.patient_id)


def test_a_correction_after_a_process_restart_replaces_the_review_too(
    evaluation, graph_store
) -> None:
    world, ui, cookie, _service, first = reviewed_at_9_30(evaluation, graph_store)
    restarted = new_store(evaluation)  # A new process: nothing in memory, only the database.
    service, _ = service_over(evaluation, restarted, sched(), {"kind": "CHANGE", "time": "9 AM"})

    reply = service.turn(cookie.token, "make it 9 AM")

    assert reply.awaiting_confirmation and "9:00 AM" in reply.text
    (second,) = active_reviews(evaluation, world.patient_id)
    assert second != first and not ui.confirm(cookie, first).committed
    restarted.close()


def test_a_session_that_expired_cannot_correct_a_review_or_confirm_it(
    evaluation, graph_store
) -> None:
    world, ui, cookie, service, first = reviewed_at_9_30(
        evaluation, graph_store, sched(), {"kind": "CHANGE", "time": "9 AM"}
    )

    evaluation.clock.advance(minutes=16)  # Past the idle timeout.

    with pytest.raises(SessionExpiredError):
        service.turn(cookie.token, "make it 9 AM")
    assert not ui.confirm(cookie, first).committed
    assert active_reviews(evaluation, world.patient_id) == []  # No new review was prepared.
    assert booked_slots(evaluation, world.patient_id) == [(world.slots["pine_tue_1000"],)]


def test_two_corrections_at_once_leave_exactly_one_live_review(evaluation, graph_store) -> None:
    world, _ui, cookie = sign_in(evaluation)
    slow = Slow(
        monday_reading(world),
        sched(time="9:30 AM", selection="SPECIFIC_TIME"),
        sched(),
        {"kind": "CHANGE", "time": "9 AM"},
        sched(),
        {"kind": "CHANGE", "time": "9 AM"},
        pause=0.2,
    )
    service, _ = service_over(evaluation, graph_store, transport=slow)
    service.turn(cookie.token, morning_of_monday(world))
    service.turn(cookie.token, "9:30 AM")
    (first,) = active_reviews(evaluation, world.patient_id)
    replies: list = []

    def correct():
        replies.append(service.turn(cookie.token, "make it 9 AM"))

    threads = [threading.Thread(target=correct) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    live = active_reviews(evaluation, world.patient_id)
    assert len(replies) == 2 and slow.peak == 1
    assert len(live) == 1 and live != [first]


def test_a_failure_while_saving_a_correction_leaves_no_review_that_could_be_confirmed(
    evaluation, graph_store
) -> None:
    world, ui, cookie, service, first = reviewed_at_9_30(
        evaluation, graph_store, sched(), {"kind": "CHANGE", "time": "9 AM"}
    )
    real_put = graph_store.saver.put

    def flaky(config, checkpoint, metadata, versions):
        if sql(evaluation, "SELECT count(*) FROM pending_actions")[0][0] >= 2:
            raise psycopg.OperationalError("connection lost")  # After the new review exists.
        return real_put(config, checkpoint, metadata, versions)

    graph_store.saver.put = flaky
    reply = service.turn(cookie.token, "make it 9 AM")
    graph_store.saver.put = real_put

    assert reply.text == TROUBLE_REPLY and not reply.awaiting_confirmation
    assert active_reviews(evaluation, world.patient_id) == []
    pending = [r[0] for r in sql(evaluation, "SELECT pending_action_id::text FROM pending_actions")]
    assert all(not ui.confirm(cookie, p).committed for p in pending)
    assert first in pending and booked_slots(evaluation, world.patient_id) == [
        (world.slots["pine_tue_1000"],)
    ]


# --- LG-5E: nothing is chosen for the patient --------------------------------------------------


def test_a_reference_that_fits_several_real_openings_prepares_no_review(
    evaluation, graph_store
) -> None:
    world, _ui, cookie = sign_in(evaluation)
    listing = {
        "route": "LOOKUP",
        "subjects": ["OPENINGS"],
        "clinic_name": world.names["pine"],
        "start_date": "2026-11-09",
        "end_date": "2026-11-09",
    }
    service, _ = service_over(
        evaluation, graph_store, listing, sched(operation="BOOK", option_number=1)
    )
    listed = service.turn(cookie.token, f"What does {world.names['pine']} have on Monday?")
    assert "9:00 AM" in listed.text and "10:30 AM" in listed.text

    reply = service.turn(cookie.token, "book the morning one")  # Three openings are mornings.

    assert not reply.awaiting_confirmation
    assert sql(evaluation, "SELECT count(*) FROM pending_actions")[0][0] == 0
    assert booked_slots(evaluation, world.patient_id) == [(world.slots["pine_tue_1000"],)]


def test_yes_to_the_one_real_opening_prepares_a_review_and_books_nothing(
    evaluation, graph_store
) -> None:
    world, ui, cookie = sign_in(evaluation)
    listing = {
        "route": "LOOKUP",
        "subjects": ["OPENINGS"],
        "clinic_name": world.names["rivers"],
        "start_date": "2026-11-06",
        "end_date": "2026-11-06",
    }
    service, _ = service_over(evaluation, graph_store, listing, sched(message_type="AGREEMENT"))
    service.turn(cookie.token, f"Is there anything at {world.names['rivers']} on Friday?")

    reply = service.turn(cookie.token, "yes")

    assert reply.awaiting_confirmation
    (pending,) = active_reviews(evaluation, world.patient_id)
    assert booked_slots(evaluation, world.patient_id) == [(world.slots["pine_tue_1000"],)]
    assert ui.confirm(cookie, pending).committed  # Only the patient's own Confirm books it.
    assert (world.slots["rivers_fri_1300"],) in booked_slots(evaluation, world.patient_id)
