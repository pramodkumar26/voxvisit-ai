"""Voice-to-browser review handoff.

A review prepared in a voice session is confirmed, or not, in the patient's own browser session,
and the voice session learns the outcome from the database alone.

Everything here is real: PostgreSQL, the authentication service, the prepare and confirm services,
the HTTP routes with their Origin and CSRF checks. The browser is played by authenticated requests
under its own session; the voice session by the service calls the voice agent makes."""

import uuid

import pytest
from action_support import (
    age_pending_action,
    appointment_row,
    book_body,
    cancel_body,
    changed_tables,
    confirm,
    error_of,
    new_key,
    new_patient,
    patient_session,
    prepared,
    principal_of,
    reschedule_body,
    rows,
    scalar,
    seed_appointment,
    slot_row,
    snapshot,
    world,
)
from auth_support import call

from app.core.errors import ResourceNotAvailableError
from app.services import review_handoffs as handoffs

CLAIM = "/me/voice-handoffs/claim"
REVIEW = "/me/voice-handoffs/review"
DECLINE = "/me/voice-handoffs/decline"


def booked(env, slots) -> int:
    """Confirmed appointments on these slots (the seeded database holds others of its own)."""
    ids = [str(slot) for slot in (slots if isinstance(slots, list | tuple) else [slots])]
    return scalar(
        env.engine,
        "SELECT count(*) FROM appointments WHERE slot_id::text = ANY(:ids)"
        " AND status = 'CONFIRMED'",
        ids=ids,
    )


def ticket_for(env, voice, pending=None):
    """The voice session asks for a ticket, as the voice agent does."""
    return handoffs.create_handoff(
        env.engine, principal=principal_of(env, voice), request_id=uuid.uuid4(), now=env.clock()
    )


def status_of(env, voice, ticket):
    return handoffs.handoff_status(
        env.engine,
        principal=principal_of(env, voice),
        handoff_id=ticket.handoff_id,
        now=env.clock(),
    )


def claim(env, browser, code, **kwargs):
    return call(env, "POST", CLAIM, browser, json={"code": code}, **kwargs)


class Review:
    """A patient with a voice session, a prepared booking, its ticket and a browser session."""

    def __init__(self, env, code="PT-1001", slots=3):
        self.env = env
        self.scene = world(env.engine, count=slots)
        self.slot = self.scene.slots[0]
        self.code = code
        self.voice = patient_session(env, code)
        self.voice_pending = prepared(env, self.voice, book_body(self.slot))
        self.ticket = ticket_for(env, self.voice)
        self.browser = patient_session(env, code)

    def claim(self, **kwargs):
        return claim(self.env, self.browser, self.ticket.code, **kwargs)

    def claimed(self):
        response = self.claim()
        assert response.status_code == 200, response.text
        return response.json()["pending_action_id"]

    def status(self):
        return status_of(self.env, self.voice, self.ticket)


@pytest.fixture
def review(auth_env) -> Review:
    return Review(auth_env)


# --- the whole workflow -----------------------------------------------------------------------


def test_a_voice_prepared_booking_is_confirmed_in_the_browser_and_the_voice_session_sees_it(
    auth_env, review
) -> None:
    env = auth_env
    before = snapshot(env.engine)
    assert review.status().state == handoffs.WAITING

    response = review.claim()

    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    shown = response.json()
    proposed = shown["summary"]["proposed"]
    assert shown["action"] == "BOOK" and shown["ui_status"] == "AWAITING_UI_CONFIRMATION"
    assert {"clinic", "doctor", "specialty", "date", "time", "timezone"} <= set(proposed)
    assert proposed["clinic"] == review.scene.clinic.name
    # Preparing a review reserves and changes nothing: no appointment, no slot, no history.
    assert slot_row(env.engine, review.slot).status == "AVAILABLE"
    assert changed_tables(before, snapshot(env.engine)) <= {"pending", "idempotency", "audit"}
    assert rows(env.engine, "SELECT count(*) FROM appointments WHERE slot_id = :s", s=review.slot)[
        0
    ] == (0,)
    assert review.status().state == handoffs.IN_BROWSER

    outcome = confirm(env, review.browser, shown["pending_action_id"])

    assert outcome.status_code == 200 and outcome.json()["result"] == "BOOKED"
    assert slot_row(env.engine, review.slot).status == "BOOKED"
    appointment = appointment_row(env.engine, outcome.json()["appointment_id"])
    assert appointment.status == "CONFIRMED" and appointment.slot_id == review.slot
    verified = review.status()
    assert verified.state == handoffs.COMMITTED and verified.result == "BOOKED"
    assert verified.summary.proposed.clinic == review.scene.clinic.name


def test_the_voice_review_is_retired_and_only_one_review_is_live(auth_env, review) -> None:
    review.claimed()

    states = dict(
        rows(
            auth_env.engine,
            "SELECT pending_action_id::text, status FROM pending_actions"
            " WHERE session_id IN (SELECT session_id FROM auth_sessions WHERE patient_id ="
            " (SELECT patient_id FROM patients WHERE patient_code = :c))",
            c=review.code,
        )
    )

    assert states[review.voice_pending] == "INVALIDATED"
    assert sorted(states.values()).count("ACTIVE") == 1


def test_a_reschedule_and_a_cancellation_can_be_handed_off_and_confirmed(auth_env) -> None:
    env = auth_env
    scene = world(env.engine, count=4)
    appointment = seed_appointment(env.engine, "PT-1001", scene.slots[0])
    voice = patient_session(env)
    prepared(env, voice, reschedule_body(appointment, scene.slots[1]))
    ticket = ticket_for(env, voice)
    browser = patient_session(env)

    shown = claim(env, browser, ticket.code).json()
    moved = confirm(env, browser, shown["pending_action_id"])

    assert shown["action"] == "RESCHEDULE" and moved.json()["result"] == "RESCHEDULED"
    assert status_of(env, voice, ticket).result == "RESCHEDULED"
    assert slot_row(env.engine, scene.slots[0]).status == "AVAILABLE"

    for finished in (voice, browser):  # A patient holds three sessions at most; end this pair.
        env.service.logout(principal_of(env, finished), request_id=uuid.uuid4())
    voice2 = patient_session(env)
    prepared(env, voice2, cancel_body(appointment))
    ticket2 = ticket_for(env, voice2)
    browser2 = patient_session(env)
    claimed2 = claim(env, browser2, ticket2.code)
    assert claimed2.status_code == 200, claimed2.text
    shown2 = claimed2.json()
    cancelled = confirm(env, browser2, shown2["pending_action_id"])

    assert shown2["action"] == "CANCEL" and cancelled.json()["result"] == "CANCELLED"
    assert status_of(env, voice2, ticket2).state == handoffs.COMMITTED


# --- the ticket -------------------------------------------------------------------------------


def test_the_code_is_never_stored_only_its_hash(auth_env, review) -> None:
    stored = rows(auth_env.engine, "SELECT row_to_json(h)::text FROM review_handoffs h")

    assert stored and review.ticket.code not in "".join(row[0] for row in stored)
    assert len(review.ticket.code) >= 40
    assert scalar(auth_env.engine, "SELECT length(code_hash) FROM review_handoffs") == 32


def test_a_ticket_is_single_use_but_the_same_browser_gets_its_own_review_back(
    auth_env, review
) -> None:
    first = review.claim().json()
    again = review.claim()
    elsewhere = claim(auth_env, patient_session(auth_env), review.ticket.code)

    assert (
        again.status_code == 200 and again.json()["pending_action_id"] == first["pending_action_id"]
    )
    assert elsewhere.status_code == 404 and error_of(elsewhere)["code"] == "HANDOFF_INVALID"
    live = scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status = 'ACTIVE'")
    assert live == 1  # Presenting it twice did not create a second review.


def test_an_interrupted_claim_can_be_finished_only_by_the_session_that_began_it(
    auth_env, review
) -> None:
    """The ticket is spent the moment a session reserves it, before its review exists. If that
    session is interrupted there, no other session may take over; the same one can finish."""
    began = principal_of(auth_env, review.browser).session_id
    with auth_env.engine.begin() as connection:
        connection.execute(
            __import__("sqlalchemy").text(
                "UPDATE review_handoffs SET claimed_session_id = :s, claimed_at = now()"
            ),
            {"s": began},
        )
    other_session = patient_session(auth_env)

    taken_over = claim(auth_env, other_session, review.ticket.code)
    finished = review.claim()

    assert taken_over.status_code == 404 and error_of(taken_over)["code"] == "HANDOFF_INVALID"
    assert finished.status_code == 200
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status='ACTIVE'") == 1
    )
    assert review.status().state == handoffs.IN_BROWSER


def test_another_patient_cannot_claim_it_and_learns_nothing_from_the_refusal(
    auth_env, review
) -> None:
    stranger = patient_session(auth_env, new_patient(auth_env))

    wrong_patient = claim(auth_env, stranger, review.ticket.code)
    unknown = claim(auth_env, stranger, "x" * 43)

    assert wrong_patient.status_code == unknown.status_code == 404
    assert wrong_patient.json() == unknown.json()  # Not told apart.
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status='ACTIVE'") == 1
    )
    assert slot_row(auth_env.engine, review.slot).status == "AVAILABLE"


def test_a_ticket_that_has_expired_cannot_be_claimed(auth_env, review) -> None:
    with auth_env.engine.begin() as connection:
        connection.execute(
            __import__("sqlalchemy").text(
                "UPDATE review_handoffs SET created_at = clock_timestamp() - interval '9 minutes',"
                " expires_at = clock_timestamp() - interval '4 minutes'"
            )
        )

    response = review.claim()

    assert response.status_code == 404 and error_of(response)["code"] == "HANDOFF_INVALID"
    assert review.status().state == handoffs.EXPIRED
    assert booked(auth_env, review.scene.slots) == 0


def test_a_review_changed_by_voice_before_the_claim_is_stale_and_cannot_be_claimed(
    auth_env, review
) -> None:
    prepared(auth_env, review.voice, book_body(review.scene.slots[1]))  # The patient changed it.

    response = review.claim()

    assert response.status_code == 404 and error_of(response)["code"] == "HANDOFF_INVALID"
    assert booked(auth_env, review.scene.slots) == 0


def test_a_voice_session_that_has_ended_leaves_a_ticket_nobody_can_redeem(auth_env, review) -> None:
    auth_env.service.logout(principal_of(auth_env, review.voice), request_id=uuid.uuid4())

    response = review.claim()

    assert response.status_code == 404 and error_of(response)["code"] == "HANDOFF_INVALID"
    assert booked(auth_env, review.scene.slots) == 0


def test_a_ticket_is_replaced_by_a_newer_one_and_the_old_code_stops_working(
    auth_env, review
) -> None:
    prepared(auth_env, review.voice, book_body(review.scene.slots[1]))
    newer = ticket_for(auth_env, review.voice)

    old = review.claim()
    fresh = claim(auth_env, review.browser, newer.code)

    assert old.status_code == 404 and fresh.status_code == 200
    assert fresh.json()["summary"]["proposed"]["time"] is not None


def test_there_is_nothing_to_hand_off_without_a_live_review(auth_env) -> None:
    voice = patient_session(auth_env)

    with pytest.raises(ResourceNotAvailableError):
        ticket_for(auth_env, voice)

    pending = prepared(auth_env, voice, book_body(world(auth_env.engine).slots[0]))
    age_pending_action(auth_env.engine, pending, minutes=11)
    with pytest.raises(ResourceNotAvailableError):
        ticket_for(auth_env, voice)


# --- who may do what --------------------------------------------------------------------------


def test_claiming_needs_a_signed_in_patient_an_allowed_origin_and_the_csrf_token(
    auth_env, review
) -> None:
    anonymous = call(auth_env, "POST", CLAIM, None, json={"code": review.ticket.code})
    no_csrf = review.claim(csrf=None)
    wrong_csrf = review.claim(csrf="not-the-token")
    bad_origin = review.claim(origin="http://evil.example")
    no_origin = review.claim(origin=None)
    staff = claim(
        auth_env,
        __import__("action_support").staff_session(auth_env),
        review.ticket.code,
    )

    assert anonymous.status_code == 401
    assert {no_csrf.status_code, wrong_csrf.status_code, bad_origin.status_code} == {403}
    assert no_origin.status_code == 403 and staff.status_code in (401, 403)
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status='ACTIVE'") == 1
    )


def test_the_ticket_travels_in_the_body_never_in_the_url(auth_env, review) -> None:
    in_url = call(auth_env, "POST", f"{CLAIM}?code={review.ticket.code}", review.browser)
    malformed = review.claim() if False else claim(auth_env, review.browser, "short")

    assert in_url.status_code == 422 and malformed.status_code == 422
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status='ACTIVE'") == 1
    )


def test_a_browser_session_cannot_confirm_the_voice_sessions_review(auth_env, review) -> None:
    other_browser = patient_session(auth_env)
    claimed = review.claimed()

    by_browser = confirm(auth_env, review.browser, review.voice_pending)
    by_other_session = confirm(auth_env, other_browser, claimed)
    by_voice = confirm(auth_env, review.voice, claimed)

    assert {by_browser.status_code, by_other_session.status_code, by_voice.status_code} == {404}
    assert booked(auth_env, review.scene.slots) == 0


def test_another_patient_cannot_confirm_the_browsers_review(auth_env, review) -> None:
    claimed = review.claimed()
    stranger = patient_session(auth_env, new_patient(auth_env))

    response = confirm(auth_env, stranger, claimed)

    assert response.status_code == 404 and booked(auth_env, review.scene.slots) == 0


def test_confirming_without_authorization_commits_nothing(auth_env, review) -> None:
    claimed = review.claimed()
    before = snapshot(auth_env.engine)

    no_session = call(
        auth_env,
        "POST",
        f"/me/appointment-actions/{claimed}/confirm",
        None,
        json={"confirmed": True},
        headers={"Idempotency-Key": new_key()},
    )
    no_csrf = confirm(auth_env, review.browser, claimed, csrf=None)
    not_true = confirm(auth_env, review.browser, claimed, body={"confirmed": "yes"})

    assert no_session.status_code == 401 and no_csrf.status_code == 403
    assert not_true.status_code == 422
    assert snapshot(auth_env.engine) == before


def test_the_voice_session_can_read_only_its_own_handoffs(auth_env, review) -> None:
    other_voice = patient_session(auth_env)
    stranger_voice = patient_session(auth_env, new_patient(auth_env))

    for session in (other_voice, stranger_voice):
        with pytest.raises(ResourceNotAvailableError):
            status_of(auth_env, session, review.ticket)


def test_an_expired_voice_session_cannot_ask_for_a_ticket_or_a_status(auth_env, review) -> None:
    from app.core.errors import SessionExpiredError

    auth_env.clock.advance(minutes=16)

    with pytest.raises(SessionExpiredError):
        review.status()
    with pytest.raises(SessionExpiredError):
        ticket_for(auth_env, review.voice)


# --- not confirmed ----------------------------------------------------------------------------


def test_declining_changes_nothing_and_the_voice_session_is_told_it_was_not_committed(
    auth_env, review
) -> None:
    claimed = review.claimed()
    before = snapshot(auth_env.engine)

    declined = call(auth_env, "POST", DECLINE, review.browser)
    late_confirm = confirm(auth_env, review.browser, claimed)

    assert declined.status_code == 200 and declined.json() == {"declined": True}
    assert (
        late_confirm.status_code == 409
        and error_of(late_confirm)["reason"] == "CONFIRMATION_INVALIDATED"
    )
    assert review.status().state == handoffs.NOT_COMMITTED and review.status().result is None
    assert changed_tables(before, snapshot(auth_env.engine)) <= {"pending", "audit"}
    assert slot_row(auth_env.engine, review.slot).status == "AVAILABLE"
    assert booked(auth_env, review.scene.slots) == 0


def test_declining_with_nothing_waiting_is_harmless(auth_env) -> None:
    browser = patient_session(auth_env)

    response = call(auth_env, "POST", DECLINE, browser)

    assert response.status_code == 200 and response.json() == {"declined": False}


def test_a_browser_review_that_lapses_cannot_be_confirmed(auth_env, review) -> None:
    claimed = review.claimed()
    age_pending_action(auth_env.engine, claimed, minutes=11)

    response = confirm(auth_env, review.browser, claimed)

    assert response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_EXPIRED"
    assert review.status().state == handoffs.EXPIRED
    assert call(auth_env, "GET", REVIEW, review.browser).status_code == 404
    assert booked(auth_env, review.scene.slots) == 0


def test_a_slot_taken_before_the_claim_is_reported_and_nothing_is_opened(auth_env, review) -> None:
    seed_appointment(auth_env.engine, new_patient(auth_env), review.slot)

    response = review.claim()

    assert response.status_code == 409 and error_of(response)["code"] == "SLOT_UNAVAILABLE"
    assert review.status().state == handoffs.SUPERSEDED
    assert (
        scalar(auth_env.engine, "SELECT count(*) FROM pending_actions WHERE status='ACTIVE'") == 0
    )
    assert review.claim().status_code == 404  # The spent ticket does not open anything later.


def test_a_slot_taken_after_the_claim_cannot_be_confirmed(auth_env, review) -> None:
    claimed = review.claimed()
    seed_appointment(auth_env.engine, new_patient(auth_env), review.slot)

    response = confirm(auth_env, review.browser, claimed)

    assert response.status_code == 409
    assert review.status().state != handoffs.COMMITTED
    # The only confirmed appointment on the slot is the other patient's; PT-1001 has none there.
    assert booked(auth_env, review.slot) == 1
    assert (
        scalar(
            auth_env.engine,
            "SELECT count(*) FROM appointments a JOIN patients p USING (patient_id)"
            " WHERE a.slot_id = :s AND p.patient_code = 'PT-1001'",
            s=review.slot,
        )
        == 0
    )


def test_speaking_again_retires_the_ticket_and_the_review_it_opened(auth_env, review) -> None:
    claimed = review.claimed()

    retired = handoffs.supersede_open_handoffs(
        auth_env.engine,
        principal=principal_of(auth_env, review.voice),
        request_id=uuid.uuid4(),
        now=auth_env.clock(),
    )
    response = confirm(auth_env, review.browser, claimed)

    assert retired == 1 and review.status().state == handoffs.SUPERSEDED
    assert (
        response.status_code == 409 and error_of(response)["reason"] == "CONFIRMATION_INVALIDATED"
    )
    assert booked(auth_env, review.scene.slots) == 0


def test_retiring_never_undoes_a_commit_that_already_happened(auth_env, review) -> None:
    claimed = review.claimed()
    assert confirm(auth_env, review.browser, claimed).status_code == 200

    handoffs.supersede_open_handoffs(
        auth_env.engine,
        principal=principal_of(auth_env, review.voice),
        request_id=uuid.uuid4(),
        now=auth_env.clock(),
    )

    assert booked(auth_env, review.scene.slots) == 1
    assert (
        scalar(
            auth_env.engine,
            "SELECT status FROM pending_actions WHERE pending_action_id = :p",
            p=claimed,
        )
        == "CONSUMED"
    )


# --- retries and duplicates -------------------------------------------------------------------


def test_confirming_twice_commits_once_and_a_retry_with_the_same_key_replays(
    auth_env, review
) -> None:
    claimed = review.claimed()
    key = new_key()

    first = confirm(auth_env, review.browser, claimed, key=key)
    replay = confirm(auth_env, review.browser, claimed, key=key)
    second_key = confirm(auth_env, review.browser, claimed)

    assert first.status_code == 200 and replay.status_code == 200
    assert replay.headers.get("idempotent-replay") == "true"
    assert replay.json()["appointment_id"] == first.json()["appointment_id"]
    assert (
        second_key.status_code == 409 and error_of(second_key)["reason"] == "CONFIRMATION_CONSUMED"
    )
    assert booked(auth_env, review.scene.slots) == 1
    assert review.status().state == handoffs.COMMITTED


def test_the_browser_can_reload_and_find_its_review_without_the_ticket(auth_env, review) -> None:
    claimed = review.claimed()

    response = call(auth_env, "GET", REVIEW, review.browser)

    assert response.status_code == 200 and response.json()["pending_action_id"] == claimed
    assert call(auth_env, "GET", REVIEW, patient_session(auth_env)).status_code == 404
    assert call(auth_env, "GET", REVIEW, None).status_code == 401


# --- audit and the page -----------------------------------------------------------------------


def test_the_handoff_is_audited_without_the_code(auth_env, review) -> None:
    review.claimed()
    claim(auth_env, patient_session(auth_env, new_patient(auth_env)), review.ticket.code)

    events = rows(
        auth_env.engine,
        "SELECT event_type, result, details::text FROM security_audit_events"
        " WHERE event_type LIKE 'REVIEW_HANDOFF_%' ORDER BY occurred_at",
    )

    kinds = [(e[0], e[1]) for e in events]
    assert ("REVIEW_HANDOFF_CREATED", "SUCCESS") in kinds
    assert ("REVIEW_HANDOFF_CLAIMED", "SUCCESS") in kinds
    assert ("REVIEW_HANDOFF_REJECTED", "DENIED") in kinds
    assert review.ticket.code not in "".join(e[2] for e in events)


def test_the_page_is_served_with_a_strict_policy_and_holds_no_secret(auth_env) -> None:
    page = auth_env.client.get("/voice-review")
    script = auth_env.client.get("/voice-review/app.js")
    style = auth_env.client.get("/voice-review/app.css")

    assert page.status_code == script.status_code == style.status_code == 200
    for response in (page, script, style):
        assert response.headers["cache-control"] == "no-store"
        policy = response.headers["content-security-policy"]
        assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
        assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
    assert "text/html" in page.headers["content-type"]
    assert "javascript" in script.headers["content-type"]
    assert "voxvisit_session" not in page.text + script.text  # No cookie is read by script.
    assert "<script>" not in page.text and "onclick" not in page.text  # No inline code.


def test_the_pages_script_never_stores_the_pin_or_writes_markup_from_the_server(auth_env) -> None:
    script = auth_env.client.get("/voice-review/app.js").text

    for forbidden in (
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "document.cookie",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "eval(",
        "new Function",
    ):
        assert forbidden not in script, forbidden
    assert 'type="password"' in auth_env.client.get("/voice-review").text
    assert "window.history.replaceState" in script  # The ticket is removed from the address bar.
