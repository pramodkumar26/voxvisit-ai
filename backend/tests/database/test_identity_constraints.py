import uuid
from datetime import timedelta

from factories import (
    BASE_TIME,
    FAR_FUTURE,
    make,
    make_context,
    make_patient_session,
    make_slot,
    make_staff_session,
    make_waitlist_entry,
    reject,
    reject_insert,
)
from sqlalchemy import Connection, delete, func, select

from app.db.models import (
    Appointment,
    AppointmentHistory,
    AuthSession,
    HandoffNote,
    HandoffTicket,
    IdempotencyRecord,
    NotificationEvent,
    OutboxEvent,
    Patient,
    PatientCredential,
    PendingAction,
    SecurityAuditEvent,
    SlotOffer,
    StaffCredential,
    StaffUser,
)


def test_patient_and_staff_identity_constraints(conn: Connection) -> None:
    patient = make(conn, Patient)
    staff = make(conn, StaffUser)

    reject_insert(conn, Patient, "uq_patients_patient_code", patient_code=patient.patient_code)
    reject_insert(
        conn,
        StaffUser,
        "uq_staff_users_username_normalized",
        username=staff.username.upper(),
    )


def test_credentials_require_an_existing_account_and_non_negative_counters(
    conn: Connection,
) -> None:
    patient = make(conn, Patient)
    staff = make(conn, StaffUser)

    reject_insert(
        conn,
        PatientCredential,
        "fk_patient_credentials_patient_id_patients",
        patient_id=uuid.uuid4(),
    )
    reject_insert(
        conn,
        PatientCredential,
        "ck_patient_credentials_failed_attempt_count_non_negative",
        patient_id=patient.patient_id,
        failed_attempt_count=-1,
    )
    reject_insert(
        conn,
        StaffCredential,
        "ck_staff_credentials_failed_attempt_count_non_negative",
        staff_id=staff.staff_id,
        failed_attempt_count=-1,
    )
    make(conn, PatientCredential, patient_id=patient.patient_id)
    reject_insert(conn, PatientCredential, "pk_patient_credentials", patient_id=patient.patient_id)


def test_session_must_belong_to_exactly_one_matching_actor(conn: Connection) -> None:
    patient = make(conn, Patient)
    staff = make(conn, StaffUser)
    base = {"actor_type": "PATIENT", "role": "PATIENT", "patient_id": patient.patient_id}

    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_exactly_one_actor",
        **{**base, "staff_id": staff.staff_id},
    )
    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_exactly_one_actor",
        actor_type="STAFF",
        role="STAFF",
        patient_id=patient.patient_id,
    )
    reject_insert(
        conn, AuthSession, "ck_auth_sessions_role_matches_actor", **{**base, "role": "STAFF"}
    )


def test_session_status_deadlines_and_termination_are_consistent(conn: Connection) -> None:
    patient = make(conn, Patient)

    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_status_valid",
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=patient.patient_id,
        status="SUSPENDED",
    )
    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_termination_matches_status",
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=patient.patient_id,
        status="ACTIVE",
        terminated_at=BASE_TIME,
    )
    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_termination_matches_status",
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=patient.patient_id,
        status="REVOKED",
    )
    reject_insert(
        conn,
        AuthSession,
        "ck_auth_sessions_deadlines_follow_creation",
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=patient.patient_id,
        created_at=BASE_TIME,
        last_activity_at=BASE_TIME,
        idle_expires_at=BASE_TIME,
        absolute_expires_at=BASE_TIME + timedelta(minutes=30),
    )


def test_session_token_hash_is_unique(conn: Connection) -> None:
    session = make_patient_session(conn, make(conn, Patient))

    reject_insert(
        conn,
        AuthSession,
        "uq_auth_sessions_token_hash",
        actor_type="PATIENT",
        role="PATIENT",
        patient_id=make(conn, Patient).patient_id,
        token_hash=session.token_hash,
    )


def test_staff_may_have_only_one_active_session(conn: Connection) -> None:
    staff = make(conn, StaffUser)
    make_staff_session(conn, staff)

    reject_insert(
        conn,
        AuthSession,
        "uq_auth_sessions_staff_active",
        actor_type="STAFF",
        role="STAFF",
        staff_id=staff.staff_id,
        token_hash=uuid.uuid4().bytes,
        idle_expires_at=FAR_FUTURE,
        absolute_expires_at=FAR_FUTURE,
    )


def test_staff_session_replacement_and_other_staff_are_allowed(conn: Connection) -> None:
    staff = make(conn, StaffUser)
    make_staff_session(conn, staff, status="REVOKED", terminated_at=BASE_TIME)
    make_staff_session(conn, staff, status="LOGGED_OUT", terminated_at=BASE_TIME)

    make_staff_session(conn, staff)
    make_staff_session(conn, make(conn, StaffUser))


def test_patient_session_count_is_a_service_rule_not_a_database_constraint(
    conn: Connection,
) -> None:
    patient = make(conn, Patient)

    for _ in range(4):
        make_patient_session(conn, patient)

    active = conn.execute(
        select(func.count()).where(
            AuthSession.patient_id == patient.patient_id, AuthSession.status == "ACTIVE"
        )
    ).scalar_one()
    assert active == 4


def test_pending_action_expiry_window_and_values(conn: Connection) -> None:
    session = make_patient_session(conn, make(conn, Patient))
    base = {"session_id": session.session_id}

    reject_insert(
        conn,
        PendingAction,
        "ck_pending_actions_expiry_window_valid",
        expires_at=BASE_TIME + timedelta(minutes=10, seconds=1),
        **base,
    )
    reject_insert(
        conn, PendingAction, "ck_pending_actions_expiry_window_valid", expires_at=BASE_TIME, **base
    )
    reject_insert(conn, PendingAction, "ck_pending_actions_status_valid", status="PENDING", **base)
    reject_insert(
        conn,
        PendingAction,
        "ck_pending_actions_action_type_valid",
        action_type="RESCHEDULED",
        **base,
    )
    action = make(conn, PendingAction, **base)
    reject_insert(
        conn, PendingAction, "uq_pending_actions_request_id", request_id=action.request_id, **base
    )


def test_pending_actions_require_a_session(conn: Connection) -> None:
    reject_insert(
        conn, PendingAction, "fk_pending_actions_session_id_auth_sessions", session_id=uuid.uuid4()
    )


def test_session_cleanup_removes_pending_actions_but_preserves_historical_records(
    conn: Connection,
) -> None:
    context = make_context(conn)
    patient = make(conn, Patient)
    staff = make(conn, StaffUser)
    slot = make_slot(conn, context)
    session = make_patient_session(conn, patient)

    pending = make(conn, PendingAction, session_id=session.session_id, target_id=slot.slot_id)
    ticket = make(
        conn,
        HandoffTicket,
        patient_id=patient.patient_id,
        created_by_session_id=session.session_id,
    )
    make(conn, HandoffNote, ticket_id=ticket.ticket_id, staff_id=staff.staff_id)
    idempotency = make(
        conn,
        IdempotencyRecord,
        actor_id=patient.patient_id,
        session_id=session.session_id,
    )
    audit = make(
        conn,
        SecurityAuditEvent,
        actor_type="PATIENT",
        actor_id=patient.patient_id,
        session_id=session.session_id,
    )
    appointment = make(conn, Appointment, patient_id=patient.patient_id, slot_id=slot.slot_id)
    make(
        conn,
        AppointmentHistory,
        appointment_id=appointment.appointment_id,
        actor_type="PATIENT",
        actor_id=patient.patient_id,
        new_slot_id=slot.slot_id,
    )
    entry = make_waitlist_entry(conn, patient, context.clinic)
    offer = make(
        conn,
        SlotOffer,
        waitlist_id=entry.waitlist_id,
        patient_id=patient.patient_id,
        slot_id=make_slot(conn, context).slot_id,
    )
    make(conn, NotificationEvent, patient_id=patient.patient_id, offer_id=offer.offer_id)
    make(conn, OutboxEvent, aggregate_id=slot.slot_id)

    preserved = [
        Appointment,
        AppointmentHistory,
        HandoffTicket,
        HandoffNote,
        IdempotencyRecord,
        NotificationEvent,
        OutboxEvent,
        SecurityAuditEvent,
        SlotOffer,
    ]
    before = {
        model: conn.execute(select(func.count()).select_from(model)).scalar_one()
        for model in preserved
    }

    conn.execute(delete(AuthSession).where(AuthSession.session_id == session.session_id))

    assert (
        conn.execute(
            select(func.count()).where(PendingAction.pending_action_id == pending.pending_action_id)
        ).scalar_one()
        == 0
    )
    after = {
        model: conn.execute(select(func.count()).select_from(model)).scalar_one()
        for model in preserved
    }
    assert after == before
    for model, key, column in (
        (HandoffTicket, ticket.ticket_id, "created_by_session_id"),
        (IdempotencyRecord, idempotency.idempotency_id, "session_id"),
        (SecurityAuditEvent, audit.audit_event_id, "session_id"),
    ):
        pk = next(iter(model.__table__.primary_key.columns))
        value = conn.execute(select(model.__table__.c[column]).where(pk == key)).scalar_one()
        assert value is None


def test_accounts_with_sessions_cannot_be_deleted(conn: Connection) -> None:
    patient = make(conn, Patient)
    staff = make(conn, StaffUser)
    make_patient_session(conn, patient)
    make_staff_session(conn, staff)

    reject(
        conn,
        "fk_auth_sessions_patient_id_patients",
        delete(Patient).where(Patient.patient_id == patient.patient_id),
    )
    reject(
        conn,
        "fk_auth_sessions_staff_id_staff_users",
        delete(StaffUser).where(StaffUser.staff_id == staff.staff_id),
    )
