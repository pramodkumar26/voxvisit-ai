import uuid

from factories import (
    BASE_TIME,
    make,
    reject,
    reject_insert,
)
from sqlalchemy import Connection, delete

from app.db.models import (
    HandoffNote,
    HandoffTicket,
    IdempotencyRecord,
    NotificationEvent,
    OutboxEvent,
    Patient,
    SecurityAuditEvent,
    StaffUser,
)


def test_handoff_ticket_status_summary_and_resolution(conn: Connection) -> None:
    reject_insert(conn, HandoffTicket, "ck_handoff_tickets_status_valid", status="CLOSED")
    reject_insert(
        conn,
        HandoffTicket,
        "ck_handoff_tickets_redacted_summary_length",
        redacted_summary="x" * 501,
    )
    reject_insert(
        conn, HandoffTicket, "ck_handoff_tickets_redacted_summary_not_blank", redacted_summary=" "
    )
    reject_insert(
        conn, HandoffTicket, "ck_handoff_tickets_resolution_matches_status", status="RESOLVED"
    )
    reject_insert(
        conn,
        HandoffTicket,
        "ck_handoff_tickets_resolution_matches_status",
        status="OPEN",
        resolved_at=BASE_TIME,
    )
    make(conn, HandoffTicket, redacted_summary="x" * 500)
    make(conn, HandoffTicket, status="RESOLVED", resolved_at=BASE_TIME)


def test_patient_specific_ticket_requires_an_existing_patient(conn: Connection) -> None:
    reject_insert(
        conn, HandoffTicket, "fk_handoff_tickets_patient_id_patients", patient_id=uuid.uuid4()
    )


def test_handoff_notes_are_attached_to_tickets_and_staff(conn: Connection) -> None:
    ticket = make(conn, HandoffTicket)
    staff = make(conn, StaffUser)
    note = make(conn, HandoffNote, ticket_id=ticket.ticket_id, staff_id=staff.staff_id)

    reject_insert(
        conn,
        HandoffNote,
        "uq_handoff_notes_request_id",
        ticket_id=ticket.ticket_id,
        staff_id=staff.staff_id,
        request_id=note.request_id,
    )
    reject(
        conn,
        "fk_handoff_notes_ticket_id_handoff_tickets",
        delete(HandoffTicket).where(HandoffTicket.ticket_id == ticket.ticket_id),
    )


def test_notification_status_read_timestamp_and_request_id(conn: Connection) -> None:
    patient = make(conn, Patient)
    base = {"patient_id": patient.patient_id}

    reject_insert(
        conn, NotificationEvent, "ck_notification_events_status_valid", status="SENT", **base
    )
    reject_insert(
        conn, NotificationEvent, "ck_notification_events_read_matches_status", status="READ", **base
    )
    notification = make(conn, NotificationEvent, **base)
    reject_insert(
        conn,
        NotificationEvent,
        "uq_notification_events_request_id",
        request_id=notification.request_id,
        **base,
    )
    reject_insert(
        conn,
        NotificationEvent,
        "fk_notification_events_offer_patient",
        offer_id=uuid.uuid4(),
        **base,
    )


def test_idempotency_scope_is_unique_per_actor_operation_and_key(conn: Connection) -> None:
    actor = uuid.uuid4()
    record = make(conn, IdempotencyRecord, actor_id=actor)

    reject_insert(
        conn,
        IdempotencyRecord,
        "uq_idempotency_records_scope_key",
        actor_id=actor,
        operation=record.operation,
        idempotency_key=record.idempotency_key,
    )
    make(conn, IdempotencyRecord, actor_id=uuid.uuid4(), idempotency_key=record.idempotency_key)
    make(
        conn,
        IdempotencyRecord,
        actor_id=actor,
        idempotency_key=record.idempotency_key,
        operation="cancel_appointment",
    )


def test_idempotency_completion_cannot_precede_creation(conn: Connection) -> None:
    reject_insert(
        conn,
        IdempotencyRecord,
        "ck_idempotency_records_completion_follows_creation",
        actor_id=uuid.uuid4(),
        created_at=BASE_TIME,
        completed_at=BASE_TIME.replace(year=2029),
    )


def test_outbox_dedupe_status_and_attempts(conn: Connection) -> None:
    event = make(conn, OutboxEvent, aggregate_id=uuid.uuid4())

    reject_insert(
        conn,
        OutboxEvent,
        "uq_outbox_events_dedupe_key",
        aggregate_id=uuid.uuid4(),
        dedupe_key=event.dedupe_key,
    )
    reject_insert(
        conn,
        OutboxEvent,
        "ck_outbox_events_status_valid",
        aggregate_id=uuid.uuid4(),
        status="FAILED",
    )
    reject_insert(
        conn,
        OutboxEvent,
        "ck_outbox_events_processed_matches_status",
        aggregate_id=uuid.uuid4(),
        status="PROCESSED",
    )
    reject_insert(
        conn,
        OutboxEvent,
        "ck_outbox_events_attempt_count_non_negative",
        aggregate_id=uuid.uuid4(),
        attempt_count=-1,
    )
    assert event.status == "PENDING"


def test_audit_events_allow_polymorphic_references_without_foreign_keys(
    conn: Connection,
) -> None:
    audit = make(
        conn,
        SecurityAuditEvent,
        actor_type="PATIENT",
        actor_id=uuid.uuid4(),
        resource_type="APPOINTMENT",
        resource_id=uuid.uuid4(),
    )

    assert audit.session_id is None
    reject_insert(
        conn,
        SecurityAuditEvent,
        "fk_security_audit_events_session_id_auth_sessions",
        session_id=uuid.uuid4(),
    )
