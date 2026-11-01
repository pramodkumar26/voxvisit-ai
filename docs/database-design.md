# PostgreSQL Database Design

**Status:** Approved  
**Scope:** Authoritative application persistence

## Purpose

This document defines the approved PostgreSQL design for the VoxVisit AI workflows. It contains synthetic data only. PostgreSQL is authoritative for authentication, permissions, pending actions, appointments, slots, waitlists, offers, handoffs, idempotency, and retryable background work.

## PostgreSQL Hosting and Connections

- Development and automated integration tests use local PostgreSQL through Docker.
- The preferred hosted database for the public demonstration is Neon serverless PostgreSQL. The FastAPI backend remains on GCP Cloud Run.
- All hosted connections use TLS and environment-managed credentials. Credentials are never committed or exposed to the AI model.
- Application traffic uses pooled connections sized for Cloud Run concurrency and provider limits.
- Schema migrations use a direct database connection because transaction pooling may not support every migration operation safely.
- Development and hosted environments should use the same supported PostgreSQL major version where practical.
- Provider version, required extensions, connection limits, pooling behavior, and compatibility are verified during implementation before the public demo is enabled.

## Design Principles

- Use PostgreSQL `READ COMMITTED`, row-level locks, consistent lock ordering, and database constraints.
- Store timestamps as `timestamptz` in UTC. Store clinic timezone as an IANA timezone name and apply clinic-local rules when matching or displaying schedules.
- Use UUID primary keys and explicit foreign keys.
- Use `text` status columns with check constraints containing only states approved in the corresponding state-machine documents. PostgreSQL enum types are not used in v1.
- Add a `version bigint` to mutable records used by pending actions for stale-data detection.
- Store credentials and opaque session-token hashes only. Never expose them to the AI model.
- Use `jsonb` only for sanitized pending-action summaries and expected versions, in-app notification payloads, transactional outbox payloads, sanitized audit details, and sanitized idempotency response snapshots. Validate each payload structure in the backend.
- Record retryable work in `outbox_events` inside the originating transaction.
- Enforce cross-table consistency in backend transactions without database triggers in v1.
- Use explicit foreign keys for known relationships and approved polymorphic references only where the target type varies.

## Data Types and Conventions

- IDs: `uuid`.
- Instants: `timestamptz`, stored in UTC.
- Local recurring schedules: `date`, `time`, and clinic timezone.
- Status and action values: `text` with check constraints containing only values approved in this document or the corresponding state-machine document.
- Flexible payloads: approved sanitized `jsonb` fields only. Never store credentials, tokens, medical records, or sensitive conversation transcripts in them.
- Counters and versions: `integer` or `bigint` with non-negative checks.
- User-facing text: bounded `varchar` where practical; sanitized longer content as `text`.
- Every mutable table includes `created_at`; most include `updated_at` and `version`.

## Core Scheduling Tables

### `clinics`

- Purpose: Fictional clinic identity and timezone.
- Columns: `clinic_id uuid`, `name varchar`, `timezone varchar`, synthetic address fields, `active boolean`, timestamps.
- Keys and constraints: primary key `clinic_id`; unique clinic name; timezone must be a configured IANA value.
- Indexes: `active`; normalized name for lookup.

### `clinic_operating_hours`

- Purpose: Recurring clinic-local operating hours.
- Columns: `hours_id uuid`, `clinic_id uuid`, `day_of_week smallint`, `opens_at time`, `closes_at time`.
- Keys and relationships: primary key `hours_id`; `clinic_id` references `clinics`.
- Constraints: day 0 through 6; open before close; unique clinic and weekday window.
- Indexes: `clinic_id, day_of_week`.

### `doctors`

- Purpose: Fictional doctor directory.
- Columns: `doctor_id uuid`, `display_name varchar`, `specialty varchar`, `active boolean`, timestamps.
- Keys and constraints: primary key `doctor_id`; non-empty display name and specialty.
- Indexes: `specialty, active`; normalized display name.

### `doctor_clinics`

- Purpose: Many-to-many assignment of doctors to clinics.
- Columns: `doctor_id uuid`, `clinic_id uuid`, `active boolean`.
- Keys and constraints: composite primary key; foreign keys to `doctors` and `clinics`.
- Indexes: reverse index on `clinic_id, doctor_id`.

### `doctor_availability_rules`

- Purpose: Recurring doctor working hours at a clinic.
- Columns: `availability_id uuid`, doctor and clinic foreign keys, weekday, local start and end times, effective start and end dates, `active boolean`.
- Keys and relationships: primary key `availability_id`; `doctor_id` references `doctors`; `clinic_id` references `clinics`; the doctor and clinic pair references an active `doctor_clinics` assignment.
- Constraints: weekday 0 through 6; start before end; effective end not before start; doctor-clinic assignment must exist.
- Indexes: doctor, clinic, weekday, and effective dates.

### `clinic_holidays`

- Purpose: Clinic closures or reduced hours.
- Columns: `holiday_id uuid`, `clinic_id uuid`, `local_date date`, optional local start and end, sanitized label.
- Keys and relationships: primary key `holiday_id`; `clinic_id` references `clinics`.
- Constraints: unique clinic, date, and time window; both times null for full-day closure or both present with start before end.
- Indexes: `clinic_id, local_date`.

### `doctor_leave`

- Purpose: Doctor-specific leave independent of recurring availability.
- Columns: `leave_id uuid`, `doctor_id uuid`, `clinic_id uuid`, `starts_at timestamptz`, `ends_at timestamptz`, sanitized reason code.
- Keys and relationships: primary key `leave_id`; doctor and clinic foreign keys reference `doctors` and `clinics`; the pair references `doctor_clinics`.
- Constraints: end after start; doctor-clinic assignment must exist.
- Indexes: doctor and time range; clinic and time range.

### `slots`

- Purpose: Materialized appointment inventory and authoritative slot state.
- Columns: `slot_id uuid`, doctor and clinic foreign keys, `starts_at timestamptz`, `ends_at timestamptz`, `status text`, `version bigint`, optional `removed_at`, timestamps.
- Keys and relationships: primary key `slot_id`; doctor and clinic foreign keys reference `doctors` and `clinics`; the pair references `doctor_clinics`.
- Constraints: status is `AVAILABLE`, `HELD`, or `BOOKED`; end after start; unique doctor and start time; `removed_at` may be set only while status is `AVAILABLE`. A removed slot cannot transition to `HELD` or `BOOKED` because that would violate the removal check.
- Overlap prevention: a `btree_gist` exclusion constraint prevents two slots for the same doctor from having overlapping `tstzrange(starts_at, ends_at, '[)')` ranges while `removed_at` is null. Adjacent slots are allowed. The rule applies to every slot status and across clinics, because a doctor cannot be in two places. Soft-removed slots are excluded from the constraint and remain stored.
- Indexes: clinic and start time; doctor and start time; `status, starts_at`; partial index for future slots where status is `AVAILABLE` and `removed_at` is null.
- Removal: staff may soft-remove only `AVAILABLE` slots. Availability searches exclude rows with `removed_at` set. No additional slot status is introduced.
- History: confirmed appointments are never deleted automatically when availability, holiday, or leave changes.

## Identity and Authentication Tables

### `patients`

- Purpose: Synthetic patient identity and staff-visible fields.
- Columns: `patient_id uuid`, unique `patient_code varchar`, `display_name varchar`, `active boolean`, timestamps.
- Keys and relationships: primary key `patient_id`; referenced by patient credentials, sessions, appointments, waitlists, notifications, and patient-specific handoffs.
- Constraints: only synthetic data; staff projection exposes patient code and display name only.
- Indexes: unique patient code; normalized display name for permitted staff search.

### `staff_users`

- Purpose: Demo staff identity.
- Columns: `staff_id uuid`, unique `username varchar`, `display_name varchar`, `active boolean`, timestamps.
- Keys and relationships: primary key `staff_id`; referenced by staff credentials, sessions, handoff notes, and sanitized audit events.
- Indexes: unique normalized username; `active`.

### `patient_credentials`

- Purpose: Patient PIN verification and lockout state.
- Columns: `patient_id uuid`, `pin_hash text`, failed-attempt count and window, `locked_until timestamptz`, `disabled boolean`, `updated_at`.
- Keys and constraints: patient ID is primary key and foreign key; six-digit format is enforced before hashing; counters are non-negative.
- Security: Argon2id hash only. No plaintext PIN or recoverable secret.

### `staff_credentials`

- Purpose: Staff password verification, lockout, and administrative reset state.
- Columns: `staff_id uuid`, `password_hash text`, failed-attempt fields, `locked_until`, `reset_required boolean`, `updated_at`.
- Keys and constraints: staff ID is primary key and foreign key; passwords require at least 12 characters and reject approved common or compromised values before hashing.
- Security: Argon2id hash only. No automated email or SMS recovery.

### `auth_sessions`

- Purpose: Opaque server-managed patient and staff sessions.
- Columns: `session_id uuid`, `actor_type text`, `role text`, nullable patient and staff foreign keys, unique `token_hash bytea`, `status text`, creation and activity times, idle and absolute deadlines, termination time and reason.
- Keys and relationships: primary key `session_id`; `patient_id` references `patients`; `staff_id` references `staff_users`; pending actions and optional historical records may reference the session.
- Constraints: actor is exactly one patient or staff record; role must match the actor type; status is `ACTIVE`, `EXPIRED`, `LOGGED_OUT`, or `REVOKED`; deadlines follow creation time.
- Indexes: unique token hash; actor and active status; expiration deadlines. A partial unique index permits one active staff session.
- Concurrency: patient login locks the patient credential row, creates the new session, and when three are active revokes the oldest active session atomically.
- Retention: expired records are removed seven days after expiration. Terminal sessions never regain access.

### `pending_actions`

- Purpose: Server-side UI confirmation state, separate from appointment status.
- Columns: `pending_action_id uuid`, `session_id uuid`, `action_type text`, `target_type text`, `target_id uuid`, sanitized `summary jsonb`, `expected_versions jsonb`, `status text`, `expires_at timestamptz`, `request_id uuid`, timestamps.
- Keys and relationships: primary key `pending_action_id`; `session_id` references `auth_sessions` with deletion cascading only when the expired session row is cleaned up; target references are polymorphic and validated by the backend.
- Constraints: status is `ACTIVE`, `CONSUMED`, `INVALIDATED`, or `EXPIRED`; active actions expire after 10 minutes; session ownership is mandatory.
- Indexes: session and active status; expiration; unique request ID.
- Security: no credentials, tokens, PINs, or sensitive conversation content.

## Appointment Tables

### `appointments`

- Purpose: Current and historical patient appointments.
- Columns: `appointment_id uuid`, `patient_id uuid`, `slot_id uuid`, `status text`, `version bigint`, cancellation timestamp, timestamps.
- Keys and relationships: primary key `appointment_id`; `patient_id` references `patients`; `slot_id` references `slots`; history events reference the appointment.
- Constraints: status is `CONFIRMED` or `CANCELLED`; a partial unique constraint allows only one `CONFIRMED` appointment per slot.
- Indexes: patient and status; slot. Time-based searches use the indexed slot start time.
- History: cancelled rows are preserved and cannot be restored directly.

### `appointment_history`

- Purpose: Immutable event history for every appointment mutation.
- Columns: `event_id uuid`, `appointment_id uuid`, `action text`, `actor_id uuid`, `actor_type text`, nullable old and new slot foreign keys, `timestamp timestamptz`, `request_id uuid`.
- Keys and relationships: primary key `event_id`; appointment and slot IDs reference `appointments` and `slots`; `actor_id` is polymorphic across patient, staff, and system actors.
- Constraints: action is `BOOKED`, `RESCHEDULED`, or `CANCELLED`; actor type is `PATIENT`, `STAFF`, or `SYSTEM`; unique appointment and request ID.
- Indexes: appointment and timestamp; request ID; actor and timestamp.
- Immutability: application database roles may insert and read but not update or delete history rows. This is enforced through restricted database-role grants, not triggers. The local development and test database uses an owner role that is not restricted, so the guarantee must be configured and verified before any public deployment.
- Retention: preserved with the appointment history. Session cleanup never deletes these events.
- Security: no PINs, tokens, or sensitive conversation content.

## Waitlist and Offer Tables

### `waitlist_entries`

- Purpose: Patient matching preferences and FIFO priority.
- Columns: `waitlist_id uuid`, patient and clinic foreign keys, `specialty varchar`, inclusive start and end dates, `time_preference text`, `status text`, `version bigint`, timestamps.
- Keys and relationships: primary key `waitlist_id`; `patient_id` references `patients`; `clinic_id` references `clinics`; offers reference the waitlist entry.
- Constraints: status is `ACTIVE`, `FULFILLED`, or `CANCELLED`; preference is `ANYTIME`, `MORNING`, or `AFTERNOON`; end date is not before start date.
- Uniqueness: a partial exclusion constraint uses inclusive date ranges and `btree_gist` to prohibit overlapping `ACTIVE` entries for the same patient, clinic, and specialty, regardless of time preference. A unique constraint on `waitlist_id, patient_id` supports offer ownership enforcement. Terminal entries remain stored and do not participate in the exclusion constraint.
- Indexes: status, clinic, specialty, dates, `created_at`, `waitlist_id`; patient and status.
- Priority: order eligible entries by `created_at`, then `waitlist_id`.

### `slot_offers`

- Purpose: Exclusive waitlist hold for one patient and slot.
- Columns: `offer_id uuid`, `waitlist_id uuid`, `slot_id uuid`, `patient_id uuid`, `status text`, `created_at timestamptz`, `expires_at timestamptz`, `responded_at timestamptz`, `version bigint`, `request_id uuid`.
- Keys and relationships: primary key `offer_id`; `slot_id` references `slots`; `patient_id` references `patients`; the composite `waitlist_id, patient_id` foreign key references the same pair on `waitlist_entries`; a unique `offer_id, patient_id` pair supports notification ownership enforcement; optional notifications may reference the offer.
- Constraints: status is `ACTIVE`, `ACCEPTED`, `DECLINED`, or `EXPIRED`; expiry follows creation; the composite foreign key enforces that the offer patient owns the waitlist entry.
- Uniqueness: partial unique constraints allow one `ACTIVE` offer per slot and per patient; unique patient and slot prevents reoffering the same slot after decline or expiration.
- Indexes: active expiration; waitlist and created time; patient and status; slot and status.
- Time rule: final acceptance checks `clock_timestamp() < expires_at` after locks are acquired.
- History: terminal offers are preserved.

## Handoff and Notification Tables

### `handoff_tickets`

- Purpose: Simulated general or patient-specific staff assistance.
- Columns: `ticket_id uuid`, optional `patient_id uuid`, optional creator session, required `category text`, `status text`, `redacted_summary text`, `version bigint`, created, updated, and resolved timestamps.
- Keys and relationships: primary key `ticket_id`; optional `patient_id` references `patients`; optional `created_by_session_id` references `auth_sessions` with `ON DELETE SET NULL`; notes reference the ticket.
- Constraints: status is `OPEN`, `IN_PROGRESS`, or `RESOLVED`; category is `SCHEDULING`, `CLINIC_INFORMATION`, `ACCESSIBILITY`, `TECHNICAL_SUPPORT`, or `OTHER_ADMINISTRATIVE` with no database default, so the creating service must supply it explicitly; patient-specific tickets require patient ownership; summaries are sanitized. A ticket is general when `patient_id` is null and patient-specific when it is set. Tickets that predate categories were backfilled once with `OTHER_ADMINISTRATIVE`.
- Indexes: status and created time; patient and created time; resolved time for retention cleanup.
- Retention: unresolved tickets remain until resolved. Resolved tickets and their notes are retained for 30 days after resolution.

### `handoff_notes`

- Purpose: Sanitized staff notes attached to a handoff.
- Columns: `note_id uuid`, `ticket_id uuid`, `staff_id uuid`, `note_text text`, `created_at timestamptz`, `request_id uuid`.
- Keys and relationships: primary key `note_id`; `ticket_id` references `handoff_tickets`; `staff_id` references `staff_users`.
- Constraints: unique request ID; append-only notes.
- Security: no authentication secrets, medical records, or sensitive conversation content.
- Indexes: ticket and created time.
- Retention: follows the owning handoff ticket and is deleted with it after the approved retention period.

### `notification_events`

- Purpose: In-app notifications for waitlist offers and other approved demo events.
- Columns: `notification_id uuid`, `patient_id uuid`, optional `offer_id uuid`, `notification_type text`, `status text`, sanitized `payload jsonb`, created and read timestamps, `request_id uuid`.
- Keys and relationships: primary key `notification_id`; `patient_id` references `patients`; optional `offer_id` references `slot_offers`. When `offer_id` is present, the composite `offer_id, patient_id` foreign key requires the notification patient to own the offer. Notifications without an offer are unaffected.
- Constraints: status is `UNREAD` or `READ`; delivery channel is in-app only; payload contains no credentials or sensitive conversation content.
- Indexes: patient, status, created time; offer ID; unique request ID; created time for retention cleanup.
- Retention: retained for 30 days after creation.

## Reliability and Audit Tables

### `idempotency_records`

- Purpose: Return the original outcome for repeated state-changing requests.
- Columns: `idempotency_id uuid`, `actor_type text`, `actor_id uuid`, optional `session_id uuid`, `operation text`, `idempotency_key varchar`, `request_hash bytea`, `status text`, `response_code`, `result_type`, `result_id`, optional `response_snapshot jsonb`, `created_at timestamptz`, `completed_at timestamptz`.
- Keys and relationships: primary key `idempotency_id`; optional `session_id` references `auth_sessions` with `ON DELETE SET NULL`; actor reference is polymorphic.
- Constraints: `actor_type` is `PATIENT`, `STAFF`, or `ANONYMOUS`; `actor_id` is required; unique actor type, actor ID, operation, and key; `response_snapshot` is either SQL `NULL` or a JSON object. The same key with a different request hash is rejected by the service.
- Anonymous scope: public handoff requests have no account, so they use `actor_type = ANONYMOUS`. The backend must generate the `actor_id` as a high-entropy random identifier and keep it stable across retries of the same request. It is never accepted from the model or the client as an identity claim, grants no patient or staff permission, and is isolated from every account scope by `actor_type`. How the scope is carried between retries (session or cookie handling) and per-source rate limiting are deferred to the authentication and public API work.
- Response snapshot: lets the future idempotency service return the original successful response after a retry. It stores minimal approved response fields only, such as identifiers, status, and display values already visible to the caller. It never stores session tokens, PINs, passwords, credential hashes, clinical records, or unrestricted request bodies. The database enforces only that the value is an object. The service must validate the allowed fields before writing, bound the size, and return a snapshot only after confirming the request hash matches.
- Indexes: unique lookup key; completion time for retention cleanup.
- Retention: completed records are retained for 30 days after completion.

### `outbox_events`

- Purpose: Durable post-commit processing such as `SLOT_RELEASED`, next-patient matching, and notification delivery.
- Columns: `outbox_event_id uuid`, `event_type text`, aggregate type and ID, sanitized `payload jsonb`, `status text`, `dedupe_key varchar`, attempt count, `available_at`, created, processed, and last-attempt timestamps, sanitized last error.
- Keys and relationships: primary key `outbox_event_id`; aggregate references are polymorphic and are not cascading foreign keys.
- Constraints: unique dedupe key; non-negative attempts; status is `PENDING`, `PROCESSING`, or `PROCESSED`.
- Indexes: status and available time; aggregate; created time; processed time for retention cleanup.
- Reliability: the originating transaction inserts the event. Workers claim and retry it idempotently after commit.
- Retention: successfully processed events are retained for 14 days after processing. `PENDING`, `PROCESSING`, failed, or otherwise retryable work is never removed by retention cleanup.

### `security_audit_events`

- Purpose: Sanitized authentication, authorization, protected-read, and mutation audit trail.
- Columns: `audit_event_id uuid`, event type, optional actor and session references, resource type and ID, result, UTC timestamp, request ID, sanitized network metadata and details.
- Keys and relationships: primary key `audit_event_id`; optional `session_id` references `auth_sessions` with `ON DELETE SET NULL`; actor and resource references are polymorphic.
- Constraints: no credentials, token values, hashes, PINs, or sensitive conversation content.
- Indexes: actor and timestamp; event type and timestamp; request ID; timestamp for retention cleanup.
- Retention: authentication and non-authentication operational events are retained for 30 days.

## Database Constraints and Indexes

- Partial uniqueness enforces one `CONFIRMED` appointment per slot.
- A `btree_gist` exclusion constraint prevents overlapping non-removed slot ranges for the same doctor.
- The composite `offer_id, patient_id` foreign key prevents a notification from naming a patient who does not own the referenced offer.
- Partial uniqueness enforces one `ACTIVE` offer per slot and per patient.
- Patient and slot uniqueness on offers prevents reoffering a declined or missed slot to the same patient.
- A partial `btree_gist` exclusion constraint prevents overlapping inclusive date ranges for `ACTIVE` waitlists by patient, clinic, and specialty, regardless of time preference.
- Status checks preserve approved state sets while terminal history remains stored.
- The composite waitlist and patient foreign key prevents an offer from naming a patient who does not own the waitlist entry.
- The slot removal check prevents a row with `removed_at` set from becoming `HELD` or `BOOKED`. Booking and offer transactions also lock the slot and require status `AVAILABLE` with `removed_at` null.
- Foreign keys use restrictive deletion for appointments, offers, histories, credentials, and other preserved records. Synthetic parent records are deactivated instead of deleting referenced history.
- Write-path indexes support session-token lookup, patient ownership, available-slot search, FIFO waitlist selection, offer expiry, outbox polling, and audit retention.
- Cross-table parity, such as `CONFIRMED` with `BOOKED`, is enforced by backend transactions, row locks, consistent lock ordering, foreign keys, uniqueness and check constraints, and reconciliation tests. V1 does not use database triggers for this purpose.

## Foreign-Key Deletion Behavior

- Historical and authoritative records use restrictive deletion. Patients, staff, slots, appointments, waitlists, offers, and other referenced synthetic parent records are deactivated or soft-removed instead of deleted.
- `pending_actions.session_id` uses cascading deletion because pending actions are session-bound and unusable after the session record reaches cleanup.
- Optional session references on handoffs, idempotency records, and security audit events use `ON DELETE SET NULL`. Their actor, request, and resource metadata remains available after the expired session row is removed.
- Session cleanup never cascades into appointment history, handoff tickets or notes, idempotency records, notifications, outbox work, or audit events.
- Cleanup jobs apply only the approved retention rules and never delete retryable outbox work.

## Authorization Boundaries

- Patient queries always derive `patient_id` from the verified session and filter owned appointments, waitlists, offers, pending actions, notifications, and patient-specific handoffs.
- Staff queries expose only patient code, display name, appointment doctor, date, time, status, and relevant sanitized handoff information.
- Staff cannot read credential tables, token hashes, CSRF material, medical records, or unrelated personal information.
- Staff cannot read patient waitlist entries or offers and cannot mutate appointments, waitlists, or offers in v1.
- The AI tool layer never receives credential records, token hashes, or raw security audit details.

## Canonical Lock-Acquisition Order

Conflicting write operations acquire row locks in this order and skip table types they do not need:

1. Credential or account-control row, for login and administrative revocation only.
2. `auth_sessions`.
3. `pending_actions`.
4. `appointments`.
5. `waitlist_entries`.
6. `slot_offers`.
7. `slots`.
8. `handoff_tickets`.

Within one table, multiple rows are locked by primary key in ascending order. Idempotency and claimed outbox work are resolved before the ranked business-row locks. New history, audit, idempotency-result, notification, and outbox rows are inserted after validation and do not introduce a reverse lock dependency.

If a workflow must discover identifiers in a different order, it performs an unlocked lookup first, acquires the required locks in canonical order, and then rechecks every relevant state and relationship. No transaction may acquire an earlier-ranked lock after holding a later-ranked lock.

## Transaction Boundaries

All state-changing transactions first resolve idempotency and recheck the active session, authorization, ownership, pending confirmation, and current database state.

### Direct appointment booking

Following the canonical order, lock the session, pending action, and slot. Recheck the 10-minute confirmation, ownership, status `AVAILABLE`, and `removed_at` is null. In one transaction set the slot `BOOKED`, create the `CONFIRMED` appointment, append `BOOKED` history, consume the pending action, and store the idempotent result. The slot removal check and confirmed-appointment uniqueness constraint provide final database enforcement.

### Appointment rescheduling

Following the canonical order, lock the session, pending action, appointment, and old and new slots. Lock both slots by ascending `slot_id`. Recheck ownership, clinic-local cutoff, confirmation, and that the new slot is `AVAILABLE` and not removed. Update the existing appointment, set old slot `AVAILABLE`, set new slot `BOOKED`, append `RESCHEDULED` history, insert `SLOT_RELEASED` outbox work, consume confirmation, and store the result. Any failure preserves the original appointment.

### Appointment cancellation

Following the canonical order, lock the session, pending action, appointment, and slot. Recheck ownership, cutoff, and confirmation. Set appointment `CANCELLED`, set slot `AVAILABLE`, append `CANCELLED` history, insert `SLOT_RELEASED`, consume confirmation, and store the result atomically.

### Waitlist offer creation

Claim the committed `SLOT_RELEASED` event and identify the candidate waitlist entry without locking business rows. Following the canonical order, lock the selected waitlist entry and then the slot. Recheck FIFO eligibility, the one-offer-per-patient rule, slot status `AVAILABLE`, and `removed_at` is null. After locking, use PostgreSQL `clock_timestamp()` to set offer creation and expiration from the globally configured duration. Atomically set the slot `HELD`, create the owned `ACTIVE` offer and in-app notification work, and mark or advance outbox processing. Duplicate delivery creates no duplicate hold.

### Waitlist offer acceptance

Following the canonical order, lock the session, pending action, waitlist entry, offer, and slot. Recheck session and offer ownership, all related states, and the matching patient relationship. After every required lock is held, evaluate `clock_timestamp() < expires_at`. Atomically set the offer `ACCEPTED`, entry `FULFILLED`, and slot `BOOKED`; create the `CONFIRMED` appointment and `BOOKED` history; consume confirmation; and save the idempotent result.

### Offer decline and expiration

For a patient decline, first lock the session and pending action. For either decline or worker-driven expiration, then lock the waitlist entry, offer, and slot in canonical order. Recheck ownership and state. For expiration, evaluate `clock_timestamp()` only after every required lock is held. Set the terminal offer state, release the slot to `AVAILABLE`, preserve the `ACTIVE` waitlist entry, and insert retryable next-patient outbox work in the same transaction.

### Waitlist cancellation with an active offer

Following the canonical order, lock the session, pending action, waitlist entry, active offer, and slot. Recheck ownership, UI confirmation, and all related states. Set the entry `CANCELLED`, offer `DECLINED`, and slot `AVAILABLE`; insert retryable next-patient work; and store the idempotent result atomically. This order matches offer acceptance and prevents an offer-waitlist lock inversion.

### Session creation, expiration, and revocation

Lock the credential or actor row while applying login counters and session limits. Session creation stores only a token hash. A fourth patient login creates the new session, revokes the oldest active session, and invalidates its pending actions in one transaction. For staff login, lock the staff account and current active session, mark the previous session `REVOKED`, invalidate its pending actions, and only then insert the replacement `ACTIVE` session in the same transaction. This ordering respects the partial unique index that permits one active staff session. Expiration is enforced logically on every request; logout, revocation, credential reset, and pending-action invalidation commit together.

### Handoff ticket updates

Following the canonical order, lock and validate the staff session and then lock the ticket. Update only an approved status, append any sanitized staff note, create the audit event, and store the idempotent result in one transaction.

## Data Retention

- Retention periods are configurable v1 defaults for synthetic demo data.
- Expired session rows are removed seven days after expiration. Deleted or terminal sessions cannot regain access.
- Sanitized authentication security audit events are retained for 30 days.
- Resolved handoff tickets and their notes are retained for 30 days after resolution. Unresolved handoffs remain until resolved.
- In-app notifications are retained for 30 days after creation.
- Completed idempotency records are retained for 30 days after completion.
- Successfully processed outbox events are retained for 14 days after processing. Unprocessed or retryable outbox work is never automatically deleted.
- Non-authentication operational audit events are retained for 30 days.
- Cancelled appointments, immutable appointment history, fulfilled or cancelled waitlists, and terminal offers remain preserved under their approved history rules.
- Retention cleanup never converts synthetic demo retention into a claim of production healthcare compliance.

## Implementation-Time Verification

These checks validate the approved design and do not reopen architecture decisions:

1. Verify the local and Neon PostgreSQL versions and select a compatible major version.
2. Verify Neon support and permissions for `btree_gist`. If unavailable, document and test an alternative with equivalent database-enforced concurrency safety before implementation proceeds.
3. Measure Neon connection limits, pooled application behavior, direct migration connectivity, TLS configuration, and Cloud Run pool sizing.
4. Test serverless suspend and resume behavior, transaction semantics, row locks, and `clock_timestamp()` expiration behavior.
5. Run concurrency and reconciliation tests for slot, appointment, waitlist, and offer state parity without triggers.
6. Test retention cleanup, foreign-key behavior, and proof that historical records survive session cleanup while retryable outbox work remains intact.
7. Before public deployment, create restricted application and migration database roles, grant the application role only `INSERT` and `SELECT` on `appointment_history`, and test that updates and deletes of history rows are denied.
