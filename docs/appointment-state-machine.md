# Appointment State Machine

**Status:** Approved  
**Scope:** Patient appointment lifecycle

## Purpose

This document defines the approved lifecycle of a patient appointment. PostgreSQL and backend services control appointment records. The AI agent may request approved tools but cannot create, update, or delete appointments directly.

## Appointment States

| State | Meaning |
|---|---|
| `CONFIRMED` | The appointment is committed and owns one `BOOKED` slot. |
| `CANCELLED` | The appointment was cancelled and remains as history. |

V1 uses only `CONFIRMED` and `CANCELLED`. `COMPLETED` and `NO_SHOW` are excluded from v1.

## Pending Confirmation and Rescheduling

- Pending UI confirmation is not an appointment state and does not create an appointment record.
- A pending confirmation expires 10 minutes after it is created.
- A selected slot remains `AVAILABLE` until a booking transaction commits.
- `PENDING` is represented by the pending action shown to the patient, not by an appointment status.
- After confirmation expires, the patient must review the current details and confirm a new pending action.
- The backend always rechecks authorization and availability before commit.
- V1 requires confirmation through the explicit UI control. Voice may prepare an action but cannot authorize execution.
- `RESCHEDULED` is an action and immutable historical event, not an appointment status.
- A successful reschedule updates the existing appointment record and leaves it `CONFIRMED`.

## State Transitions

The absence of a record is shown as `NONE` for clarity and is not a stored appointment state.

| From | To | Trigger | Slot effect |
|---|---|---|---|
| `NONE` | `CONFIRMED` | Verified patient confirms a direct booking | `AVAILABLE` to `BOOKED` |
| `NONE` | `CONFIRMED` | Intended patient accepts a valid waitlist offer | `HELD` to `BOOKED` |
| `CONFIRMED` | `CONFIRMED` | Atomic rescheduling action | Old slot `BOOKED` to `AVAILABLE`; new slot `AVAILABLE` to `BOOKED` |
| `CONFIRMED` | `CANCELLED` | Verified patient confirms cancellation | `BOOKED` to `AVAILABLE` |

Booking, rescheduling, and cancellation succeed only after their database transactions commit.

## Valid and Invalid Examples

Valid examples:

- `NONE` to `CONFIRMED` after verification, explicit confirmation, and a successful commit.
- `CONFIRMED` to `CONFIRMED` when an atomic reschedule moves the appointment to another available slot.
- `CONFIRMED` to `CANCELLED` while preserving the appointment record.

Invalid examples:

- Creating `CONFIRMED` from a pending UI confirmation without committing a booking transaction.
- Committing an action from a confirmation that is more than 10 minutes old.
- Creating an appointment for an `AVAILABLE` slot without rechecking authorization and slot state.
- Cancelling, rescheduling, or viewing another patient's appointment based on a model-provided patient ID.
- Creating a second active appointment for a slot that already has an active booking.
- Performing a patient self-service cancellation or reschedule at or after the appointment start time.
- Restoring a `CANCELLED` appointment instead of creating a new booking.

## Creation by Direct Booking

1. The patient verifies through the private authentication flow.
2. The UI displays the exact doctor, clinic, date, and time for confirmation.
3. Pending confirmation expires after 10 minutes, creates no appointment, and reserves no slot.
4. After confirmation, the backend locks the slot and rechecks the confirmation age, session, authorization, idempotency key, and `AVAILABLE` state.
5. One transaction creates a `CONFIRMED` appointment and changes the slot to `BOOKED`.
6. The assistant reports success only after commit.

## Creation from a Waitlist Offer

1. The backend locks the `HELD` slot and its active offer.
2. It verifies the intended patient, session, offer ownership, idempotency key, and matching hold.
3. Acceptance is valid only when PostgreSQL database time is strictly before `expires_at`.
4. One transaction creates a `CONFIRMED` appointment, marks the offer accepted, and changes the slot from `HELD` to `BOOKED`.
5. A late, invalid, or unauthorized acceptance creates no appointment.

## Rescheduling

- The existing appointment must be `CONFIRMED`, owned by the verified patient, and explicitly confirmed for rescheduling.
- The pending rescheduling confirmation must be no more than 10 minutes old.
- The backend locks the old and new slots in a consistent order.
- The new slot must still be `AVAILABLE` after locking.
- One transaction updates the existing appointment record, changes the old slot from `BOOKED` to `AVAILABLE`, changes the new slot from `AVAILABLE` to `BOOKED`, appends immutable rescheduling history, and records a reliable `SLOT_RELEASED` event after releasing the old slot.
- Waitlist processing consumes `SLOT_RELEASED` only after commit.
- If any check or write fails, the full transaction rolls back and the original appointment remains unchanged.

## Cancellation and History

- Cancellation requires a verified session, ownership, and explicit confirmation.
- The pending cancellation confirmation must be no more than 10 minutes old.
- The appointment row is preserved and changes from `CONFIRMED` to `CANCELLED`.
- The same transaction changes the slot from `BOOKED` to `AVAILABLE` and records a reliable `SLOT_RELEASED` event.
- Historical cancelled appointments do not count as active bookings for slot uniqueness.
- Waitlist processing consumes the release event only after commit.
- The assistant does not report cancellation until the transaction commits.

## Self-Service Cutoff

- Patients may cancel or reschedule only before the scheduled appointment start time.
- At or after the start time, the backend rejects patient self-service changes and commits no appointment or slot changes.
- The appointment start time is interpreted using the clinic's configured timezone.
- The cutoff is rechecked before commit.
- After rejection, the assistant offers simulated staff assistance without claiming that the appointment was changed.

## Cancelled Appointment Restoration

A `CANCELLED` appointment cannot be restored directly. The patient must create a new booking, and the original cancelled appointment remains unchanged as history.

## Immutable Appointment History

Every booking, rescheduling, and cancellation appends a history record with these fields:

- `event_id`
- `appointment_id`
- `action`: `BOOKED`, `RESCHEDULED`, or `CANCELLED`
- `actor_id`
- `actor_type`: `PATIENT`, `STAFF`, or `SYSTEM`
- `old_slot_id`
- `new_slot_id`
- `timestamp` in UTC
- `request_id`

History is append-only and is written in the same transaction as the appointment change. History never stores PINs, authentication tokens, or sensitive conversation content.

## Authorization and Idempotency

- Patient identity comes from the server-managed session, never from model arguments.
- Every protected action checks appointment ownership.
- Every write requires the approved confirmation boundary.
- Every mutation uses an idempotency key. A duplicate retry returns the original result without creating another appointment or event.
- Database uniqueness enforcement allows at most one active booking per slot while retaining cancelled appointment history.
- Every mutation appends the approved immutable, sanitized history record.
- Authenticated demo staff may view clinic schedules and limited fictional patient information but may not book, reschedule, or cancel patient appointments in v1.
- Staff authentication remains separate from patient authentication.
- Voice-only confirmation is deferred to a future version and requires a security evaluation.

## Transaction Boundaries

- Transactions use PostgreSQL `READ COMMITTED` isolation with row-level locking and unique constraints.
- Direct booking commits the appointment and `AVAILABLE` to `BOOKED` slot transition together.
- Waitlist acceptance commits the appointment, offer acceptance, and `HELD` to `BOOKED` transition together.
- Rescheduling commits the existing appointment update, both slot changes, immutable history, and `SLOT_RELEASED` event together.
- Cancellation commits the `CANCELLED` status, slot release, history, and `SLOT_RELEASED` event together.
- Confirmation age, authorization, ownership, availability, and the self-service cutoff are rechecked before commit.
- Validation or database failure rolls back every change in the transaction.

## Relationship to Slot States

- A `CONFIRMED` appointment must reference one `BOOKED` slot.
- A `BOOKED` slot may have at most one active appointment.
- A `CANCELLED` appointment remains in history but no longer owns an active slot.
- A successful reschedule transfers the active appointment from the old `BOOKED` slot to the new slot atomically.
- An accepted waitlist offer creates the same `CONFIRMED` appointment type as a direct booking.

## Failure and Concurrency Scenarios

| Scenario | Expected database outcome |
|---|---|
| Session expires before commit | No appointment or slot changes commit. |
| Pending confirmation is older than 10 minutes | Reject the action. The patient must review current details and confirm again. |
| Slot is taken after selection but before confirmation | Booking fails. No appointment is created. |
| Two patients book the same slot | One transaction may commit. The other fails without creating an active appointment. |
| Duplicate booking request uses the same idempotency key | Return the original result. Do not create a second appointment. |
| New slot becomes unavailable during rescheduling | Roll back all changes. The original appointment and old slot remain unchanged. |
| Patient cancels or reschedules at or after the clinic-local start time | Reject self-service. Preserve all appointment and slot state and offer simulated staff assistance. |
| Cancellation and rescheduling race | One transaction commits first. The other rechecks current state and fails safely. |
| Cancellation is retried | Return the original cancellation result. Do not emit another release event. |
| Waitlist offer expires before acceptance | Create no appointment. Reconcile the hold through the approved offer workflow. |
| Wrong patient attempts a mutation | Reject the request. No protected data or state changes are committed. |
| Database failure occurs during a mutation | Roll back appointment, slot, history, event, and idempotency changes. |
| `SLOT_RELEASED` processing is delayed | Keep the committed event available for idempotent retry. The appointment transaction remains committed. |
