# Appointment Slot State Machine

**Status:** Approved  
**Scope:** Appointment slot state only

## Purpose

This document defines the approved appointment slot state machine. PostgreSQL is the source of truth. The backend, not the conversation agent, validates and commits every state change.

Appointment, waitlist-entry, slot-offer, and session state machines are outside this document and will be reviewed separately.

Slot status is stored directly in `slots`. Appointment details remain in `appointments`, and waitlist offer details remain in `slot_offers`. Transactions and database constraints keep these records consistent.

## Slot States

| State | Meaning |
|---|---|
| `AVAILABLE` | The slot can be booked. |
| `HELD` | The slot is temporarily reserved for one exclusive waitlist offer. The default hold is five minutes and is configurable. |
| `BOOKED` | The slot has one confirmed appointment. |

A pending booking confirmation does not reserve a slot. The slot remains `AVAILABLE` and may be booked by another valid transaction before confirmation is committed.

## Valid Transitions

| From | To | Trigger | Required checks |
|---|---|---|---|
| `AVAILABLE` | `BOOKED` | Patient confirms a booking | Valid session, ownership rules, explicit confirmation, idempotency, and current availability |
| `AVAILABLE` | `HELD` | Backend creates an exclusive waitlist offer | Slot is still available and no other active offer exists for it |
| `HELD` | `BOOKED` | Intended patient accepts during the configured hold | Valid session, offer ownership, database time strictly before `expires_at`, matching hold, and idempotency |
| `HELD` | `AVAILABLE` | Offer is declined, expires, or becomes invalid | Matching hold and offer are rechecked together |
| `BOOKED` | `AVAILABLE` | Confirmed appointment is cancelled | Valid session, appointment ownership, explicit confirmation, and matching appointment |

Each transition succeeds only after its database transaction commits. A duplicate request that returns a stored idempotent result does not perform another transition.

## Invalid Transitions

- `AVAILABLE` to `AVAILABLE` is not a state transition. Searches and pending confirmations do not change the slot.
- `HELD` to `HELD` cannot create or replace an offer. A held slot has at most one active exclusive offer.
- `BOOKED` to `BOOKED` cannot create another appointment for the same slot.
- `HELD` to `BOOKED` is invalid for the wrong patient, an expired offer, or a mismatched hold.
- `BOOKED` to `HELD` is not direct. The appointment must first be cancelled, producing `BOOKED` to `AVAILABLE`. A later waitlist action may produce `AVAILABLE` to `HELD`.
- `AVAILABLE` to `HELD` is invalid when the slot has already been booked or another active offer exists.
- Staff may remove only `AVAILABLE` slots. A `HELD` slot must follow the approved offer release or expiration workflow, and a `BOOKED` slot must follow the approved cancellation or rescheduling workflow.
- Any transition requested without required authorization or confirmation is rejected without changing state.

## Booking and Double-Booking Rules

- A slot may have at most one confirmed appointment.
- A `HELD` slot is unavailable for general booking and cannot be offered to another patient.
- Pending confirmation does not lock or reserve an `AVAILABLE` slot.
- At commit time, the backend rechecks the authenticated session, authorization, explicit confirmation, idempotency key, and current slot state.
- Database uniqueness enforcement allows at most one active booking and one active offer per slot. Historical cancelled appointments and expired or declined offers remain stored without violating these constraints.
- The assistant reports a booking only after the backend confirms a committed `BOOKED` state.

## Transaction and Concurrency Requirements

- Transactions use PostgreSQL `READ COMMITTED` isolation.
- The backend locks the affected slot row before validating and changing its state.
- The related appointment or offer record and the slot transition are written in the same transaction.
- A transaction rechecks current database state after acquiring the lock. Earlier search results are not trusted.
- Concurrent operations must have one winner. Later operations observe the committed state and fail safely.
- Booking, holding, accepting, releasing, cancelling, and rescheduling use idempotency keys for safe retries.
- Rescheduling locks both affected slots in a consistent order to reduce deadlock risk.
- Any validation or database failure rolls back the entire transaction.
- Concurrency tests must race two bookings, a booking against a hold, two offer acceptances, and rescheduling against another booking.

## Hold Expiration

- The default hold lasts five minutes and is configurable.
- PostgreSQL database time exclusively sets the hold creation time and calculates `expires_at`.
- PostgreSQL database time exclusively determines whether an offer is still valid. Browser and application-server clocks are never used for expiration checks.
- Acceptance is valid only when PostgreSQL database time is strictly earlier than `expires_at`.
- A retry does not extend the original expiration time.
- At or after expiration, the offer is logically expired even if a cleanup process has not run.
- A late acceptance is rejected and never revives the hold.
- A read-only request may report the offer as logically expired but does not change stored state.
- The next controlled expiration or protected write operation must reconcile `HELD` to `AVAILABLE` transactionally before further waitlist processing.
- If another eligible waitlist entry exists, a later valid transition may move the slot from `AVAILABLE` to `HELD` for a new offer.

## Rescheduling

Rescheduling changes two slots atomically:

1. The old slot starts as `BOOKED` for the patient's current appointment.
2. The new slot must still be `AVAILABLE` when the transaction obtains its locks.
3. The backend rechecks the session, ownership, explicit confirmation, and both slot states.
4. In one transaction, the old slot changes `BOOKED` to `AVAILABLE` and the new slot changes `AVAILABLE` to `BOOKED`.
5. The transaction records a reliable `SLOT_RELEASED` event for the old slot.
6. If any check or write fails, all changes roll back. The old slot remains `BOOKED` and the new slot remains unchanged.
7. Waitlist processing consumes the `SLOT_RELEASED` event only after the rescheduling transaction commits.

Selecting a new slot or displaying a pending reschedule does not reserve the new slot.

## Cancellation and Slot Release

- Cancellation changes the slot from `BOOKED` to `AVAILABLE` and records a reliable `SLOT_RELEASED` event in the same transaction.
- Waitlist processing consumes the event only after the cancellation commits.
- Event retries are idempotent and cannot create duplicate holds or offers.
- The event handler locks and rechecks the slot before changing `AVAILABLE` to `HELD`.

## Doctor Availability

- Doctor working hours, holidays, and leave are managed separately from slot status.
- Slots are generated only within periods when the doctor is available.
- A later schedule, holiday, or leave change never automatically deletes a confirmed appointment.
- A confirmed appointment remains `BOOKED` until an explicit approved workflow changes it.

## Failure Scenarios

| Scenario | Expected outcome |
|---|---|
| Session expires before commit | Reject the operation. No slot state changes. |
| Authorization or ownership check fails | Reject the operation and expose no protected data. No state changes. |
| Slot is booked after search but before confirmation | Booking fails as unavailable. Existing `BOOKED` state remains. |
| Two patients confirm the same available slot | One transaction may change it to `BOOKED`. The other fails without creating an appointment. |
| Booking and waitlist hold race for one available slot | The transaction that locks and commits first wins. The other rechecks the state and fails. |
| Two requests accept the same offer | At most one changes `HELD` to `BOOKED`. An idempotent retry returns the original result. |
| Intended patient accepts after expiration | Reject as expired. No appointment is created and the hold is reconciled toward `AVAILABLE`. |
| Cancellation is retried | Return the stored idempotent result. Do not release the slot twice. |
| `SLOT_RELEASED` processing is delayed or retried | Keep the event available for retry. Recheck the slot and do not create duplicate holds or offers. |
| New slot becomes unavailable during rescheduling | Roll back the full reschedule. The original appointment and old `BOOKED` slot remain unchanged. |
| Database write fails during a transition | Roll back all related slot, appointment, offer, and idempotency changes. |
| Expiration worker is delayed or stops | Treat the offer as logically expired using database time and reconcile it on the next relevant operation. |
