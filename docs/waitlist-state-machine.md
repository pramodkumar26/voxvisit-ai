# Waitlist and Offer State Machine

**Status:** Approved  
**Scope:** Patient waitlist entries and exclusive slot offers

## Purpose

This document defines the approved waitlist and offer lifecycles. PostgreSQL and backend services control all state changes. The AI agent may request approved tools but cannot create, update, accept, decline, or cancel waitlist and offer records directly.

## Waitlist Entry States

| State | Meaning |
|---|---|
| `ACTIVE` | The verified patient is waiting for a matching appointment. |
| `FULFILLED` | The patient successfully booked an appointment through an accepted offer. |
| `CANCELLED` | The patient removed themselves from the waitlist. |

## Waitlist Entry Transitions

The absence of a record is shown as `NONE` for clarity and is not a stored state.

| From | To | Trigger |
|---|---|---|
| `NONE` | `ACTIVE` | Verified patient confirms joining the waitlist |
| `ACTIVE` | `FULFILLED` | An owned offer is accepted and its appointment transaction commits |
| `ACTIVE` | `CANCELLED` | Verified patient confirms removal; any active offer is declined atomically |

A declined or expired offer does not change the waitlist entry from `ACTIVE`. The patient remains eligible for other matching slots.

## Offer States

| State | Meaning |
|---|---|
| `ACTIVE` | One slot is exclusively held for one eligible patient. |
| `ACCEPTED` | The intended patient accepted and the booking committed. |
| `DECLINED` | The intended patient rejected the offer. |
| `EXPIRED` | The offer was not accepted before its deadline. |

## Offer Transitions

| From | To | Trigger | Slot effect |
|---|---|---|---|
| `NONE` | `ACTIVE` | Worker atomically creates the hold and offer | `AVAILABLE` to `HELD` |
| `ACTIVE` | `ACCEPTED` | Intended patient accepts before `expires_at` and booking commits | `HELD` to `BOOKED` |
| `ACTIVE` | `DECLINED` | Intended patient declines or cancels the owning waitlist entry | `HELD` to `AVAILABLE` |
| `ACTIVE` | `EXPIRED` | PostgreSQL `clock_timestamp()` reaches or passes `expires_at` | `HELD` to `AVAILABLE` |

`ACCEPTED`, `DECLINED`, and `EXPIRED` are terminal offer states. An expired offer cannot be revived.

## Matching and Priority

- Selection orders eligible `ACTIVE` entries by `created_at` ascending, then by `waitlist_id` ascending when timestamps match.
- Eligibility matches clinic, specialty, date range, and preferred time.
- `Anytime` matches any eligible slot during clinic operating hours.
- `Morning` matches clinic-local times from 8:00 AM inclusive to 12:00 PM exclusive.
- `Afternoon` matches clinic-local times from 12:00 PM inclusive to 5:00 PM exclusive.
- Matching uses the clinic's configured timezone.
- Overlapping `ACTIVE` requests for the same patient, clinic, and specialty are prohibited. Historical `FULFILLED` and `CANCELLED` entries remain stored.
- A database constraint prevents more than one `ACTIVE` offer for the same slot while preserving terminal offer history.
- A patient may have at most one `ACTIVE` offer at a time in v1. Database constraints and transactional checks enforce this rule.
- A patient who declined or missed an offer for a slot is never offered that same slot again.
- Decline or expiration does not remove the patient from the waitlist. The entry may match another slot.
- The worker rechecks eligibility and current state after acquiring database locks.

## Offer Creation and Expiration

- A committed `SLOT_RELEASED` event triggers waitlist processing.
- An idempotent background worker selects the next eligible patient.
- In one transaction, the worker locks and rechecks the slot, changes it from `AVAILABLE` to `HELD`, and creates one `ACTIVE` offer.
- The default hold duration is five minutes and is configured globally for v1. Application logic uses the shared configuration rather than repeating a hardcoded duration.
- PostgreSQL database time exclusively creates the offer time and calculates `expires_at`.
- After acquiring the required locks, the backend uses PostgreSQL `clock_timestamp()` for the final expiration decision. Transaction-start `now()` is not used for this check.
- Acceptance is valid only when `clock_timestamp() < expires_at`.
- The offer's own deadline controls acceptance. The general 10-minute pending-confirmation window never extends the offer.
- At or after `expires_at`, the offer is logically expired even if a worker has not yet updated its stored state.
- A read-only request may report logical expiration but never changes the offer, waitlist, or slot state.
- Expiration state changes, slot release, and retryable next-patient work occur only through a controlled backend expiration or protected write operation.

## Offer Acceptance

- Acceptance requires a valid patient session, offer ownership, and explicit UI confirmation.
- Voice may prepare the action but cannot authorize acceptance in v1.
- The backend locks the offer, slot, and waitlist entry and rechecks their state and ownership.
- After acquiring those locks, the backend evaluates `clock_timestamp() < expires_at` before accepting.
- One transaction changes the offer to `ACCEPTED`, the waitlist entry to `FULFILLED`, and the slot from `HELD` to `BOOKED`.
- The same transaction creates one `CONFIRMED` appointment and its immutable `BOOKED` history record.
- The assistant reports success only after commit.

## Decline and Expiration Processing

- Decline requires a valid patient session and ownership of the `ACTIVE` offer.
- Decline changes the offer to `DECLINED` and releases the slot in one transaction.
- Expiration changes the offer to `EXPIRED` and releases the slot in one transaction.
- The waitlist entry remains `ACTIVE` after decline or expiration.
- The terminal offer record prevents the same patient from receiving the same slot again.
- The decline or expiration transaction durably records retryable next-patient processing for execution after commit.
- After commit, idempotent processing may create an offer for the next eligible patient.
- Cleanup delay does not extend a hold or make a late acceptance valid.

## Waitlist Cancellation with an Active Offer

- The backend verifies the patient session and waitlist ownership and requires explicit UI confirmation.
- One transaction changes the waitlist entry to `CANCELLED`, changes its `ACTIVE` offer to `DECLINED`, and changes the slot from `HELD` to `AVAILABLE`.
- The transaction records the retryable next-patient processing work for execution after commit.
- Idempotency prevents duplicate cancellation, offer decline, slot release, or next-patient processing.
- If the transaction fails, the waitlist entry, offer, and slot remain unchanged.

## Notifications

- V1 uses in-app notifications only.
- V1 does not use SMS or paid notification services.
- Notifications contain only the minimum synthetic information needed to review the offer.
- Notifications never contain PINs, session tokens, authentication secrets, or sensitive conversation content.
- Notification failure does not extend `expires_at`; delivery may be retried idempotently.

## Authorization and Safety

- Joining or cancelling a waitlist requires a verified patient session and explicit UI confirmation.
- Accepting an offer requires verification, ownership, and explicit UI confirmation.
- Protected patient identity comes from the server session, not model arguments.
- Demo staff have no waitlist mutation permissions in v1.
- Mutations and restricted reads create sanitized audit events.
- The AI agent cannot directly modify waitlist, offer, slot, or appointment records.

## Transaction Boundaries

- Transactions use PostgreSQL `READ COMMITTED` isolation with row-level locking and database constraints.
- Offer creation commits the slot hold and `ACTIVE` offer together.
- Database constraints and transactional checks prevent overlapping active waitlist entries and enforce at most one `ACTIVE` offer per slot and per patient.
- Acceptance commits the offer, waitlist entry, slot, appointment, immutable history, and idempotency outcome together.
- Decline and expiration commit the terminal offer state, slot release, and retryable next-patient work together.
- Waitlist cancellation without an offer commits the `CANCELLED` entry state and its audit event together.
- Waitlist cancellation with an `ACTIVE` offer commits the entry cancellation, offer decline, slot release, audit event, and retryable next-patient work together.
- Event and mutation processing is idempotent. Duplicate requests return the original result without repeating state changes.
- Any validation or database failure rolls back every change in the transaction.

## Relationship to Slots and Appointments

- An `ACTIVE` waitlist entry may exist without an active offer.
- An `ACTIVE` offer requires its slot to be `HELD` for the intended patient.
- An `ACCEPTED` offer requires the slot to be `BOOKED`, the appointment to be `CONFIRMED`, and the waitlist entry to be `FULFILLED`.
- A `DECLINED` or `EXPIRED` offer releases the slot to `AVAILABLE` before any later `AVAILABLE` to `HELD` transition.
- Appointment cancellation or rescheduling may create the committed `SLOT_RELEASED` event that begins matching.
- A waitlist cancellation creates no appointment and does not alter appointment history.

## Valid and Invalid Transitions

Valid examples:

- `ACTIVE` waitlist to `FULFILLED` only when offer acceptance and appointment creation commit.
- `ACTIVE` offer to `DECLINED` when the intended patient rejects it.
- `ACTIVE` offer to `EXPIRED` when PostgreSQL database time reaches `expires_at`.
- `ACTIVE` waitlist to `CANCELLED` together with `ACTIVE` offer to `DECLINED` when the owning patient confirms cancellation.

Invalid examples:

- Creating two `ACTIVE` offers for the same slot.
- Creating overlapping `ACTIVE` entries for the same patient, clinic, and specialty.
- Creating a second `ACTIVE` offer for a patient who already has one.
- Accepting an offer for another patient.
- Accepting when database time is equal to or later than `expires_at`.
- Moving `DECLINED` or `EXPIRED` back to `ACTIVE`.
- Reoffering the same slot to the patient who declined or missed it.
- Marking a waitlist `FULFILLED` before the appointment transaction commits.
- Allowing voice-only confirmation or an AI-generated ownership claim to authorize acceptance.

## Failure and Concurrency Scenarios

| Scenario | Expected database outcome |
|---|---|
| Duplicate `SLOT_RELEASED` delivery | Idempotent processing creates at most one active hold and offer. |
| Two overlapping waitlist requests are submitted together | At most one `ACTIVE` entry commits for the patient, clinic, and specialty. Historical entries remain unchanged. |
| Two workers process the same available slot | One may commit the hold. The other rechecks state and creates no offer. |
| Two workers try to offer different slots to the same patient | At most one `ACTIVE` offer commits. The other worker selects another eligible patient or safely stops. |
| Worker fails before offer creation commits | Roll back the hold and offer. The slot remains unchanged and the event remains retryable. |
| Worker fails after commit but before notification | Keep the committed offer and retry the in-app notification without extending expiry. |
| Two acceptance requests arrive together | At most one commits. An idempotent retry returns the original result. |
| Acceptance races expiration | Locks and `clock_timestamp()` decide the result. Acceptance succeeds only strictly before expiry. |
| Acceptance occurs exactly at `expires_at` | Reject acceptance, create no appointment, and process the offer as expired. |
| Acceptance races waitlist cancellation | One transaction commits first. The other rechecks the terminal state and fails without partial changes. |
| Wrong patient attempts acceptance or decline | Reject the request and commit no protected state changes. |
| Offer expires before UI confirmation | Create no appointment, mark or treat the offer as `EXPIRED`, and release the hold safely. |
| Appointment creation fails during acceptance | Roll back offer, waitlist, slot, appointment, history, and idempotency changes. |
| Decline or expiration processing is retried | Return or reproduce the terminal result without releasing the slot twice. |
| Waitlist cancellation with an active offer is retried | Return the original result without repeating cancellation, decline, release, or next-patient work. |
| Next-patient processing is delayed | The expired offer remains logically expired and cannot be accepted or revived. |
