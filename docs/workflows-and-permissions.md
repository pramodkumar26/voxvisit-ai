# VoxVisit AI Workflows and Permissions

**Status:** Approved  
**Source:** VoxVisit AI Project Blueprint, Version 3

## Purpose

This document defines the initial user workflows and permission boundaries for VoxVisit AI. The conversation agent may guide these workflows, but the backend controls authentication, authorization, confirmation, and all state changes.

## Roles

### Public user

A public user may:

- View clinic information and administrative policies.
- Search doctors and general appointment availability.
- Request general staff assistance.

A public user may not:

- View a patient's appointments or waitlist entries.
- Book, reschedule, or cancel an appointment.
- Join an account-specific waitlist or respond to an offer.
- Access the staff dashboard.

### Verified patient

A verified patient may:

- View their own appointments.
- Book, reschedule, or cancel their own appointments after explicit confirmation.
- Join a waitlist and respond to offers intended for them.
- Request patient-specific staff assistance.

A verified patient may not access or modify another patient's records.

### Demo staff

Demo staff use authentication that is separate from patient authentication. Staff permissions are not inherited from patient sessions.

Authenticated demo staff may:

- View clinic appointment schedules.
- View fictional patient ID and display name.
- View appointment doctor, date, time, and status.
- View relevant sanitized handoff information.
- Manage doctor availability according to the approved slot rules.
- View and update staff handoff tickets.

Authenticated demo staff may not:

- Book, reschedule, or cancel patient appointments in v1.
- Access patient PINs, session tokens, or authentication secrets.
- Access medical records or unrelated personal information.

## Permission Matrix

"Own" means the verified patient's records only.

| Action | Public user | Verified patient | Authenticated demo staff |
|---|---|---|---|
| View clinic information and policies | Allowed | Allowed | Allowed |
| Search doctors and general availability | Allowed | Allowed | Allowed |
| View clinic appointment schedules | Not allowed | Own appointments only | Doctor, date, time, and status only |
| View fictional patient information | Not allowed | Own information only | Patient ID and display name only |
| Book, reschedule, or cancel appointments | Not allowed | Own only, with confirmation | Not allowed |
| Join a waitlist or respond to an offer | Not allowed | Own only, with confirmation | Not allowed |
| Manage doctor availability | Not allowed | Not allowed | Allowed, subject to approved slot rules |
| Request general assistance | Allowed | Allowed | Allowed |
| Request patient-specific assistance | Not allowed | Own only | Not applicable |
| Access the staff dashboard | Not allowed | Not allowed | Allowed |
| View and update handoff tickets | Not allowed | Not allowed | Allowed |

## Authentication Boundary

- The patient enters their fictional patient ID and six-digit demo PIN in a private UI form.
- The PIN is sent directly to the authentication endpoint and never through the conversation agent.
- Patient PINs and strong staff passwords are hashed with Argon2id and never stored in plaintext.
- The backend verifies credentials and creates a server-managed session.
- Patient sessions have a 30-minute absolute lifetime, 15-minute inactivity timeout, and maximum of three active sessions.
- Staff sessions have a 20-minute absolute lifetime, 10-minute inactivity timeout, and maximum of one active session. A new staff login revokes the previous session.
- Five failed attempts within 15 minutes trigger a 15-minute patient lock or 30-minute staff lock. Network sources are limited to 20 authentication attempts in 15 minutes.
- Authentication failures use generic responses to prevent account enumeration.
- Protected operations obtain patient identity from the session, not from model-generated arguments.
- PINs, session tokens, and authentication secrets are excluded from prompts and logs.

## Confirmation Boundary

- Every appointment or waitlist state change requires explicit confirmation.
- The UI displays the exact pending action before confirmation.
- The displayed summary includes the relevant doctor, clinic, date, time, and operation.
- The patient confirms through an explicit UI control within 10 minutes.
- Voice interaction may prepare an action but cannot authorize its execution in v1.
- The backend rechecks the session, ownership, current state, and availability before committing.
- Session expiration, logout, revocation, or a relevant record-version change invalidates the pending action.
- The assistant reports success only after the backend transaction commits.
- Voice-only confirmation is deferred to a future version and requires a security evaluation.

## Booking Workflow

1. The patient searches available slots.
2. The patient selects a slot.
3. The backend requires verification if there is no valid patient session.
4. The backend creates a pending booking summary.
5. The UI displays the doctor, clinic, date, and time.
6. The patient selects Confirm.
7. The backend rechecks authorization and availability in one transaction.
8. The assistant reports the result returned by the backend.

## Rescheduling Workflow

1. The patient selects one of their current appointments.
2. The patient searches for and selects a replacement slot.
3. The UI displays both the current and proposed appointment details.
4. The patient selects Confirm.
5. The backend performs the move in one transaction.
6. If the replacement fails, the original appointment remains unchanged.
7. The assistant reports the committed result.

## Cancellation Workflow

1. The patient selects one of their appointments.
2. The UI displays the exact appointment to be cancelled.
3. The patient selects Confirm.
4. The backend cancels the appointment in one transaction.
5. The released slot triggers waitlist processing when appropriate.
6. The assistant reports success only after commit.

## Waitlist Workflow

1. A verified patient selects a clinic, specialty, date range, and time preference.
2. The UI displays the waitlist request and requires confirmation.
3. When a matching slot opens, the backend creates one exclusive five-minute offer.
4. The held slot cannot be booked or offered to another patient during that period.
5. The intended patient may accept or decline through the UI.
6. Acceptance rechecks identity, ownership, expiry, and slot state in one transaction.
7. An expired or declined offer may be offered to the next eligible patient.
8. A late acceptance always fails and cannot revive the offer.

## Staff Handoff Workflow

1. Any user may request general assistance.
2. Patient-specific assistance requires a verified patient session.
3. The backend creates an `OPEN` handoff ticket with a redacted summary and an approved category.
4. The ticket excludes PINs, session tokens, and unnecessary transcript content.
5. Handoff status is `OPEN`, `IN_PROGRESS`, or `RESOLVED`.
6. Authorized staff may view relevant sanitized handoff information, update status, and add sanitized notes.
7. Staff notes exclude authentication secrets, medical records, and sensitive conversation content.
8. The assistant identifies the handoff as simulated and does not promise a real staff response or response time.

## Failure Scenarios

### Authentication

- Failed verification creates no patient session and reveals no protected information.
- An expired session blocks the protected operation and requires verification again.
- A pending action is not committed while its session is invalid.

### Availability and Clinic Information

- If availability or policy data cannot be retrieved, the assistant reports that the information is unavailable and does not invent an answer.

### Booking

- Failed verification prevents the booking from reaching confirmation.
- If the session expires before commit, no appointment is created and the patient must verify again.
- If confirmation is not completed within 10 minutes, no appointment is created and the patient must review the current details again.
- If the selected slot becomes unavailable, the booking fails without changing any appointment data. The patient may search again.

### Rescheduling

- Failed verification or an expired session prevents the reschedule.
- An expired confirmation or a request at or after the clinic-local appointment start time is rejected without changing the appointment. The assistant offers simulated staff assistance for the cutoff case.
- If the replacement slot becomes unavailable, the transaction fails and the original appointment remains unchanged.
- If the current appointment no longer matches the pending action, the reschedule is rejected and its current state is shown.

### Cancellation

- Failed verification or an expired session prevents the cancellation.
- An expired confirmation or a request at or after the clinic-local appointment start time is rejected without changing the appointment. The assistant offers simulated staff assistance for the cutoff case.
- If the appointment was already cancelled or changed, no second mutation occurs and its current state is shown.
- A cancelled appointment cannot be restored. The patient must create a new booking, and the cancelled record remains as history.

### Waitlist

- Failed verification or an expired session prevents joining a waitlist or responding to an offer.
- An expired offer is rejected, does not create an appointment, and cannot be revived.
- If the held slot is no longer valid, acceptance fails without changing the patient's appointments.

### Staff Handoff

- Failed verification or an expired session prevents creation of a patient-specific ticket.
- If ticket creation fails, the assistant reports the failure and does not claim that staff were notified.

## Backend Enforcement Rules

- The conversation agent cannot grant access or override permissions.
- Protected record ownership is checked on every request.
- Model-provided patient identifiers are never treated as authentication.
- State-changing requests are idempotent and transactionally validated.
- Ambiguous doctor, date, time, appointment, or offer details must be clarified before confirmation.
- Restricted reads and all write attempts create sanitized audit events.
- Appointment history is append-only, written in the same transaction as the change, and excludes PINs, authentication tokens, and sensitive conversation content.
- Demo staff cannot book, reschedule, or cancel patient appointments in v1.
- Staff data access is limited to patient ID, display name, appointment doctor, date, time, status, and relevant sanitized handoff information.
- Staff authentication remains separate from patient authentication.

## Deployment Configuration Deferred

- Final `SameSite`, cookie domain, cookie path, and Vercel-to-Cloud Run cookie topology after deployed browser testing.
