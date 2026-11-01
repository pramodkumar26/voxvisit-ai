# Agent Tool Contracts

**Status:** Approved  
**Scope:** V1 conversational-agent access to FastAPI services

## Purpose

This document proposes the backend tools available to the single VoxVisit AI conversational agent. The tools expose only approved capabilities and sanitized synthetic data. They do not give the model direct PostgreSQL access, staff-dashboard permissions, authentication secrets, or authority to commit appointment or waitlist mutations.

## Architecture and Trust Boundary

- V1 uses one conversational agent with tools scoped to the current intent and verified permission level.
- The agent calls typed FastAPI service tools. Only FastAPI services access PostgreSQL or perform protected operations.
- The tool runtime attaches the validated session, role, request ID, and idempotency context outside model-generated arguments.
- Patient identity always comes from the validated backend session. Tool schemas do not accept a patient ID, actor ID, role, session token, or permission claim from the model.
- Public tools return no protected patient records. Patient tools reject public and staff sessions.
- PINs, passwords, session tokens, CSRF tokens, credential hashes, and database credentials never enter prompts, tool arguments, or tool results.
- Appointment and waitlist mutations require a separate explicit UI confirmation. Preparation tools create only a pending action.
- The agent cannot call the confirmation endpoint. The authenticated UI calls it with CSRF protection and an idempotency key.
- Booking, rescheduling, cancellation, waitlist, and offer success is reported only from a trusted backend result after the transaction commits.
- Staff dashboard operations use separate authenticated endpoints and are not exposed as unrestricted conversational-agent tools.

## Tool Inventory

| Tool | Access | Mode | Purpose |
|---|---|---|---|
| `search_clinic_directory` | Public, patient | Read-only | Search fictional clinics, doctors, specialties, hours, and administrative details |
| `search_appointment_availability` | Public, patient | Read-only | Search currently bookable, non-removed slots |
| `get_my_appointments` | Verified patient | Read-only | Retrieve appointments owned by the session patient |
| `prepare_appointment_action` | Verified patient | Pending action | Prepare booking, rescheduling, or cancellation for UI confirmation |
| `get_my_waitlist` | Verified patient | Read-only | Retrieve owned waitlist entries and offers |
| `prepare_waitlist_action` | Verified patient | Pending action | Prepare joining or cancelling a waitlist |
| `prepare_offer_action` | Verified patient | Pending action | Prepare accepting or declining an owned active offer |
| `request_staff_handoff` | Public, patient | Direct controlled side effect | Create a simulated, sanitized staff handoff ticket |
| `search_clinic_policies` | Public, patient | Read-only | Retrieve approved administrative policy and FAQ passages |

## Shared Request and Response Rules

Model-visible arguments contain only fields documented for the selected tool. The runtime adds trusted context separately:

- `request_id`
- session and actor context, when present
- allowed tool scope
- runtime-generated idempotency key for pending-action and handoff tools
- trace and timeout metadata

Every response uses a common sanitized envelope:

```json
{
  "ok": true,
  "request_id": "33333333-3333-4333-8333-333333333333",
  "data": {},
  "error": null
}
```

Errors use the same envelope and never reveal whether another patient's protected record exists:

```json
{
  "ok": false,
  "request_id": "33333333-3333-4333-8333-333333333333",
  "data": null,
  "error": {
    "code": "RESOURCE_NOT_AVAILABLE",
    "message": "That item is unavailable or cannot be accessed.",
    "retryable": false
  }
}
```

Responses contain no raw SQL errors, stack traces, internal table names, credentials, tokens, or unrestricted conversation transcripts.

## Tool Contracts

### `search_clinic_directory`

- Purpose: Find fictional clinics and doctors using administrative search criteria.
- Permitted users: Public users and verified patients. Staff use the separate dashboard.
- Inputs: Optional `clinic_name`, `doctor_name`, `specialty`, and `clinic_id`; bounded normalized strings; opaque IDs must use the expected format. At least one filter or an explicit directory-browse request is required. The backend enforces the 20-result page limit.
- Response: Up to 20 matching clinics and doctors per page, with clinic ID, name, timezone, public hours, administrative details, doctor ID, display name, specialty, clinic assignments, optional ambiguity choices, and pagination cursor.
- Authorization: No patient session is required. Public rate limits and tool-scope validation apply.
- Mode and side effects: Read-only. No pending action or database mutation.
- Idempotency: Repeated requests are safe and do not require stored idempotency records.
- Errors: `INVALID_ARGUMENT`, `AMBIGUOUS_MATCH`, `NO_RESULTS`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "I could not find a matching clinic or doctor. Please check the name or specialty."

### `search_appointment_availability`

- Purpose: Return slots that are currently searchable for booking.
- Permitted users: Public users and verified patients.
- Inputs: Required `clinic_id`, `start_date`, and `end_date`; optional `doctor_id`, `specialty`, and `time_preference` of `ANYTIME`, `MORNING`, or `AFTERNOON`. Dates use ISO format and the clinic timezone.
- Validation: Date order; maximum 31-day inclusive search window; doctor-clinic relationship; approved time-preference boundaries; clinic and doctor active status. The backend enforces all limits.
- Response: Up to 20 slots per page, with available slot ID, clinic and doctor summary, UTC start and end, clinic-local start and end, clinic timezone, and pagination cursor. Only slots with status `AVAILABLE` and `removed_at` null are returned.
- Authorization: No patient session is required. The result contains no patient or appointment-owner information.
- Mode and side effects: Read-only. Searching never holds or reserves a slot.
- Idempotency: Repeated requests are safe. Results may change between calls and are never proof that a later booking will succeed.
- Errors: `INVALID_ARGUMENT`, `AMBIGUOUS_MATCH`, `NO_AVAILABILITY`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "No matching times are currently available for that date range."

Illustrative request and response:

```json
{
  "tool": "search_appointment_availability",
  "arguments": {
    "clinic_id": "11111111-1111-4111-8111-111111111111",
    "specialty": "cardiology",
    "start_date": "2026-11-03",
    "end_date": "2026-11-07",
    "time_preference": "AFTERNOON"
  }
}
```

```json
{
  "ok": true,
  "request_id": "33333333-3333-4333-8333-333333333331",
  "data": {
    "slots": [
      {
        "slot_id": "22222222-2222-4222-8222-222222222222",
        "clinic_name": "Central Clinic",
        "doctor_name": "Dr. Rivera",
        "specialty": "Cardiology",
        "starts_at_utc": "2026-11-05T21:00:00Z",
        "clinic_local_start": "2026-11-05T14:00:00",
        "clinic_timezone": "America/Denver"
      }
    ],
    "next_cursor": null
  },
  "error": null
}
```

### `get_my_appointments`

- Purpose: Retrieve the verified patient's own confirmed or cancelled appointments.
- Permitted users: Verified patients only.
- Inputs: Optional `status` of `CONFIRMED` or `CANCELLED`, date range, and pagination cursor. No patient ID is accepted.
- Validation: Approved status values, valid date range, and a backend-enforced maximum of 20 results per page.
- Response: Up to 20 owned appointments per page, with appointment ID, status, clinic, doctor, specialty, UTC time, clinic-local time, timezone, and pagination cursor. It does not return credentials or unrelated patient data.
- Authorization: Requires an active patient session. The backend derives and filters by the session patient ID on every request.
- Mode and side effects: Read-only.
- Idempotency: Repeated requests are safe.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "Please verify your demo patient account to view appointments."

### `prepare_appointment_action`

- Purpose: Validate and prepare an appointment mutation for explicit UI confirmation.
- Permitted users: Verified patients only.
- Inputs: `action` with one of the following field sets:
  - `BOOK`: required `slot_id`.
  - `RESCHEDULE`: required `appointment_id` and `new_slot_id`.
  - `CANCEL`: required `appointment_id`.
- Validation: Active patient session; ownership for existing appointments; approved self-service cutoff; current appointment and slot state; non-removed slot; unambiguous identifiers; no unsupported fields.
- Response: Pending action ID, action type, stored status `ACTIVE`, UI label `AWAITING_UI_CONFIRMATION`, exact display summary, creation time, 10-minute `expires_at`, and `confirmation_required: true`. Internal expected versions remain server-side.
- Authorization: Patient identity and ownership come only from the session. Public and staff sessions are rejected.
- Mode and side effects: Creates or returns an idempotent `pending_actions` record only. It does not create, move, cancel, hold, or reserve an appointment or slot.
- Idempotency: The runtime supplies an idempotency key. Retrying the same preparation with the same normalized arguments returns the original pending action. Reusing the key with different arguments fails.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `RESOURCE_NOT_AVAILABLE`, `SLOT_UNAVAILABLE`, `SELF_SERVICE_CUTOFF`, `CONFLICT`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "That appointment change cannot be prepared. Please review the current appointment details."

Illustrative booking preparation:

```json
{
  "tool": "prepare_appointment_action",
  "arguments": {
    "action": "BOOK",
    "slot_id": "22222222-2222-4222-8222-222222222222"
  }
}
```

```json
{
  "ok": true,
  "request_id": "33333333-3333-4333-8333-333333333332",
  "data": {
    "pending_action_id": "44444444-4444-4444-8444-444444444444",
    "action": "BOOK",
    "stored_status": "ACTIVE",
    "ui_status": "AWAITING_UI_CONFIRMATION",
    "summary": {
      "clinic": "Central Clinic",
      "doctor": "Dr. Rivera",
      "date": "2026-11-05",
      "time": "2:00 PM",
      "timezone": "America/Denver"
    },
    "created_at": "2026-11-01T18:00:00Z",
    "expires_at": "2026-11-01T18:10:00Z",
    "confirmation_required": true
  },
  "error": null
}
```

This response means only that the action is ready for review. It is not a booking result.

### `get_my_waitlist`

- Purpose: Retrieve the verified patient's waitlist entries and related offers.
- Permitted users: Verified patients only.
- Inputs: Optional `waitlist_id`, waitlist status, offer status, and pagination cursor. No patient ID is accepted.
- Validation: Approved waitlist and offer states, identifier format, and a backend-enforced maximum of 20 results per page.
- Response: Up to 20 owned waitlist entries per page, with preferences, status, related offer ID, offer state, slot summary, `expires_at`, database-derived remaining time when applicable, and pagination cursor.
- Authorization: Requires an active patient session and ownership filtering by the backend.
- Mode and side effects: Read-only. Reading an offer does not extend or revive it.
- Idempotency: Repeated requests are safe.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `RESOURCE_NOT_AVAILABLE`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "That waitlist item or offer is unavailable or cannot be accessed."

### `prepare_waitlist_action`

- Purpose: Prepare joining or cancelling a patient waitlist entry.
- Permitted users: Verified patients only.
- Inputs: `action` with one of the following field sets:
  - `JOIN`: required `clinic_id`, `specialty`, `start_date`, `end_date`, and `time_preference` of `ANYTIME`, `MORNING`, or `AFTERNOON`.
  - `CANCEL`: required `waitlist_id`.
- Validation: Active session; clinic and specialty; clinic-local preference rules; date order; no overlapping active request for the same patient, clinic, and specialty; ownership and current state for cancellation.
- Response: Pending action ID, stored status `ACTIVE`, UI label `AWAITING_UI_CONFIRMATION`, sanitized waitlist summary, 10-minute `expires_at`, and `confirmation_required: true`.
- Authorization: Patient identity and ownership come only from the session.
- Mode and side effects: Creates or returns a pending action only. It does not create or cancel a waitlist entry, decline an offer, or release a slot.
- Idempotency: The runtime supplies the key. Same-key retries return the original preparation; changed arguments are rejected.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `OVERLAPPING_WAITLIST`, `RESOURCE_NOT_AVAILABLE`, `CONFLICT`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "That waitlist request overlaps an active request already associated with your account."

### `prepare_offer_action`

- Purpose: Prepare acceptance or decline of an active offer for UI confirmation.
- Permitted users: The verified patient who owns the offer.
- Inputs: Required `action` of `ACCEPT` or `DECLINE` and required `offer_id`. No patient, waitlist, or slot ownership fields are accepted from the model.
- Validation: Active session; offer ownership; matching waitlist and patient; offer status `ACTIVE`; matching `HELD` slot; PostgreSQL `clock_timestamp()` strictly before `expires_at`.
- Response: Pending action ID, action, stored status `ACTIVE`, UI label `AWAITING_UI_CONFIRMATION`, exact offer and slot summary, offer `expires_at`, pending-action `expires_at`, and `confirmation_required: true`.
- Authorization: Backend session identity and database relationships determine ownership.
- Mode and side effects: Creates or returns a pending action only. It does not accept, decline, book, release, extend, or revive the offer.
- Expiration: The pending action ends at the earlier of its 10-minute limit or the offer's own deadline. The offer deadline always controls.
- Idempotency: Same-key retries return the original pending action. A retry never extends either deadline.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `RESOURCE_NOT_AVAILABLE`, `OFFER_EXPIRED`, `OFFER_NOT_ACTIVE`, `CONFLICT`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "That offer is no longer active. Please review your current waitlist status."

### `request_staff_handoff`

- Purpose: Create a simulated handoff ticket without promising a real staff response.
- Permitted users: Public users for general assistance; verified patients for general or patient-specific assistance.
- Inputs: Required `assistance_type` of `GENERAL` or `PATIENT_SPECIFIC`; required `category` of `SCHEDULING`, `CLINIC_INFORMATION`, `ACCESSIBILITY`, `TECHNICAL_SUPPORT`, or `OTHER_ADMINISTRATIVE`; and a sanitized `summary` of at most 500 characters. No transcript, credential, or model-supplied patient ID is accepted.
- Validation: Public users may request only `GENERAL`; patient-specific requests require an active patient session; the category must match the approved allowlist; the sanitized summary must not exceed 500 characters; sensitive content is redacted or rejected.
- Response: Ticket ID, status `OPEN`, `simulated: true`, and a message that no response time is promised.
- Authorization: Patient identity for a patient-specific ticket comes only from the session.
- Mode and side effects: Directly creates one sanitized `OPEN` handoff ticket and stores the validated `category`. The ticket is general when no patient is attached and patient-specific otherwise. It does not change appointments, waitlists, offers, or staff assignment.
- Idempotency: Required. Same-key retries return the original ticket and never create duplicates. Authenticated requests use the session patient as the scope. Public requests use a backend-generated, high-entropy `ANONYMOUS` scope that stays stable across retries, is never supplied by the model, and grants no permissions. Keys are scoped to actor, operation, and request.
- Errors: `AUTH_REQUIRED`, `SESSION_EXPIRED`, `INVALID_ARGUMENT`, `SENSITIVE_CONTENT_REJECTED`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message example: "I could not create the simulated handoff ticket. No staff notification was sent."

### `search_clinic_policies`

- Purpose: Retrieve approved fictional clinic policy and FAQ passages for grounded administrative answers.
- Permitted users: Public users and verified patients.
- Inputs: Required bounded `query`; optional `clinic_id`. The backend restricts retrieval to the approved synthetic administrative corpus and at most five passages.
- Validation: Query length, allowed administrative scope, active clinic, and approved corpus only.
- Response: At most five ranked passages. Every passage contains source ID, title, section, sanitized excerpt, and clinic scope. The tool returns evidence, not a fabricated policy answer.
- Authorization: No patient session is required. Retrieved content contains no protected patient data.
- Mode and side effects: Read-only.
- Idempotency: Repeated requests are safe.
- Errors: `INVALID_ARGUMENT`, `POLICY_NOT_FOUND`, `CLINICAL_REQUEST_UNSUPPORTED`, `RATE_LIMITED`, `DEPENDENCY_UNAVAILABLE`.
- Safe message examples: "I could not find an approved clinic policy that answers that question." and "This demo cannot provide medical advice."

Illustrative grounded retrieval response:

```json
{
  "ok": true,
  "request_id": "33333333-3333-4333-8333-333333333333",
  "data": {
    "matches": [
      {
        "source_id": "55555555-5555-4555-8555-555555555555",
        "title": "Central Clinic Visitor Information",
        "section": "Parking",
        "excerpt": "Validated parking is available for fictional clinic visitors."
      }
    ]
  },
  "error": null
}
```

## Confirmation Boundary

1. A preparation tool validates the request and stores a session-bound pending action.
2. The UI displays the exact operation and relevant clinic, doctor, date, time, appointment, waitlist, or offer details.
3. The patient confirms through a dedicated authenticated UI control.
4. The UI calls a separate FastAPI confirmation endpoint with the pending action ID, CSRF protection, and an idempotency key.
5. The backend rechecks the session, ownership, pending-action deadline, record versions, current states, and applicable appointment or offer deadline.
6. The backend follows the approved database lock order and commits the full transaction or rolls it back.
7. A trusted backend event delivers the committed result to the UI and conversation runtime. Only that event may support a success statement.

Pending confirmation lasts at most 10 minutes and never reserves a slot. Offer acceptance additionally requires PostgreSQL `clock_timestamp() < expires_at` after database locks are acquired. The default offer duration remains five minutes and is not extended by preparation or confirmation.

## Ambiguous Requests

- Tools return `AMBIGUOUS_MATCH` with a small list of sanitized choices when a clinic, doctor, appointment, date, time, or offer is not uniquely identified.
- The agent asks the user to choose and does not select a default on the user's behalf.
- Relative dates must be resolved to an explicit date using the clinic timezone and repeated to the user before preparation.
- Preparation tools require opaque IDs obtained from current backend results rather than relying on names alone.
- If details change after preparation, confirmation fails as stale and the patient must review a new summary.

## Standard Error Taxonomy

| Code | Meaning | Retry behavior |
|---|---|---|
| `INVALID_ARGUMENT` | Missing, malformed, unsupported, or contradictory input | Correct input before retry |
| `AMBIGUOUS_MATCH` | More than one valid clinic, doctor, appointment, or time | Ask the user to choose |
| `AUTH_REQUIRED` | A protected tool has no valid patient session | Use the private verification UI |
| `SESSION_EXPIRED` | The session is no longer active | Reauthenticate and review again |
| `FORBIDDEN` | The authenticated role cannot use the tool | Do not retry with the same role |
| `RESOURCE_NOT_AVAILABLE` | Protected item is absent, inaccessible, or no longer usable | Refresh owned state; do not reveal existence |
| `NO_RESULTS` | A public directory search found no match | Refine the query |
| `NO_AVAILABILITY` | No matching available slots were found | Change search criteria |
| `SLOT_UNAVAILABLE` | A selected slot cannot be prepared or committed | Search again |
| `SELF_SERVICE_CUTOFF` | Cancellation or rescheduling is at or after appointment start | Offer simulated staff assistance |
| `OVERLAPPING_WAITLIST` | An overlapping active waitlist entry exists | Review current waitlist status |
| `OFFER_EXPIRED` | Database time reached the offer deadline | Refresh waitlist status |
| `OFFER_NOT_ACTIVE` | The offer is already terminal or otherwise unusable | Refresh waitlist status |
| `CONFLICT` | Current backend state conflicts with the requested preparation | Refresh and review |
| `POLICY_NOT_FOUND` | Approved sources contain no supported answer | State that no approved answer was found |
| `CLINICAL_REQUEST_UNSUPPORTED` | The request asks for diagnosis, treatment, or unsupported clinical guidance | Decline and offer appropriate assistance |
| `SENSITIVE_CONTENT_REJECTED` | A handoff summary contains prohibited content | Ask for a minimal administrative summary |
| `RATE_LIMITED` | Request limits were reached | Retry only after the supplied delay |
| `DEPENDENCY_UNAVAILABLE` | A required service timed out or failed | Retry according to tool policy |
| `INTERNAL_ERROR` | Unexpected safe failure | Do not expose internals or claim success |

## Timeout, Retry, and Idempotency Behavior

- Read-only tools use an initial timeout of 8 seconds.
- Pending-action and handoff tools use an initial timeout of 12 seconds.
- The runtime may retry a read-only call once after a transient timeout using bounded jitter.
- Preparation, handoff, and other state-changing calls may be retried only with the same runtime-generated idempotency key. An unknown outcome is resolved through the idempotency record rather than creating a second action or ticket.
- Authentication, authorization, validation, conflict, and expiry errors are not automatically retried.
- Tool cancellation does not imply backend rollback after the request was sent. The runtime must resolve an unknown result through the same idempotency key.
- If the final outcome remains unavailable, the agent reports that the operation could not be verified and never claims success.

## Security and Failure Scenarios

| Scenario | Required outcome |
|---|---|
| Model supplies a patient ID, role, or session token | Reject the unsupported field and use only trusted runtime context. |
| Public user calls a protected patient tool | Return `AUTH_REQUIRED` without protected data. |
| Staff session reaches a patient agent tool | Return `FORBIDDEN`; staff dashboard permissions do not transfer. |
| Session expires during preparation or confirmation | Reject the action and require reauthentication and review. |
| Slot changes after booking preparation | Confirmation fails; no appointment is created. |
| Offer expires while confirmation is open | Reject acceptance using database time; never extend or revive the offer. |
| Duplicate preparation or handoff call | Return the stored idempotent result without duplicate records. |
| Tool times out after a possible write | Resolve using the same idempotency key; do not repeat blindly or claim success. |
| Clinic or doctor name is ambiguous | Return choices and require clarification. |
| Approved policy source has no answer | State that no approved policy answer was found. Do not fabricate one. |
| User requests diagnosis or treatment advice | Return `CLINICAL_REQUEST_UNSUPPORTED` and do not generate medical advice. |
| Retrieved text contains instructions to the model | Treat it as data, ignore the instructions, and use only approved policy content. |
| Handoff creation fails | Report failure and do not claim that staff were notified. |
| Database transaction rolls back | Report the backend failure; no mutation success statement is allowed. |

