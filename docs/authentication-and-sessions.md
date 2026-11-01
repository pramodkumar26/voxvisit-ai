# Authentication and Session Management

**Status:** Approved  
**Scope:** Fictional patient and demo staff authentication

## Purpose

This document defines the approved authentication and session rules for the VoxVisit AI demo. It uses synthetic identities only and does not claim production healthcare identity proofing, HIPAA compliance, or suitability for real patient data.

## Authentication Architecture

- The browser sends credentials directly to dedicated FastAPI authentication endpoints over HTTPS.
- Patient and staff credentials use separate login flows, credential records, roles, and permission checks.
- The backend validates credentials and creates a server-managed session.
- The browser receives an opaque session cookie. The database stores only a hash of the session token.
- The AI agent never receives PINs, staff passwords, session tokens, or authentication secrets.
- Protected services receive actor identity and role from the verified server session, never from LLM-generated arguments.

## Patient Authentication

- Patients sign in with a fictional patient ID and six-digit demo PIN through a private UI form outside the AI conversation.
- The patient ID identifies the synthetic account but is not proof of identity by itself.
- PINs are hashed with Argon2id. Plaintext PINs are never stored or logged.
- The backend returns a generic failure response for unknown patient IDs and incorrect PINs.
- Successful authentication binds the session to the verified patient ID and patient role.

## Staff Authentication

- Demo staff use separate staff credentials and a separate login endpoint.
- Staff passwords require at least 12 characters, reject common or compromised values, and are hashed with Argon2id. Staff never share patient credential records.
- Staff authentication never inherits or reuses a patient session.
- Staff cannot book, reschedule, cancel, join waitlists, or respond to offers in v1.
- A staff credential reset or security revocation invalidates all sessions for that staff account.
- Staff credential recovery is administrative reset only in v1. Automated email and SMS recovery are not supported.

## Permission Boundaries

Verified patients may access only their own appointments, waitlists, offers, and patient-specific handoffs. Appointment and waitlist writes require the approved explicit UI confirmation.

Authenticated demo staff may:

- View clinic appointment schedules.
- View approved limited fictional patient information.
- Manage doctor availability according to the approved slot rules.
- View and update staff handoff tickets.

Demo staff may not access patient PINs, session tokens, authentication secrets, or patient appointment mutation tools.

## Approved Session Defaults

These values are approved for v1:

| Setting | Patient | Staff | Reason |
|---|---:|---:|---|
| Absolute session lifetime | 30 minutes | 20 minutes | Limits exposure if a session is left open |
| Inactivity timeout | 15 minutes | 10 minutes | Staff views have broader access |
| Concurrent sessions | Up to 3 | 1 | Supports patient device changes while limiting staff exposure |
| Failed attempts before temporary lock | 5 within 15 minutes | 5 within 15 minutes | Slows guessing without making the demo unusable |
| Temporary lock duration | 15 minutes | 30 minutes | Staff credentials receive the stricter default |

Approved timing rules remain unchanged:

- A pending UI confirmation expires after 10 minutes.
- A waitlist offer uses the globally configured five-minute default and its own `expires_at` deadline.
- The offer deadline takes priority and is never extended by the pending-action window.

## Session Storage and Cookies

- Sessions are stored server-side in PostgreSQL with actor ID, actor type, role, token hash, creation time, last activity, idle deadline, absolute deadline, and revocation details.
- Session tokens are cryptographically random, opaque, and never stored in plaintext in the database.
- Prefer a same-site arrangement between the Vercel frontend and GCP Cloud Run API for secure session cookies.
- The browser uses a `Secure`, `HttpOnly` session cookie to call the API.
- CSRF protection and restricted credentialed CORS are required.
- Final `SameSite`, domain, path, and frontend-to-API cookie topology remain deferred until deployment domains are selected and tested in the deployed browser environment.
- Cross-site cookie behavior is never assumed to work without that test.
- Session tokens are never stored in browser local storage or included in URLs.

## Session States and Transitions

| From | To | Trigger |
|---|---|---|
| `NONE` | `ACTIVE` | Backend validates credentials and creates a session |
| `ACTIVE` | `ACTIVE` | Valid activity refreshes the idle deadline without extending the absolute lifetime |
| `ACTIVE` | `EXPIRED` | Idle or absolute deadline is reached |
| `ACTIVE` | `LOGGED_OUT` | User logs out of the current session |
| `ACTIVE` | `REVOKED` | Security action, credential reset, role change, or concurrent-session policy revokes it |

`EXPIRED`, `LOGGED_OUT`, and `REVOKED` are terminal. Reauthentication creates a new `ACTIVE` session and never revives the old one. A session is logically expired when its deadline passes even if cleanup has not run.

## Renewal, Logout, and Revocation

- Valid protected activity may refresh only the inactivity deadline and never extend the absolute lifetime.
- Session deadlines are evaluated against the application's clock. Recorded activity never moves backwards, so an instance whose clock lags slightly cannot fail a request or shorten a session.
- Renewal after absolute expiration requires credential reauthentication and creates a new session token.
- Token rotation occurs on successful reauthentication and security-sensitive renewal.
- Logout immediately invalidates the current session and its unfinished pending actions.
- Users may log out of their current session. Logout-all-sessions is restricted to administrative actions in v1.
- Administrative logout-all-sessions and credential reset revoke every active session for the account.
- Patient sessions in separate browsers remain independent. Pending actions are bound to the session that created them.
- When a patient logs in with three active sessions, the successful fourth login atomically revokes the oldest active session and invalidates its pending actions.
- A new successful staff login revokes the previous staff session and its pending actions.

## Login Protection

- Apply rate limits by account identifier and network source.
- Five failed attempts within 15 minutes lock a patient account for 15 minutes or a staff account for 30 minutes.
- Allow no more than 20 authentication attempts per network source in 15 minutes, in addition to the account lock limits.
- Use generic error messages so callers cannot distinguish unknown accounts from incorrect credentials.
- Successful login clears the account's failed-attempt counter but does not erase audit history.
- Credential verification uses Argon2id. Cost parameters are benchmarked before implementation.
- Staff password creation and reset reject values shorter than 12 characters and values found on the approved common or compromised password list.
- Logs never contain submitted PINs, passwords, session tokens, or credential hashes.

## Authorization and CSRF Protection

- Every protected request validates the session state, idle deadline, absolute deadline, role, and required permission.
- Patient-specific endpoints also verify ownership using the patient ID bound to the session.
- Write transactions recheck session validity and authorization before commit.
- State-changing browser requests require an approved Origin and a session-bound CSRF token sent outside the cookie.
- Restricted CORS allows credentials only from the approved frontend origin.
- Authentication and write endpoints use HTTPS only and reject credentials in query strings.

## Pending Actions

- Pending actions are stored server-side and bound to the actor and exact session that created them.
- A pending action contains only the action type, target identifiers, displayed summary, expected record versions, creation time, expiration time, and request or idempotency reference.
- Pending actions never contain PINs, staff passwords, session tokens, or sensitive conversation content.
- If a session expires, is logged out, or is revoked before confirmation, the pending action becomes unusable. Reauthentication requires the patient to review current details again.
- Logging in from another browser creates an independent session. It does not transfer or authorize pending actions from the first browser.
- If appointment, slot, offer, or ownership details change, the version check invalidates the pending action and no write commits.
- A pending action expires after 10 minutes. A waitlist offer may expire sooner, and the five-minute offer deadline remains controlling.

## Audit Logging

- Record login success and failure, logout, expiration, revocation, credential reset, protected reads, denied requests, and write attempts.
- Audit records include the event type, actor ID when known, actor type, session reference, result, UTC timestamp, request ID, and minimal sanitized network metadata.
- Never log PINs, passwords, raw session tokens, CSRF tokens, credential hashes, or sensitive conversation content.
- Appointment history remains the separate append-only record defined by the approved appointment state machine.

## Data Retention

- Expired session records are removed seven days after expiration.
- `EXPIRED`, `LOGGED_OUT`, and `REVOKED` sessions lose access immediately. Cleanup cannot reactivate them, and deleted session tokens fail lookup.
- Sanitized authentication audit events are retained for 30 days and then removed.
- Session and audit retention never stores authentication secrets.

## Failure and Security Scenarios

| Scenario | Expected outcome |
|---|---|
| Invalid patient ID or PIN | Return a generic failure, record a sanitized audit event, and update rate-limit counters. |
| Failed-attempt limit is reached | Temporarily block authentication for the account while preserving existing session policy. |
| Session reaches inactivity or absolute deadline | Treat it as `EXPIRED`, reject protected access, and invalidate pending actions. |
| Session expires during a booking transaction | Recheck before commit, roll back all writes, and require reauthentication and review. |
| User logs out with an unfinished action | Mark the session `LOGGED_OUT` and invalidate the pending action. |
| Patient signs in with fewer than three active sessions | Create an independent session and do not transfer pending actions. |
| Patient signs in with three active sessions | Create the new session and atomically revoke the oldest active session and its pending actions. |
| New staff login occurs while another is active | Revoke the older staff session and its pending work. |
| Appointment or slot changes after action preparation | Reject the stale action and require current details to be reviewed again. |
| Session cookie is copied but later revoked | Reject it because the server-side session is no longer `ACTIVE`. |
| CSRF token or allowed Origin check fails | Reject the state-changing request without modifying application data. |
| Waitlist offer expires while confirmation is open | Reject acceptance using the offer's database deadline and release the hold through the approved workflow. |

## Deployment Configuration Deferred

- Select and browser-test the final `SameSite`, cookie domain, cookie path, and Vercel-to-Cloud Run cookie topology after deployment domains are chosen.
