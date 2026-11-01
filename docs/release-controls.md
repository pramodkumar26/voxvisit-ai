# Release Controls

**Status:** Tracking list (open requirements, not design decisions)
**Scope:** Controls that must exist and be verified before any public or shared deployment

## Purpose

This list records work that is intentionally not built yet so it cannot be forgotten. It adds no product rules: each item points to the approved document that already requires it. An item moves to the roadmap as verified work only after it is implemented and tested.

## Open Requirements

### Protected write-endpoint rate limiting

- Applies to `POST /me/appointment-actions`, its `/confirm` operation, and later waitlist and handoff writes. Only login has limits today. The read-only `/integration/fhir-like` endpoints need the same public-read limits as the other read endpoints.
- The tool contract already defines `RATE_LIMITED` (with a retry delay) for these tools. Limits per patient, per session, and per network source still need to be chosen, enforced in PostgreSQL or an approved shared store, and covered by tests. Public read endpoints need limits as well.

### Production cookie and reverse-proxy configuration

- Final `SameSite`, cookie domain, and path for the Vercel-to-Cloud Run topology, tested in deployed browsers.
- HTTPS only, `Secure` cookies and the `__Host-` cookie name, credentialed CORS limited to the real frontend origin (the allowed headers already include `X-CSRF-Token` and `Idempotency-Key`), and response security headers such as HSTS.
- Whether `X-Forwarded-For` can be trusted behind Cloud Run and the proxy hop count. `AUTH_TRUST_FORWARDED_FOR` stays off until verified.
- A production `AUTH_SECRET_KEY` held in a secret manager and stable across instances.
- Application instances must keep reasonably synchronized clocks (NTP). Session refresh tolerates small skew, and pending-action expiry uses the database clock, but business time (slot start, cancellation cutoff) uses the application clock.

### Appointment-history database-role permissions

- `appointment_history` is append-only by service convention only. The application role must have `INSERT` and `SELECT` on it and no `UPDATE`, `DELETE`, or `TRUNCATE`, with a separate role for migrations. The same least-privilege review applies to `outbox_events` (no deletes outside retention jobs) and `idempotency_records`.
- Requires integration tests that run the services as the restricted role.

### Runtime agent tool allowlisting

- The confirmation operation is separated structurally today: its own router, the only caller of the commit service, and tests enforcing both. The future agent runtime must also enforce it at runtime with an explicit tool registry that exposes `prepare_appointment_action` and the other approved tools and never `confirm`, rejects any model-supplied patient, role, session, or idempotency field, and attaches trusted context itself.
- Needs a registry-level test when the agent is built: every registered tool is on the approved list and none maps to a confirmation route.

### Retention and cleanup jobs

- Approved retention is not enforced yet: sessions (7 days after expiry), audit events (30 days), completed idempotency records (30 days after completion), processed outbox events (14 days), notifications (30 days).
- Pending actions have no cleanup. Consumed, invalidated, and logically expired rows should be reconciled to `EXPIRED` and removed after a retention period that still needs to be chosen.
- Cleanup must never delete unprocessed outbox work, and needs tests for foreign-key behavior and surviving history.

### Outbox consumption (waitlist phase)

- Booking changes already write `SLOT_RELEASED` events in the releasing transaction, one per release (deduplicated by slot and version), and nothing consumes them. They accumulate as `PENDING`, which loses nothing.
- The waitlist phase must add the consumer: claim events, recover abandoned `PROCESSING` claims, retry idempotently, and follow the canonical lock order (waitlist entry, offer, then slot) when creating offers.

### Synthetic FHIR-like adapter

- It is a simplified, synthetic, read-only demonstration of an integration boundary and must keep that label (no conformance, certification, or Epic claim) in any public material.
- Any real EHR integration is out of scope until it has separate credentials, a security review, mapping validation, conformance testing, and approval.
- EV-020 must pass in full before the final public release; the deterministic mapping and authorization checks exist now, the agent-facing checks come later.

### Other release prerequisites

- PostgreSQL 17 or newer is required: write and read transactions use `transaction_timeout`. Confirm the hosted provider's version before deployment.
- The CI workflow has been exercised locally but not yet on GitHub. Review the first real Actions run, and refresh the pinned action versions periodically.
- Argon2id cost benchmarking, the approved common-password list, and administrative logout-all and credential-reset tooling (see the backend README).
