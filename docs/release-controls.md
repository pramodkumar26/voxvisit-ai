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

- Implemented in the tool layer (`app/agent_tools`): a fixed four-tool allowlist, a registration guard that refuses confirmation or commit names and handlers from outside the package, argument models that reject any model-supplied patient, role, session, or idempotency field, and registry-level and import-boundary tests. The confirmation operation also stays on its own router, the only caller of the commit service.
- Met by the conversation layer (`app/evaluation/conversation.py`, `hybrid.py`): the model never sees or chooses a tool (it only returns a structured intent), every call goes through the runtime, and the principal is the verified session's, never model output; the only non-tool operation, retiring a dropped pending action, is a trusted runtime method that cannot confirm anything. Each new tool must still be added to the allowlist deliberately, with tests.

### Retention and cleanup jobs

- Status: deferred to pre-release hardening (roadmap Phase 7), not cancelled. The jobs and their tests must exist and pass before any public or shared deployment.
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
- The CI workflow passed its first real GitHub Actions run (run 41270577286, commit `653db6b`). Refresh the pinned action versions periodically.
- Argon2id cost benchmarking, the approved common-password list, and administrative logout-all and credential-reset tooling.
- Neon compatibility for the PostgreSQL 17 features used (`transaction_timeout`, `btree_gist`, row-lock behavior) is unverified, as is behavior under load and in deployed browsers.

## Open Decisions

Behavior the implementation has settled for v1 but that the approved documents did not decide. Each is working as described; none changes the approved architecture. They need an explicit decision before the related feature ships.

1. **Clock for the self-service cutoff and "bookable in the future".** v1 uses the application's injected clock, the same clock availability search uses, while pending-action and offer deadlines use PostgreSQL time. The alternative is PostgreSQL time for the cutoff too, which removes application-clock skew from the decision at the cost of harder test control.
2. **Replay across sessions.** A replay is served only to the session that owns the pending action, because pending actions are session-bound. A patient who logs in again and retries an old key receives a not-found response, not the original result; their appointment is visible in their appointment list. The alternative is allowing the same patient to replay across sessions.
3. **Strictness of staleness.** Any version change to the appointment or a slot it depends on, even one that does not alter what the patient was shown, makes a pending action stale. This is the safest reading of "a relevant record-version change invalidates the pending action" but may cause avoidable re-confirmations once staff can edit slots.
4. **Reschedule scope.** A reschedule may move to any bookable slot at any clinic or with any doctor. The documents do not restrict it to the same doctor or clinic.
5. **Overlapping appointments.** Nothing prevents one patient from holding overlapping confirmed appointments with different doctors, and a patient may hold several active pending actions for the same target. Whether either should be restricted is undecided.
6. **Pending-action status.** A rejected confirmation does not write the status, so an expired action stays `ACTIVE` until a reconciliation job exists. The retention period for consumed, invalidated, and expired actions is not chosen.
7. **Retention tests in Phase 1.** The original Phase 1 sequence listed retention tests. They are deferred, not cancelled, to pre-release hardening together with the cleanup jobs, which do not exist yet, and they gate any public or shared deployment.
8. **Idempotency conflict before session recheck.** A key reused with different parameters is reported before the session is rechecked. It is reachable only with the same patient's own key and discloses nothing beyond that the key was used.
9. **Error body versus tool envelope.** The REST endpoints return plain models and `{"error": {...}}`. The tool runtime now provides the contract's envelope, `retryable`, `NO_RESULTS`, `NO_AVAILABILITY`, and `AMBIGUOUS_MATCH` for the four implemented tools; the remaining tools will add theirs. `AMBIGUOUS_MATCH` for the directory is implemented as: a clinic or doctor name matching several entities with no exact match, with at most five choices. The contract did not define that rule precisely, so it needs confirmation.
10. **Evaluation thresholds and the model adapter.** The harness reports baseline measurements only; the strategy defers task-completion, tool-selection, and latency thresholds until measured baselines exist. A local Ollama adapter (`app/evaluation/llm.py`) now implements the `Model` protocol, supplies response text, runs with response text required, bounds each call (timeout, output tokens, calls per conversation), and turns unreadable replies and provider faults into structured failures; its first measurements are in `docs/evaluation-strategy.md`. Still open: thresholds; whether EV-001 should let the synthetic patient answer a question (the model tested asks before preparing, so the trusted-confirmation step has not been reached by a real model); checking the details a message states against the trusted result, and non-English or unusual phrasings (the claim screen is rule-based and detects only that a success is claimed); and whether the evaluation CLI, or the real-model pilot, should run in CI.
