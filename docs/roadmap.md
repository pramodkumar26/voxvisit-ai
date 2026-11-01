# Development Roadmap

## Purpose

This document tracks VoxVisit AI development phases and progress separately from technical design documents. Scope, architecture, and fixed project decisions remain defined in the project blueprint.

## Current Status

Phase 0 Design and Decisions is complete. The master blueprint, workflows, state machines, authentication, PostgreSQL database design, agent tool contracts, evaluation strategy, and 20 initial scenarios are approved. Phase 1 Backend Foundation is in progress. The FastAPI foundation is implemented and verified against local Docker PostgreSQL 17. The approved PostgreSQL schema is implemented as SQLAlchemy models and three reviewed Alembic migrations: `0001` (24 tables, `btree_gist`, no triggers), `0002` (database hardening), and `0003` (a login rate-limit index). A reproducible, idempotent synthetic data seeder (`python -m app.seed`) loads fictional demonstration data. Patient and staff authentication, PostgreSQL-backed sessions, account lockout, network rate limiting, authorization dependencies, and CSRF and origin protection are implemented and verified locally, with login, logout, and current-session endpoints. PostgreSQL integration tests verify the migrations, constraints, seeder, authentication, and scheduling reads on disposable databases. Read-only scheduling APIs are implemented and verified locally: public clinic directory search, public appointment availability search with a 31-day window, and a verified patient's own appointment list, all with 20-item pages, clinic-timezone-aware times, and an 8-second read timeout. Appointment booking, rescheduling, and cancellation are implemented and verified locally as separate prepare and confirm operations, with atomic READ COMMITTED transactions in the canonical lock order, patient-scoped idempotency, immutable history, and transactional `SLOT_RELEASED` outbox events, covered by PostgreSQL race and rollback tests. A GitHub Actions workflow (PostgreSQL 17 service, locked dependencies, Ruff, `uv lock --check`, migration and schema-drift checks, and the complete database-enabled suite) is written and has been exercised locally against a disposable PostgreSQL 17, but has not yet run on GitHub. A reliability review of the appointment transactions corrected the write and read timeouts to bound the whole transaction, repeated the account checks after the locks, and made session refresh tolerate clock skew. A synthetic FHIR-like adapter (Patient, Practitioner, Schedule, Slot, and Appointment resources, read-only, behind the existing authorization, with PostgreSQL as the only store) is implemented and verified locally; it is deliberately not HL7 FHIR conformant and contacts no external system. Waitlist workflows, the `SLOT_RELEASED` consumer, retention jobs, write-endpoint rate limiting, and the runtime agent tool allowlist have not begun; outstanding release requirements are listed in `docs/release-controls.md`. The first real GitHub Actions run remains an outstanding verification requirement. Production deployment security controls, listed in the backend README, still await verification, including appointment-history immutability through restricted database-role grants.

## Development Phases

| Phase | Status | Estimate | Focus | Exit criterion |
|---|---|---:|---|---|
| 0. Design and decisions | Complete | 2 days | Requirements, architecture, verification, state rules, data entities, and initial scenarios | Design documents and scenarios reviewed |
| 1. Backend foundation | In Progress | 5 to 7 days | PostgreSQL, FastAPI, synthetic data, scheduling, authorization, transactions, tests, and initial CI | Booking APIs and security and concurrency tests pass |
| 2. Agent and early evaluation | Not Started | 5 to 7 days | Local model, tool schemas, conversation state, confirmation, and initial evaluation harness | Text agent performs booking reliably and can be regression tested |
| 3. Local voice | Not Started | 5 to 7 days | Push-to-talk speech input, spoken output, retry behavior, and latency measurement | First complete local voice booking succeeds |
| 4. RAG patient services | Not Started | 4 to 6 days | Approved clinic knowledge, retrieval, grounded answers, abstention, and evaluations | Administrative answers are grounded in source documents |
| 5. Waitlist and workflows | Not Started | 5 to 7 days | Exclusive holds, offers, expiration, race tests, multi-intent tasks, and simulated escalation | Waitlist concurrency tests prevent overbooking |
| 6. Complete interfaces | Not Started | 5 to 7 days | Patient interface, staff dashboard, operational views, and Vercel preview | Interfaces reflect authoritative backend state |
| 7. Hardening and live portfolio release | Not Started | 4 to 6 days | Cloud integration, live AI and voice validation, expanded evaluations, security checks, documentation, and teardown plan | Public demo works independently of the development Mac |

The total estimate remains 35 to 49 working days. Estimates are planning guides, not completion promises.

## Phase 1 Implementation Sequence

1. Establish the FastAPI application, configuration, error handling, and test foundation.
2. Define PostgreSQL schema migrations from the approved database design and run them against local Docker PostgreSQL.
3. Add deterministic synthetic clinics, doctors, patients, credentials, schedules, slots, and appointment fixtures.
4. Implement patient and staff authentication, sessions, authorization, ownership checks, CSRF boundaries, and audit events.
5. Implement clinic, doctor, and available-slot reads.
6. Implement booking, rescheduling, and cancellation with canonical locks, atomic transactions, idempotency, immutable history, and transactional `SLOT_RELEASED` events.
7. Add unit, PostgreSQL integration, authorization, concurrency, idempotency, and retention tests for the approved Phase 1 scenarios.
8. Add the initial GitHub Actions workflow for deterministic tests.
9. Implement the minimal synthetic FHIR-like adapter toward the end of Phase 1. EV-020 does not block the Phase 1 scheduling exit criterion but must pass before the final public release.

This sequence is approved planning. Step 1 is implemented and verified locally. Step 2 is implemented and verified locally: the initial migration applies to a clean database, downgrades cleanly on a disposable database, and matches the models. Rules that need transactional service enforcement, such as the three-session patient limit and cross-table state parity, are deliberately not database constraints and remain for steps 4 and 6. Migration `0002` hardened that schema after code review. Step 3 is implemented and verified locally: a deterministic seeder with explicit-anchor reproducibility, clinic-local daylight-saving-safe schedules, safe repeat execution, and Argon2id credentials read from local environment configuration. Step 4 is implemented and verified locally: separate patient and staff login, hashed server-side sessions with the approved limits, durable lockout and network rate limiting, reusable authorization dependencies, and CSRF protection, with no frontend and no business APIs. Its production cookie, proxy, and CORS behavior remains an implementation-time verification item. Step 5 is implemented and verified locally: the read-only clinic, doctor, and available-slot reads, plus a verified patient's own appointment list, with no booking, rescheduling, cancellation, or waitlist behavior. Step 6 is implemented and verified locally: booking, rescheduling, and cancellation as a prepare operation (a pending action that reserves nothing) and a separate UI-only confirm operation, with canonical locks, atomic transactions, idempotency, immutable history, and transactional `SLOT_RELEASED` events; doctor leave is now scoped to the clinic where it was recorded, in search and booking alike. Step 7 is partly complete: the unit, PostgreSQL integration, authorization, concurrency, idempotency, rollback, and clock-handling tests exist, while the retention-cleanup tests wait for the cleanup jobs. Step 8 is written and verified locally: the workflow's own steps pass against a disposable PostgreSQL 17 from a clean checkout, and each guard (version, skipped tests, leaked databases, schema drift) fails clearly when forced; its first real GitHub Actions run is still pending. Step 9 is implemented and verified locally: the read-only synthetic FHIR-like adapter with stable identifiers and references, approved status translations, patient isolation and staff permissions, and no writes, covered by unit and PostgreSQL integration tests. Its deterministic mapping assertions are the EV-020 checks that can run before the agent exists; EV-020 must still pass in full before the final public release. The waitlist, agent, and later phases have not begun. The remaining Phase 1 items are the first real GitHub Actions run, the retention-cleanup tests that wait for the cleanup jobs, and the release controls that are tracked in `docs/release-controls.md` for later phases.

## Implementation-Time Verification

- Neon PostgreSQL compatibility, version, connection limits, and required extension support.
- Hosted inference provider tool-calling, accuracy, latency, quota, terms, and cost benchmarking.
- Browser speech compatibility, fallback behavior, privacy, and latency.
- Vercel-to-Cloud Run cookie, CORS, CSRF, and deployed browser behavior.
- Argon2id cost benchmarking on the target runtime.
- Hosted RAG storage, embedding model, and provider compatibility.
- Cloud quotas, costs, budget controls, secrets, and deployment configuration.
- EV-020 synthetic FHIR adapter consistency before the final public release.

## Tracking Rules

- Technical documents use descriptive, professional, lowercase kebab-case filenames.
- Phase and step numbers do not appear in technical document filenames, titles, or metadata.
- Development phases and progress are tracked in this roadmap or in GitHub Issues.
- GitHub Issues may track individual tasks after the repository and issue workflow are established.
- Technical decisions remain in their relevant design documents and are not changed by roadmap status updates.
