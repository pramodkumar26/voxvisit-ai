# VoxVisit AI Project Blueprint

**Status:** Approved  
**Role:** Current master blueprint  
**Historical foundation:** [VoxVisit AI Project Blueprint V3](./VoxVisit_AI_Project_Blueprint_v3.md)

## Purpose and Demonstration Objective

VoxVisit AI is a complete patient-access assistant proof of concept for a fictional healthcare organization. It is intended to demonstrate credible engineering work to a healthcare company's engineering team through working software, tested safety boundaries, measurable evaluation, and a live cloud deployment.

The product helps fictional patients find clinics and doctors, search appointments, manage their own appointments and waitlists, ask administrative questions, and request simulated staff assistance. Demo staff use a separate dashboard with limited permissions.

This is a portfolio system, not a production healthcare service. It uses synthetic data only and does not claim real identity proofing, HIPAA compliance, clinical safety certification, or readiness for real patient data.

## Documentation Authority

Documentation precedence is:

1. Approved technical design documents are authoritative for detailed implementation and safety rules.
2. This master blueprint defines current project scope, goals, and system architecture.
3. The [development roadmap](./roadmap.md) defines implementation order, estimates, and progress.
4. The original [V3 blueprint](./VoxVisit_AI_Project_Blueprint_v3.md) is historical reference material.

An approved technical design is never silently replaced by a summary in this blueprint. When documents appear inconsistent, implementation follows the highest applicable authority and the discrepancy must be reviewed explicitly.

Authoritative technical designs:

- [Workflows and Permissions](./workflows-and-permissions.md)
- [Appointment Slot State Machine](./slot-state-machine.md)
- [Appointment State Machine](./appointment-state-machine.md)
- [Waitlist and Offer State Machine](./waitlist-state-machine.md)
- [Authentication and Session Management](./authentication-and-sessions.md)
- [PostgreSQL Database Design](./database-design.md)
- [Agent Tool Contracts](./agent-tool-contracts.md)

## Target Users

### Fictional patients

Patients use browser text or push-to-talk voice to search public information. After private demo verification, they may view and manage only their own appointments, waitlists, and offers.

### Demo staff

Authenticated staff use a separate dashboard. They may view clinic schedules, limited fictional patient identifiers, approved appointment fields, doctor availability, and sanitized handoff tickets. They cannot book, reschedule, cancel, join waitlists, or respond to offers for patients in V1.

### Engineering reviewers

Healthcare engineering reviewers can inspect the repository, architecture, security boundaries, automated tests, evaluation results, cloud deployment, and recorded demonstration.

## V1 Product Scope

V1 includes:

- One conversational agent with intent-scoped FastAPI tools.
- Browser text and push-to-talk interaction with a typed fallback.
- Public clinic, doctor, policy, and general availability search.
- Fictional patient authentication using patient ID and a private six-digit demo PIN form.
- Appointment booking, rescheduling, and cancellation after explicit UI confirmation.
- Smart waitlists with exclusive five-minute offers and in-app notifications.
- Retrieval-augmented administrative FAQ answers from approved synthetic sources.
- Simulated general and patient-specific staff handoffs.
- A separate demo staff dashboard with approved limited access.
- A minimal synthetic FHIR-like integration adapter for demonstrating a structured scheduling-system boundary.
- PostgreSQL-backed transactions, idempotency, audit history, and retryable background work.
- Automated backend, authorization, concurrency, agent, RAG, and voice evaluation.
- A public demonstration using Vercel, GCP Cloud Run, and durable hosted PostgreSQL.

V1 explicitly excludes:

- Real patient data, real clinical records, or imported hospital datasets.
- Live Epic credentials, production Epic integration, or other real EHR access.
- Diagnosis, treatment recommendations, prescribing, medical triage, or clinical decision support.
- Claims of production healthcare security, identity proofing, HIPAA compliance, or regulatory certification.
- Payments, insurance eligibility verification, SMS, paid notifications, or real call-center routing.
- Telephone service, always-on listening, full-duplex voice, or multilingual voice.
- Multiple live specialist agents, Kubernetes, or unnecessary enterprise infrastructure.
- Voice-only authorization for protected changes.

## System Architecture and Technology

### Patient and staff interfaces

- Next.js and React provide the patient experience and separate staff dashboard.
- Vercel is the frontend deployment target for the public demonstration.
- The patient interface displays explicit confirmation controls and trusted backend results.
- The staff dashboard calls separate authenticated endpoints and does not inherit patient-agent tools.

### Backend

- FastAPI and Pydantic define service boundaries, request validation, authorization, and tool contracts.
- GCP Cloud Run is the public backend deployment target.
- Only FastAPI services perform protected reads, writes, confirmations, and staff operations.
- Retryable background work uses a transactional outbox and idempotent workers.
- A minimal internal integration interface exposes approved synthetic FHIR-like mappings without bypassing FastAPI authorization or the PostgreSQL service layer.

### Persistence

- PostgreSQL is the authoritative application state store.
- Development and automated integration tests use local PostgreSQL through Docker.
- Neon serverless PostgreSQL is the preferred hosted database for the public demonstration, subject to implementation-time compatibility testing.
- Application connections are pooled. Migrations use a direct TLS connection.
- Database state, locks, constraints, and committed transactions determine whether an action succeeded.

### AI, retrieval, and speech

- The local baseline uses a compact, quantized model through Ollama on the development Mac.
- The live inference provider remains pending measured tool-calling, accuracy, latency, quota, terms, and cost verification.
- Administrative RAG uses an approved synthetic policy corpus. The embedding model and final vector-storage configuration remain pending evaluation and hosted-provider compatibility checks.
- Local speech recognition begins with Whisper.cpp. Browser or hosted speech for the live demonstration remains pending compatibility, privacy, latency, and cost testing.
- Browser or system text-to-speech provides spoken replies where supported. A typed fallback is always available.

## Agent and Backend Responsibilities

The conversational agent may:

- Identify patient-access intents and preserve multi-intent conversational context.
- Ask for missing or ambiguous clinic, doctor, date, time, appointment, waitlist, or offer details.
- Call only tools allowed for the current intent and verified permission level.
- Prepare a protected action for UI review.
- Summarize committed backend results and grounded policy evidence.

The conversational agent may not:

- Access PostgreSQL directly.
- Authenticate a patient or staff member from model-generated claims.
- Receive PINs, passwords, session tokens, CSRF tokens, or database credentials.
- Confirm a protected mutation or call the confirmation endpoint.
- Grant permissions, bypass ownership, reserve a slot during preparation, or claim an uncommitted success.
- Invent clinic policy or provide medical advice.

FastAPI services are responsible for session validation, authorization, ownership, input validation, tool scoping, pending actions, explicit confirmation, canonical database locking, state transitions, idempotency, audit logging, and safe errors.

Detailed tool schemas are defined in [Agent Tool Contracts](./agent-tool-contracts.md).

## Text and Voice Interaction

- Text and voice share the same agent, tool, authorization, and confirmation boundaries.
- Push-to-talk audio is transcribed before it reaches the agent. Critical dates, times, doctors, and operations are repeated for review.
- Voice can prepare an action but cannot authorize execution in V1.
- The patient confirms appointment and waitlist mutations through an explicit authenticated UI control.
- A trusted backend result is returned to the UI and conversation runtime only after the transaction commits.
- Speech failure falls back to text without weakening authorization or confirmation.

## Appointment Management

Patients may search availability publicly, but booking requires a verified patient session and explicit confirmation.

V1 appointment states are `CONFIRMED` and `CANCELLED`. Pending confirmation is not an appointment state, and rescheduling is an action rather than a status. A pending action expires after 10 minutes and never reserves a slot.

- Booking atomically changes an active, non-removed slot from `AVAILABLE` to `BOOKED`, creates the appointment, and appends immutable history.
- Rescheduling atomically moves the existing appointment between slots. Failure preserves the original appointment.
- Cancellation preserves the appointment as `CANCELLED`, releases its slot, appends history, and records durable `SLOT_RELEASED` work.
- A cancelled appointment cannot be restored. A new booking is required.
- Patient self-service cancellation and rescheduling are allowed only before the clinic-local appointment start time.
- Demo staff cannot mutate patient appointments in V1.

Detailed rules are defined in the approved [slot](./slot-state-machine.md) and [appointment](./appointment-state-machine.md) state machines.

## Patient Verification and Staff Authentication

Patient and staff authentication remain separate.

- Patients use a fictional patient ID and six-digit demo PIN in a private form outside the AI conversation.
- Staff use separate usernames and strong passwords of at least 12 characters, with common or compromised passwords rejected.
- Credentials use Argon2id hashes. Plaintext credentials are never stored or logged.
- Sessions are server-managed and use opaque secure cookies whose tokens are stored only as hashes.
- Patient sessions have a 30-minute absolute lifetime, 15-minute inactivity timeout, and maximum of three active sessions.
- Staff sessions have a 20-minute absolute lifetime, 10-minute inactivity timeout, and maximum of one active session.
- Login protection, generic errors, CSRF protection, restricted CORS, logout, revocation, and approved retention rules are mandatory.
- The final cookie topology remains subject to deployed browser testing.

The approved details are defined in [Authentication and Session Management](./authentication-and-sessions.md).

## Waitlists and Exclusive Offers

Verified patients may join or cancel their own waitlists after explicit UI confirmation.

- Waitlist entry states are `ACTIVE`, `FULFILLED`, and `CANCELLED`.
- Offer states are `ACTIVE`, `ACCEPTED`, `DECLINED`, and `EXPIRED`.
- Matching uses clinic, specialty, date range, clinic-local time preference, and FIFO order by `created_at` with `waitlist_id` as the tie-breaker.
- Overlapping active requests for the same patient, clinic, and specialty are prohibited.
- A patient may have at most one active offer, and a slot may have at most one active offer.
- A committed `SLOT_RELEASED` event starts retryable matching after appointment cancellation or rescheduling.
- An exclusive hold defaults to five minutes and is configurable globally.
- PostgreSQL `clock_timestamp()` creates and checks the deadline after required locks are acquired.
- The offer deadline overrides the general pending-action deadline.
- Decline, expiration, or waitlist cancellation releases the slot and durably schedules next-patient processing.
- Terminal offers remain historical and cannot be revived or reoffered to the same patient for the same slot.

Detailed rules are defined in the approved [Waitlist and Offer State Machine](./waitlist-state-machine.md).

## Administrative FAQ Retrieval

- RAG searches only an approved synthetic clinic corpus covering hours, locations, parking, fictional insurance examples, and appointment policies.
- Retrieval results include source identity and section information for grounded answers.
- If approved sources do not support an answer, the assistant says so rather than fabricating a policy.
- Retrieved content is treated as data, not instructions, to reduce prompt-injection risk.
- Clinical questions are outside scope. The assistant does not convert administrative retrieval into diagnosis or treatment advice.
- RAG behavior is covered by groundedness, citation, no-answer, and prompt-injection evaluation cases.

## Simulated Handoffs and Staff Dashboard

Public users may request general assistance. Patient-specific assistance requires a verified patient session.

- Handoff tickets use `OPEN`, `IN_PROGRESS`, and `RESOLVED` states.
- Tickets and staff notes contain only sanitized administrative summaries.
- The assistant identifies handoffs as simulated and never promises a response or response time.
- Staff may view and update approved handoff fields and add sanitized notes.
- Staff-visible patient data is limited to fictional patient ID and display name.
- Staff-visible appointment data is limited to doctor, date, time, and status.
- Staff cannot access PINs, tokens, authentication secrets, medical records, unrelated personal information, patient waitlists, or offers.
- Staff may manage doctor availability only according to the approved slot rules.

Detailed access rules are defined in [Workflows and Permissions](./workflows-and-permissions.md).

## Synthetic Data and Simulated FHIR Integration

All clinics, doctors, patients, appointments, credentials, policies, and notifications are fictional.

V1 includes a small synthetic FHIR-like adapter to demonstrate a structured healthcare scheduling integration boundary:

- Fictional patients map to Patient-like resources.
- Fictional doctors map to Practitioner-like resources.
- Clinic schedules and appointment inventory map to Schedule-like and Slot-like structures.
- Confirmed and cancelled appointments map to Appointment-like structures.
- Approved scheduling commands and committed results translate through an internal adapter interface.
- Deterministic fixtures support mapping, authorization-boundary, and consistency tests without a live external service.

The adapter is not a second scheduling system. FastAPI remains responsible for authentication, ownership, confirmation, validation, and transaction orchestration. PostgreSQL remains authoritative. The adapter cannot write around the service layer or bypass approved appointment, slot, waitlist, audit, or idempotency rules.

The internal adapter interface marks the future boundary where an authorized EHR integration could be introduced. Such an integration would require separate credentials, security review, mapping validation, conformance testing, and approval. V1 uses no Epic credentials, production FHIR server, real patient data, or live hospital connection and makes no FHIR conformance, certification, endorsement, or production-interoperability claim.

## Safety, Authorization, and Audit Controls

- Synthetic data only and clear non-clinical demo labeling.
- Server-verified identity and ownership on every protected request.
- Explicit UI confirmation for appointment and waitlist mutations.
- Ten-minute pending-action expiration and five-minute waitlist-offer expiration.
- PostgreSQL `READ COMMITTED`, canonical row-lock order, uniqueness and check constraints, atomic transactions, and idempotency.
- Append-only appointment history committed with each appointment mutation.
- Durable outbox work committed with the originating transaction.
- Sanitized authentication, authorization, protected-read, and mutation audit events.
- No secrets, medical records, or sensitive conversation transcripts in prompts, JSON payloads, logs, notifications, or audits.
- Safe error messages that do not reveal another patient's records.
- Refusal of diagnosis, treatment, clinical triage, and unsupported policy claims.
- Separate staff and patient roles with least-privilege access.

The database and retention details are defined in [PostgreSQL Database Design](./database-design.md).

## Evaluation and Regression Strategy

Evaluation begins before model and voice integration.

### Backend truth tests

- Unit and PostgreSQL integration tests for validation, authorization, ownership, idempotency, and retention.
- Transaction and concurrency tests for double booking, slot holds, simultaneous offer acceptance, cancellation, rescheduling, session replacement, and retryable outbox work.
- Deterministic integration tests for synthetic Patient-like, Practitioner-like, Schedule-like, Slot-like, and Appointment-like mappings and proof that the adapter cannot bypass backend authorization or PostgreSQL transactions.
- Automated safety targets: zero unauthorized mutations, zero double bookings, zero hold breaches, and confirmation-before-write compliance across the test suite.

### Agent evaluation

- The approved 20 initial scenarios form the versioned regression baseline. Their backend, agent, RAG, voice, waitlist, and adapter portions are added in the phases defined by the evaluation strategy.
- Deterministic fake-model tests run in CI against tool scope, pending actions, safe errors, ambiguity, and multi-intent state.
- Real local-model evaluations measure task completion, tool selection, invalid calls, clarification behavior, and database final state.
- Prompt and tool changes rerun the regression suite and compare against the previous baseline.

### RAG and voice evaluation

- RAG tests measure source support, correct abstention, citation behavior, and prompt-injection resistance.
- Voice tests include ambiguous names and dates, transcription errors, retries, confirmation display, and typed fallback.
- Report p50 and p95 transcription, model, tool, speech, and full-turn latency from measured runs.

### Release evidence

- Expand to 60 to 100 synthetic scenarios by the final hardening phase.
- Report task success, tool correctness, unauthorized-change count, concurrency failures, groundedness, and regression trends.
- Set non-safety performance thresholds only after baseline measurements. Do not invent accuracy or latency claims.
- The public demo is not complete until an external user can complete a synthetic workflow without the developer's Mac running.

## Local Development and Cloud Deployment

Local development targets an Apple Silicon Mac with 16 GB unified memory:

- Next.js frontend.
- FastAPI backend.
- Docker PostgreSQL.
- Ollama compact-model baseline.
- Whisper.cpp and browser or system speech output.
- Pytest, deterministic agent tests, and local real-model evaluation.
- Synthetic FHIR fixtures and adapter integration tests.

The public demonstration targets:

- Vercel for the Next.js frontend.
- GCP Cloud Run for FastAPI.
- Neon serverless PostgreSQL for durable application state.
- A verified hosted inference adapter that can call approved tools.
- A verified browser or hosted speech path with typed fallback.
- GitHub Actions for deterministic tests, not costly real-model evaluation.

Provider compatibility, inference, speech, final cookie behavior, quotas, and cost remain explicit implementation-time verification gates. No provider is treated as working until it passes the relevant tests.

## Cost and Shutdown Strategy

- Develop locally first and provision cloud components only with explicit approval.
- Keep Cloud Run minimum instances at zero where compatible with the demonstration.
- Use connection pooling and provider limits appropriate for serverless workloads.
- Configure GCP budgets and alerts, while recognizing that alerts are not hard spending caps.
- Do not upgrade the GCP trial billing account to paid for this project without a separate decision.
- Verify current Vercel, Neon, inference, and speech limits and pricing before enabling the public demo.
- Avoid always-on compute and paid notification services.
- Record the final demo while approved trial resources are available.
- After the demonstration, disable or delete unneeded cloud services, database resources, credentials, secrets, scheduled jobs, and compute resources. Verify that no billable resource remains active.
- Preserve only synthetic, non-secret documentation and evaluation artifacts needed for the portfolio.

## Development Plan

The [Development Roadmap](./roadmap.md) is authoritative for phase order, progress, and estimates. The approved plan remains eight phases and 35 to 49 working days. Phase 0 design and decisions are complete. Phase 1 backend foundation is in progress.

## Final Deliverables

- A working public patient-access demonstration with text, voice, authenticated synthetic workflows, RAG, waitlists, and simulated handoffs.
- A separate limited-permission staff dashboard.
- A version-controlled GitHub repository with no secrets, real patient data, model weights, or private recordings.
- A clear README covering architecture, setup, local execution, testing, limitations, and live-demo status.
- Approved design documents and documented architectural decisions.
- Automated unit, integration, authorization, concurrency, and deterministic agent tests.
- A tested synthetic FHIR-like adapter demonstrating a controlled healthcare integration boundary.
- Versioned evaluation scenarios, results, failure analysis, and measured latency.
- A recorded walkthrough showing the product, backend state changes, safety boundaries, tests, and evaluation evidence.
- Cloud deployment and teardown documentation.
- A reproducible local fallback after the time-limited public deployment is removed.

## Synchronization Changes from Historical V3

The following later decisions supersede open or less-specific statements in V3:

- Neon is now the preferred hosted PostgreSQL provider; V3 left the provider open.
- Authentication now has approved session lifetimes, concurrency limits, lockouts, Argon2id requirements, CSRF controls, and retention rules.
- Protected mutations require explicit UI confirmation. Voice and model tool calls cannot authorize execution.
- Slots use `AVAILABLE`, `HELD`, and `BOOKED`, with soft removal through `removed_at` only while available.
- Appointments use only `CONFIRMED` and `CANCELLED` in V1. Pending and rescheduled are actions or history, not persisted statuses.
- Waitlist offers use `ACTIVE`, `ACCEPTED`, `DECLINED`, and `EXPIRED`, rather than the historical diagram's `PENDING` label.
- Waitlist priority, time preferences, one-active-offer rules, overlap prevention, database-time expiry, and retryable next-patient processing are now explicit.
- `SLOT_RELEASED` is produced by both cancellation and rescheduling and is recorded in the originating transaction.
- Staff access is narrower than the historical dashboard description and excludes patient waitlists, offers, secrets, medical records, and appointment mutations.
- Agent tools prepare protected actions only. A separate authenticated endpoint performs confirmation and mutation.
- The nine agent tools, response envelope, search limits, timeout behavior, confirmation-result event, handoff categories, and RAG passage limits are approved.
- The database now defines canonical lock ordering, immutable appointment history, idempotency, transactional outbox work, soft slot removal, and approved retention defaults.
- V1 now includes a minimal synthetic FHIR-like adapter behind the authoritative FastAPI and PostgreSQL service layer.

## Implementation-Time Verification

No unresolved Phase 0 architecture decision remains. Implementation must still verify Neon PostgreSQL and extension compatibility, hosted inference, browser speech, Vercel-to-Cloud Run cookie behavior, Argon2id cost parameters, hosted RAG storage and embeddings, connection behavior, cloud quotas, costs, and deployment configuration.

The synthetic FHIR-like adapter is implemented toward the end of Phase 1. EV-020 does not block the Phase 1 scheduling exit criterion, but it must pass before the final public release.
