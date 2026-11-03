# Phase 3.3: secure browser confirmation

Status: **complete for the local prototype and verified locally** (synthetic data only). The voice agent can prepare a
booking review; the patient confirms or declines it in a browser; the voice agent announces a
booking only after the database shows the confirmation service committed it.

```
voice request ─▶ review prepared ─▶ browser link ─▶ patient signs in ─▶ Confirm / Decline
   (LG-5F)        (voice session)    (single use)     (own PIN, own        (existing endpoint)
                                                       session)                  │
voice acknowledgement ◀── verified outcome read from PostgreSQL ◀───────────────┘
```

## What was built

| Piece | Where |
|---|---|
| Handoff rules (create, claim, status, supersede, decline) | `backend/app/services/review_handoffs.py` |
| Table for the single-use tickets (SHA-256 of the code only) | `ReviewHandoff` model, migration `0004_review_handoffs` |
| Browser routes: claim, current review, decline | `backend/app/api/routes/review_handoff.py` (`/me/voice-handoffs/…`) |
| The page (one HTML file, one script, one stylesheet) | `backend/app/web/review/`, served at `/voice-review` by `routes/review_page.py` |
| Voice side: link, wait, verified announcement | `backend/app/voice/browser_review.py`; loop changes in `voice_agent.py` |

There was no Next.js frontend in the repository, so the page is plain HTML served by the backend. That
keeps it on an origin the API already allows (cookie stays same-origin, Origin and CSRF checks apply
unchanged), needs no build step, and adds no dependency.

## How the secure handoff works

A pending action (a "review") belongs to the one session that prepared it, and retirement, expiry
and ownership all rely on that. So the handoff **moves no credential and reassigns nothing**:

1. **The voice session asks for a ticket.** `create_handoff` finds the session's one live review and
   stores a row naming the patient, the voice session and that review, with a random 256-bit code of
   which only the SHA-256 is kept. It lasts at most 5 minutes, never longer than the review. A newer
   ticket replaces any earlier one.
2. **The link carries the code in the URL fragment**
   (`/voice-review#h=<code>&p=<patient code>`): never sent to the server, never in a log line. The
   page reads it once and removes it from the address bar. No token, PIN or capability to confirm
   is in the URL; the patient code is an identifier, not a secret.
3. **The browser signs in on its own**, with the patient's PIN in a password field, like any other
   browser. The ticket alone does nothing: it is a *claim*, redeemable only by an authenticated
   session of **the same patient**, once.
4. **Claiming** (`POST /me/voice-handoffs/claim`, session cookie + allowed Origin + CSRF token)
   atomically reserves the ticket, then prepares an equivalent review **in the browser's own
   session** through the ordinary prepare service (the slot is checked again), and retires the voice
   session's review. One live review exists at any time, and it can be confirmed only by the session
   that owns it.
5. **Confirm or Decline.** Confirm is the existing `POST /me/appointment-actions/{id}/confirm`,
   unchanged (Idempotency-Key, CSRF, single transaction). Decline retires the review.
6. **The voice session learns the outcome from the database only** (`handoff_status`). A review is
   reported COMMITTED only when the confirmation service marked it `CONSUMED`, in the same
   transaction that changed the appointment. Nothing the browser, the model or the patient's words
   claim counts.
7. **Only then** is the conversation told (as the trusted event the orchestrator is built to
   receive) and the booking announced, with the doctor, clinic, day and time from the verified
   record. Declined, expired, replaced or unavailable: "nothing was changed".

If the patient speaks again, or the voice agent exits, the ticket and any review it opened in the
browser are retired first (`supersede_open_handoffs`), so a review the conversation has left behind
stops working, as it already does within one session.

## Security properties (each has tests)

| Requirement | How |
|---|---|
| Browser cannot confirm another patient's or session's action | Confirm is unchanged and session-scoped; the browser holds only its own review. Claim needs the same patient. |
| A leaked link is useless alone | Needs the patient's PIN-authenticated session, once, within minutes. All failures (unknown, used, expired, other patient, replaced, voice session gone, review no longer live) return one identical 404. |
| No credentials in URLs or logs | Fragment only; code stored only as SHA-256; audit events carry no code; the page uses no storage APIs, clears the PIN field after submit, and writes text with `textContent` only. |
| Existing protections preserved | Session ownership, expiry (10 min review lifetime), Origin and CSRF checks, idempotency, patient scoping: all reused, none altered. |
| Voice/LLM/CLI can never commit | `browser_review.py` has no path to the commit service (test); the only conversation event it sends is a commit, only after the database says so (test). |
| No false booking announcement | Every other outcome says nothing was changed (tests assert no booking words). A voice session that ended before it could check says so honestly. |
| Retries and duplicates | Same Idempotency-Key replays; a second key gets `CONFIRMATION_CONSUMED`; presenting a spent ticket again returns the same browser's own review, never a second one. |
| Strict page policy | `Content-Security-Policy` (same-origin script and style only, no framing), `no-store`, `no-referrer`, `nosniff`. |

New audit events: `REVIEW_HANDOFF_CREATED`, `_CLAIMED`, `_SUPERSEDED`, `_REJECTED` (never the code).

## Run it

Prerequisites: PostgreSQL running (`docker compose up -d`), Ollama serving `qwen2.5:7b`, the Whisper model
at `~/models/ggml-base.en.bin`, and `.env` set up as before.

```bash
# Run everything from the repository root: the settings read .env from the folder you run in.

# 1. Apply the new migration (adds the review_handoffs table)
uv run alembic -c backend/alembic.ini upgrade head

# 2. Terminal A: the backend, which also serves the review page
PYTHONPATH=backend uv run uvicorn app.main:app --port 8000

# 3. Terminal B: the voice agent (LG-5F is the default)
PYTHONPATH=backend uv run --group voice python -m app.voice.voice_agent \
  --model ~/models/ggml-base.en.bin --patient-code PT-1001
```

If `uv run --env-file .env` complains about the first line of `.env`, quote the name:
`APP_NAME="VoxVisit AI Backend"` (a value with spaces must be quoted for that parser).

Then speak a booking request. When a review is prepared the agent prints a link and opens your
browser. Sign in there (patient code is prefilled; enter the seeded PIN), check the details, and press
**Confirm** or **Decline**. The voice agent, which waits meanwhile, speaks the result. Press Enter
in the terminal at any time to cancel the review and speak instead.

Options: `--no-browser` shows the link without opening it; `--review-url` (or
`VOXVISIT_REVIEW_BASE_URL`) sets where the page is served (default `http://localhost:8000`, which must be
one of the backend's `ALLOWED_ORIGINS`). LG-5F is selected by default. Use `--agent legacy` only
when the original text-agent path is needed for compatibility or comparison.

## Supervised local pilot

A supervised microphone pilot completed 14 timed turns on the local synthetic-data prototype. One
voice-to-browser booking completed for Riverbend Family Health with Dr. Elena Vasquez (Pediatrics)
on November 5, 2026 at 9:00 AM (`America/Denver`). The spoken review and browser review matched.
The appointment was committed only after authenticated browser confirmation, and the voice agent
announced the committed result afterward.

The local timing log recorded transcription, agent, first-spoken-word, speech, and machine-turn
durations. P50 and p95 below use nearest-rank percentiles. With 14 observations, p95 is the highest
observed value and is useful only as an initial local baseline.

| Measurement | P50 | P95 |
|---|---:|---:|
| Speech-to-text | 0.38 s | 0.48 s |
| Agent processing | 2.75 s | 7.11 s |
| Time to first spoken word | 3.10 s | 7.60 s |
| Spoken-response duration | 13.30 s | 40.10 s |
| Machine turn (speech-to-text through completed speech) | 18.38 s | 41.97 s |
| Full observed turn (including recording and operator review) | 30.26 s | 59.76 s |

Observed limitations: quiet-office speech caused transcription misrecognitions and corrections; one
recording was not recognized; one spoken response lasted about 40 seconds; and one confirmation
announcement was delayed. The 14-turn, single-machine sample does not establish latency targets,
reliability targets, or production readiness.

## Limits

- Local, single-machine design: the voice agent and the backend share the database; the voice side polls
  every 1.5 s (no push).
- The page needs the patient to type the patient code and PIN once per browser session. That is
  deliberate (the ticket must not be a login) and is the project's stated identity model.
- Only the voice session's single live review can be handed off; one ticket is open at a time.
- Voice usability still needs broader microphone testing, shorter non-availability responses, and a
  faster confirmation-result experience. These are usability improvements, not exceptions to the
  UI-only confirmation boundary.
- There is no automated browser test in the suite. The page is covered by tests of what the server
  sends and of the script's source (no storage APIs, no inline code, `textContent` only), and during
  development its real script was run in Node against a running server. The supervised manual pilot
  also completed one browser-confirmed booking, but broader browser compatibility has not been tested.
- Not covered here: production hosting, HTTPS and `Secure` cookies, rate limiting of claim attempts beyond
  the existing authentication limits, and a patient dashboard.
