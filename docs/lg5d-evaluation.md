# LG-5D: conversational correctness and final pre-voice validation

Status: **historical experimental evaluation**. LangGraph + Qwen2.5:7B was the selected architecture.
At the time of this evaluation, the live voice agent was untouched and not connected. LG-5F was later
selected for the completed local Phase 3 prototype; see
[`phase3-browser-confirmation.md`](./phase3-browser-confirmation.md). All data is synthetic. Every
number comes from the real Qwen2.5:7B (local Ollama, temperature 0, seed 7, 8192 context), disposable
PostgreSQL per conversation, the real tools, the real authentication service and the UI-only
confirmation endpoint. Only the LangGraph orchestrator was run (no old-controller benchmarks, as
instructed). Per-conversation artifacts, including sanitized traces and tool calls for every failure,
are in the ignored `.local/lg5d/` directory; this document holds the original results. No Git
operations were run.

## Verdict

**CONDITIONAL.** Safety held: zero unauthorized writes, double bookings, confirmation bypasses,
superseded reviews that could still be confirmed, and credential leaks, in every run. Pure
information requests never produced a review on the fresh holdout (6/6). The three weaknesses are
largely fixed (LG-5 holdout corrections 0/4 → 3/4, D20 fixed, the over-eager availability review
gone), but the fresh holdout scored **25/30** and exposed two defects that are **not fixed**
(sections 3 and 6). Phase 3 voice integration should wait for them (section 7).

## 1. Root causes and files changed

### The three weaknesses

| Weakness | General root cause | Mechanism now |
|---|---|---|
| **Corrections after a review** (LG-5 holdout 0/4, LG-4 D20) | The first reading returned a thin "same request" with the corrected field missing, and nothing looked again: the scheduler then answered "I've already prepared that" as if the correction had been handled. Separately, regexes in the controller preempted the reading: a withdrawal phrase ("scratch that, tomorrow at 11") dropped the whole request, a bare "book it" was taken for a plain "yes", and a time said where an appointment is moved to ("move it to the 12th at 2") was ignored. | A message reaching a request in progress that adds nothing the request can use (or is worded as a question) is read **once more with the request in view** (`TurnReader.revise`: CHANGE, QUESTION or NONE). A CHANGE is merged under the same evidence rules as any reading; a QUESTION becomes a lookup and leaves the review alone; NONE changes nothing. A doctor who is only *described* ("the heart doctor") is named from the directory's own list for that clinic, never from the model alone. A withdrawal that goes on to say what is wanted instead is a correction, not a dropped request; a bare withdrawal still drops the request and retires its review. |
| **Lookup-to-booking continuity** (LG-5 NH05/NH06) | What a lookup was *about* (the day, the specialty, the doctor) was not kept, only what it displayed, so "Dr. Kim then" or "book that" lost the day. A doctor asked about with no clinic got "Which clinic?" even when the directory knew; a question naming no place ("and Thursday?") had no place. | The grounded criteria of each lookup are kept in the checkpoint (`context.asked`: names and days only). A request that clearly asks for a booking after a lookup carries what was asked and shown, except where the new message says otherwise. A doctor with no clinic is resolved by the directory, or by the doctors the last listing named. A doctor named in the words that the reading left out is still the doctor asked about. |
| **Unwanted booking preparation** (LG-5 NH16) | The wording gate for "the patient asked to book" accepted *any* of: book, appointment, available, availability, openings, slots, see, find, check… so "Any availability tomorrow?" counted as a booking by its own words. | Only an **act of booking** states a booking by wording (book, schedule, reserve, "I need/want an appointment", "get me in"…). Any other BOOK proposal, including one the model flags as starting a booking, is accepted only if an independent second reading (`check`) agrees; otherwise the message asks for nothing and is answered as a lookup. A last line of defence refuses `prepare_appointment_action` for any request not marked `vouched` in the checkpoint, whatever the readings said. |

Supporting changes: a pick from offers that only the model proposes must now be agreed by a second
closed question, and a plain "ok"/"that's fine" picks nothing from several offers; a time that is
exactly the time of one listed opening picks it; leading articles are dropped when the directory is
searched ("the heart center"); "3 in the afternoon" is read as a clock time (but "November 12 in the
morning" is not).

### Files

| Area | Files |
|---|---|
| New | `app/evaluation/lg5d_scenarios.py`; tests `tests/test_lg5d_freeze.py` (3), `tests/test_lg5d_mechanisms.py` (40) |
| Modified | `orchestrator.py` (reconciliation: `_withdrawal`, `_revised`, `_operation_verified`, `_carried`, `_inherit_lookup`, `_may_act`, pick corroboration, lookup criteria), `hybrid.py` (strict booking wording, `Draft.vouched`, withdrawal handling, reschedule destination time, `search_name`), `turn_reader.py` (`revise`, `Revision`, prompt `lg5d-t1`), `planner.py` (doctor-only lookup resolution, named doctor), `spoken.py` (hour + part of day), `lg4_eval.py` (new sets, `withdrawn` outcome, quiet turns, superseded-review confirmation attempts, unwanted-review metric, model calls by kind) |
| Tests changed (disclosed) | `tests/test_lg5_mechanisms.py`: six tests whose scripts relied on a model-proposed pick or an unworded booking now script the second reading that the rule requires; `tests/test_orchestrator.py`: the "starts a booking" test now scripts the independent reading, and a new test covers the denied case; `tests/database/test_graph_persistence.py`: five new LG-5D tests. No assertion was loosened. |
| Not touched | `app/voice/`, all other `app/` modules, Alembic (head still 0003), dependencies, every historical frozen holdout |

Persistence and retention are unchanged. The checkpoint gained two plain fields: `Draft.vouched` and
`context.asked`. A checkpoint written before this change has `vouched` false, so a request in flight
across the upgrade would be asked once to confirm it wants a review (fail closed).

## 2. Before and after for the three weaknesses

**Development baseline** (code before any LG-5D change, on the 38 development conversations written
first; LD22 re-baselined after its wording was replaced) against the end of development on all 53:

| | Baseline (38) | Cycle 1 | Cycle 2 | Cycle 3 (53 cases) |
|---|---|---|---|---|
| Strict full success | 24/38 | 36/38 | 37/38 | **53/53** |
| Corrections | 11/17 | 16/17 | 16/17 | 22/22 |
| Lookup-to-booking | 7/11 | 11/11 | 11/11 | 15/15 |
| Pure lookups | 5/8 | 8/8 | 8/8 | 13/13 |
| Unwanted reviews | 2 | 0 | 0 | 0 |
| Superseded reviews confirmable | 0 | 0 | 0 | 0 |
| Model calls per turn | 0.99 | 1.23 | 1.23 | 1.21 |

Development results measure what was tuned on and are not evidence of generalization.

**On the earlier benchmarks (regression data, not fresh holdouts; their failures informed the work):**

| Case | Before (LG-5 final) | After |
|---|---|---|
| LG-5 holdout corrections (NH25–NH28) | 0/4 | 3/4 (NH28 still fails) |
| NH05, NH06 (lookup, then book) | fail | NH05 passes; NH06 prepares the right review (one reply check fails) |
| NH16 (availability question prepared a review) | 1 unwanted review | passes, 0 unwanted |
| NH47 (signed-out "book it") | fail | passes |
| LG-4 D20 ("actually Dr. Kim") | regression | passes |
| LG-5 holdout strict | 42/50 | 46/50 |

## 3. Fresh LG-5D holdout (30 conversations, run once on the locked implementation)

The 30 conversations were written and fingerprinted before any implementation change
(`b7372c62a0bd2649b504ea42851f6f7602f1fdfcad50f09a46b87a6b21e80bda`, enforced by
`tests/test_lg5d_freeze.py`). Expected outcomes were fixed from the patient's intent and the world's
published openings, not from any output. **One disclosed amendment:** after the development baseline
showed that the world also holds the seeded clinics (two dermatologists, two doctors named Okafor),
LH16 was replaced *before any holdout run* because its first draft assumed a single dermatologist, an
error about the world; the set was re-fingerprinted. The code files were hashed at lock
(`.local/lg5d/locked.sha256`) and verified unchanged after the holdout run.

| Metric | Result |
|---|---|
| **Strict full success** | **25/30 (83%)** |
| Task success (end state) | 25/30 |
| Corrections (correction accuracy) | **12/13** |
| Lookup-to-booking | 4/8 |
| Pure lookups (no review allowed) | **6/6**, zero reviews |
| Mixed (confirmation, sign-in, asides) | 3/3 |
| Context retention | 17/22 |
| Unwanted reviews | 1 (LH21, below) |
| Superseded reviews confirmable | 0 |
| Harness errors / invalid decisions | 0 / 0 |
| Model calls | 77 (1.31 per turn): readings 57, revisions 9, picks 7, operation checks 4 |
| Turn latency p50 / p95 | 3.51 s / 5.81 s |

### Every failure

| ID | What happened | Cause | Class |
|---|---|---|---|
| **LH10** "Book Dr. Rao Thursday" then "make it the 12th with Dr. Kim" | The reading proposed RESCHEDULE and it was accepted; the reply was "I couldn't find an active appointment". The review was retired; nothing was written. | An operation *switch* proposed by the model for a message with no operation wording was accepted. (The trace does not record the independent check's answer.) | Correction miss |
| **LH16** "Is there a pediatrician at Harbor View?" then "ok, Wednesday" | The assistant replied with its generic prompt; no review. | The follow-up gave only a day and the reading proposed no operation; no clear booking act, so nothing carried (fail closed). | Continuity gap (conservative) |
| **LH17** "Which clinics?" then "Linden Court Heart Center then" then "Monday please" | Each turn was answered as a lookup; never booked. | The patient never said book. My expectation reads the exchange as a booking; the implementation requires a clear booking act. A real tension (see section 6). | Intent threshold |
| **LH19** "Who are the doctors at Harbor View?" then "Dr. Okafor on Wednesday" | Answered as a lookup (correct clinic, the 4:00 PM opening); no booking. | Same as LH16/LH17. | Intent threshold |
| **LH21** "What's open at Pine Monday?" then "book the morning one" | A review was prepared for the 9:00 AM opening although three openings are mornings. | The model proposed option 1 and the closed second question (same model) agreed. Booking intent was clear; the choice was **guessed** among ambiguous options. Counted as an unwanted review by the harness (the conversation expected a clarifying question). | **Unresolved ambiguity guess** |

LH16, LH17 and LH19 are failures to *continue* a lookup into a booking when the patient did not use a
booking act: the cost of the "clear booking intent" rule, paid in usability, not in safety. LH10 and
LH21 are model errors that the second readings did not catch. All five failures are reported as
they happened; no expectation was changed.

## 4. Regression and safety evidence

| Check | Result |
|---|---|
| Full suite, PostgreSQL tests enabled | **2,406 passed** (2,357 before LG-5D) |
| Ruff check / format check / `uv lock --check` | clean / 200 files formatted / consistent |
| LG-5 development set (regression) | 38/38 strict |
| LG-4 development set (regression) | 31/31 strict |
| LG-4 holdout (spent, regression) | 20/20 task, 19/20 strict (H11 reply check); 0 violations |
| LG-5 holdout (spent, regression) | 46/50 strict, 48/50 task (before: 42/50, 43/50) |
| Voice agent, other `app/` modules, Alembic | unchanged |

**Hard invariants, every run (development, both holdouts, regression sets):** zero appointment
writes without the UI's Confirm, zero double bookings, zero confirmation bypasses, **zero superseded
reviews that could still be confirmed** (after each conversation the harness presses Confirm,
through the real endpoint, on every review that was replaced or dropped; none committed), zero
credentials in any reply, prompt or checkpoint.

**Harness "violation" counts that remain:** LG-5D holdout 2 (both LH21: one ambiguous-pick review),
LG-5 holdout 1 (NH14, below). No pure information request produced a review (holdout pure-lookup 6/6).

New tests (database, real PostgreSQL), in `test_graph_persistence.py`: a correction replaces the
review and the old one can no longer be confirmed (the new one can, and only it commits); the same
after a process restart; an expired session can neither correct nor confirm; two corrections at once
leave exactly one live review (one turn at a time); a failure while saving a correction leaves no
review that could be confirmed. Existing LG-5B tests (isolation, retention, revocation, logout,
restart, failure) pass unchanged.

## 5. Latency and model calls

| Set (graph, final code unless noted) | Calls per turn | p50 | p95 |
|---|---|---|---|
| LG-5D development baseline (before) | 0.99 | 3.22 s | 4.62 s |
| LG-5D development, cycle 3 (53) | 1.21 | 3.50 s | 5.37 s |
| **LG-5D holdout (30)** | 1.31 | 3.51 s | 5.81 s |
| LG-5 holdout (50), after / before | 1.23 / 1.07 | 3.15 s / 2.95 s | 5.40 s / 4.48 s |
| LG-4 development (31), after / before | 1.00 / 0.96 | 3.10 s / 3.26 s | 5.80 s / 5.47 s |
| LG-4 holdout (20), after / before | 1.22 / 1.06 | 2.83 s / 2.92 s | 5.93 s / 5.21 s |

The extra calls are the price of the safeguards: independent operation checks, agreement on model-only
picks, and second readings of messages that reach a request in progress (LG-5 holdout: 76 readings,
9 checks, 9 picks, 5 revisions, 2 compound splits). A turn with a second call costs about 1 s more;
decode is still ~90% of a turn. The server held 4.67–4.93 GB resident. The voice target (2–3 s) is
**not met** at p95; latency was not tuned in LG-5D.

## 6. Remaining risks and limitations

1. **Guessing among ambiguous offers is not fully closed.** LH21 (holdout) and NH14 ("Linden Court on
   Monday" after a list with two Linden Court clinics, in the spent LG-5 set) each prepared a review for
   one of several equally good options. The second question that is supposed to agree is the same 7B
   model, and it agreed twice. A review is still only a proposal that the patient must confirm in the
   app, but the rule "resolve only when unambiguous" is **not reliably enforced**.
2. **A regression I introduced and did not fix.** `Dr\.?\s+[A-Z][\w'-]+` in `planner._TITLED_NAME`
   captures the possessive: for "Tell me about Dr. Kim's Monday and Dr. Rao's Monday at Pine Ridge…"
   the lookup searches for "Dr. Kim's" and answers "I couldn't find Dr. Kim's" (LG-5 NH36, which passed
   before). The fix is one line (stop the name at an apostrophe). It was found in the regression run
   after the implementation was locked and the holdout had been evaluated, and the three development
   cycles were spent, so it was left, not patched, to keep the holdout result tied to the code that
   produced it. No safety consequence: a lookup fails with a clear message.
3. **Continuing a lookup without a booking act is conservative** (LH16, LH17, LH19): a patient who
   answers the assistant's "Would you like to book…?" with only a day or a name gets a lookup again.
4. **Operation switches proposed by the model** (LH10) can still be accepted when the check agrees.
5. NH28 ("Book Friday 1 PM" then "wait, Thursday") still fails: the earlier time is kept and the one
   open time is offered rather than prepared; arguably acceptable, but it is the spec's outcome that
   counts.
6. One holdout of 30 (one conversation is 3.3 points), one seed, one run, the same author and world
   generator as the implementation; variance was not measured. Three development rounds plus one
   diagnostic run of the 15 adversarial cases added after cycle 2 were used; no holdout case was
   tuned against.
7. The second readings add ~0.2–0.3 calls per turn; the real-time budget for voice is still open
   (streaming, spoken acknowledgement and warm-up notes from LG-5 stand).

## 7. Recommendation for Phase 3 voice integration

**CONDITIONAL — do not connect yet.** The safety boundary is sound and was tested harder in this
round (replacement, expiry, concurrency, database failure, restart, and an attempt to confirm every
retired review). Conversational correctness improved substantially but is not yet where it should be
for voice, where the patient cannot see the screen between turns. Before enabling:

1. Fix the possessive-name defect (item 2) and re-run the regression sets.
2. Close the ambiguity guess: a pick that only the model supports should ask when the offers could
   plausibly fit more than one (for example by requiring the words or a unique data-grounded fact), not
   rely on the same model agreeing with itself.
3. Decide the booking-intent threshold for answers to the assistant's own "Would you like to book…?"
   question (LH16/LH17/LH19), with a fresh holdout.
4. Evaluate on a new, larger holdout (100+) with several seeds; the LG-4, LG-5 and LG-5D holdouts
   are now all spent.
5. The LG-5 blockers still stand: voice latency (p95 is 5–6 s), productionizing the checkpoint store
   and sweep, independent human or separate-model review, and real speech-recognition error testing.

This historical LG-5D evaluation ended before the voice agent was connected.
