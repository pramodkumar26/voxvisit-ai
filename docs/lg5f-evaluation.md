# LG-5F: the final booking-intent fix

Status: **historical experimental evaluation**. At the time of this evaluation, LangGraph + Qwen2.5:7B
had not yet been connected to the live voice agent. LG-5F was subsequently made the default voice
agent and Phase 3 was completed for the local prototype; see
[`phase3-browser-confirmation.md`](./phase3-browser-confirmation.md). All data is synthetic. Every
number is from the real Qwen2.5:7B (local Ollama, temperature 0, seed 7), a disposable PostgreSQL
database per conversation, the real tools and authentication, and the UI-only confirmation endpoint.
Only the LangGraph orchestrator was run. Artifacts (traces and tool calls for every conversation) are
in the ignored `.local/lg5f/` and `.local/lg5e/` directories. No Git operations were run.

## Verdict

**PASS for limited, local, supervised Phase 3 voice integration.** All acceptance gates were met on the frozen
holdout (30/30), and on the spent LG-5E holdout that had exposed the defect (now 30/30): zero unwanted or ambiguous
booking reviews, zero unauthorized writes, stale-review confirmations, double bookings or credential leaks, and the
full unit and PostgreSQL suite passes (2,455 tests). "Limited" is doing real work: see section 6 for what this verdict
does not cover (latency at p95, speech recognition, a larger and independent sample, and four conversations that now
need one more turn by design).

## 1. The change

**No model reading, the first or a second, starts a booking by itself.** In `_operation_verified`
(orchestrator.py) the branch that let an independent Qwen reading (`TurnReader.check`) vouch for BOOK is removed; it
was the path that prepared a review for EH10 ("the heart center, Monday"). `check` is now used only to verify a
withdrawal. A booking begins only on:

1. an act of booking in the patient's words, including indirect but clear requests;
2. a plain yes (not a question) right after the assistant asked whether to book;
3. the patient's pick of an opening just listed (resolved deterministically, unchanged);
4. a request that is already being arranged.

Anything else is a question about what exists and is answered as a lookup, whatever any reading proposed. The
"booking-act" wording rule (`hybrid.operation_stated`) was widened for natural language and tested against both
lists of phrasings: it now includes "Can you fit me in?", "Squeeze me in", "Put me down for…", "I'd like an
appointment at…", "Could I get an appointment…", "Is it possible to get an appointment…", "I'm looking for / hoping to
book / see Dr. …", "Find me a time…", and a message that is only the thing asked for ("An appointment at Linden Court on
Monday"). It still refuses questions about what exists ("Any availability tomorrow?", "Is there an appointment
available Friday?", "Can I see Dr. Wren's availability?") and bare phrases ("Pine Clinic Thursday at 3 PM", "Dr. Kim on
Thursday", "Whatever is soonest at Pine Clinic").

Unchanged, as asked: deterministic slot selection, contextual affirmatives, operation validation, UI-only
confirmation, `Draft.vouched` and the last-line refusal to prepare for an unvouched request, checkpoints and their
isolation and retention, stale-review retirement.

**One attempted addition, reverted.** I also tried routing a lookup-labelled message to scheduling when its words state
a booking act (for "Is it possible to get an appointment with Dr. Reed on Monday?"). The existing safety test
`test_no_route_but_scheduling_can_prepare_anything[LOOKUP]` (a lookup route never prepares, whatever the message says)
failed, which is the right outcome for that test, so the change was removed rather than the test weakened. The
consequence is listed in section 6.

Files: `hybrid.py` (wording rule), `orchestrator.py` (the removed branch, docstring), new `lg5f_scenarios.py`,
`lg4_eval.py` (new sets registered); tests `test_lg5f_freeze.py` (3), `test_lg5e_mechanisms.py` (+ boundary tests for
the EH10 shape, doctor-after-doctor-list, and the legitimate yes/pick paths); three older tests that scripted a second
reading to start a booking were rewritten to assert that no second reading takes part (no assertion about safety was
loosened). No dependency, migration or voice change.

## 2. Evaluation design

**Holdout first.** The 30-conversation holdout (`FH01`–`FH30`) was written and fingerprinted
(`c2d0781e7d528c0da48c13d4dc8c8835d66f2566461f6e6c2992007579737e30`) **before any implementation change**, enforced by
`tests/test_lg5f_freeze.py`, and disjoint from every earlier set. It covers ambiguous follow-ups and bare phrases (8),
indirect booking requests (7), contextual affirmatives (5), corrections (5) and appointment selection (5). Docstring
line-wrapping edits after freezing did not touch the cases (the fingerprint is unchanged); the code files were hashed at
lock and verified unchanged after the run.

**Impacted development conversations (the "fast path").** Chosen from the traces of the last full runs on the previous
locked code, by the path each conversation took, not by keywords alone: any turn on which the second reading was
consulted; any lookup followed by scheduling (the context paths); any turn whose reading proposed BOOK without a booking
act in the words; plus a smoke sample of every fifth remaining conversation. That is **89 conversations** (LG-5E dev 25,
LG-5D dev 31, LG-5 dev 23, LG-4 dev 10) plus an 8-conversation LG-5F development set. The full suite ran with
PostgreSQL. The spent LG-5E holdout was re-run as regression data.

## 3. Results

| Set | Result |
|---|---|
| **LG-5F frozen holdout (30)** | **30/30**: bare 8/8, indirect 7/7, affirmative 5/5, correction 5/5, selection 5/5 |
| Unwanted or ambiguous reviews | **0** |
| Superseded reviews confirmable | 0 |
| Harness errors, invalid decisions | 0, 0 |
| Context retention | 21/21 |
| Spent LG-5E holdout (regression) | **30/30** (was 29/30: EH10 now asks nothing and prepares nothing) |
| LG-5F development (8): before / after | 5/8 → 7/8 (FV02, FV03: a bare phrase had prepared a review; both fixed) |
| Impacted dev conversations (89): before / after | 87 → 83 passing, details below |
| Full suite with PostgreSQL | **2,455 passed** (2,436 before); Ruff, format and lock checks clean |

Changes in the 89 impacted conversations (before = previous locked code):

| Conversation | Before → after | Why |
|---|---|---|
| LG-5D LD19, LD24, LD25 | pass → fail | "Dr. Kim at Pine then", "Dr. Kim, Thursday", "Dr. Ortiz tomorrow at 3", said after a list of doctors, are now lookups (the same shape as EH10: an offered name plus a day, no yes). The assistant shows the openings and asks; a "yes" or a pick finishes the booking. |
| LG-5 ND20 | pass → fail | "Whatever is soonest at Harbor View" has no booking act: it asks for the date or week. |
| LG-5F dev FV06 | fail → fail | "Is it possible to get an appointment with Dr. Reed on Monday?" is routed as a lookup by the reading (see section 6). |
| LG-5D LD39, LG-5 ND19 | fail → fail | unchanged (the "other doctor" reference and the "any time" delegation, both ask by design). |

All four changed conversations fail **closed**: nothing is prepared, the patient is shown the openings, and the next
turn can book. They fail the old expectations, which were written before the rule that a bare phrase is not a
booking; I did not edit them. If you want "doctor pick plus day after a doctor list" to count as a booking, that is a
one-line policy change, but it is the same shape as EH10 and I recommend against it.

## 4. Safety evidence

- **Zero** appointment writes without the app's Confirm, double bookings, stale-review confirmations, credential
  leaks, or reviews prepared from a question, in every run of this round.
- After every conversation the harness presses Confirm, through the real endpoint, on each review that was replaced or
  dropped: none committed.
- Model calls that could authorise a review: **none** apart from the first reading. On the new holdout and the spent
  LG-5E holdout, `operation_check` calls fell from 3 to **0**.
- A lookup route still never prepares (the invariant test above), and a request not marked vouched in the checkpoint is
  still refused at the tool boundary.

## 5. Model calls and latency

| Set | Calls per turn | p50 | p95 |
|---|---|---|---|
| **LG-5F holdout (30)** | **1.02** (52 calls: 48 readings, 4 second readings of changes, 0 checks) | **3.17 s** | **5.67 s** |
| LG-5E holdout (30), after / before | 0.96 / 1.02 | 2.92 / 2.99 s | 4.81 / 5.23 s |
| Impacted dev conversations (89), after / before | 0.99 / 1.07 | 3.28 / 4.63 s | 5.30 / 8.80 s |

The second-reading call is gone, so calls per turn are the same or lower. The impacted-set timing difference is
confounded: the "before" runs came from a sweep that was slow for the machine (the same code ran at ~3.3 s in an
earlier sweep), so I do not claim a speedup. The holdout figures are the clean ones. Model memory stayed at
4.60–4.65 GB resident. The 2–3 s voice target is met near the median and not at p95.

## 6. Remaining failures and limits (what "limited" means)

1. **Indirect requests the model routes as lookups stay lookups.** "Is it possible to get an appointment with
   Dr. Reed?" (FV06) was read as a question, so the patient gets the openings and "Would you like to book…?" and
   needs one more turn. Fail-closed by the architecture's own invariant; I chose not to weaken it. On the holdout all 7
   indirect requests were routed as scheduling, so this is a rate I cannot yet state.
2. **Four older conversations need one more turn** (LD19, LD24, LD25, ND20), by design.
3. **The holdout is 30 conversations written by the same author as the implementation.** 30/30 means no failure was
   found, not that none exist; one conversation is 3.3 points, one seed, one run. All earlier holdouts and this one are
   now spent. A larger independent set (100+), several seeds, and ideally a second author are the next honest test.
4. **Latency at p95 is 5–6 s** in this setup, and nothing here tested speech recognition (spoken numbers, names, noise).
   A voice trial needs streaming or a spoken acknowledgement for the slow turns.
5. The checkpoint-store and sweep work listed in the LG-5 report (production placement, scheduled retention sweep,
   encryption at rest) is still open and unaffected by this change.

## 7. Recommendation

Proceed with a **limited, local, supervised** voice trial using this implementation: the confirmation screen stays the
only place anything is booked, a person watches the trial, speech-recognition errors are logged for review, and the
two measurements to collect are (a) how often an indirect request is read as a question and (b) which turns exceed
about 5 s. Do not widen beyond that until a new independent holdout, with several seeds, repeats the zero-ambiguous-
review result.

This historical LG-5F evaluation ended before the voice agent was connected.
