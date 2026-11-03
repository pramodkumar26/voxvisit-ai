# LG-5: conversational robustness, PostgreSQL checkpoints, and measured latency

Status: **historical experimental evaluation**. At the time of this evaluation, LangGraph was not
connected to the live voice agent; nothing here switched the live model or touched `app/voice/`.
LG-5F was later selected for the completed local Phase 3 prototype; see
[`phase3-browser-confirmation.md`](./phase3-browser-confirmation.md). All data is synthetic. Model:
Qwen2.5:7B via local Ollama (temperature 0, seed 7, 8192 context) for every number below.
Per-conversation artifacts (replies, tool calls, per-turn timings, traces) are in the ignored
`.local/lg5/` directory; this document holds the original results. No Git operations were run (the
directory is not a repository).

## 1. Summary

| Question | Answer |
|---|---|
| Does the graph handle conversations it was not tuned on? | On a **frozen, fresh 50-conversation holdout**: graph **42/50** vs previous controller **20/50** (strict full-success metric). Eight graph failures remain, five of them one mechanism (corrections after a prepared review). |
| Safety | **0 unauthorized writes, double bookings, confirmation bypasses, credential leaks** in every run (development, holdout, LG-4 regression, tests). The harness flagged one *over-eager review* (prepared, never confirmable without the patient) in the holdout; see section 3. |
| Is persistence real? | Yes. PostgreSQL checkpointer (`langgraph-checkpoint-postgres`) with a pooled connection, strict deserialization, per-turn backend authentication, 12 h idle retention, deletion at sign-out. 23 PostgreSQL tests cover restart, expiry, logout, revocation, isolation, concurrency, DB failure and cleanup. |
| Latency | Dev workload graph turn **p50 4.48 s -> 3.42 s, p95 6.36 s -> 6.02 s**, model calls per turn 1.30 -> 1.11, with accuracy unchanged. Holdout graph p50 **2.95 s**, p95 **4.48 s**. The "about 2-3 s" voice target is **not met** for p95 or for two-call turns (p50 4.2 s, p95 8.7 s). |
| Regression | Full suite **2,357 passed** (was 2,234 at LG-4), Ruff and format clean, lockfile consistent, Phase 2 smoke 15/15. LG-4 development set shows **one regression** (D20, section 6). |
| Verdicts | **LG-5A CONDITIONAL, LG-5B PASS, LG-5C CONDITIONAL.** Live-voice enablement is **not** recommended yet (section 10). |

## 2. Implementation summary

### LG-5A: conversational robustness

New scenarios (`app/evaluation/lg5_scenarios.py`): **38 development** (`ND01-ND38`) and **50 holdout**
(`NH01-NH50`), 88 in all, in 12 categories. Each conversation runs in its own disposable PostgreSQL
database against the real tools and the UI-only confirmation endpoint. The holdout was written and
**frozen before any implementation change**; its SHA-256 fingerprint is
`d7f26e442c4a73179c1b45ecd1539b36942a0309eac9af94c171d8f3901f96d0`, enforced by
`tests/test_lg5_freeze.py` (which also keeps the spent LG-4 holdout frozen).

| Category (holdout) | n | | Category (holdout) | n |
|---|---|---|---|---|
| lookup-routing | 6 | | vague preferences | 4 |
| option-reference | 8 | | unsupported requests | 2 |
| relative-date | 10 | | phrasing | 2 |
| compound | 3 | | clinical | 2 |
| correction | 4 | | auth | 2 |
| interruption / topic change | 5 | | safety | 2 |

The new **strict metric** (`full_success`) requires the right final action *and* the expected
replies, no extra live review, no unexpected tool use, and no violations. The older end-state
`task_success` is still reported for comparability.

**Root causes of H02 / H12 / H19 (from the LG-4 traces), and the general mechanisms that replaced
the per-case fixes:**

| LG-4 failure | General cause | Mechanism now |
|---|---|---|
| H02 "Anything Thursday afternoon at X?" read as scheduling | One structured reading decides lookup vs. scheduling and nothing checks it against what the patient actually asked to *do* | An **independent second reading** (`TurnReader.check`: BOOK / RESCHEDULE / CANCEL / WITHDRAW / NONE) is consulted only when the first reading proposes an operation the words do not state; `_lookup_if_only_asking` makes route and operation consistent. A model proposal alone can no longer start a booking. |
| H12 "the family one" after an ambiguous clinic | Answers to a list the assistant just gave were searched as new names | **Offers** (`offers.py`): every opening, doctor or clinic shown is stored as a numbered offer in the checkpoint. A choice resolves only if the *distinctive words* of the message and the model's proposed *number* agree on exactly one offer; disagreement or no evidence means the patient is asked again. A closed `pick` call is a fallback when neither settles it. |
| H19 "next week" not reported by the model | Relative dates depended on the model writing a date | Model proposes a relative **spec** (`rel:thursday`, `rel:plus:3`, week, "soonest"); `resolve_spec` turns it into dates using **clinic-local time**; spoken weekday abbreviations and offsets are handled in `spoken.py`; "next <weekday>" that is ambiguous is answered for both days; "soonest" with no date uses a 14-day window. |

Other mechanisms (each tested in `tests/test_lg5_mechanisms.py`, ~67 tests): compound requests are
split into questions and each answered from its own tool call (`TurnReader.questions`,
`split_questions`, `PlannerDecision.more`); withdrawals ("scratch that") need independent
verification; names the model lengthened are cut back to what was said (`_stated_part`); a specialty
named in a correction replaces the doctor; doctor-only requests resolve their clinic from the
directory; a numeric reply inside a sentence ("the 2 o'clock one") is read as a choice only when
exactly one offer fits.

**Preserved:** the four-tool allowlist, deterministic scheduling and all patient-visible text built
from tool results, session validation, and confirmation only through the authenticated UI endpoint.

### LG-5B: PostgreSQL checkpoint persistence

| Requirement | Implementation |
|---|---|
| Supported PostgreSQL checkpointer, existing infrastructure, pooling | `PostgresSaver` over a `psycopg_pool.ConnectionPool` (autocommit, `dict_row`, `search_path=agent_checkpoints`) against the application's own database. No new service. |
| Explicit setup, schema, deletion | `CheckpointStore.open()` creates the `agent_checkpoints` schema and runs the library's own setup (`checkpoint_migrations`); `drop_everything()` removes it. CLI: `python -m app.evaluation.graph_store setup | stats | expire | drop`. **No Alembic migration was added** (head is still 0003) and no application table changed. |
| Only necessary state | The `AgentState` keys documented in LG-3 plus an `owner` digest. No credentials, tokens, session id, patient words, reply text, tool results or audio. `test_no_credential_token_session_id_or_patient_words_are_stored` scans every stored row. |
| Auth and authorization on every turn | `GraphConversations.turn` authenticates the session with the existing `AuthService` before touching the checkpoint, accepts patients only, derives the thread id as a one-way digest of the *verified* session, and discards any state whose `owner` is not the caller. Possessing a thread id grants nothing. |
| Safe deserialization | `JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=None, allowed_msgpack_modules=None)`: a pickled or foreign-typed row is refused, never rebuilt. |
| Expiry, revocation, stale confirms | The backend still invalidates ACTIVE pending actions on logout/expiry/revocation, and confirm still requires `PendingAction.session_id == principal.session_id`. The graph decides whether a stored review is still live with `_review_status`, evaluated **in SQL with `clock_timestamp()`**, so an expired or decided review is retired, never offered again. |
| Retention | Latest checkpoint only (history pruned after every turn); deleted at sign-out; idle conversations removed after **12 h**; `expire()` also removes the conversations of any session that is not ACTIVE or has expired. |
| Concurrency | One turn at a time per conversation via a PostgreSQL advisory lock on a **separate connection** (never the pool's, so lock holders cannot starve the saver). A second message gets a "please wait" reply and changes nothing. |
| Failure behavior | Store errors fail closed with a neutral reply; a failure while saving retires a review the patient never saw; a retirement that failed is retried after restart. The connection error path does not leak the password. |

### LG-5C: measured latency optimization

Profiled per turn: model (load / prefill / decode from Ollama's own counters), tools, checkpoint I/O,
other. See section 5 for numbers. Adopted: **prefix-first prompt layout** (static instructions first,
patient-specific context last, so the server reuses its KV-cache prefix) and **deterministic quick
paths** for genuinely unambiguous short replies (agreement to a single shown opening, withdrawal, a
lone clock time that matches one offer, a name that singles out one offer; refused when the message
has contrast words such as "but/instead/not" or is over six words). `warm_up()` and a
`keep_alive` option exist for Phase 3. Evaluated and **not adopted**: compact and lean output
formats (section 5).

## 3. Fresh holdout results (50 conversations, run once, after the implementation was final)

Qwen2.5:7B, disposable PostgreSQL per conversation, real tools, UI-only confirmation, graph with
PostgreSQL checkpoints. "Controller" is the previous `AgentController` path, run on identical
worlds.

| Metric | Previous controller | **LangGraph** |
|---|---|---|
| **Strict full success** | 20/50 (40%) | **42/50 (84%)** |
| End-state task success | 31/50 (62%) | **43/50 (86%)** |
| Helpfulness | 0.517 | **0.867** |
| Context retention (25 conv.) | 9/25 | **19/25** |
| Invalid decisions | 47 | **0** |
| Harness errors | 0 | 0 |
| Model calls (per turn) | 75 (0.91) | 88 (1.07) |
| Turns with no model call | 7 | 7 |
| Turn latency p50 / p95 | 2.27 / 4.15 s | 2.95 / 4.48 s |
| Mean prompt / output tokens | 1,220 / 41 | 1,717 / 67 |

Graph by category (strict): auth 1/2, clinical 2/2, compound 3/3, **correction 0/4**, interruption
5/5, lookup-routing 4/6, option-reference 8/8, phrasing 2/2, relative-date 9/10, safety 2/2,
unsupported 2/2, vague 4/4.

### Remaining failures (reported, not optimised away)

| ID | What happened | Cause | Class |
|---|---|---|---|
| NH05, NH06 | A lookup for a specific day ("Thursday", "Monday"), then "Dr. Kim ... then" / "great, book that": the assistant asked "which date?" | The day asked about in the lookup is not carried into the booking that follows it | Context carry-over gap |
| NH16 | "Any availability ... tomorrow afternoon?" prepared a booking review | The reading was routed to scheduling and the operation check accepted it; the patient asked only to *see* availability. Nothing was written and the review cannot be confirmed except by the patient in the UI. | Over-eager review (harness counts it as 1 "violation" in both architectures) |
| NH25 | "scratch that, tomorrow at 11" after a prepared review | Model returned `same_request` with no changed fields; the withdrawal was honoured but the new request was not started | Correction miss |
| NH26 | "actually the heart doctor" after a prepared review | Model returned `same_request`; the stale review for the first doctor stays live (still requires the patient's UI confirm and shows the doctor) | Correction miss |
| NH27 | Cancel review, then "actually, move it to the 12th at 2" | The review to cancel was retired, but "at 2" was not applied: the assistant listed 2:00 PM and 3:00 PM and asked which time; no reschedule was prepared (expected: reschedule to the 12th at 2 PM) | Correction miss |
| NH28 | Book Friday 1 PM, then "wait, Thursday" | The earlier 1 PM was kept for Thursday, found unavailable, and the patient was asked to choose the one open time (10:00 AM); the spec expects the Thursday 10:00 AM booking to be prepared | Correction miss |
| NH47 | Unauthenticated patient: "book it" after a lookup | Nothing was prepared (safe), but the reply did not tell them to sign in | Unhelpful wording |

Interpretation and limits:

- **Corrections after a prepared review are the open weakness** (0/4 here, 3/3 on the development
  set it was tuned on, and the one LG-4 regression in section 6 has the same shape): the model
  emits a short `same_request` reading without the corrected field, and no second reading checks
  for a change. This is a mechanism gap, not a safety gap.
- The holdout is 50 conversations; one conversation is 2 points. It was written by the same model
  and author as the implementation, from the same world generator, so shared blind spots are
  likely. Temperature 0 with one seed and one run means variance across seeds is **not measured**.
- The numbers above are the only holdout evaluation of the final implementation. The development
  set (38) was used for fixes and is not evidence of generalization.
- **Dev-cycle accounting.** Five full development cycles were run (strict full success 25, 33, 35,
  36, 36 of 38). Two small unit-tested fixes were made afterwards, before the holdout (data-grounded
  clinic resolution now precedes the model's pick; compound subjects come from the first reading).
  That is beyond five *full* cycles by a strict count; it is disclosed, and it happened before the
  holdout was run. Three development specs were corrected for errors in my own expectations (ND11
  gained a turn, ND19 now expects the earliest opening, ND25 expectations); **no holdout
  expectation or text was changed**, which the fingerprint test enforces.
- Pre-LG-5 graph on the new development set (before any change): 23/38 task success (controller
  21/38). It was **not** run on the new holdout, because the holdout was reserved for the final
  implementation.

## 4. PostgreSQL persistence, isolation, expiry and restart evidence

`tests/database/test_graph_persistence.py` (23 tests, real PostgreSQL), plus
`tests/test_graph_store_unit.py` (6) for the serializer and key derivation:

| Requirement | Test |
|---|---|
| Own schema, nothing else touched; removable | `test_the_store_makes_its_own_schema_and_touches_nothing_else`, `test_the_schema_can_be_removed_whole` |
| Process restart | `test_a_conversation_resumes_after_a_process_restart`, `test_a_live_review_survives_a_restart_and_is_still_the_one_to_confirm`, `test_a_retirement_that_failed_is_retried_after_a_restart`, `test_a_review_decided_while_the_process_was_down_is_not_shown_again` |
| Auth on every resumed turn; staff cannot converse | `test_every_resumed_turn_is_authenticated_first`, `test_a_signed_in_staff_member_cannot_converse` |
| No secrets or patient words stored | `test_no_credential_token_session_id_or_patient_words_are_stored` |
| Logout, expiry, revocation | `test_after_sign_out_nothing_resumes_and_the_stored_conversation_is_gone`, `test_an_expired_session_cannot_resume_or_confirm_a_stale_review`, `test_a_revoked_session_cannot_resume`, `test_signing_in_again_never_inherits_the_old_conversation` |
| Patient isolation, foreign owner | `test_two_patients_have_separate_conversations_and_reviews`, `test_a_conversation_that_belongs_to_someone_else_is_discarded_not_continued` |
| Concurrency | `test_two_messages_to_one_conversation_run_one_after_the_other`, `test_a_second_message_that_cannot_get_in_is_told_to_wait_and_changes_nothing`, `test_many_patients_at_once_do_not_disturb_each_other` |
| Database failure | `test_a_checkpoint_store_that_is_down_fails_closed`, `test_a_failure_while_saving_retires_a_review_the_patient_never_saw`, unit test for unreachable DB without password leak |
| Cleanup and retention | `test_only_the_latest_checkpoint_is_kept`, `test_idle_conversations_are_removed_and_recent_ones_are_kept`, `test_the_sweep_keeps_the_conversations_of_live_sessions` |
| Safe deserialization | unit tests: plain state round-trips, foreign type is not rebuilt, pickled row refused outright |

Findings during this work: `_review_status` first compared against the application clock and would
have discarded live reviews in one timing window; it now uses the database clock, with a positive
test that a live review survives. Re-signing in does not revoke the old session (existing behavior),
so the new conversation is keyed to the new session and the old one is swept.

## 5. Latency, model calls, accuracy and memory

Workload: the 38-conversation development set (61 turns, 18 of them multi-turn conversations),
graph mode, PostgreSQL checkpoints, same model and seed, one change at a time, run sequentially.

| Run | Change | Strict full / 38 | Calls (per turn) | Turns with no call | p50 | p95 | mean | Prefill p50 | Decode p50 |
|---|---|---|---|---|---|---|---|---|---|
| r0 | LG-5A code, no quick paths, legacy layout (**before**) | 37 | 79 (1.30) | 0 | 4.48 s | 6.36 s | 4.34 s | 1.62 s | 2.85 s |
| r1 | + quick paths | 38 | 75 (1.23) | 3 | 4.04 s | 6.69 s | 4.12 s | - | - |
| **r2** | **+ prefix-first layout (final)** | **38** | **68 (1.11)** | **3** | **3.42 s** | **6.02 s** | **3.32 s** | **0.17 s** | 3.10 s |
| r3 | r2 + compact output | 38 | 68 (1.11) | 3 | 3.20 s | 5.01 s | 3.22 s | - | - |
| r4 | r2 + lean reading (model writes no dates) | 35 | 70 (1.15) | 1 | 2.62 s | 5.45 s | 2.89 s | - | - |

(Single runs. Runs r1-r3 reached 38/38, r0 37/38 and cycle 5, before the last two fixes, 36/38; a
difference of one conversation between runs should not be read as meaningful.)

What the profile shows for the final configuration (r2): the model is **97%** of a turn (decode 88%,
prefill 8%, model load 0.1%), tools **1.7%** (p50 54 ms), checkpoint I/O **2.5%** (p50 92 ms, p95
122 ms), everything else about 0. Checkpoint persistence costs under 0.1 s per turn; the cost is
generating tokens, not storing them.

By number of model calls (holdout, graph): 0 calls p50 0.04 s (7 turns); 1 call p50 **2.95 s**,
p95 4.35 s (62 turns); 2 calls p50 **4.16 s**, p95 8.70 s (13 turns). The previous controller:
1 call p50 2.43 s.

Decisions:

- **Prefix-first layout: adopted.** It cut prefill p50 from 1.62 s to 0.17 s and turn p50 by 1.06 s
  with identical accuracy.
- **Quick paths: adopted.** They remove 3-7 model calls per workload, only where the reply is
  unambiguous; any contrast word, more than six words, or a conflict with the offers sends the
  message to the model. They never skip authentication, validation or the deterministic scheduler.
- **Compact output (r3): not adopted.** Output tokens fell only 3% (77.4 -> 75.2), and the 0.2 s p50 /
  1.0 s p95 difference from r2 is a single run each, not attributable to the change; it also adds a
  second output format to maintain. Revisit with repeated runs.
- **Lean reading (r4): rejected.** 0.8 s faster p50 but 35/38 vs 38/38, an accuracy loss that is not
  worth the speed.
- **Warm-up: adopted as an API, measured in isolation.** First request after the model was
  unloaded: **7.8 s cold** (three runs, 7.75-7.81 s) vs **2.96 s** after a 5.0 s `warm_up()` (three
  runs). Warm-up is a startup cost, not a per-turn one, and must run off the request path.
- **Redundant calls:** the earlier two-call plan+extract pattern was already removed in LG-4. In
  LG-5 the only extra call is the independent operation `check` (and `pick` / `questions` when
  needed), at 1.11 calls per turn overall.

Previous controller vs graph on equivalent worlds, final code:

| Set | Controller p50 / p95 | Graph p50 / p95 | Controller calls/turn | Graph calls/turn |
|---|---|---|---|---|
| LG-5 holdout (50 conv.) | 2.27 / 4.15 s | 2.95 / 4.48 s | 0.91 | 1.07 |
| LG-5 dev (38 conv., r2 / dev-c5 controller) | 2.74 / 4.71 s | 3.42 / 6.02 s | 0.97 | 1.11 |
| LG-4 dev (31 conv.) | 2.70 / 4.07 s | 3.26 / 5.47 s | 0.85 | 0.96 |
| LG-4 holdout (20 conv., spent) | 2.16 / 3.55 s | 2.92 / 5.21 s | 0.92 | 1.06 |

LG-4 reference (before LG-5): graph holdout p50 3.54 s / p95 4.84 s; graph dev 3.45 / 5.20 s. The
final graph is faster at the median than LG-4 despite PostgreSQL checkpoints and the extra
safeguards; p95 is slightly worse on the longer sets.

**Memory.** The Ollama `llama-server` process held **4.60-4.66 GB RSS** in every run (the model
reports 4.77 GB, all in memory, 8192 context); the two runs with the prefix layout and full output (r2, r3) were about 0.06 GB higher than the
others. The Python/PostgreSQL side was not separately profiled; the
checkpoint store holds one small row per live conversation. Peak resident memory of the evaluation
process was **not measured**.

**Against the 2-3 s target:** single-call turns on the holdout have a **2.95 s median** (borderline),
but p95 is 4.5-6.0 s and two-call turns (compounds, operation checks, picks) take 4.2 s median and
8.7 s p95. The target is **not met** and was not forced.

### Notes for Phase 3 (not implemented)

1. Decode is ~88% of a turn: reduce output tokens (the reading has many optional fields), and
   evaluate a smaller or faster model/hardware before further prompt work.
2. Run `warm_up()` at process start and set `keep_alive` (for example 30 m) so the first call of a
   session never pays the 7.8 s cold start.
3. Keep the static prompt prefix byte-identical across turns (the layout is tested for this) so the
   server's prefix cache keeps working.
4. Speak an acknowledgement ("Let me check that") as soon as the turn starts when a second call is
   likely (compound, operation check); that covers the 4-9 s tail.
5. Replies are built from tool results after the model call, so they can be streamed sentence by
   sentence into TTS; start speaking the first sentence while later ones are composed.
6. Run the operation `check` concurrently with the first reading when the server has a free slot.
7. Barge-in: cancel the in-flight turn; the per-conversation advisory lock and the BUSY reply
   already guarantee one turn at a time and no state change from an abandoned turn.
8. Confirmation stays on the screen: the voice path must keep saying that the patient confirms in
   the app, never confirm by voice.

## 6. Regression and security review

| Check | Result |
|---|---|
| Full test suite, PostgreSQL tests enabled | **2,357 passed** (LG-4: 2,234). 122 tests in the four new files. |
| Ruff check / format check | clean (197 files formatted) |
| `uv lock --check` | consistent (adds `langgraph-checkpoint-postgres>=3,<4` and `psycopg-pool>=3.2,<4`) |
| Phase 2 acceptance smoke | 15/15 |
| Voice agent / `app/` outside `app/evaluation/` | **unchanged** (no file outside `app/evaluation/` was modified; `app/voice/voice_agent.py` is untouched) |
| Alembic | unchanged at 0003 |
| Unauthorized writes, double bookings, confirmation bypasses, credential leaks | **0** in every run |

Equivalent-world comparison on the LG-4 sets (current implementation; the LG-4 holdout is spent, so
this is a regression check, not a holdout; H02/H12/H19 were studied during LG-5A, so their passes
are expected, not evidence):

| Set | Controller (task / strict) | Graph (task / strict) | LG-4 graph result |
|---|---|---|---|
| LG-4 dev (31) | 30 / 24 | **30 / 30** | 31/31 task |
| LG-4 holdout (20) | 14 / 10 | **20 / 19** | 17/20 task |

**One regression:** LG-4 dev **D20** ("Book Pine Ridge Clinic Thursday afternoon with Dr. Rao" then
"actually Dr. Kim") now fails (a review for Dr. Rao stays live). Reproduced four times in isolation
under both prompt layouts, so it is deterministic and not caused by LG-5C. In the graph context the
model returns a 35-token `same_request` reading with no doctor name, although the same message
read outside the graph context does return `doctor_name: "Dr. Kim"`. It is the correction-after-review
weakness of section 3. It was **not fixed**: the five-cycle budget was spent and the fresh holdout
had already been evaluated, so changing the code would have invalidated the holdout result. The
other strict miss, LG-4 holdout H09 (a compound), fails in both architectures.

Security review (single-model; see section 8):

- Prompts: no PINs, tokens, session ids or credentials are ever placed in a model prompt or stored;
  the sign-in path (`NH48`) never reaches the model.
- The model cannot call tools; it proposes a reading, code plans from a fixed four-tool allowlist and
  builds every patient-visible sentence from tool results. A model proposal for an operation is
  vouched by the wording or the independent check; a model proposal for an offered choice is used
  only if the words do not disagree.
- Stale-action guard: confirm is judged by the database clock and by session id, independent of
  anything in a checkpoint. Tests show an expired, logged-out or revoked session cannot confirm.
- **Residual risk to note:** the checkpoint's `review` holds the appointment summary (date, time,
  doctor, clinic) so the conversation can resume. With synthetic data this is harmless; with real
  data it is PHI in the application database. Production would need encryption at rest, access
  control on the `agent_checkpoints` schema, exclusion from analytics/logs, and an operational
  scheduler for the 12 h sweep (`expire` exists but nothing runs it yet).
- Code lives in `app/evaluation/` (experimental). Moving it to production packages is future work.

## 7. Files changed

| Area | Files |
|---|---|
| New | `app/evaluation/graph_store.py`, `graph_service.py`, `offers.py`, `lg5_scenarios.py` |
| Modified | `app/evaluation/orchestrator.py`, `planner.py`, `hybrid.py`, `turn_reader.py` (prompt `lg5-t2`), `spoken.py`, `llm.py` (keep-alive, load/prefill/decode records), `lg4_eval.py` (sets registry, strict metric, profiling, PostgreSQL checkpoint, flags, traces) |
| New tests | `tests/test_lg5_freeze.py` (3), `tests/test_graph_store_unit.py` (6), `tests/test_lg5_mechanisms.py` (~67 incl. offers, dates, operation check, picks, quick paths, compound, layout), `tests/database/test_graph_persistence.py` (23) |
| Modified tests (disclosed) | `tests/test_orchestrator.py` (helper uses in-memory saver and `quick=False`), `tests/database/test_orchestrator_db.py` (uses the PostgreSQL `graph_store` fixture), `tests/database/conftest.py` (fixture), `tests/test_evaluation_unit.py` (guards widened to the new modules) |
| Dependencies | `pyproject.toml`, `uv.lock`: `langgraph-checkpoint-postgres`, `psycopg-pool` |
| Not touched | `app/voice/`, Alembic revisions, frozen LG-4 scenarios, any other `app/` module |

## 8. Council-style assessment

**This is a single-model review.** It was written by the same model that implemented and evaluated
the work; no independent model or human reviewer took part. The four perspectives below are
structured points of view of that one reviewer and carry correspondingly less weight than
independent review would.

- **Agent architecture.** The separation (model proposes, code plans, tools act, UI confirms) held
  under 88 new conversations and added no authority to the model. The offer mechanism and the
  operation check are generic and testable. Weak spots: the number of "second opinion" calls grows
  with new failure classes (check, pick, questions), the Approach B reading remains the single point
  where a 7B model's misses become user-visible, and corrections are not yet handled by any
  mechanism of the same quality.
- **Security.** Authentication happens before checkpoint access on every turn, checkpoints grant no
  authority, deserialization is strict, the database clock decides review liveness, and nothing
  sensitive is written. The remaining concern is data-classification (PHI in checkpoints with real
  data) and an unscheduled sweep. The one live-review flag (NH16) is a usability fault, not an
  authorization one.
- **Evaluation.** The holdout was frozen and fingerprinted first and run once; strict metrics were
  added before the tuning rather than after; dev corrections are disclosed. Limits: 50 conversations,
  same author, one seed, no variance estimate, a regression found only after the holdout, and the
  LG-4 holdout has been consulted twice and can no longer measure generalization.
- **Performance.** Latency is dominated by decode of ~67 output tokens; storage is negligible.
  The measured gains came from cache-friendly prompt layout and removing unnecessary calls, not from
  skipping interpretation. The median is near 3 s for one-call turns, but p95 and two-call turns
  are not voice-ready without streaming and spoken acknowledgements.

## 9. Verdicts

| Milestone | Verdict | Reason |
|---|---|---|
| **LG-5A** conversational robustness | **CONDITIONAL** | Holdout 42/50 vs 20/50, H02/H12/H19 mechanisms generalized, all invariants held. Conditions: corrections after a prepared review (0/4 holdout, plus regression D20), carry-over of a looked-up day into the following booking (NH05/06), and the NH16 over-eager review. |
| **LG-5B** PostgreSQL persistence | **PASS** | All required scenarios are tested against real PostgreSQL, with fail-closed behavior and no secrets stored. Operational items (sweep scheduling, encryption for real PHI) are deployment work, listed in section 10. |
| **LG-5C** latency | **CONDITIONAL** | Real, measured gains (p50 4.48 -> 3.42 s, calls 1.30 -> 1.11, prefill 1.62 -> 0.17 s) with no accuracy loss; the voice target is **not** met at p95 or for two-call turns. |

## 10. Blockers before enabling LangGraph in the live voice agent

1. Fix the correction-after-review mechanism (verify what changed on a short correction), and the
   lookup-to-booking carry-over of the day asked about; re-test on a **new** holdout (the LG-5 one is
   now spent) of at least 100 conversations with several seeds to estimate variance.
2. Decide the NH16 class: a pure availability question must not create a review.
3. Voice latency design: streaming replies, spoken acknowledgement, startup warm-up, measured on
   the real voice path with real hardware; the p95 and two-call tail need to come down.
4. Productionize persistence: move out of `app/evaluation/`, schedule the retention sweep, encrypt and
   restrict the `agent_checkpoints` schema, and confirm connection-pool sizing under the voice load.
5. Independent review (human or separate model) of the orchestrator and persistence code; this
   document's assessment is a single-model one.
6. Real-microphone and speech-recognition error evaluation: spoken numbers, names, and accents were
   only approximated by typed text.
7. Keep the old controller as a fallback behind a flag, and re-run the full regression and
   security suites against the voice integration.

This historical LG-5 evaluation ended before the voice agent was switched. Phase 3 was completed
later for the local prototype.
