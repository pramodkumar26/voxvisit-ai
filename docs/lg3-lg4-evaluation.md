# LG-3 and LG-4: LangGraph orchestration, independent evaluation, and the Qwen3:8B comparison

Status: **historical experimental evaluation**. At the time of this evaluation, the new graph was
not connected to the live voice agent and nothing switched the live model. All data is synthetic.
LG-5F was later selected for the completed local Phase 3 prototype; see
[`phase3-browser-confirmation.md`](./phase3-browser-confirmation.md). Detailed per-conversation
artifacts are in the ignored `.local/lg4/` directory (replies, tool calls, timings); this document
holds the original results.

## 1. Summary

| Question | Answer |
|---|---|
| Does the graph beat the old controller on conversations it was not tuned on? | Yes on the frozen holdout (Qwen2.5:7B): **17/20 vs 14/20** task success, 0 invalid decisions vs 19. It is about 1.2 s slower per turn at the median. |
| Did combining planning and intent extraction help? | Yes, measured: **1.0 vs 1.61 model calls per turn**, equal or better success (holdout 17/20 vs 15/20), lower p95 latency (4.8 s vs 6.7 s). |
| Safety | **0 unauthorized writes, double bookings, confirmation bypasses, secrets to a model, or clinical advice** in every run of every mode (the old controller included). Full regression: 2,234 tests pass. |
| Qwen3:8B vs Qwen2.5:7B on the same graph | Statistically indistinguishable (holdout 16/20 vs 17/20; dev 30/31 vs 31/31). Qwen3 is ~10–15% faster per turn and needs ~0.7 GB more memory. **Keep Qwen2.5:7B.** |
| Do LG-3 and LG-4 meet their acceptance criteria? | Safety and regression gates: **met**. Quality: **met against the old controller**, with three honest holdout failures that are model-reading misses (section 6). The graph stays experimental. |

## 2. LG-3 architecture

### State ownership

The checkpointed LangGraph state is the conversation. In LG-1/LG-2 the controller owned the
request and the graph held a copy. In LG-3 `OrchestratedSession` (`orchestrator.py`) builds the
deterministic scheduler (`HybridControllerModel`, unchanged scheduling code) from the checkpoint at
the start of each turn and writes its state back at the end. The scheduler holds nothing between
turns, so there is no second source of truth, and a checkpoint is enough to resume a conversation.

| Checkpoint key | Holds |
|---|---|
| `request` | the draft being built: clinic, doctor, dates, preferences, selection |
| `dialogue` | the question awaited, the choices offered, picks pinned for the next search |
| `review` | the change awaiting Confirm: id, summary, the details the UI's report-back needs |
| `retiring` | a review that could not yet be retired (retried next turn) |
| `context` | what the last lookup showed (so "I'll take 2 PM" has a referent); a lookup left open for one missing detail |
| `interruptions` | asides taken without leaving the request, and whether they were resumed |

Never checkpointed: the principal, the session id (the thread id is a one-way hash of it), tokens,
the tool runtime, the model, tool results, the patient's words, and reply text. In-memory saver;
deleted at sign-out, expiry, eviction, or when the session id changes patient.

### Flow

```
intake -> understand -> lookup -------------------------> finish
                     \-> (clarify / out of scope) ------> finish
                     \-> schedule: decide <-> tool -----> finish
```

- **understand** is one model call per message (`TurnReader`, `turn_reader.py`) returning a
  `TurnReading`: the route (SCHEDULING, LOOKUP, CLARIFY, OUT_OF_SCOPE), what to look up (a list of
  subjects, so compound questions work), whether it corrects the request in hand, whether it asks
  for an appointment without a booking word, whether it describes an emergency, and the request
  fields. The reading is only a proposal: every name, date and time is checked against the patient's
  own words; dates are read from the words, not the model; the model names no tool and no
  identifier; any failure falls back to the old extractor ladder.
- **lookup** answers from read-only tools in a fixed order (directory, then availability), at most
  five calls per message, several questions per message. Every sentence is built from tool results.
  A clinic the model missed is found in the directory itself; an ambiguous clinic is put to the
  patient, never chosen.
- **schedule** is the existing deterministic path. It alone plans and prepares booking,
  rescheduling and cancellation reviews. Confirmation is only the authenticated UI's: no node, tool
  or route in the graph can confirm.
- Planner "hints" from LG-2 are gone. What they carried is now explicit state (`context`) or
  decided in one place (`_reconcile`). The one remaining flag, `operation_vouched`, lets a reading
  vouch for an unspoken BOOK ("Pine Clinic Monday 9:30").

### Targeted improvements found by the evaluation (shared with the old controller)

These are fixes in shared scheduler code, so the old controller benefits too; the comparison is of
the architectures, not of who got the fixes.

1. A clock time that names none of the patient's appointments is the time they want, not a filter
   (grounded in tool data). Cause: "move my appointment … to November 12 at 2 PM" was read with
   `appointment_time = 2 PM`, so the real appointment was never found.
2. `selection = SPECIFIC_TIME` needs a stated time; the model had been returning it for "Book X on
   Thursday", which made the agent ask "What time?" without searching.
3. A bare "9:30" is settled from the schedule when exactly one reading is open, and the review says
   so; if both are open, or neither, it still asks. (This changed the order for two older tests: see
   section 7.)
4. "instead" alone no longer turns a booking into a reschedule.
5. A "doctor" whose words are the rest of a clinic's own name was a mis-split name; found by the
   directory and corrected.
6. Bare numbers answering "which time" ("11") resolve to the one offered time they read as.
7. Emergency symptoms get a fixed emergency notice (graph path only), never medical advice.

## 3. LG-4 evaluation design

- **Conversations:** 51 synthetic multi-turn conversations. **31 development** (used to find and fix
  defects) and **20 holdout**, written before any evaluation, fingerprinted
  (`HOLDOUT_FINGERPRINT`, enforced by `tests/test_lg4_freeze.py` and by the runner), run once on the
  finished implementation, and never edited or tuned against. Categories: scheduling, discovery,
  interruption, correction, ambiguous, phrasing, compound, authentication boundary, unsupported
  clinical, safety.
- **World:** each conversation has its own disposable PostgreSQL database (migrated and seeded),
  four clinics, five doctors, a patient with one booking, and another patient holding a slot. Real
  `ToolRuntime`, real verified session, real FastAPI confirmation endpoint, controllable clock. The
  UI driver presses Confirm; the agent can never.
- **Measures:** task success (database end state: the right review, or the right commit, or
  nothing); helpfulness (reply checks plus task success); context retention (success on
  conversations whose outcome depends on earlier turns); tool correctness and invalid decisions
  (tool errors other than normal "none found/ambiguous/sign in", unknown tools, invalid model
  readings); safety (unauthorized write, double booking, extra live reviews, claiming a change
  before the app confirmed it, secrets, private text, clinical advice, a credential reaching a
  model); model calls; p50/p95 turn latency.
- **Model and settings:** Qwen2.5:7B (digest `845dbda0ea48`), temperature 0, seed 7, context 8,192
  tokens, 512 output tokens, one reply per call. Apple M5, 16 GB; Ollama 0.33.3.
- **Modes:** `controller` (old path), `graph` (LG-3, one call per message, the default), `graph-split`
  (LG-3 state with LG-2-style planner + extractor, two calls), to measure the combined call.
- **Improvement cycles:** 2 of the allowed 5 were used. The development set was saturated by cycle 2,
  so no further cycles were run.

## 4. Results: old controller vs LangGraph (Qwen2.5:7B)

### Frozen holdout (20 conversations, run once, after the implementation was frozen)

| Metric | Controller | **Graph (1 call)** | Graph split (2 calls) |
|---|---|---|---|
| Task success | 14/20 (70%) | **17/20 (85%)** | 15/20 (75%) |
| Helpfulness | 0.617 | **0.833** | 0.733 |
| Context retention (9 convs.) | 4/9 | **6/9** | 4/9 |
| Invalid decisions | 19 | **0** | 18 |
| Safety violations | 0 | 0 | 0 |
| Model calls per turn | 0.92 | 1.00 | 1.61 |
| Turn latency p50 / p95 | **2.29 / 3.37 s** | 3.54 / 4.84 s | 3.57 / 6.67 s |
| By category (graph) | | ambiguous 1/2, auth 2/2, clinical 2/2, compound 1/1, correction 2/2, discovery 1/1, interruption 3/3, safety 1/1, scheduling 4/6 | |

(Controller "invalid decisions" are strict-reading rejections by the extractor that the controller
then salvaged; they are recovered, not user-visible.)

### Development set (31 conversations) across the improvement cycles

| Cycle | Controller | Graph (1 call) | Graph split |
|---|---|---|---|
| 0 (baseline) | 24/31 | 23/31 | 26/31 |
| 1 (five fixes) | 30/31 | 29/31 | 31/31 |
| 2 (final) | 30/31 | **31/31** | 31/31 |

Final development detail: helpfulness 0.862 / 0.952 / 0.978; model calls per turn 0.85 / 0.98 / 1.64;
p50 latency 2.64 / 3.45 / 3.84 s; p95 4.13 / 5.20 / 6.70 s. The baseline's one "safety violation" per
mode was a false positive in my harness (it flagged "please don't share a PIN"); fixed before cycle
1 by reusing the Phase 2 negation-aware detector.

### Holdout failures (reported, not optimised away)

| Conversation | Graph | Cause |
|---|---|---|
| H02 "Anything Thursday afternoon at X?" then "ok the 2 PM" | fail | the model read the first message as scheduling, not a lookup; no tools ran |
| H12 "the family one" after an ambiguous clinic | fail | the answer is searched in the directory as a name; it should resolve against the offered options |
| H19 "What does X have next week?" | fail | the model did not report "next week", so the agent asked for a date |

The controller additionally failed H05, H07, H08 (a lookup that should seed the next request, an
aside that dropped the request, a commit scenario that never reached a review).

## 5. Qwen2.5:7B vs Qwen3:8B (same graph, prompts, fixtures, frozen scenarios)

`qwen3:8b` was not installed; it was downloaded (digest `500a1f067a9f`, Q4_K_M, 8.2B, 5.2 GB). It
works with the existing structured interface with **no code change**, with thinking **off**
(`think: false`, short outputs, no schema failures). The thinking mode was not benchmarked.
Settings identical for both: temperature 0, seed 7, context 8,192, 512 output tokens, thinking off,
run sequentially (one model loaded at a time), same code, prompt (`lg3-t1`), and worlds. The
Qwen2.5 holdout rerun reproduced its first result exactly (17/20), as expected for seed 7.

| Metric | Qwen2.5:7B | Qwen3:8B |
|---|---|---|
| Holdout task success | **17/20 (85%)** | 16/20 (80%) |
| Dev task success | **31/31** | 30/31 |
| Holdout helpfulness | **0.833** | 0.808 |
| Holdout context retention | **6/9** | 5/9 |
| Holdout interruptions / scheduling | **3/3** / 4/6 | 1/3 / **5/6** |
| Safety violations (both sets) | 0 | 0 |
| Invalid decisions | 0 | 0 |
| Model calls per turn | 1.00 | 1.00 |
| Turn latency p50 / p95 (holdout) | 3.51 / 5.04 s | **2.85 / 4.49 s** |
| Turn latency p50 / p95 (dev) | 3.52 / 5.14 s | **3.08 / 5.21 s** |
| Wall time (holdout / dev) | 136 / 174 s | **122 / 162 s** |
| Ollama resident memory (peak sampled) | **4.4 GB** | 5.1 GB |
| Model size loaded (8,192 ctx, all on GPU) | **4.9 GB** | 6.0 GB |

Qwen3 won H02 and was better on scheduling, but lost two interruption conversations (an
aside misread, and a Monday-morning reading that returned no openings). The difference (one
conversation in 20) is within noise; this evaluation cannot separate the models on accuracy.

**Recommendation: keep Qwen2.5:7B as the default for VoxVisit AI.** It is at least as accurate on
every set and category that matters (interruptions, context retention), smaller on a 16 GB machine
(~0.7 GB), and already the validated model for the earlier phases. Qwen3:8B is ~10–15% faster per
turn and is a credible alternative; revisit it when the evaluation set is large enough to resolve a
5-point difference and after the prompt is tuned per model. Neither model was retuned for this
comparison, and the live agent was not switched.

## 6. Safety verification

- **Zero** across all runs of all modes (controller, graph, graph split; both models; development and
  holdout): appointment rows changed without a UI commit, double bookings, more than one live
  review, a live review for a conversation that asked for none, claiming a change before the app
  confirmed it, asking for or echoing a secret, another patient's code in a reply, clinical advice,
  a credential reaching a model, a signed-out visitor preparing a change.
- Enforced by tests that run in the default suite (no model needed): the orchestrator cannot confirm
  (`test_orchestrator_db.py`: no `CONSUMED` pending action without the UI), reviews are retired when
  dropped (including a failed-retirement retry kept in the checkpoint), a signed-out visitor can look
  things up but not prepare, sessions are isolated, checkpoints hold no secret/identifier/spoken
  word, and cleanup deletes only the ended session's checkpoints.
- Credentials are screened before any model call (graph and controller); the credential conversation
  recorded 0 model calls.
- Regression: full suite with PostgreSQL **2,234 passed**; Ruff check and format clean; `uv lock
  --check` consistent; Alembic head 0003 unchanged; the Phase 2 conversation smoke (15 conversations,
  real Qwen) still 15/15 with 0 safety violations; the frozen Phase 2 acceptance manifest and
  benchmark files have no diff. The Phase 2 acceptance run itself was not repeated (it is a
  one-time frozen run).

## 7. Changes to existing tests (disclosed)

None was weakened to get a score; each encoded an ordering or wording the new, deliberate behaviour
changed.

- Two unit tests and one database test asserted that **no availability search happens before** the
  "AM or PM?" question. The agent may now read the schedule first (read-only) to see if only one
  reading is open. They now assert the question is still asked when both readings are open and that
  nothing is prepared before the answer; two new tests cover the one-reading and no-reading cases.
- One LG-2 test used "sorry, I meant Thursday instead" as a message the controller reads as a move;
  "instead" is no longer a move word, so it now uses "let me change that to Thursday".
- The architecture test that only `runner.py` imports the UI driver now allows `lg4_eval.py` too, and
  additionally asserts no agent-side module (`hybrid`, `conversation`, `orchestrator`, `planner`,
  `turn_reader`, `model`) imports it.

## 8. Files

New: `orchestrator.py`, `turn_reader.py`, `planner.py` (LG-2, extended with compound questions),
`graph_session.py` (LG-1/LG-2, kept as a predecessor), `lg4_scenarios.py`, `lg4_eval.py`;
`tests/test_orchestrator.py` (32), `tests/test_graph_planner.py`, `tests/test_graph_session.py`,
`tests/test_lg4_freeze.py`, `tests/database/test_orchestrator_db.py`. Changed: `hybrid.py` (shared
fixes above, `operation_vouched`, public helpers), `spoken.py` ("3 o'clock"), `llm.py` (`think`),
`conversation.py` (steps split out of the loop; registry `session_factory` and `close`), and the tests
named in section 7. `pyproject.toml`/`uv.lock`: `langgraph` as a dev dependency. Not touched: the voice
agent, the frozen benchmarks and acceptance files, the scheduling services, authentication.

## 9. Unresolved risks and limitations

- The holdout is 20 conversations; one conversation is 5 points. Differences of one or two
  conversations (model vs model) should not be read as real.
- The three holdout failures are model-reading misses. The one-call reading is somewhat more
  sensitive to misrouting than the two-call path on lookups (H02), and less costly everywhere else.
- A lookup now seeds the next request only for a single clinic/day/doctor shown; a week listing or
  several clinics does not (the patient is asked).
- "the family one" style answers to an offered list of clinics are not matched to the offered names.
- Latency: the graph adds ~1.2 s at the median over the controller (one extra model call for every
  message, including simple scheduling ones that the old path read in about the same time).
- The evaluation used `think: false`; Qwen3 in thinking mode was not tested.
- The old controller path keeps its LG-1 stand-in, `GraphSession` (LG-2), which should be removed once
  the orchestrator is approved.

## 10. Remaining work before reconnecting the live voice agent

1. Review and approve the architecture; decide the fate of `GraphSession` (LG-2).
2. Resolve the three failure causes: offered-option resolution for free-text answers; a safeguard
   when a lookup-like message is routed to scheduling with no tools run; reading "next week" without
   the model.
3. A larger, independent holdout (100+ conversations) to resolve model differences, and repeated
   runs at a non-zero temperature to measure variance.
4. Persist the checkpoint to the database (the in-memory saver loses the conversation on restart)
   with the same no-secrets guarantees, and add expiry matching the session lifetime.
5. Wire `OrchestratedSession` into the voice agent behind a flag; add a latency budget (the voice
   path needs streaming or a spoken acknowledgement for a 3–6 s turn), and keep the controller as a
   fallback.
6. Real-microphone verification of the combined path, and an emergency-notice review by someone
   qualified.
7. Update `docs/roadmap.md` and the conversation-layer docs once approved.

## 11. Acceptance

| Criterion | Result |
|---|---|
| Zero unauthorized writes, double bookings, confirmation bypasses | **Met** (all runs; tests) |
| Full regression and security/concurrency suite green | **Met** (2,234 passed) |
| Holdout untouched by tuning | **Met** (fingerprint enforced; run once) |
| Graph at least as good as the old controller on the holdout | **Met** (17/20 vs 14/20; 0 vs 19 invalid decisions) |
| State authoritative in the graph, no competing state | **Met** (scheduler is rebuilt each turn; test edits the checkpoint and behaviour follows) |
| Fewer duplicate model calls, measured | **Met** (1.00 vs 1.61 calls per turn; success equal or higher) |
| Latency | **Not improved**: +1.2 s p50 over the old controller; acceptable for the experimental path, not yet for voice |
| Qwen3:8B reliability | **Met**: runs reliably with thinking off; no blocker |
