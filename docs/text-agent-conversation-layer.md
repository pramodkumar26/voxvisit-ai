# Text Agent (Approach B): Conversation Layer

**Status:** implemented and tested with the real backend tools; bounded real-model smoke run done (results below).
**Scope:** Approach B only (Qwen reads the intent, deterministic code does everything else). Approach A was not developed further.
**What this is not:** a benchmark pass. The frozen architecture comparison's mechanical decision remains `NEITHER_SELECTED`; Approach B was chosen by the human reviewer as the development architecture. The frozen benchmark, its manifest, thresholds and scorer were not modified, tuned against, or rerun.

## 1. What changed and why

The official comparison left Approach B with these known failures. Each was fixed at its cause, not per scenario.

| Failure | Root cause | Fix |
|---|---|---|
| "I don't see an opening at None" | The model chose "specific time" but left the time empty; the controller printed the empty value | An intent is validated before any tool is called. A specific time with no readable time asks "What time would you like?". Every reply passes one guard that replaces empty text, `None`, `null`, `NaN` or braces with a plain notice |
| Doctor looked up by substring, no appointment found | "Dr. Ames" is not a substring of "Dr. Rina Ames" | Names are matched word by word against trusted tool data (titles ignored). One match proceeds; several are put to the patient; none asks again. Clinics match the same way |
| Extraction rejected, patient left with no reply | One unsupported field made the whole reading invalid, and the error ended the turn | A bounded ladder (see below). The patient always gets a reply |
| Authentication | B never asked for a PIN, but nothing guaranteed it, and a patient could type one | Any message that mentions a PIN, password, login code or token, or is only digits, is answered with a notice, is never read, stored, sent to the model or repeated. "Not signed in" sends the patient to the secure sign-in screen |
| Dates | Only dates with a year were understood | "November 17", "17th of November", "tomorrow", "next Friday", "the 18th" (in the month under discussion) are read by code and checked against the message |
| Over-reading | The 7B model read "book an appointment" as "earliest", "the 17th" as a move, and a time as a new time | A value is kept only if the message supports it: names, dates, clock times, time of day, positions ("the second one") and selection words must appear in it. An operation needs words that mean it |
| Date roles when rescheduling | The model put "to November 17" in the wrong field | When a message has dates, the words decide which is the existing appointment's day and which is where it moves to |

### Reading a message: a bounded ladder (at most two model calls per message)

1. A strict reading: a reply that is not valid, or states anything the message does not, is rejected whole.
2. **Salvage** (no model call): the rejected reply is reused, keeping what the message supports and nulling the rest. The controller then asks about what is missing.
3. If there was nothing to salvage (not JSON, wrong shape), one more strict reading with a repair note.
4. Rules over the message itself (kind of request, dates, clock times, time of day, "the first one"; a short answer to a clinic or doctor question is used as the name, and the directory search checks it). A timeout or an unreachable server goes straight here.

When the reading came from rules, the agent **never picks the appointment to cancel or move**: even a single match is put to the patient, who names it. Nothing is invented, and a reply never claims a change happened.

## 2. How multi-turn state works

One conversation is one `HybridControllerModel` instance, owned by one `TextSession`, found through a `ConversationRegistry` by the **verified session id**. State is memory only: no table, no stored availability, no authorization.

`ConversationState`:

| Field | Meaning |
|---|---|
| `draft` | The merged request so far (operation, clinic, doctor, dates, time, selection) |
| `awaiting`, `last_question`, `offered` | What the agent last asked, and the choices it listed (each with the thing it stands for) |
| `prepared` | The change last prepared for review (its pending-action id and wording), or none |
| `pin_appointment`, `pin_slot` | A choice the patient made by position or time. Used only if the next fresh search still returns it |
| `turns` | Bounded at 40, then the conversation starts over |

Each patient message goes through the same steps:

1. **Screens, no model:** a credential-like message is answered with a notice; a bare "yes / okay / go ahead / confirm" and "never mind / forget it" are handled by rules.
2. **Read** (the ladder above), given the question just asked and the request so far, and asked to return only what the *latest* message says.
3. **Merge** into the draft; the latest message wins. A new operation starts a new request. A new clinic forgets the old clinic's doctor. A lone date in a reschedule means where to, once the existing day is known. A choice by position means what was offered in that position.
4. **Ask** if something essential is missing, otherwise **plan again from the start with fresh tool calls**: directory, appointments, availability, then prepare. Nothing found in an earlier turn is reused, so a changed date or doctor is a new search.
5. **Say** a reply written from the tool results.

| The patient… | The agent… |
|---|---|
| answers a question | merges the answer and continues |
| changes the date, time, doctor or clinic, or corrects it | merges it, drops the earlier preparation, searches again, prepares a new action |
| picks between choices ("the second one", "the 2 PM one", "Dr. Lee South") | resolves it against the offered list and the fresh results; asks again if it no longer fits |
| says yes / okay / go ahead | says it cannot confirm and points to Confirm in the app. No tool is called and the model is not asked |
| says never mind | forgets the request and says not to confirm the earlier review |
| asks something unrelated | says what it can help with and **forgets** the old request, so a later "9 AM then" cannot continue it |
| says nothing new | says the change is already prepared |

**Session scoping.** The registry key is the verified session id. A session id that now belongs to a different patient, or to none, gets a new conversation, never the old one. Signing in creates a new session, so an anonymous conversation does not carry into a signed-in one. Conversations expire when idle (30 minutes) and the total is bounded (500). A session calls tools only with its own principal; the model supplies no identity, session, or confirmation argument (the tool schemas forbid them).

**Safety boundary (unchanged and re-tested).** The model returns an intent and nothing else. It cannot call a tool, and the registry has no confirmation tool. Agreement in words is never treated as confirmation. The trusted UI confirms (authenticated, CSRF, Origin, idempotency key) and then tells the session with `confirmation_event`; only a reported commit of the *current* action is described as done, and a commit of an action the conversation has moved on from is acknowledged without the current details.

**Phase 3 interface.** `TextSession.respond(utterance) -> AgentReply(text, awaiting_confirmation)` and `confirmation_event(payload)`. Speech-to-text feeds the first and text-to-speech reads the reply; the agent knows nothing about either.

## 3. Test results

### Deterministic tests (no model; real tools, real PostgreSQL, real trusted-UI confirmation)

| What | Result |
|---|---|
| `tests/test_conversation_unit.py`: dates and clock times as patients write them; the extractor's strict reading, salvage and repair; the controller on a fake backend (missing time, partial doctor names, ambiguity, choices by position, changed requests, stale actions, agreement, dropping a request, unrelated questions, credentials, sign-in, every tool failure code, session isolation, expiry, bounds) | 115 tests |
| `tests/database/test_conversation.py`: 15 multi-turn conversations (below) run with the real tools against PostgreSQL, a stand-in for the model's reading, and a stand-in patient who presses Confirm only if the prepared action is what they asked for; plus end-state checks on replies, fresh searches, credentials and recovery; plus a `TextSession` driven through the real tool runtime and confirmed in the app | 27 tests |
| Safety checks inside every conversation | 0 safety violations: no double booking, no slot-state violation, no change without a trusted confirmation, no protected-data leak, no confirmation attempt by the agent |
| Mutation spot checks (restored byte-for-byte after each) | Removing the yes-is-not-confirmation rule, the credential screen, the no-guessing rule without the model, the stale-event guard, the placeholder guard, or the clearing of the earlier preparation each fails a test. The last one survived the first version of the suite and led to a new test |

The 15 conversations: **CV-01** clarification, then the date changes after a review (the design's example) · **CV-02** a different doctor after a review · **CV-03** a different clinic after a review · **CV-04** a partial doctor name fits two doctors · **CV-05** two appointments match, the patient says which by time · **CV-06** a reschedule, then another new day after the review · **CV-07** two appointments, the second chosen by position · **CV-08** the patient drops a prepared cancellation · **CV-09** a vague request completed by answering questions · **CV-10** an unrelated question must not let a later message continue the old booking · **CV-11** not signed in, then the patient types a PIN · **CV-12** no openings, then another day · **CV-13** the first reading is invalid · **CV-14** the model is unavailable for the whole conversation · **CV-15** the patient corrects the day with only its number. In each, the final appointment state is checked in the database, the trusted UI confirms only a matching action, and the patient's earlier, superseded request leaves nothing behind.

Related suites run after the change: the evaluation harness, the benchmark execution infrastructure, the real-model adapter, and the A/B unit tests (534 tests) all pass. Their stand-in readers only gained optional keyword arguments.

### Real-model smoke evaluation (local `qwen2.5:7b`, digest `845dbda0ea48`, temperature 0, seed 7, prompt `ab-b5`)

Run after the deterministic tests passed, through the real extractor, the real tools and the trusted UI, one run per conversation, nothing retried. Raw conversations are in Git-ignored `.local/conversation/`.

| Run | Result | What it showed |
|---|---|---|
| First (core, 15) | 9 of 15 reached the right end state; 0 safety violations | Five real defects, each fixed generally: the destination date of a move put in the wrong field (CV-05); "the 9 AM one" read as the time wanted (CV-06); a shortened clinic name matched exactly (CV-07); "book an appointment" read as "earliest" so the agent picked a time unasked (CV-10); "how about the 17th" returned as a move date (CV-12, CV-15) |
| Second (core, 15) | 14 of 15; 0 safety violations | CV-13 failed because the test wrapper discarded a recoverable reply; fixed in the wrapper |
| **Final (core, 15)** | **15 of 15; 0 safety violations; 0 provider errors** | 28 model calls; 13 readings were rejected as stating something the message did not and were salvaged; one conversation (CV-14, the outage) used no model |
| **Final (8 wording variants, written before they were run and not edited afterwards)** | **8 of 8; 0 safety violations** | 18 model calls; 10 readings salvaged |

How to read this: the three runs were used to find and fix defects, so the core 15/15 is a development result and optimistic. The variants are the fairer estimate, but they reuse the same worlds and one model, seed and wording each, so they do not show robustness to the range of real speech. At temperature 0 a second seed would not add variation. Most real-model readings needed correction (roughly half), which is why the checks against the message matter: the model alone would not have passed.

The frozen benchmark was not run, rerun, or tuned against. The official Approach B result (21/24 outcomes) belongs to prompt `ab-b4`; this layer is `ab-b5` and has no benchmark score.


## 6. Spoken dates and times

The text agent now reads dates and times the way people say them aloud. The code is `app/evaluation/spoken.py` (pure functions, no clock, no I/O) plus the merge in `hybrid.py`; the scheduling controller, the tools and the confirmation boundary are unchanged.

**How it works**

1. **The model reads what was said.** Qwen receives the patient's own words (names stay intact) for the operation, clinic, doctor and the like.
2. **The controller reads dates and times itself** from the same words, after masking the names the model found (a clinic called "Tomorrow Family Clinic" is not a date) and rewriting spoken numbers in date and time phrases into digits. A number word outside such a phrase is left alone: "Two Rivers Clinic" stays a name. The model's own date or time is used only where the words support it.
3. **A date said relative to today keeps how it was said** ("tomorrow", "next Tuesday", "November 17" with no year). Before the first availability search it is read again on the clinic's own calendar day, from the clinic's time zone (`ZoneInfo`, so daylight saving is handled by the standard library), not UTC.
4. **A phrase with two reasonable meanings is asked about:** "next Tuesday" said on a Monday (the coming Tuesday or the one after), and a time with no am or pm ("quarter past two"), unless a morning or afternoon word in the same message settles it.
5. **The answer is merged like any other:** "the tenth", "the second one", "PM", "in the afternoon", or the full date or time. What the patient said about the part of the day, or "the first available", is kept while the question is asked.

**Understood** (digits or words): "November seventeenth", "the twenty-fourth of November", "17 November 2027", "twenty twenty-six" as a year, "tomorrow", "the day after tomorrow", "this Friday", "next Monday", "the 18th" (in the month under discussion); "eleven thirty in the morning", "two forty-five in the afternoon", "ten o'clock", "nine oh five", "quarter past two", "half past three", "quarter to three", "twenty to four", "noon"; "after ten", "before eleven", "the first available appointment after ten"; "tomorrow afternoon", "next Tuesday morning".

**Handled safely**

| Case | What happens |
|---|---|
| A date that does not exist ("February thirtieth", "November thirty-first") | Not read as a date; the agent asks for the date |
| A date already past on the clinic's calendar | Not searched; asked about |
| A search longer than the backend's 31-day window | Not attempted; the patient is asked to narrow it |
| A time the clinic's clocks skip (02:30 on the spring change) | No opening at that time; the real times are offered |
| The same clock time twice (01:30 when the clocks go back) | Both are offered, told apart by their UTC offset, and the one chosen is booked |
| An evening request | Only mornings, afternoons or a particular time are searchable, so the patient is asked |
| Digits said aloud ("four eight two one") | Treated as a credential: a notice, never read, stored or sent to the model |
| A clinic or doctor name that contains a number or date word | Left intact |

**Not changed:** the backend validation, the prepare and confirm separation, the UI-only confirmation, session scoping, and the retirement of outdated reviews.

**Tests** (no model unless stated)

| What | Result |
|---|---|
| `tests/test_spoken_unit.py`: number words to digits (and what is left alone), every date form, ambiguity of "next weekday", re-reading a date on another calendar day, clock times settled or ambiguous, parts of the day | 84 tests |
| `tests/test_conversation_unit.py` (spoken section): the controller on a fake backend with clinics in two time zones: each spoken phrase, every question and its answers, corrections across turns, invalid and past dates, the 31-day limit, clinic names holding number or date words, spoken digits as a credential, daylight-saving cases | 41 new tests |
| `tests/database/test_conversation.py` (spoken section): 7 spoken conversations on the real tools, run with a reader that leaves every date and time to the controller; the hour the clocks go back, with both 01:30 slots offered and the chosen one booked | 10 new tests |
| Mutation spot checks (restored after each) | Reading "tomorrow" on the UTC day, guessing "next Tuesday", not asking about an ambiguous time, allowing a past date, not treating spoken digits as a credential, not telling the repeated hour apart, skipping the 31-day limit, and assuming am for a bare hour each fail a test |

Two defects were found by the database scenarios, not by the unit tests, and fixed generally: "after ten at the clinic" was not recognised as a time (the phrase was only converted before a few words), and a clinic whose name contains "Tomorrow" had its name read as a second date. Dates and times are now scanned with the names the model found masked out.

**Real model** (`qwen2.5:7b`, digest `845dbda0ea48`, temperature 0, seed 7, prompt `ab-b6`; one run each, nothing retried, no prompt tuning; raw conversations in Git-ignored `.local/conversation/`)

| Set | Result |
|---|---|
| 7 spoken conversations (SP-01 to SP-07) | **7 of 7**, 0 safety violations, 11 model calls |
| 4 spoken wording variants, written before they were run and not edited | **3 of 4**, 0 safety violations |
| The 15 earlier conversations, rerun on the new code | 15 of 15, 0 safety violations |
| The 8 earlier wording variants, rerun | 8 of 8, 0 safety violations |

The one failure, **SP-01V**, was "Could I get in at the clinic on November sixteenth, say half past eleven in the morning?". The agent asked "Would you like to book, move, or cancel an appointment?" and booked nothing. The cause is not the date or the time: the rule that an operation must be backed by words in the message does not list "get in" as a booking phrase, so the model's correct reading was set aside. That is the safe direction (a question, no action), and it is a vocabulary gap in a rule written before this work. I did not widen the list after seeing the failure, so the variant stays a fair record of the miss.

How to read this: the spoken conversations were written while the layer was developed, and the defects they found were fixed before the real model was run on them, so 7 of 7 is a development result. The variants are the fairer estimate, and 4 conversations do not estimate a failure rate. In the real runs the controller, not the model, did the date and time reading: the model's own reading of a spoken time was usually unusable (it was rejected and salvaged in 10 of 11 calls), which is what the design assumes.


## 4. Remaining limitations

- **One live review per session, enforced in the backend.** A patient who changes or drops a request cannot confirm the old review from a screen that is still open. A newer preparation retires the session's other `ACTIVE` pending actions in its own transaction; when a request is dropped or changed with nothing new to prepare, the session retires the review through a trusted runtime method (`discard_pending_action`, not a tool). See section 5. Limits: (1) a patient may hold up to three independent sessions (authentication policy), each with its own review and its own conversation; a change in one session never retires another session's review, by design, and the oldest session is revoked, with its review, when a fourth signs in. (2) The retirement after a drop happens when the agent's reply is produced, so a Confirm pressed in the same instant as the message can still win (it is a real, explicit confirmation of what the patient was shown). (3) If the retirement fails (database down), the review stays confirmable until it expires; the reply then says plainly that the earlier review may still be open and not to confirm it, and the next turn tries again. Nothing is reported as retired that the backend did not confirm. A review screen cannot learn on its own that it was retired until it is used: the confirm call is the check. A screen that polled the pending action's status, or a durable retry of failed retirements, would close this; both are outside this change.
- **State is in process memory.** It is lost on restart and not shared between workers. Phase 3 needs sticky routing for a session, or the draft stored with the session (it holds no secrets, only what the patient said).
- **Spoken language.** Dates and times said in words are now read (section 6). Still unsupported: years said in words beyond 2020-2039 ("two thousand and forty"), "around" or "about" a time, ranges ("between ten and two"), "half an hour from now", "in two weeks", "a week from Friday", "Tuesday week", "noon-ish", dates in other orders or calendars, and a day number alone from the first to the fifth ("the third": ambiguous with a choice from a list). A misheard clinic or doctor name is not corrected by fuzzy matching.
- **What can identify an appointment:** clinic, doctor, date, a clock time, or a position in the list offered. "The afternoon one", "the later one", and "the one with Dr. Kim last week" are not understood; the agent asks again.
- **One request at a time.** A message that books one appointment and cancels another is read as the first operation only.
- **Unrelated questions are not answered** (hours, address, directions). They end the old request on purpose. A parked request that resumes would be friendlier and is a larger change.
- **Relative dates use the clinic's calendar, but only once a clinic is known.** Until then a provisional UTC day is used and never sent to a tool; it is read again on the clinic's own day before the first availability search. A patient who is in a different time zone from the clinic gets the clinic's "tomorrow", by design. For cancel and move requests the appointments list is fetched with a two-day margin and filtered by each appointment's own clinic calendar. A date with no year is its next occurrence; a date that has already passed on the clinic's calendar is asked about, not searched.
- **The model.** Qwen 7B reads each message once; about half of the real readings contained at least one value the message did not support. The checks make that safe, but each rejected value becomes a question for the patient, so a weaker reading means more questions, not wrong actions. Determinism at temperature 0 is not robustness. Latency is one model call per message plus the tools.
- **Evidence is small.** 23 real-model conversations, one run each, on synthetic data. They show that the design works and that safety held; they do not estimate a failure rate. A larger, independent set of conversations should precede any claim about reliability.
- **Reply wording is fixed text** built from tool results. It is accurate by construction, not natural, and it spells out dates in full ("Monday, November 16, 2026"), which is long for speech.

### Is the text agent ready for Phase 3 voice integration?

**At the interface level, yes; as a voice product, not yet.** The agent is text in, text out behind `TextSession`, is scoped to the verified session, never confirms anything, never asks for a credential, and recovers from a failed reading without leaving the patient silent. What must be settled first: (1) number-word and spoken-time normalization (or readers for them); (2) where the conversation state lives across workers; (3) shorter spoken wording for dates and lists; and (4) a larger independent conversation set, ideally with real speech transcripts of synthetic callers, before any reliability claim.

## 5. Outdated reviews cannot commit

Before this change a review (a pending action waiting for Confirm) stayed `ACTIVE` for ten minutes whatever the patient said next. Now:

| Event | What the backend does |
|---|---|
| A newer preparation in the session | In the preparing transaction, after the session row lock, every other `ACTIVE` pending action of that session becomes `INVALIDATED`. Audit: the prepared event records how many (`superseded`) |
| The patient drops or changes a request and nothing new is prepared | The session calls `ToolRuntime.discard_pending_action(principal, id)`, which retires that one action of its own session and writes an `APPOINTMENT_ACTION_DISCARDED` audit event. It is not in the tool registry, so a model cannot name it |
| Confirm on a retired review | `409 CONFLICT`, reason `CONFIRMATION_INVALIDATED`, nothing committed |
| Idempotent replay of an old preparation, or a preparation that fails | Retires nothing |
| A committed (`CONSUMED`) action | Keeps its status |
| Another session of the same patient, or another patient | Never touched (a patient may hold up to three independent sessions, each with its own review); discarding a foreign or unknown id is a quiet no-op that cannot be told apart |

The code is in `app/services/pending_action_retirement.py`, separate from the prepare surface (a test pins that surface to one exported function). Both retirements and confirmation take the session row lock first, so they cannot interleave: a confirmation either commits before (and the action is `CONSUMED`) or finds the action `INVALIDATED`. Retiring only removes the ability to confirm; it creates, moves and cancels nothing, and authentication, idempotency, session isolation and UI-only confirmation are unchanged.

Tests: `tests/database/test_pending_action_retirement.py` (supersession, replay, failed preparation, other patients, committed actions, audit, discard, authorization, and three races run 12, 12 and 8 times each: confirm against a newer preparation, confirm against a discard, and two preparations at once, all checking that an appointment exists exactly when its action was consumed); `tests/database/test_conversation.py` (an open screen cannot commit after "never mind", after a change with no new review, or after a change that prepares a new one); `tests/test_conversation_unit.py`. Six older tests assumed several live reviews in one session; their setup was changed to keep what each proves (see the change report), not their assertions.
