# Architecture Benchmark: Execution Infrastructure

**Status:** built, tested and used for the official comparison, which was run once (96 of 96 runs). Its frozen decision rule output is `NEITHER_SELECTED`. The tests themselves use no language model. See "Outcome and unresolved historical items" at the end.
**Belongs to:** `docs/agent-architecture-evaluation-protocol.md`, which is frozen and unchanged. That document's "Status of the pieces" table is superseded by the "Outcome and unresolved historical items" section at the end.

This is the machinery that runs the frozen 24-scenario benchmark identically for Approach A and Approach B, and scores it with the frozen protocol. The manifest (SHA-256 `077e9fc0d74c691274edb1b14b17a05e5950b50d48aa8dc727f08abd53e21c9f`) and the thresholds (`0a729d6067bb6262a642229b9b3134a9d472eea41c64b1f75df2140cd06e48a7`) are unchanged, and tests pin both.

## How a run works

For each scenario, each seed, and each approach:

1. A **fresh disposable PostgreSQL database** is created (migrated and loaded with the demonstration seed), and dropped afterwards.
2. The scenario's world is **built from the manifest** by one function (`benchmark_builder.build_world`): its clinics, doctors, slots, the verified patient, the other patient, and the existing bookings.
3. The database is **read back** through separate SQL and compared with the manifest's own definition of the initial state. The real public search tools are also called and compared with what the manifest says an honest search must return. If anything differs, the run stops with `BenchmarkEnvironmentError` and is never scored.
4. The approach runs through the **existing evaluation runner**: the real tools, a real authenticated session (or none, for the unauthenticated scenarios), and the trusted-UI confirmation over the real HTTP endpoint. The simulated patient confirms only if the stored pending action is what the manifest says the patient wanted (any acceptable slot, for a reschedule).
5. The database is read back again and compared with the manifest's expected end state, and the demonstration data is checked to be untouched.
6. The result is scored with the frozen protocol functions.

The approach is only a factory. Nothing else differs between A and B. Approaches alternate scenario by scenario (the one going first alternates), after one unscored warm-up request.

## What was verified independently

| Claim | How it is checked |
|---|---|
| Each of the 24 worlds matches the manifest | Parametrized test builds each in its own database, reads it back, and compares it with the manifest's state **and** with the frozen fingerprint function |
| A database that does not match is detected | 12 tamper tests: a moved, removed, or lengthened slot; a slot booked with no appointment; an open slot with an appointment; a renamed doctor or clinic; a changed specialty; a wrong time zone (alone, and shifted consistently with the times); another scenario's clinic present |
| The manifest agrees with the real tools | Availability and directory searches are run for each scenario and compared with what the manifest derives; tampering with a specialty or surname is caught |
| An invalid world is never scored | A test makes the builder fault; the run raises and scores nothing |
| Scenarios are isolated | Every run gets its own database (counted for two approaches, two seeds); a scenario leaves nothing for the next; a second scenario built into the same database is flagged |
| Runs are reproducible | The same scenario gives the same rows, patient, and demonstration-data fingerprint twice; a repeated run gives identical scores apart from timing |
| Equivalent starting conditions | Both approaches get the same initial-state fingerprint, request, patient, and demonstration data for scenarios of every kind, including unauthenticated; comparability fails if any of them, the model, or the manifest differs, or is unrecorded |
| A correct run is scored correct | A scripted reference approach solving all 24 through the real tools passes every gate and core criterion (24/24 outcomes and intent, 25/25 complete action replies, zero safety, accuracy, or leak findings) |
| Each defect is scored as that defect | Pipeline tests with one defect at a time: wrong target, a corrected wrong target, an unrequested filter, an invalid tool call, a wrong time in the review or the final reply, a success claim before confirmation, a clarification that omits the choices, a leak to an unauthenticated caller, a preparation where none belongs, a confirmation attempt, "done" in place of a refusal |
| Fixed templates are neither favored nor penalized | A template is accurate and adequate but incomplete; it fails a clarification; it passes an honest failure notice; and a template that claims "booked" after a rejected confirmation is caught |
| Defects across a whole run fail the matching gates | Ten defects scattered through a 24-scenario run fail exactly the expected gates and core criteria, and the decision rule never selects the flawed run over a sound one |
| Nothing published holds conversation content | A sentinel in every reply, plus the requests, tool arguments, patient codes, and display names, are searched for in the report; the report's own guard is shown to fire |
| The review sheet is blind | No approach, seed, scenario id, or rule verdict appears on it; items are shuffled; a separate key maps them back |
| The review sheet picks what the protocol says | Every inaccurate or mixed-verdict reply, plus a random 25% of the rest, repeatable by seed |
| The 10% rule | A verdict is reliable at exactly 10% disagreement and unreliable above it |
| Raw conversations stay out of Git | Output goes only to a path Git ignores (`.local/` is now ignored) or outside the repository; others are refused and not created |
| The two approaches are built identically | Same weights, sampling, limits, timeout, and seed in both configurations |
| A remote model server is refused | Before any connection is attempted |

The scoring code was also mutation-tested: about 50 deliberate faults in the builder, scorer, runner, review, and command line, each required to be caught by a test. Several survived at first and each exposed a real gap in a test, now closed.

## Defects this work found and fixed

- **The published metadata contained the synthetic patients' login codes.** The comparison needs them equal, not published; it now records a hash.
- **The runner reports `UNEXPECTED_TOOL_CALL` on every call when there is no reference call path.** That is an artifact of the benchmark prescribing no path, so it is never published as a finding.
- **The first command line ran one approach entirely and then the other,** contradicting the protocol's alternation. It now alternates and warms up.

## Changes to shared code

All additive and off by default; the evaluation tests that already existed still pass.

- `runner.py`: an optional fixed run suffix (so the same scenario gives the same patient), and an optional in-memory conversation record (the scorer needs the actual reply text and tool arguments; reports never contain it).
- `intent.py`: a `OneOf` target for the trusted-UI policy, so a reschedule to any acceptable new slot is confirmed.
- `.gitignore`: `.local/`, the place for raw transcripts and review sheets.

Production scheduling code, tools and their schemas, prompts, and both approaches are untouched.

## Commands

```
python -m app.evaluation.benchmark_run run --output-dir .local/benchmark/<date>   # both approaches, alternating
python -m app.evaluation.benchmark_run assemble --output-dir .local/benchmark/<date>
python -m app.evaluation.benchmark_run audit --output-dir .local/benchmark/<date> --judgments <file>
```

`run` needs the local model server and exits with status 2 if it cannot be used. `assemble` writes `comparison.json` (a decision only if the runs are comparable), the review sheet, the key, and a blank judgments file. `audit` reads the reviewer's judgments and reports whether each verdict can be trusted.

## Outcome and unresolved historical items

**Outcome.** The official run (two approaches, 24 scenarios, two seeds, alternating, after an unscored warm-up) finished with no interruption. The frozen rule's mechanical output is `NEITHER_SELECTED`: Approach A failed two gates and all six core criteria; Approach B passed every gate and missed three core criteria by one scenario each (21 of 24 correct outcomes against 13 of 24 for A, under the prompt versions then current). The seeds gave identical outcomes at temperature 0, so that is determinism, not robustness. This is not a benchmark pass for either approach. Approach B was afterwards selected for continued development by the human reviewer, which is an engineering decision, not a result of the rule. It was later changed (a new prompt version, multi-turn state and spoken-language handling), so the benchmark score does not describe the current agent; the current agent was judged on a separate, frozen acceptance evaluation (`docs/phase2-acceptance-evaluation.md`).

**Unresolved or limiting items from that run, recorded rather than resolved:**

- **The blinded human audit was not carried out.** The protocol requires a person to check a sample of the rule-based reply verdicts (accurate, adequate, complete) and to report a verdict as unreliable if they disagree on more than 10 percent. A 59-item review sheet and its separate key were generated (Approach A 42, Approach B 17), but no judgments were recorded, so the agreement was never measured. The reply-level verdicts are therefore rule-based and unaudited, and no claim in the repository depends on them being reliable. No audit result has been invented.
- **A scorer limitation, found while auditing, was not corrected** because the protocol was frozen: the intent scorer does not strip a title from a doctor filter, so Approach A's clarification in BM-018 with a filter such as "Dr. Park" was marked as an unauthorized doctor filter. It lowers Approach A's intent score by at most one (10 to 11) and changes no gate, core result or decision. The protocol's procedure for a scorer fix (correct it once and re-score both approaches) could not be fully applied, because tool arguments and end states were not saved, only replies and their verdicts.
- **Approach A scored zero on the unauthenticated scenarios in part because it asked for a PIN in chat in two of them.** No gate names credential solicitation; it was caught only through reply adequacy. Approach B never does this.
- **Single wording per scenario, fixed date formats, single-turn only.** The benchmark cannot show a statistical difference and did not test follow-up turns or relative dates.
- **Raw transcripts and the review materials are kept outside Git** (`.local/`), as the protocol requires; only sanitized counts and codes are reported.
