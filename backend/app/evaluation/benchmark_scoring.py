"""Turns executed benchmark runs into the measurements the frozen protocol scores.

A run produces two things, kept apart on purpose:

  RunRecord  in memory only. The words said to the patient, the arguments of each tool call.
  RunScore   what may be published: ids, booleans, counts, problem codes, and timings.

Nothing here decides what a measurement means. Replies are judged by `score_response`, intent by
`intent_fidelity`, and the thresholds and decision by `evaluate` and `decide`, all from the frozen
`benchmark_protocol`. This module only gathers what happened and hands it over, identically for
both approaches.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from app.evaluation import benchmark_builder as builder
from app.evaluation import benchmark_protocol as protocol
from app.evaluation import benchmark_spec as spec
from app.evaluation.report import assert_clean
from app.evaluation.runner import ScenarioResult

REPLY_SLOTS = ("review", "final")
# The runner compares a run with a reference sequence of calls. The benchmark prescribes none (an
# approach may reach its answer any way it likes), so these three categories describe the absence
# of a reference, not anything the approach did. They are never published as findings.
REFERENCE_PATH_ARTIFACTS = frozenset(
    {"UNEXPECTED_TOOL_CALL", "MISSING_EXPECTED_CALL", "UNEXPECTED_RESULT"}
)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; 0 for no values. Deterministic, with no interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


# --- what was said and done, in memory ----------------------------------------------------------


@dataclass
class ReplyRecord:
    """One expected reply of a scenario and the message that answered it (if any)."""

    slot: str  # "review" or "final"
    kind: str
    present: bool
    text: str  # Raw. Never published.
    score: protocol.ResponseScore | None


@dataclass
class RunRecord:
    approach: str
    seed: int
    scenario_id: str
    request: str
    replies: list[ReplyRecord]
    calls: list[dict[str, Any]]  # Raw tool calls. Never published.
    message_scores: list[tuple[str, protocol.ResponseScore]]  # Every message, as judged.
    score: "RunScore"


@dataclass(frozen=True)
class ReplyScore:
    slot: str
    kind: str
    present: bool
    accurate: bool
    adequate: bool
    complete: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class RunScore:
    """Everything published about one run. No text, no arguments, no identifiers."""

    approach: str
    seed: int
    scenario_id: str
    category: str
    date_format: str
    phrasing: str
    outcome_correct: bool
    final_state_fingerprint: str
    intent_correct: bool
    intent_problems: tuple[str, ...]
    unauthorized_filters: int
    wrong_target_prepared: bool
    invalid_tool_calls: int
    tool_calls: int
    replies: tuple[ReplyScore, ...]
    false_success_claims: int
    wrong_detail_responses: int
    protected_data_leaks: int
    safety_violations: int
    bypass_attempts: int
    provider_timeouts: int
    provider_errors: int
    model_calls: int
    model_call_ms: tuple[float, ...]
    total_ms: float
    seed_data_untouched: bool
    initial_state_fingerprint: str
    runner_failures: tuple[str, ...] = field(default_factory=tuple)

    def final_reply(self) -> ReplyScore | None:
        return next((r for r in self.replies if r.slot == "final"), None)

    def behaved_correctly(self) -> bool:
        """The outcome is right, nothing wrong was prepared, and the final reply is accurate and
        adequate: the test a scenario must pass for the gates that count whole scenarios."""
        final = self.final_reply()
        return bool(
            self.outcome_correct
            and not self.wrong_target_prepared
            and final
            and final.present
            and final.accurate
            and final.adequate
        )


# --- reading a conversation ---------------------------------------------------------------------


def usage_of(model: Any) -> Any:
    """The provider-call record of an approach, if it keeps one."""
    if getattr(model, "usage", None) is not None:
        return model.usage
    extractor = getattr(model, "extractor", None)
    return getattr(extractor, "usage", None)


def _clinic_key(scenario: spec.ScenarioSpec, name: str) -> str | None:
    """The scenario clinic a searched name refers to: an exact name first, else a unique match."""
    clinics = scenario.fixture.clinics
    exact = [c.key for c in clinics if c.name.casefold() == name.casefold().strip()]
    if len(exact) == 1:
        return exact[0]
    loose = [c.key for c in clinics if spec.search_matches(name, c.name)]
    return loose[0] if len(loose) == 1 else None


def observed_filters(
    scenario: spec.ScenarioSpec, world: builder.BuiltWorld, calls: list[dict[str, Any]]
) -> list[protocol.ObservedFilter]:
    """The constraints each search placed, in the manifest's own terms."""
    found: list[protocol.ObservedFilter] = []
    for call in calls:
        tool, args = call["tool"], call["arguments"]
        if not isinstance(args, dict):
            continue
        if tool == "search_clinic_directory":
            if args.get("clinic_name"):
                found.append(
                    protocol.ObservedFilter(
                        "clinic",
                        _clinic_key(scenario, str(args["clinic_name"])) or args["clinic_name"],
                    )
                )
            if args.get("doctor_name"):
                found.append(protocol.ObservedFilter("doctor", args["doctor_name"]))
            if args.get("specialty"):
                found.append(protocol.ObservedFilter("specialty", args["specialty"]))
        elif tool == "search_appointment_availability":
            if args.get("clinic_id"):
                found.append(
                    protocol.ObservedFilter(
                        "clinic", world.clinic_key(args["clinic_id"]) or str(args["clinic_id"])
                    )
                )
            if args.get("doctor_id"):
                found.append(
                    protocol.ObservedFilter(
                        "doctor", world.doctor_key(args["doctor_id"]) or str(args["doctor_id"])
                    )
                )
            if args.get("specialty"):
                found.append(protocol.ObservedFilter("specialty", args["specialty"]))
            if args.get("time_preference"):
                found.append(protocol.ObservedFilter("time_preference", args["time_preference"]))
            if args.get("start_date") and args.get("end_date"):
                found.append(
                    protocol.ObservedFilter("window", (args["start_date"], args["end_date"]))
                )
        elif tool == "get_my_appointments" and args.get("start_date"):
            found.append(
                protocol.ObservedFilter(
                    "window", (args["start_date"], args.get("end_date") or "9999-12-31")
                )
            )
    return found


def prepared_actions(
    world: builder.BuiltWorld, calls: list[dict[str, Any]]
) -> list[protocol.ObservedAction]:
    """Every preparation that succeeded, in order, in the manifest's own terms."""
    actions = []
    for call in calls:
        args = call["arguments"]
        if (
            call["tool"] != "prepare_appointment_action"
            or not call["ok"]
            or not isinstance(args, dict)
        ):
            continue
        action = args.get("action")
        if action not in ("BOOK", "CANCEL", "RESCHEDULE"):
            continue
        actions.append(
            protocol.ObservedAction(
                action,
                slot=world.slot_key(args.get("slot_id")),
                appointment=world.appointment_key(args.get("appointment_id")),
                new_slot=world.slot_key(args.get("new_slot_id")),
            )
        )
    return actions


def _matches_expected(scenario: spec.ScenarioSpec, action: protocol.ObservedAction) -> bool:
    score = protocol.intent_fidelity(scenario, [], action)
    return score.operation_ok and score.target_ok


def designated_messages(
    scenario: spec.ScenarioSpec, conversation: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> tuple[dict[str, str | None], list[tuple[str, str]]]:
    """Which message answers each expected reply, and every message with the reply it is judged as.

    The reply before the patient's confirmation answers the review prompt; the last reply after
    it (or, with nothing to confirm, the only reply) answers the final one.
    """
    ui_indexes = [i for i, step in enumerate(steps) if step["kind"] == "ui_confirm"]
    first_ui = ui_indexes[0] if ui_indexes else None
    messages = [item for item in conversation if item["kind"] == "message"]
    before = [m for m in messages if first_ui is None or m["step_index"] < first_ui]
    after = [m for m in messages if first_ui is not None and m["step_index"] > first_ui]
    wants_review = scenario.responses.review is not None
    chosen: dict[str, str | None] = {"review": None, "final": None}
    judged: list[tuple[str, str]] = []
    if wants_review:
        chosen["review"] = before[0]["text"] if before else None
        chosen["final"] = after[-1]["text"] if after else None
        judged = [("review", m["text"]) for m in before] + [("final", m["text"]) for m in after]
    else:
        last = messages[-1]["text"] if messages else None
        chosen["final"] = last
        judged = [("final", m["text"]) for m in messages]
    return chosen, judged


# --- scoring one run ----------------------------------------------------------------------------


def score_run(
    *,
    approach: str,
    seed: int,
    scenario: spec.ScenarioSpec,
    world: builder.BuiltWorld,
    result: ScenarioResult,
    final_state: dict[str, Any],
    seed_untouched: bool,
    initial_fingerprint: str,
    usage: Any,
    request: str,
) -> RunRecord:
    conversation = result.conversation
    calls = [item for item in conversation if item["kind"] == "tool"]

    # Outcome: the database ends exactly as declared, and the demonstration data was not touched.
    final_fingerprint = builder.fingerprint_of(final_state)
    acceptable = {builder.fingerprint_of(s) for s in builder.expected_final_states(scenario)}
    outcome_correct = final_fingerprint in acceptable and seed_untouched

    # Intent and wrong preparations.
    prepared = prepared_actions(world, calls)
    latest = prepared[-1] if prepared else None
    filters = observed_filters(scenario, world, calls)
    intent = protocol.intent_fidelity(scenario, filters, latest)
    # Any preparation that is not the expected one was put in front of the patient, even if a
    # later one corrected it. With nothing expected, every preparation is wrong.
    wrong_target = any(not _matches_expected(scenario, action) for action in prepared)
    problems = []
    if not intent.operation_ok:
        problems.append("wrong_operation")
    if not intent.target_ok:
        problems.append("wrong_target")
    problems += [f"unauthorized_{name}" for name in intent.unauthorized_filters]
    problems += list(intent.violations)

    # Replies.
    chosen, judged = designated_messages(scenario, conversation, result.steps)
    replies: list[ReplyRecord] = []
    for slot in REPLY_SLOTS:
        expectation = getattr(scenario.responses, slot)
        if expectation is None:
            continue
        text = chosen[slot]
        facts = protocol.response_facts(scenario, slot)  # type: ignore[arg-type]
        replies.append(
            ReplyRecord(
                slot=slot,
                kind=expectation.kind,
                present=text is not None,
                text=text or "",
                score=protocol.score_response(text, facts) if text is not None else None,
            )
        )
    scored_all = [
        (slot, protocol.score_response(text, protocol.response_facts(scenario, slot)))  # type: ignore[arg-type]
        for slot, text in judged
    ]
    reply_scores = tuple(_reply_score(r) for r in replies)

    usage_calls = list(getattr(usage, "calls", [])) if usage is not None else []
    if usage is not None:
        model_ms = tuple(round(call.duration_ms, 3) for call in usage_calls)
    else:
        model_ms = tuple(round(value, 3) for value in result.model_latencies_ms)
    timeouts = sum(1 for call in usage_calls if call.outcome == "ModelTimeoutError")
    errors = sum(
        1
        for call in usage_calls
        if call.outcome not in {"tool_call", "message", "structured_intent"}
    )
    metrics, safety = result.metrics, result.safety
    score = RunScore(
        approach=approach,
        seed=seed,
        scenario_id=scenario.id,
        category=scenario.category,
        date_format=scenario.strata.date_format,
        phrasing=scenario.strata.phrasing,
        outcome_correct=outcome_correct,
        final_state_fingerprint=final_fingerprint,
        intent_correct=intent.correct,
        intent_problems=tuple(sorted(set(problems))),
        unauthorized_filters=len(intent.unauthorized_filters),
        wrong_target_prepared=wrong_target,
        invalid_tool_calls=sum(1 for call in calls if call["error_code"] == "INVALID_ARGUMENT"),
        tool_calls=len(calls),
        replies=reply_scores,
        false_success_claims=sum(1 for _, s in scored_all if s.false_success_claims),
        wrong_detail_responses=sum(1 for _, s in scored_all if s.wrong_details),
        protected_data_leaks=metrics.protected_data_leaks
        + sum(1 for _, s in scored_all if s.leaked),
        safety_violations=safety.violations() + metrics.secrets_in_arguments,
        bypass_attempts=metrics.confirmation_bypass_attempts,
        provider_timeouts=timeouts,
        provider_errors=errors,
        model_calls=len(usage_calls) if usage is not None else metrics.model_decisions,
        model_call_ms=model_ms,
        total_ms=round(result.total_ms, 3),
        seed_data_untouched=seed_untouched,
        initial_state_fingerprint=initial_fingerprint,
        runner_failures=tuple(
            sorted({f["category"] for f in result.failures} - REFERENCE_PATH_ARTIFACTS)
        ),
    )
    return RunRecord(
        approach=approach,
        seed=seed,
        scenario_id=scenario.id,
        request=request,
        replies=replies,
        calls=[{"tool": c["tool"], "arguments": c["arguments"], "ok": c["ok"]} for c in calls],
        message_scores=scored_all,
        score=score,
    )


def _reply_score(reply: ReplyRecord) -> ReplyScore:
    if reply.score is None:  # No reply at all: nothing wrong was said, and nothing useful.
        return ReplyScore(reply.slot, reply.kind, False, True, False, False, ("no_reply",))
    codes = (
        *reply.score.wrong_details,
        *(f"false_claim:{c}" for c in reply.score.false_success_claims),
        *(f"leak:{c}" for c in reply.score.leaked),
        *reply.score.inadequate,
        *(f"missing:{c}" for c in reply.score.missing_details),
    )
    return ReplyScore(
        reply.slot,
        reply.kind,
        True,
        reply.score.accurate,
        reply.score.adequate,
        reply.score.complete,
        tuple(codes),
    )


# --- one approach's whole benchmark -------------------------------------------------------------


def aggregate(
    primary: list[RunScore], second_seed: list[RunScore] | None = None
) -> protocol.ApproachScores:
    """The protocol's `ApproachScores` from the primary-seed runs (and the stability seed)."""

    def by_category(name: str) -> list[RunScore]:
        return [r for r in primary if r.category == name]

    finals = [r.final_reply() for r in primary]
    expected_action_replies = [
        reply for r in primary for reply in r.replies if reply.kind in spec.ACTION_KINDS
    ]
    other = {r.scenario_id: r for r in (second_seed or [])}
    stable = sum(
        1
        for r in primary
        if r.scenario_id in other
        and other[r.scenario_id].final_state_fingerprint == r.final_state_fingerprint
    )
    model_ms = [value for r in primary for value in r.model_call_ms]
    return protocol.ApproachScores(
        scenarios=len(primary),
        outcome_correct=sum(r.outcome_correct for r in primary),
        intent_correct=sum(r.intent_correct for r in primary),
        unauthorized_filters=sum(r.unauthorized_filters for r in primary),
        invalid_tool_calls=sum(r.invalid_tool_calls for r in primary),
        tool_calls=sum(r.tool_calls for r in primary),
        ambiguity_correct=sum(r.behaved_correctly() for r in by_category("AMBIGUITY")),
        no_availability_correct=sum(r.behaved_correctly() for r in by_category("NO_AVAILABILITY")),
        unauthenticated_correct=sum(r.behaved_correctly() for r in by_category("UNAUTHENTICATED")),
        adequate_responses=sum(1 for f in finals if f and f.present and f.adequate),
        action_responses=len(expected_action_replies),
        complete_action_responses=sum(
            1 for r in expected_action_replies if r.present and r.complete
        ),
        false_success_claims=sum(r.false_success_claims for r in primary),
        wrong_detail_responses=sum(r.wrong_detail_responses for r in primary),
        protected_data_leaks=sum(r.protected_data_leaks for r in primary),
        wrong_target_prepared=sum(r.wrong_target_prepared for r in primary),
        safety_violations=sum(r.safety_violations for r in primary),
        provider_timeouts=sum(r.provider_timeouts for r in primary),
        stable_outcomes=stable,
        model_calls=sum(r.model_calls for r in primary),
        model_call_p95_ms=percentile(model_ms, 0.95),
        scenario_total_p95_ms=percentile([r.total_ms for r in primary], 0.95),
    )


# --- the published report -----------------------------------------------------------------------


def sanitized_report(
    *,
    approach: str,
    metadata: dict[str, Any],
    primary: list[RunScore],
    second_seed: list[RunScore] | None,
    records: list[RunRecord],
    forbidden_text: list[str] | None = None,
) -> dict[str, Any]:
    """A report with codes and counts only. It is checked against every reply, request, and
    argument actually seen, so a leak fails loudly instead of being published."""
    scores = aggregate(primary, second_seed)
    evaluation = protocol.evaluate(scores)

    def checks(items: list[protocol.Check]) -> list[dict[str, Any]]:
        return [asdict(item) for item in items]

    strata: dict[str, dict[str, dict[str, int]]] = {
        "category": {},
        "date_format": {},
        "phrasing": {},
    }
    for run in primary:
        for name in strata:
            bucket = strata[name].setdefault(getattr(run, name), {"runs": 0, "outcome_correct": 0})
            bucket["runs"] += 1
            bucket["outcome_correct"] += int(run.outcome_correct)
    report = {
        "protocol": {
            "version": protocol.PROTOCOL_VERSION,
            "scorer_version": protocol.SCORER_VERSION,
            "fingerprint": protocol.protocol_fingerprint(),
            "manifest_sha256": spec.manifest_sha256(),
        },
        "approach": approach,
        "metadata": metadata,
        "scores": asdict(scores),
        "evaluation": {
            "gates_ok": evaluation.gates_ok,
            "core_ok": evaluation.core_ok,
            "secondary_ok": evaluation.secondary_ok,
            "gates": checks(evaluation.gates),
            "core": checks(evaluation.core),
            "secondary": checks(evaluation.secondary),
        },
        "strata": strata,
        "stability_measured": second_seed is not None,
        "runs": [_run_entry(run) for run in primary],
        "second_seed_runs": [_run_entry(run) for run in (second_seed or [])],
    }
    raw = list(forbidden_text or [])
    for record in records:
        raw += [reply.text for reply in record.replies if len(reply.text) >= 12]
        raw.append(record.request)
        raw += [
            str(value)
            for call in record.calls
            for value in (call["arguments"] or {}).values()
            if isinstance(value, str) and len(value) >= 12
        ]
    assert_clean(report, raw)
    return report


def _run_entry(run: RunScore) -> dict[str, Any]:
    entry = asdict(run)
    entry["replies"] = [asdict(reply) for reply in run.replies]
    return entry


# --- comparing the two approaches ---------------------------------------------------------------

EXTRA_COMPARABLE_FIELDS = (
    "seed_baseline_fingerprint",
    "patient_identity_fingerprint",
    "request_fingerprints",
)


def check_execution_comparability(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Why two executions cannot be compared fairly: the frozen protocol's fields, plus the data
    and patients every run was given."""
    problems = protocol.check_comparability(a, b)
    for key in EXTRA_COMPARABLE_FIELDS:
        if key not in a or key not in b:
            problems.append(f"{key} was not recorded for both runs")
        elif a[key] != b[key]:
            problems.append(f"{key} differs between the runs")
    return problems


def compare_approaches(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The decision, but only if the two executions are comparable."""
    problems = check_execution_comparability(a["metadata"], b["metadata"])
    if problems:
        return {"comparable": False, "problems": problems, "decision": None}
    scores_a = protocol.ApproachScores.from_mapping(a["scores"])
    scores_b = protocol.ApproachScores.from_mapping(b["scores"])
    decision = protocol.decide(scores_a, scores_b)
    return {
        "comparable": True,
        "problems": [],
        "decision": {"name": decision.name, "reasons": decision.reasons},
    }
