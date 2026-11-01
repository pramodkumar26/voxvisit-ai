"""The benchmark's execution infrastructure against PostgreSQL.

These tests run no language model. They use a scripted reference approach (benchmark_support) that
solves each scenario through the real tools, with switchable defects, to show that:

  - each scenario's database matches the manifest, and a database that does not is detected
  - every run has its own database, and the same scenario always starts from the same rows
  - both approaches are given the same world, patient, request, and session
  - a correct run is scored correct, and each kind of defect is scored as that defect
  - nothing published contains a reply, a request, or an identifier
"""

import dataclasses
import hashlib
import json
from collections import Counter

import pytest
from benchmark_support import SENTINEL, ReferenceModel
from sqlalchemy import text

from app.evaluation import benchmark_builder as builder
from app.evaluation import benchmark_protocol as protocol
from app.evaluation import benchmark_run as run
from app.evaluation import benchmark_scoring as scoring
from app.evaluation import benchmark_spec as spec
from app.evaluation import environment as environment_module
from app.evaluation.environment import create_disposable_environment
from app.evaluation.report import ReportContainsProtectedTextError

MANIFEST_SHA256 = "077e9fc0d74c691274edb1b14b17a05e5950b50d48aa8dc727f08abd53e21c9f"
PROTOCOL_FINGERPRINT = "0a729d6067bb6262a642229b9b3134a9d472eea41c64b1f75df2140cd06e48a7"


@pytest.fixture(scope="module")
def manifest() -> spec.ManifestSpec:
    return spec.load_manifest()


def scenario_of(manifest, sid: str) -> spec.ScenarioSpec:
    return next(item for item in manifest.scenarios if item.id == sid)


def reference(variant: str = "correct", overrides: dict[str, str] | None = None):
    chosen = overrides or {}
    return lambda scenario, seed, world: ReferenceModel(
        scenario, world, chosen.get(scenario.id, variant)
    )


def execute(manifest, sid: str, variant: str = "correct", seed: int = 7) -> scoring.RunRecord:
    return run.execute_run("REF", scenario_of(manifest, sid), seed, reference(variant)).record


@pytest.fixture(scope="module")
def perfect(manifest) -> run.ApproachRun:
    """The correct reference approach on the whole benchmark at both seeds."""
    return run.run_approach("REF", reference("correct"), seeds=(7, 19))


# --- 1. the frozen artifacts are untouched -----------------------------------------------------


def test_the_frozen_manifest_and_thresholds_are_unchanged() -> None:
    assert spec.manifest_sha256() == MANIFEST_SHA256
    assert hashlib.sha256(spec.MANIFEST_PATH.read_bytes()).hexdigest() == MANIFEST_SHA256
    assert protocol.protocol_fingerprint() == PROTOCOL_FINGERPRINT


def test_every_report_carries_the_frozen_hashes(perfect) -> None:
    metadata = run.execution_metadata(perfect, **STAND_IN)

    report = run.build_report(perfect, metadata)

    assert report["protocol"]["manifest_sha256"] == MANIFEST_SHA256
    assert report["protocol"]["fingerprint"] == PROTOCOL_FINGERPRINT
    assert metadata["manifest_sha256"] == MANIFEST_SHA256
    assert metadata["protocol_fingerprint"] == PROTOCOL_FINGERPRINT


STAND_IN = {
    "model": "stand-in",
    "model_digest": "none",
    "temperature": 0.0,
    "context_tokens": 8192,
    "max_output_tokens": 512,
    "request_timeout_s": 120.0,
}


# --- 2. each scenario's database matches its independently defined state ---------------------


@pytest.mark.parametrize("sid", [f"BM-{n:03d}" for n in range(1, 25)])
def test_each_scenarios_database_matches_the_manifest(manifest, sid) -> None:
    scenario = scenario_of(manifest, sid)
    with create_disposable_environment() as env:
        baseline = builder.seed_baseline_fingerprint(env.engine)
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for(sid), pin=env.pin
        )

        assert builder.verify_world(env.engine, scenario, world) == []
        observed = builder.observe_state(
            env.engine, patient_id=world.patient_id, other_id=world.other_id, order=scenario.fixture
        )
        # The state read back from the database hashes to the manifest's own fingerprint.
        assert builder.fingerprint_of(observed) == spec.initial_state_fingerprint(scenario.fixture)
        assert observed == spec.initial_state(scenario.fixture)
        # Building the scenario did not touch the demonstration data.
        assert builder.seed_baseline_fingerprint(env.engine) == baseline


def tampered(manifest, sid, mutate) -> list[str]:
    """Build a scenario's world, change one thing behind the builder's back, and verify it."""
    scenario = scenario_of(manifest, sid)
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for(sid), pin=env.pin
        )
        with env.engine.begin() as connection:
            mutate(connection, world, env)
        return builder.verify_world(env.engine, scenario, world)


def slot(world, key):
    return {"id": world.slot_ids[key]}


TAMPERING = [
    (
        "a slot moved an hour",
        "BM-001",
        lambda c, w, e: c.execute(
            text(
                "UPDATE slots SET starts_at = starts_at + interval '1 hour',"
                " ends_at = ends_at + interval '1 hour' WHERE slot_id = :id"
            ),
            slot(w, "s1"),
        ),
    ),
    (
        "a slot removed",
        "BM-001",
        lambda c, w, e: c.execute(text("DELETE FROM slots WHERE slot_id = :id"), slot(w, "s2")),
    ),
    (
        "a slot longer than one interval",
        "BM-001",
        lambda c, w, e: c.execute(
            text("UPDATE slots SET ends_at = ends_at + interval '15 minutes' WHERE slot_id = :id"),
            slot(w, "s4"),
        ),
    ),
    (
        "a slot marked booked with no appointment",
        "BM-001",
        lambda c, w, e: c.execute(
            text("UPDATE slots SET status = 'BOOKED' WHERE slot_id = :id"), slot(w, "s1")
        ),
    ),
    (
        "an appointment cancelled behind a booked slot",
        "BM-007",
        lambda c, w, e: c.execute(
            text(
                "UPDATE appointments SET status = 'CANCELLED', cancelled_at = now()"
                " WHERE slot_id = :id"
            ),
            slot(w, "s1"),
        ),
    ),
    (
        "a doctor renamed",
        "BM-002",
        lambda c, w, e: c.execute(
            text("UPDATE doctors SET display_name = 'Dr. Priya Nairn' WHERE doctor_id = :id"),
            {"id": w.doctor_ids["nair"]},
        ),
    ),
    (
        "a doctor's specialty changed",
        "BM-003",
        lambda c, w, e: c.execute(
            text("UPDATE doctors SET specialty = 'Cardiology' WHERE doctor_id = :id"),
            {"id": w.doctor_ids["chen"]},
        ),
    ),
    (
        "a clinic in the wrong time zone",
        "BM-001",
        lambda c, w, e: c.execute(
            text("UPDATE clinics SET timezone = 'America/New_York' WHERE clinic_id = :id"),
            {"id": w.clinic_ids["c1"]},
        ),
    ),
    (
        "a clinic renamed",
        "BM-005",
        lambda c, w, e: c.execute(
            text(
                "UPDATE clinics SET name = 'Harborgate Community Clinic East' WHERE clinic_id = :id"
            ),
            {"id": w.clinic_ids["c1"]},
        ),
    ),
]


TAMPERING += [
    (
        "an open slot that still has a confirmed appointment",
        "BM-007",
        lambda c, w, e: c.execute(
            text("UPDATE slots SET status = 'AVAILABLE' WHERE slot_id = :id"), slot(w, "s1")
        ),
    ),
    (
        "a clinic's zone and times shifted together so local times still agree",
        "BM-001",
        lambda c, w, e: (
            c.execute(
                text("UPDATE clinics SET timezone = 'America/New_York' WHERE clinic_id = :id"),
                {"id": w.clinic_ids["c1"]},
            ),
            c.execute(
                text(
                    "UPDATE slots SET starts_at = starts_at - interval '2 hours',"
                    " ends_at = ends_at - interval '2 hours' WHERE clinic_id = :id"
                ),
                {"id": w.clinic_ids["c1"]},
            ),
        ),
    ),
]


@pytest.mark.parametrize(("label", "sid", "mutate"), TAMPERING, ids=[t[0] for t in TAMPERING])
def test_a_database_that_differs_from_the_manifest_is_detected(
    manifest, label, sid, mutate
) -> None:
    assert tampered(manifest, sid, mutate), label


@pytest.mark.parametrize(
    ("sid", "key", "statement"),
    [
        ("BM-001", "s1", "UPDATE slots SET status = 'BOOKED' WHERE slot_id = :id"),
        ("BM-007", "s1", "UPDATE slots SET status = 'AVAILABLE' WHERE slot_id = :id"),
    ],
    ids=["booked with no appointment", "open with a confirmed appointment"],
)
def test_a_slot_whose_status_and_appointment_disagree_is_labelled_inconsistent(
    manifest, sid, key, statement
) -> None:
    scenario = scenario_of(manifest, sid)
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for(sid), pin=env.pin
        )
        with env.engine.begin() as connection:
            connection.execute(text(statement), {"id": world.slot_ids[key]})
        state = builder.observe_state(
            env.engine, patient_id=world.patient_id, other_id=world.other_id, order=scenario.fixture
        )

    assert [row[-1] for row in state["slots"]].count("inconsistent") == 1


def test_an_extra_clinic_from_another_scenario_is_detected(manifest) -> None:
    scenario = scenario_of(manifest, "BM-001")
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for("BM-001"), pin=env.pin
        )
        other = scenario_of(manifest, "BM-002")
        builder.build_world(env.engine, other, suffix=builder.suffix_for("BM-002"), pin=env.pin)

        problems = builder.verify_world(env.engine, scenario, world)

    assert any("unexpected clinic" in problem for problem in problems)
    assert any("Cedar Hollow Heart Center" in problem for problem in problems)


def test_verifying_a_world_includes_the_real_search_check(manifest, monkeypatch) -> None:
    scenario = scenario_of(manifest, "BM-001")
    monkeypatch.setattr(
        builder, "verify_search_behavior", lambda engine, scenario, world: ["a search disagrees"]
    )
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for("BM-001"), pin=env.pin
        )

        assert builder.verify_world(env.engine, scenario, world) == ["a search disagrees"]


def test_the_demonstration_data_fingerprint_notices_a_change_to_it(manifest) -> None:
    scenario = scenario_of(manifest, "BM-001")
    with create_disposable_environment() as env:
        before = builder.seed_baseline_fingerprint(env.engine)
        builder.build_world(env.engine, scenario, suffix=builder.suffix_for("BM-001"), pin=env.pin)
        assert (
            builder.seed_baseline_fingerprint(env.engine) == before
        )  # The scenario is not seed data.

        with env.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE slots SET status = 'BOOKED' WHERE slot_id = (SELECT slot_id FROM slots"
                    " s JOIN clinics c USING (clinic_id) WHERE c.name = :name"
                    " AND s.status = 'AVAILABLE' LIMIT 1)"
                ),
                {"name": builder.SEED_CLINIC_NAMES[0]},
            )
        after_slot = builder.seed_baseline_fingerprint(env.engine)
        with env.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE appointments SET status = 'CANCELLED', cancelled_at = now()"
                    " WHERE appointment_id = (SELECT a.appointment_id FROM appointments a"
                    " JOIN patients p USING (patient_id) WHERE p.patient_code NOT LIKE 'EVP-%'"
                    " AND a.status = 'CONFIRMED' LIMIT 1)"
                )
            )
        after_appointment = builder.seed_baseline_fingerprint(env.engine)

    assert len({before, after_slot, after_appointment}) == 3


def test_the_real_search_tools_are_checked_against_the_manifest(manifest) -> None:
    scenario = scenario_of(manifest, "BM-003")
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for("BM-003"), pin=env.pin
        )
        assert builder.verify_search_behavior(env.engine, scenario, world) == []
        with env.engine.begin() as connection:  # Chen is no longer a dermatologist.
            connection.execute(
                text("UPDATE doctors SET specialty = 'Cardiology' WHERE doctor_id = :id"),
                {"id": world.doctor_ids["chen"]},
            )

        problems = builder.verify_search_behavior(env.engine, scenario, world)

    assert any("availability finds" in problem for problem in problems)


def test_an_ambiguous_surname_is_checked_against_the_real_directory(manifest) -> None:
    scenario = scenario_of(manifest, "BM-018")
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for("BM-018"), pin=env.pin
        )
        assert builder.verify_search_behavior(env.engine, scenario, world) == []
        with env.engine.begin() as connection:  # The second Park becomes someone else.
            connection.execute(
                text("UPDATE doctors SET display_name = 'Dr. David Quill' WHERE doctor_id = :id"),
                {"id": world.doctor_ids["dpark"]},
            )

        problems = builder.verify_search_behavior(env.engine, scenario, world)

    assert any("ambiguous" in problem for problem in problems)


def test_a_run_in_a_world_the_manifest_does_not_describe_is_never_scored(
    manifest, monkeypatch
) -> None:
    real = builder.build_world

    def faulty(engine, scenario, **kwargs):
        world = real(engine, scenario, **kwargs)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE slots SET starts_at = starts_at + interval '1 hour',"
                    " ends_at = ends_at + interval '1 hour' WHERE slot_id = :id"
                ),
                {"id": world.slot_ids["s1"]},
            )
        return world

    monkeypatch.setattr(builder, "build_world", faulty)

    with pytest.raises(builder.BenchmarkEnvironmentError):
        run.execute_run("REF", scenario_of(manifest, "BM-001"), 7, reference())


# --- 3. isolation and reproducibility ----------------------------------------------------------


def test_every_run_gets_its_own_database(manifest, monkeypatch) -> None:
    created: list[str] = []
    real = environment_module.create_disposable_environment

    from contextlib import contextmanager

    @contextmanager
    def recording():
        with real() as env:
            created.append(env.engine.url.database)
            yield env

    monkeypatch.setattr(run, "create_disposable_environment", recording)

    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-002", "BM-020")]
    run.run_approach("REF", reference(), seeds=(7, 19), scenarios=chosen)

    assert len(created) == 6 == len(set(created))  # Six runs, six databases, none reused.


def test_a_scenario_leaves_nothing_behind_for_the_next(manifest) -> None:
    first, second = scenario_of(manifest, "BM-001"), scenario_of(manifest, "BM-002")
    run.execute_run("REF", first, 7, reference())  # Its database is dropped when it ends.

    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, second, suffix=builder.suffix_for("BM-002"), pin=env.pin
        )
        state = builder.observe_state(
            env.engine, patient_id=world.patient_id, other_id=world.other_id, order=second.fixture
        )

    assert [c["name"] for c in state["clinics"]] == ["Cedar Hollow Heart Center"]


def test_the_same_scenario_starts_from_the_same_rows_every_time(manifest) -> None:
    scenario = scenario_of(manifest, "BM-008")
    seen = []
    for _ in range(2):
        with create_disposable_environment() as env:
            baseline = builder.seed_baseline_fingerprint(env.engine)
            world = builder.build_world(
                env.engine, scenario, suffix=builder.suffix_for("BM-008"), pin=env.pin
            )
            state = builder.observe_state(
                env.engine,
                patient_id=world.patient_id,
                other_id=world.other_id,
                order=scenario.fixture,
            )
            seen.append((builder.fingerprint_of(state), baseline, world.patient_code))

    assert seen[0] == seen[1]
    assert seen[0][2] == f"EVP-{builder.suffix_for('BM-008').upper()}-A"


def test_a_repeated_run_gives_identical_scores_apart_from_timing(manifest) -> None:
    def comparable(record):
        data = dataclasses.asdict(record.score)
        for timing in ("total_ms", "model_call_ms"):
            data.pop(timing)
        return data

    first = execute(manifest, "BM-011")
    second = execute(manifest, "BM-011")

    assert comparable(first) == comparable(second)
    assert [r.text for r in first.replies] == [r.text for r in second.replies]


def test_stability_counts_only_scenarios_whose_outcome_is_the_same_under_both_seeds(
    perfect,
) -> None:
    primary = perfect.scores(7)
    second = perfect.scores(19)
    changed = [dataclasses.replace(second[0], final_state_fingerprint="a different end state")]

    stable = scoring.aggregate(primary, changed + second[1:]).stable_outcomes

    assert stable == 23
    assert (
        scoring.aggregate(primary, None).stable_outcomes == 0
    )  # Not measured counts as not stable.


def test_the_two_seeds_give_the_same_world_and_the_same_outcome(perfect) -> None:
    primary, other = perfect.scores(7), perfect.scores(19)

    assert [r.initial_state_fingerprint for r in primary] == [
        r.initial_state_fingerprint for r in other
    ]
    assert scoring.aggregate(primary, other).stable_outcomes == 24


# --- 4. equivalent starting conditions for the two approaches ----------------------------------


def test_both_approaches_start_from_the_same_world_patient_and_request(manifest) -> None:
    chosen = [
        scenario_of(manifest, sid) for sid in ("BM-001", "BM-005", "BM-009", "BM-018", "BM-022")
    ]
    one = run.run_approach("A", reference("correct"), seeds=(7,), scenarios=chosen)
    other = run.run_approach("B", reference("template"), seeds=(7,), scenarios=chosen)

    for first, second in zip(one.records[7], other.records[7], strict=True):
        assert first.score.initial_state_fingerprint == second.score.initial_state_fingerprint
        assert first.request == second.request
        assert first.score.seed_data_untouched and second.score.seed_data_untouched
    assert one.patient_codes == other.patient_codes
    assert all(code.startswith("EVP-") for code in one.patient_codes.values())
    assert one.request_fingerprints == other.request_fingerprints
    assert one.seed_baselines == other.seed_baselines and len(one.seed_baselines) == 1
    metadata = [run.execution_metadata(r, **STAND_IN) for r in (one, other)]
    assert scoring.check_execution_comparability(*metadata) == []


def test_an_unauthenticated_scenario_is_unauthenticated_for_both_approaches(manifest) -> None:
    for variant in ("correct", "template"):
        record = execute(manifest, "BM-022", variant)

        assert record.score.outcome_correct
        assert not [
            c for c in record.calls if c["tool"] == "prepare_appointment_action" and c["ok"]
        ]


def test_the_patient_identity_fingerprint_changes_when_the_patients_do(perfect) -> None:
    same = run.execution_metadata(perfect, **STAND_IN)["patient_identity_fingerprint"]
    codes = {**perfect.patient_codes, "BM-001": "EVP-SOMEONE-ELSE-A"}

    other = run.execution_metadata(dataclasses.replace(perfect, patient_codes=codes), **STAND_IN)

    assert other["patient_identity_fingerprint"] != same
    assert not any("EVP-" in str(value) for value in other.values())  # Never the codes themselves.


def test_a_difference_in_conditions_makes_the_comparison_invalid(perfect) -> None:
    base = run.execution_metadata(perfect, **STAND_IN)
    twist = {
        "a different patient": {"patient_identity_fingerprint": "0" * 64},
        "a different request": {
            "request_fingerprints": {**base["request_fingerprints"], "BM-001": "0" * 64}
        },
        "different demonstration data": {"seed_baseline_fingerprint": ["different"]},
        "a different starting world": {
            "initial_state_fingerprints": {**base["initial_state_fingerprints"], "BM-001": "0" * 64}
        },
        "a different model": {"model_digest": "other"},
        "a different manifest": {"manifest_sha256": "0" * 64},
    }
    for label, change in twist.items():
        problems = scoring.check_execution_comparability(base, {**base, **change})
        assert problems, label

    missing = {key: value for key, value in base.items() if key != "patient_identity_fingerprint"}
    assert scoring.check_execution_comparability(base, missing) == [
        "patient_identity_fingerprint was not recorded for both runs"
    ]


# --- 5. a correct run is scored correct --------------------------------------------------------


def test_the_correct_reference_approach_passes_every_gate_and_core_criterion(perfect) -> None:
    scores = scoring.aggregate(perfect.scores(7), perfect.scores(19))
    evaluation = protocol.evaluate(scores)

    assert evaluation.gates_ok and evaluation.core_ok
    assert scores.outcome_correct == scores.intent_correct == 24
    assert scores.adequate_responses == 24
    assert scores.action_responses == scores.complete_action_responses == 25
    assert (scores.ambiguity_correct, scores.no_availability_correct) == (4, 3)
    assert scores.unauthenticated_correct == 3
    assert scores.false_success_claims == scores.wrong_detail_responses == 0
    assert scores.protected_data_leaks == scores.wrong_target_prepared == 0
    assert scores.safety_violations == scores.provider_timeouts == scores.invalid_tool_calls == 0
    assert scores.unauthorized_filters == 0 and scores.stable_outcomes == 24


def test_a_perfect_run_is_not_marked_down_for_having_no_reference_call_path(perfect) -> None:
    published = {category for r in perfect.scores(7) for category in r.runner_failures}

    assert published <= {"UNREQUESTED_NARROWING_FILTER"} or not published
    assert not published & scoring.REFERENCE_PATH_ARTIFACTS


def test_any_acceptable_new_slot_is_a_correct_reschedule(manifest) -> None:
    record = execute(manifest, "BM-011", "alternate_acceptable")  # The later of the two.

    assert record.score.outcome_correct and record.score.intent_correct
    assert not record.score.wrong_target_prepared
    assert (
        record.score.final_state_fingerprint
        != execute(manifest, "BM-011").score.final_state_fingerprint
    )


def test_the_trusted_ui_still_commits_only_what_the_patient_wanted(manifest) -> None:
    right = execute(manifest, "BM-012")
    wrong = execute(manifest, "BM-012", "wrong_slot")  # The afternoon slot, not a morning one.

    assert right.score.outcome_correct
    assert not wrong.score.outcome_correct and wrong.score.wrong_target_prepared


# --- 6. negative controls: each defect is scored as that defect -------------------------------


@pytest.mark.parametrize("sid", ["BM-001", "BM-007", "BM-011"])
def test_preparing_the_wrong_thing_is_a_wrong_target_and_changes_nothing(manifest, sid) -> None:
    score = execute(manifest, sid, "wrong_slot").score

    assert score.wrong_target_prepared and not score.intent_correct
    assert not score.outcome_correct  # The simulated patient declined the wrong action.
    assert "wrong_target" in score.intent_problems
    assert not score.behaved_correctly()


def test_a_wrong_action_that_is_corrected_was_still_put_in_front_of_the_patient(manifest) -> None:
    score = execute(manifest, "BM-001", "wrong_then_right").score

    assert score.wrong_target_prepared  # A gate: the patient was shown a wrong action.
    assert score.intent_correct and score.outcome_correct  # The one finally confirmed was right.


def test_a_run_that_alters_the_demonstration_data_is_not_correct(manifest, monkeypatch) -> None:
    real = builder.seed_baseline_fingerprint
    calls = iter(range(100))
    monkeypatch.setattr(
        builder,
        "seed_baseline_fingerprint",
        lambda engine: real(engine) if next(calls) == 0 else "the data changed",
    )

    score = execute(manifest, "BM-001").score

    assert not score.seed_data_untouched and not score.outcome_correct


def test_a_filter_the_patient_never_asked_for_is_unauthorized(manifest) -> None:
    score = execute(manifest, "BM-001", "unrequested_filter").score

    assert score.unauthorized_filters >= 1 and not score.intent_correct
    assert "unauthorized_specialty" in score.intent_problems
    assert score.outcome_correct  # It still found the right slot; the filter is its own finding.


def test_a_rejected_tool_call_is_counted_as_invalid(manifest) -> None:
    score = execute(manifest, "BM-001", "invalid_arguments").score

    assert score.invalid_tool_calls >= 1


def test_an_incorrect_appointment_detail_is_inaccurate_and_incomplete(manifest) -> None:
    record = execute(manifest, "BM-001", "wrong_detail_review")
    review = next(r for r in record.score.replies if r.slot == "review")

    assert not review.accurate and not review.complete
    assert any(code.startswith("time:") for code in review.problems)
    assert record.score.wrong_detail_responses == 1
    assert record.score.outcome_correct  # The booking was right; what it said was not.


def test_a_wrong_detail_after_the_booking_is_inaccurate(manifest) -> None:
    final = next(
        r
        for r in execute(manifest, "BM-002", "wrong_detail_final").score.replies
        if r.slot == "final"
    )

    assert not final.accurate and any(code.startswith("time:") for code in final.problems)


def test_claiming_success_before_the_patient_confirms_is_a_false_claim(manifest) -> None:
    score = execute(manifest, "BM-001", "false_claim_review").score

    assert score.false_success_claims >= 1
    review = next(r for r in score.replies if r.slot == "review")
    assert not review.accurate and "false_claim:BOOKED" in review.problems
    assert not score.outcome_correct  # The confirmation was not offered, so nothing was booked.


@pytest.mark.parametrize("sid", ["BM-018", "BM-019"])
def test_a_clarification_that_omits_the_choices_is_inadequate(manifest, sid) -> None:
    score = execute(manifest, sid, "incomplete_clarification").score
    final = score.final_reply()

    assert final and not final.adequate and final.accurate
    assert any(code.startswith("omits_option") for code in final.problems)
    assert not score.behaved_correctly()


def test_a_fixed_template_is_accurate_but_incomplete_and_fails_a_clarification(manifest) -> None:
    booking = execute(manifest, "BM-001", "template").score
    ambiguity = execute(manifest, "BM-018", "template").score

    for reply in booking.replies:
        assert reply.accurate and reply.adequate and not reply.complete
        assert {"missing:doctor", "missing:time"} <= set(reply.problems)
    assert booking.outcome_correct and booking.intent_correct
    assert not ambiguity.final_reply().adequate and not ambiguity.behaved_correctly()


def test_a_template_that_claims_success_after_a_failed_confirmation_is_caught(manifest) -> None:
    score = execute(manifest, "BM-006", "misleading_template").score
    final = score.final_reply()

    assert score.outcome_correct  # The database is right: the slot went to someone else.
    assert not final.accurate and "false_claim:BOOKED" in final.problems
    assert score.false_success_claims == 1


def test_an_honest_template_after_a_failed_confirmation_is_not_penalized(manifest) -> None:
    final = execute(manifest, "BM-006", "template").score.final_reply()

    assert final.accurate and final.adequate


@pytest.mark.parametrize("sid", ["BM-018", "BM-015"])
def test_preparing_an_action_that_should_not_exist_is_a_wrong_target(manifest, sid) -> None:
    score = execute(manifest, sid, "prepare_anyway").score

    assert score.wrong_target_prepared and not score.intent_correct
    assert {"wrong_operation", "wrong_target"} <= set(score.intent_problems)


def test_revealing_the_patients_own_appointment_to_an_unauthenticated_caller_is_a_leak(
    manifest,
) -> None:
    score = execute(manifest, "BM-023", "leaky_auth").score

    assert score.protected_data_leaks >= 1
    assert not score.final_reply().accurate and not score.behaved_correctly()


def test_an_attempt_to_confirm_is_recorded_and_changes_nothing(manifest) -> None:
    score = execute(manifest, "BM-001", "bypass").score

    assert score.bypass_attempts >= 1
    assert score.safety_violations == 0  # The attempt failed; nothing was committed by the model.
    assert score.outcome_correct
    assert "CONFIRMATION_BYPASS_ATTEMPT" in score.runner_failures


def test_claiming_the_job_is_done_in_place_of_asking_or_refusing_is_caught(manifest) -> None:
    for sid in ("BM-018", "BM-015"):
        final = execute(manifest, sid, "claims_done").score.final_reply()

        assert not final.accurate and not final.adequate


def test_defects_scattered_through_a_whole_run_fail_the_matching_gates(manifest, perfect) -> None:
    defects = {
        "BM-001": "wrong_slot",
        "BM-002": "wrong_detail_review",
        "BM-003": "false_claim_review",
        "BM-004": "unrequested_filter",
        "BM-005": "invalid_arguments",
        "BM-006": "misleading_template",
        "BM-010": "bypass",
        "BM-015": "prepare_anyway",
        "BM-018": "incomplete_clarification",
        "BM-023": "leaky_auth",
    }
    flawed = run.run_approach("REF", reference("correct", defects), seeds=(7,))
    scores = scoring.aggregate(flawed.scores(7))
    evaluation = protocol.evaluate(scores)
    failed_gates = {c.name for c in evaluation.gates if not c.passed}
    failed_core = {c.name for c in evaluation.core if not c.passed}

    assert scores.wrong_target_prepared == 2  # BM-001 and BM-015
    assert scores.false_success_claims == 2  # BM-003's review, BM-006's final
    assert scores.wrong_detail_responses >= 1 and scores.protected_data_leaks >= 1
    assert scores.unauthorized_filters >= 1 and scores.invalid_tool_calls >= 1
    assert {
        "false_success_claims",
        "wrong_detail_responses",
        "protected_data_leaks",
    } <= failed_gates
    assert {"wrong_target_prepared"} <= failed_gates
    assert {"unauthorized_filters", "invalid_tool_calls", "ambiguity_correct"} <= failed_core
    assert not evaluation.gates_ok and not evaluation.core_ok
    sound = scoring.aggregate(perfect.scores(7), perfect.scores(19))
    # A flawed approach is never selected over a sound one, however the sound one is judged.
    assert protocol.decide(scores, sound).name == "SELECT_B"
    assert protocol.decide(sound, scores).name == "SELECT_A"


# --- 7. nothing published contains raw conversation content ------------------------------------


def test_a_published_report_holds_codes_and_counts_never_words(manifest) -> None:
    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-018", "BM-023")]
    sentinel_run = run.run_approach("REF", reference("sentinel"), seeds=(7, 19), scenarios=chosen)

    report = run.build_report(sentinel_run, run.execution_metadata(sentinel_run, **STAND_IN))
    body = json.dumps(report)

    assert SENTINEL not in body
    for record in (r for seed in (7, 19) for r in sentinel_run.records[seed]):
        assert record.request not in body
        for reply in record.replies:
            assert reply.text not in body
        for call in record.calls:
            assert not any(
                isinstance(v, str) and len(v) >= 12 and v in body
                for v in call["arguments"].values()
            )
    for private in ("EVP-", "Evaluation Patient", "csrf", "Set-Cookie", "password", "pin_hash"):
        assert private not in body
    for code in sentinel_run.patient_codes.values():
        assert code not in body  # The login identifiers of the synthetic patients.
    # The raw words really were recorded in memory, so the report is clean by construction, not
    # because nothing was kept.
    assert any(SENTINEL in reply.text for r in sentinel_run.records[7] for reply in r.replies)


def test_the_report_guard_fails_loudly_if_a_reply_would_leak(manifest) -> None:
    chosen = [scenario_of(manifest, "BM-001")]
    one = run.run_approach("REF", reference(), seeds=(7,), scenarios=chosen)

    with pytest.raises(ReportContainsProtectedTextError):
        scoring.sanitized_report(
            approach="REF",
            metadata={},
            primary=one.scores(7),
            second_seed=None,
            records=one.records[7],
            forbidden_text=["BM-001"],  # Present in the report, so it must be refused.
        )


def test_the_runner_records_nothing_unless_asked(manifest) -> None:
    from app.evaluation.runner import run_scenario

    scenario = scenario_of(manifest, "BM-020")
    with create_disposable_environment() as env:
        world = builder.build_world(
            env.engine, scenario, suffix=builder.suffix_for("BM-020"), pin=env.pin
        )
        runnable = run.scenario_for_run(
            scenario, world, run.run_fixtures(env.engine, scenario, world)
        )
        plain = run_scenario(
            runnable, env, model_factory=lambda s, r: ReferenceModel(scenario, world)
        )

    assert plain.conversation == []


# --- 8. the whole path, end to end -------------------------------------------------------------


def test_a_comparison_is_decided_only_between_comparable_runs(manifest) -> None:
    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-018", "BM-022")]
    a = run.run_approach("A", reference("correct"), seeds=(7, 19), scenarios=chosen)
    b = run.run_approach("B", reference("template"), seeds=(7, 19), scenarios=chosen)
    reports = [run.build_report(r, run.execution_metadata(r, **STAND_IN)) for r in (a, b)]

    comparison = scoring.compare_approaches(*reports)
    assert comparison["comparable"] and comparison["decision"]["name"]

    reports[1]["metadata"]["model_digest"] = "something else"
    refused = scoring.compare_approaches(*reports)
    assert not refused["comparable"] and refused["decision"] is None
    assert refused["problems"] == ["model_digest differs between the runs"]


def test_the_outcome_by_category_is_reported_without_content(perfect) -> None:
    report = run.build_report(perfect, run.execution_metadata(perfect, **STAND_IN))

    assert report["strata"]["category"]["BOOKING"] == {"runs": 5, "outcome_correct": 5}
    assert Counter(r["category"] for r in report["runs"]) == {
        "BOOKING": 5,
        "CONFLICT": 1,
        "CANCELLATION": 4,
        "RESCHEDULING": 4,
        "NO_AVAILABILITY": 3,
        "AMBIGUITY": 4,
        "UNAUTHENTICATED": 3,
    }
    assert report["stability_measured"] is True
    assert {"gates_ok", "core_ok", "secondary_ok"} <= set(report["evaluation"])


def test_assembling_two_runs_writes_a_decision_and_a_blinded_sheet(manifest, tmp_path) -> None:
    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-018", "BM-022")]
    for label, variant in (("A", "correct"), ("B", "template")):
        approach = run.run_approach(label, reference(variant), seeds=(7, 19), scenarios=chosen)
        run.save_outputs(tmp_path, approach, run.execution_metadata(approach, **STAND_IN))

    assert run.main(["assemble", "--output-dir", str(tmp_path)]) == 0

    comparison = json.loads((tmp_path / "comparison.json").read_text())
    sheet = (tmp_path / "review-sheet.md").read_text()
    key = json.loads((tmp_path / "review-key.json").read_text())
    assert comparison["comparable"] and comparison["decision"]["name"]
    assert {entry["approach"] for entry in key.values()} == {"A", "B"}  # Both are in the audit.
    assert "approach" not in sheet.lower() and "seed" not in sheet.lower()
    # The published reports hold no reply text; the raw files, which do, are the local-only ones.
    for label in ("A", "B"):
        published = (tmp_path / f"report-{label}.json").read_text()
        raw = json.loads((tmp_path / f"raw-{label}.json").read_text())
        assert raw and all(item["text"] not in published for item in raw)


def test_assembling_runs_that_are_not_comparable_refuses_to_decide(manifest, tmp_path) -> None:
    chosen = [scenario_of(manifest, "BM-001")]
    for label in ("A", "B"):
        approach = run.run_approach(label, reference(), seeds=(7,), scenarios=chosen)
        metadata = run.execution_metadata(approach, **STAND_IN)
        if label == "B":
            metadata["model_digest"] = "a different model"
        run.save_outputs(tmp_path, approach, metadata)

    assert run.main(["assemble", "--output-dir", str(tmp_path)]) == 1

    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert not comparison["comparable"] and comparison["decision"] is None


# --- 9. alternation, the warm-up, and the orchestration of a real run ---------------------------


def test_the_approaches_take_turns_going_first_and_each_runs_every_scenario_once(
    manifest, monkeypatch
) -> None:
    order: list[tuple[str, str, int]] = []

    def tracking(label):
        def factory(scenario, seed, world):
            order.append((label, scenario.id, seed))
            return ReferenceModel(scenario, world)

        return factory

    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-002", "BM-020", "BM-022")]
    outcomes = run.run_approaches(
        {"A": tracking("A"), "B": tracking("B")}, seeds=(7, 19), scenarios=chosen
    )

    first_seed = [(label, sid) for label, sid, seed in order if seed == 7]
    assert first_seed == [
        ("A", "BM-001"), ("B", "BM-001"),
        ("B", "BM-002"), ("A", "BM-002"),
        ("A", "BM-020"), ("B", "BM-020"),
        ("B", "BM-022"), ("A", "BM-022"),
    ]  # fmt: skip
    assert Counter(order) == dict.fromkeys(
        ((label, s.id, seed) for label in "AB" for s in chosen for seed in (7, 19)), 1
    )
    assert [r.scenario_id for r in outcomes["A"].records[7]] == [s.id for s in chosen]
    assert [r.scenario_id for r in outcomes["B"].records[19]] == [s.id for s in chosen]


def test_two_approaches_on_the_same_scenarios_still_get_separate_databases(
    manifest, monkeypatch
) -> None:
    from contextlib import contextmanager

    created: list[str] = []
    real = environment_module.create_disposable_environment

    @contextmanager
    def recording():
        with real() as env:
            created.append(env.engine.url.database)
            yield env

    monkeypatch.setattr(run, "create_disposable_environment", recording)
    chosen = [scenario_of(manifest, sid) for sid in ("BM-001", "BM-020")]

    run.run_approaches(
        {"A": reference(), "B": reference("template")}, seeds=(7, 19), scenarios=chosen
    )

    assert len(created) == 8 == len(set(created))  # Two approaches, two scenarios, two seeds.


def test_the_warm_up_is_one_small_request_that_belongs_to_neither_approach() -> None:
    from app.evaluation.llm import LlmConfig

    sent = []
    config = LlmConfig()

    run.warm_up(
        config, transport=lambda path, payload, timeout: sent.append((path, payload, timeout))
    )

    ((path, payload, timeout),) = sent
    assert path == "/api/chat" and timeout == config.request_timeout_s
    assert "tools" not in payload and "format" not in payload
    assert payload["options"]["num_predict"] <= 8
    assert payload["model"] == config.model


def test_a_full_run_warms_up_once_then_alternates_and_writes_every_output(
    manifest, tmp_path, monkeypatch
) -> None:
    from app.evaluation import cli

    events: list[str] = []
    subset = manifest.model_copy(
        update={"scenarios": [scenario_of(manifest, sid) for sid in ("BM-001", "BM-018")]}
    )
    monkeypatch.setattr(run.spec, "load_manifest", lambda: subset)
    monkeypatch.setattr(cli, "served_model_digest", lambda config: "digest-123")
    monkeypatch.setattr(run, "warm_up", lambda config, transport=None: events.append("warm-up"))

    def factory_for(label, model, url):
        def build(scenario, seed, world):
            events.append(f"{label}:{scenario.id}:{seed}")
            return ReferenceModel(scenario, world, "correct" if label == "A" else "template")

        return build

    monkeypatch.setattr(run, "approach_factory", factory_for)

    assert run.main(["run", "--output-dir", str(tmp_path), "--seeds", "7"]) == 0

    assert events == ["warm-up", "A:BM-001:7", "B:BM-001:7", "B:BM-018:7", "A:BM-018:7"]
    for label in ("A", "B"):
        report = json.loads((tmp_path / f"report-{label}.json").read_text())
        assert report["approach"] == label
        assert report["metadata"]["model_digest"] == "digest-123"
        assert report["metadata"]["manifest_sha256"] == MANIFEST_SHA256
        assert report["protocol"]["fingerprint"] == PROTOCOL_FINGERPRINT
        assert (tmp_path / f"raw-{label}.json").is_file()
    # The two reports describe the same world, so they can be compared.
    assert run.main(["assemble", "--output-dir", str(tmp_path)]) == 0


def test_a_single_approach_can_be_run_for_debugging_without_the_other(
    manifest, tmp_path, monkeypatch
) -> None:
    from app.evaluation import cli

    subset = manifest.model_copy(update={"scenarios": [scenario_of(manifest, "BM-020")]})
    monkeypatch.setattr(run.spec, "load_manifest", lambda: subset)
    monkeypatch.setattr(cli, "served_model_digest", lambda config: "digest-123")
    monkeypatch.setattr(run, "warm_up", lambda config, transport=None: None)
    monkeypatch.setattr(run, "approach_factory", lambda label, model, url: reference())

    assert run.main(["run", "--approach", "B", "--output-dir", str(tmp_path), "--seeds", "7"]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "partial-raw.jsonl",
        "partial-scores.jsonl",
        "raw-B.json",
        "report-B.json",
        "run-status.json",
    ]
    assert not (tmp_path / "report-A.json").exists()  # Only the approach that was asked for.


def test_a_warm_up_that_fails_stops_the_run_before_anything_is_scored(
    tmp_path, monkeypatch
) -> None:
    from app.evaluation import cli
    from app.evaluation.llm import ModelTimeoutError

    monkeypatch.setattr(cli, "served_model_digest", lambda config: "digest-123")

    def fail(config, transport=None):
        raise ModelTimeoutError

    ran = []
    monkeypatch.setattr(run, "warm_up", fail)
    monkeypatch.setattr(run, "run_approaches", lambda *a, **k: ran.append(1))

    with pytest.raises(SystemExit) as stop:
        run.main(["run", "--output-dir", str(tmp_path)])

    assert stop.value.code == 2 and not ran and not list(tmp_path.iterdir())


def test_the_audit_command_reports_whether_the_rules_can_be_trusted(tmp_path) -> None:
    key = {
        f"R-{n:03d}": {"rules": {"accurate": True, "adequate": True, "complete": True}}
        for n in range(1, 11)
    }
    (tmp_path / "review-key.json").write_text(json.dumps(key))
    agree = {name: {"accurate": True, "adequate": True, "complete": True} for name in key}
    judgments = tmp_path / "judgments.json"

    judgments.write_text(json.dumps(agree))
    assert run.main(["audit", "--output-dir", str(tmp_path), "--judgments", str(judgments)]) == 0

    disagree = {**agree, "R-001": {**agree["R-001"], "accurate": False},
                "R-002": {**agree["R-002"], "accurate": False}}  # fmt: skip
    judgments.write_text(json.dumps(disagree))
    assert run.main(["audit", "--output-dir", str(tmp_path), "--judgments", str(judgments)]) == 1
    result = json.loads((tmp_path / "audit-agreement.json").read_text())
    assert result["per_verdict"]["accurate"]["reliable"] is False
    assert result["per_verdict"]["adequate"]["reliable"] is True


def test_each_finished_run_is_written_at_once_and_an_interruption_is_recorded(
    manifest, tmp_path, monkeypatch
) -> None:
    from app.evaluation import cli

    subset = manifest.model_copy(
        update={"scenarios": [scenario_of(manifest, sid) for sid in ("BM-001", "BM-018", "BM-022")]}
    )
    monkeypatch.setattr(run.spec, "load_manifest", lambda: subset)
    monkeypatch.setattr(cli, "served_model_digest", lambda config: "digest-123")
    monkeypatch.setattr(run, "warm_up", lambda config, transport=None: None)
    started = []

    def factory_for(label, model, url):
        def build(scenario, seed, world):
            started.append((label, scenario.id))
            if len(started) == 4:  # The fourth run is interrupted, as by a usage limit.
                raise KeyboardInterrupt
            return ReferenceModel(scenario, world)

        return build

    monkeypatch.setattr(run, "approach_factory", factory_for)

    with pytest.raises(KeyboardInterrupt):
        run.main(["run", "--output-dir", str(tmp_path), "--seeds", "7"])

    status = json.loads((tmp_path / "run-status.json").read_text())
    scores = [
        json.loads(line) for line in (tmp_path / "partial-scores.jsonl").read_text().splitlines()
    ]
    raw = (tmp_path / "partial-raw.jsonl").read_text().splitlines()
    assert status == {
        "state": "interrupted",
        "completed_runs": 3,
        "interrupted_by": "KeyboardInterrupt",
    }
    assert len(scores) == len(raw) == 3  # The three finished runs survive.
    assert [s["scenario_id"] for s in scores] == ["BM-001", "BM-001", "BM-018"]
    assert not (tmp_path / "report-A.json").exists()  # No report was made from an incomplete run.
    scored_text = (tmp_path / "partial-scores.jsonl").read_text()
    assert "EVP-" not in scored_text and not any(
        item["text"] in scored_text for line in raw for item in json.loads(line)
    )


def test_a_complete_run_says_so(manifest, tmp_path, monkeypatch) -> None:
    from app.evaluation import cli

    subset = manifest.model_copy(update={"scenarios": [scenario_of(manifest, "BM-020")]})
    monkeypatch.setattr(run.spec, "load_manifest", lambda: subset)
    monkeypatch.setattr(cli, "served_model_digest", lambda config: "digest-123")
    monkeypatch.setattr(run, "warm_up", lambda config, transport=None: None)
    monkeypatch.setattr(run, "approach_factory", lambda label, model, url: reference())

    assert run.main(["run", "--output-dir", str(tmp_path), "--seeds", "7"]) == 0

    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status == {"state": "complete", "completed_runs": 2, "interrupted_by": None}
