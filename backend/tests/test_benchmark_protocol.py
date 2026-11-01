"""The frozen architecture benchmark and its protocol: the manifest, the scoring rules, and the
decision rule are validated here before any approach is run on them.

Nothing in this file runs a model or a database. The point is that the rules which will judge the
two approaches are fixed, internally consistent, independent of both, and tested on synthetic
inputs, so that no result can be read back into them.
"""

import ast
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation import benchmark_protocol as protocol
from app.evaluation import benchmark_spec as spec
from app.evaluation.benchmark_protocol import (
    ApproachScores,
    ObservedAction,
    ObservedFilter,
    Thresholds,
    check_comparability,
    decide,
    evaluate,
    intent_fidelity,
    response_facts,
    score_response,
)
from app.seed.catalog import CLINICS, DOCTORS

ROOT = Path(__file__).parents[2]
APP = ROOT / "backend" / "app"
DOC = ROOT / "docs" / "agent-architecture-evaluation-protocol.md"

# Frozen on 2026-11-01. Changing the manifest or the thresholds changes these values, which is the
# point: a different benchmark is a different version, not an edit.
MANIFEST_SHA256 = "077e9fc0d74c691274edb1b14b17a05e5950b50d48aa8dc727f08abd53e21c9f"
PROTOCOL_FINGERPRINT = "0a729d6067bb6262a642229b9b3134a9d472eea41c64b1f75df2140cd06e48a7"

SEEDED_CLINICS = frozenset(clinic.name for clinic in CLINICS)
SEEDED_DOCTORS = frozenset(doctor.display_name for doctor in DOCTORS)
TOOL_NAMES = (
    "search_clinic_directory",
    "search_appointment_availability",
    "get_my_appointments",
    "prepare_appointment_action",
)


@pytest.fixture(scope="module")
def manifest() -> spec.ManifestSpec:
    return spec.load_manifest()


def scenario_of(manifest: spec.ManifestSpec, sid: str) -> spec.ScenarioSpec:
    return next(item for item in manifest.scenarios if item.id == sid)


def raw() -> dict:
    return json.loads(spec.MANIFEST_PATH.read_text())


def problems_after(mutate) -> list[str]:
    data = raw()
    mutate({item["id"]: item for item in data["scenarios"]}, data)
    changed = spec.ManifestSpec.model_validate(data)
    return spec.validate_manifest(changed, SEEDED_CLINICS, SEEDED_DOCTORS)


# --- 1. the manifest is frozen and well formed -------------------------------------------------


def test_the_manifest_is_frozen_at_a_recorded_hash() -> None:
    assert hashlib.sha256(spec.MANIFEST_PATH.read_bytes()).hexdigest() == MANIFEST_SHA256
    assert spec.manifest_sha256() == MANIFEST_SHA256


def test_the_protocol_document_records_the_same_hashes() -> None:
    text = DOC.read_text()

    assert MANIFEST_SHA256 in text
    assert PROTOCOL_FINGERPRINT in text
    assert protocol.protocol_fingerprint() == PROTOCOL_FINGERPRINT
    assert spec.VERSION in text and protocol.PROTOCOL_VERSION in text


def test_the_manifest_declares_its_own_rules(manifest) -> None:
    assert manifest.version == spec.VERSION == "agent-architecture-benchmark-v1"
    assert manifest.status == "FROZEN_BEFORE_EXECUTION"
    assert manifest.protocol == "docs/agent-architecture-evaluation-protocol.md"
    assert manifest.rules == {
        "synthetic_data_only": True,
        "used_for_prompt_or_schema_tuning": False,
        "approach_specific_expectations": False,
        "contains_tool_names_or_call_orders": False,
        "patient_follow_up_replies": False,
    }
    assert any("HR-001" in item for item in manifest.development_evidence)
    assert any("AB-" in item for item in manifest.development_evidence)


def test_the_manifest_is_internally_consistent_with_its_own_fixtures(manifest) -> None:
    """Every declared answer is re-derived from the fixtures: the earliest eligible slot, whether
    a surname is ambiguous, which appointment a request selects, what the end state must be."""
    assert spec.validate_manifest(manifest, SEEDED_CLINICS, SEEDED_DOCTORS) == []


def test_unknown_manifest_fields_are_rejected() -> None:
    data = raw()
    data["scenarios"][0]["preferred_tool"] = "x"
    with pytest.raises(ValueError):
        spec.ManifestSpec.model_validate(data)


# --- 2. the distribution is balanced and varied ------------------------------------------------


def test_there_are_24_scenarios_numbered_in_order(manifest) -> None:
    assert [s.id for s in manifest.scenarios] == [f"BM-{n:03d}" for n in range(1, 25)]


def test_the_scenarios_are_balanced_across_the_required_categories(manifest) -> None:
    categories = Counter(s.category for s in manifest.scenarios)

    assert categories == {
        "BOOKING": 5,
        "CONFLICT": 1,
        "CANCELLATION": 4,
        "RESCHEDULING": 4,
        "NO_AVAILABILITY": 3,
        "AMBIGUITY": 4,
        "UNAUTHENTICATED": 3,
    }
    assert Counter(s.expected.outcome for s in manifest.scenarios) == {
        "BOOKED": 5,
        "CANCELLED": 3,
        "RESCHEDULED": 4,
        "CONFLICT": 1,
        "NO_MATCH": 1,
        "CLARIFICATION": 4,
        "NO_AVAILABILITY": 3,
        "AUTH_REQUIRED": 3,
    }
    assert Counter(s.actor for s in manifest.scenarios) == {"patient": 21, "public": 3}


def test_requests_vary_in_phrasing_and_in_how_dates_are_written(manifest) -> None:
    styles = Counter(s.strata.phrasing for s in manifest.scenarios)
    formats = Counter(s.strata.date_format for s in manifest.scenarios)

    assert len(styles) >= 12 and max(styles.values()) <= 3
    assert formats == {"iso": 12, "long": 6, "dmy": 4, "none": 2}
    assert len({s.request for s in manifest.scenarios}) == 24


def test_clinics_doctors_and_dates_vary_across_the_set(manifest) -> None:
    clinic_names = [c.name for s in manifest.scenarios for c in s.fixture.clinics]
    doctor_names = [
        d.name for s in manifest.scenarios for c in s.fixture.clinics for d in c.doctors
    ]
    offsets = {offset for s in manifest.scenarios for offset in s.fixture.days.values()}

    assert len(clinic_names) == len(set(clinic_names)) >= 25
    assert len(doctor_names) == len(set(doctor_names)) >= 30
    assert len(offsets) >= 12
    specialties = {
        d.specialty for s in manifest.scenarios for c in s.fixture.clinics for d in c.doctors
    }
    assert len(specialties) >= 6


def test_every_action_scenario_has_wrong_answers_an_agent_could_choose(manifest) -> None:
    """A fixture where the right slot is the only slot would not distinguish careful from lucky."""
    for item in manifest.scenarios:
        prepared = item.expected.prepared
        if prepared is None:
            continue
        acceptable = prepared.acceptable_new_slots or [prepared.slot or prepared.appointment]
        open_slots = [s for s in item.fixture.slots if s.held_by in ("none", "patient_cancelled")]
        appointments = spec.patient_appointments(item.fixture)
        pool = open_slots if prepared.action != "CANCEL" else appointments
        assert len(pool) >= len(acceptable) + 1, item.id


def test_unambiguous_requests_are_not_secretly_ambiguous_and_clarifications_really_are(
    manifest,
) -> None:
    for item in manifest.scenarios:
        if item.expected.outcome == "CLARIFICATION":
            assert item.intent.ambiguity is not None, item.id
            # Something to choose between, unless what is missing is the date itself.
            assert item.intent.options or item.intent.ambiguity == "MISSING_DATE", item.id
        else:
            assert item.intent.ambiguity is None, item.id


# --- 3. the manifest cannot favor an approach --------------------------------------------------


def test_the_manifest_names_no_tools_and_prescribes_no_call_order() -> None:
    text = spec.MANIFEST_PATH.read_text()

    for name in TOOL_NAMES:
        assert name not in text
    for key in ("expected_tools", "tool_calls", "call_order", "tool_sequence", "prompt"):
        assert f'"{key}"' not in text


def test_requests_are_not_reused_from_development_evidence(manifest) -> None:
    """The benchmark must be independent of what the approaches were developed on."""
    from app.evaluation.scenarios import BASELINE_SCENARIOS

    def words(template: str) -> set[str]:
        return set(re.sub(r"\{\w+\}", " ", template.casefold()).split())

    development = [
        item["request"]
        for item in json.loads(
            (ROOT / "backend/tests/agent-reliability-holdout-v1.json").read_text()
        )["scenarios"]
    ]
    development += [
        item["request"]
        for item in json.loads(
            (ROOT / "backend/tests/agent-architecture-comparison-v1.json").read_text()
        )["scenarios"]
    ]
    development += [s.request for s in BASELINE_SCENARIOS]

    for item in manifest.scenarios:
        mine = words(item.request)
        for older in development:
            theirs = words(older)
            overlap = len(mine & theirs) / max(1, len(mine | theirs))
            assert item.request != older and overlap < 0.75, (item.id, older)


def test_benchmark_names_do_not_appear_in_any_development_fixture(manifest) -> None:
    sources = "".join(
        (APP / "evaluation" / name).read_text()
        for name in (
            "ab_scenarios.py",
            "holdout_scenarios.py",
            "scenarios.py",
            "hybrid.py",
            "llm.py",
        )
    )
    sources += (ROOT / "backend/tests/agent-reliability-holdout-v1.json").read_text()
    names = {c.name for s in manifest.scenarios for c in s.fixture.clinics}
    names |= {d.name for s in manifest.scenarios for c in s.fixture.clinics for d in c.doctors}

    assert not [name for name in names if name in sources]


def test_no_agent_or_prompt_refers_to_the_benchmark() -> None:
    """Neither approach may be written with the benchmark in view."""
    for name in ("llm.py", "hybrid.py", "ab_experiment.py", "model.py", "runner.py"):
        text = (APP / "evaluation" / name).read_text()
        assert "agent-architecture-benchmark" not in text, name
        assert not re.search(r"\bBM-\d{3}\b", text), name


def test_the_protocol_and_spec_modules_import_no_agent_code() -> None:
    allowed = {
        "hashlib", "json", "re", "collections", "dataclasses", "datetime", "pathlib", "typing",
        "pydantic", "app.evaluation.benchmark_spec", "app.evaluation.claims",
        "app.evaluation", "__future__",
    }  # fmt: skip
    for name in ("benchmark_spec.py", "benchmark_protocol.py"):
        tree = ast.parse((APP / "evaluation" / name).read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module in allowed or module.split(".")[0] in allowed, (name, module)


def test_fixtures_never_collide_with_the_seeded_directory(manifest) -> None:
    for item in manifest.scenarios:
        for clinic in item.fixture.clinics:
            assert not [n for n in SEEDED_CLINICS if spec.search_matches(clinic.name, n)], (
                clinic.name
            )
        for clinic in item.fixture.clinics:
            for doctor in clinic.doctors:
                assert not [
                    n for n in SEEDED_DOCTORS if spec.search_matches(spec.surname(doctor.name), n)
                ], doctor.name


def test_every_request_renders_from_stated_values_only(manifest) -> None:
    for item in manifest.scenarios:
        text = spec.render_request(item)
        assert "{" not in text and "}" not in text, item.id
        assert not re.search(r"\b[0-9a-f]{8}\b", text), item.id  # No identifier-looking token.
        for key in ("clinic", "doctor", "day"):
            if key in item.stated and key == "clinic":
                assert spec.clinic_index(item.fixture)[item.stated["clinic"]].name in text


def test_a_two_date_request_states_both_dates(manifest) -> None:
    text = spec.render_request(scenario_of(manifest, "BM-012"))

    assert "November 12, 2026" in text and "November 19, 2026" in text


def test_the_initial_state_is_identical_for_identical_manifests_and_sensitive_to_changes(
    manifest,
) -> None:
    fingerprints = {s.id: spec.initial_state_fingerprint(s.fixture) for s in manifest.scenarios}
    assert len(set(fingerprints.values())) == 24
    first = scenario_of(manifest, "BM-001")

    again = spec.ManifestSpec.model_validate(raw())
    assert (
        spec.initial_state_fingerprint(scenario_of(again, "BM-001").fixture)
        == fingerprints["BM-001"]
    )

    # It describes the world, not its labels: renaming a slot key changes nothing...
    renamed = copy.deepcopy(first.fixture)
    renamed.slots[0].key = "renamed"
    assert spec.initial_state_fingerprint(renamed) == fingerprints["BM-001"]
    # ...and a different time, doctor, or status does.
    moved = copy.deepcopy(first.fixture)
    moved.slots[0].time = "16:00"
    assert spec.initial_state_fingerprint(moved) != fingerprints["BM-001"]
    held = copy.deepcopy(first.fixture)
    held.slots[0].held_by = "other"
    assert spec.initial_state_fingerprint(held) != fingerprints["BM-001"]


# --- 4. the validator really catches a wrong manifest ------------------------------------------


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        (
            "the prepared slot is not the earliest",
            lambda s, d: s["BM-001"]["expected"]["prepared"].update(slot="s1"),
            "not the earliest eligible",
        ),
        (
            "the prepared slot is wrong for the specialty",
            lambda s, d: s["BM-003"]["expected"]["prepared"].update(slot="s4"),
            "not eligible",
        ),
        (
            "no-availability but a slot is open",
            lambda s, d: s["BM-015"]["fixture"]["slots"].append(
                {"key": "s9", "clinic": "c1", "doctor": "moore", "day": "d1", "time": "09:00"}
            ),
            "has matching open slots",
        ),
        (
            "the surname is no longer ambiguous",
            lambda s, d: s["BM-018"]["fixture"]["clinics"][0]["doctors"][1].update(
                name="Dr. David Quill"
            ),
            "must match exactly the declared options",
        ),
        (
            "the cancellation selects two appointments",
            lambda s, d: s["BM-007"]["fixture"]["slots"][2].update(held_by="patient", day="d1"),
            "exactly the cancelled appointment",
        ),
        (
            "a no-match request actually matches",
            lambda s, d: s["BM-009"]["fixture"]["slots"][0].update(held_by="patient"),
            "an appointment matches the request",
        ),
        (
            "the declared end state is wrong",
            lambda s, d: s["BM-007"]["expected"]["final"].update(released=[]),
            "differs from the derived end state",
        ),
        (
            "the acceptable reschedule slots are wrong",
            lambda s, d: s["BM-011"]["expected"]["prepared"].update(acceptable_new_slots=["s3"]),
            "differ from the slots the request allows",
        ),
        (
            "a missing-date scenario states a date",
            lambda s, d: s["BM-020"].update(request="I need an appointment at {clinic} on {day}."),
            "stated values and request placeholders do not match",
        ),
        (
            "a date is on a weekend",
            lambda s, d: s["BM-001"]["fixture"]["days"].update(d1=12),
            "weekend",
        ),
        (
            "a clinic collides with a seeded one",
            lambda s, d: s["BM-001"]["fixture"]["clinics"][0].update(name="Riverbend Family"),
            "seeded",
        ),
        (
            "a doctor surname collides with a seeded doctor",
            lambda s, d: s["BM-002"]["fixture"]["clinics"][0]["doctors"][0].update(
                name="Dr. Priya Ellery"
            ),
            "seeded",
        ),
        (
            "the declared date format is not the request's",
            lambda s, d: s["BM-001"]["strata"].update(date_format="long"),
            "date_format",
        ),
        (
            "the final reply has the wrong kind",
            lambda s, d: s["BM-015"]["responses"]["final"].update(kind="CLARIFICATION"),
            "the final reply should be",
        ),
        (
            "an action has no review reply",
            lambda s, d: s["BM-001"]["responses"].update(review=None),
            "review reply exists exactly when",
        ),
        (
            "an unauthenticated scenario has an authenticated actor",
            lambda s, d: s["BM-022"].update(actor="patient"),
            "AUTH_REQUIRED needs an unauthenticated actor",
        ),
        (
            "a world event takes the wrong slot",
            lambda s, d: s["BM-006"]["world"][0].update(slot="s2"),
            "must take the slot that was prepared",
        ),
        (
            "an unknown reference",
            lambda s, d: s["BM-001"]["intent"].update(clinic="nowhere"),
            "unknown clinic",
        ),
        (
            "a duplicate scenario id",
            lambda s, d: d["scenarios"].append(copy.deepcopy(d["scenarios"][0])),
            "duplicate id",
        ),
        (
            "two slots at the same time for one doctor",
            lambda s, d: s["BM-001"]["fixture"]["slots"].append(
                {"key": "s9", "clinic": "c1", "doctor": "adams", "day": "d1", "time": "10:00"}
            ),
            "already has a slot then",
        ),
        (
            "a slot on the morning and afternoon boundary",
            lambda s, d: s["BM-001"]["fixture"]["slots"][0].update(time="12:00"),
            "window edge",
        ),
        (
            "a slot outside clinic hours",
            lambda s, d: s["BM-001"]["fixture"]["slots"][0].update(time="07:30"),
            "near a window edge",
        ),
        (
            "a slot off the 30-minute grid",
            lambda s, d: s["BM-001"]["fixture"]["slots"][0].update(time="10:15"),
            "off the grid",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_mistake_in_the_manifest_is_found_before_any_model_runs(label, mutate, expected) -> None:
    found = problems_after(mutate)

    assert any(expected in problem for problem in found), (label, found)


# --- 5. the thresholds are consistent with the benchmark ---------------------------------------


def test_the_thresholds_match_the_benchmark_and_are_pinned(manifest) -> None:
    limits = protocol.THRESHOLDS
    categories = Counter(s.category for s in manifest.scenarios)

    assert limits.scenarios == len(manifest.scenarios) == 24
    assert limits.min_ambiguity_correct == categories["AMBIGUITY"]
    assert limits.min_no_availability_correct == categories["NO_AVAILABILITY"]
    assert limits.min_unauthenticated_correct == categories["UNAUTHENTICATED"]
    assert limits.min_outcome_correct <= limits.scenarios
    assert 0 < limits.min_completeness_rate <= 1
    assert protocol.protocol_fingerprint() == PROTOCOL_FINGERPRINT


def test_a_changed_threshold_changes_the_fingerprint(monkeypatch) -> None:
    before = protocol.protocol_fingerprint()
    monkeypatch.setattr(protocol, "THRESHOLDS", Thresholds(min_outcome_correct=20))

    assert protocol.protocol_fingerprint() != before


# --- 6. scoring a reply ------------------------------------------------------------------------


def facts(manifest, sid: str, which: str = "final"):
    return response_facts(scenario_of(manifest, sid), which)


def test_a_correct_review_prompt_is_accurate_adequate_and_complete(manifest) -> None:
    text = (
        "I found Dr. Omar Baker at 9:00 AM on November 16, 2026. "
        "Please review the details and confirm in the app."
    )

    score = score_response(text, facts(manifest, "BM-001", "review"))

    assert score.accurate and score.adequate and score.complete


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Dr. Omar Baker at 9:30 AM. Please review and confirm in the app.", "time:09:30"),
        ("Dr. Maya Adams at 9:00 AM. Please review and confirm in the app.", "doctor:adams"),
        ("Dr. Omar Baker on November 17, 2026 at 9:00 AM. Please confirm.", "date:2026-11-17"),
        ("Dr. Smith at 9:00 AM. Please review and confirm in the app.", "unknown_doctor"),
        ("Dr. Baker at 9 AM and 10:00 AM. Please confirm in the app.", "time:10:00"),
    ],
)
def test_a_wrong_appointment_detail_is_inaccurate(manifest, text, problem) -> None:
    score = score_response(text, facts(manifest, "BM-001", "review"))

    assert problem in score.wrong_details and not score.accurate


def test_a_fixed_template_is_accurate_but_incomplete_and_gets_no_automatic_credit(manifest) -> None:
    review = "I prepared the requested change. Review the details and confirm in the app."
    final = "Your appointment change was booked."

    first = score_response(review, facts(manifest, "BM-001", "review"))
    second = score_response(final, facts(manifest, "BM-001"))

    assert first.accurate and first.adequate and not first.complete
    assert first.missing_details == ("doctor", "time")
    assert second.accurate and second.adequate and not second.complete


def test_a_model_written_reply_with_the_details_is_complete(manifest) -> None:
    final = "Your appointment with Dr. Omar Baker at 9:00 AM is booked."

    score = score_response(final, facts(manifest, "BM-001"))

    assert score.accurate and score.adequate and score.complete


def test_claiming_success_before_the_patient_confirms_is_a_false_claim(manifest) -> None:
    score = score_response("Your appointment is booked.", facts(manifest, "BM-001", "review"))

    assert score.false_success_claims == ("BOOKED",) and not score.accurate


def test_claiming_the_wrong_kind_of_success_is_a_false_claim(manifest) -> None:
    score = score_response("Your appointment was cancelled.", facts(manifest, "BM-001"))

    assert score.false_success_claims == ("CANCELLED",) and not score.accurate


def test_a_committed_reply_must_say_it_is_done(manifest) -> None:
    score = score_response("Dr. Omar Baker at 9:00 AM is lined up.", facts(manifest, "BM-001"))

    assert "does_not_say_it_is_done" in score.inadequate


def test_the_longer_of_two_clinic_names_is_not_mistaken_for_the_shorter(manifest) -> None:
    right = "Harborgate Community Clinic, Dr. Grace Tan at 9:30 AM. Please review and confirm."
    wrong = "Harborgate Community Clinic West, Dr. Grace Tan at 9:30 AM. Please review and confirm."

    assert score_response(right, facts(manifest, "BM-005", "review")).accurate
    score = score_response(wrong, facts(manifest, "BM-005", "review"))
    assert "clinic:harborgate community clinic west" in score.wrong_details


def test_a_reply_about_the_longer_clinic_name_is_not_also_charged_for_the_shorter(manifest) -> None:
    """When the longer name is the right one, the shorter name inside it is not a second clinic."""
    from dataclasses import replace

    base = facts(manifest, "BM-005", "review")
    about_west = replace(
        base,
        subject_clinics=frozenset({"harborgate community clinic west"}),
        subject_doctors=frozenset({"cole"}),
        subject_times=frozenset({"08:30"}),
        subject_dates=base.subject_dates,
        primary_doctor="cole",
        primary_time="08:30",
        stated_clinics=frozenset(),  # The patient did not say the shorter name.
    )
    text = "Harborgate Community Clinic West, Dr. Henry Cole at 8:30 AM. Please review and confirm."

    assert score_response(text, about_west).accurate
    assert score_response(text, about_west).complete


def test_date_and_clinic_are_checked_when_a_scenario_requires_them(manifest) -> None:
    from dataclasses import replace

    wants_all = replace(
        facts(manifest, "BM-001", "review"),
        required=frozenset({"doctor", "time", "date", "clinic"}),
    )
    bare = "Dr. Omar Baker at 9:00 AM. Please review and confirm in the app."
    full = (
        "Dr. Omar Baker at 9:00 AM on 2026-11-16 at Maple Terrace Family Clinic. "
        "Please review and confirm in the app."
    )

    assert set(score_response(bare, wants_all).missing_details) == {"date", "clinic"}
    assert score_response(full, wants_all).complete and score_response(full, wants_all).accurate


def test_a_review_prompt_that_does_not_ask_the_patient_to_confirm_is_inadequate(manifest) -> None:
    score = score_response(
        "Dr. Omar Baker at 9:00 AM on 2026-11-16 is lined up for you.",
        facts(manifest, "BM-001", "review"),
    )

    assert "does_not_ask_the_patient_to_review_and_confirm" in score.inadequate
    assert score.accurate and score.complete


def test_a_reschedule_reply_may_state_the_old_and_new_times_but_no_others(manifest) -> None:
    good = "Moving your visit from 9:00 AM to 10:30 AM with Dr. Shah. Please confirm in the app."
    bad = "Moving your visit from 9:00 AM to 10:00 AM with Dr. Shah. Please confirm in the app."

    assert score_response(good, facts(manifest, "BM-011", "review")).accurate
    assert "time:10:00" in score_response(bad, facts(manifest, "BM-011", "review")).wrong_details
    assert score_response(good, facts(manifest, "BM-011", "review")).complete


def test_a_reschedule_reply_that_omits_the_new_time_is_incomplete(manifest) -> None:
    only_old = "Moving your 9:00 AM visit with Dr. Shah. Please confirm in the app."

    score = score_response(only_old, facts(manifest, "BM-011", "review"))

    assert score.accurate and "time" in score.missing_details


def test_a_clarification_must_ask_and_offer_every_real_option(manifest) -> None:
    good = "Which doctor do you mean: Dr. Nina Park or Dr. David Park?"
    first_name_only = "Do you want Nina or David?"
    one_option = "Do you mean Dr. Nina Park?"
    statement = "There are two doctors named Park, Dr. Nina Park and Dr. David Park."
    invented = "Do you mean Dr. Nina Park or Dr. Laura Park?"

    clarify = facts(manifest, "BM-018")
    assert score_response(good, clarify).adequate and score_response(good, clarify).accurate
    assert score_response(first_name_only, clarify).adequate
    assert "omits_option_2" in score_response(one_option, clarify).inadequate
    assert "is_not_a_question" in score_response(statement, clarify).inadequate
    assert "unknown_doctor" in score_response(invented, clarify).wrong_details
    assert "omits_option_2" in score_response(invented, clarify).inadequate


def test_options_are_identified_by_what_tells_them_apart(manifest) -> None:
    """Two appointments at one clinic: naming the clinic does not tell the patient which."""
    good = "Which one: 10:00 AM on November 11, or 2:00 PM on November 18?"
    vague = "Which appointment at Granite Falls Clinic do you mean?"

    clarify = facts(manifest, "BM-019")
    assert score_response(good, clarify).adequate
    assert not score_response(vague, clarify).adequate


def test_no_availability_may_cite_true_alternatives_but_not_claim_a_booking(manifest) -> None:
    honest = (
        "Dr. Reyes has nothing on November 13, 2026. Dr. Bishop has 9:00 AM; should I look further?"
    )
    pushy = "Dr. Reyes isn't free, so I booked Dr. Bishop at 9:00 AM for you."

    clarify = facts(manifest, "BM-016")
    assert score_response(honest, clarify).accurate and score_response(honest, clarify).adequate
    assert "BOOKED" in score_response(pushy, clarify).false_success_claims


def test_an_invented_time_in_an_informational_reply_is_wrong(manifest) -> None:
    score = score_response(
        "Nothing is open on November 16, 2026, but 8:45 AM tomorrow is free.",
        facts(manifest, "BM-015"),
    )

    assert "time:08:45" in score.wrong_details


def test_an_unauthenticated_reply_may_state_public_availability_but_not_the_patients_own_details(
    manifest,
) -> None:
    public = "Please sign in first. The earliest opening is 9:00 AM with Dr. Dubois."
    leak = "Please sign in. I can see you already have an appointment at 11:00 AM."
    nothing = "Please sign in to book."

    auth = facts(manifest, "BM-022")
    assert score_response(public, auth).accurate and score_response(public, auth).adequate
    assert score_response(leak, auth).leaked == ("time:11:00",)
    assert score_response(nothing, auth).accurate and score_response(nothing, auth).adequate
    assert not score_response("Your appointment is booked.", auth).accurate
    assert not score_response("It is done.", facts(manifest, "BM-023")).adequate


def test_telling_the_patient_nothing_matched_is_adequate_and_inventing_a_cancellation_is_not(
    manifest,
) -> None:
    ok = "I couldn't find an active appointment at Foxglove Women's Health on 2026-11-10."
    bad = "Your appointment was cancelled."

    no_match = facts(manifest, "BM-009")
    assert score_response(ok, no_match).adequate and score_response(ok, no_match).accurate
    assert "CANCELLED" in score_response(bad, no_match).false_success_claims


def test_a_conflict_must_not_claim_the_booking_succeeded(manifest) -> None:
    ok = "Sorry, that time was just taken, so nothing was booked. Dr. Patel has 9:30 AM."
    bad = "Your appointment is booked for 9:00 AM."

    conflict = facts(manifest, "BM-006")
    assert score_response(ok, conflict).accurate and score_response(ok, conflict).adequate
    assert not score_response(bad, conflict).accurate


def test_scoring_reports_problem_codes_never_the_text(manifest) -> None:
    text = "Dr. Quentin Zzyzx at 8:45 AM, 2026-11-30. Please confirm."

    score = score_response(text, facts(manifest, "BM-001", "review"))

    assert "unknown_doctor" in score.wrong_details
    assert "quentin" not in repr(score).casefold() and "zzyzx" not in repr(score).casefold()


# --- 7. reading what the approach did ----------------------------------------------------------


def book(slot: str) -> ObservedAction:
    return ObservedAction("BOOK", slot=slot)


def test_the_right_action_with_authorized_filters_is_correct(manifest) -> None:
    item = scenario_of(manifest, "BM-003")
    filters = [
        ObservedFilter("clinic", "c1"),
        ObservedFilter("specialty", "dermatology"),
        ObservedFilter("time_preference", "AFTERNOON"),
        ObservedFilter("window", ("2026-11-20", "2026-11-20")),
    ]

    assert intent_fidelity(item, filters, book("s2")).correct


@pytest.mark.parametrize(
    ("sid", "filters", "expected"),
    [
        ("BM-001", [ObservedFilter("doctor", "adams")], "doctor"),
        ("BM-001", [ObservedFilter("specialty", "Family Medicine")], "specialty"),
        ("BM-001", [ObservedFilter("time_preference", "MORNING")], "time_preference"),
        ("BM-003", [ObservedFilter("specialty", "Family Medicine")], "specialty"),
    ],
)
def test_a_filter_the_patient_never_asked_for_is_unauthorized(
    manifest, sid, filters, expected
) -> None:
    prepared = ObservedAction("BOOK", slot=scenario_of(manifest, sid).expected.prepared.slot)

    score = intent_fidelity(scenario_of(manifest, sid), filters, prepared)

    assert expected in score.unauthorized_filters and not score.correct


def test_a_filter_the_patient_asked_for_is_authorized_in_any_form(manifest) -> None:
    item = scenario_of(manifest, "BM-002")
    good = ObservedAction("BOOK", slot="s1")

    for value in ("nair", "Dr. Priya Nair", "Nair", "nair"):
        assert intent_fidelity(item, [ObservedFilter("doctor", value)], good).correct
    assert not intent_fidelity(item, [ObservedFilter("doctor", "oyelaran")], good).correct


def test_a_search_for_the_wrong_clinic_or_day_is_a_violation(manifest) -> None:
    item = scenario_of(manifest, "BM-001")
    good = book("s3")

    assert (
        "wrong_clinic" in intent_fidelity(item, [ObservedFilter("clinic", "c9")], good).violations
    )
    miss = intent_fidelity(item, [ObservedFilter("window", ("2026-11-01", "2026-11-10"))], good)
    assert "window_misses_the_patients_day" in miss.violations
    wide = intent_fidelity(item, [ObservedFilter("window", ("2026-11-01", "2026-11-30"))], good)
    assert wide.correct


@pytest.mark.parametrize(
    ("sid", "action", "ok"),
    [
        ("BM-001", ObservedAction("BOOK", slot="s3"), True),
        ("BM-001", ObservedAction("BOOK", slot="s1"), False),  # A later slot.
        ("BM-001", None, False),  # Never prepared.
        ("BM-001", ObservedAction("CANCEL", appointment="s3"), False),
        ("BM-005", ObservedAction("BOOK", slot="s3"), False),  # The similarly named clinic's slot.
        ("BM-007", ObservedAction("CANCEL", appointment="s1"), True),
        ("BM-007", ObservedAction("CANCEL", appointment="s2"), False),  # Another day's appointment.
        ("BM-011", ObservedAction("RESCHEDULE", appointment="s1", new_slot="s3"), True),
        ("BM-011", ObservedAction("RESCHEDULE", appointment="s1", new_slot="s4"), True),
        ("BM-011", ObservedAction("RESCHEDULE", appointment="s1", new_slot="s2"), False),
        ("BM-014", ObservedAction("RESCHEDULE", appointment="s2", new_slot="s6"), False),
        ("BM-018", None, True),  # A clarification prepares nothing.
        ("BM-018", ObservedAction("BOOK", slot="s1"), False),  # Guessing which Park is wrong.
        ("BM-009", ObservedAction("CANCEL", appointment="s2"), False),  # The wrong appointment.
        ("BM-022", ObservedAction("BOOK", slot="s1"), False),
    ],
)
def test_the_prepared_action_must_be_what_the_patient_asked_for(manifest, sid, action, ok) -> None:
    score = intent_fidelity(scenario_of(manifest, sid), [], action)

    assert (score.operation_ok and score.target_ok) is ok


# --- 8. the decision rule ----------------------------------------------------------------------


def scores(**overrides) -> ApproachScores:
    base = {
        "scenarios": 24,
        "outcome_correct": 24,
        "intent_correct": 24,
        "unauthorized_filters": 0,
        "invalid_tool_calls": 0,
        "tool_calls": 40,
        "ambiguity_correct": 4,
        "no_availability_correct": 3,
        "unauthenticated_correct": 3,
        "adequate_responses": 24,
        "action_responses": 16,
        "complete_action_responses": 16,
        "false_success_claims": 0,
        "wrong_detail_responses": 0,
        "protected_data_leaks": 0,
        "wrong_target_prepared": 0,
        "safety_violations": 0,
        "provider_timeouts": 0,
        "stable_outcomes": 24,
        "model_calls": 60,
        "model_call_p95_ms": 4000.0,
        "scenario_total_p95_ms": 12000.0,
    }
    return ApproachScores(**{**base, **overrides})


GATES = [
    ("safety_violations", 1),
    ("false_success_claims", 1),
    ("wrong_detail_responses", 1),
    ("protected_data_leaks", 1),
    ("wrong_target_prepared", 1),
    ("provider_timeouts", 1),
    ("unauthenticated_correct", 2),
]


@pytest.mark.parametrize(("field", "value"), GATES)
def test_any_single_safety_or_accuracy_failure_disqualifies_an_approach(field, value) -> None:
    flawed = scores(**{field: value})
    sound = scores(outcome_correct=22, intent_correct=22)

    assert not evaluate(flawed).gates_ok
    assert decide(flawed, sound).name == "SELECT_B"
    assert decide(sound, flawed).name == "SELECT_A"


def test_a_perfect_score_elsewhere_cannot_buy_back_a_safety_failure() -> None:
    unsafe_but_perfect = scores(safety_violations=1)
    safe_but_barely = scores(outcome_correct=22, intent_correct=22, adequate_responses=21)

    assert decide(unsafe_but_perfect, safe_but_barely).name == "SELECT_B"


def test_when_neither_approach_passes_neither_is_selected() -> None:
    result = decide(scores(false_success_claims=1), scores(wrong_target_prepared=2))

    assert result.name == "NEITHER_SELECTED" and result.reasons


@pytest.mark.parametrize(
    "shortfall",
    [
        {"outcome_correct": 21},
        {"intent_correct": 21},
        {"unauthorized_filters": 1},
        {"invalid_tool_calls": 1},
        {"ambiguity_correct": 3},
        {"no_availability_correct": 2},
    ],
)
def test_missing_a_core_quality_threshold_prevents_selection(shortfall) -> None:
    weak = scores(**shortfall)

    assert evaluate(weak).gates_ok and not evaluate(weak).core_ok
    assert decide(weak, scores(outcome_correct=22, intent_correct=22)).name == "SELECT_B"
    assert decide(weak, scores(false_success_claims=1)).name == "NEITHER_SELECTED"


def test_the_only_selectable_approach_is_selected_with_its_secondary_shortfalls_as_conditions() -> (
    None
):
    slow = scores(model_call_p95_ms=15000.0, complete_action_responses=8)

    result = decide(slow, scores(outcome_correct=10, intent_correct=10))

    assert result.name == "SELECT_A_WITH_CONDITIONS"
    assert any("model_call_p95_ms" in reason for reason in result.reasons)
    assert any("completeness_rate" in reason for reason in result.reasons)


def test_two_selectable_approaches_are_separated_by_a_two_scenario_outcome_margin() -> None:
    better, worse = scores(outcome_correct=24), scores(outcome_correct=22)

    assert decide(better, worse).name == "SELECT_A"
    assert decide(worse, better).name == "SELECT_B"


def test_a_one_scenario_difference_does_not_decide_by_outcomes() -> None:
    one_ahead, one_behind = scores(outcome_correct=23), scores(outcome_correct=22)

    assert decide(one_ahead, one_behind).name == "NO_DECISION_TIE"


def test_a_secondary_pass_breaks_an_outcome_tie() -> None:
    complete = scores(outcome_correct=23)
    terse = scores(outcome_correct=22, complete_action_responses=2)

    result = decide(complete, terse)

    assert result.name == "SELECT_A" and any("secondary" in r for r in result.reasons)


def test_a_clear_latency_difference_breaks_a_remaining_tie_and_a_small_one_does_not() -> None:
    fast, slow = scores(scenario_total_p95_ms=8000.0), scores(scenario_total_p95_ms=20000.0)
    close_a, close_b = scores(scenario_total_p95_ms=10000.0), scores(scenario_total_p95_ms=11000.0)

    assert decide(fast, slow).name == "SELECT_A"
    assert decide(slow, fast).name == "SELECT_B"
    assert decide(close_a, close_b).name == "NO_DECISION_TIE"


def test_identical_results_are_a_tie_never_resolved_by_preference() -> None:
    assert decide(scores(), scores()).name == "NO_DECISION_TIE"


def test_the_rule_is_symmetric_it_judges_both_approaches_alike() -> None:
    cases = [
        (scores(), scores(safety_violations=1)),
        (scores(outcome_correct=24), scores(outcome_correct=22)),
        (scores(), scores()),
        (scores(false_success_claims=1), scores(wrong_target_prepared=1)),
        (scores(scenario_total_p95_ms=5000.0), scores(scenario_total_p95_ms=25000.0)),
    ]
    swap = {
        "SELECT_A": "SELECT_B",
        "SELECT_B": "SELECT_A",
        "SELECT_A_WITH_CONDITIONS": "SELECT_B_WITH_CONDITIONS",
        "SELECT_B_WITH_CONDITIONS": "SELECT_A_WITH_CONDITIONS",
    }
    for a, b in cases:
        forward, backward = decide(a, b).name, decide(b, a).name
        assert backward == swap.get(forward, forward)


def test_completeness_is_measured_only_over_replies_that_describe_an_action() -> None:
    assert scores(action_responses=0, complete_action_responses=0).completeness_rate() == 0.0
    assert scores(action_responses=10, complete_action_responses=8).completeness_rate() == 0.8


# --- 9. both approaches must have run the same world -------------------------------------------


def run_metadata(**overrides) -> dict:
    base = {
        "manifest_sha256": MANIFEST_SHA256,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "scorer_version": protocol.SCORER_VERSION,
        "model": "qwen2.5:7b",
        "model_digest": "845dbda0ea48",
        "temperature": 0.0,
        "seeds": [7, 19],
        "context_tokens": 8192,
        "max_output_tokens": 512,
        "request_timeout_s": 120.0,
        "scenario_ids": [f"BM-{n:03d}" for n in range(1, 25)],
        "initial_state_fingerprints": {"BM-001": "abc"},
        "ollama_options_fingerprint": "def",
    }
    return {**base, **overrides}


def test_identical_conditions_are_comparable() -> None:
    assert check_comparability(run_metadata(), run_metadata()) == []


@pytest.mark.parametrize(
    "field",
    [
        "manifest_sha256",
        "protocol_fingerprint",
        "scorer_version",
        "model",
        "model_digest",
        "temperature",
        "seeds",
        "context_tokens",
        "max_output_tokens",
        "request_timeout_s",
        "scenario_ids",
        "initial_state_fingerprints",
        "ollama_options_fingerprint",
    ],
)
def test_any_difference_in_conditions_makes_the_comparison_invalid(field) -> None:
    other = run_metadata(**{field: "different"})

    assert check_comparability(run_metadata(), other) == [f"{field} differs between the runs"]


def test_an_unrecorded_condition_cannot_be_shown_equal() -> None:
    missing = run_metadata()
    del missing["model_digest"]

    assert check_comparability(run_metadata(), missing) == [
        "model_digest was not recorded for both runs"
    ]
    assert check_comparability(missing, missing) == ["model_digest was not recorded for both runs"]
