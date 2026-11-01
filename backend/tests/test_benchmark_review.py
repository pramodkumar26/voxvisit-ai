"""The blinded review sheet, the safeguards around raw conversations, and the scorer's helpers.

None of this needs a database or a model. It tests the parts of the benchmark's execution that
turn what happened into measurements and into material a person can check without bias.
"""

import json
import math
import shutil
import types
import uuid
from pathlib import Path

import pytest

from app.evaluation import benchmark_builder as builder
from app.evaluation import benchmark_review as review
from app.evaluation import benchmark_run as run
from app.evaluation import benchmark_scoring as scoring
from app.evaluation import benchmark_spec as spec
from app.evaluation.intent import OneOf, pending_mismatches

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, spec.ScenarioSpec]:
    return {s.id: s for s in spec.load_manifest().scenarios}


def source(
    n: int, *, approach="approach-alpha", accurate=True, adequate=True, complete=True, sid=None
):
    return review.AuditSource(
        approach=approach,
        seed=7,
        scenario_id=sid or "BM-001",
        request=f"request number {n}",
        slot="final",
        kind="COMMITTED",
        text=f"reply number {n} says something",
        accurate=accurate,
        adequate=adequate,
        complete=complete,
    )


def test_no_two_slots_in_a_clinic_start_together_so_tool_ordering_cannot_vary_the_answer(
    scenarios,
) -> None:
    """Tools order equal start times by identifier, and identifiers differ in every database."""
    for scenario in scenarios.values():
        starts = [(s.clinic, s.day, s.time) for s in scenario.fixture.slots]
        assert len(starts) == len(set(starts)), scenario.id


# --- choosing what to review -------------------------------------------------------------------


def test_every_flagged_reply_is_reviewed_and_a_quarter_of_the_rest() -> None:
    flagged = [
        source(1, accurate=False),
        source(2, adequate=False),
        source(3, complete=False),  # The verdicts disagree: accurate and adequate, not complete.
    ]
    clean = [source(n) for n in range(10, 50)]  # 40 replies with all three verdicts agreeing.

    chosen = review.select_for_audit(flagged + clean, seed=3)

    assert all(item in chosen for item in flagged)
    assert len(chosen) == 3 + math.ceil(0.25 * 40)
    assert len({item.text for item in chosen}) == len(chosen)  # No reply twice.


def test_the_random_sample_is_repeatable_and_depends_on_the_seed() -> None:
    clean = [source(n) for n in range(60)]

    first = review.select_for_audit(clean, seed=5)

    assert first == review.select_for_audit(clean, seed=5)
    assert first != review.select_for_audit(clean, seed=6)


def test_a_reply_that_was_never_given_is_not_reviewed() -> None:
    def reply(present):
        return types.SimpleNamespace(slot="final", kind="COMMITTED", present=present, text="words")

    verdicts = types.SimpleNamespace(accurate=True, adequate=True, complete=True)
    record = types.SimpleNamespace(
        approach="x",
        seed=7,
        scenario_id="BM-001",
        request="ask",
        replies=[reply(False), reply(True)],
        score=types.SimpleNamespace(replies=[verdicts, verdicts]),
    )

    assert len(review.sources_from([record])) == 1


# --- the sheet does not reveal what it must not ------------------------------------------------


def materials(tmp_path, scenarios, items, *, seed=7):
    return review.write_review_materials(tmp_path, items, scenarios, seed=seed)


def varied_sources(count=24):
    items = []
    for n in range(count):
        items.append(
            source(
                n,
                approach="approach-alpha" if n % 2 else "approach-beta",
                accurate=n % 5 != 0,
                adequate=n % 3 != 0,
                sid=f"BM-{1 + n % 24:03d}",
            )
        )
    return items


def test_the_sheet_hides_the_approach_the_seed_the_scenario_and_the_rules_verdicts(
    tmp_path, scenarios
) -> None:
    sheet_path, _, _ = materials(tmp_path, scenarios, varied_sources())
    sheet = sheet_path.read_text()

    for hidden in ("approach-alpha", "approach-beta", "seed", "BM-0", "BM-1", "True", "False"):
        assert hidden not in sheet
    assert sheet.count("**The patient said:**") == sheet.count("## R-") > 0
    assert "reply number" in sheet and "What is true" in sheet


def test_the_key_is_separate_and_maps_every_item_back(tmp_path, scenarios) -> None:
    items = varied_sources()
    sheet_path, key_path, template_path = materials(tmp_path, scenarios, items)
    key = json.loads(key_path.read_text())
    template = json.loads(template_path.read_text())

    assert (
        set(key)
        == set(template)
        == {
            line.removeprefix("## ")
            for line in sheet_path.read_text().splitlines()
            if line.startswith("## R-")
        }
    )
    for entry in key.values():
        assert entry["approach"] in {"approach-alpha", "approach-beta"}
        assert set(entry["rules"]) == {"accurate", "adequate", "complete"}
    assert all(set(v.values()) == {None} for v in template.values())
    assert "approach-alpha" not in template_path.read_text()


def test_items_are_shuffled_so_position_does_not_reveal_the_approach(tmp_path, scenarios) -> None:
    items = [source(n, approach="approach-alpha", accurate=False) for n in range(10)]
    items += [source(n + 10, approach="approach-beta", accurate=False) for n in range(10)]

    _, key_path, _ = materials(tmp_path, scenarios, items)
    order = [entry["approach"] for entry in json.loads(key_path.read_text()).values()]

    assert order != sorted(order) and order != sorted(order, reverse=True)


def test_the_same_seed_gives_the_same_sheet_and_another_seed_a_different_one(
    tmp_path, scenarios
) -> None:
    items = varied_sources()
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()

    one = materials(tmp_path / "a", scenarios, items, seed=1)
    again = materials(tmp_path / "b", scenarios, items, seed=1)
    other = materials(tmp_path / "c", scenarios, items, seed=2)

    assert one[0].read_text() == again[0].read_text()
    assert one[0].read_text() != other[0].read_text()


def test_the_facts_card_gives_only_what_a_correct_reply_may_say(scenarios) -> None:
    action = review.facts_card(scenarios["BM-001"], "review")
    clarification = review.facts_card(scenarios["BM-018"], "final")
    refusal = review.facts_card(scenarios["BM-023"], "final")

    assert "Omar Baker" in action and "9:00 AM" in action and "Maya Adams" not in action
    assert "Nina Park" in clarification and "David Park" in clarification
    assert "own appointment details may not" in refusal
    assert "10:00 AM" not in refusal  # The patient's protected appointment is not on the card.


# --- how far the rules can be trusted ----------------------------------------------------------


def audit_key(count=10):
    return {
        f"R-{n:03d}": {"rules": {"accurate": True, "adequate": True, "complete": True}}
        for n in range(1, count + 1)
    }


def verdicts(**flipped):
    return {"accurate": True, "adequate": True, "complete": True, **flipped}


def test_a_verdict_is_reliable_up_to_a_ten_percent_disagreement() -> None:
    key = audit_key()
    judgments = {name: verdicts() for name in key}
    judgments["R-001"] = verdicts(accurate=False)  # 1 of 10 disagrees: exactly 10%.

    report = review.audit_agreement(key, judgments)["per_verdict"]

    assert report["accurate"] == {"compared": 10, "disagreements": 1, "rate": 0.1, "reliable": True}
    assert report["adequate"]["reliable"] and report["complete"]["reliable"]


def test_more_than_ten_percent_disagreement_makes_the_verdict_unreliable() -> None:
    key = audit_key()
    judgments = {name: verdicts() for name in key}
    judgments["R-001"] = verdicts(adequate=False)
    judgments["R-002"] = verdicts(adequate=False)  # 2 of 10.

    report = review.audit_agreement(key, judgments)["per_verdict"]

    assert report["adequate"]["rate"] == 0.2 and not report["adequate"]["reliable"]
    assert report["accurate"]["reliable"]  # The other verdicts are judged on their own.


def test_an_unjudged_verdict_is_skipped_and_an_empty_audit_is_not_reliable() -> None:
    key = audit_key(2)

    skipped = review.audit_agreement(key, {"R-001": verdicts(complete=None), "R-002": verdicts()})
    empty = review.audit_agreement(key, {})

    assert skipped["per_verdict"]["complete"]["compared"] == 1
    assert empty["per_verdict"]["accurate"] == {
        "compared": 0,
        "disagreements": 0,
        "rate": None,
        "reliable": False,
    }


def test_a_judgment_for_an_item_that_is_not_on_the_sheet_is_an_error() -> None:
    with pytest.raises(ValueError, match="R-999"):
        review.audit_agreement(audit_key(2), {"R-999": verdicts()})


# --- raw conversations stay out of Git ---------------------------------------------------------


def test_a_directory_outside_the_repository_is_accepted(tmp_path) -> None:
    assert run.ensure_untracked_directory(tmp_path / "runs").is_dir()


@pytest.mark.parametrize("relative", ["docs/benchmark", "backend/benchmark-runs", "benchmark-runs"])
def test_a_directory_Git_would_track_is_refused_and_not_created(relative) -> None:
    target = ROOT / relative
    existed = target.exists()

    with pytest.raises(ValueError, match="Git does not ignore"):
        run.ensure_untracked_directory(target)

    assert target.exists() == existed


def test_the_local_only_directory_is_accepted_because_Git_ignores_it() -> None:
    created = not (ROOT / ".local").exists()
    target = ROOT / ".local" / "pytest-benchmark-guard"
    try:
        assert run.ensure_untracked_directory(target) == target.resolve()
    finally:
        shutil.rmtree(target, ignore_errors=True)
        if created:
            shutil.rmtree(ROOT / ".local", ignore_errors=True)


# --- the command line --------------------------------------------------------------------------


def test_an_unreachable_model_server_exits_cleanly_before_running_anything(tmp_path) -> None:
    with pytest.raises(SystemExit) as stop:
        run.main(
            ["run", "--approach", "A", "--output-dir", str(tmp_path), "--url", "http://127.0.0.1:9"]
        )

    assert stop.value.code == 2 and not list(tmp_path.iterdir())


def test_a_model_server_that_is_not_on_this_machine_is_refused_before_any_connection(
    tmp_path, monkeypatch
) -> None:
    from app.evaluation import cli

    def must_not_connect(config):
        raise AssertionError("a connection was attempted to a remote server")

    monkeypatch.setattr(cli, "served_model_digest", must_not_connect)

    with pytest.raises(SystemExit) as stop:
        run.main(
            [
                "run",
                "--approach",
                "B",
                "--output-dir",
                str(tmp_path),
                "--url",
                "http://example.com:11434",
            ]
        )

    assert stop.value.code == 2 and not list(tmp_path.iterdir())


def test_the_two_approaches_are_built_with_identical_model_settings() -> None:
    from app.evaluation.hybrid import HybridControllerModel
    from app.evaluation.llm import APPROACH_A_SYSTEM_PROMPT, OllamaModel

    a = run.approach_factory("A", "qwen2.5:7b", "http://127.0.0.1:11434")(None, 7, None)
    b = run.approach_factory("B", "qwen2.5:7b", "http://127.0.0.1:11434")(None, 7, None)

    assert isinstance(a, OllamaModel) and isinstance(b, HybridControllerModel)
    assert a.config == b.extractor.config  # Same weights, sampling, limits, timeout, and seed.
    assert a.real_transport and b.extractor.real_transport
    assert a._messages[0]["content"] == APPROACH_A_SYSTEM_PROMPT
    other_seed = run.approach_factory("A", "qwen2.5:7b", "http://127.0.0.1:11434")(None, 19, None)
    assert other_seed.config.seed == 19 and other_seed.config != a.config
    with pytest.raises(ValueError):
        run.approach_factory("C", "qwen2.5:7b", "http://127.0.0.1:11434")


# --- the trusted-UI policy accepts any of several acceptable targets ---------------------------


def stored(**overrides):
    base = {
        "owner": "p1",
        "status": "ACTIVE",
        "action_type": "RESCHEDULE",
        "target_id": "appt",
        "expected_versions": {"new_slot_id": "n2", "old_slot_id": "o"},
        "summary": {},
    }
    return {**base, **overrides}


def mismatches(stored_action, **expected):
    return pending_mismatches(
        stored_action,
        action="RESCHEDULE",
        target="appt",
        expected=expected,
        summary={},
        actor_patient_id="p1",
    )


def test_one_of_accepts_each_acceptable_value_and_nothing_else() -> None:
    allowed = OneOf("n1", "n2")

    assert mismatches(stored(), new_slot_id=allowed) == []
    assert mismatches(stored(expected_versions={"new_slot_id": "n1"}), new_slot_id=allowed) == []
    assert mismatches(stored(expected_versions={"new_slot_id": "n9"}), new_slot_id=allowed) == [
        "expected.new_slot_id"
    ]
    assert mismatches(stored(), new_slot_id="n1") == ["expected.new_slot_id"]  # Exact still exact.
    assert OneOf(uuid.UUID(int=1)).accepts(str(uuid.UUID(int=1)))


# --- reading a conversation --------------------------------------------------------------------


def fake_world():
    ids = {name: uuid.uuid4() for name in ("c1", "adams", "s1", "s3", "a1")}
    return builder.BuiltWorld(
        patient_code="EVP-TEST-A",
        other_code="EVP-TEST-B",
        patient_id=uuid.uuid4(),
        other_id=uuid.uuid4(),
        clinic_ids={"c1": ids["c1"]},
        doctor_ids={"adams": ids["adams"]},
        slot_ids={"s1": ids["s1"], "s3": ids["s3"]},
        appointment_ids={"s1": ids["a1"]},
    ), ids


def call(tool, arguments, ok=True, code=None):
    return {"kind": "tool", "tool": tool, "arguments": arguments, "ok": ok, "error_code": code}


def test_filters_are_read_in_the_manifests_own_terms(scenarios) -> None:
    world, ids = fake_world()
    scenario = scenarios["BM-001"]
    calls = [
        call("search_clinic_directory", {"clinic_name": "Maple Terrace Family Clinic"}),
        call("search_clinic_directory", {"clinic_name": "Maple Terrace", "doctor_name": "Adams"}),
        call(
            "search_appointment_availability",
            {
                "clinic_id": str(ids["c1"]),
                "doctor_id": str(ids["adams"]),
                "start_date": "2026-11-16",
                "end_date": "2026-11-16",
                "specialty": "Family Medicine",
                "time_preference": "MORNING",
            },
        ),
        call("search_appointment_availability", {"clinic_id": "not-a-uuid"}),
        call("get_my_appointments", {"start_date": "2026-11-16"}),
    ]

    found = [(f.field, f.value) for f in scoring.observed_filters(scenario, world, calls)]

    assert ("clinic", "c1") in found  # An exact clinic name, and the real id, both become a key.
    assert ("doctor", "adams") in found and ("doctor", "Adams") in found
    assert ("window", ("2026-11-16", "2026-11-16")) in found
    assert ("window", ("2026-11-16", "9999-12-31")) in found  # No end date: open-ended.
    assert ("clinic", "not-a-uuid") in found  # An unknown id stays as it was said.
    assert ("time_preference", "MORNING") in found and ("specialty", "Family Medicine") in found


def test_only_successful_well_formed_preparations_count_as_prepared() -> None:
    world, ids = fake_world()
    calls = [
        call("prepare_appointment_action", {"action": "BOOK", "slot_id": str(ids["s1"])}),
        call("prepare_appointment_action", {"action": "BOOK", "slot_id": str(ids["s3"])}, ok=False),
        call("prepare_appointment_action", {"action": {"x": 1}}),
        call("prepare_appointment_action", {"action": "CANCEL", "appointment_id": str(ids["a1"])}),
        call("search_clinic_directory", {"clinic_name": "x"}),
    ]

    prepared = scoring.prepared_actions(world, calls)

    assert [(p.action, p.slot, p.appointment) for p in prepared] == [
        ("BOOK", "s1", None),
        ("CANCEL", None, "s1"),
    ]


def test_replies_are_matched_to_the_review_and_the_final_message(scenarios) -> None:
    steps = [{"kind": "say"}, {"kind": "ui_confirm"}, {"kind": "say"}]
    messages = [
        {"kind": "message", "step_index": 0, "text": "before"},
        {"kind": "message", "step_index": 2, "text": "after"},
    ]

    chosen, judged = scoring.designated_messages(scenarios["BM-001"], messages, steps)

    assert chosen == {"review": "before", "final": "after"}
    assert judged == [("review", "before"), ("final", "after")]


def test_a_missing_reply_is_none_not_a_guess(scenarios) -> None:
    only_review = [{"kind": "say"}, {"kind": "ui_confirm"}]
    messages = [{"kind": "message", "step_index": 0, "text": "before"}]

    chosen, _ = scoring.designated_messages(scenarios["BM-001"], messages, only_review)
    nothing, _ = scoring.designated_messages(scenarios["BM-001"], [], [])

    assert chosen == {"review": "before", "final": None}
    assert nothing == {"review": None, "final": None}


def test_a_reply_after_the_confirmation_is_never_mistaken_for_the_review(scenarios) -> None:
    steps = [{"kind": "ui_confirm"}, {"kind": "say"}]
    messages = [{"kind": "message", "step_index": 1, "text": "after only"}]

    chosen, _ = scoring.designated_messages(scenarios["BM-001"], messages, steps)

    assert chosen == {"review": None, "final": "after only"}


def test_a_scenario_with_nothing_to_confirm_judges_its_only_reply_as_final(scenarios) -> None:
    messages = [{"kind": "message", "step_index": 1, "text": "only"}]

    chosen, judged = scoring.designated_messages(scenarios["BM-018"], messages, [{"kind": "tool"}])

    assert chosen == {"review": None, "final": "only"} and judged == [("final", "only")]


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [([], 0.95, 0.0), ([5.0], 0.95, 5.0), (list(map(float, range(1, 21))), 0.95, 19.0),
     ([3.0, 1.0, 2.0], 0.5, 2.0), ([10.0, 20.0], 0.95, 20.0)],
)  # fmt: skip
def test_the_percentile_is_nearest_rank_and_deterministic(values, fraction, expected) -> None:
    assert scoring.percentile(values, fraction) == expected


def test_no_instruction_or_example_shares_a_distinctive_name_with_the_benchmark(scenarios) -> None:
    """Neither approach's prompts or examples may be written with the benchmark's names in view,
    not even a surname or a place word."""
    from app.evaluation.hybrid import INTENT_PROMPT
    from app.evaluation.llm import APPROACH_A_SYSTEM_PROMPT, SYSTEM_PROMPT

    generic = set(
        [
            *("clinic", "health", "medical", "center", "family", "care", "medicine"),
            *("associates", "partners", "group", "practice", "walk-in", "community"),
            *("primary", "wellness", "internal", "specialty", "women's", "and"),
            *("creek", "falls", "valley", "ridge"),
        ]
    )
    names = set()
    for scenario in scenarios.values():
        for clinic in scenario.fixture.clinics:
            names |= {w.strip(".,").casefold() for w in clinic.name.split()}
            for doctor in clinic.doctors:
                names |= {w.strip(".,").casefold() for w in doctor.name.split()}
    distinctive = {w for w in names if len(w) > 3 and w not in generic and w != "dr"}
    for prompt in (INTENT_PROMPT, SYSTEM_PROMPT, APPROACH_A_SYSTEM_PROMPT):
        words = {w.strip(".,\"'():").casefold() for w in prompt.split()}
        assert not (words & distinctive), sorted(words & distinctive)
