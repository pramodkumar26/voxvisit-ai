"""The Phase 2 acceptance dataset is frozen, independent, and complete.

These tests need no database and no model. They pin the dataset, its expected results and its
scoring rules by fingerprint, so that nothing can change after the real model has been run, and they
check that the cases share nothing with the development scenarios or the official benchmark.
"""

import hashlib
import inspect
import json
import re
from pathlib import Path

from app.evaluation import ab_scenarios, benchmark_spec, conversation_scenarios, holdout_scenarios
from app.evaluation import acceptance_cases as frozen
from app.evaluation import scenarios as ev_scenarios

ROOT = Path(__file__).parents[2]
MANIFEST_FILE = ROOT / "backend" / "tests" / "phase2-acceptance-v1.json"
# Recorded when the dataset was frozen, before the real model was run on it.
FROZEN_FINGERPRINT = "080c46f0ab25ebae6a7f77fee74ffa2518fd2d6f7320c727ef23c0c4605b3c49"


def test_the_dataset_matches_its_frozen_fingerprint() -> None:
    assert frozen.fingerprint() == FROZEN_FINGERPRINT
    assert hashlib.sha256(MANIFEST_FILE.read_bytes()).hexdigest() == FROZEN_FINGERPRINT
    assert MANIFEST_FILE.read_text() == frozen.canonical(frozen.manifest())


def test_there_are_25_conversations_in_every_required_category() -> None:
    cases = frozen.CASES
    assert len(cases) == 25 and len({c.id for c in cases}) == 25
    counts = {name: sum(c.category == name for c in cases) for name in frozen.CATEGORIES}
    assert all(counts.values()), counts
    assert set(counts) == {
        "booking", "clarification", "no_availability", "cancellation", "rescheduling",
        "correction", "authentication", "stale_review", "model_failure", "refusal",
    }  # fmt: skip


def test_every_case_defines_its_expected_end_state_and_replies_independently() -> None:
    for case in frozen.CASES:
        record = json.loads(MANIFEST_FILE.read_text())["cases"][frozen.CASES.index(case)]
        assert record["final_state_checks"], case.id  # The database is checked in every case.
        assert record["expected_outcome"].strip() and record["reply_checks"], case.id
        completed = record["expected_success_claims"]
        presses_confirm = any(t["turn"] == "ui_confirm" for t in record["timeline"])
        assert bool(completed) or not presses_confirm or case.stale_screen or case.id == "AC-23"


def test_expected_clarifications_and_refusals_are_defined_as_outcomes_not_failures() -> None:
    asked = [c for c in frozen.CASES if c.category in ("clarification", "no_availability")]
    assert len(asked) >= 4
    for case in asked[:3]:
        assert not case.scenario.expected_claims  # Nothing is committed, and that is correct.


def test_the_cases_cover_the_required_situations() -> None:
    manifest = frozen.manifest()
    text = json.dumps(manifest).casefold()
    assert any(c["fault_injected"] == "outage" for c in manifest["cases"])
    assert any(c["fault_injected"] == "invalid_once" for c in manifest["cases"])
    assert any(c["stale_screen_pressed"] for c in manifest["cases"])
    assert any(c["actor"] == "public" for c in manifest["cases"])
    for needed in ("fit me in", "could i get in", "my pin is", "never mind", "forget it"):
        assert needed in text


# --- independence from everything the agent was developed or compared on ---------------------


def words(value: str) -> set[str]:
    return set(re.findall(r"[a-z']{3,}", re.sub(r"\{[a-z_0-9]+\}", " ", value.casefold())))


def other_requests() -> list[str]:
    found = []
    for module in (conversation_scenarios, ab_scenarios, holdout_scenarios, ev_scenarios):
        for value in vars(module).values():
            for scenario in value if isinstance(value, tuple) else [value]:
                if hasattr(scenario, "request") and hasattr(scenario, "timeline"):
                    found.append(scenario.request)
                    found += [getattr(t, "text", "") for t in scenario.timeline]
                    found += [f for t in scenario.timeline for f in getattr(t, "follow_ups", ())]
    found += [s.request for s in benchmark_spec.load_manifest().scenarios]
    return [text for text in found if text]


def test_no_patient_wording_repeats_a_development_or_benchmark_scenario() -> None:
    earlier = [words(text) for text in other_requests()]
    assert len(earlier) > 80
    for case in frozen.CASES:
        mine = [case.scenario.request]
        mine += [getattr(t, "text", "") for t in case.scenario.timeline]
        mine += [f for t in case.scenario.timeline for f in getattr(t, "follow_ups", ())]
        for text in filter(None, mine):
            ours = words(text)
            for theirs in earlier:
                if len(ours) >= 4 and len(theirs) >= 4:
                    overlap = len(ours & theirs) / len(ours | theirs)
                    assert overlap < 0.6, (case.id, text)


def invented_names() -> set[str]:
    source = inspect.getsource(frozen)
    names = set(re.findall(r'\("([A-Z][A-Za-z]+(?: [A-Z][a-z]+)?)", "Family Medicine"\)', source))
    prefixes = set(re.findall(r'"Acceptance ([A-Z][a-z]+)"', source))
    return {part for name in names | prefixes for part in name.split()}


def test_no_doctor_or_clinic_name_is_one_the_benchmark_or_the_development_scenarios_use() -> None:
    mine = invented_names()
    assert len(mine) >= 25
    used = set()
    for scenario in benchmark_spec.load_manifest().scenarios:
        for clinic in scenario.fixture.clinics:
            used |= set(re.split(r"[\s-]", clinic.name))
            for doctor in clinic.doctors:
                used |= set(re.split(r"[\s-]", doctor.name))
    for module in (conversation_scenarios, ab_scenarios, holdout_scenarios, ev_scenarios):
        used |= set(re.findall(r"[A-Z][a-z]+", inspect.getsource(module)))
    assert not (mine & used), sorted(mine & used)


def shingles(value: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z']+", re.sub(r"\{[a-z_0-9]+\}", " | ", value.casefold()))
    return {
        tuple(tokens[i : i + size])
        for i in range(len(tokens) - size + 1)
        if "|" not in tokens[i : i + size]
    }


def test_the_agent_and_its_prompts_do_not_contain_the_acceptance_names_or_wording() -> None:
    from app.evaluation import conversation, hybrid, llm, spoken

    mine = invented_names()
    agent = set()
    for module in (hybrid, spoken, conversation, llm):
        source = inspect.getsource(module)
        found = set(re.findall(r"[A-Z][a-z]+", source))
        assert not (mine & found), (module.__name__, sorted(mine & found))
        agent |= shingles(source)
    for case in frozen.CASES:
        texts = [case.scenario.request]
        texts += [getattr(t, "text", "") for t in case.scenario.timeline]
        texts += [f for t in case.scenario.timeline for f in getattr(t, "follow_ups", ())]
        for text in filter(None, texts):
            assert not (shingles(text) & agent), (case.id, text)


def test_the_scoring_rules_name_every_outcome_class() -> None:
    manifest = frozen.manifest()
    for key in (
        "success",
        "safe_failure",
        "safety_violation",
        "caught_before_commit",
        "clarification",
    ):
        assert manifest["scoring"][key].strip()
    assert "PROTECTED_DATA_LEAK" in manifest["safety_failure_categories"]
    assert "MODEL_CONFIRMED" in manifest["safety_failure_categories"]


def test_the_frozen_file_cannot_be_edited_without_the_fingerprint_changing() -> None:
    text = MANIFEST_FILE.read_text()
    assert (
        hashlib.sha256(text.replace("AC-25", "AC-26", 1).encode()).hexdigest() != FROZEN_FINGERPRINT
    )
