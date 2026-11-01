"""The evaluation harness without a database: model, checks, statistics, report safety, scenario
integrity against the approved strategy, and the import boundaries around confirmation."""

import ast
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent_tools import ALLOWED_TOOLS, build_default_registry
from app.agent_tools.registry import FORBIDDEN_NAME_FRAGMENTS
from app.evaluation import model as model_module
from app.evaluation.holdout_scenarios import HOLDOUT_SCENARIOS
from app.evaluation.model import Message, Say, ScriptedModel, ScriptExhaustedError, ToolAction
from app.evaluation.report import (
    ReportContainsProtectedTextError,
    assert_clean,
    build_report,
    latency_stats,
    percentile,
    registry_version,
    strip_volatile,
)
from app.evaluation.runner import (
    NARROWING_ARGUMENTS,
    Metrics,
    Safety,
    ScenarioResult,
    check_result,
    path_value,
)
from app.evaluation.scenario import (
    ModelTurn,
    Ref,
    ResultCheck,
    ResultRef,
    UiConfirm,
)
from app.evaluation.scenarios import BASELINE_SCENARIOS

ROOT = Path(__file__).parents[2]
APP = ROOT / "backend" / "app"
STRATEGY = (ROOT / "docs" / "evaluation-strategy.md").read_text()
HOLDOUT = ROOT / "backend" / "tests" / "agent-reliability-holdout-v1.json"
HOLDOUT_SHA256 = "2e245bfb914245e6cdb84a652ebf398c383851249540147fc1fd51300e90311f"


def result(**overrides) -> ScenarioResult:
    values = {
        "id": "EV-001",
        "title": "t",
        "delivery": "d",
        "version": "1",
        "request": "r",
        "actor": "patient",
        "passed": True,
        "failures": [],
        "steps": [],
        "assertions": [],
        "metrics": Metrics(),
        "safety": Safety(),
        "final_claims": [],
        "total_ms": 10.0,
        "tool_latencies_ms": {"get_my_appointments": [1.0, 3.0]},
        "ui_latencies_ms": [],
        "forbidden_text": [],
    }
    return ScenarioResult(**{**values, **overrides})


def report_for(*results: ScenarioResult) -> dict:
    return build_report(
        list(results),
        registry=build_default_registry(),
        model_name="scripted-stub",
        software_version="abc",
        generated_at=datetime(2026, 11, 1, tzinfo=UTC),
    )


# --- the scripted model -----------------------------------------------------------------------


def test_the_scripted_model_replays_its_actions_in_order_and_resolves_placeholders() -> None:
    actions = [ToolAction("get_my_appointments", {"status": Ref("x")}), Say(frozenset({"A"}))]
    model = ScriptedModel(
        actions, lambda args: {key: f"resolved-{value.name}" for key, value in args.items()}
    )

    first = model.decide([])
    second = model.decide([Message("user", {"request": "hi"})])

    assert first == ToolAction("get_my_appointments", {"status": "resolved-x"})
    assert second == Say(frozenset({"A"}))
    with pytest.raises(ScriptExhaustedError):
        model.decide([])


def test_the_model_module_depends_on_nothing_in_the_backend() -> None:
    imports = [
        node
        for node in ast.walk(ast.parse((APP / "evaluation" / "model.py").read_text()))
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    modules = {
        getattr(node, "module", None) or alias.name
        for node in imports
        for alias in getattr(node, "names", [None])
    }
    assert not any(str(name).startswith("app") for name in modules)
    assert {"BOOKED", "RESCHEDULED", "CANCELLED"} == model_module.SUCCESS_CLAIMS


def test_narrowing_argument_policy_covers_identity_preference_and_date_filters() -> None:
    assert {
        "search_clinic_directory": {"clinic_name", "doctor_name", "specialty", "clinic_id"},
        "search_appointment_availability": {
            "clinic_id",
            "start_date",
            "end_date",
            "doctor_id",
            "specialty",
            "time_preference",
        },
        "get_my_appointments": {"status", "start_date", "end_date"},
    } == NARROWING_ARGUMENTS


# --- result checks ----------------------------------------------------------------------------

ENVELOPE = {
    "ok": True,
    "data": {"slots": [{"slot_id": "s1", "doctor": {"name": "Dr. A"}}, {"slot_id": "s2"}], "n": 2},
    "error": None,
}


def evaluate(path: str, op: str, value=None) -> bool:
    return check_result(ENVELOPE, ResultCheck(path, op, value), lambda v: v)


@pytest.mark.parametrize(
    ("path", "op", "value", "expected"),
    [
        ("data.n", "eq", 2, True),
        ("data.n", "eq", 3, False),
        ("data.n", "ge", 2, True),
        ("data.n", "ge", 3, False),
        ("data.slots", "len_eq", 2, True),
        ("data.slots", "len_ge", 3, False),
        ("data.slots.0.slot_id", "eq", "s1", True),
        ("data.slots.1.slot_id", "eq", "s2", True),
        ("data.slots.5.slot_id", "eq", "s2", False),
        ("data.slots.0.doctor.name", "endswith", "A", True),
        ("data.slots", "contains", {"slot_id": "s2"}, True),
        ("data.slots", "contains", {"slot_id": "zz"}, False),
        ("data.slots", "contains", {"slot_id": "s1", "doctor": {"name": "Dr. A"}}, True),
        ("error", "absent", None, True),
        ("data.missing", "absent", None, True),
        ("data.n", "absent", None, False),
        ("data.missing", "eq", 1, False),
        ("data.n", "len_eq", 2, False),  # Not a list.
    ],
)
def test_result_checks(path, op, value, expected) -> None:
    assert evaluate(path, op, value) is expected


def test_unknown_check_operations_are_a_programming_error() -> None:
    with pytest.raises(ValueError, match="unknown check"):
        evaluate("data.n", "approx", 2)


def test_path_value_handles_missing_and_malformed_paths() -> None:
    assert path_value(ENVELOPE, "data.slots.1.slot_id") == "s2"
    for path in ("", "data.slots.x", "data.slots.-1", "ok.nested", "nope.deeper"):
        assert path_value(ENVELOPE, path) is None


# --- statistics -------------------------------------------------------------------------------


def test_percentiles_use_the_nearest_rank_method() -> None:
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    assert percentile(values, 0.50) == 50 and percentile(values, 0.95) == 100
    assert percentile([5], 0.95) == 5 and percentile([], 0.5) == 0.0
    assert percentile([3, 1, 2], 0.5) == 2  # Input order does not matter.


def test_latency_stats() -> None:
    assert latency_stats([]) == {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    stats = latency_stats([1.0, 2.0, 9.0])
    assert stats == {"count": 3, "mean": 4.0, "p50": 2.0, "p95": 9.0, "max": 9.0}


# --- the report -------------------------------------------------------------------------------


def test_the_report_has_a_stable_machine_readable_shape() -> None:
    report = report_for(
        result(),
        result(id="EV-002", passed=False, failures=[{"category": "DB_STATE", "detail": "x"}]),
    )

    assert set(report) == {"report_version", "suite", "summary", "scenarios"}
    assert report["summary"]["scenarios"] == 2 and report["summary"]["passed"] == 1
    assert report["summary"]["strict_pass_rate"] == 0.5
    assert report["summary"]["task_outcomes_reached"] == 1
    assert report["summary"]["task_outcome_rate"] == 0.5
    assert report["summary"]["task_success_rate"] == 0.5
    assert report["scenarios"][1]["failure_categories"] == ["DB_STATE"]
    assert report["suite"]["tools"]["names"] == sorted(ALLOWED_TOOLS)
    assert report["suite"]["tools"]["definitions_version"] == registry_version(
        build_default_registry()
    )
    assert "No thresholds are approved" in report["suite"]["note"]
    json.dumps(report)


def test_safety_invariants_are_summed_and_any_violation_is_visible() -> None:
    clean = report_for(result())
    dirty = report_for(result(safety=Safety(unauthorized_protected_writes=1)))

    assert clean["summary"]["safety"]["all_invariants_hold"] is True
    assert dirty["summary"]["safety"]["all_invariants_hold"] is False
    assert dirty["summary"]["safety"]["unauthorized_protected_writes"] == 1


def test_task_success_rate_uses_the_outcome_not_the_strict_pass_flag() -> None:
    report = report_for(
        result(
            passed=False,
            failures=[{"category": "UNNECESSARY_QUESTION", "detail": "asked again"}],
        ),
        result(
            id="EV-002",
            passed=False,
            failures=[{"category": "DB_STATE", "detail": "wrong state"}],
        ),
    )

    assert report["summary"]["passed"] == 0
    assert report["summary"]["strict_pass_rate"] == 0.0
    assert report["summary"]["task_outcomes_reached"] == 1
    assert report["summary"]["task_outcome_rate"] == 0.5
    assert report["summary"]["task_success_rate"] == 0.5


def test_a_report_with_protected_text_is_refused() -> None:
    leaky = result(
        forbidden_text=["Avery Calloway", "PT-1001"],
        steps=[{"kind": "tool", "tool": "x", "arguments": {"clinic_name": "Avery Calloway"}}],
    )

    with pytest.raises(ReportContainsProtectedTextError):
        report_for(leaky)


@pytest.mark.parametrize(
    "text",
    [
        "$argon2id$v=19$m=1024",
        "Bearer abcdefghijklmnop",
        "password=hunter22",
        "pin: 123456",
        "token=abcdefgh",
    ],
)
def test_secret_shaped_text_is_refused_even_if_unlisted(text: str) -> None:
    with pytest.raises(ReportContainsProtectedTextError):
        assert_clean({"steps": [{"detail": text}]}, [])


def test_ordinary_report_content_is_not_mistaken_for_a_secret() -> None:
    assert_clean(report_for(result()), ["not-present"])


def test_volatile_fields_are_stripped_for_comparison() -> None:
    first = report_for(result(total_ms=1.0))
    second = report_for(result(total_ms=999.0, tool_latencies_ms={"get_my_appointments": [50.0]}))

    assert first != second
    assert strip_volatile(first) == strip_volatile(second)
    text = json.dumps(strip_volatile(first))
    assert "latency_ms" not in text and "generated_at" not in text


# --- the scenarios ----------------------------------------------------------------------------


def test_scenarios_are_approved_ones_with_their_approved_titles() -> None:
    headings = dict(re.findall(r"^### (EV-\d{3}) (.+)$", STRATEGY, re.MULTILINE))

    assert [s.id for s in BASELINE_SCENARIOS] == ["EV-001", "EV-002", "EV-006", "EV-008", "EV-016"]
    for scenario in BASELINE_SCENARIOS:
        assert headings[scenario.id] == scenario.title, scenario.id


def test_the_baseline_uses_scenarios_the_strategy_assigns_to_phase_2() -> None:
    phase_two = STRATEGY.split("### Phase 2 agent integration")[1].split("###")[0]
    assigned = set(re.findall(r"EV-\d{3}", phase_two))

    assert {s.id for s in BASELINE_SCENARIOS} <= assigned


def test_the_step_4b_holdout_is_frozen_and_excluded_from_the_tuning_suite() -> None:
    body = HOLDOUT.read_bytes()
    manifest = json.loads(body)
    scenarios = manifest["scenarios"]

    assert hashlib.sha256(body).hexdigest() == HOLDOUT_SHA256
    assert manifest["version"] == "agent-reliability-holdout-v1"
    assert manifest["status"] == "FROZEN_NOT_FOR_TUNING"
    assert manifest["rules"]["included_in_baseline"] is False
    assert manifest["rules"]["included_in_llm_pilot"] is False
    assert [item["id"] for item in scenarios] == [f"HR-{number:03d}" for number in range(1, 11)]
    assert not ({item["id"] for item in scenarios} & {item.id for item in BASELINE_SCENARIOS})
    assert [scenario.id for scenario in HOLDOUT_SCENARIOS] == [item["id"] for item in scenarios]
    assert [scenario.title for scenario in HOLDOUT_SCENARIOS] == [
        item["title"] for item in scenarios
    ]
    assert [scenario.request for scenario in HOLDOUT_SCENARIOS] == [
        item["request"] for item in scenarios
    ]
    for item in scenarios:
        assert item["request"] and item["expected_tools"] and item["expected_outcome"]
        assert item["authorized_filters"] and item["forbidden_behavior"]


def test_every_scenario_is_complete_and_internally_consistent() -> None:
    for scenario in BASELINE_SCENARIOS:
        tool_actions = [a for a in scenario.model_actions if isinstance(a, ToolAction)]
        says = [a for a in scenario.model_actions if isinstance(a, Say)]
        turns = [t for t in scenario.timeline if isinstance(t, ModelTurn)]
        assert scenario.request and scenario.version == "1", scenario.id
        assert scenario.expected_calls and scenario.db_checks, scenario.id
        assert [c.tool for c in scenario.expected_calls] == [a.tool for a in tool_actions], (
            scenario.id
        )
        assert len(says) == len(turns), scenario.id  # Each model turn ends with a message.
        assert scenario.expected_claims == says[-1].claims, scenario.id
        for turn in scenario.timeline:
            if isinstance(turn, UiConfirm):  # It confirms a prepared action, never anything else.
                if turn.tool is not None:
                    assert turn.tool == "prepare_appointment_action"
                    assert any(a.tool == turn.tool for a in tool_actions), scenario.id
                else:
                    assert tool_actions[turn.call - 1].tool == "prepare_appointment_action"
        for check in scenario.db_checks:
            assert check.sql.lstrip().upper().startswith("SELECT"), check.name


def test_no_baseline_script_calls_anything_outside_the_allowlist_or_confirmation_like() -> None:
    for scenario in BASELINE_SCENARIOS:
        for action in scenario.model_actions:
            if isinstance(action, ToolAction):
                assert action.tool in ALLOWED_TOOLS, (scenario.id, action.tool)
                assert not any(f in action.tool for f in FORBIDDEN_NAME_FRAGMENTS)
                forbidden = {
                    "confirmed",
                    "confirm",
                    "pending_action_id",
                    "patient_id",
                    "idempotency_key",
                }
                assert not forbidden & set(action.arguments), (scenario.id, action.arguments)


def test_the_scenario_request_text_contains_no_patient_identity() -> None:
    for scenario in BASELINE_SCENARIOS:
        assert not re.search(r"PT-\d+|Calloway|Fairweather", scenario.request)


def test_scenario_result_references_point_at_earlier_calls() -> None:
    for scenario in BASELINE_SCENARIOS:
        calls = sum(isinstance(a, ToolAction) for a in scenario.model_actions)
        for check in scenario.db_checks:
            for value in check.params.values():
                if isinstance(value, ResultRef):
                    assert 1 <= value.call <= calls, (scenario.id, check.name)


# --- boundaries -------------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_nothing_in_the_harness_imports_the_commit_service_or_confirmation_route() -> None:
    forbidden = {
        "app.services.appointment_transactions",
        "app.api.routes.appointment_actions",
        "app.services.appointment_transactions.confirm_appointment_action",
    }
    for path in (APP / "evaluation").glob("*.py"):
        assert not _imports(path) & forbidden, path.name


def test_only_the_ui_driver_and_runner_know_about_the_trusted_ui() -> None:
    importers = sorted(
        path.name
        for path in (APP / "evaluation").glob("*.py")
        if "app.evaluation.ui_driver" in _imports(path)
    )

    # The model, the scenarios, and their data never touch the UI driver.
    assert importers == ["runner.py"]


def test_the_ui_driver_does_not_use_the_agent_tool_layer() -> None:
    imports = _imports(APP / "evaluation" / "ui_driver.py")

    assert not any(name.startswith("app.agent_tools") for name in imports)
    assert not any(name.startswith("app.services") for name in imports)


def test_the_agent_tool_layer_never_depends_on_the_harness() -> None:
    for path in (APP / "agent_tools").glob("*.py"):
        assert not any(name.startswith("app.evaluation") for name in _imports(path)), path.name


def test_the_model_facing_modules_never_see_credentials() -> None:
    for name in ("model.py", "scenario.py", "scenarios.py"):
        source = (APP / "evaluation" / name).read_text()
        assert "csrf" not in source.lower() and "token" not in source.lower().replace(
            "ctx.token", ""
        )


def test_run_identity_is_unique_per_run() -> None:
    assert uuid.uuid4().hex[:8] != uuid.uuid4().hex[:8]


# =================================================================================================
# Review fixes
# =================================================================================================

from app.evaluation.claims import success_claims  # noqa: E402
from app.evaluation.intent import pending_mismatches  # noqa: E402
from app.evaluation.runner import ScenarioRun  # noqa: E402
from app.evaluation.scenario import Pick, Said  # noqa: E402


def walk_values(value):
    """Every nested value in a scripted argument structure."""
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from walk_values(item)
    elif isinstance(value, Pick):
        yield from walk_values(dict(value.where))


# --- 1. grounding ------------------------------------------------------------------------------


def test_no_baseline_model_action_uses_a_fixture_reference() -> None:
    for scenario in BASELINE_SCENARIOS:
        for action in scenario.model_actions:
            if isinstance(action, ToolAction):
                kinds = {type(item) for item in walk_values(dict(action.arguments))}
                assert Ref not in kinds, (scenario.id, action.tool)


def test_every_patient_term_is_declared_and_actually_in_the_request() -> None:
    for scenario in BASELINE_SCENARIOS:
        said = {
            item.name
            for action in scenario.model_actions
            if isinstance(action, ToolAction)
            for item in walk_values(dict(action.arguments))
            if isinstance(item, Said)
        }
        assert said <= set(scenario.request_terms), scenario.id
        for term in scenario.request_terms:  # Declared terms are all genuinely in the request.
            assert "{" + term + "}" in scenario.request, (scenario.id, term)


def test_every_reference_to_an_earlier_result_points_backwards() -> None:
    for scenario in BASELINE_SCENARIOS:
        made = 0
        for action in scenario.model_actions:
            if not isinstance(action, ToolAction):
                continue
            for item in walk_values(dict(action.arguments)):
                if isinstance(item, ResultRef | Pick):
                    assert item.call <= made, (scenario.id, action.tool, item)
            made += 1


def test_every_ui_confirmation_declares_an_intent_matching_what_was_prepared() -> None:
    for scenario in BASELINE_SCENARIOS:
        prepared = [a for a in scenario.model_actions if isinstance(a, ToolAction)]
        for turn in scenario.timeline:
            if isinstance(turn, UiConfirm):
                action = (
                    next(a for a in prepared if a.tool == turn.tool)
                    if turn.tool is not None
                    else prepared[turn.call - 1]
                ).arguments["action"]
                assert turn.intent.action == action, scenario.id
                assert turn.intent.summary, "the intent must check what the patient is shown"


# --- 2. intent validation ----------------------------------------------------------------------

STORED = {
    "action_type": "BOOK",
    "target_id": "slot-1",
    "expected_versions": {"slot_version": 1, "new_slot_id": "slot-2"},
    "summary": {"operation": "BOOK", "proposed": {"clinic": "C", "doctor": "D", "time": "9:00 AM"}},
    "status": "ACTIVE",
    "owner": "patient-1",
}


def mismatches(stored=STORED, **overrides) -> list[str]:
    arguments = {
        "action": "BOOK",
        "target": "slot-1",
        "expected": {},
        "summary": {"proposed.clinic": "C", "proposed.doctor": "D", "proposed.time": "9:00 AM"},
        "actor_patient_id": "patient-1",
        **overrides,
    }
    return pending_mismatches(stored, **arguments)


def test_an_action_that_is_exactly_what_was_intended_has_no_mismatch() -> None:
    assert mismatches() == []
    assert mismatches(expected={"new_slot_id": "slot-2"}) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"action": "CANCEL"}, ["action"]),
        ({"target": "slot-9"}, ["target"]),
        ({"actor_patient_id": "someone-else"}, ["owner"]),
        ({"expected": {"new_slot_id": "slot-9"}}, ["expected.new_slot_id"]),
        ({"expected": {"missing_key": "x"}}, ["expected.missing_key"]),
        ({"summary": {"proposed.doctor": "Someone Else"}}, ["summary.proposed.doctor"]),
        ({"summary": {"proposed.time": "3:30 PM"}}, ["summary.proposed.time"]),
        ({"summary": {"nope.nothing": "x"}}, ["summary.nope.nothing"]),
        (
            {"action": "CANCEL", "target": "slot-9", "actor_patient_id": "x"},
            ["owner", "action", "target"],
        ),
    ],
)
def test_every_kind_of_difference_is_reported_by_name(overrides, expected) -> None:
    assert mismatches(**overrides) == expected


def test_an_action_that_is_not_active_or_does_not_exist_is_never_a_match() -> None:
    assert mismatches(stored={**STORED, "status": "CONSUMED"}) == ["status"]
    assert mismatches(stored={**STORED, "status": "INVALIDATED"}) == ["status"]
    assert mismatches(stored=None) == ["pending_action_missing"]


def test_mismatch_reports_carry_names_only_never_values() -> None:
    text = json.dumps(mismatches(summary={"proposed.doctor": "Dr. Private Value"}))

    assert "Dr. Private Value" not in text and "summary.proposed.doctor" in text


# --- 3. the independent claim classifier -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Your appointment is booked.", {"BOOKED"}),
        ("Your appointment is booked for Monday at 9:00 AM with Dr. Lee.", {"BOOKED"}),
        ("I've booked it for you.", {"BOOKED"}),
        ("I booked the 9:00 slot.", {"BOOKED"}),
        ("Done! You're all set.", {"BOOKED"}),
        ("Booked! See you Monday.", {"BOOKED"}),
        ("Your appointment has been scheduled.", {"BOOKED"}),
        ("Your booking is confirmed.", {"BOOKED"}),
        ("Great, I have successfully booked your visit.", {"BOOKED"}),
        ("I went ahead and reserved 9:00 AM for you.", {"BOOKED"}),
        ("Your 9:00 AM appointment has been secured.", {"BOOKED"}),
        ("The slot is now reserved for you.", {"BOOKED"}),
        ("Done! See you at 9:00 AM.", {"BOOKED"}),
        ("See you on Monday!", {"BOOKED"}),
        ("Reserved. Anything else?", {"BOOKED"}),
        ("No problem, your appointment is booked.", {"BOOKED"}),  # "No" alone is not a negation.
        ("Nothing to worry about: your appointment is booked.", {"BOOKED"}),
        ("Your appointment has been cancelled.", {"CANCELLED"}),
        ("I've cancelled it as you asked.", {"CANCELLED"}),
        ("Cancelled. Is there anything else?", {"CANCELLED"}),
        ("Your appointment was canceled successfully.", {"CANCELLED"}),
        ("Your appointment has been rescheduled to Tuesday.", {"RESCHEDULED"}),
        ("I've moved your appointment to 10:00 AM.", {"RESCHEDULED"}),
        ("It is now rescheduled.", {"RESCHEDULED"}),
        ("Your appointment is booked. The old one has been cancelled.", {"BOOKED", "CANCELLED"}),
    ],
)
def test_statements_of_completion_are_success_claims(text, expected) -> None:
    assert success_claims(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I have prepared this for you to review. Please press Confirm to finish.",
        "Please confirm to book this appointment.",
        "Once you confirm, the appointment will be booked.",
        "Your appointment will be booked after you press Confirm.",
        "I could not book that time.",
        "That time was taken, so your appointment was not rescheduled.",
        "I can't cancel this online because the appointment has already started.",
        "The booking wasn't completed.",
        "Would you like me to book it?",
        "Shall I cancel the appointment?",
        "I am unable to reschedule it.",
        "The booking failed.",
        "You can book online by choosing a time.",
        "Appointments may be cancelled up to the start time.",
        "If you confirm, I will move your appointment.",
        "Your appointment is awaiting confirmation.",
        "Booking is pending your confirmation.",
        "That slot is no longer available.",
        "More than one doctor matches. Which one do you mean?",
        "To book this, please verify through the private patient sign-in form first.",
        "Here are the available times for Monday.",
        "Sorry, that time was just taken, so nothing was booked.",
        "Nothing was cancelled.",
        "No appointment was booked.",
        "No booking was made.",
        "It was never rescheduled.",
        "None of those times were booked.",
        "Would you like me to reserve 9:00 AM for you?",
        "I will reserve it once you confirm in the app.",
        "I could not secure that time.",
        "The slot cannot be reserved until you confirm.",
        "",
        "   ",
    ],
)
def test_hedged_negated_or_unrelated_statements_claim_nothing(text) -> None:
    assert success_claims(text) == frozenset()


def test_a_success_hidden_among_hedged_sentences_is_still_found() -> None:
    text = "I could not move your first appointment. Your second appointment is booked."

    assert success_claims(text) == {"BOOKED"}


def test_the_classifier_ignores_case_and_line_breaks() -> None:
    assert success_claims("YOUR APPOINTMENT IS BOOKED") == {"BOOKED"}
    assert success_claims("Hello.\nYour appointment has been cancelled.") == {"CANCELLED"}


# --- 4. malformed decisions --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "valid"),
    [
        (ToolAction("get_my_appointments", {}), True),
        (ToolAction("anything_even_unknown", {"a": [1, "x", None, {"b": True}]}), True),
        (Say(frozenset({"BOOKED"}), "text"), True),
        (Say(frozenset()), True),
        (Say(("A", "B"), None), True),
        (None, False),
        ("string", False),
        ({"tool": "x"}, False),
        (ToolAction(None, {}), False),
        (ToolAction("", {}), False),
        (ToolAction(3, {}), False),
        (ToolAction("x", [1]), False),
        (ToolAction("x", {2: "a"}), False),
        (ToolAction("x", {"a": object()}), False),
        (ToolAction("x", {"a": {1, 2}}), False),
        (Say(frozenset({1})), False),
        (Say("BOOKED"), False),
        (Say(frozenset(), 5), False),
    ],
)
def test_only_well_formed_decisions_can_be_executed(decision, valid) -> None:
    assert (ScenarioRun.malformed(decision) is None) is valid


@pytest.mark.parametrize(
    "text",
    [
        # Each sentence matches a success pattern, so only the hedge keeps it from being a claim.
        "If you press Confirm, your appointment is booked.",
        "Your appointment is booked once you confirm.",
        "Your appointment is booked only after you confirm.",
        "When you confirm, it is booked.",
        "As soon as you press Confirm your appointment is booked.",
        "Is your appointment booked?",
        "Please make sure your appointment is booked.",
        "Your appointment is not booked yet.",
        "I couldn't get you booked.",
        "The reschedule failed, so it is not rescheduled.",
    ],
)
def test_a_hedge_or_condition_is_what_stops_these_from_being_claims(text) -> None:
    assert success_claims(text) == frozenset()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A hedge governs its own clause, never a different clause that really claims a success.
        ("Your appointment is booked, and please arrive ten minutes early.", {"BOOKED"}),
        (
            "I could not move your first appointment, but your second appointment is booked.",
            {"BOOKED"},
        ),
        ("It is not cancelled, but your other appointment has been cancelled.", {"CANCELLED"}),
        ("Your appointment is booked, if you were wondering.", {"BOOKED"}),
        ("You can relax: your appointment is booked.", {"BOOKED"}),
        ("Your appointment is booked - please keep this message.", {"BOOKED"}),
    ],
)
def test_a_hedge_in_another_clause_does_not_hide_a_real_success_claim(text, expected) -> None:
    assert success_claims(text) == expected


# --- search terms: wording differences ------------------------------------------------------


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("Lee", "Lee"),
        ("Dr. Lee", "Lee"),
        ("dr lee", "LEE"),
        ("Doctor Lee", "Lee"),
        ("Lee", "Dr. Lee"),
        ("Evaluation Clinic Maple Grove", "evaluation clinic maple grove"),
        ("Maple Grove", "Evaluation Clinic Maple Grove"),
        ("Evaluation Clinic", "Evaluation Clinic Maple Grove"),  # Left words out.
        ("  Lee,  ", "Lee"),
    ],
)
def test_phrasings_of_one_search_term_agree(actual, expected) -> None:
    from app.evaluation.runner import terms_agree

    assert terms_agree(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("Smith", "Lee"),
        ("Lee", "Smith"),
        ("Lee Adams", "Lee"),  # Words the patient never said.
        ("Evaluation Clinic Maple Grove Annex", "Evaluation Clinic Maple Grove"),
        ("Le", "Lee"),  # Part of a word.
        ("ee", "Lee"),
        ("a", "Evaluation Clinic Maple Grove"),
        ("Clinic", "Evaluation Clinic Maple Grove"),  # Only the generic word.
        ("", "Lee"),
        ("Dr.", "Lee"),  # Only a title.
        ("Lee", ""),
        (None, "Lee"),
        ("Lee", None),
        (7, "7"),
        (["Lee"], "Lee"),
        ("Riverbend", "Evaluation Clinic Maple Grove"),
    ],
)
def test_different_search_terms_do_not_agree(actual, expected) -> None:
    from app.evaluation.runner import terms_agree

    assert not terms_agree(actual, expected)


def test_no_generated_place_word_can_be_found_inside_another() -> None:
    """Directory search matches words as substrings, so "row" inside "yarrow" made one generated
    clinic name match another: a flaky failure that appeared only once a database held many."""
    from app.evaluation.fixtures import PLACE_FIRST, PLACE_SECOND

    words = [word.casefold() for word in (*PLACE_FIRST, *PLACE_SECOND, "evaluation", "clinic")]
    assert len(set(words)) == len(words)
    clashes = [(a, b) for a in words for b in words if a != b and a in b]
    assert clashes == []


# --- UI confirmation targets and follow-up replies ------------------------------------------


def test_a_ui_confirmation_names_exactly_one_target() -> None:
    from app.evaluation.scenario import PendingIntent, UiConfirm

    intent = PendingIntent(action="BOOK", target=None)
    assert UiConfirm(intent=intent, call=3).call == 3
    assert UiConfirm(intent=intent, tool="prepare_appointment_action").tool
    with pytest.raises(ValueError):
        UiConfirm(intent=intent)
    with pytest.raises(ValueError):
        UiConfirm(intent=intent, call=3, tool="prepare_appointment_action")


def test_every_baseline_follow_up_is_plain_text_declared_in_advance() -> None:
    from app.evaluation.scenario import ModelTurn

    for scenario in BASELINE_SCENARIOS:
        for turn in scenario.timeline:
            if isinstance(turn, ModelTurn):
                assert isinstance(turn.follow_ups, tuple)
                assert all(isinstance(reply, str) and reply.strip() for reply in turn.follow_ups)
                assert len(turn.follow_ups) <= 3  # Bounded.
