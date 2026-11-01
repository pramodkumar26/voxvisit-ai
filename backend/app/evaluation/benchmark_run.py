"""Runs the frozen benchmark: one fresh database per run, the same path for either approach.

For each scenario and seed this module creates a disposable database, builds the manifest's world
in it, checks that world against the manifest, runs the approach through the existing evaluation
runner (the real tools, the real authenticated session, the trusted-UI confirmation), reads the
database back, and scores the result with the frozen protocol.

The approach is only a `model_factory`. Nothing here depends on which approach it builds, and
nothing about the world, the request, the patient, the confirmation policy, or the scoring is
different between them.

An invalid world (the database does not match the manifest) stops the run with
`BenchmarkEnvironmentError`. It is never scored, because a result from a world the manifest does
not describe says nothing about the approach.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.evaluation import benchmark_builder as builder
from app.evaluation import benchmark_protocol as protocol
from app.evaluation import benchmark_scoring as scoring
from app.evaluation import benchmark_spec as spec
from app.evaluation.environment import EvalEnvironment, create_disposable_environment
from app.evaluation.fixtures import book_slot_for, patient_identity
from app.evaluation.intent import OneOf
from app.evaluation.model import Model
from app.evaluation.runner import run_scenario
from app.evaluation.scenario import (
    PUBLIC,
    ModelTurn,
    PendingIntent,
    Scenario,
    UiConfirm,
    World,
    WorldContext,
)
from app.evaluation.scenario import Fixtures as RunFixtures

PREPARE = "prepare_appointment_action"
SEARCH = "search_clinic_directory"
AVAILABILITY = "search_appointment_availability"
MINE = "get_my_appointments"
DELIVERY = "Phase 2 architecture benchmark v1"

ModelFactory = Callable[[spec.ScenarioSpec, int, builder.BuiltWorld], Model]


# --- the scenario, as the existing runner understands it ---------------------------------------


def _intent(scenario: spec.ScenarioSpec, world: builder.BuiltWorld) -> PendingIntent | None:
    """What the simulated patient checks before pressing Confirm: the stored pending action must
    be the one the manifest says the patient wanted (any acceptable new slot, for a reschedule)."""
    prepared = scenario.expected.prepared
    if prepared is None:
        return None
    if prepared.action == "BOOK":
        return PendingIntent(action="BOOK", target=world.slot_ids[prepared.slot or ""])
    appointment = world.appointment_ids[prepared.appointment or ""]
    if prepared.action == "CANCEL":
        return PendingIntent(action="CANCEL", target=appointment)
    acceptable = prepared.acceptable_new_slots or [prepared.new_slot]
    return PendingIntent(
        action="RESCHEDULE",
        target=appointment,
        expected={
            "old_slot_id": world.slot_ids[prepared.appointment or ""],
            "new_slot_id": OneOf(*(world.slot_ids[key or ""] for key in acceptable)),
        },
    )


def _authorized_filters(scenario: spec.ScenarioSpec) -> dict[str, frozenset[str]]:
    intent = scenario.intent
    named_doctor = bool(intent.doctor) or "doctor" in scenario.stated
    directory = {"clinic_name", "clinic_id"}
    availability = {"clinic_id", "start_date", "end_date"}
    if named_doctor:
        directory.add("doctor_name")
        availability.add("doctor_id")
    if intent.specialty:
        directory.add("specialty")
        availability.add("specialty")
    if intent.time_preference != "ANYTIME":
        availability.add("time_preference")
    return {
        SEARCH: frozenset(directory),
        AVAILABILITY: frozenset(availability),
        MINE: frozenset({"status", "start_date", "end_date"}),
    }


def scenario_for_run(
    scenario: spec.ScenarioSpec, world: builder.BuiltWorld, fixtures: RunFixtures
) -> Scenario:
    """The manifest scenario as a runner `Scenario`: the rendered request, the prebuilt world, and
    the timeline of turns. No expected tool calls and no scripted actions: the approach decides."""
    intent = _intent(scenario, world)
    timeline: list[Any] = [ModelTurn()]
    for event in scenario.world:

        def other_patient_books(context: WorldContext, key: str = event.slot) -> None:
            book_slot_for(context.engine, world.other_code, world.slot_ids[key])

        timeline.append(World("another patient takes the slot", other_patient_books))
    if intent is not None:
        timeline.append(
            UiConfirm(
                tool=PREPARE,
                intent=intent,
                expect="COMMITTED" if scenario.expected.ui == "COMMITTED" else "SLOT_UNAVAILABLE",
            )
        )
        timeline.append(ModelTurn())
    claim = protocol.OUTCOME_CLAIM.get(scenario.expected.outcome)
    return Scenario(
        id=scenario.id,
        title=scenario.title,
        delivery=DELIVERY,
        request=spec.render_request(scenario),
        actor=PUBLIC if scenario.actor == "public" else "patient",
        setup=lambda _context: fixtures,
        model_actions=[],
        timeline=timeline,
        expected_calls=[],
        expected_claims=frozenset({claim}) if claim else frozenset(),
        db_checks=[],
        authorized_filters=_authorized_filters(scenario),
        version="1",
    )


def run_fixtures(
    engine: Any, scenario: spec.ScenarioSpec, world: builder.BuiltWorld
) -> RunFixtures:
    values: dict[str, Any] = {"patient": world.patient_id}
    values.update({f"slot_{key}": value for key, value in world.slot_ids.items()})
    values.update({f"clinic_{key}": value for key, value in world.clinic_ids.items()})
    return RunFixtures(
        values=values,
        slots=list(world.slot_ids.values()),
        protected=[
            *patient_identity(engine, world.patient_code),
            *patient_identity(engine, world.other_code),
        ],
        actor_code=None if scenario.actor == "public" else world.patient_code,
    )


# --- one run ------------------------------------------------------------------------------------


@dataclass
class ExecutedRun:
    record: scoring.RunRecord
    world_summary: dict[str, str]  # Scenario facts that must match across approaches.


def execute_run(
    approach: str,
    scenario: spec.ScenarioSpec,
    seed: int,
    model_factory: ModelFactory,
    *,
    environment: EvalEnvironment | None = None,
) -> ExecutedRun:
    """Build the world, verify it, run the approach, and score it.

    With no `environment` a fresh disposable database is created and dropped, which is what the
    protocol requires. A caller may pass one in only to test the builder's own checks.
    """
    if environment is not None:
        return _execute(approach, scenario, seed, model_factory, environment)
    with create_disposable_environment() as fresh:
        return _execute(approach, scenario, seed, model_factory, fresh)


def _execute(
    approach: str,
    scenario: spec.ScenarioSpec,
    seed: int,
    model_factory: ModelFactory,
    env: EvalEnvironment,
) -> ExecutedRun:
    engine = env.engine
    baseline_before = builder.seed_baseline_fingerprint(engine)
    suffix = builder.suffix_for(scenario.id)
    world = builder.build_world(engine, scenario, suffix=suffix, pin=env.pin)
    problems = builder.verify_world(engine, scenario, world)
    if problems:
        raise builder.BenchmarkEnvironmentError(
            f"{scenario.id}: the database does not match the manifest: {problems[:3]}"
        )
    initial = builder.fingerprint_of(
        builder.observe_state(
            engine, patient_id=world.patient_id, other_id=world.other_id, order=scenario.fixture
        )
    )
    models: list[Model] = []

    def factory(_spec: Scenario, _resolve: Callable) -> Model:
        model = model_factory(scenario, seed, world)
        models.append(model)
        return model

    runnable = scenario_for_run(scenario, world, run_fixtures(engine, scenario, world))
    result = run_scenario(
        runnable,
        env,
        model_factory=factory,
        require_response_text=True,
        suffix=suffix,
        record_conversation=True,
    )
    final = builder.observe_state(
        engine, patient_id=world.patient_id, other_id=world.other_id, order=scenario.fixture
    )
    untouched = builder.seed_baseline_fingerprint(engine) == baseline_before
    record = scoring.score_run(
        approach=approach,
        seed=seed,
        scenario=scenario,
        world=world,
        result=result,
        final_state=final,
        seed_untouched=untouched,
        initial_fingerprint=initial,
        usage=scoring.usage_of(models[0]) if models else None,
        request=runnable.request,
    )
    return ExecutedRun(
        record,
        {
            "seed_baseline_fingerprint": baseline_before,
            "patient_code": world.patient_code,
            "request_fingerprint": hashlib.sha256(runnable.request.encode()).hexdigest(),
        },
    )


# --- one approach, the whole benchmark ----------------------------------------------------------


@dataclass
class ApproachRun:
    approach: str
    seeds: tuple[int, ...]
    records: dict[int, list[scoring.RunRecord]] = field(default_factory=dict)
    seed_baselines: set[str] = field(default_factory=set)
    patient_codes: dict[str, str] = field(default_factory=dict)
    request_fingerprints: dict[str, str] = field(default_factory=dict)

    def scores(self, seed: int) -> list[scoring.RunScore]:
        return [record.score for record in self.records[seed]]


def run_approaches(
    factories: dict[str, ModelFactory],
    *,
    seeds: tuple[int, ...] = (7, 19),
    scenarios: list[spec.ScenarioSpec] | None = None,
    progress: Callable[[str], None] | None = None,
    on_run: Callable[[str, int, scoring.RunRecord], None] | None = None,
) -> dict[str, ApproachRun]:
    """Every approach on every scenario at every seed, alternating between approaches.

    The protocol requires alternation so that model loading, caching, and the state of the machine
    do not fall on one approach: within each scenario the approaches take turns going first. Each
    run still gets its own fresh database.
    """
    chosen = scenarios if scenarios is not None else spec.load_manifest().scenarios
    labels = list(factories)
    outcomes = {label: ApproachRun(approach=label, seeds=seeds) for label in labels}
    for seed in seeds:
        for outcome in outcomes.values():
            outcome.records[seed] = []
        for position, scenario in enumerate(chosen):
            turn = labels if position % 2 == 0 else labels[::-1]
            for label in turn:
                if progress:
                    progress(f"{label} seed {seed} {scenario.id}")
                executed = execute_run(label, scenario, seed, factories[label])
                outcome = outcomes[label]
                outcome.records[seed].append(executed.record)
                if on_run:
                    on_run(label, seed, executed.record)
                outcome.seed_baselines.add(executed.world_summary["seed_baseline_fingerprint"])
                outcome.patient_codes[scenario.id] = executed.world_summary["patient_code"]
                outcome.request_fingerprints[scenario.id] = executed.world_summary[
                    "request_fingerprint"
                ]
    return outcomes


def run_approach(
    approach: str,
    model_factory: ModelFactory,
    *,
    seeds: tuple[int, ...] = (7, 19),
    scenarios: list[spec.ScenarioSpec] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ApproachRun:
    """One approach on its own. The official comparison uses `run_approaches` instead."""
    return run_approaches(
        {approach: model_factory}, seeds=seeds, scenarios=scenarios, progress=progress
    )[approach]


def warm_up(config: Any, transport: Any = None) -> None:
    """One throwaway request so the model is loaded before the first scored call. It is not
    attributed to either approach and is never recorded."""
    from app.evaluation.llm import http_transport

    post = transport or http_transport(config.base_url)
    post(
        "/api/chat",
        {
            "model": config.model,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4, "num_ctx": config.context_tokens},
        },
        config.request_timeout_s,
    )


def execution_metadata(
    run: ApproachRun,
    *,
    model: str,
    model_digest: str,
    temperature: float,
    context_tokens: int,
    max_output_tokens: int,
    request_timeout_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What must be equal for two executions to be comparable, plus approach-specific details
    (which are recorded but never compared)."""
    primary = run.records[run.seeds[0]]
    options = {
        "temperature": temperature,
        "num_ctx": context_tokens,
        "num_predict": max_output_tokens,
    }
    return {
        "manifest_sha256": spec.manifest_sha256(),
        "protocol_fingerprint": protocol.protocol_fingerprint(),
        "scorer_version": protocol.SCORER_VERSION,
        "model": model,
        "model_digest": model_digest,
        "temperature": temperature,
        "seeds": list(run.seeds),
        "context_tokens": context_tokens,
        "max_output_tokens": max_output_tokens,
        "request_timeout_s": request_timeout_s,
        "scenario_ids": [record.scenario_id for record in primary],
        "initial_state_fingerprints": {
            record.scenario_id: record.score.initial_state_fingerprint for record in primary
        },
        "ollama_options_fingerprint": hashlib.sha256(
            json.dumps(options, sort_keys=True).encode()
        ).hexdigest(),
        "seed_baseline_fingerprint": sorted(run.seed_baselines),
        # Equal across approaches means the same patient signed in each time. The codes themselves
        # are login identifiers and are never published, only their hash.
        "patient_identity_fingerprint": hashlib.sha256(
            json.dumps(run.patient_codes, sort_keys=True).encode()
        ).hexdigest(),
        "request_fingerprints": run.request_fingerprints,
        **(extra or {}),
    }


def build_report(run: ApproachRun, metadata: dict[str, Any]) -> dict[str, Any]:
    primary_seed, *others = run.seeds
    return scoring.sanitized_report(
        approach=run.approach,
        metadata=metadata,
        primary=run.scores(primary_seed),
        second_seed=run.scores(others[0]) if others else None,
        records=[record for seed in run.seeds for record in run.records[seed]],
    )


def ensure_untracked_directory(directory: Path) -> Path:
    """Raw conversations are written only where Git will not pick them up.

    A directory inside the repository is refused unless Git already ignores it.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[3]
    resolved = directory.resolve()
    if repo in (resolved, *resolved.parents):
        ignored = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", str(resolved)],
            capture_output=True,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("raw conversations must not be written to a path Git does not ignore")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# --- the command line ---------------------------------------------------------------------------
#
#   python -m app.evaluation.benchmark_run run --output-dir DIR       (both approaches, alternating)
#   python -m app.evaluation.benchmark_run assemble --output-dir DIR
#   python -m app.evaluation.benchmark_run audit --output-dir DIR --judgments FILE
#
# `run` needs the local model server; `assemble` does not. Nothing here is run by the tests with a
# real model: they use the same functions with a scripted stand-in.


def approach_factory(approach: str, model: str, base_url: str) -> ModelFactory:
    """The two candidates, built identically except for what they are."""
    from app.agent_tools import build_default_registry
    from app.evaluation.hybrid import HybridControllerModel, StructuredIntentExtractor
    from app.evaluation.llm import (
        APPROACH_A_SYSTEM_PROMPT,
        LlmConfig,
        OllamaModel,
    )

    definitions = build_default_registry().definitions()

    def config(seed: int) -> LlmConfig:
        return LlmConfig(model=model, base_url=base_url, seed=seed)

    if approach == "A":
        return lambda _s, seed, _w: OllamaModel(
            definitions, config(seed), system_prompt=APPROACH_A_SYSTEM_PROMPT
        )
    if approach == "B":
        return lambda _s, seed, _w: HybridControllerModel(StructuredIntentExtractor(config(seed)))
    raise ValueError(f"unknown approach {approach!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_run(arguments: Any) -> int:
    import sys

    import httpx

    from app.evaluation.cli import served_model_digest, unusable
    from app.evaluation.hybrid import HYBRID_PROMPT_VERSION
    from app.evaluation.llm import (
        APPROACH_A_PROMPT_VERSION,
        LlmConfig,
        ModelConfigurationError,
        ModelTimeoutError,
        ModelUnavailableError,
    )
    from app.evaluation.model import MalformedModelResponse

    directory = ensure_untracked_directory(arguments.output_dir)
    try:
        config = LlmConfig(model=arguments.model, base_url=arguments.url)  # Local servers only.
    except ModelConfigurationError as error:
        raise unusable(str(error)) from None
    try:
        digest = served_model_digest(config)
    except (httpx.HTTPError, ValueError):
        raise unusable(f"cannot reach a model server at {arguments.url}") from None
    if digest is None:
        raise unusable(f"the model server has no model {arguments.model!r}")
    try:
        warm_up(config)
    except (ModelTimeoutError, ModelUnavailableError, MalformedModelResponse):
        raise unusable("the model server did not answer the warm-up request") from None
    labels = ["A", "B"] if arguments.approach == "both" else [arguments.approach]
    status = {"state": "running", "completed_runs": 0, "interrupted_by": None}

    def save_status() -> None:
        _write_json(directory / "run-status.json", status)

    def keep(label: str, seed: int, record: scoring.RunRecord) -> None:
        """Write each finished run at once, so an interruption loses at most the run in progress."""
        append_partial(directory, label, seed, record)
        status["completed_runs"] += 1
        save_status()

    save_status()
    try:
        runs = run_approaches(
            {label: approach_factory(label, arguments.model, arguments.url) for label in labels},
            seeds=tuple(arguments.seeds),
            progress=lambda line: print(line, file=sys.stderr),
            on_run=keep,
        )
    except BaseException as error:  # Including Ctrl-C: record it, keep what was written, and stop.
        status.update(state="interrupted", interrupted_by=type(error).__name__)
        save_status()
        raise
    status["state"] = "complete"
    save_status()
    for label, outcome in runs.items():
        details = APPROACH_A_PROMPT_VERSION if label == "A" else HYBRID_PROMPT_VERSION
        metadata = execution_metadata(
            outcome,
            model=arguments.model,
            model_digest=digest,
            temperature=config.temperature,
            context_tokens=config.context_tokens,
            max_output_tokens=config.max_output_tokens,
            request_timeout_s=config.request_timeout_s,
            extra={"approach_details": {"prompt_version": details}},
        )
        save_outputs(directory, outcome, metadata)
    return 0


def append_partial(directory: Path, label: str, seed: int, record: scoring.RunRecord) -> None:
    """One finished run, appended: the sanitized score to one file and the raw replies (local
    only) to another."""
    from dataclasses import asdict

    from app.evaluation.benchmark_review import sources_from, sources_to_json

    with (directory / "partial-scores.jsonl").open("a") as handle:
        handle.write(json.dumps(asdict(record.score)) + "\n")
    with (directory / "partial-raw.jsonl").open("a") as handle:
        handle.write(json.dumps(sources_to_json(sources_from([record]))) + "\n")


def save_outputs(directory: Path, run: ApproachRun, metadata: dict[str, Any]) -> None:
    """The sanitized report (publishable) and the raw replies (local only) for one approach."""
    from app.evaluation.benchmark_review import sources_from, sources_to_json

    _write_json(directory / f"report-{run.approach}.json", build_report(run, metadata))
    records = [record for seed in run.seeds for record in run.records[seed]]
    _write_json(directory / f"raw-{run.approach}.json", sources_to_json(sources_from(records)))


def command_assemble(arguments: Any) -> int:
    from app.evaluation.benchmark_review import (
        sources_from_json,
        write_review_materials,
    )

    directory = ensure_untracked_directory(arguments.output_dir)
    reports = {x: json.loads((directory / f"report-{x}.json").read_text()) for x in ("A", "B")}
    comparison = scoring.compare_approaches(reports["A"], reports["B"])
    _write_json(directory / "comparison.json", comparison)
    sources = []
    for x in ("A", "B"):
        sources += sources_from_json(json.loads((directory / f"raw-{x}.json").read_text()))
    scenarios = {s.id: s for s in spec.load_manifest().scenarios}
    write_review_materials(directory, sources, scenarios, seed=arguments.seed)
    return 0 if comparison["comparable"] else 1


def command_audit(arguments: Any) -> int:
    from app.evaluation.benchmark_review import audit_agreement

    directory = ensure_untracked_directory(arguments.output_dir)
    key = json.loads((directory / "review-key.json").read_text())
    judgments = json.loads(arguments.judgments.read_text())
    result = audit_agreement(key, judgments)
    _write_json(directory / "audit-agreement.json", result)
    return 0 if all(v["reliable"] for v in result["per_verdict"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the frozen architecture benchmark.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one approach on every scenario and seed.")
    run.add_argument(
        "--approach",
        choices=["A", "B", "both"],
        default="both",
        help="The official comparison runs both, alternating; a single approach is for debugging.",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", default="qwen2.5:7b")
    run.add_argument("--url", default="http://127.0.0.1:11434")
    run.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    assemble = commands.add_parser("assemble", help="Compare both runs and write the review sheet.")
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--seed", type=int, default=7)
    audit = commands.add_parser("audit", help="Measure how far a reviewer agrees with the rules.")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--judgments", type=Path, required=True)
    arguments = parser.parse_args(argv)
    handler = {"run": command_run, "assemble": command_assemble, "audit": command_audit}
    return handler[arguments.command](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
