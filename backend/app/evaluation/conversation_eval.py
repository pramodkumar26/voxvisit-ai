"""Run the multi-turn conversation scenarios against Approach B, and summarize them safely.

    PYTHONPATH=backend uv run python -m app.evaluation.conversation_eval smoke --output-dir DIR

The smoke run uses the local Qwen model through the real extractor and the real tools, in one
disposable database. It is a bounded check of the conversation layer, not a benchmark: every
scenario runs once, nothing is retried, and nothing here selects an architecture. Raw conversations
are written only to a directory Git ignores. The summary has codes and counts, never the words.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.conversation_scenarios import (
    ALL_SCENARIOS,
    CONVERSATION_SCENARIOS,
    CONVERSATION_VARIANTS,
    FAULTS,
    SPOKEN_SCENARIOS,
    SPOKEN_VARIANTS,
)
from app.evaluation.hybrid import HYBRID_PROMPT_VERSION, HybridControllerModel
from app.evaluation.model import MalformedModelResponse
from app.evaluation.report import OUTCOME_FAILURES, assert_clean
from app.evaluation.runner import ScenarioResult, run_scenario
from app.evaluation.scenario import Scenario


class FaultInjectingReader:
    """Wraps a reader (the real extractor or a stand-in) and makes its readings fail on purpose.

    `invalid_once`: the first reading is rejected as invalid, as a malformed model reply would be.
    `outage`: every reading fails as if the model server could not be reached. Nothing is then
    asked of the wrapped reader, and nothing can be salvaged.
    """

    def __init__(self, inner: Any, mode: str) -> None:
        if mode not in {"invalid_once", "outage"}:
            raise ValueError(f"unknown fault {mode!r}")
        self.inner, self.mode, self.calls = inner, mode, 0

    def extract(self, request: str, **context: Any):
        self.calls += 1
        if self.mode == "outage":
            raise TimeoutError("the model server did not answer")
        if self.mode == "invalid_once" and self.calls == 1:
            raise MalformedModelResponse("the structured intent was not valid")
        return self.inner.extract(request, **context)

    def salvage(self, *args: Any, **kwargs: Any):
        # The wrapped reader has nothing to salvage unless it really served a reply.
        return self.inner.salvage(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class Outcome:
    result: ScenarioResult
    model: HybridControllerModel


ReaderFor = Callable[[Scenario, Callable], Any]


def run_conversations(
    environment: Any,
    reader_for: ReaderFor,
    scenarios: Sequence[Scenario] = CONVERSATION_SCENARIOS,
    *,
    faults: dict[str, str] | None = None,
    suffix_base: str = "c0ffee",
    pause_every: int | None = None,
) -> list[Outcome]:
    """Each scenario in order, in the same database, with a fresh conversation and fresh patients.

    `pause_every` moves the shared clock on 16 minutes after that many scenarios, so that a long
    run does not reach the application's limit on sign-ins from one source.

    `suffix_base` (hex) makes the synthetic names repeatable; use a new one for each call that
    shares a database with an earlier call."""
    faults = FAULTS if faults is None else faults
    outcomes: list[Outcome] = []
    for index, scenario in enumerate(scenarios):
        holder: list[HybridControllerModel] = []

        def factory(spec, resolve, _holder=holder):
            reader = reader_for(spec, resolve)
            if spec.id in faults:
                reader = FaultInjectingReader(reader, faults[spec.id])
            model = HybridControllerModel(reader, clock=environment.clock)
            _holder.append(model)
            return model

        result = run_scenario(
            scenario,
            environment,
            model_factory=factory,
            require_response_text=True,
            suffix=f"{index + 1:02x}{suffix_base}"[:8],
            record_conversation=True,
        )
        outcomes.append(Outcome(result, holder[0]))
        if pause_every and (index + 1) % pause_every == 0:
            # Every conversation signs in from the same source, and the application limits sign-ins
            # from one source (20 in 15 minutes). The clock, not the application, is moved on.
            environment.clock.advance(minutes=16)
    return outcomes


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Codes and counts only: no patient or assistant words, no identifiers."""
    entries = []
    for item in outcomes:
        result = item.result
        categories = sorted({failure["category"] for failure in result.failures})
        entries.append(
            {
                "id": result.id,
                "title": result.title,
                "outcome_reached": not set(categories) & OUTCOME_FAILURES,
                "failure_categories": categories,
                "recoveries": sorted(set(item.model.recoveries)),
                "tool_calls": result.metrics.tool_calls,
                "model_decisions": result.metrics.model_decisions,
                "patient_turns": 1 + result.metrics.patient_follow_ups,
                "safety_violations": result.safety.violations(),
                "confirmations_committed": result.metrics.ui_confirmations_committed,
            }
        )
    return {
        "scenarios": len(entries),
        "outcomes_reached": sum(e["outcome_reached"] for e in entries),
        "safety_violations": sum(e["safety_violations"] for e in entries),
        "entries": entries,
    }


def raw_conversations(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    return [
        {"id": item.result.id, "conversation": item.result.conversation, "steps": item.result.steps}
        for item in outcomes
    ]


# --- the command line -------------------------------------------------------------------------


def command_smoke(arguments: argparse.Namespace) -> int:
    from pathlib import Path

    import httpx

    from app.evaluation.benchmark_run import ensure_untracked_directory
    from app.evaluation.cli import served_model_digest, unusable
    from app.evaluation.environment import create_disposable_environment
    from app.evaluation.hybrid import StructuredIntentExtractor
    from app.evaluation.llm import LlmConfig, ModelConfigurationError

    directory = ensure_untracked_directory(Path(arguments.output_dir))
    try:
        config = LlmConfig(model=arguments.model, base_url=arguments.url)
        digest = served_model_digest(config)
    except ModelConfigurationError as error:
        raise unusable(str(error)) from None
    except (httpx.HTTPError, ValueError):
        raise unusable(f"cannot reach a model server at {config.base_url}") from None
    if digest is None:
        raise unusable(f"the server at {config.base_url} has no model {config.model!r}")
    scenarios = {
        "core": CONVERSATION_SCENARIOS,
        "variants": CONVERSATION_VARIANTS,
        "spoken": SPOKEN_SCENARIOS,
        "spoken-variants": SPOKEN_VARIANTS,
        "all": ALL_SCENARIOS,
    }[arguments.set]
    if arguments.only:
        scenarios = tuple(s for s in scenarios if s.id in arguments.only)
        if not scenarios:
            raise unusable("no such scenario")
    extractors: list[StructuredIntentExtractor] = []

    def reader_for(_spec, _resolve):
        extractor = StructuredIntentExtractor(config)
        extractors.append(extractor)
        return extractor

    with create_disposable_environment() as environment:
        outcomes = run_conversations(environment, reader_for, scenarios)
    forbidden = [text for item in outcomes for text in item.result.forbidden_text]
    summary = summarize(outcomes)
    calls = [call for extractor in extractors for call in extractor.usage.calls]
    summary["model"] = {
        "name": config.model,
        "digest": digest,
        "prompt_version": HYBRID_PROMPT_VERSION,
        "seed": config.seed,
        "temperature": config.temperature,
        "provider_calls": len(calls),
        "invalid_readings": sum(call.outcome == "invalid_structured_intent" for call in calls),
        "provider_errors": sum(
            call.outcome not in {"structured_intent", "invalid_structured_intent"} for call in calls
        ),
    }
    assert_clean(summary, forbidden)
    (directory / "smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raw = raw_conversations(outcomes)
    (directory / "smoke-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2))
    for entry in summary["entries"]:
        mark = "ok  " if entry["outcome_reached"] else "FAIL"
        print(f"{mark} {entry['id']} {entry['failure_categories']} {entry['recoveries']}")
    return 1 if summary["safety_violations"] else 0


def command_acceptance(arguments: argparse.Namespace) -> int:
    """Run the frozen Phase 2 acceptance conversations once with the real model."""
    import platform
    from pathlib import Path

    import httpx

    from app.evaluation import acceptance_cases as frozen
    from app.evaluation.acceptance_score import score, summarize
    from app.evaluation.benchmark_run import ensure_untracked_directory
    from app.evaluation.cli import served_model_digest, unusable
    from app.evaluation.environment import EvalClock, create_disposable_environment
    from app.evaluation.hybrid import StructuredIntentExtractor
    from app.evaluation.llm import LlmConfig, ModelConfigurationError

    directory = ensure_untracked_directory(Path(arguments.output_dir))
    frozen_file = Path(arguments.manifest)
    current = frozen.fingerprint()
    if current != arguments.expect_fingerprint or frozen_file.read_text() != frozen.canonical(
        frozen.manifest()
    ):
        raise unusable("the acceptance cases differ from the frozen manifest; nothing was run")
    try:
        config = LlmConfig(model=arguments.model, base_url=arguments.url)
        digest = served_model_digest(config)
    except ModelConfigurationError as error:
        raise unusable(str(error)) from None
    except (httpx.HTTPError, ValueError):
        raise unusable(f"cannot reach a model server at {config.base_url}") from None
    if digest is None:
        raise unusable(f"the server at {config.base_url} has no model {config.model!r}")
    extractors: list[StructuredIntentExtractor] = []

    def reader_for(_spec, _resolve):
        extractor = StructuredIntentExtractor(config)
        extractors.append(extractor)
        return extractor

    cases = frozen.CASES
    with create_disposable_environment() as environment:
        runs = run_conversations(
            environment,
            reader_for,
            [case.scenario for case in cases],
            faults={case.id: case.fault for case in cases if case.fault},
            pause_every=10,
        )
        clock_start = EvalClock().now.isoformat()
    scored = [score(case, run.result, run.model) for case, run in zip(cases, runs, strict=True)]
    summary = summarize(scored, frozen.CATEGORIES)
    forbidden = [text for run in runs for text in run.result.forbidden_text]
    report = {
        "evaluation": frozen.VERSION,
        "fingerprint": current,
        "run": {
            "times": "each case run once; nothing retried or edited after a result",
            "model": config.model,
            "model_digest": digest,
            "prompt_version": HYBRID_PROMPT_VERSION,
            "seed": config.seed,
            "temperature": config.temperature,
            "context_tokens": config.context_tokens,
            "output_limit": config.max_output_tokens,
            "call_timeout_s": config.request_timeout_s,
            "clock_start": clock_start,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        **summary,
    }
    assert_clean(report, forbidden)
    (directory / "acceptance-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "acceptance-raw.json").write_text(
        json.dumps(raw_conversations(runs), indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for item in scored:
        flag = "ok  " if item.success else ("VIOL" if item.violations else "fail")
        print(flag, item.id, item.category, item.reasons, item.violations)
    return 1 if report["safety_violations"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    from app.evaluation.llm import DEFAULT_BASE_URL, DEFAULT_MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="run the conversations with the local model")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--url", default=DEFAULT_BASE_URL)
    smoke.add_argument("--only", nargs="*", help="scenario ids to run")
    smoke.add_argument(
        "--set", choices=["core", "variants", "spoken", "spoken-variants", "all"], default="core"
    )
    acceptance = commands.add_parser("acceptance", help="run the frozen acceptance conversations")
    acceptance.add_argument("--output-dir", required=True)
    acceptance.add_argument("--expect-fingerprint", required=True)
    acceptance.add_argument(
        "--manifest", default=str(Path(__file__).parents[2] / "tests" / "phase2-acceptance-v1.json")
    )
    acceptance.add_argument("--model", default=DEFAULT_MODEL)
    acceptance.add_argument("--url", default=DEFAULT_BASE_URL)
    arguments = parser.parse_args(argv)
    return (
        command_acceptance(arguments)
        if arguments.command == "acceptance"
        else command_smoke(arguments)
    )


if __name__ == "__main__":
    sys.exit(main())
